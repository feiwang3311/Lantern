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

long HEAP_SIZE = 4294967304; // this is for GPU

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
int32_t x272 = open("../../cifar10_data/cifar-10-batches-bin/data_batch_1.bin",0);
int32_t x273 = fsize(x272);
int64_t x275 = (int64_t)x273;
int64_t x276 = x275 / 3073LL;
int32_t x277 = (int32_t)x276;
int32_t x278 = x277 * 3072;
float* x279 = (float*)myMalloc(x278 * sizeof(float));;
int* x280 = (int32_t*)myMalloc(x277 * sizeof(int32_t));;
char* x274 = (char*)mmap(0, x273, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x272, 0);
for(int x282=0; x282 < x277; x282++) {
int32_t x283 = x282 * 3073;
char x284 = x274[x283];
int32_t x285 = (int32_t)(unsigned char)x284;
x280[x282] = x285;
int32_t x291 = x283 + 1;
int32_t x289 = x282 * 3072;
for(int x288=0; x288 < 3072; x288++) {
int32_t x292 = x291 + x288;
char x293 = x274[x292];
int32_t x290 = x289 + x288;
float x294 = (float)(unsigned char)x293;
float x295 = x294 / 255.0f;
x279[x290] = x295;

}

}
int32_t x301 = x277 / 64;
int32_t x2 = open("/u/data/u99/wang603/TiarkMlEnv/Lantern/src/out/PLDI19evaluation/resnet50/resnet50.onnx.bin",0);
int32_t x3 = fsize(x2);
float* x4 = (float*)mmap(0, x3, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x2, 0);
float* x151 = x4+0;
float* x39 = x4+1856;
float* x109 = x4+1920;
float* x205 = x4+1728;
float* x250 = x4+1792;
float* x232 = x4+1984;
float* x113 = x4+6208;
float* x50 = x4+6272;
float* x25 = x4+6080;
float* x52 = x4+6144;
float* x89 = x4+6336;
float* x104 = x4+43328;
float* x157 = x4+43392;
float* x163 = x4+43200;
float* x48 = x4+43264;
float* x31 = x4+43456;
float* x70 = x4+60352;
float* x35 = x4+60608;
float* x198 = x4+59840;
float* x125 = x4+60096;
float* x161 = x4+60864;
float* x263 = x4+77760;
float* x242 = x4+78016;
float* x75 = x4+77248;
float* x202 = x4+77504;
float* x170 = x4+78272;
float* x9 = x4+94784;
float* x101 = x4+94848;
float* x141 = x4+94656;
float* x59 = x4+94720;
float* x82 = x4+94912;
float* x43 = x4+131904;
float* x243 = x4+131968;
float* x207 = x4+131776;
float* x152 = x4+131840;
float* x129 = x4+132032;
float* x90 = x4+148928;
float* x165 = x4+149184;
float* x57 = x4+148416;
float* x6 = x4+148672;
float* x149 = x4+149440;
float* x256 = x4+165952;
float* x186 = x4+166016;
float* x80 = x4+165824;
float* x23 = x4+165888;
float* x72 = x4+166080;
float* x178 = x4+203072;
float* x117 = x4+203136;
float* x71 = x4+202944;
float* x134 = x4+203008;
float* x86 = x4+203200;
float* x183 = x4+220096;
float* x132 = x4+220352;
float* x36 = x4+219584;
float* x246 = x4+219840;
float* x10 = x4+220608;
float* x203 = x4+253632;
float* x133 = x4+253760;
float* x83 = x4+253376;
float* x171 = x4+253504;
float* x26 = x4+253888;
float* x127 = x4+401600;
float* x42 = x4+401728;
float* x251 = x4+401344;
float* x189 = x4+401472;
float* x105 = x4+401856;
float* x148 = x4+468416;
float* x100 = x4+468928;
float* x144 = x4+467392;
float* x209 = x4+467904;
float* x257 = x4+469440;
float* x41 = x4+601536;
float* x22 = x4+602048;
float* x206 = x4+600512;
float* x118 = x4+601024;
float* x255 = x4+602560;
float* x99 = x4+668352;
float* x176 = x4+668480;
float* x221 = x4+668096;
float* x16 = x4+668224;
float* x234 = x4+668608;
float* x34 = x4+816320;
float* x224 = x4+816448;
float* x7 = x4+816064;
float* x94 = x4+816192;
float* x110 = x4+816576;
float* x146 = x4+883136;
float* x87 = x4+883648;
float* x51 = x4+882112;
float* x245 = x4+882624;
float* x195 = x4+884160;
float* x111 = x4+949952;
float* x8 = x4+950080;
float* x44 = x4+949696;
float* x169 = x4+949824;
float* x190 = x4+950208;
float* x216 = x4+1097920;
float* x265 = x4+1098048;
float* x126 = x4+1097664;
float* x60 = x4+1097792;
float* x40 = x4+1098176;
float* x24 = x4+1164736;
float* x222 = x4+1165248;
float* x166 = x4+1163712;
float* x81 = x4+1164224;
float* x131 = x4+1165760;
float* x235 = x4+1231552;
float* x260 = x4+1231680;
float* x38 = x4+1231296;
float* x241 = x4+1231424;
float* x164 = x4+1231808;
float* x267 = x4+1379520;
float* x147 = x4+1379648;
float* x78 = x4+1379264;
float* x37 = x4+1379392;
float* x54 = x4+1379776;
float* x18 = x4+1446336;
float* x233 = x4+1446848;
float* x155 = x4+1445312;
float* x53 = x4+1445824;
float* x179 = x4+1447360;
float* x130 = x4+1578944;
float* x197 = x4+1579200;
float* x269 = x4+1578432;
float* x20 = x4+1578688;
float* x174 = x4+1579456;
float* x228 = x4+2169792;
float* x98 = x4+2170048;
float* x107 = x4+2169280;
float* x15 = x4+2169536;
float* x268 = x4+2170304;
float* x215 = x4+2434496;
float* x266 = x4+2435520;
float* x17 = x4+2432448;
float* x116 = x4+2433472;
float* x74 = x4+2436544;
float* x85 = x4+2962880;
float* x210 = x4+2963904;
float* x28 = x4+2960832;
float* x219 = x4+2961856;
float* x12 = x4+2964928;
float* x258 = x4+3227584;
float* x156 = x4+3227840;
float* x29 = x4+3227072;
float* x218 = x4+3227328;
float* x30 = x4+3228096;
float* x199 = x4+3818432;
float* x236 = x4+3818688;
float* x270 = x4+3817920;
float* x95 = x4+3818176;
float* x55 = x4+3818944;
float* x181 = x4+4083136;
float* x142 = x4+4084160;
float* x19 = x4+4081088;
float* x231 = x4+4082112;
float* x217 = x4+4085184;
float* x177 = x4+4347840;
float* x173 = x4+4348096;
float* x128 = x4+4347328;
float* x196 = x4+4347584;
float* x13 = x4+4348352;
float* x123 = x4+4938688;
float* x62 = x4+4938944;
float* x227 = x4+4938176;
float* x191 = x4+4938432;
float* x115 = x4+4939200;
float* x139 = x4+5203392;
float* x187 = x4+5204416;
float* x262 = x4+5201344;
float* x56 = x4+5202368;
float* x5 = x4+5205440;
float* x162 = x4+5468096;
float* x97 = x4+5468352;
float* x91 = x4+5467584;
float* x240 = x4+5467840;
float* x248 = x4+5468608;
float* x185 = x4+6058944;
float* x229 = x4+6059200;
float* x73 = x4+6058432;
float* x135 = x4+6058688;
float* x88 = x4+6059456;
float* x230 = x4+6323648;
float* x160 = x4+6324672;
float* x237 = x4+6321600;
float* x145 = x4+6322624;
float* x21 = x4+6325696;
float* x253 = x4+6588352;
float* x68 = x4+6588608;
float* x76 = x4+6587840;
float* x184 = x4+6588096;
float* x261 = x4+6588864;
float* x249 = x4+7179200;
float* x103 = x4+7179456;
float* x167 = x4+7178688;
float* x108 = x4+7178944;
float* x220 = x4+7179712;
float* x208 = x4+7443904;
float* x271 = x4+7444928;
float* x58 = x4+7441856;
float* x119 = x4+7442880;
float* x150 = x4+7445952;
float* x79 = x4+7708608;
float* x175 = x4+7708864;
float* x84 = x4+7708096;
float* x252 = x4+7708352;
float* x225 = x4+7709120;
float* x69 = x4+8299456;
float* x239 = x4+8299712;
float* x140 = x4+8298944;
float* x188 = x4+8299200;
float* x96 = x4+8299968;
float* x121 = x4+8564160;
float* x182 = x4+8565184;
float* x247 = x4+8562112;
float* x92 = x4+8563136;
float* x138 = x4+8566208;
float* x66 = x4+9091520;
float* x120 = x4+9092032;
float* x200 = x4+9090496;
float* x223 = x4+9091008;
float* x33 = x4+9092544;
float* x112 = x4+11452864;
float* x49 = x4+11453376;
float* x204 = x4+11451840;
float* x158 = x4+11452352;
float* x211 = x4+11453888;
float* x114 = x4+12506560;
float* x192 = x4+12508608;
float* x238 = x4+12502464;
float* x61 = x4+12504512;
float* x213 = x4+12510656;
float* x63 = x4+14611904;
float* x124 = x4+14613952;
float* x172 = x4+14607808;
float* x106 = x4+14609856;
float* x214 = x4+14616000;
float* x153 = x4+15665600;
float* x64 = x4+15666112;
float* x45 = x4+15664576;
float* x136 = x4+15665088;
float* x154 = x4+15666624;
float* x137 = x4+18026944;
float* x194 = x4+18027456;
float* x159 = x4+18025920;
float* x65 = x4+18026432;
float* x46 = x4+18027968;
float* x67 = x4+19080640;
float* x244 = x4+19082688;
float* x93 = x4+19076544;
float* x143 = x4+19078592;
float* x264 = x4+19084736;
float* x212 = x4+20134336;
float* x254 = x4+20134848;
float* x14 = x4+20133312;
float* x77 = x4+20133824;
float* x27 = x4+20135360;
float* x11 = x4+22495680;
float* x201 = x4+22496192;
float* x193 = x4+22494656;
float* x168 = x4+22495168;
float* x32 = x4+22496704;
float* x259 = x4+23549376;
float* x122 = x4+23551424;
float* x102 = x4+23545280;
float* x180 = x4+23547328;
float* x226 = x4+23553472;
float* x47 = x4+23573952;
for(int x303=0; x303 < x301; x303++) {
int32_t x304 = x303 * 64;
int32_t x305 = x304 * 3072;
float* x306 = x279+x305;
int* x307 = x280+x304;
printf("input (size 64 x 3 x 32 x 32)\n");
float x309 = 0.0f;
for(int x311=0; x311 < 196608; x311++) {
float x312 = x309;
float x313 = x306[x311];
float x314 = fabs(x313);
float x315 = fabs(x312);
bool x316 = x314 > x315;
float x317;
if (x316) {
x317 = x313;
} else {
x317 = x312;
}
x309 = x317;

}
float x321 = x309;
printf("Max Abs: %.5f || ",x321);
for(int x324=0; x324 < 10; x324++) {
float x325 = x306[x324];
printf("%.5f ",x325);

}
printf("\n");
float* x330 = (float*)myMalloc(4194304 * sizeof(float));;
float* x331 = (float*)myMalloc(1769472 * sizeof(float));;
for(int x333=0; x333 < 64; x333++) {
int32_t x334 = x333 * 3072;
float* x335 = x306+x334;
int32_t x336 = x333 * 65536;
float* x337 = x330+x336;
int32_t x338 = x333 * 27648;
float* x339 = x331+x338;
for(int x341=0; x341 < 27; x341++) {
int32_t x342 = x341 / 9;
int32_t x346 = x342 * 3;
int32_t x347 = x346 * 3;
int32_t x348 = x347 * 32;
int32_t x349 = x348 * 32;
int32_t x343 = x341 % 9;
int32_t x344 = x343 / 3;
int32_t x350 = x344 * 3;
int32_t x351 = x350 * 32;
int32_t x352 = x351 * 32;
int32_t x353 = x349 + x352;
int32_t x345 = x343 % 3;
int32_t x354 = x345 * 32;
int32_t x355 = x354 * 32;
int32_t x356 = x353 + x355;
float* x357 = x339+x356;
int32_t x358 = x342 * 32;
int32_t x359 = x358 * 32;
float* x360 = x335+x359;
int32_t x373 = 1 - x345;
bool x374 = x373 > 0;
int32_t x375;
if (x374) {
x375 = x373;
} else {
x375 = 0;
}
int32_t x376 = 3 - x345;
int32_t x377 = x376 - 1;
int32_t x378 = 1 - x377;
bool x379 = x378 > 0;
int32_t x380;
if (x379) {
x380 = x378;
} else {
x380 = 0;
}
int32_t x381 = 32 - x380;
int32_t x382 = x381 - x375;
bool x383 = x382 <= 0;
bool x387 = x375 > 0;
int32_t x372 = -1 + x345;
bool x400 = x380 > 0;
for(int x362=0; x362 < 32; x362++) {
int32_t x363 = x362 - 1;
int32_t x364 = x363 + x344;
bool x365 = x364 < 0;
bool x366 = x364 >= 32;
bool x367 = x365 || x366;
if (x367) {
int32_t x368 = x362 * 32;
float* x369 = x357+x368;
memset(x369, 0, 4 * 32);;
} else {
if (x383) {
int32_t x368 = x362 * 32;
float* x384 = x357+x368;
memset(x384, 0, 4 * 32);;
} else {
int32_t x368 = x362 * 32;
if (x387) {
float* x388 = x357+x368;
memset(x388, 0, 4 * x375);;
} else {
}
// may have segfault here
int32_t x393 = x368 + x375;
float* x394 = x357+x393;
int32_t x395 = x364 * 32;
int32_t x396 = x395 + x372;
int32_t x397 = x396 + x375;
float* x398 = x360+x397;
memcpy(x394, x398, 4 * x382);;
if (x400) {
int32_t x401 = x368 + 32;
int32_t x402 = x401 - x380;
float* x403 = x357+x402;
memset(x403, 0, 4 * x380);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,1024,27,1,x151,27,x339,1024,1,x337,1024);

}
// resize to WrappedArray(-1, 1, 1)
float* x419 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x420 = 0;
int32_t x421 = 0;
int32_t x422 = 0;
for(int x423=0; x423 < 64; x423++) {
int32_t x424 = x421;
int32_t x425 = x422;
int32_t x426 = x420;
int32_t x427 = x426;
int32_t x428 = x424;
int32_t x429 = x425;
for(int x430=0; x430 < 64; x430++) {
int32_t x431 = x428;
int32_t x432 = x429;
int32_t x433 = x427;
int32_t x434 = x433;
int32_t x435 = x431;
int32_t x436 = x432;
for(int x437=0; x437 < 32; x437++) {
int32_t x438 = x435;
int32_t x439 = x436;
int32_t x440 = x434;
int32_t x441 = x440;
int32_t x442 = x438;
int32_t x443 = x439;
for(int x444=0; x444 < 32; x444++) {
int32_t x445 = x441;
int32_t x446 = x442;
float x447 = x330[x446];
int32_t x448 = x443;
float x449 = x39[x448];
float x450 = x447 - x449;
x419[x445] = x450;
x441 += 1;
x442 += 1;

}
x434 += 32;
x435 += 32;

}
x427 += 1024;
x428 += 1024;
x429 += 1;

}
x420 += 65536;
x421 += 65536;

}
float* x469 = (float*)myMalloc(64 * sizeof(float));;
for(int x470=0; x470 < 64; x470++) {
float x471 = x109[x470];
float x472 = x471 + 1.0E-5f;
x469[x470] = x472;

}
float* x476 = (float*)myMalloc(64 * sizeof(float));;
for(int x477=0; x477 < 64; x477++) {
float x478 = x469[x477];
double x479 = (double)x478;
double x480 = sqrt(x479);
float x481 = (float)x480;
x476[x477] = x481;

}
// resize to WrappedArray(-1, 1, 1)
float* x486 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x487 = 0;
int32_t x488 = 0;
int32_t x489 = 0;
for(int x490=0; x490 < 64; x490++) {
int32_t x491 = x488;
int32_t x492 = x489;
int32_t x493 = x487;
int32_t x494 = x493;
int32_t x495 = x491;
int32_t x496 = x492;
for(int x497=0; x497 < 64; x497++) {
int32_t x498 = x495;
int32_t x499 = x496;
int32_t x500 = x494;
int32_t x501 = x500;
int32_t x502 = x498;
int32_t x503 = x499;
for(int x504=0; x504 < 32; x504++) {
int32_t x505 = x502;
int32_t x506 = x503;
int32_t x507 = x501;
int32_t x508 = x507;
int32_t x509 = x505;
int32_t x510 = x506;
for(int x511=0; x511 < 32; x511++) {
int32_t x512 = x508;
int32_t x513 = x509;
float x514 = x419[x513];
int32_t x515 = x510;
float x516 = x476[x515];
float x517 = x514 / x516;
x486[x512] = x517;
x508 += 1;
x509 += 1;

}
x501 += 32;
x502 += 32;

}
x494 += 1024;
x495 += 1024;
x496 += 1;

}
x487 += 65536;
x488 += 65536;

}
// resize to WrappedArray(-1, 1, 1)
float* x537 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x538 = 0;
int32_t x539 = 0;
int32_t x540 = 0;
for(int x541=0; x541 < 64; x541++) {
int32_t x542 = x539;
int32_t x543 = x540;
int32_t x544 = x538;
int32_t x545 = x544;
int32_t x546 = x542;
int32_t x547 = x543;
for(int x548=0; x548 < 64; x548++) {
int32_t x549 = x546;
int32_t x550 = x547;
int32_t x551 = x545;
int32_t x552 = x551;
int32_t x553 = x549;
int32_t x554 = x550;
for(int x555=0; x555 < 32; x555++) {
int32_t x556 = x553;
int32_t x557 = x554;
int32_t x558 = x552;
int32_t x559 = x558;
int32_t x560 = x556;
int32_t x561 = x557;
for(int x562=0; x562 < 32; x562++) {
int32_t x563 = x559;
int32_t x564 = x560;
float x565 = x486[x564];
int32_t x566 = x561;
float x567 = x205[x566];
float x568 = x565 * x567;
x537[x563] = x568;
x559 += 1;
x560 += 1;

}
x552 += 32;
x553 += 32;

}
x545 += 1024;
x546 += 1024;
x547 += 1;

}
x538 += 65536;
x539 += 65536;

}
// resize to WrappedArray(-1, 1, 1)
float* x588 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x589 = 0;
int32_t x590 = 0;
int32_t x591 = 0;
for(int x592=0; x592 < 64; x592++) {
int32_t x593 = x590;
int32_t x594 = x591;
int32_t x595 = x589;
int32_t x596 = x595;
int32_t x597 = x593;
int32_t x598 = x594;
for(int x599=0; x599 < 64; x599++) {
int32_t x600 = x597;
int32_t x601 = x598;
int32_t x602 = x596;
int32_t x603 = x602;
int32_t x604 = x600;
int32_t x605 = x601;
for(int x606=0; x606 < 32; x606++) {
int32_t x607 = x604;
int32_t x608 = x605;
int32_t x609 = x603;
int32_t x610 = x609;
int32_t x611 = x607;
int32_t x612 = x608;
for(int x613=0; x613 < 32; x613++) {
int32_t x614 = x610;
int32_t x615 = x611;
float x616 = x537[x615];
int32_t x617 = x612;
float x618 = x250[x617];
float x619 = x616 + x618;
x588[x614] = x619;
x610 += 1;
x611 += 1;

}
x603 += 32;
x604 += 32;

}
x596 += 1024;
x597 += 1024;
x598 += 1;

}
x589 += 65536;
x590 += 65536;

}
float* x638 = (float*)myMalloc(4194304 * sizeof(float));;
for(int x640=0; x640 < 4194304; x640++) {
float x641 = x588[x640];
bool x642 = x641 < 0.0f;
if (x642) {
x638[x640] = 0.0f;
} else {
float x645 = x588[x640];
x638[x640] = x645;
}

}
float* x651 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x653=0; x653 < 1048576; x653++) {
x651[x653] = -3.4028235E38f;

}
int* x657 = (int32_t*)myMalloc(1048576 * sizeof(int32_t));;
for(int x658=0; x658 < 64; x658++) {
int32_t x659 = x658 * 65536;
float* x660 = x638+x659;
int32_t x661 = x658 * 16384;
float* x662 = x651+x661;
int* x663 = x657+x661;
int32_t x664 = 0;
int32_t x665 = 0;
for(int x666=0; x666 < 64; x666++) {
int32_t x667 = x664;
int32_t x668 = x667;
int32_t x669 = x665;
int32_t x670 = x669;
for(int x672=0; x672 < 16; x672++) {
int32_t x673 = x668;
int32_t x674 = x673;
int32_t x675 = x670;
int32_t x676 = x675;
for(int x677=0; x677 < 16; x677++) {
int32_t x678 = x676;
int32_t x679 = x678;
int32_t x680 = x679;
int32_t x681 = x680;
int32_t x682 = x681;
float x683 = x660[x682];
int32_t x684 = x674;
float x685 = x662[x684];
bool x686 = x683 > x685;
if (x686) {
float x687 = x660[x682];
x662[x684] = x687;
int32_t x689 = x682 + x659;
x663[x684] = x689;
} else {
}
x681 += 1;
int32_t x694 = x681;
float x695 = x660[x694];
float x696 = x662[x684];
bool x697 = x695 > x696;
if (x697) {
float x698 = x660[x694];
x662[x684] = x698;
int32_t x700 = x694 + x659;
x663[x684] = x700;
} else {
}
x681 += 1;
x679 += 32;
int32_t x706 = x679;
int32_t x707 = x706;
int32_t x708 = x707;
float x709 = x660[x708];
float x710 = x662[x684];
bool x711 = x709 > x710;
if (x711) {
float x712 = x660[x708];
x662[x684] = x712;
int32_t x714 = x708 + x659;
x663[x684] = x714;
} else {
}
x707 += 1;
int32_t x719 = x707;
float x720 = x660[x719];
float x721 = x662[x684];
bool x722 = x720 > x721;
if (x722) {
float x723 = x660[x719];
x662[x684] = x723;
int32_t x725 = x719 + x659;
x663[x684] = x725;
} else {
}
x707 += 1;
x679 += 32;
x674 += 1;
x676 += 2;

}
x668 += 16;
x670 += 64;

}
x664 += 256;
x665 += 1024;

}

}
float* x745 = (float*)myMalloc(1048576 * sizeof(float));;
float* x746 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x747=0; x747 < 64; x747++) {
int32_t x748 = x747 * 16384;
float* x749 = x651+x748;
float* x750 = x745+x748;
float* x751 = x746+x748;
for(int x752=0; x752 < 64; x752++) {
int32_t x753 = x752 / 1;
int32_t x757 = x753 * 16;
int32_t x758 = x757 * 16;
int32_t x754 = x752 % 1;
int32_t x755 = x754 / 1;
int32_t x759 = x755 * 16;
int32_t x760 = x759 * 16;
int32_t x761 = x758 + x760;
int32_t x756 = x754 % 1;
int32_t x762 = x756 * 16;
int32_t x763 = x762 * 16;
int32_t x764 = x761 + x763;
float* x765 = x751+x764;
float* x766 = x749+x758;
for(int x767=0; x767 < 16; x767++) {
int32_t x769 = x767 * 16;
float* x770 = x765+x769;
int32_t x768 = x767 + x755;
int32_t x771 = x768 * 16;
int32_t x772 = x771 + x756;
float* x773 = x766+x772;
memcpy(x770, x773, 4 * 16);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,256,64,1,x232,64,x751,256,1,x750,256);

}
// resize to WrappedArray(-1, 1, 1)
float* x783 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x784 = 0;
int32_t x785 = 0;
int32_t x786 = 0;
for(int x787=0; x787 < 64; x787++) {
int32_t x788 = x785;
int32_t x789 = x786;
int32_t x790 = x784;
int32_t x791 = x790;
int32_t x792 = x788;
int32_t x793 = x789;
for(int x794=0; x794 < 64; x794++) {
int32_t x795 = x792;
int32_t x796 = x793;
int32_t x797 = x791;
int32_t x798 = x797;
int32_t x799 = x795;
int32_t x800 = x796;
for(int x801=0; x801 < 16; x801++) {
int32_t x802 = x799;
int32_t x803 = x800;
int32_t x804 = x798;
int32_t x805 = x804;
int32_t x806 = x802;
int32_t x807 = x803;
for(int x808=0; x808 < 16; x808++) {
int32_t x809 = x805;
int32_t x810 = x806;
float x811 = x745[x810];
int32_t x812 = x807;
float x813 = x113[x812];
float x814 = x811 - x813;
x783[x809] = x814;
x805 += 1;
x806 += 1;

}
x798 += 16;
x799 += 16;

}
x791 += 256;
x792 += 256;
x793 += 1;

}
x784 += 16384;
x785 += 16384;

}
float* x833 = (float*)myMalloc(64 * sizeof(float));;
for(int x834=0; x834 < 64; x834++) {
float x835 = x50[x834];
float x836 = x835 + 1.0E-5f;
x833[x834] = x836;

}
float* x840 = (float*)myMalloc(64 * sizeof(float));;
for(int x841=0; x841 < 64; x841++) {
float x842 = x833[x841];
double x843 = (double)x842;
double x844 = sqrt(x843);
float x845 = (float)x844;
x840[x841] = x845;

}
// resize to WrappedArray(-1, 1, 1)
float* x850 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x851 = 0;
int32_t x852 = 0;
int32_t x853 = 0;
for(int x854=0; x854 < 64; x854++) {
int32_t x855 = x852;
int32_t x856 = x853;
int32_t x857 = x851;
int32_t x858 = x857;
int32_t x859 = x855;
int32_t x860 = x856;
for(int x861=0; x861 < 64; x861++) {
int32_t x862 = x859;
int32_t x863 = x860;
int32_t x864 = x858;
int32_t x865 = x864;
int32_t x866 = x862;
int32_t x867 = x863;
for(int x868=0; x868 < 16; x868++) {
int32_t x869 = x866;
int32_t x870 = x867;
int32_t x871 = x865;
int32_t x872 = x871;
int32_t x873 = x869;
int32_t x874 = x870;
for(int x875=0; x875 < 16; x875++) {
int32_t x876 = x872;
int32_t x877 = x873;
float x878 = x783[x877];
int32_t x879 = x874;
float x880 = x840[x879];
float x881 = x878 / x880;
x850[x876] = x881;
x872 += 1;
x873 += 1;

}
x865 += 16;
x866 += 16;

}
x858 += 256;
x859 += 256;
x860 += 1;

}
x851 += 16384;
x852 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x901 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x902 = 0;
int32_t x903 = 0;
int32_t x904 = 0;
for(int x905=0; x905 < 64; x905++) {
int32_t x906 = x903;
int32_t x907 = x904;
int32_t x908 = x902;
int32_t x909 = x908;
int32_t x910 = x906;
int32_t x911 = x907;
for(int x912=0; x912 < 64; x912++) {
int32_t x913 = x910;
int32_t x914 = x911;
int32_t x915 = x909;
int32_t x916 = x915;
int32_t x917 = x913;
int32_t x918 = x914;
for(int x919=0; x919 < 16; x919++) {
int32_t x920 = x917;
int32_t x921 = x918;
int32_t x922 = x916;
int32_t x923 = x922;
int32_t x924 = x920;
int32_t x925 = x921;
for(int x926=0; x926 < 16; x926++) {
int32_t x927 = x923;
int32_t x928 = x924;
float x929 = x850[x928];
int32_t x930 = x925;
float x931 = x25[x930];
float x932 = x929 * x931;
x901[x927] = x932;
x923 += 1;
x924 += 1;

}
x916 += 16;
x917 += 16;

}
x909 += 256;
x910 += 256;
x911 += 1;

}
x902 += 16384;
x903 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x952 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x953 = 0;
int32_t x954 = 0;
int32_t x955 = 0;
for(int x956=0; x956 < 64; x956++) {
int32_t x957 = x954;
int32_t x958 = x955;
int32_t x959 = x953;
int32_t x960 = x959;
int32_t x961 = x957;
int32_t x962 = x958;
for(int x963=0; x963 < 64; x963++) {
int32_t x964 = x961;
int32_t x965 = x962;
int32_t x966 = x960;
int32_t x967 = x966;
int32_t x968 = x964;
int32_t x969 = x965;
for(int x970=0; x970 < 16; x970++) {
int32_t x971 = x968;
int32_t x972 = x969;
int32_t x973 = x967;
int32_t x974 = x973;
int32_t x975 = x971;
int32_t x976 = x972;
for(int x977=0; x977 < 16; x977++) {
int32_t x978 = x974;
int32_t x979 = x975;
float x980 = x901[x979];
int32_t x981 = x976;
float x982 = x52[x981];
float x983 = x980 + x982;
x952[x978] = x983;
x974 += 1;
x975 += 1;

}
x967 += 16;
x968 += 16;

}
x960 += 256;
x961 += 256;
x962 += 1;

}
x953 += 16384;
x954 += 16384;

}
float* x1002 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x1003=0; x1003 < 1048576; x1003++) {
float x1004 = x952[x1003];
bool x1005 = x1004 < 0.0f;
if (x1005) {
x1002[x1003] = 0.0f;
} else {
float x1008 = x952[x1003];
x1002[x1003] = x1008;
}

}
float* x1014 = (float*)myMalloc(1048576 * sizeof(float));;
float* x1015 = (float*)myMalloc(9437184 * sizeof(float));;
for(int x1016=0; x1016 < 64; x1016++) {
int32_t x1017 = x1016 * 16384;
float* x1018 = x1002+x1017;
float* x1019 = x1014+x1017;
int32_t x1020 = x1016 * 147456;
float* x1021 = x1015+x1020;
for(int x1023=0; x1023 < 576; x1023++) {
int32_t x1024 = x1023 / 9;
int32_t x1028 = x1024 * 3;
int32_t x1029 = x1028 * 3;
int32_t x1030 = x1029 * 16;
int32_t x1031 = x1030 * 16;
int32_t x1025 = x1023 % 9;
int32_t x1026 = x1025 / 3;
int32_t x1032 = x1026 * 3;
int32_t x1033 = x1032 * 16;
int32_t x1034 = x1033 * 16;
int32_t x1035 = x1031 + x1034;
int32_t x1027 = x1025 % 3;
int32_t x1036 = x1027 * 16;
int32_t x1037 = x1036 * 16;
int32_t x1038 = x1035 + x1037;
float* x1039 = x1021+x1038;
int32_t x1040 = x1024 * 16;
int32_t x1041 = x1040 * 16;
float* x1042 = x1018+x1041;
int32_t x1054 = 1 - x1027;
bool x1055 = x1054 > 0;
int32_t x1056;
if (x1055) {
x1056 = x1054;
} else {
x1056 = 0;
}
int32_t x1057 = 3 - x1027;
int32_t x1058 = x1057 - 1;
int32_t x1059 = 1 - x1058;
bool x1060 = x1059 > 0;
int32_t x1061;
if (x1060) {
x1061 = x1059;
} else {
x1061 = 0;
}
int32_t x1062 = 16 - x1061;
int32_t x1063 = x1062 - x1056;
bool x1064 = x1063 <= 0;
bool x1068 = x1056 > 0;
int32_t x1053 = -1 + x1027;
bool x1081 = x1061 > 0;
for(int x1043=0; x1043 < 16; x1043++) {
int32_t x1044 = x1043 - 1;
int32_t x1045 = x1044 + x1026;
bool x1046 = x1045 < 0;
bool x1047 = x1045 >= 16;
bool x1048 = x1046 || x1047;
if (x1048) {
int32_t x1049 = x1043 * 16;
float* x1050 = x1039+x1049;
memset(x1050, 0, 4 * 16);;
} else {
if (x1064) {
int32_t x1049 = x1043 * 16;
float* x1065 = x1039+x1049;
memset(x1065, 0, 4 * 16);;
} else {
int32_t x1049 = x1043 * 16;
if (x1068) {
float* x1069 = x1039+x1049;
memset(x1069, 0, 4 * x1056);;
} else {
}
// may have segfault here
int32_t x1074 = x1049 + x1056;
float* x1075 = x1039+x1074;
int32_t x1076 = x1045 * 16;
int32_t x1077 = x1076 + x1053;
int32_t x1078 = x1077 + x1056;
float* x1079 = x1042+x1078;
memcpy(x1075, x1079, 4 * x1063);;
if (x1081) {
int32_t x1082 = x1049 + 16;
int32_t x1083 = x1082 - x1061;
float* x1084 = x1039+x1083;
memset(x1084, 0, 4 * x1061);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,256,576,1,x89,576,x1021,256,1,x1019,256);

}
// resize to WrappedArray(-1, 1, 1)
float* x1100 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x1101 = 0;
int32_t x1102 = 0;
int32_t x1103 = 0;
for(int x1104=0; x1104 < 64; x1104++) {
int32_t x1105 = x1102;
int32_t x1106 = x1103;
int32_t x1107 = x1101;
int32_t x1108 = x1107;
int32_t x1109 = x1105;
int32_t x1110 = x1106;
for(int x1111=0; x1111 < 64; x1111++) {
int32_t x1112 = x1109;
int32_t x1113 = x1110;
int32_t x1114 = x1108;
int32_t x1115 = x1114;
int32_t x1116 = x1112;
int32_t x1117 = x1113;
for(int x1118=0; x1118 < 16; x1118++) {
int32_t x1119 = x1116;
int32_t x1120 = x1117;
int32_t x1121 = x1115;
int32_t x1122 = x1121;
int32_t x1123 = x1119;
int32_t x1124 = x1120;
for(int x1125=0; x1125 < 16; x1125++) {
int32_t x1126 = x1122;
int32_t x1127 = x1123;
float x1128 = x1014[x1127];
int32_t x1129 = x1124;
float x1130 = x104[x1129];
float x1131 = x1128 - x1130;
x1100[x1126] = x1131;
x1122 += 1;
x1123 += 1;

}
x1115 += 16;
x1116 += 16;

}
x1108 += 256;
x1109 += 256;
x1110 += 1;

}
x1101 += 16384;
x1102 += 16384;

}
float* x1150 = (float*)myMalloc(64 * sizeof(float));;
for(int x1151=0; x1151 < 64; x1151++) {
float x1152 = x157[x1151];
float x1153 = x1152 + 1.0E-5f;
x1150[x1151] = x1153;

}
float* x1157 = (float*)myMalloc(64 * sizeof(float));;
for(int x1158=0; x1158 < 64; x1158++) {
float x1159 = x1150[x1158];
double x1160 = (double)x1159;
double x1161 = sqrt(x1160);
float x1162 = (float)x1161;
x1157[x1158] = x1162;

}
// resize to WrappedArray(-1, 1, 1)
float* x1167 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x1168 = 0;
int32_t x1169 = 0;
int32_t x1170 = 0;
for(int x1171=0; x1171 < 64; x1171++) {
int32_t x1172 = x1169;
int32_t x1173 = x1170;
int32_t x1174 = x1168;
int32_t x1175 = x1174;
int32_t x1176 = x1172;
int32_t x1177 = x1173;
for(int x1178=0; x1178 < 64; x1178++) {
int32_t x1179 = x1176;
int32_t x1180 = x1177;
int32_t x1181 = x1175;
int32_t x1182 = x1181;
int32_t x1183 = x1179;
int32_t x1184 = x1180;
for(int x1185=0; x1185 < 16; x1185++) {
int32_t x1186 = x1183;
int32_t x1187 = x1184;
int32_t x1188 = x1182;
int32_t x1189 = x1188;
int32_t x1190 = x1186;
int32_t x1191 = x1187;
for(int x1192=0; x1192 < 16; x1192++) {
int32_t x1193 = x1189;
int32_t x1194 = x1190;
float x1195 = x1100[x1194];
int32_t x1196 = x1191;
float x1197 = x1157[x1196];
float x1198 = x1195 / x1197;
x1167[x1193] = x1198;
x1189 += 1;
x1190 += 1;

}
x1182 += 16;
x1183 += 16;

}
x1175 += 256;
x1176 += 256;
x1177 += 1;

}
x1168 += 16384;
x1169 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x1218 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x1219 = 0;
int32_t x1220 = 0;
int32_t x1221 = 0;
for(int x1222=0; x1222 < 64; x1222++) {
int32_t x1223 = x1220;
int32_t x1224 = x1221;
int32_t x1225 = x1219;
int32_t x1226 = x1225;
int32_t x1227 = x1223;
int32_t x1228 = x1224;
for(int x1229=0; x1229 < 64; x1229++) {
int32_t x1230 = x1227;
int32_t x1231 = x1228;
int32_t x1232 = x1226;
int32_t x1233 = x1232;
int32_t x1234 = x1230;
int32_t x1235 = x1231;
for(int x1236=0; x1236 < 16; x1236++) {
int32_t x1237 = x1234;
int32_t x1238 = x1235;
int32_t x1239 = x1233;
int32_t x1240 = x1239;
int32_t x1241 = x1237;
int32_t x1242 = x1238;
for(int x1243=0; x1243 < 16; x1243++) {
int32_t x1244 = x1240;
int32_t x1245 = x1241;
float x1246 = x1167[x1245];
int32_t x1247 = x1242;
float x1248 = x163[x1247];
float x1249 = x1246 * x1248;
x1218[x1244] = x1249;
x1240 += 1;
x1241 += 1;

}
x1233 += 16;
x1234 += 16;

}
x1226 += 256;
x1227 += 256;
x1228 += 1;

}
x1219 += 16384;
x1220 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x1269 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x1270 = 0;
int32_t x1271 = 0;
int32_t x1272 = 0;
for(int x1273=0; x1273 < 64; x1273++) {
int32_t x1274 = x1271;
int32_t x1275 = x1272;
int32_t x1276 = x1270;
int32_t x1277 = x1276;
int32_t x1278 = x1274;
int32_t x1279 = x1275;
for(int x1280=0; x1280 < 64; x1280++) {
int32_t x1281 = x1278;
int32_t x1282 = x1279;
int32_t x1283 = x1277;
int32_t x1284 = x1283;
int32_t x1285 = x1281;
int32_t x1286 = x1282;
for(int x1287=0; x1287 < 16; x1287++) {
int32_t x1288 = x1285;
int32_t x1289 = x1286;
int32_t x1290 = x1284;
int32_t x1291 = x1290;
int32_t x1292 = x1288;
int32_t x1293 = x1289;
for(int x1294=0; x1294 < 16; x1294++) {
int32_t x1295 = x1291;
int32_t x1296 = x1292;
float x1297 = x1218[x1296];
int32_t x1298 = x1293;
float x1299 = x48[x1298];
float x1300 = x1297 + x1299;
x1269[x1295] = x1300;
x1291 += 1;
x1292 += 1;

}
x1284 += 16;
x1285 += 16;

}
x1277 += 256;
x1278 += 256;
x1279 += 1;

}
x1270 += 16384;
x1271 += 16384;

}
float* x1319 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x1320=0; x1320 < 1048576; x1320++) {
float x1321 = x1269[x1320];
bool x1322 = x1321 < 0.0f;
if (x1322) {
x1319[x1320] = 0.0f;
} else {
float x1325 = x1269[x1320];
x1319[x1320] = x1325;
}

}
float* x1331 = (float*)myMalloc(4194304 * sizeof(float));;
float* x1332 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x1333=0; x1333 < 64; x1333++) {
int32_t x1334 = x1333 * 16384;
float* x1335 = x1319+x1334;
int32_t x1336 = x1333 * 65536;
float* x1337 = x1331+x1336;
float* x1338 = x1332+x1334;
for(int x1339=0; x1339 < 64; x1339++) {
int32_t x1340 = x1339 / 1;
int32_t x1344 = x1340 * 16;
int32_t x1345 = x1344 * 16;
int32_t x1341 = x1339 % 1;
int32_t x1342 = x1341 / 1;
int32_t x1346 = x1342 * 16;
int32_t x1347 = x1346 * 16;
int32_t x1348 = x1345 + x1347;
int32_t x1343 = x1341 % 1;
int32_t x1349 = x1343 * 16;
int32_t x1350 = x1349 * 16;
int32_t x1351 = x1348 + x1350;
float* x1352 = x1338+x1351;
float* x1353 = x1335+x1345;
for(int x1354=0; x1354 < 16; x1354++) {
int32_t x1356 = x1354 * 16;
float* x1357 = x1352+x1356;
int32_t x1355 = x1354 + x1342;
int32_t x1358 = x1355 * 16;
int32_t x1359 = x1358 + x1343;
float* x1360 = x1353+x1359;
memcpy(x1357, x1360, 4 * 16);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,256,64,1,x31,64,x1338,256,1,x1337,256);

}
// resize to WrappedArray(-1, 1, 1)
float* x1370 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x1371 = 0;
int32_t x1372 = 0;
int32_t x1373 = 0;
for(int x1374=0; x1374 < 64; x1374++) {
int32_t x1375 = x1372;
int32_t x1376 = x1373;
int32_t x1377 = x1371;
int32_t x1378 = x1377;
int32_t x1379 = x1375;
int32_t x1380 = x1376;
for(int x1382=0; x1382 < 256; x1382++) {
int32_t x1383 = x1379;
int32_t x1384 = x1380;
int32_t x1385 = x1378;
int32_t x1386 = x1385;
int32_t x1387 = x1383;
int32_t x1388 = x1384;
for(int x1389=0; x1389 < 16; x1389++) {
int32_t x1390 = x1387;
int32_t x1391 = x1388;
int32_t x1392 = x1386;
int32_t x1393 = x1392;
int32_t x1394 = x1390;
int32_t x1395 = x1391;
for(int x1396=0; x1396 < 16; x1396++) {
int32_t x1397 = x1393;
int32_t x1398 = x1394;
float x1399 = x1331[x1398];
int32_t x1400 = x1395;
float x1401 = x70[x1400];
float x1402 = x1399 - x1401;
x1370[x1397] = x1402;
x1393 += 1;
x1394 += 1;

}
x1386 += 16;
x1387 += 16;

}
x1378 += 256;
x1379 += 256;
x1380 += 1;

}
x1371 += 65536;
x1372 += 65536;

}
float* x1421 = (float*)myMalloc(256 * sizeof(float));;
for(int x1422=0; x1422 < 256; x1422++) {
float x1423 = x35[x1422];
float x1424 = x1423 + 1.0E-5f;
x1421[x1422] = x1424;

}
float* x1428 = (float*)myMalloc(256 * sizeof(float));;
for(int x1429=0; x1429 < 256; x1429++) {
float x1430 = x1421[x1429];
double x1431 = (double)x1430;
double x1432 = sqrt(x1431);
float x1433 = (float)x1432;
x1428[x1429] = x1433;

}
// resize to WrappedArray(-1, 1, 1)
float* x1438 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x1439 = 0;
int32_t x1440 = 0;
int32_t x1441 = 0;
for(int x1442=0; x1442 < 64; x1442++) {
int32_t x1443 = x1440;
int32_t x1444 = x1441;
int32_t x1445 = x1439;
int32_t x1446 = x1445;
int32_t x1447 = x1443;
int32_t x1448 = x1444;
for(int x1449=0; x1449 < 256; x1449++) {
int32_t x1450 = x1447;
int32_t x1451 = x1448;
int32_t x1452 = x1446;
int32_t x1453 = x1452;
int32_t x1454 = x1450;
int32_t x1455 = x1451;
for(int x1456=0; x1456 < 16; x1456++) {
int32_t x1457 = x1454;
int32_t x1458 = x1455;
int32_t x1459 = x1453;
int32_t x1460 = x1459;
int32_t x1461 = x1457;
int32_t x1462 = x1458;
for(int x1463=0; x1463 < 16; x1463++) {
int32_t x1464 = x1460;
int32_t x1465 = x1461;
float x1466 = x1370[x1465];
int32_t x1467 = x1462;
float x1468 = x1428[x1467];
float x1469 = x1466 / x1468;
x1438[x1464] = x1469;
x1460 += 1;
x1461 += 1;

}
x1453 += 16;
x1454 += 16;

}
x1446 += 256;
x1447 += 256;
x1448 += 1;

}
x1439 += 65536;
x1440 += 65536;

}
// resize to WrappedArray(-1, 1, 1)
float* x1489 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x1490 = 0;
int32_t x1491 = 0;
int32_t x1492 = 0;
for(int x1493=0; x1493 < 64; x1493++) {
int32_t x1494 = x1491;
int32_t x1495 = x1492;
int32_t x1496 = x1490;
int32_t x1497 = x1496;
int32_t x1498 = x1494;
int32_t x1499 = x1495;
for(int x1500=0; x1500 < 256; x1500++) {
int32_t x1501 = x1498;
int32_t x1502 = x1499;
int32_t x1503 = x1497;
int32_t x1504 = x1503;
int32_t x1505 = x1501;
int32_t x1506 = x1502;
for(int x1507=0; x1507 < 16; x1507++) {
int32_t x1508 = x1505;
int32_t x1509 = x1506;
int32_t x1510 = x1504;
int32_t x1511 = x1510;
int32_t x1512 = x1508;
int32_t x1513 = x1509;
for(int x1514=0; x1514 < 16; x1514++) {
int32_t x1515 = x1511;
int32_t x1516 = x1512;
float x1517 = x1438[x1516];
int32_t x1518 = x1513;
float x1519 = x198[x1518];
float x1520 = x1517 * x1519;
x1489[x1515] = x1520;
x1511 += 1;
x1512 += 1;

}
x1504 += 16;
x1505 += 16;

}
x1497 += 256;
x1498 += 256;
x1499 += 1;

}
x1490 += 65536;
x1491 += 65536;

}
// resize to WrappedArray(-1, 1, 1)
float* x1540 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x1541 = 0;
int32_t x1542 = 0;
int32_t x1543 = 0;
for(int x1544=0; x1544 < 64; x1544++) {
int32_t x1545 = x1542;
int32_t x1546 = x1543;
int32_t x1547 = x1541;
int32_t x1548 = x1547;
int32_t x1549 = x1545;
int32_t x1550 = x1546;
for(int x1551=0; x1551 < 256; x1551++) {
int32_t x1552 = x1549;
int32_t x1553 = x1550;
int32_t x1554 = x1548;
int32_t x1555 = x1554;
int32_t x1556 = x1552;
int32_t x1557 = x1553;
for(int x1558=0; x1558 < 16; x1558++) {
int32_t x1559 = x1556;
int32_t x1560 = x1557;
int32_t x1561 = x1555;
int32_t x1562 = x1561;
int32_t x1563 = x1559;
int32_t x1564 = x1560;
for(int x1565=0; x1565 < 16; x1565++) {
int32_t x1566 = x1562;
int32_t x1567 = x1563;
float x1568 = x1489[x1567];
int32_t x1569 = x1564;
float x1570 = x125[x1569];
float x1571 = x1568 + x1570;
x1540[x1566] = x1571;
x1562 += 1;
x1563 += 1;

}
x1555 += 16;
x1556 += 16;

}
x1548 += 256;
x1549 += 256;
x1550 += 1;

}
x1541 += 65536;
x1542 += 65536;

}
float* x1590 = (float*)myMalloc(4194304 * sizeof(float));;
float* x1591 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x1592=0; x1592 < 64; x1592++) {
int32_t x1593 = x1592 * 16384;
float* x1594 = x651+x1593;
int32_t x1595 = x1592 * 65536;
float* x1596 = x1590+x1595;
float* x1597 = x1591+x1593;
for(int x1598=0; x1598 < 64; x1598++) {
int32_t x1599 = x1598 / 1;
int32_t x1603 = x1599 * 16;
int32_t x1604 = x1603 * 16;
int32_t x1600 = x1598 % 1;
int32_t x1601 = x1600 / 1;
int32_t x1605 = x1601 * 16;
int32_t x1606 = x1605 * 16;
int32_t x1607 = x1604 + x1606;
int32_t x1602 = x1600 % 1;
int32_t x1608 = x1602 * 16;
int32_t x1609 = x1608 * 16;
int32_t x1610 = x1607 + x1609;
float* x1611 = x1597+x1610;
float* x1612 = x1594+x1604;
for(int x1613=0; x1613 < 16; x1613++) {
int32_t x1615 = x1613 * 16;
float* x1616 = x1611+x1615;
int32_t x1614 = x1613 + x1601;
int32_t x1617 = x1614 * 16;
int32_t x1618 = x1617 + x1602;
float* x1619 = x1612+x1618;
memcpy(x1616, x1619, 4 * 16);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,256,64,1,x161,64,x1597,256,1,x1596,256);

}
// resize to WrappedArray(-1, 1, 1)
float* x1629 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x1630 = 0;
int32_t x1631 = 0;
int32_t x1632 = 0;
for(int x1633=0; x1633 < 64; x1633++) {
int32_t x1634 = x1631;
int32_t x1635 = x1632;
int32_t x1636 = x1630;
int32_t x1637 = x1636;
int32_t x1638 = x1634;
int32_t x1639 = x1635;
for(int x1640=0; x1640 < 256; x1640++) {
int32_t x1641 = x1638;
int32_t x1642 = x1639;
int32_t x1643 = x1637;
int32_t x1644 = x1643;
int32_t x1645 = x1641;
int32_t x1646 = x1642;
for(int x1647=0; x1647 < 16; x1647++) {
int32_t x1648 = x1645;
int32_t x1649 = x1646;
int32_t x1650 = x1644;
int32_t x1651 = x1650;
int32_t x1652 = x1648;
int32_t x1653 = x1649;
for(int x1654=0; x1654 < 16; x1654++) {
int32_t x1655 = x1651;
int32_t x1656 = x1652;
float x1657 = x1590[x1656];
int32_t x1658 = x1653;
float x1659 = x263[x1658];
float x1660 = x1657 - x1659;
x1629[x1655] = x1660;
x1651 += 1;
x1652 += 1;

}
x1644 += 16;
x1645 += 16;

}
x1637 += 256;
x1638 += 256;
x1639 += 1;

}
x1630 += 65536;
x1631 += 65536;

}
float* x1679 = (float*)myMalloc(256 * sizeof(float));;
for(int x1680=0; x1680 < 256; x1680++) {
float x1681 = x242[x1680];
float x1682 = x1681 + 1.0E-5f;
x1679[x1680] = x1682;

}
float* x1686 = (float*)myMalloc(256 * sizeof(float));;
for(int x1687=0; x1687 < 256; x1687++) {
float x1688 = x1679[x1687];
double x1689 = (double)x1688;
double x1690 = sqrt(x1689);
float x1691 = (float)x1690;
x1686[x1687] = x1691;

}
// resize to WrappedArray(-1, 1, 1)
float* x1696 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x1697 = 0;
int32_t x1698 = 0;
int32_t x1699 = 0;
for(int x1700=0; x1700 < 64; x1700++) {
int32_t x1701 = x1698;
int32_t x1702 = x1699;
int32_t x1703 = x1697;
int32_t x1704 = x1703;
int32_t x1705 = x1701;
int32_t x1706 = x1702;
for(int x1707=0; x1707 < 256; x1707++) {
int32_t x1708 = x1705;
int32_t x1709 = x1706;
int32_t x1710 = x1704;
int32_t x1711 = x1710;
int32_t x1712 = x1708;
int32_t x1713 = x1709;
for(int x1714=0; x1714 < 16; x1714++) {
int32_t x1715 = x1712;
int32_t x1716 = x1713;
int32_t x1717 = x1711;
int32_t x1718 = x1717;
int32_t x1719 = x1715;
int32_t x1720 = x1716;
for(int x1721=0; x1721 < 16; x1721++) {
int32_t x1722 = x1718;
int32_t x1723 = x1719;
float x1724 = x1629[x1723];
int32_t x1725 = x1720;
float x1726 = x1686[x1725];
float x1727 = x1724 / x1726;
x1696[x1722] = x1727;
x1718 += 1;
x1719 += 1;

}
x1711 += 16;
x1712 += 16;

}
x1704 += 256;
x1705 += 256;
x1706 += 1;

}
x1697 += 65536;
x1698 += 65536;

}
// resize to WrappedArray(-1, 1, 1)
float* x1747 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x1748 = 0;
int32_t x1749 = 0;
int32_t x1750 = 0;
for(int x1751=0; x1751 < 64; x1751++) {
int32_t x1752 = x1749;
int32_t x1753 = x1750;
int32_t x1754 = x1748;
int32_t x1755 = x1754;
int32_t x1756 = x1752;
int32_t x1757 = x1753;
for(int x1758=0; x1758 < 256; x1758++) {
int32_t x1759 = x1756;
int32_t x1760 = x1757;
int32_t x1761 = x1755;
int32_t x1762 = x1761;
int32_t x1763 = x1759;
int32_t x1764 = x1760;
for(int x1765=0; x1765 < 16; x1765++) {
int32_t x1766 = x1763;
int32_t x1767 = x1764;
int32_t x1768 = x1762;
int32_t x1769 = x1768;
int32_t x1770 = x1766;
int32_t x1771 = x1767;
for(int x1772=0; x1772 < 16; x1772++) {
int32_t x1773 = x1769;
int32_t x1774 = x1770;
float x1775 = x1696[x1774];
int32_t x1776 = x1771;
float x1777 = x75[x1776];
float x1778 = x1775 * x1777;
x1747[x1773] = x1778;
x1769 += 1;
x1770 += 1;

}
x1762 += 16;
x1763 += 16;

}
x1755 += 256;
x1756 += 256;
x1757 += 1;

}
x1748 += 65536;
x1749 += 65536;

}
// resize to WrappedArray(-1, 1, 1)
float* x1798 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x1799 = 0;
int32_t x1800 = 0;
int32_t x1801 = 0;
for(int x1802=0; x1802 < 64; x1802++) {
int32_t x1803 = x1800;
int32_t x1804 = x1801;
int32_t x1805 = x1799;
int32_t x1806 = x1805;
int32_t x1807 = x1803;
int32_t x1808 = x1804;
for(int x1809=0; x1809 < 256; x1809++) {
int32_t x1810 = x1807;
int32_t x1811 = x1808;
int32_t x1812 = x1806;
int32_t x1813 = x1812;
int32_t x1814 = x1810;
int32_t x1815 = x1811;
for(int x1816=0; x1816 < 16; x1816++) {
int32_t x1817 = x1814;
int32_t x1818 = x1815;
int32_t x1819 = x1813;
int32_t x1820 = x1819;
int32_t x1821 = x1817;
int32_t x1822 = x1818;
for(int x1823=0; x1823 < 16; x1823++) {
int32_t x1824 = x1820;
int32_t x1825 = x1821;
float x1826 = x1747[x1825];
int32_t x1827 = x1822;
float x1828 = x202[x1827];
float x1829 = x1826 + x1828;
x1798[x1824] = x1829;
x1820 += 1;
x1821 += 1;

}
x1813 += 16;
x1814 += 16;

}
x1806 += 256;
x1807 += 256;
x1808 += 1;

}
x1799 += 65536;
x1800 += 65536;

}
int32_t x1848 = 0;
int32_t x1849 = 0;
int32_t x1850 = 0;
for(int x1851=0; x1851 < 64; x1851++) {
int32_t x1852 = x1849;
int32_t x1853 = x1850;
int32_t x1854 = x1848;
int32_t x1855 = x1854;
int32_t x1856 = x1852;
int32_t x1857 = x1853;
for(int x1858=0; x1858 < 256; x1858++) {
int32_t x1859 = x1856;
int32_t x1860 = x1857;
int32_t x1861 = x1855;
int32_t x1862 = x1861;
int32_t x1863 = x1859;
int32_t x1864 = x1860;
for(int x1865=0; x1865 < 16; x1865++) {
int32_t x1866 = x1863;
int32_t x1867 = x1864;
int32_t x1868 = x1862;
int32_t x1869 = x1868;
int32_t x1870 = x1866;
int32_t x1871 = x1867;
for(int x1872=0; x1872 < 16; x1872++) {
int32_t x1873 = x1870;
float x1874 = x1540[x1873];
int32_t x1875 = x1871;
float x1876 = x1798[x1875];
float x1877 = x1874 + x1876;
x1540[x1873] = x1877;
x1869 += 1;
x1870 += 1;
x1871 += 1;

}
x1862 += 16;
x1863 += 16;
x1864 += 16;

}
x1855 += 256;
x1856 += 256;
x1857 += 256;

}
x1848 += 65536;
x1849 += 65536;
x1850 += 65536;

}
float* x1899 = (float*)myMalloc(4194304 * sizeof(float));;
for(int x1900=0; x1900 < 4194304; x1900++) {
float x1901 = x1540[x1900];
bool x1902 = x1901 < 0.0f;
if (x1902) {
x1899[x1900] = 0.0f;
} else {
float x1905 = x1540[x1900];
x1899[x1900] = x1905;
}

}
float* x1911 = (float*)myMalloc(1048576 * sizeof(float));;
float* x1912 = (float*)myMalloc(4194304 * sizeof(float));;
for(int x1913=0; x1913 < 64; x1913++) {
int32_t x1914 = x1913 * 65536;
float* x1915 = x1899+x1914;
int32_t x1916 = x1913 * 16384;
float* x1917 = x1911+x1916;
float* x1918 = x1912+x1914;
for(int x1919=0; x1919 < 256; x1919++) {
int32_t x1920 = x1919 / 1;
int32_t x1924 = x1920 * 16;
int32_t x1925 = x1924 * 16;
int32_t x1921 = x1919 % 1;
int32_t x1922 = x1921 / 1;
int32_t x1926 = x1922 * 16;
int32_t x1927 = x1926 * 16;
int32_t x1928 = x1925 + x1927;
int32_t x1923 = x1921 % 1;
int32_t x1929 = x1923 * 16;
int32_t x1930 = x1929 * 16;
int32_t x1931 = x1928 + x1930;
float* x1932 = x1918+x1931;
float* x1933 = x1915+x1925;
for(int x1934=0; x1934 < 16; x1934++) {
int32_t x1936 = x1934 * 16;
float* x1937 = x1932+x1936;
int32_t x1935 = x1934 + x1922;
int32_t x1938 = x1935 * 16;
int32_t x1939 = x1938 + x1923;
float* x1940 = x1933+x1939;
memcpy(x1937, x1940, 4 * 16);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,256,256,1,x170,256,x1918,256,1,x1917,256);

}
// resize to WrappedArray(-1, 1, 1)
float* x1950 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x1951 = 0;
int32_t x1952 = 0;
int32_t x1953 = 0;
for(int x1954=0; x1954 < 64; x1954++) {
int32_t x1955 = x1952;
int32_t x1956 = x1953;
int32_t x1957 = x1951;
int32_t x1958 = x1957;
int32_t x1959 = x1955;
int32_t x1960 = x1956;
for(int x1961=0; x1961 < 64; x1961++) {
int32_t x1962 = x1959;
int32_t x1963 = x1960;
int32_t x1964 = x1958;
int32_t x1965 = x1964;
int32_t x1966 = x1962;
int32_t x1967 = x1963;
for(int x1968=0; x1968 < 16; x1968++) {
int32_t x1969 = x1966;
int32_t x1970 = x1967;
int32_t x1971 = x1965;
int32_t x1972 = x1971;
int32_t x1973 = x1969;
int32_t x1974 = x1970;
for(int x1975=0; x1975 < 16; x1975++) {
int32_t x1976 = x1972;
int32_t x1977 = x1973;
float x1978 = x1911[x1977];
int32_t x1979 = x1974;
float x1980 = x9[x1979];
float x1981 = x1978 - x1980;
x1950[x1976] = x1981;
x1972 += 1;
x1973 += 1;

}
x1965 += 16;
x1966 += 16;

}
x1958 += 256;
x1959 += 256;
x1960 += 1;

}
x1951 += 16384;
x1952 += 16384;

}
float* x2000 = (float*)myMalloc(64 * sizeof(float));;
for(int x2001=0; x2001 < 64; x2001++) {
float x2002 = x101[x2001];
float x2003 = x2002 + 1.0E-5f;
x2000[x2001] = x2003;

}
float* x2007 = (float*)myMalloc(64 * sizeof(float));;
for(int x2008=0; x2008 < 64; x2008++) {
float x2009 = x2000[x2008];
double x2010 = (double)x2009;
double x2011 = sqrt(x2010);
float x2012 = (float)x2011;
x2007[x2008] = x2012;

}
// resize to WrappedArray(-1, 1, 1)
float* x2017 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x2018 = 0;
int32_t x2019 = 0;
int32_t x2020 = 0;
for(int x2021=0; x2021 < 64; x2021++) {
int32_t x2022 = x2019;
int32_t x2023 = x2020;
int32_t x2024 = x2018;
int32_t x2025 = x2024;
int32_t x2026 = x2022;
int32_t x2027 = x2023;
for(int x2028=0; x2028 < 64; x2028++) {
int32_t x2029 = x2026;
int32_t x2030 = x2027;
int32_t x2031 = x2025;
int32_t x2032 = x2031;
int32_t x2033 = x2029;
int32_t x2034 = x2030;
for(int x2035=0; x2035 < 16; x2035++) {
int32_t x2036 = x2033;
int32_t x2037 = x2034;
int32_t x2038 = x2032;
int32_t x2039 = x2038;
int32_t x2040 = x2036;
int32_t x2041 = x2037;
for(int x2042=0; x2042 < 16; x2042++) {
int32_t x2043 = x2039;
int32_t x2044 = x2040;
float x2045 = x1950[x2044];
int32_t x2046 = x2041;
float x2047 = x2007[x2046];
float x2048 = x2045 / x2047;
x2017[x2043] = x2048;
x2039 += 1;
x2040 += 1;

}
x2032 += 16;
x2033 += 16;

}
x2025 += 256;
x2026 += 256;
x2027 += 1;

}
x2018 += 16384;
x2019 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x2068 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x2069 = 0;
int32_t x2070 = 0;
int32_t x2071 = 0;
for(int x2072=0; x2072 < 64; x2072++) {
int32_t x2073 = x2070;
int32_t x2074 = x2071;
int32_t x2075 = x2069;
int32_t x2076 = x2075;
int32_t x2077 = x2073;
int32_t x2078 = x2074;
for(int x2079=0; x2079 < 64; x2079++) {
int32_t x2080 = x2077;
int32_t x2081 = x2078;
int32_t x2082 = x2076;
int32_t x2083 = x2082;
int32_t x2084 = x2080;
int32_t x2085 = x2081;
for(int x2086=0; x2086 < 16; x2086++) {
int32_t x2087 = x2084;
int32_t x2088 = x2085;
int32_t x2089 = x2083;
int32_t x2090 = x2089;
int32_t x2091 = x2087;
int32_t x2092 = x2088;
for(int x2093=0; x2093 < 16; x2093++) {
int32_t x2094 = x2090;
int32_t x2095 = x2091;
float x2096 = x2017[x2095];
int32_t x2097 = x2092;
float x2098 = x141[x2097];
float x2099 = x2096 * x2098;
x2068[x2094] = x2099;
x2090 += 1;
x2091 += 1;

}
x2083 += 16;
x2084 += 16;

}
x2076 += 256;
x2077 += 256;
x2078 += 1;

}
x2069 += 16384;
x2070 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x2119 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x2120 = 0;
int32_t x2121 = 0;
int32_t x2122 = 0;
for(int x2123=0; x2123 < 64; x2123++) {
int32_t x2124 = x2121;
int32_t x2125 = x2122;
int32_t x2126 = x2120;
int32_t x2127 = x2126;
int32_t x2128 = x2124;
int32_t x2129 = x2125;
for(int x2130=0; x2130 < 64; x2130++) {
int32_t x2131 = x2128;
int32_t x2132 = x2129;
int32_t x2133 = x2127;
int32_t x2134 = x2133;
int32_t x2135 = x2131;
int32_t x2136 = x2132;
for(int x2137=0; x2137 < 16; x2137++) {
int32_t x2138 = x2135;
int32_t x2139 = x2136;
int32_t x2140 = x2134;
int32_t x2141 = x2140;
int32_t x2142 = x2138;
int32_t x2143 = x2139;
for(int x2144=0; x2144 < 16; x2144++) {
int32_t x2145 = x2141;
int32_t x2146 = x2142;
float x2147 = x2068[x2146];
int32_t x2148 = x2143;
float x2149 = x59[x2148];
float x2150 = x2147 + x2149;
x2119[x2145] = x2150;
x2141 += 1;
x2142 += 1;

}
x2134 += 16;
x2135 += 16;

}
x2127 += 256;
x2128 += 256;
x2129 += 1;

}
x2120 += 16384;
x2121 += 16384;

}
float* x2169 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x2170=0; x2170 < 1048576; x2170++) {
float x2171 = x2119[x2170];
bool x2172 = x2171 < 0.0f;
if (x2172) {
x2169[x2170] = 0.0f;
} else {
float x2175 = x2119[x2170];
x2169[x2170] = x2175;
}

}
float* x2181 = (float*)myMalloc(1048576 * sizeof(float));;
float* x2182 = (float*)myMalloc(9437184 * sizeof(float));;
for(int x2183=0; x2183 < 64; x2183++) {
int32_t x2184 = x2183 * 16384;
float* x2185 = x2169+x2184;
float* x2186 = x2181+x2184;
int32_t x2187 = x2183 * 147456;
float* x2188 = x2182+x2187;
for(int x2189=0; x2189 < 576; x2189++) {
int32_t x2190 = x2189 / 9;
int32_t x2194 = x2190 * 3;
int32_t x2195 = x2194 * 3;
int32_t x2196 = x2195 * 16;
int32_t x2197 = x2196 * 16;
int32_t x2191 = x2189 % 9;
int32_t x2192 = x2191 / 3;
int32_t x2198 = x2192 * 3;
int32_t x2199 = x2198 * 16;
int32_t x2200 = x2199 * 16;
int32_t x2201 = x2197 + x2200;
int32_t x2193 = x2191 % 3;
int32_t x2202 = x2193 * 16;
int32_t x2203 = x2202 * 16;
int32_t x2204 = x2201 + x2203;
float* x2205 = x2188+x2204;
int32_t x2206 = x2190 * 16;
int32_t x2207 = x2206 * 16;
float* x2208 = x2185+x2207;
int32_t x2220 = 1 - x2193;
bool x2221 = x2220 > 0;
int32_t x2222;
if (x2221) {
x2222 = x2220;
} else {
x2222 = 0;
}
int32_t x2223 = 3 - x2193;
int32_t x2224 = x2223 - 1;
int32_t x2225 = 1 - x2224;
bool x2226 = x2225 > 0;
int32_t x2227;
if (x2226) {
x2227 = x2225;
} else {
x2227 = 0;
}
int32_t x2228 = 16 - x2227;
int32_t x2229 = x2228 - x2222;
bool x2230 = x2229 <= 0;
bool x2234 = x2222 > 0;
int32_t x2219 = -1 + x2193;
bool x2247 = x2227 > 0;
for(int x2209=0; x2209 < 16; x2209++) {
int32_t x2210 = x2209 - 1;
int32_t x2211 = x2210 + x2192;
bool x2212 = x2211 < 0;
bool x2213 = x2211 >= 16;
bool x2214 = x2212 || x2213;
if (x2214) {
int32_t x2215 = x2209 * 16;
float* x2216 = x2205+x2215;
memset(x2216, 0, 4 * 16);;
} else {
if (x2230) {
int32_t x2215 = x2209 * 16;
float* x2231 = x2205+x2215;
memset(x2231, 0, 4 * 16);;
} else {
int32_t x2215 = x2209 * 16;
if (x2234) {
float* x2235 = x2205+x2215;
memset(x2235, 0, 4 * x2222);;
} else {
}
// may have segfault here
int32_t x2240 = x2215 + x2222;
float* x2241 = x2205+x2240;
int32_t x2242 = x2211 * 16;
int32_t x2243 = x2242 + x2219;
int32_t x2244 = x2243 + x2222;
float* x2245 = x2208+x2244;
memcpy(x2241, x2245, 4 * x2229);;
if (x2247) {
int32_t x2248 = x2215 + 16;
int32_t x2249 = x2248 - x2227;
float* x2250 = x2205+x2249;
memset(x2250, 0, 4 * x2227);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,256,576,1,x82,576,x2188,256,1,x2186,256);

}
// resize to WrappedArray(-1, 1, 1)
float* x2266 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x2267 = 0;
int32_t x2268 = 0;
int32_t x2269 = 0;
for(int x2270=0; x2270 < 64; x2270++) {
int32_t x2271 = x2268;
int32_t x2272 = x2269;
int32_t x2273 = x2267;
int32_t x2274 = x2273;
int32_t x2275 = x2271;
int32_t x2276 = x2272;
for(int x2277=0; x2277 < 64; x2277++) {
int32_t x2278 = x2275;
int32_t x2279 = x2276;
int32_t x2280 = x2274;
int32_t x2281 = x2280;
int32_t x2282 = x2278;
int32_t x2283 = x2279;
for(int x2284=0; x2284 < 16; x2284++) {
int32_t x2285 = x2282;
int32_t x2286 = x2283;
int32_t x2287 = x2281;
int32_t x2288 = x2287;
int32_t x2289 = x2285;
int32_t x2290 = x2286;
for(int x2291=0; x2291 < 16; x2291++) {
int32_t x2292 = x2288;
int32_t x2293 = x2289;
float x2294 = x2181[x2293];
int32_t x2295 = x2290;
float x2296 = x43[x2295];
float x2297 = x2294 - x2296;
x2266[x2292] = x2297;
x2288 += 1;
x2289 += 1;

}
x2281 += 16;
x2282 += 16;

}
x2274 += 256;
x2275 += 256;
x2276 += 1;

}
x2267 += 16384;
x2268 += 16384;

}
float* x2316 = (float*)myMalloc(64 * sizeof(float));;
for(int x2317=0; x2317 < 64; x2317++) {
float x2318 = x243[x2317];
float x2319 = x2318 + 1.0E-5f;
x2316[x2317] = x2319;

}
float* x2323 = (float*)myMalloc(64 * sizeof(float));;
for(int x2324=0; x2324 < 64; x2324++) {
float x2325 = x2316[x2324];
double x2326 = (double)x2325;
double x2327 = sqrt(x2326);
float x2328 = (float)x2327;
x2323[x2324] = x2328;

}
// resize to WrappedArray(-1, 1, 1)
float* x2333 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x2334 = 0;
int32_t x2335 = 0;
int32_t x2336 = 0;
for(int x2337=0; x2337 < 64; x2337++) {
int32_t x2338 = x2335;
int32_t x2339 = x2336;
int32_t x2340 = x2334;
int32_t x2341 = x2340;
int32_t x2342 = x2338;
int32_t x2343 = x2339;
for(int x2344=0; x2344 < 64; x2344++) {
int32_t x2345 = x2342;
int32_t x2346 = x2343;
int32_t x2347 = x2341;
int32_t x2348 = x2347;
int32_t x2349 = x2345;
int32_t x2350 = x2346;
for(int x2351=0; x2351 < 16; x2351++) {
int32_t x2352 = x2349;
int32_t x2353 = x2350;
int32_t x2354 = x2348;
int32_t x2355 = x2354;
int32_t x2356 = x2352;
int32_t x2357 = x2353;
for(int x2358=0; x2358 < 16; x2358++) {
int32_t x2359 = x2355;
int32_t x2360 = x2356;
float x2361 = x2266[x2360];
int32_t x2362 = x2357;
float x2363 = x2323[x2362];
float x2364 = x2361 / x2363;
x2333[x2359] = x2364;
x2355 += 1;
x2356 += 1;

}
x2348 += 16;
x2349 += 16;

}
x2341 += 256;
x2342 += 256;
x2343 += 1;

}
x2334 += 16384;
x2335 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x2384 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x2385 = 0;
int32_t x2386 = 0;
int32_t x2387 = 0;
for(int x2388=0; x2388 < 64; x2388++) {
int32_t x2389 = x2386;
int32_t x2390 = x2387;
int32_t x2391 = x2385;
int32_t x2392 = x2391;
int32_t x2393 = x2389;
int32_t x2394 = x2390;
for(int x2395=0; x2395 < 64; x2395++) {
int32_t x2396 = x2393;
int32_t x2397 = x2394;
int32_t x2398 = x2392;
int32_t x2399 = x2398;
int32_t x2400 = x2396;
int32_t x2401 = x2397;
for(int x2402=0; x2402 < 16; x2402++) {
int32_t x2403 = x2400;
int32_t x2404 = x2401;
int32_t x2405 = x2399;
int32_t x2406 = x2405;
int32_t x2407 = x2403;
int32_t x2408 = x2404;
for(int x2409=0; x2409 < 16; x2409++) {
int32_t x2410 = x2406;
int32_t x2411 = x2407;
float x2412 = x2333[x2411];
int32_t x2413 = x2408;
float x2414 = x207[x2413];
float x2415 = x2412 * x2414;
x2384[x2410] = x2415;
x2406 += 1;
x2407 += 1;

}
x2399 += 16;
x2400 += 16;

}
x2392 += 256;
x2393 += 256;
x2394 += 1;

}
x2385 += 16384;
x2386 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x2435 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x2436 = 0;
int32_t x2437 = 0;
int32_t x2438 = 0;
for(int x2439=0; x2439 < 64; x2439++) {
int32_t x2440 = x2437;
int32_t x2441 = x2438;
int32_t x2442 = x2436;
int32_t x2443 = x2442;
int32_t x2444 = x2440;
int32_t x2445 = x2441;
for(int x2446=0; x2446 < 64; x2446++) {
int32_t x2447 = x2444;
int32_t x2448 = x2445;
int32_t x2449 = x2443;
int32_t x2450 = x2449;
int32_t x2451 = x2447;
int32_t x2452 = x2448;
for(int x2453=0; x2453 < 16; x2453++) {
int32_t x2454 = x2451;
int32_t x2455 = x2452;
int32_t x2456 = x2450;
int32_t x2457 = x2456;
int32_t x2458 = x2454;
int32_t x2459 = x2455;
for(int x2460=0; x2460 < 16; x2460++) {
int32_t x2461 = x2457;
int32_t x2462 = x2458;
float x2463 = x2384[x2462];
int32_t x2464 = x2459;
float x2465 = x152[x2464];
float x2466 = x2463 + x2465;
x2435[x2461] = x2466;
x2457 += 1;
x2458 += 1;

}
x2450 += 16;
x2451 += 16;

}
x2443 += 256;
x2444 += 256;
x2445 += 1;

}
x2436 += 16384;
x2437 += 16384;

}
float* x2485 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x2486=0; x2486 < 1048576; x2486++) {
float x2487 = x2435[x2486];
bool x2488 = x2487 < 0.0f;
if (x2488) {
x2485[x2486] = 0.0f;
} else {
float x2491 = x2435[x2486];
x2485[x2486] = x2491;
}

}
float* x2497 = (float*)myMalloc(4194304 * sizeof(float));;
float* x2498 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x2499=0; x2499 < 64; x2499++) {
int32_t x2500 = x2499 * 16384;
float* x2501 = x2485+x2500;
int32_t x2502 = x2499 * 65536;
float* x2503 = x2497+x2502;
float* x2504 = x2498+x2500;
for(int x2505=0; x2505 < 64; x2505++) {
int32_t x2506 = x2505 / 1;
int32_t x2510 = x2506 * 16;
int32_t x2511 = x2510 * 16;
int32_t x2507 = x2505 % 1;
int32_t x2508 = x2507 / 1;
int32_t x2512 = x2508 * 16;
int32_t x2513 = x2512 * 16;
int32_t x2514 = x2511 + x2513;
int32_t x2509 = x2507 % 1;
int32_t x2515 = x2509 * 16;
int32_t x2516 = x2515 * 16;
int32_t x2517 = x2514 + x2516;
float* x2518 = x2504+x2517;
float* x2519 = x2501+x2511;
for(int x2520=0; x2520 < 16; x2520++) {
int32_t x2522 = x2520 * 16;
float* x2523 = x2518+x2522;
int32_t x2521 = x2520 + x2508;
int32_t x2524 = x2521 * 16;
int32_t x2525 = x2524 + x2509;
float* x2526 = x2519+x2525;
memcpy(x2523, x2526, 4 * 16);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,256,64,1,x129,64,x2504,256,1,x2503,256);

}
// resize to WrappedArray(-1, 1, 1)
float* x2536 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x2537 = 0;
int32_t x2538 = 0;
int32_t x2539 = 0;
for(int x2540=0; x2540 < 64; x2540++) {
int32_t x2541 = x2538;
int32_t x2542 = x2539;
int32_t x2543 = x2537;
int32_t x2544 = x2543;
int32_t x2545 = x2541;
int32_t x2546 = x2542;
for(int x2547=0; x2547 < 256; x2547++) {
int32_t x2548 = x2545;
int32_t x2549 = x2546;
int32_t x2550 = x2544;
int32_t x2551 = x2550;
int32_t x2552 = x2548;
int32_t x2553 = x2549;
for(int x2554=0; x2554 < 16; x2554++) {
int32_t x2555 = x2552;
int32_t x2556 = x2553;
int32_t x2557 = x2551;
int32_t x2558 = x2557;
int32_t x2559 = x2555;
int32_t x2560 = x2556;
for(int x2561=0; x2561 < 16; x2561++) {
int32_t x2562 = x2558;
int32_t x2563 = x2559;
float x2564 = x2497[x2563];
int32_t x2565 = x2560;
float x2566 = x90[x2565];
float x2567 = x2564 - x2566;
x2536[x2562] = x2567;
x2558 += 1;
x2559 += 1;

}
x2551 += 16;
x2552 += 16;

}
x2544 += 256;
x2545 += 256;
x2546 += 1;

}
x2537 += 65536;
x2538 += 65536;

}
float* x2586 = (float*)myMalloc(256 * sizeof(float));;
for(int x2587=0; x2587 < 256; x2587++) {
float x2588 = x165[x2587];
float x2589 = x2588 + 1.0E-5f;
x2586[x2587] = x2589;

}
float* x2593 = (float*)myMalloc(256 * sizeof(float));;
for(int x2594=0; x2594 < 256; x2594++) {
float x2595 = x2586[x2594];
double x2596 = (double)x2595;
double x2597 = sqrt(x2596);
float x2598 = (float)x2597;
x2593[x2594] = x2598;

}
// resize to WrappedArray(-1, 1, 1)
float* x2603 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x2604 = 0;
int32_t x2605 = 0;
int32_t x2606 = 0;
for(int x2607=0; x2607 < 64; x2607++) {
int32_t x2608 = x2605;
int32_t x2609 = x2606;
int32_t x2610 = x2604;
int32_t x2611 = x2610;
int32_t x2612 = x2608;
int32_t x2613 = x2609;
for(int x2614=0; x2614 < 256; x2614++) {
int32_t x2615 = x2612;
int32_t x2616 = x2613;
int32_t x2617 = x2611;
int32_t x2618 = x2617;
int32_t x2619 = x2615;
int32_t x2620 = x2616;
for(int x2621=0; x2621 < 16; x2621++) {
int32_t x2622 = x2619;
int32_t x2623 = x2620;
int32_t x2624 = x2618;
int32_t x2625 = x2624;
int32_t x2626 = x2622;
int32_t x2627 = x2623;
for(int x2628=0; x2628 < 16; x2628++) {
int32_t x2629 = x2625;
int32_t x2630 = x2626;
float x2631 = x2536[x2630];
int32_t x2632 = x2627;
float x2633 = x2593[x2632];
float x2634 = x2631 / x2633;
x2603[x2629] = x2634;
x2625 += 1;
x2626 += 1;

}
x2618 += 16;
x2619 += 16;

}
x2611 += 256;
x2612 += 256;
x2613 += 1;

}
x2604 += 65536;
x2605 += 65536;

}
// resize to WrappedArray(-1, 1, 1)
float* x2654 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x2655 = 0;
int32_t x2656 = 0;
int32_t x2657 = 0;
for(int x2658=0; x2658 < 64; x2658++) {
int32_t x2659 = x2656;
int32_t x2660 = x2657;
int32_t x2661 = x2655;
int32_t x2662 = x2661;
int32_t x2663 = x2659;
int32_t x2664 = x2660;
for(int x2665=0; x2665 < 256; x2665++) {
int32_t x2666 = x2663;
int32_t x2667 = x2664;
int32_t x2668 = x2662;
int32_t x2669 = x2668;
int32_t x2670 = x2666;
int32_t x2671 = x2667;
for(int x2672=0; x2672 < 16; x2672++) {
int32_t x2673 = x2670;
int32_t x2674 = x2671;
int32_t x2675 = x2669;
int32_t x2676 = x2675;
int32_t x2677 = x2673;
int32_t x2678 = x2674;
for(int x2679=0; x2679 < 16; x2679++) {
int32_t x2680 = x2676;
int32_t x2681 = x2677;
float x2682 = x2603[x2681];
int32_t x2683 = x2678;
float x2684 = x57[x2683];
float x2685 = x2682 * x2684;
x2654[x2680] = x2685;
x2676 += 1;
x2677 += 1;

}
x2669 += 16;
x2670 += 16;

}
x2662 += 256;
x2663 += 256;
x2664 += 1;

}
x2655 += 65536;
x2656 += 65536;

}
// resize to WrappedArray(-1, 1, 1)
float* x2705 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x2706 = 0;
int32_t x2707 = 0;
int32_t x2708 = 0;
for(int x2709=0; x2709 < 64; x2709++) {
int32_t x2710 = x2707;
int32_t x2711 = x2708;
int32_t x2712 = x2706;
int32_t x2713 = x2712;
int32_t x2714 = x2710;
int32_t x2715 = x2711;
for(int x2716=0; x2716 < 256; x2716++) {
int32_t x2717 = x2714;
int32_t x2718 = x2715;
int32_t x2719 = x2713;
int32_t x2720 = x2719;
int32_t x2721 = x2717;
int32_t x2722 = x2718;
for(int x2723=0; x2723 < 16; x2723++) {
int32_t x2724 = x2721;
int32_t x2725 = x2722;
int32_t x2726 = x2720;
int32_t x2727 = x2726;
int32_t x2728 = x2724;
int32_t x2729 = x2725;
for(int x2730=0; x2730 < 16; x2730++) {
int32_t x2731 = x2727;
int32_t x2732 = x2728;
float x2733 = x2654[x2732];
int32_t x2734 = x2729;
float x2735 = x6[x2734];
float x2736 = x2733 + x2735;
x2705[x2731] = x2736;
x2727 += 1;
x2728 += 1;

}
x2720 += 16;
x2721 += 16;

}
x2713 += 256;
x2714 += 256;
x2715 += 1;

}
x2706 += 65536;
x2707 += 65536;

}
int32_t x2755 = 0;
int32_t x2756 = 0;
int32_t x2757 = 0;
for(int x2758=0; x2758 < 64; x2758++) {
int32_t x2759 = x2756;
int32_t x2760 = x2757;
int32_t x2761 = x2755;
int32_t x2762 = x2761;
int32_t x2763 = x2759;
int32_t x2764 = x2760;
for(int x2765=0; x2765 < 256; x2765++) {
int32_t x2766 = x2763;
int32_t x2767 = x2764;
int32_t x2768 = x2762;
int32_t x2769 = x2768;
int32_t x2770 = x2766;
int32_t x2771 = x2767;
for(int x2772=0; x2772 < 16; x2772++) {
int32_t x2773 = x2770;
int32_t x2774 = x2771;
int32_t x2775 = x2769;
int32_t x2776 = x2775;
int32_t x2777 = x2773;
int32_t x2778 = x2774;
for(int x2779=0; x2779 < 16; x2779++) {
int32_t x2780 = x2777;
float x2781 = x2705[x2780];
int32_t x2782 = x2778;
float x2783 = x1899[x2782];
float x2784 = x2781 + x2783;
x2705[x2780] = x2784;
x2776 += 1;
x2777 += 1;
x2778 += 1;

}
x2769 += 16;
x2770 += 16;
x2771 += 16;

}
x2762 += 256;
x2763 += 256;
x2764 += 256;

}
x2755 += 65536;
x2756 += 65536;
x2757 += 65536;

}
float* x2806 = (float*)myMalloc(4194304 * sizeof(float));;
for(int x2807=0; x2807 < 4194304; x2807++) {
float x2808 = x2705[x2807];
bool x2809 = x2808 < 0.0f;
if (x2809) {
x2806[x2807] = 0.0f;
} else {
float x2812 = x2705[x2807];
x2806[x2807] = x2812;
}

}
float* x2818 = (float*)myMalloc(1048576 * sizeof(float));;
float* x2819 = (float*)myMalloc(4194304 * sizeof(float));;
for(int x2820=0; x2820 < 64; x2820++) {
int32_t x2821 = x2820 * 65536;
float* x2822 = x2806+x2821;
int32_t x2823 = x2820 * 16384;
float* x2824 = x2818+x2823;
float* x2825 = x2819+x2821;
for(int x2826=0; x2826 < 256; x2826++) {
int32_t x2827 = x2826 / 1;
int32_t x2831 = x2827 * 16;
int32_t x2832 = x2831 * 16;
int32_t x2828 = x2826 % 1;
int32_t x2829 = x2828 / 1;
int32_t x2833 = x2829 * 16;
int32_t x2834 = x2833 * 16;
int32_t x2835 = x2832 + x2834;
int32_t x2830 = x2828 % 1;
int32_t x2836 = x2830 * 16;
int32_t x2837 = x2836 * 16;
int32_t x2838 = x2835 + x2837;
float* x2839 = x2825+x2838;
float* x2840 = x2822+x2832;
for(int x2841=0; x2841 < 16; x2841++) {
int32_t x2843 = x2841 * 16;
float* x2844 = x2839+x2843;
int32_t x2842 = x2841 + x2829;
int32_t x2845 = x2842 * 16;
int32_t x2846 = x2845 + x2830;
float* x2847 = x2840+x2846;
memcpy(x2844, x2847, 4 * 16);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,256,256,1,x149,256,x2825,256,1,x2824,256);

}
// resize to WrappedArray(-1, 1, 1)
float* x2857 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x2858 = 0;
int32_t x2859 = 0;
int32_t x2860 = 0;
for(int x2861=0; x2861 < 64; x2861++) {
int32_t x2862 = x2859;
int32_t x2863 = x2860;
int32_t x2864 = x2858;
int32_t x2865 = x2864;
int32_t x2866 = x2862;
int32_t x2867 = x2863;
for(int x2868=0; x2868 < 64; x2868++) {
int32_t x2869 = x2866;
int32_t x2870 = x2867;
int32_t x2871 = x2865;
int32_t x2872 = x2871;
int32_t x2873 = x2869;
int32_t x2874 = x2870;
for(int x2875=0; x2875 < 16; x2875++) {
int32_t x2876 = x2873;
int32_t x2877 = x2874;
int32_t x2878 = x2872;
int32_t x2879 = x2878;
int32_t x2880 = x2876;
int32_t x2881 = x2877;
for(int x2882=0; x2882 < 16; x2882++) {
int32_t x2883 = x2879;
int32_t x2884 = x2880;
float x2885 = x2818[x2884];
int32_t x2886 = x2881;
float x2887 = x256[x2886];
float x2888 = x2885 - x2887;
x2857[x2883] = x2888;
x2879 += 1;
x2880 += 1;

}
x2872 += 16;
x2873 += 16;

}
x2865 += 256;
x2866 += 256;
x2867 += 1;

}
x2858 += 16384;
x2859 += 16384;

}
float* x2907 = (float*)myMalloc(64 * sizeof(float));;
for(int x2908=0; x2908 < 64; x2908++) {
float x2909 = x186[x2908];
float x2910 = x2909 + 1.0E-5f;
x2907[x2908] = x2910;

}
float* x2914 = (float*)myMalloc(64 * sizeof(float));;
for(int x2915=0; x2915 < 64; x2915++) {
float x2916 = x2907[x2915];
double x2917 = (double)x2916;
double x2918 = sqrt(x2917);
float x2919 = (float)x2918;
x2914[x2915] = x2919;

}
// resize to WrappedArray(-1, 1, 1)
float* x2924 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x2925 = 0;
int32_t x2926 = 0;
int32_t x2927 = 0;
for(int x2928=0; x2928 < 64; x2928++) {
int32_t x2929 = x2926;
int32_t x2930 = x2927;
int32_t x2931 = x2925;
int32_t x2932 = x2931;
int32_t x2933 = x2929;
int32_t x2934 = x2930;
for(int x2935=0; x2935 < 64; x2935++) {
int32_t x2936 = x2933;
int32_t x2937 = x2934;
int32_t x2938 = x2932;
int32_t x2939 = x2938;
int32_t x2940 = x2936;
int32_t x2941 = x2937;
for(int x2942=0; x2942 < 16; x2942++) {
int32_t x2943 = x2940;
int32_t x2944 = x2941;
int32_t x2945 = x2939;
int32_t x2946 = x2945;
int32_t x2947 = x2943;
int32_t x2948 = x2944;
for(int x2949=0; x2949 < 16; x2949++) {
int32_t x2950 = x2946;
int32_t x2951 = x2947;
float x2952 = x2857[x2951];
int32_t x2953 = x2948;
float x2954 = x2914[x2953];
float x2955 = x2952 / x2954;
x2924[x2950] = x2955;
x2946 += 1;
x2947 += 1;

}
x2939 += 16;
x2940 += 16;

}
x2932 += 256;
x2933 += 256;
x2934 += 1;

}
x2925 += 16384;
x2926 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x2975 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x2976 = 0;
int32_t x2977 = 0;
int32_t x2978 = 0;
for(int x2979=0; x2979 < 64; x2979++) {
int32_t x2980 = x2977;
int32_t x2981 = x2978;
int32_t x2982 = x2976;
int32_t x2983 = x2982;
int32_t x2984 = x2980;
int32_t x2985 = x2981;
for(int x2986=0; x2986 < 64; x2986++) {
int32_t x2987 = x2984;
int32_t x2988 = x2985;
int32_t x2989 = x2983;
int32_t x2990 = x2989;
int32_t x2991 = x2987;
int32_t x2992 = x2988;
for(int x2993=0; x2993 < 16; x2993++) {
int32_t x2994 = x2991;
int32_t x2995 = x2992;
int32_t x2996 = x2990;
int32_t x2997 = x2996;
int32_t x2998 = x2994;
int32_t x2999 = x2995;
for(int x3000=0; x3000 < 16; x3000++) {
int32_t x3001 = x2997;
int32_t x3002 = x2998;
float x3003 = x2924[x3002];
int32_t x3004 = x2999;
float x3005 = x80[x3004];
float x3006 = x3003 * x3005;
x2975[x3001] = x3006;
x2997 += 1;
x2998 += 1;

}
x2990 += 16;
x2991 += 16;

}
x2983 += 256;
x2984 += 256;
x2985 += 1;

}
x2976 += 16384;
x2977 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x3026 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x3027 = 0;
int32_t x3028 = 0;
int32_t x3029 = 0;
for(int x3030=0; x3030 < 64; x3030++) {
int32_t x3031 = x3028;
int32_t x3032 = x3029;
int32_t x3033 = x3027;
int32_t x3034 = x3033;
int32_t x3035 = x3031;
int32_t x3036 = x3032;
for(int x3037=0; x3037 < 64; x3037++) {
int32_t x3038 = x3035;
int32_t x3039 = x3036;
int32_t x3040 = x3034;
int32_t x3041 = x3040;
int32_t x3042 = x3038;
int32_t x3043 = x3039;
for(int x3044=0; x3044 < 16; x3044++) {
int32_t x3045 = x3042;
int32_t x3046 = x3043;
int32_t x3047 = x3041;
int32_t x3048 = x3047;
int32_t x3049 = x3045;
int32_t x3050 = x3046;
for(int x3051=0; x3051 < 16; x3051++) {
int32_t x3052 = x3048;
int32_t x3053 = x3049;
float x3054 = x2975[x3053];
int32_t x3055 = x3050;
float x3056 = x23[x3055];
float x3057 = x3054 + x3056;
x3026[x3052] = x3057;
x3048 += 1;
x3049 += 1;

}
x3041 += 16;
x3042 += 16;

}
x3034 += 256;
x3035 += 256;
x3036 += 1;

}
x3027 += 16384;
x3028 += 16384;

}
float* x3076 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x3077=0; x3077 < 1048576; x3077++) {
float x3078 = x3026[x3077];
bool x3079 = x3078 < 0.0f;
if (x3079) {
x3076[x3077] = 0.0f;
} else {
float x3082 = x3026[x3077];
x3076[x3077] = x3082;
}

}
float* x3088 = (float*)myMalloc(1048576 * sizeof(float));;
float* x3089 = (float*)myMalloc(9437184 * sizeof(float));;
for(int x3090=0; x3090 < 64; x3090++) {
int32_t x3091 = x3090 * 16384;
float* x3092 = x3076+x3091;
float* x3093 = x3088+x3091;
int32_t x3094 = x3090 * 147456;
float* x3095 = x3089+x3094;
for(int x3096=0; x3096 < 576; x3096++) {
int32_t x3097 = x3096 / 9;
int32_t x3101 = x3097 * 3;
int32_t x3102 = x3101 * 3;
int32_t x3103 = x3102 * 16;
int32_t x3104 = x3103 * 16;
int32_t x3098 = x3096 % 9;
int32_t x3099 = x3098 / 3;
int32_t x3105 = x3099 * 3;
int32_t x3106 = x3105 * 16;
int32_t x3107 = x3106 * 16;
int32_t x3108 = x3104 + x3107;
int32_t x3100 = x3098 % 3;
int32_t x3109 = x3100 * 16;
int32_t x3110 = x3109 * 16;
int32_t x3111 = x3108 + x3110;
float* x3112 = x3095+x3111;
int32_t x3113 = x3097 * 16;
int32_t x3114 = x3113 * 16;
float* x3115 = x3092+x3114;
int32_t x3127 = 1 - x3100;
bool x3128 = x3127 > 0;
int32_t x3129;
if (x3128) {
x3129 = x3127;
} else {
x3129 = 0;
}
int32_t x3130 = 3 - x3100;
int32_t x3131 = x3130 - 1;
int32_t x3132 = 1 - x3131;
bool x3133 = x3132 > 0;
int32_t x3134;
if (x3133) {
x3134 = x3132;
} else {
x3134 = 0;
}
int32_t x3135 = 16 - x3134;
int32_t x3136 = x3135 - x3129;
bool x3137 = x3136 <= 0;
bool x3141 = x3129 > 0;
int32_t x3126 = -1 + x3100;
bool x3154 = x3134 > 0;
for(int x3116=0; x3116 < 16; x3116++) {
int32_t x3117 = x3116 - 1;
int32_t x3118 = x3117 + x3099;
bool x3119 = x3118 < 0;
bool x3120 = x3118 >= 16;
bool x3121 = x3119 || x3120;
if (x3121) {
int32_t x3122 = x3116 * 16;
float* x3123 = x3112+x3122;
memset(x3123, 0, 4 * 16);;
} else {
if (x3137) {
int32_t x3122 = x3116 * 16;
float* x3138 = x3112+x3122;
memset(x3138, 0, 4 * 16);;
} else {
int32_t x3122 = x3116 * 16;
if (x3141) {
float* x3142 = x3112+x3122;
memset(x3142, 0, 4 * x3129);;
} else {
}
// may have segfault here
int32_t x3147 = x3122 + x3129;
float* x3148 = x3112+x3147;
int32_t x3149 = x3118 * 16;
int32_t x3150 = x3149 + x3126;
int32_t x3151 = x3150 + x3129;
float* x3152 = x3115+x3151;
memcpy(x3148, x3152, 4 * x3136);;
if (x3154) {
int32_t x3155 = x3122 + 16;
int32_t x3156 = x3155 - x3134;
float* x3157 = x3112+x3156;
memset(x3157, 0, 4 * x3134);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,256,576,1,x72,576,x3095,256,1,x3093,256);

}
// resize to WrappedArray(-1, 1, 1)
float* x3173 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x3174 = 0;
int32_t x3175 = 0;
int32_t x3176 = 0;
for(int x3177=0; x3177 < 64; x3177++) {
int32_t x3178 = x3175;
int32_t x3179 = x3176;
int32_t x3180 = x3174;
int32_t x3181 = x3180;
int32_t x3182 = x3178;
int32_t x3183 = x3179;
for(int x3184=0; x3184 < 64; x3184++) {
int32_t x3185 = x3182;
int32_t x3186 = x3183;
int32_t x3187 = x3181;
int32_t x3188 = x3187;
int32_t x3189 = x3185;
int32_t x3190 = x3186;
for(int x3191=0; x3191 < 16; x3191++) {
int32_t x3192 = x3189;
int32_t x3193 = x3190;
int32_t x3194 = x3188;
int32_t x3195 = x3194;
int32_t x3196 = x3192;
int32_t x3197 = x3193;
for(int x3198=0; x3198 < 16; x3198++) {
int32_t x3199 = x3195;
int32_t x3200 = x3196;
float x3201 = x3088[x3200];
int32_t x3202 = x3197;
float x3203 = x178[x3202];
float x3204 = x3201 - x3203;
x3173[x3199] = x3204;
x3195 += 1;
x3196 += 1;

}
x3188 += 16;
x3189 += 16;

}
x3181 += 256;
x3182 += 256;
x3183 += 1;

}
x3174 += 16384;
x3175 += 16384;

}
float* x3223 = (float*)myMalloc(64 * sizeof(float));;
for(int x3224=0; x3224 < 64; x3224++) {
float x3225 = x117[x3224];
float x3226 = x3225 + 1.0E-5f;
x3223[x3224] = x3226;

}
float* x3230 = (float*)myMalloc(64 * sizeof(float));;
for(int x3231=0; x3231 < 64; x3231++) {
float x3232 = x3223[x3231];
double x3233 = (double)x3232;
double x3234 = sqrt(x3233);
float x3235 = (float)x3234;
x3230[x3231] = x3235;

}
// resize to WrappedArray(-1, 1, 1)
float* x3240 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x3241 = 0;
int32_t x3242 = 0;
int32_t x3243 = 0;
for(int x3244=0; x3244 < 64; x3244++) {
int32_t x3245 = x3242;
int32_t x3246 = x3243;
int32_t x3247 = x3241;
int32_t x3248 = x3247;
int32_t x3249 = x3245;
int32_t x3250 = x3246;
for(int x3251=0; x3251 < 64; x3251++) {
int32_t x3252 = x3249;
int32_t x3253 = x3250;
int32_t x3254 = x3248;
int32_t x3255 = x3254;
int32_t x3256 = x3252;
int32_t x3257 = x3253;
for(int x3258=0; x3258 < 16; x3258++) {
int32_t x3259 = x3256;
int32_t x3260 = x3257;
int32_t x3261 = x3255;
int32_t x3262 = x3261;
int32_t x3263 = x3259;
int32_t x3264 = x3260;
for(int x3265=0; x3265 < 16; x3265++) {
int32_t x3266 = x3262;
int32_t x3267 = x3263;
float x3268 = x3173[x3267];
int32_t x3269 = x3264;
float x3270 = x3230[x3269];
float x3271 = x3268 / x3270;
x3240[x3266] = x3271;
x3262 += 1;
x3263 += 1;

}
x3255 += 16;
x3256 += 16;

}
x3248 += 256;
x3249 += 256;
x3250 += 1;

}
x3241 += 16384;
x3242 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x3291 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x3292 = 0;
int32_t x3293 = 0;
int32_t x3294 = 0;
for(int x3295=0; x3295 < 64; x3295++) {
int32_t x3296 = x3293;
int32_t x3297 = x3294;
int32_t x3298 = x3292;
int32_t x3299 = x3298;
int32_t x3300 = x3296;
int32_t x3301 = x3297;
for(int x3302=0; x3302 < 64; x3302++) {
int32_t x3303 = x3300;
int32_t x3304 = x3301;
int32_t x3305 = x3299;
int32_t x3306 = x3305;
int32_t x3307 = x3303;
int32_t x3308 = x3304;
for(int x3309=0; x3309 < 16; x3309++) {
int32_t x3310 = x3307;
int32_t x3311 = x3308;
int32_t x3312 = x3306;
int32_t x3313 = x3312;
int32_t x3314 = x3310;
int32_t x3315 = x3311;
for(int x3316=0; x3316 < 16; x3316++) {
int32_t x3317 = x3313;
int32_t x3318 = x3314;
float x3319 = x3240[x3318];
int32_t x3320 = x3315;
float x3321 = x71[x3320];
float x3322 = x3319 * x3321;
x3291[x3317] = x3322;
x3313 += 1;
x3314 += 1;

}
x3306 += 16;
x3307 += 16;

}
x3299 += 256;
x3300 += 256;
x3301 += 1;

}
x3292 += 16384;
x3293 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x3342 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x3343 = 0;
int32_t x3344 = 0;
int32_t x3345 = 0;
for(int x3346=0; x3346 < 64; x3346++) {
int32_t x3347 = x3344;
int32_t x3348 = x3345;
int32_t x3349 = x3343;
int32_t x3350 = x3349;
int32_t x3351 = x3347;
int32_t x3352 = x3348;
for(int x3353=0; x3353 < 64; x3353++) {
int32_t x3354 = x3351;
int32_t x3355 = x3352;
int32_t x3356 = x3350;
int32_t x3357 = x3356;
int32_t x3358 = x3354;
int32_t x3359 = x3355;
for(int x3360=0; x3360 < 16; x3360++) {
int32_t x3361 = x3358;
int32_t x3362 = x3359;
int32_t x3363 = x3357;
int32_t x3364 = x3363;
int32_t x3365 = x3361;
int32_t x3366 = x3362;
for(int x3367=0; x3367 < 16; x3367++) {
int32_t x3368 = x3364;
int32_t x3369 = x3365;
float x3370 = x3291[x3369];
int32_t x3371 = x3366;
float x3372 = x134[x3371];
float x3373 = x3370 + x3372;
x3342[x3368] = x3373;
x3364 += 1;
x3365 += 1;

}
x3357 += 16;
x3358 += 16;

}
x3350 += 256;
x3351 += 256;
x3352 += 1;

}
x3343 += 16384;
x3344 += 16384;

}
float* x3392 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x3393=0; x3393 < 1048576; x3393++) {
float x3394 = x3342[x3393];
bool x3395 = x3394 < 0.0f;
if (x3395) {
x3392[x3393] = 0.0f;
} else {
float x3398 = x3342[x3393];
x3392[x3393] = x3398;
}

}
float* x3404 = (float*)myMalloc(4194304 * sizeof(float));;
float* x3405 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x3406=0; x3406 < 64; x3406++) {
int32_t x3407 = x3406 * 16384;
float* x3408 = x3392+x3407;
int32_t x3409 = x3406 * 65536;
float* x3410 = x3404+x3409;
float* x3411 = x3405+x3407;
for(int x3412=0; x3412 < 64; x3412++) {
int32_t x3413 = x3412 / 1;
int32_t x3417 = x3413 * 16;
int32_t x3418 = x3417 * 16;
int32_t x3414 = x3412 % 1;
int32_t x3415 = x3414 / 1;
int32_t x3419 = x3415 * 16;
int32_t x3420 = x3419 * 16;
int32_t x3421 = x3418 + x3420;
int32_t x3416 = x3414 % 1;
int32_t x3422 = x3416 * 16;
int32_t x3423 = x3422 * 16;
int32_t x3424 = x3421 + x3423;
float* x3425 = x3411+x3424;
float* x3426 = x3408+x3418;
for(int x3427=0; x3427 < 16; x3427++) {
int32_t x3429 = x3427 * 16;
float* x3430 = x3425+x3429;
int32_t x3428 = x3427 + x3415;
int32_t x3431 = x3428 * 16;
int32_t x3432 = x3431 + x3416;
float* x3433 = x3426+x3432;
memcpy(x3430, x3433, 4 * 16);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,256,64,1,x86,64,x3411,256,1,x3410,256);

}
// resize to WrappedArray(-1, 1, 1)
float* x3443 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x3444 = 0;
int32_t x3445 = 0;
int32_t x3446 = 0;
for(int x3447=0; x3447 < 64; x3447++) {
int32_t x3448 = x3445;
int32_t x3449 = x3446;
int32_t x3450 = x3444;
int32_t x3451 = x3450;
int32_t x3452 = x3448;
int32_t x3453 = x3449;
for(int x3454=0; x3454 < 256; x3454++) {
int32_t x3455 = x3452;
int32_t x3456 = x3453;
int32_t x3457 = x3451;
int32_t x3458 = x3457;
int32_t x3459 = x3455;
int32_t x3460 = x3456;
for(int x3461=0; x3461 < 16; x3461++) {
int32_t x3462 = x3459;
int32_t x3463 = x3460;
int32_t x3464 = x3458;
int32_t x3465 = x3464;
int32_t x3466 = x3462;
int32_t x3467 = x3463;
for(int x3468=0; x3468 < 16; x3468++) {
int32_t x3469 = x3465;
int32_t x3470 = x3466;
float x3471 = x3404[x3470];
int32_t x3472 = x3467;
float x3473 = x183[x3472];
float x3474 = x3471 - x3473;
x3443[x3469] = x3474;
x3465 += 1;
x3466 += 1;

}
x3458 += 16;
x3459 += 16;

}
x3451 += 256;
x3452 += 256;
x3453 += 1;

}
x3444 += 65536;
x3445 += 65536;

}
float* x3493 = (float*)myMalloc(256 * sizeof(float));;
for(int x3494=0; x3494 < 256; x3494++) {
float x3495 = x132[x3494];
float x3496 = x3495 + 1.0E-5f;
x3493[x3494] = x3496;

}
float* x3500 = (float*)myMalloc(256 * sizeof(float));;
for(int x3501=0; x3501 < 256; x3501++) {
float x3502 = x3493[x3501];
double x3503 = (double)x3502;
double x3504 = sqrt(x3503);
float x3505 = (float)x3504;
x3500[x3501] = x3505;

}
// resize to WrappedArray(-1, 1, 1)
float* x3510 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x3511 = 0;
int32_t x3512 = 0;
int32_t x3513 = 0;
for(int x3514=0; x3514 < 64; x3514++) {
int32_t x3515 = x3512;
int32_t x3516 = x3513;
int32_t x3517 = x3511;
int32_t x3518 = x3517;
int32_t x3519 = x3515;
int32_t x3520 = x3516;
for(int x3521=0; x3521 < 256; x3521++) {
int32_t x3522 = x3519;
int32_t x3523 = x3520;
int32_t x3524 = x3518;
int32_t x3525 = x3524;
int32_t x3526 = x3522;
int32_t x3527 = x3523;
for(int x3528=0; x3528 < 16; x3528++) {
int32_t x3529 = x3526;
int32_t x3530 = x3527;
int32_t x3531 = x3525;
int32_t x3532 = x3531;
int32_t x3533 = x3529;
int32_t x3534 = x3530;
for(int x3535=0; x3535 < 16; x3535++) {
int32_t x3536 = x3532;
int32_t x3537 = x3533;
float x3538 = x3443[x3537];
int32_t x3539 = x3534;
float x3540 = x3500[x3539];
float x3541 = x3538 / x3540;
x3510[x3536] = x3541;
x3532 += 1;
x3533 += 1;

}
x3525 += 16;
x3526 += 16;

}
x3518 += 256;
x3519 += 256;
x3520 += 1;

}
x3511 += 65536;
x3512 += 65536;

}
// resize to WrappedArray(-1, 1, 1)
float* x3561 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x3562 = 0;
int32_t x3563 = 0;
int32_t x3564 = 0;
for(int x3565=0; x3565 < 64; x3565++) {
int32_t x3566 = x3563;
int32_t x3567 = x3564;
int32_t x3568 = x3562;
int32_t x3569 = x3568;
int32_t x3570 = x3566;
int32_t x3571 = x3567;
for(int x3572=0; x3572 < 256; x3572++) {
int32_t x3573 = x3570;
int32_t x3574 = x3571;
int32_t x3575 = x3569;
int32_t x3576 = x3575;
int32_t x3577 = x3573;
int32_t x3578 = x3574;
for(int x3579=0; x3579 < 16; x3579++) {
int32_t x3580 = x3577;
int32_t x3581 = x3578;
int32_t x3582 = x3576;
int32_t x3583 = x3582;
int32_t x3584 = x3580;
int32_t x3585 = x3581;
for(int x3586=0; x3586 < 16; x3586++) {
int32_t x3587 = x3583;
int32_t x3588 = x3584;
float x3589 = x3510[x3588];
int32_t x3590 = x3585;
float x3591 = x36[x3590];
float x3592 = x3589 * x3591;
x3561[x3587] = x3592;
x3583 += 1;
x3584 += 1;

}
x3576 += 16;
x3577 += 16;

}
x3569 += 256;
x3570 += 256;
x3571 += 1;

}
x3562 += 65536;
x3563 += 65536;

}
// resize to WrappedArray(-1, 1, 1)
float* x3612 = (float*)myMalloc(4194304 * sizeof(float));;
int32_t x3613 = 0;
int32_t x3614 = 0;
int32_t x3615 = 0;
for(int x3616=0; x3616 < 64; x3616++) {
int32_t x3617 = x3614;
int32_t x3618 = x3615;
int32_t x3619 = x3613;
int32_t x3620 = x3619;
int32_t x3621 = x3617;
int32_t x3622 = x3618;
for(int x3623=0; x3623 < 256; x3623++) {
int32_t x3624 = x3621;
int32_t x3625 = x3622;
int32_t x3626 = x3620;
int32_t x3627 = x3626;
int32_t x3628 = x3624;
int32_t x3629 = x3625;
for(int x3630=0; x3630 < 16; x3630++) {
int32_t x3631 = x3628;
int32_t x3632 = x3629;
int32_t x3633 = x3627;
int32_t x3634 = x3633;
int32_t x3635 = x3631;
int32_t x3636 = x3632;
for(int x3637=0; x3637 < 16; x3637++) {
int32_t x3638 = x3634;
int32_t x3639 = x3635;
float x3640 = x3561[x3639];
int32_t x3641 = x3636;
float x3642 = x246[x3641];
float x3643 = x3640 + x3642;
x3612[x3638] = x3643;
x3634 += 1;
x3635 += 1;

}
x3627 += 16;
x3628 += 16;

}
x3620 += 256;
x3621 += 256;
x3622 += 1;

}
x3613 += 65536;
x3614 += 65536;

}
int32_t x3662 = 0;
int32_t x3663 = 0;
int32_t x3664 = 0;
for(int x3665=0; x3665 < 64; x3665++) {
int32_t x3666 = x3663;
int32_t x3667 = x3664;
int32_t x3668 = x3662;
int32_t x3669 = x3668;
int32_t x3670 = x3666;
int32_t x3671 = x3667;
for(int x3672=0; x3672 < 256; x3672++) {
int32_t x3673 = x3670;
int32_t x3674 = x3671;
int32_t x3675 = x3669;
int32_t x3676 = x3675;
int32_t x3677 = x3673;
int32_t x3678 = x3674;
for(int x3679=0; x3679 < 16; x3679++) {
int32_t x3680 = x3677;
int32_t x3681 = x3678;
int32_t x3682 = x3676;
int32_t x3683 = x3682;
int32_t x3684 = x3680;
int32_t x3685 = x3681;
for(int x3686=0; x3686 < 16; x3686++) {
int32_t x3687 = x3684;
float x3688 = x3612[x3687];
int32_t x3689 = x3685;
float x3690 = x2806[x3689];
float x3691 = x3688 + x3690;
x3612[x3687] = x3691;
x3683 += 1;
x3684 += 1;
x3685 += 1;

}
x3676 += 16;
x3677 += 16;
x3678 += 16;

}
x3669 += 256;
x3670 += 256;
x3671 += 256;

}
x3662 += 65536;
x3663 += 65536;
x3664 += 65536;

}
float* x3713 = (float*)myMalloc(4194304 * sizeof(float));;
for(int x3714=0; x3714 < 4194304; x3714++) {
float x3715 = x3612[x3714];
bool x3716 = x3715 < 0.0f;
if (x3716) {
x3713[x3714] = 0.0f;
} else {
float x3719 = x3612[x3714];
x3713[x3714] = x3719;
}

}
float* x3725 = (float*)myMalloc(2097152 * sizeof(float));;
float* x3726 = (float*)myMalloc(4194304 * sizeof(float));;
for(int x3727=0; x3727 < 64; x3727++) {
int32_t x3728 = x3727 * 65536;
float* x3729 = x3713+x3728;
int32_t x3730 = x3727 * 32768;
float* x3731 = x3725+x3730;
float* x3732 = x3726+x3728;
for(int x3733=0; x3733 < 256; x3733++) {
int32_t x3734 = x3733 / 1;
int32_t x3738 = x3734 * 16;
int32_t x3739 = x3738 * 16;
int32_t x3735 = x3733 % 1;
int32_t x3736 = x3735 / 1;
int32_t x3740 = x3736 * 16;
int32_t x3741 = x3740 * 16;
int32_t x3742 = x3739 + x3741;
int32_t x3737 = x3735 % 1;
int32_t x3743 = x3737 * 16;
int32_t x3744 = x3743 * 16;
int32_t x3745 = x3742 + x3744;
float* x3746 = x3732+x3745;
float* x3747 = x3729+x3739;
for(int x3748=0; x3748 < 16; x3748++) {
int32_t x3750 = x3748 * 16;
float* x3751 = x3746+x3750;
int32_t x3749 = x3748 + x3736;
int32_t x3752 = x3749 * 16;
int32_t x3753 = x3752 + x3737;
float* x3754 = x3747+x3753;
memcpy(x3751, x3754, 4 * 16);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,256,256,1,x10,256,x3732,256,1,x3731,256);

}
// resize to WrappedArray(-1, 1, 1)
float* x3764 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x3765 = 0;
int32_t x3766 = 0;
int32_t x3767 = 0;
for(int x3768=0; x3768 < 64; x3768++) {
int32_t x3769 = x3766;
int32_t x3770 = x3767;
int32_t x3771 = x3765;
int32_t x3772 = x3771;
int32_t x3773 = x3769;
int32_t x3774 = x3770;
for(int x3776=0; x3776 < 128; x3776++) {
int32_t x3777 = x3773;
int32_t x3778 = x3774;
int32_t x3779 = x3772;
int32_t x3780 = x3779;
int32_t x3781 = x3777;
int32_t x3782 = x3778;
for(int x3783=0; x3783 < 16; x3783++) {
int32_t x3784 = x3781;
int32_t x3785 = x3782;
int32_t x3786 = x3780;
int32_t x3787 = x3786;
int32_t x3788 = x3784;
int32_t x3789 = x3785;
for(int x3790=0; x3790 < 16; x3790++) {
int32_t x3791 = x3787;
int32_t x3792 = x3788;
float x3793 = x3725[x3792];
int32_t x3794 = x3789;
float x3795 = x203[x3794];
float x3796 = x3793 - x3795;
x3764[x3791] = x3796;
x3787 += 1;
x3788 += 1;

}
x3780 += 16;
x3781 += 16;

}
x3772 += 256;
x3773 += 256;
x3774 += 1;

}
x3765 += 32768;
x3766 += 32768;

}
float* x3815 = (float*)myMalloc(128 * sizeof(float));;
for(int x3816=0; x3816 < 128; x3816++) {
float x3817 = x133[x3816];
float x3818 = x3817 + 1.0E-5f;
x3815[x3816] = x3818;

}
float* x3822 = (float*)myMalloc(128 * sizeof(float));;
for(int x3823=0; x3823 < 128; x3823++) {
float x3824 = x3815[x3823];
double x3825 = (double)x3824;
double x3826 = sqrt(x3825);
float x3827 = (float)x3826;
x3822[x3823] = x3827;

}
// resize to WrappedArray(-1, 1, 1)
float* x3832 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x3833 = 0;
int32_t x3834 = 0;
int32_t x3835 = 0;
for(int x3836=0; x3836 < 64; x3836++) {
int32_t x3837 = x3834;
int32_t x3838 = x3835;
int32_t x3839 = x3833;
int32_t x3840 = x3839;
int32_t x3841 = x3837;
int32_t x3842 = x3838;
for(int x3843=0; x3843 < 128; x3843++) {
int32_t x3844 = x3841;
int32_t x3845 = x3842;
int32_t x3846 = x3840;
int32_t x3847 = x3846;
int32_t x3848 = x3844;
int32_t x3849 = x3845;
for(int x3850=0; x3850 < 16; x3850++) {
int32_t x3851 = x3848;
int32_t x3852 = x3849;
int32_t x3853 = x3847;
int32_t x3854 = x3853;
int32_t x3855 = x3851;
int32_t x3856 = x3852;
for(int x3857=0; x3857 < 16; x3857++) {
int32_t x3858 = x3854;
int32_t x3859 = x3855;
float x3860 = x3764[x3859];
int32_t x3861 = x3856;
float x3862 = x3822[x3861];
float x3863 = x3860 / x3862;
x3832[x3858] = x3863;
x3854 += 1;
x3855 += 1;

}
x3847 += 16;
x3848 += 16;

}
x3840 += 256;
x3841 += 256;
x3842 += 1;

}
x3833 += 32768;
x3834 += 32768;

}
// resize to WrappedArray(-1, 1, 1)
float* x3883 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x3884 = 0;
int32_t x3885 = 0;
int32_t x3886 = 0;
for(int x3887=0; x3887 < 64; x3887++) {
int32_t x3888 = x3885;
int32_t x3889 = x3886;
int32_t x3890 = x3884;
int32_t x3891 = x3890;
int32_t x3892 = x3888;
int32_t x3893 = x3889;
for(int x3894=0; x3894 < 128; x3894++) {
int32_t x3895 = x3892;
int32_t x3896 = x3893;
int32_t x3897 = x3891;
int32_t x3898 = x3897;
int32_t x3899 = x3895;
int32_t x3900 = x3896;
for(int x3901=0; x3901 < 16; x3901++) {
int32_t x3902 = x3899;
int32_t x3903 = x3900;
int32_t x3904 = x3898;
int32_t x3905 = x3904;
int32_t x3906 = x3902;
int32_t x3907 = x3903;
for(int x3908=0; x3908 < 16; x3908++) {
int32_t x3909 = x3905;
int32_t x3910 = x3906;
float x3911 = x3832[x3910];
int32_t x3912 = x3907;
float x3913 = x83[x3912];
float x3914 = x3911 * x3913;
x3883[x3909] = x3914;
x3905 += 1;
x3906 += 1;

}
x3898 += 16;
x3899 += 16;

}
x3891 += 256;
x3892 += 256;
x3893 += 1;

}
x3884 += 32768;
x3885 += 32768;

}
// resize to WrappedArray(-1, 1, 1)
float* x3934 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x3935 = 0;
int32_t x3936 = 0;
int32_t x3937 = 0;
for(int x3938=0; x3938 < 64; x3938++) {
int32_t x3939 = x3936;
int32_t x3940 = x3937;
int32_t x3941 = x3935;
int32_t x3942 = x3941;
int32_t x3943 = x3939;
int32_t x3944 = x3940;
for(int x3945=0; x3945 < 128; x3945++) {
int32_t x3946 = x3943;
int32_t x3947 = x3944;
int32_t x3948 = x3942;
int32_t x3949 = x3948;
int32_t x3950 = x3946;
int32_t x3951 = x3947;
for(int x3952=0; x3952 < 16; x3952++) {
int32_t x3953 = x3950;
int32_t x3954 = x3951;
int32_t x3955 = x3949;
int32_t x3956 = x3955;
int32_t x3957 = x3953;
int32_t x3958 = x3954;
for(int x3959=0; x3959 < 16; x3959++) {
int32_t x3960 = x3956;
int32_t x3961 = x3957;
float x3962 = x3883[x3961];
int32_t x3963 = x3958;
float x3964 = x171[x3963];
float x3965 = x3962 + x3964;
x3934[x3960] = x3965;
x3956 += 1;
x3957 += 1;

}
x3949 += 16;
x3950 += 16;

}
x3942 += 256;
x3943 += 256;
x3944 += 1;

}
x3935 += 32768;
x3936 += 32768;

}
float* x3984 = (float*)myMalloc(2097152 * sizeof(float));;
for(int x3986=0; x3986 < 2097152; x3986++) {
float x3987 = x3934[x3986];
bool x3988 = x3987 < 0.0f;
if (x3988) {
x3984[x3986] = 0.0f;
} else {
float x3991 = x3934[x3986];
x3984[x3986] = x3991;
}

}
float* x3997 = (float*)myMalloc(524288 * sizeof(float));;
float* x3998 = (float*)myMalloc(4718592 * sizeof(float));;
for(int x3999=0; x3999 < 64; x3999++) {
int32_t x4000 = x3999 * 32768;
float* x4001 = x3984+x4000;
int32_t x4002 = x3999 * 8192;
float* x4003 = x3997+x4002;
int32_t x4004 = x3999 * 73728;
float* x4005 = x3998+x4004;
for(int x4007=0; x4007 < 1152; x4007++) {
int32_t x4008 = x4007 / 9;
int32_t x4012 = x4008 * 3;
int32_t x4013 = x4012 * 3;
int32_t x4014 = x4013 * 8;
int32_t x4015 = x4014 * 8;
int32_t x4009 = x4007 % 9;
int32_t x4010 = x4009 / 3;
int32_t x4016 = x4010 * 3;
int32_t x4017 = x4016 * 8;
int32_t x4018 = x4017 * 8;
int32_t x4019 = x4015 + x4018;
int32_t x4011 = x4009 % 3;
int32_t x4020 = x4011 * 8;
int32_t x4021 = x4020 * 8;
int32_t x4022 = x4019 + x4021;
float* x4023 = x4005+x4022;
int32_t x4024 = x4008 * 16;
int32_t x4025 = x4024 * 16;
float* x4026 = x4001+x4025;
for(int x4028=0; x4028 < 8; x4028++) {
int32_t x4029 = x4028 * 2;
int32_t x4030 = x4029 - 1;
int32_t x4031 = x4030 + x4010;
bool x4032 = x4031 < 0;
bool x4033 = x4031 >= 16;
bool x4034 = x4032 || x4033;
if (x4034) {
int32_t x4035 = x4028 * 8;
float* x4036 = x4023+x4035;
memset(x4036, 0, 4 * 8);;
} else {
int32_t x4035 = x4028 * 8;
int32_t x4051 = x4031 * 16;
for(int x4039=0; x4039 < 8; x4039++) {
int32_t x4040 = x4039 * 2;
int32_t x4041 = x4040 - 1;
int32_t x4042 = x4041 + x4011;
bool x4043 = x4042 < 0;
bool x4044 = x4042 >= 16;
bool x4045 = x4043 || x4044;
if (x4045) {
int32_t x4046 = x4035 + x4039;
float* x4047 = x4023+x4046;
memset(x4047, 0, 4 * 1);;
} else {
int32_t x4046 = x4035 + x4039;
float* x4050 = x4023+x4046;
int32_t x4052 = x4051 + x4042;
float* x4053 = x4026+x4052;
memcpy(x4050, x4053, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,64,1152,1,x26,1152,x4005,64,1,x4003,64);

}
// resize to WrappedArray(-1, 1, 1)
float* x4069 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x4070 = 0;
int32_t x4071 = 0;
int32_t x4072 = 0;
for(int x4073=0; x4073 < 64; x4073++) {
int32_t x4074 = x4071;
int32_t x4075 = x4072;
int32_t x4076 = x4070;
int32_t x4077 = x4076;
int32_t x4078 = x4074;
int32_t x4079 = x4075;
for(int x4080=0; x4080 < 128; x4080++) {
int32_t x4081 = x4078;
int32_t x4082 = x4079;
int32_t x4083 = x4077;
int32_t x4084 = x4083;
int32_t x4085 = x4081;
int32_t x4086 = x4082;
for(int x4087=0; x4087 < 8; x4087++) {
int32_t x4088 = x4085;
int32_t x4089 = x4086;
int32_t x4090 = x4084;
int32_t x4091 = x4090;
int32_t x4092 = x4088;
int32_t x4093 = x4089;
for(int x4094=0; x4094 < 8; x4094++) {
int32_t x4095 = x4091;
int32_t x4096 = x4092;
float x4097 = x3997[x4096];
int32_t x4098 = x4093;
float x4099 = x127[x4098];
float x4100 = x4097 - x4099;
x4069[x4095] = x4100;
x4091 += 1;
x4092 += 1;

}
x4084 += 8;
x4085 += 8;

}
x4077 += 64;
x4078 += 64;
x4079 += 1;

}
x4070 += 8192;
x4071 += 8192;

}
float* x4119 = (float*)myMalloc(128 * sizeof(float));;
for(int x4120=0; x4120 < 128; x4120++) {
float x4121 = x42[x4120];
float x4122 = x4121 + 1.0E-5f;
x4119[x4120] = x4122;

}
float* x4126 = (float*)myMalloc(128 * sizeof(float));;
for(int x4127=0; x4127 < 128; x4127++) {
float x4128 = x4119[x4127];
double x4129 = (double)x4128;
double x4130 = sqrt(x4129);
float x4131 = (float)x4130;
x4126[x4127] = x4131;

}
// resize to WrappedArray(-1, 1, 1)
float* x4136 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x4137 = 0;
int32_t x4138 = 0;
int32_t x4139 = 0;
for(int x4140=0; x4140 < 64; x4140++) {
int32_t x4141 = x4138;
int32_t x4142 = x4139;
int32_t x4143 = x4137;
int32_t x4144 = x4143;
int32_t x4145 = x4141;
int32_t x4146 = x4142;
for(int x4147=0; x4147 < 128; x4147++) {
int32_t x4148 = x4145;
int32_t x4149 = x4146;
int32_t x4150 = x4144;
int32_t x4151 = x4150;
int32_t x4152 = x4148;
int32_t x4153 = x4149;
for(int x4154=0; x4154 < 8; x4154++) {
int32_t x4155 = x4152;
int32_t x4156 = x4153;
int32_t x4157 = x4151;
int32_t x4158 = x4157;
int32_t x4159 = x4155;
int32_t x4160 = x4156;
for(int x4161=0; x4161 < 8; x4161++) {
int32_t x4162 = x4158;
int32_t x4163 = x4159;
float x4164 = x4069[x4163];
int32_t x4165 = x4160;
float x4166 = x4126[x4165];
float x4167 = x4164 / x4166;
x4136[x4162] = x4167;
x4158 += 1;
x4159 += 1;

}
x4151 += 8;
x4152 += 8;

}
x4144 += 64;
x4145 += 64;
x4146 += 1;

}
x4137 += 8192;
x4138 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x4187 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x4188 = 0;
int32_t x4189 = 0;
int32_t x4190 = 0;
for(int x4191=0; x4191 < 64; x4191++) {
int32_t x4192 = x4189;
int32_t x4193 = x4190;
int32_t x4194 = x4188;
int32_t x4195 = x4194;
int32_t x4196 = x4192;
int32_t x4197 = x4193;
for(int x4198=0; x4198 < 128; x4198++) {
int32_t x4199 = x4196;
int32_t x4200 = x4197;
int32_t x4201 = x4195;
int32_t x4202 = x4201;
int32_t x4203 = x4199;
int32_t x4204 = x4200;
for(int x4205=0; x4205 < 8; x4205++) {
int32_t x4206 = x4203;
int32_t x4207 = x4204;
int32_t x4208 = x4202;
int32_t x4209 = x4208;
int32_t x4210 = x4206;
int32_t x4211 = x4207;
for(int x4212=0; x4212 < 8; x4212++) {
int32_t x4213 = x4209;
int32_t x4214 = x4210;
float x4215 = x4136[x4214];
int32_t x4216 = x4211;
float x4217 = x251[x4216];
float x4218 = x4215 * x4217;
x4187[x4213] = x4218;
x4209 += 1;
x4210 += 1;

}
x4202 += 8;
x4203 += 8;

}
x4195 += 64;
x4196 += 64;
x4197 += 1;

}
x4188 += 8192;
x4189 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x4238 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x4239 = 0;
int32_t x4240 = 0;
int32_t x4241 = 0;
for(int x4242=0; x4242 < 64; x4242++) {
int32_t x4243 = x4240;
int32_t x4244 = x4241;
int32_t x4245 = x4239;
int32_t x4246 = x4245;
int32_t x4247 = x4243;
int32_t x4248 = x4244;
for(int x4249=0; x4249 < 128; x4249++) {
int32_t x4250 = x4247;
int32_t x4251 = x4248;
int32_t x4252 = x4246;
int32_t x4253 = x4252;
int32_t x4254 = x4250;
int32_t x4255 = x4251;
for(int x4256=0; x4256 < 8; x4256++) {
int32_t x4257 = x4254;
int32_t x4258 = x4255;
int32_t x4259 = x4253;
int32_t x4260 = x4259;
int32_t x4261 = x4257;
int32_t x4262 = x4258;
for(int x4263=0; x4263 < 8; x4263++) {
int32_t x4264 = x4260;
int32_t x4265 = x4261;
float x4266 = x4187[x4265];
int32_t x4267 = x4262;
float x4268 = x189[x4267];
float x4269 = x4266 + x4268;
x4238[x4264] = x4269;
x4260 += 1;
x4261 += 1;

}
x4253 += 8;
x4254 += 8;

}
x4246 += 64;
x4247 += 64;
x4248 += 1;

}
x4239 += 8192;
x4240 += 8192;

}
float* x4288 = (float*)myMalloc(524288 * sizeof(float));;
for(int x4290=0; x4290 < 524288; x4290++) {
float x4291 = x4238[x4290];
bool x4292 = x4291 < 0.0f;
if (x4292) {
x4288[x4290] = 0.0f;
} else {
float x4295 = x4238[x4290];
x4288[x4290] = x4295;
}

}
float* x4301 = (float*)myMalloc(2097152 * sizeof(float));;
float* x4302 = (float*)myMalloc(524288 * sizeof(float));;
for(int x4303=0; x4303 < 64; x4303++) {
int32_t x4304 = x4303 * 8192;
float* x4305 = x4288+x4304;
int32_t x4306 = x4303 * 32768;
float* x4307 = x4301+x4306;
float* x4308 = x4302+x4304;
for(int x4309=0; x4309 < 128; x4309++) {
int32_t x4310 = x4309 / 1;
int32_t x4314 = x4310 * 8;
int32_t x4315 = x4314 * 8;
int32_t x4311 = x4309 % 1;
int32_t x4312 = x4311 / 1;
int32_t x4316 = x4312 * 8;
int32_t x4317 = x4316 * 8;
int32_t x4318 = x4315 + x4317;
int32_t x4313 = x4311 % 1;
int32_t x4319 = x4313 * 8;
int32_t x4320 = x4319 * 8;
int32_t x4321 = x4318 + x4320;
float* x4322 = x4308+x4321;
float* x4323 = x4305+x4315;
for(int x4324=0; x4324 < 8; x4324++) {
int32_t x4326 = x4324 * 8;
float* x4327 = x4322+x4326;
int32_t x4325 = x4324 + x4312;
int32_t x4328 = x4325 * 8;
int32_t x4329 = x4328 + x4313;
float* x4330 = x4323+x4329;
memcpy(x4327, x4330, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,64,128,1,x105,128,x4308,64,1,x4307,64);

}
// resize to WrappedArray(-1, 1, 1)
float* x4340 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x4341 = 0;
int32_t x4342 = 0;
int32_t x4343 = 0;
for(int x4344=0; x4344 < 64; x4344++) {
int32_t x4345 = x4342;
int32_t x4346 = x4343;
int32_t x4347 = x4341;
int32_t x4348 = x4347;
int32_t x4349 = x4345;
int32_t x4350 = x4346;
for(int x4352=0; x4352 < 512; x4352++) {
int32_t x4353 = x4349;
int32_t x4354 = x4350;
int32_t x4355 = x4348;
int32_t x4356 = x4355;
int32_t x4357 = x4353;
int32_t x4358 = x4354;
for(int x4359=0; x4359 < 8; x4359++) {
int32_t x4360 = x4357;
int32_t x4361 = x4358;
int32_t x4362 = x4356;
int32_t x4363 = x4362;
int32_t x4364 = x4360;
int32_t x4365 = x4361;
for(int x4366=0; x4366 < 8; x4366++) {
int32_t x4367 = x4363;
int32_t x4368 = x4364;
float x4369 = x4301[x4368];
int32_t x4370 = x4365;
float x4371 = x148[x4370];
float x4372 = x4369 - x4371;
x4340[x4367] = x4372;
x4363 += 1;
x4364 += 1;

}
x4356 += 8;
x4357 += 8;

}
x4348 += 64;
x4349 += 64;
x4350 += 1;

}
x4341 += 32768;
x4342 += 32768;

}
float* x4391 = (float*)myMalloc(512 * sizeof(float));;
for(int x4392=0; x4392 < 512; x4392++) {
float x4393 = x100[x4392];
float x4394 = x4393 + 1.0E-5f;
x4391[x4392] = x4394;

}
float* x4398 = (float*)myMalloc(512 * sizeof(float));;
for(int x4399=0; x4399 < 512; x4399++) {
float x4400 = x4391[x4399];
double x4401 = (double)x4400;
double x4402 = sqrt(x4401);
float x4403 = (float)x4402;
x4398[x4399] = x4403;

}
// resize to WrappedArray(-1, 1, 1)
float* x4408 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x4409 = 0;
int32_t x4410 = 0;
int32_t x4411 = 0;
for(int x4412=0; x4412 < 64; x4412++) {
int32_t x4413 = x4410;
int32_t x4414 = x4411;
int32_t x4415 = x4409;
int32_t x4416 = x4415;
int32_t x4417 = x4413;
int32_t x4418 = x4414;
for(int x4419=0; x4419 < 512; x4419++) {
int32_t x4420 = x4417;
int32_t x4421 = x4418;
int32_t x4422 = x4416;
int32_t x4423 = x4422;
int32_t x4424 = x4420;
int32_t x4425 = x4421;
for(int x4426=0; x4426 < 8; x4426++) {
int32_t x4427 = x4424;
int32_t x4428 = x4425;
int32_t x4429 = x4423;
int32_t x4430 = x4429;
int32_t x4431 = x4427;
int32_t x4432 = x4428;
for(int x4433=0; x4433 < 8; x4433++) {
int32_t x4434 = x4430;
int32_t x4435 = x4431;
float x4436 = x4340[x4435];
int32_t x4437 = x4432;
float x4438 = x4398[x4437];
float x4439 = x4436 / x4438;
x4408[x4434] = x4439;
x4430 += 1;
x4431 += 1;

}
x4423 += 8;
x4424 += 8;

}
x4416 += 64;
x4417 += 64;
x4418 += 1;

}
x4409 += 32768;
x4410 += 32768;

}
// resize to WrappedArray(-1, 1, 1)
float* x4459 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x4460 = 0;
int32_t x4461 = 0;
int32_t x4462 = 0;
for(int x4463=0; x4463 < 64; x4463++) {
int32_t x4464 = x4461;
int32_t x4465 = x4462;
int32_t x4466 = x4460;
int32_t x4467 = x4466;
int32_t x4468 = x4464;
int32_t x4469 = x4465;
for(int x4470=0; x4470 < 512; x4470++) {
int32_t x4471 = x4468;
int32_t x4472 = x4469;
int32_t x4473 = x4467;
int32_t x4474 = x4473;
int32_t x4475 = x4471;
int32_t x4476 = x4472;
for(int x4477=0; x4477 < 8; x4477++) {
int32_t x4478 = x4475;
int32_t x4479 = x4476;
int32_t x4480 = x4474;
int32_t x4481 = x4480;
int32_t x4482 = x4478;
int32_t x4483 = x4479;
for(int x4484=0; x4484 < 8; x4484++) {
int32_t x4485 = x4481;
int32_t x4486 = x4482;
float x4487 = x4408[x4486];
int32_t x4488 = x4483;
float x4489 = x144[x4488];
float x4490 = x4487 * x4489;
x4459[x4485] = x4490;
x4481 += 1;
x4482 += 1;

}
x4474 += 8;
x4475 += 8;

}
x4467 += 64;
x4468 += 64;
x4469 += 1;

}
x4460 += 32768;
x4461 += 32768;

}
// resize to WrappedArray(-1, 1, 1)
float* x4510 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x4511 = 0;
int32_t x4512 = 0;
int32_t x4513 = 0;
for(int x4514=0; x4514 < 64; x4514++) {
int32_t x4515 = x4512;
int32_t x4516 = x4513;
int32_t x4517 = x4511;
int32_t x4518 = x4517;
int32_t x4519 = x4515;
int32_t x4520 = x4516;
for(int x4521=0; x4521 < 512; x4521++) {
int32_t x4522 = x4519;
int32_t x4523 = x4520;
int32_t x4524 = x4518;
int32_t x4525 = x4524;
int32_t x4526 = x4522;
int32_t x4527 = x4523;
for(int x4528=0; x4528 < 8; x4528++) {
int32_t x4529 = x4526;
int32_t x4530 = x4527;
int32_t x4531 = x4525;
int32_t x4532 = x4531;
int32_t x4533 = x4529;
int32_t x4534 = x4530;
for(int x4535=0; x4535 < 8; x4535++) {
int32_t x4536 = x4532;
int32_t x4537 = x4533;
float x4538 = x4459[x4537];
int32_t x4539 = x4534;
float x4540 = x209[x4539];
float x4541 = x4538 + x4540;
x4510[x4536] = x4541;
x4532 += 1;
x4533 += 1;

}
x4525 += 8;
x4526 += 8;

}
x4518 += 64;
x4519 += 64;
x4520 += 1;

}
x4511 += 32768;
x4512 += 32768;

}
float* x4560 = (float*)myMalloc(2097152 * sizeof(float));;
float* x4561 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x4562=0; x4562 < 64; x4562++) {
int32_t x4563 = x4562 * 65536;
float* x4564 = x3713+x4563;
int32_t x4565 = x4562 * 32768;
float* x4566 = x4560+x4565;
int32_t x4567 = x4562 * 16384;
float* x4568 = x4561+x4567;
for(int x4569=0; x4569 < 256; x4569++) {
int32_t x4570 = x4569 / 1;
int32_t x4574 = x4570 * 8;
int32_t x4575 = x4574 * 8;
int32_t x4571 = x4569 % 1;
int32_t x4572 = x4571 / 1;
int32_t x4576 = x4572 * 8;
int32_t x4577 = x4576 * 8;
int32_t x4578 = x4575 + x4577;
int32_t x4573 = x4571 % 1;
int32_t x4579 = x4573 * 8;
int32_t x4580 = x4579 * 8;
int32_t x4581 = x4578 + x4580;
float* x4582 = x4568+x4581;
int32_t x4583 = x4570 * 16;
int32_t x4584 = x4583 * 16;
float* x4585 = x4564+x4584;
for(int x4586=0; x4586 < 8; x4586++) {
int32_t x4590 = x4586 * 8;
int32_t x4587 = x4586 * 2;
int32_t x4588 = x4587 + x4572;
int32_t x4593 = x4588 * 16;
int32_t x4594 = x4593 + x4573;
for(int x4589=0; x4589 < 8; x4589++) {
int32_t x4591 = x4590 + x4589;
float* x4592 = x4582+x4591;
int32_t x4595 = x4589 * 2;
int32_t x4596 = x4594 + x4595;
float* x4597 = x4585+x4596;
memcpy(x4592, x4597, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,64,256,1,x257,256,x4568,64,1,x4566,64);

}
// resize to WrappedArray(-1, 1, 1)
float* x4609 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x4610 = 0;
int32_t x4611 = 0;
int32_t x4612 = 0;
for(int x4613=0; x4613 < 64; x4613++) {
int32_t x4614 = x4611;
int32_t x4615 = x4612;
int32_t x4616 = x4610;
int32_t x4617 = x4616;
int32_t x4618 = x4614;
int32_t x4619 = x4615;
for(int x4620=0; x4620 < 512; x4620++) {
int32_t x4621 = x4618;
int32_t x4622 = x4619;
int32_t x4623 = x4617;
int32_t x4624 = x4623;
int32_t x4625 = x4621;
int32_t x4626 = x4622;
for(int x4627=0; x4627 < 8; x4627++) {
int32_t x4628 = x4625;
int32_t x4629 = x4626;
int32_t x4630 = x4624;
int32_t x4631 = x4630;
int32_t x4632 = x4628;
int32_t x4633 = x4629;
for(int x4634=0; x4634 < 8; x4634++) {
int32_t x4635 = x4631;
int32_t x4636 = x4632;
float x4637 = x4560[x4636];
int32_t x4638 = x4633;
float x4639 = x41[x4638];
float x4640 = x4637 - x4639;
x4609[x4635] = x4640;
x4631 += 1;
x4632 += 1;

}
x4624 += 8;
x4625 += 8;

}
x4617 += 64;
x4618 += 64;
x4619 += 1;

}
x4610 += 32768;
x4611 += 32768;

}
float* x4659 = (float*)myMalloc(512 * sizeof(float));;
for(int x4660=0; x4660 < 512; x4660++) {
float x4661 = x22[x4660];
float x4662 = x4661 + 1.0E-5f;
x4659[x4660] = x4662;

}
float* x4666 = (float*)myMalloc(512 * sizeof(float));;
for(int x4667=0; x4667 < 512; x4667++) {
float x4668 = x4659[x4667];
double x4669 = (double)x4668;
double x4670 = sqrt(x4669);
float x4671 = (float)x4670;
x4666[x4667] = x4671;

}
// resize to WrappedArray(-1, 1, 1)
float* x4676 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x4677 = 0;
int32_t x4678 = 0;
int32_t x4679 = 0;
for(int x4680=0; x4680 < 64; x4680++) {
int32_t x4681 = x4678;
int32_t x4682 = x4679;
int32_t x4683 = x4677;
int32_t x4684 = x4683;
int32_t x4685 = x4681;
int32_t x4686 = x4682;
for(int x4687=0; x4687 < 512; x4687++) {
int32_t x4688 = x4685;
int32_t x4689 = x4686;
int32_t x4690 = x4684;
int32_t x4691 = x4690;
int32_t x4692 = x4688;
int32_t x4693 = x4689;
for(int x4694=0; x4694 < 8; x4694++) {
int32_t x4695 = x4692;
int32_t x4696 = x4693;
int32_t x4697 = x4691;
int32_t x4698 = x4697;
int32_t x4699 = x4695;
int32_t x4700 = x4696;
for(int x4701=0; x4701 < 8; x4701++) {
int32_t x4702 = x4698;
int32_t x4703 = x4699;
float x4704 = x4609[x4703];
int32_t x4705 = x4700;
float x4706 = x4666[x4705];
float x4707 = x4704 / x4706;
x4676[x4702] = x4707;
x4698 += 1;
x4699 += 1;

}
x4691 += 8;
x4692 += 8;

}
x4684 += 64;
x4685 += 64;
x4686 += 1;

}
x4677 += 32768;
x4678 += 32768;

}
// resize to WrappedArray(-1, 1, 1)
float* x4727 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x4728 = 0;
int32_t x4729 = 0;
int32_t x4730 = 0;
for(int x4731=0; x4731 < 64; x4731++) {
int32_t x4732 = x4729;
int32_t x4733 = x4730;
int32_t x4734 = x4728;
int32_t x4735 = x4734;
int32_t x4736 = x4732;
int32_t x4737 = x4733;
for(int x4738=0; x4738 < 512; x4738++) {
int32_t x4739 = x4736;
int32_t x4740 = x4737;
int32_t x4741 = x4735;
int32_t x4742 = x4741;
int32_t x4743 = x4739;
int32_t x4744 = x4740;
for(int x4745=0; x4745 < 8; x4745++) {
int32_t x4746 = x4743;
int32_t x4747 = x4744;
int32_t x4748 = x4742;
int32_t x4749 = x4748;
int32_t x4750 = x4746;
int32_t x4751 = x4747;
for(int x4752=0; x4752 < 8; x4752++) {
int32_t x4753 = x4749;
int32_t x4754 = x4750;
float x4755 = x4676[x4754];
int32_t x4756 = x4751;
float x4757 = x206[x4756];
float x4758 = x4755 * x4757;
x4727[x4753] = x4758;
x4749 += 1;
x4750 += 1;

}
x4742 += 8;
x4743 += 8;

}
x4735 += 64;
x4736 += 64;
x4737 += 1;

}
x4728 += 32768;
x4729 += 32768;

}
// resize to WrappedArray(-1, 1, 1)
float* x4778 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x4779 = 0;
int32_t x4780 = 0;
int32_t x4781 = 0;
for(int x4782=0; x4782 < 64; x4782++) {
int32_t x4783 = x4780;
int32_t x4784 = x4781;
int32_t x4785 = x4779;
int32_t x4786 = x4785;
int32_t x4787 = x4783;
int32_t x4788 = x4784;
for(int x4789=0; x4789 < 512; x4789++) {
int32_t x4790 = x4787;
int32_t x4791 = x4788;
int32_t x4792 = x4786;
int32_t x4793 = x4792;
int32_t x4794 = x4790;
int32_t x4795 = x4791;
for(int x4796=0; x4796 < 8; x4796++) {
int32_t x4797 = x4794;
int32_t x4798 = x4795;
int32_t x4799 = x4793;
int32_t x4800 = x4799;
int32_t x4801 = x4797;
int32_t x4802 = x4798;
for(int x4803=0; x4803 < 8; x4803++) {
int32_t x4804 = x4800;
int32_t x4805 = x4801;
float x4806 = x4727[x4805];
int32_t x4807 = x4802;
float x4808 = x118[x4807];
float x4809 = x4806 + x4808;
x4778[x4804] = x4809;
x4800 += 1;
x4801 += 1;

}
x4793 += 8;
x4794 += 8;

}
x4786 += 64;
x4787 += 64;
x4788 += 1;

}
x4779 += 32768;
x4780 += 32768;

}
int32_t x4828 = 0;
int32_t x4829 = 0;
int32_t x4830 = 0;
for(int x4831=0; x4831 < 64; x4831++) {
int32_t x4832 = x4829;
int32_t x4833 = x4830;
int32_t x4834 = x4828;
int32_t x4835 = x4834;
int32_t x4836 = x4832;
int32_t x4837 = x4833;
for(int x4838=0; x4838 < 512; x4838++) {
int32_t x4839 = x4836;
int32_t x4840 = x4837;
int32_t x4841 = x4835;
int32_t x4842 = x4841;
int32_t x4843 = x4839;
int32_t x4844 = x4840;
for(int x4845=0; x4845 < 8; x4845++) {
int32_t x4846 = x4843;
int32_t x4847 = x4844;
int32_t x4848 = x4842;
int32_t x4849 = x4848;
int32_t x4850 = x4846;
int32_t x4851 = x4847;
for(int x4852=0; x4852 < 8; x4852++) {
int32_t x4853 = x4850;
float x4854 = x4510[x4853];
int32_t x4855 = x4851;
float x4856 = x4778[x4855];
float x4857 = x4854 + x4856;
x4510[x4853] = x4857;
x4849 += 1;
x4850 += 1;
x4851 += 1;

}
x4842 += 8;
x4843 += 8;
x4844 += 8;

}
x4835 += 64;
x4836 += 64;
x4837 += 64;

}
x4828 += 32768;
x4829 += 32768;
x4830 += 32768;

}
float* x4879 = (float*)myMalloc(2097152 * sizeof(float));;
for(int x4880=0; x4880 < 2097152; x4880++) {
float x4881 = x4510[x4880];
bool x4882 = x4881 < 0.0f;
if (x4882) {
x4879[x4880] = 0.0f;
} else {
float x4885 = x4510[x4880];
x4879[x4880] = x4885;
}

}
float* x4891 = (float*)myMalloc(524288 * sizeof(float));;
float* x4892 = (float*)myMalloc(2097152 * sizeof(float));;
for(int x4893=0; x4893 < 64; x4893++) {
int32_t x4894 = x4893 * 32768;
float* x4895 = x4879+x4894;
int32_t x4896 = x4893 * 8192;
float* x4897 = x4891+x4896;
float* x4898 = x4892+x4894;
for(int x4899=0; x4899 < 512; x4899++) {
int32_t x4900 = x4899 / 1;
int32_t x4904 = x4900 * 8;
int32_t x4905 = x4904 * 8;
int32_t x4901 = x4899 % 1;
int32_t x4902 = x4901 / 1;
int32_t x4906 = x4902 * 8;
int32_t x4907 = x4906 * 8;
int32_t x4908 = x4905 + x4907;
int32_t x4903 = x4901 % 1;
int32_t x4909 = x4903 * 8;
int32_t x4910 = x4909 * 8;
int32_t x4911 = x4908 + x4910;
float* x4912 = x4898+x4911;
float* x4913 = x4895+x4905;
for(int x4914=0; x4914 < 8; x4914++) {
int32_t x4916 = x4914 * 8;
float* x4917 = x4912+x4916;
int32_t x4915 = x4914 + x4902;
int32_t x4918 = x4915 * 8;
int32_t x4919 = x4918 + x4903;
float* x4920 = x4913+x4919;
memcpy(x4917, x4920, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,64,512,1,x255,512,x4898,64,1,x4897,64);

}
// resize to WrappedArray(-1, 1, 1)
float* x4930 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x4931 = 0;
int32_t x4932 = 0;
int32_t x4933 = 0;
for(int x4934=0; x4934 < 64; x4934++) {
int32_t x4935 = x4932;
int32_t x4936 = x4933;
int32_t x4937 = x4931;
int32_t x4938 = x4937;
int32_t x4939 = x4935;
int32_t x4940 = x4936;
for(int x4941=0; x4941 < 128; x4941++) {
int32_t x4942 = x4939;
int32_t x4943 = x4940;
int32_t x4944 = x4938;
int32_t x4945 = x4944;
int32_t x4946 = x4942;
int32_t x4947 = x4943;
for(int x4948=0; x4948 < 8; x4948++) {
int32_t x4949 = x4946;
int32_t x4950 = x4947;
int32_t x4951 = x4945;
int32_t x4952 = x4951;
int32_t x4953 = x4949;
int32_t x4954 = x4950;
for(int x4955=0; x4955 < 8; x4955++) {
int32_t x4956 = x4952;
int32_t x4957 = x4953;
float x4958 = x4891[x4957];
int32_t x4959 = x4954;
float x4960 = x99[x4959];
float x4961 = x4958 - x4960;
x4930[x4956] = x4961;
x4952 += 1;
x4953 += 1;

}
x4945 += 8;
x4946 += 8;

}
x4938 += 64;
x4939 += 64;
x4940 += 1;

}
x4931 += 8192;
x4932 += 8192;

}
float* x4980 = (float*)myMalloc(128 * sizeof(float));;
for(int x4981=0; x4981 < 128; x4981++) {
float x4982 = x176[x4981];
float x4983 = x4982 + 1.0E-5f;
x4980[x4981] = x4983;

}
float* x4987 = (float*)myMalloc(128 * sizeof(float));;
for(int x4988=0; x4988 < 128; x4988++) {
float x4989 = x4980[x4988];
double x4990 = (double)x4989;
double x4991 = sqrt(x4990);
float x4992 = (float)x4991;
x4987[x4988] = x4992;

}
// resize to WrappedArray(-1, 1, 1)
float* x4997 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x4998 = 0;
int32_t x4999 = 0;
int32_t x5000 = 0;
for(int x5001=0; x5001 < 64; x5001++) {
int32_t x5002 = x4999;
int32_t x5003 = x5000;
int32_t x5004 = x4998;
int32_t x5005 = x5004;
int32_t x5006 = x5002;
int32_t x5007 = x5003;
for(int x5008=0; x5008 < 128; x5008++) {
int32_t x5009 = x5006;
int32_t x5010 = x5007;
int32_t x5011 = x5005;
int32_t x5012 = x5011;
int32_t x5013 = x5009;
int32_t x5014 = x5010;
for(int x5015=0; x5015 < 8; x5015++) {
int32_t x5016 = x5013;
int32_t x5017 = x5014;
int32_t x5018 = x5012;
int32_t x5019 = x5018;
int32_t x5020 = x5016;
int32_t x5021 = x5017;
for(int x5022=0; x5022 < 8; x5022++) {
int32_t x5023 = x5019;
int32_t x5024 = x5020;
float x5025 = x4930[x5024];
int32_t x5026 = x5021;
float x5027 = x4987[x5026];
float x5028 = x5025 / x5027;
x4997[x5023] = x5028;
x5019 += 1;
x5020 += 1;

}
x5012 += 8;
x5013 += 8;

}
x5005 += 64;
x5006 += 64;
x5007 += 1;

}
x4998 += 8192;
x4999 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x5048 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x5049 = 0;
int32_t x5050 = 0;
int32_t x5051 = 0;
for(int x5052=0; x5052 < 64; x5052++) {
int32_t x5053 = x5050;
int32_t x5054 = x5051;
int32_t x5055 = x5049;
int32_t x5056 = x5055;
int32_t x5057 = x5053;
int32_t x5058 = x5054;
for(int x5059=0; x5059 < 128; x5059++) {
int32_t x5060 = x5057;
int32_t x5061 = x5058;
int32_t x5062 = x5056;
int32_t x5063 = x5062;
int32_t x5064 = x5060;
int32_t x5065 = x5061;
for(int x5066=0; x5066 < 8; x5066++) {
int32_t x5067 = x5064;
int32_t x5068 = x5065;
int32_t x5069 = x5063;
int32_t x5070 = x5069;
int32_t x5071 = x5067;
int32_t x5072 = x5068;
for(int x5073=0; x5073 < 8; x5073++) {
int32_t x5074 = x5070;
int32_t x5075 = x5071;
float x5076 = x4997[x5075];
int32_t x5077 = x5072;
float x5078 = x221[x5077];
float x5079 = x5076 * x5078;
x5048[x5074] = x5079;
x5070 += 1;
x5071 += 1;

}
x5063 += 8;
x5064 += 8;

}
x5056 += 64;
x5057 += 64;
x5058 += 1;

}
x5049 += 8192;
x5050 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x5099 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x5100 = 0;
int32_t x5101 = 0;
int32_t x5102 = 0;
for(int x5103=0; x5103 < 64; x5103++) {
int32_t x5104 = x5101;
int32_t x5105 = x5102;
int32_t x5106 = x5100;
int32_t x5107 = x5106;
int32_t x5108 = x5104;
int32_t x5109 = x5105;
for(int x5110=0; x5110 < 128; x5110++) {
int32_t x5111 = x5108;
int32_t x5112 = x5109;
int32_t x5113 = x5107;
int32_t x5114 = x5113;
int32_t x5115 = x5111;
int32_t x5116 = x5112;
for(int x5117=0; x5117 < 8; x5117++) {
int32_t x5118 = x5115;
int32_t x5119 = x5116;
int32_t x5120 = x5114;
int32_t x5121 = x5120;
int32_t x5122 = x5118;
int32_t x5123 = x5119;
for(int x5124=0; x5124 < 8; x5124++) {
int32_t x5125 = x5121;
int32_t x5126 = x5122;
float x5127 = x5048[x5126];
int32_t x5128 = x5123;
float x5129 = x16[x5128];
float x5130 = x5127 + x5129;
x5099[x5125] = x5130;
x5121 += 1;
x5122 += 1;

}
x5114 += 8;
x5115 += 8;

}
x5107 += 64;
x5108 += 64;
x5109 += 1;

}
x5100 += 8192;
x5101 += 8192;

}
float* x5149 = (float*)myMalloc(524288 * sizeof(float));;
for(int x5150=0; x5150 < 524288; x5150++) {
float x5151 = x5099[x5150];
bool x5152 = x5151 < 0.0f;
if (x5152) {
x5149[x5150] = 0.0f;
} else {
float x5155 = x5099[x5150];
x5149[x5150] = x5155;
}

}
float* x5161 = (float*)myMalloc(524288 * sizeof(float));;
float* x5162 = (float*)myMalloc(4718592 * sizeof(float));;
for(int x5163=0; x5163 < 64; x5163++) {
int32_t x5164 = x5163 * 8192;
float* x5165 = x5149+x5164;
float* x5166 = x5161+x5164;
int32_t x5167 = x5163 * 73728;
float* x5168 = x5162+x5167;
for(int x5169=0; x5169 < 1152; x5169++) {
int32_t x5170 = x5169 / 9;
int32_t x5174 = x5170 * 3;
int32_t x5175 = x5174 * 3;
int32_t x5176 = x5175 * 8;
int32_t x5177 = x5176 * 8;
int32_t x5171 = x5169 % 9;
int32_t x5172 = x5171 / 3;
int32_t x5178 = x5172 * 3;
int32_t x5179 = x5178 * 8;
int32_t x5180 = x5179 * 8;
int32_t x5181 = x5177 + x5180;
int32_t x5173 = x5171 % 3;
int32_t x5182 = x5173 * 8;
int32_t x5183 = x5182 * 8;
int32_t x5184 = x5181 + x5183;
float* x5185 = x5168+x5184;
int32_t x5186 = x5170 * 8;
int32_t x5187 = x5186 * 8;
float* x5188 = x5165+x5187;
int32_t x5200 = 1 - x5173;
bool x5201 = x5200 > 0;
int32_t x5202;
if (x5201) {
x5202 = x5200;
} else {
x5202 = 0;
}
int32_t x5203 = 3 - x5173;
int32_t x5204 = x5203 - 1;
int32_t x5205 = 1 - x5204;
bool x5206 = x5205 > 0;
int32_t x5207;
if (x5206) {
x5207 = x5205;
} else {
x5207 = 0;
}
int32_t x5208 = 8 - x5207;
int32_t x5209 = x5208 - x5202;
bool x5210 = x5209 <= 0;
bool x5214 = x5202 > 0;
int32_t x5199 = -1 + x5173;
bool x5227 = x5207 > 0;
for(int x5189=0; x5189 < 8; x5189++) {
int32_t x5190 = x5189 - 1;
int32_t x5191 = x5190 + x5172;
bool x5192 = x5191 < 0;
bool x5193 = x5191 >= 8;
bool x5194 = x5192 || x5193;
if (x5194) {
int32_t x5195 = x5189 * 8;
float* x5196 = x5185+x5195;
memset(x5196, 0, 4 * 8);;
} else {
if (x5210) {
int32_t x5195 = x5189 * 8;
float* x5211 = x5185+x5195;
memset(x5211, 0, 4 * 8);;
} else {
int32_t x5195 = x5189 * 8;
if (x5214) {
float* x5215 = x5185+x5195;
memset(x5215, 0, 4 * x5202);;
} else {
}
// may have segfault here
int32_t x5220 = x5195 + x5202;
float* x5221 = x5185+x5220;
int32_t x5222 = x5191 * 8;
int32_t x5223 = x5222 + x5199;
int32_t x5224 = x5223 + x5202;
float* x5225 = x5188+x5224;
memcpy(x5221, x5225, 4 * x5209);;
if (x5227) {
int32_t x5228 = x5195 + 8;
int32_t x5229 = x5228 - x5207;
float* x5230 = x5185+x5229;
memset(x5230, 0, 4 * x5207);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,64,1152,1,x234,1152,x5168,64,1,x5166,64);

}
// resize to WrappedArray(-1, 1, 1)
float* x5246 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x5247 = 0;
int32_t x5248 = 0;
int32_t x5249 = 0;
for(int x5250=0; x5250 < 64; x5250++) {
int32_t x5251 = x5248;
int32_t x5252 = x5249;
int32_t x5253 = x5247;
int32_t x5254 = x5253;
int32_t x5255 = x5251;
int32_t x5256 = x5252;
for(int x5257=0; x5257 < 128; x5257++) {
int32_t x5258 = x5255;
int32_t x5259 = x5256;
int32_t x5260 = x5254;
int32_t x5261 = x5260;
int32_t x5262 = x5258;
int32_t x5263 = x5259;
for(int x5264=0; x5264 < 8; x5264++) {
int32_t x5265 = x5262;
int32_t x5266 = x5263;
int32_t x5267 = x5261;
int32_t x5268 = x5267;
int32_t x5269 = x5265;
int32_t x5270 = x5266;
for(int x5271=0; x5271 < 8; x5271++) {
int32_t x5272 = x5268;
int32_t x5273 = x5269;
float x5274 = x5161[x5273];
int32_t x5275 = x5270;
float x5276 = x34[x5275];
float x5277 = x5274 - x5276;
x5246[x5272] = x5277;
x5268 += 1;
x5269 += 1;

}
x5261 += 8;
x5262 += 8;

}
x5254 += 64;
x5255 += 64;
x5256 += 1;

}
x5247 += 8192;
x5248 += 8192;

}
float* x5296 = (float*)myMalloc(128 * sizeof(float));;
for(int x5297=0; x5297 < 128; x5297++) {
float x5298 = x224[x5297];
float x5299 = x5298 + 1.0E-5f;
x5296[x5297] = x5299;

}
float* x5303 = (float*)myMalloc(128 * sizeof(float));;
for(int x5304=0; x5304 < 128; x5304++) {
float x5305 = x5296[x5304];
double x5306 = (double)x5305;
double x5307 = sqrt(x5306);
float x5308 = (float)x5307;
x5303[x5304] = x5308;

}
// resize to WrappedArray(-1, 1, 1)
float* x5313 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x5314 = 0;
int32_t x5315 = 0;
int32_t x5316 = 0;
for(int x5317=0; x5317 < 64; x5317++) {
int32_t x5318 = x5315;
int32_t x5319 = x5316;
int32_t x5320 = x5314;
int32_t x5321 = x5320;
int32_t x5322 = x5318;
int32_t x5323 = x5319;
for(int x5324=0; x5324 < 128; x5324++) {
int32_t x5325 = x5322;
int32_t x5326 = x5323;
int32_t x5327 = x5321;
int32_t x5328 = x5327;
int32_t x5329 = x5325;
int32_t x5330 = x5326;
for(int x5331=0; x5331 < 8; x5331++) {
int32_t x5332 = x5329;
int32_t x5333 = x5330;
int32_t x5334 = x5328;
int32_t x5335 = x5334;
int32_t x5336 = x5332;
int32_t x5337 = x5333;
for(int x5338=0; x5338 < 8; x5338++) {
int32_t x5339 = x5335;
int32_t x5340 = x5336;
float x5341 = x5246[x5340];
int32_t x5342 = x5337;
float x5343 = x5303[x5342];
float x5344 = x5341 / x5343;
x5313[x5339] = x5344;
x5335 += 1;
x5336 += 1;

}
x5328 += 8;
x5329 += 8;

}
x5321 += 64;
x5322 += 64;
x5323 += 1;

}
x5314 += 8192;
x5315 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x5364 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x5365 = 0;
int32_t x5366 = 0;
int32_t x5367 = 0;
for(int x5368=0; x5368 < 64; x5368++) {
int32_t x5369 = x5366;
int32_t x5370 = x5367;
int32_t x5371 = x5365;
int32_t x5372 = x5371;
int32_t x5373 = x5369;
int32_t x5374 = x5370;
for(int x5375=0; x5375 < 128; x5375++) {
int32_t x5376 = x5373;
int32_t x5377 = x5374;
int32_t x5378 = x5372;
int32_t x5379 = x5378;
int32_t x5380 = x5376;
int32_t x5381 = x5377;
for(int x5382=0; x5382 < 8; x5382++) {
int32_t x5383 = x5380;
int32_t x5384 = x5381;
int32_t x5385 = x5379;
int32_t x5386 = x5385;
int32_t x5387 = x5383;
int32_t x5388 = x5384;
for(int x5389=0; x5389 < 8; x5389++) {
int32_t x5390 = x5386;
int32_t x5391 = x5387;
float x5392 = x5313[x5391];
int32_t x5393 = x5388;
float x5394 = x7[x5393];
float x5395 = x5392 * x5394;
x5364[x5390] = x5395;
x5386 += 1;
x5387 += 1;

}
x5379 += 8;
x5380 += 8;

}
x5372 += 64;
x5373 += 64;
x5374 += 1;

}
x5365 += 8192;
x5366 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x5415 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x5416 = 0;
int32_t x5417 = 0;
int32_t x5418 = 0;
for(int x5419=0; x5419 < 64; x5419++) {
int32_t x5420 = x5417;
int32_t x5421 = x5418;
int32_t x5422 = x5416;
int32_t x5423 = x5422;
int32_t x5424 = x5420;
int32_t x5425 = x5421;
for(int x5426=0; x5426 < 128; x5426++) {
int32_t x5427 = x5424;
int32_t x5428 = x5425;
int32_t x5429 = x5423;
int32_t x5430 = x5429;
int32_t x5431 = x5427;
int32_t x5432 = x5428;
for(int x5433=0; x5433 < 8; x5433++) {
int32_t x5434 = x5431;
int32_t x5435 = x5432;
int32_t x5436 = x5430;
int32_t x5437 = x5436;
int32_t x5438 = x5434;
int32_t x5439 = x5435;
for(int x5440=0; x5440 < 8; x5440++) {
int32_t x5441 = x5437;
int32_t x5442 = x5438;
float x5443 = x5364[x5442];
int32_t x5444 = x5439;
float x5445 = x94[x5444];
float x5446 = x5443 + x5445;
x5415[x5441] = x5446;
x5437 += 1;
x5438 += 1;

}
x5430 += 8;
x5431 += 8;

}
x5423 += 64;
x5424 += 64;
x5425 += 1;

}
x5416 += 8192;
x5417 += 8192;

}
float* x5465 = (float*)myMalloc(524288 * sizeof(float));;
for(int x5466=0; x5466 < 524288; x5466++) {
float x5467 = x5415[x5466];
bool x5468 = x5467 < 0.0f;
if (x5468) {
x5465[x5466] = 0.0f;
} else {
float x5471 = x5415[x5466];
x5465[x5466] = x5471;
}

}
float* x5477 = (float*)myMalloc(2097152 * sizeof(float));;
float* x5478 = (float*)myMalloc(524288 * sizeof(float));;
for(int x5479=0; x5479 < 64; x5479++) {
int32_t x5480 = x5479 * 8192;
float* x5481 = x5465+x5480;
int32_t x5482 = x5479 * 32768;
float* x5483 = x5477+x5482;
float* x5484 = x5478+x5480;
for(int x5485=0; x5485 < 128; x5485++) {
int32_t x5486 = x5485 / 1;
int32_t x5490 = x5486 * 8;
int32_t x5491 = x5490 * 8;
int32_t x5487 = x5485 % 1;
int32_t x5488 = x5487 / 1;
int32_t x5492 = x5488 * 8;
int32_t x5493 = x5492 * 8;
int32_t x5494 = x5491 + x5493;
int32_t x5489 = x5487 % 1;
int32_t x5495 = x5489 * 8;
int32_t x5496 = x5495 * 8;
int32_t x5497 = x5494 + x5496;
float* x5498 = x5484+x5497;
float* x5499 = x5481+x5491;
for(int x5500=0; x5500 < 8; x5500++) {
int32_t x5502 = x5500 * 8;
float* x5503 = x5498+x5502;
int32_t x5501 = x5500 + x5488;
int32_t x5504 = x5501 * 8;
int32_t x5505 = x5504 + x5489;
float* x5506 = x5499+x5505;
memcpy(x5503, x5506, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,64,128,1,x110,128,x5484,64,1,x5483,64);

}
// resize to WrappedArray(-1, 1, 1)
float* x5516 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x5517 = 0;
int32_t x5518 = 0;
int32_t x5519 = 0;
for(int x5520=0; x5520 < 64; x5520++) {
int32_t x5521 = x5518;
int32_t x5522 = x5519;
int32_t x5523 = x5517;
int32_t x5524 = x5523;
int32_t x5525 = x5521;
int32_t x5526 = x5522;
for(int x5527=0; x5527 < 512; x5527++) {
int32_t x5528 = x5525;
int32_t x5529 = x5526;
int32_t x5530 = x5524;
int32_t x5531 = x5530;
int32_t x5532 = x5528;
int32_t x5533 = x5529;
for(int x5534=0; x5534 < 8; x5534++) {
int32_t x5535 = x5532;
int32_t x5536 = x5533;
int32_t x5537 = x5531;
int32_t x5538 = x5537;
int32_t x5539 = x5535;
int32_t x5540 = x5536;
for(int x5541=0; x5541 < 8; x5541++) {
int32_t x5542 = x5538;
int32_t x5543 = x5539;
float x5544 = x5477[x5543];
int32_t x5545 = x5540;
float x5546 = x146[x5545];
float x5547 = x5544 - x5546;
x5516[x5542] = x5547;
x5538 += 1;
x5539 += 1;

}
x5531 += 8;
x5532 += 8;

}
x5524 += 64;
x5525 += 64;
x5526 += 1;

}
x5517 += 32768;
x5518 += 32768;

}
float* x5566 = (float*)myMalloc(512 * sizeof(float));;
for(int x5567=0; x5567 < 512; x5567++) {
float x5568 = x87[x5567];
float x5569 = x5568 + 1.0E-5f;
x5566[x5567] = x5569;

}
float* x5573 = (float*)myMalloc(512 * sizeof(float));;
for(int x5574=0; x5574 < 512; x5574++) {
float x5575 = x5566[x5574];
double x5576 = (double)x5575;
double x5577 = sqrt(x5576);
float x5578 = (float)x5577;
x5573[x5574] = x5578;

}
// resize to WrappedArray(-1, 1, 1)
float* x5583 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x5584 = 0;
int32_t x5585 = 0;
int32_t x5586 = 0;
for(int x5587=0; x5587 < 64; x5587++) {
int32_t x5588 = x5585;
int32_t x5589 = x5586;
int32_t x5590 = x5584;
int32_t x5591 = x5590;
int32_t x5592 = x5588;
int32_t x5593 = x5589;
for(int x5594=0; x5594 < 512; x5594++) {
int32_t x5595 = x5592;
int32_t x5596 = x5593;
int32_t x5597 = x5591;
int32_t x5598 = x5597;
int32_t x5599 = x5595;
int32_t x5600 = x5596;
for(int x5601=0; x5601 < 8; x5601++) {
int32_t x5602 = x5599;
int32_t x5603 = x5600;
int32_t x5604 = x5598;
int32_t x5605 = x5604;
int32_t x5606 = x5602;
int32_t x5607 = x5603;
for(int x5608=0; x5608 < 8; x5608++) {
int32_t x5609 = x5605;
int32_t x5610 = x5606;
float x5611 = x5516[x5610];
int32_t x5612 = x5607;
float x5613 = x5573[x5612];
float x5614 = x5611 / x5613;
x5583[x5609] = x5614;
x5605 += 1;
x5606 += 1;

}
x5598 += 8;
x5599 += 8;

}
x5591 += 64;
x5592 += 64;
x5593 += 1;

}
x5584 += 32768;
x5585 += 32768;

}
// resize to WrappedArray(-1, 1, 1)
float* x5634 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x5635 = 0;
int32_t x5636 = 0;
int32_t x5637 = 0;
for(int x5638=0; x5638 < 64; x5638++) {
int32_t x5639 = x5636;
int32_t x5640 = x5637;
int32_t x5641 = x5635;
int32_t x5642 = x5641;
int32_t x5643 = x5639;
int32_t x5644 = x5640;
for(int x5645=0; x5645 < 512; x5645++) {
int32_t x5646 = x5643;
int32_t x5647 = x5644;
int32_t x5648 = x5642;
int32_t x5649 = x5648;
int32_t x5650 = x5646;
int32_t x5651 = x5647;
for(int x5652=0; x5652 < 8; x5652++) {
int32_t x5653 = x5650;
int32_t x5654 = x5651;
int32_t x5655 = x5649;
int32_t x5656 = x5655;
int32_t x5657 = x5653;
int32_t x5658 = x5654;
for(int x5659=0; x5659 < 8; x5659++) {
int32_t x5660 = x5656;
int32_t x5661 = x5657;
float x5662 = x5583[x5661];
int32_t x5663 = x5658;
float x5664 = x51[x5663];
float x5665 = x5662 * x5664;
x5634[x5660] = x5665;
x5656 += 1;
x5657 += 1;

}
x5649 += 8;
x5650 += 8;

}
x5642 += 64;
x5643 += 64;
x5644 += 1;

}
x5635 += 32768;
x5636 += 32768;

}
// resize to WrappedArray(-1, 1, 1)
float* x5685 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x5686 = 0;
int32_t x5687 = 0;
int32_t x5688 = 0;
for(int x5689=0; x5689 < 64; x5689++) {
int32_t x5690 = x5687;
int32_t x5691 = x5688;
int32_t x5692 = x5686;
int32_t x5693 = x5692;
int32_t x5694 = x5690;
int32_t x5695 = x5691;
for(int x5696=0; x5696 < 512; x5696++) {
int32_t x5697 = x5694;
int32_t x5698 = x5695;
int32_t x5699 = x5693;
int32_t x5700 = x5699;
int32_t x5701 = x5697;
int32_t x5702 = x5698;
for(int x5703=0; x5703 < 8; x5703++) {
int32_t x5704 = x5701;
int32_t x5705 = x5702;
int32_t x5706 = x5700;
int32_t x5707 = x5706;
int32_t x5708 = x5704;
int32_t x5709 = x5705;
for(int x5710=0; x5710 < 8; x5710++) {
int32_t x5711 = x5707;
int32_t x5712 = x5708;
float x5713 = x5634[x5712];
int32_t x5714 = x5709;
float x5715 = x245[x5714];
float x5716 = x5713 + x5715;
x5685[x5711] = x5716;
x5707 += 1;
x5708 += 1;

}
x5700 += 8;
x5701 += 8;

}
x5693 += 64;
x5694 += 64;
x5695 += 1;

}
x5686 += 32768;
x5687 += 32768;

}
int32_t x5735 = 0;
int32_t x5736 = 0;
int32_t x5737 = 0;
for(int x5738=0; x5738 < 64; x5738++) {
int32_t x5739 = x5736;
int32_t x5740 = x5737;
int32_t x5741 = x5735;
int32_t x5742 = x5741;
int32_t x5743 = x5739;
int32_t x5744 = x5740;
for(int x5745=0; x5745 < 512; x5745++) {
int32_t x5746 = x5743;
int32_t x5747 = x5744;
int32_t x5748 = x5742;
int32_t x5749 = x5748;
int32_t x5750 = x5746;
int32_t x5751 = x5747;
for(int x5752=0; x5752 < 8; x5752++) {
int32_t x5753 = x5750;
int32_t x5754 = x5751;
int32_t x5755 = x5749;
int32_t x5756 = x5755;
int32_t x5757 = x5753;
int32_t x5758 = x5754;
for(int x5759=0; x5759 < 8; x5759++) {
int32_t x5760 = x5757;
float x5761 = x5685[x5760];
int32_t x5762 = x5758;
float x5763 = x4879[x5762];
float x5764 = x5761 + x5763;
x5685[x5760] = x5764;
x5756 += 1;
x5757 += 1;
x5758 += 1;

}
x5749 += 8;
x5750 += 8;
x5751 += 8;

}
x5742 += 64;
x5743 += 64;
x5744 += 64;

}
x5735 += 32768;
x5736 += 32768;
x5737 += 32768;

}
float* x5786 = (float*)myMalloc(2097152 * sizeof(float));;
for(int x5787=0; x5787 < 2097152; x5787++) {
float x5788 = x5685[x5787];
bool x5789 = x5788 < 0.0f;
if (x5789) {
x5786[x5787] = 0.0f;
} else {
float x5792 = x5685[x5787];
x5786[x5787] = x5792;
}

}
float* x5798 = (float*)myMalloc(524288 * sizeof(float));;
float* x5799 = (float*)myMalloc(2097152 * sizeof(float));;
for(int x5800=0; x5800 < 64; x5800++) {
int32_t x5801 = x5800 * 32768;
float* x5802 = x5786+x5801;
int32_t x5803 = x5800 * 8192;
float* x5804 = x5798+x5803;
float* x5805 = x5799+x5801;
for(int x5806=0; x5806 < 512; x5806++) {
int32_t x5807 = x5806 / 1;
int32_t x5811 = x5807 * 8;
int32_t x5812 = x5811 * 8;
int32_t x5808 = x5806 % 1;
int32_t x5809 = x5808 / 1;
int32_t x5813 = x5809 * 8;
int32_t x5814 = x5813 * 8;
int32_t x5815 = x5812 + x5814;
int32_t x5810 = x5808 % 1;
int32_t x5816 = x5810 * 8;
int32_t x5817 = x5816 * 8;
int32_t x5818 = x5815 + x5817;
float* x5819 = x5805+x5818;
float* x5820 = x5802+x5812;
for(int x5821=0; x5821 < 8; x5821++) {
int32_t x5823 = x5821 * 8;
float* x5824 = x5819+x5823;
int32_t x5822 = x5821 + x5809;
int32_t x5825 = x5822 * 8;
int32_t x5826 = x5825 + x5810;
float* x5827 = x5820+x5826;
memcpy(x5824, x5827, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,64,512,1,x195,512,x5805,64,1,x5804,64);

}
// resize to WrappedArray(-1, 1, 1)
float* x5837 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x5838 = 0;
int32_t x5839 = 0;
int32_t x5840 = 0;
for(int x5841=0; x5841 < 64; x5841++) {
int32_t x5842 = x5839;
int32_t x5843 = x5840;
int32_t x5844 = x5838;
int32_t x5845 = x5844;
int32_t x5846 = x5842;
int32_t x5847 = x5843;
for(int x5848=0; x5848 < 128; x5848++) {
int32_t x5849 = x5846;
int32_t x5850 = x5847;
int32_t x5851 = x5845;
int32_t x5852 = x5851;
int32_t x5853 = x5849;
int32_t x5854 = x5850;
for(int x5855=0; x5855 < 8; x5855++) {
int32_t x5856 = x5853;
int32_t x5857 = x5854;
int32_t x5858 = x5852;
int32_t x5859 = x5858;
int32_t x5860 = x5856;
int32_t x5861 = x5857;
for(int x5862=0; x5862 < 8; x5862++) {
int32_t x5863 = x5859;
int32_t x5864 = x5860;
float x5865 = x5798[x5864];
int32_t x5866 = x5861;
float x5867 = x111[x5866];
float x5868 = x5865 - x5867;
x5837[x5863] = x5868;
x5859 += 1;
x5860 += 1;

}
x5852 += 8;
x5853 += 8;

}
x5845 += 64;
x5846 += 64;
x5847 += 1;

}
x5838 += 8192;
x5839 += 8192;

}
float* x5887 = (float*)myMalloc(128 * sizeof(float));;
for(int x5888=0; x5888 < 128; x5888++) {
float x5889 = x8[x5888];
float x5890 = x5889 + 1.0E-5f;
x5887[x5888] = x5890;

}
float* x5894 = (float*)myMalloc(128 * sizeof(float));;
for(int x5895=0; x5895 < 128; x5895++) {
float x5896 = x5887[x5895];
double x5897 = (double)x5896;
double x5898 = sqrt(x5897);
float x5899 = (float)x5898;
x5894[x5895] = x5899;

}
// resize to WrappedArray(-1, 1, 1)
float* x5904 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x5905 = 0;
int32_t x5906 = 0;
int32_t x5907 = 0;
for(int x5908=0; x5908 < 64; x5908++) {
int32_t x5909 = x5906;
int32_t x5910 = x5907;
int32_t x5911 = x5905;
int32_t x5912 = x5911;
int32_t x5913 = x5909;
int32_t x5914 = x5910;
for(int x5915=0; x5915 < 128; x5915++) {
int32_t x5916 = x5913;
int32_t x5917 = x5914;
int32_t x5918 = x5912;
int32_t x5919 = x5918;
int32_t x5920 = x5916;
int32_t x5921 = x5917;
for(int x5922=0; x5922 < 8; x5922++) {
int32_t x5923 = x5920;
int32_t x5924 = x5921;
int32_t x5925 = x5919;
int32_t x5926 = x5925;
int32_t x5927 = x5923;
int32_t x5928 = x5924;
for(int x5929=0; x5929 < 8; x5929++) {
int32_t x5930 = x5926;
int32_t x5931 = x5927;
float x5932 = x5837[x5931];
int32_t x5933 = x5928;
float x5934 = x5894[x5933];
float x5935 = x5932 / x5934;
x5904[x5930] = x5935;
x5926 += 1;
x5927 += 1;

}
x5919 += 8;
x5920 += 8;

}
x5912 += 64;
x5913 += 64;
x5914 += 1;

}
x5905 += 8192;
x5906 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x5955 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x5956 = 0;
int32_t x5957 = 0;
int32_t x5958 = 0;
for(int x5959=0; x5959 < 64; x5959++) {
int32_t x5960 = x5957;
int32_t x5961 = x5958;
int32_t x5962 = x5956;
int32_t x5963 = x5962;
int32_t x5964 = x5960;
int32_t x5965 = x5961;
for(int x5966=0; x5966 < 128; x5966++) {
int32_t x5967 = x5964;
int32_t x5968 = x5965;
int32_t x5969 = x5963;
int32_t x5970 = x5969;
int32_t x5971 = x5967;
int32_t x5972 = x5968;
for(int x5973=0; x5973 < 8; x5973++) {
int32_t x5974 = x5971;
int32_t x5975 = x5972;
int32_t x5976 = x5970;
int32_t x5977 = x5976;
int32_t x5978 = x5974;
int32_t x5979 = x5975;
for(int x5980=0; x5980 < 8; x5980++) {
int32_t x5981 = x5977;
int32_t x5982 = x5978;
float x5983 = x5904[x5982];
int32_t x5984 = x5979;
float x5985 = x44[x5984];
float x5986 = x5983 * x5985;
x5955[x5981] = x5986;
x5977 += 1;
x5978 += 1;

}
x5970 += 8;
x5971 += 8;

}
x5963 += 64;
x5964 += 64;
x5965 += 1;

}
x5956 += 8192;
x5957 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x6006 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x6007 = 0;
int32_t x6008 = 0;
int32_t x6009 = 0;
for(int x6010=0; x6010 < 64; x6010++) {
int32_t x6011 = x6008;
int32_t x6012 = x6009;
int32_t x6013 = x6007;
int32_t x6014 = x6013;
int32_t x6015 = x6011;
int32_t x6016 = x6012;
for(int x6017=0; x6017 < 128; x6017++) {
int32_t x6018 = x6015;
int32_t x6019 = x6016;
int32_t x6020 = x6014;
int32_t x6021 = x6020;
int32_t x6022 = x6018;
int32_t x6023 = x6019;
for(int x6024=0; x6024 < 8; x6024++) {
int32_t x6025 = x6022;
int32_t x6026 = x6023;
int32_t x6027 = x6021;
int32_t x6028 = x6027;
int32_t x6029 = x6025;
int32_t x6030 = x6026;
for(int x6031=0; x6031 < 8; x6031++) {
int32_t x6032 = x6028;
int32_t x6033 = x6029;
float x6034 = x5955[x6033];
int32_t x6035 = x6030;
float x6036 = x169[x6035];
float x6037 = x6034 + x6036;
x6006[x6032] = x6037;
x6028 += 1;
x6029 += 1;

}
x6021 += 8;
x6022 += 8;

}
x6014 += 64;
x6015 += 64;
x6016 += 1;

}
x6007 += 8192;
x6008 += 8192;

}
float* x6056 = (float*)myMalloc(524288 * sizeof(float));;
for(int x6057=0; x6057 < 524288; x6057++) {
float x6058 = x6006[x6057];
bool x6059 = x6058 < 0.0f;
if (x6059) {
x6056[x6057] = 0.0f;
} else {
float x6062 = x6006[x6057];
x6056[x6057] = x6062;
}

}
float* x6068 = (float*)myMalloc(524288 * sizeof(float));;
float* x6069 = (float*)myMalloc(4718592 * sizeof(float));;
for(int x6070=0; x6070 < 64; x6070++) {
int32_t x6071 = x6070 * 8192;
float* x6072 = x6056+x6071;
float* x6073 = x6068+x6071;
int32_t x6074 = x6070 * 73728;
float* x6075 = x6069+x6074;
for(int x6076=0; x6076 < 1152; x6076++) {
int32_t x6077 = x6076 / 9;
int32_t x6081 = x6077 * 3;
int32_t x6082 = x6081 * 3;
int32_t x6083 = x6082 * 8;
int32_t x6084 = x6083 * 8;
int32_t x6078 = x6076 % 9;
int32_t x6079 = x6078 / 3;
int32_t x6085 = x6079 * 3;
int32_t x6086 = x6085 * 8;
int32_t x6087 = x6086 * 8;
int32_t x6088 = x6084 + x6087;
int32_t x6080 = x6078 % 3;
int32_t x6089 = x6080 * 8;
int32_t x6090 = x6089 * 8;
int32_t x6091 = x6088 + x6090;
float* x6092 = x6075+x6091;
int32_t x6093 = x6077 * 8;
int32_t x6094 = x6093 * 8;
float* x6095 = x6072+x6094;
int32_t x6107 = 1 - x6080;
bool x6108 = x6107 > 0;
int32_t x6109;
if (x6108) {
x6109 = x6107;
} else {
x6109 = 0;
}
int32_t x6110 = 3 - x6080;
int32_t x6111 = x6110 - 1;
int32_t x6112 = 1 - x6111;
bool x6113 = x6112 > 0;
int32_t x6114;
if (x6113) {
x6114 = x6112;
} else {
x6114 = 0;
}
int32_t x6115 = 8 - x6114;
int32_t x6116 = x6115 - x6109;
bool x6117 = x6116 <= 0;
bool x6121 = x6109 > 0;
int32_t x6106 = -1 + x6080;
bool x6134 = x6114 > 0;
for(int x6096=0; x6096 < 8; x6096++) {
int32_t x6097 = x6096 - 1;
int32_t x6098 = x6097 + x6079;
bool x6099 = x6098 < 0;
bool x6100 = x6098 >= 8;
bool x6101 = x6099 || x6100;
if (x6101) {
int32_t x6102 = x6096 * 8;
float* x6103 = x6092+x6102;
memset(x6103, 0, 4 * 8);;
} else {
if (x6117) {
int32_t x6102 = x6096 * 8;
float* x6118 = x6092+x6102;
memset(x6118, 0, 4 * 8);;
} else {
int32_t x6102 = x6096 * 8;
if (x6121) {
float* x6122 = x6092+x6102;
memset(x6122, 0, 4 * x6109);;
} else {
}
// may have segfault here
int32_t x6127 = x6102 + x6109;
float* x6128 = x6092+x6127;
int32_t x6129 = x6098 * 8;
int32_t x6130 = x6129 + x6106;
int32_t x6131 = x6130 + x6109;
float* x6132 = x6095+x6131;
memcpy(x6128, x6132, 4 * x6116);;
if (x6134) {
int32_t x6135 = x6102 + 8;
int32_t x6136 = x6135 - x6114;
float* x6137 = x6092+x6136;
memset(x6137, 0, 4 * x6114);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,64,1152,1,x190,1152,x6075,64,1,x6073,64);

}
// resize to WrappedArray(-1, 1, 1)
float* x6153 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x6154 = 0;
int32_t x6155 = 0;
int32_t x6156 = 0;
for(int x6157=0; x6157 < 64; x6157++) {
int32_t x6158 = x6155;
int32_t x6159 = x6156;
int32_t x6160 = x6154;
int32_t x6161 = x6160;
int32_t x6162 = x6158;
int32_t x6163 = x6159;
for(int x6164=0; x6164 < 128; x6164++) {
int32_t x6165 = x6162;
int32_t x6166 = x6163;
int32_t x6167 = x6161;
int32_t x6168 = x6167;
int32_t x6169 = x6165;
int32_t x6170 = x6166;
for(int x6171=0; x6171 < 8; x6171++) {
int32_t x6172 = x6169;
int32_t x6173 = x6170;
int32_t x6174 = x6168;
int32_t x6175 = x6174;
int32_t x6176 = x6172;
int32_t x6177 = x6173;
for(int x6178=0; x6178 < 8; x6178++) {
int32_t x6179 = x6175;
int32_t x6180 = x6176;
float x6181 = x6068[x6180];
int32_t x6182 = x6177;
float x6183 = x216[x6182];
float x6184 = x6181 - x6183;
x6153[x6179] = x6184;
x6175 += 1;
x6176 += 1;

}
x6168 += 8;
x6169 += 8;

}
x6161 += 64;
x6162 += 64;
x6163 += 1;

}
x6154 += 8192;
x6155 += 8192;

}
float* x6203 = (float*)myMalloc(128 * sizeof(float));;
for(int x6204=0; x6204 < 128; x6204++) {
float x6205 = x265[x6204];
float x6206 = x6205 + 1.0E-5f;
x6203[x6204] = x6206;

}
float* x6210 = (float*)myMalloc(128 * sizeof(float));;
for(int x6211=0; x6211 < 128; x6211++) {
float x6212 = x6203[x6211];
double x6213 = (double)x6212;
double x6214 = sqrt(x6213);
float x6215 = (float)x6214;
x6210[x6211] = x6215;

}
// resize to WrappedArray(-1, 1, 1)
float* x6220 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x6221 = 0;
int32_t x6222 = 0;
int32_t x6223 = 0;
for(int x6224=0; x6224 < 64; x6224++) {
int32_t x6225 = x6222;
int32_t x6226 = x6223;
int32_t x6227 = x6221;
int32_t x6228 = x6227;
int32_t x6229 = x6225;
int32_t x6230 = x6226;
for(int x6231=0; x6231 < 128; x6231++) {
int32_t x6232 = x6229;
int32_t x6233 = x6230;
int32_t x6234 = x6228;
int32_t x6235 = x6234;
int32_t x6236 = x6232;
int32_t x6237 = x6233;
for(int x6238=0; x6238 < 8; x6238++) {
int32_t x6239 = x6236;
int32_t x6240 = x6237;
int32_t x6241 = x6235;
int32_t x6242 = x6241;
int32_t x6243 = x6239;
int32_t x6244 = x6240;
for(int x6245=0; x6245 < 8; x6245++) {
int32_t x6246 = x6242;
int32_t x6247 = x6243;
float x6248 = x6153[x6247];
int32_t x6249 = x6244;
float x6250 = x6210[x6249];
float x6251 = x6248 / x6250;
x6220[x6246] = x6251;
x6242 += 1;
x6243 += 1;

}
x6235 += 8;
x6236 += 8;

}
x6228 += 64;
x6229 += 64;
x6230 += 1;

}
x6221 += 8192;
x6222 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x6271 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x6272 = 0;
int32_t x6273 = 0;
int32_t x6274 = 0;
for(int x6275=0; x6275 < 64; x6275++) {
int32_t x6276 = x6273;
int32_t x6277 = x6274;
int32_t x6278 = x6272;
int32_t x6279 = x6278;
int32_t x6280 = x6276;
int32_t x6281 = x6277;
for(int x6282=0; x6282 < 128; x6282++) {
int32_t x6283 = x6280;
int32_t x6284 = x6281;
int32_t x6285 = x6279;
int32_t x6286 = x6285;
int32_t x6287 = x6283;
int32_t x6288 = x6284;
for(int x6289=0; x6289 < 8; x6289++) {
int32_t x6290 = x6287;
int32_t x6291 = x6288;
int32_t x6292 = x6286;
int32_t x6293 = x6292;
int32_t x6294 = x6290;
int32_t x6295 = x6291;
for(int x6296=0; x6296 < 8; x6296++) {
int32_t x6297 = x6293;
int32_t x6298 = x6294;
float x6299 = x6220[x6298];
int32_t x6300 = x6295;
float x6301 = x126[x6300];
float x6302 = x6299 * x6301;
x6271[x6297] = x6302;
x6293 += 1;
x6294 += 1;

}
x6286 += 8;
x6287 += 8;

}
x6279 += 64;
x6280 += 64;
x6281 += 1;

}
x6272 += 8192;
x6273 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x6322 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x6323 = 0;
int32_t x6324 = 0;
int32_t x6325 = 0;
for(int x6326=0; x6326 < 64; x6326++) {
int32_t x6327 = x6324;
int32_t x6328 = x6325;
int32_t x6329 = x6323;
int32_t x6330 = x6329;
int32_t x6331 = x6327;
int32_t x6332 = x6328;
for(int x6333=0; x6333 < 128; x6333++) {
int32_t x6334 = x6331;
int32_t x6335 = x6332;
int32_t x6336 = x6330;
int32_t x6337 = x6336;
int32_t x6338 = x6334;
int32_t x6339 = x6335;
for(int x6340=0; x6340 < 8; x6340++) {
int32_t x6341 = x6338;
int32_t x6342 = x6339;
int32_t x6343 = x6337;
int32_t x6344 = x6343;
int32_t x6345 = x6341;
int32_t x6346 = x6342;
for(int x6347=0; x6347 < 8; x6347++) {
int32_t x6348 = x6344;
int32_t x6349 = x6345;
float x6350 = x6271[x6349];
int32_t x6351 = x6346;
float x6352 = x60[x6351];
float x6353 = x6350 + x6352;
x6322[x6348] = x6353;
x6344 += 1;
x6345 += 1;

}
x6337 += 8;
x6338 += 8;

}
x6330 += 64;
x6331 += 64;
x6332 += 1;

}
x6323 += 8192;
x6324 += 8192;

}
float* x6372 = (float*)myMalloc(524288 * sizeof(float));;
for(int x6373=0; x6373 < 524288; x6373++) {
float x6374 = x6322[x6373];
bool x6375 = x6374 < 0.0f;
if (x6375) {
x6372[x6373] = 0.0f;
} else {
float x6378 = x6322[x6373];
x6372[x6373] = x6378;
}

}
float* x6384 = (float*)myMalloc(2097152 * sizeof(float));;
float* x6385 = (float*)myMalloc(524288 * sizeof(float));;
for(int x6386=0; x6386 < 64; x6386++) {
int32_t x6387 = x6386 * 8192;
float* x6388 = x6372+x6387;
int32_t x6389 = x6386 * 32768;
float* x6390 = x6384+x6389;
float* x6391 = x6385+x6387;
for(int x6392=0; x6392 < 128; x6392++) {
int32_t x6393 = x6392 / 1;
int32_t x6397 = x6393 * 8;
int32_t x6398 = x6397 * 8;
int32_t x6394 = x6392 % 1;
int32_t x6395 = x6394 / 1;
int32_t x6399 = x6395 * 8;
int32_t x6400 = x6399 * 8;
int32_t x6401 = x6398 + x6400;
int32_t x6396 = x6394 % 1;
int32_t x6402 = x6396 * 8;
int32_t x6403 = x6402 * 8;
int32_t x6404 = x6401 + x6403;
float* x6405 = x6391+x6404;
float* x6406 = x6388+x6398;
for(int x6407=0; x6407 < 8; x6407++) {
int32_t x6409 = x6407 * 8;
float* x6410 = x6405+x6409;
int32_t x6408 = x6407 + x6395;
int32_t x6411 = x6408 * 8;
int32_t x6412 = x6411 + x6396;
float* x6413 = x6406+x6412;
memcpy(x6410, x6413, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,64,128,1,x40,128,x6391,64,1,x6390,64);

}
// resize to WrappedArray(-1, 1, 1)
float* x6423 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x6424 = 0;
int32_t x6425 = 0;
int32_t x6426 = 0;
for(int x6427=0; x6427 < 64; x6427++) {
int32_t x6428 = x6425;
int32_t x6429 = x6426;
int32_t x6430 = x6424;
int32_t x6431 = x6430;
int32_t x6432 = x6428;
int32_t x6433 = x6429;
for(int x6434=0; x6434 < 512; x6434++) {
int32_t x6435 = x6432;
int32_t x6436 = x6433;
int32_t x6437 = x6431;
int32_t x6438 = x6437;
int32_t x6439 = x6435;
int32_t x6440 = x6436;
for(int x6441=0; x6441 < 8; x6441++) {
int32_t x6442 = x6439;
int32_t x6443 = x6440;
int32_t x6444 = x6438;
int32_t x6445 = x6444;
int32_t x6446 = x6442;
int32_t x6447 = x6443;
for(int x6448=0; x6448 < 8; x6448++) {
int32_t x6449 = x6445;
int32_t x6450 = x6446;
float x6451 = x6384[x6450];
int32_t x6452 = x6447;
float x6453 = x24[x6452];
float x6454 = x6451 - x6453;
x6423[x6449] = x6454;
x6445 += 1;
x6446 += 1;

}
x6438 += 8;
x6439 += 8;

}
x6431 += 64;
x6432 += 64;
x6433 += 1;

}
x6424 += 32768;
x6425 += 32768;

}
float* x6473 = (float*)myMalloc(512 * sizeof(float));;
for(int x6474=0; x6474 < 512; x6474++) {
float x6475 = x222[x6474];
float x6476 = x6475 + 1.0E-5f;
x6473[x6474] = x6476;

}
float* x6480 = (float*)myMalloc(512 * sizeof(float));;
for(int x6481=0; x6481 < 512; x6481++) {
float x6482 = x6473[x6481];
double x6483 = (double)x6482;
double x6484 = sqrt(x6483);
float x6485 = (float)x6484;
x6480[x6481] = x6485;

}
// resize to WrappedArray(-1, 1, 1)
float* x6490 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x6491 = 0;
int32_t x6492 = 0;
int32_t x6493 = 0;
for(int x6494=0; x6494 < 64; x6494++) {
int32_t x6495 = x6492;
int32_t x6496 = x6493;
int32_t x6497 = x6491;
int32_t x6498 = x6497;
int32_t x6499 = x6495;
int32_t x6500 = x6496;
for(int x6501=0; x6501 < 512; x6501++) {
int32_t x6502 = x6499;
int32_t x6503 = x6500;
int32_t x6504 = x6498;
int32_t x6505 = x6504;
int32_t x6506 = x6502;
int32_t x6507 = x6503;
for(int x6508=0; x6508 < 8; x6508++) {
int32_t x6509 = x6506;
int32_t x6510 = x6507;
int32_t x6511 = x6505;
int32_t x6512 = x6511;
int32_t x6513 = x6509;
int32_t x6514 = x6510;
for(int x6515=0; x6515 < 8; x6515++) {
int32_t x6516 = x6512;
int32_t x6517 = x6513;
float x6518 = x6423[x6517];
int32_t x6519 = x6514;
float x6520 = x6480[x6519];
float x6521 = x6518 / x6520;
x6490[x6516] = x6521;
x6512 += 1;
x6513 += 1;

}
x6505 += 8;
x6506 += 8;

}
x6498 += 64;
x6499 += 64;
x6500 += 1;

}
x6491 += 32768;
x6492 += 32768;

}
// resize to WrappedArray(-1, 1, 1)
float* x6541 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x6542 = 0;
int32_t x6543 = 0;
int32_t x6544 = 0;
for(int x6545=0; x6545 < 64; x6545++) {
int32_t x6546 = x6543;
int32_t x6547 = x6544;
int32_t x6548 = x6542;
int32_t x6549 = x6548;
int32_t x6550 = x6546;
int32_t x6551 = x6547;
for(int x6552=0; x6552 < 512; x6552++) {
int32_t x6553 = x6550;
int32_t x6554 = x6551;
int32_t x6555 = x6549;
int32_t x6556 = x6555;
int32_t x6557 = x6553;
int32_t x6558 = x6554;
for(int x6559=0; x6559 < 8; x6559++) {
int32_t x6560 = x6557;
int32_t x6561 = x6558;
int32_t x6562 = x6556;
int32_t x6563 = x6562;
int32_t x6564 = x6560;
int32_t x6565 = x6561;
for(int x6566=0; x6566 < 8; x6566++) {
int32_t x6567 = x6563;
int32_t x6568 = x6564;
float x6569 = x6490[x6568];
int32_t x6570 = x6565;
float x6571 = x166[x6570];
float x6572 = x6569 * x6571;
x6541[x6567] = x6572;
x6563 += 1;
x6564 += 1;

}
x6556 += 8;
x6557 += 8;

}
x6549 += 64;
x6550 += 64;
x6551 += 1;

}
x6542 += 32768;
x6543 += 32768;

}
// resize to WrappedArray(-1, 1, 1)
float* x6592 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x6593 = 0;
int32_t x6594 = 0;
int32_t x6595 = 0;
for(int x6596=0; x6596 < 64; x6596++) {
int32_t x6597 = x6594;
int32_t x6598 = x6595;
int32_t x6599 = x6593;
int32_t x6600 = x6599;
int32_t x6601 = x6597;
int32_t x6602 = x6598;
for(int x6603=0; x6603 < 512; x6603++) {
int32_t x6604 = x6601;
int32_t x6605 = x6602;
int32_t x6606 = x6600;
int32_t x6607 = x6606;
int32_t x6608 = x6604;
int32_t x6609 = x6605;
for(int x6610=0; x6610 < 8; x6610++) {
int32_t x6611 = x6608;
int32_t x6612 = x6609;
int32_t x6613 = x6607;
int32_t x6614 = x6613;
int32_t x6615 = x6611;
int32_t x6616 = x6612;
for(int x6617=0; x6617 < 8; x6617++) {
int32_t x6618 = x6614;
int32_t x6619 = x6615;
float x6620 = x6541[x6619];
int32_t x6621 = x6616;
float x6622 = x81[x6621];
float x6623 = x6620 + x6622;
x6592[x6618] = x6623;
x6614 += 1;
x6615 += 1;

}
x6607 += 8;
x6608 += 8;

}
x6600 += 64;
x6601 += 64;
x6602 += 1;

}
x6593 += 32768;
x6594 += 32768;

}
int32_t x6642 = 0;
int32_t x6643 = 0;
int32_t x6644 = 0;
for(int x6645=0; x6645 < 64; x6645++) {
int32_t x6646 = x6643;
int32_t x6647 = x6644;
int32_t x6648 = x6642;
int32_t x6649 = x6648;
int32_t x6650 = x6646;
int32_t x6651 = x6647;
for(int x6652=0; x6652 < 512; x6652++) {
int32_t x6653 = x6650;
int32_t x6654 = x6651;
int32_t x6655 = x6649;
int32_t x6656 = x6655;
int32_t x6657 = x6653;
int32_t x6658 = x6654;
for(int x6659=0; x6659 < 8; x6659++) {
int32_t x6660 = x6657;
int32_t x6661 = x6658;
int32_t x6662 = x6656;
int32_t x6663 = x6662;
int32_t x6664 = x6660;
int32_t x6665 = x6661;
for(int x6666=0; x6666 < 8; x6666++) {
int32_t x6667 = x6664;
float x6668 = x6592[x6667];
int32_t x6669 = x6665;
float x6670 = x5786[x6669];
float x6671 = x6668 + x6670;
x6592[x6667] = x6671;
x6663 += 1;
x6664 += 1;
x6665 += 1;

}
x6656 += 8;
x6657 += 8;
x6658 += 8;

}
x6649 += 64;
x6650 += 64;
x6651 += 64;

}
x6642 += 32768;
x6643 += 32768;
x6644 += 32768;

}
float* x6693 = (float*)myMalloc(2097152 * sizeof(float));;
for(int x6694=0; x6694 < 2097152; x6694++) {
float x6695 = x6592[x6694];
bool x6696 = x6695 < 0.0f;
if (x6696) {
x6693[x6694] = 0.0f;
} else {
float x6699 = x6592[x6694];
x6693[x6694] = x6699;
}

}
float* x6705 = (float*)myMalloc(524288 * sizeof(float));;
float* x6706 = (float*)myMalloc(2097152 * sizeof(float));;
for(int x6707=0; x6707 < 64; x6707++) {
int32_t x6708 = x6707 * 32768;
float* x6709 = x6693+x6708;
int32_t x6710 = x6707 * 8192;
float* x6711 = x6705+x6710;
float* x6712 = x6706+x6708;
for(int x6713=0; x6713 < 512; x6713++) {
int32_t x6714 = x6713 / 1;
int32_t x6718 = x6714 * 8;
int32_t x6719 = x6718 * 8;
int32_t x6715 = x6713 % 1;
int32_t x6716 = x6715 / 1;
int32_t x6720 = x6716 * 8;
int32_t x6721 = x6720 * 8;
int32_t x6722 = x6719 + x6721;
int32_t x6717 = x6715 % 1;
int32_t x6723 = x6717 * 8;
int32_t x6724 = x6723 * 8;
int32_t x6725 = x6722 + x6724;
float* x6726 = x6712+x6725;
float* x6727 = x6709+x6719;
for(int x6728=0; x6728 < 8; x6728++) {
int32_t x6730 = x6728 * 8;
float* x6731 = x6726+x6730;
int32_t x6729 = x6728 + x6716;
int32_t x6732 = x6729 * 8;
int32_t x6733 = x6732 + x6717;
float* x6734 = x6727+x6733;
memcpy(x6731, x6734, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,64,512,1,x131,512,x6712,64,1,x6711,64);

}
// resize to WrappedArray(-1, 1, 1)
float* x6744 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x6745 = 0;
int32_t x6746 = 0;
int32_t x6747 = 0;
for(int x6748=0; x6748 < 64; x6748++) {
int32_t x6749 = x6746;
int32_t x6750 = x6747;
int32_t x6751 = x6745;
int32_t x6752 = x6751;
int32_t x6753 = x6749;
int32_t x6754 = x6750;
for(int x6755=0; x6755 < 128; x6755++) {
int32_t x6756 = x6753;
int32_t x6757 = x6754;
int32_t x6758 = x6752;
int32_t x6759 = x6758;
int32_t x6760 = x6756;
int32_t x6761 = x6757;
for(int x6762=0; x6762 < 8; x6762++) {
int32_t x6763 = x6760;
int32_t x6764 = x6761;
int32_t x6765 = x6759;
int32_t x6766 = x6765;
int32_t x6767 = x6763;
int32_t x6768 = x6764;
for(int x6769=0; x6769 < 8; x6769++) {
int32_t x6770 = x6766;
int32_t x6771 = x6767;
float x6772 = x6705[x6771];
int32_t x6773 = x6768;
float x6774 = x235[x6773];
float x6775 = x6772 - x6774;
x6744[x6770] = x6775;
x6766 += 1;
x6767 += 1;

}
x6759 += 8;
x6760 += 8;

}
x6752 += 64;
x6753 += 64;
x6754 += 1;

}
x6745 += 8192;
x6746 += 8192;

}
float* x6794 = (float*)myMalloc(128 * sizeof(float));;
for(int x6795=0; x6795 < 128; x6795++) {
float x6796 = x260[x6795];
float x6797 = x6796 + 1.0E-5f;
x6794[x6795] = x6797;

}
float* x6801 = (float*)myMalloc(128 * sizeof(float));;
for(int x6802=0; x6802 < 128; x6802++) {
float x6803 = x6794[x6802];
double x6804 = (double)x6803;
double x6805 = sqrt(x6804);
float x6806 = (float)x6805;
x6801[x6802] = x6806;

}
// resize to WrappedArray(-1, 1, 1)
float* x6811 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x6812 = 0;
int32_t x6813 = 0;
int32_t x6814 = 0;
for(int x6815=0; x6815 < 64; x6815++) {
int32_t x6816 = x6813;
int32_t x6817 = x6814;
int32_t x6818 = x6812;
int32_t x6819 = x6818;
int32_t x6820 = x6816;
int32_t x6821 = x6817;
for(int x6822=0; x6822 < 128; x6822++) {
int32_t x6823 = x6820;
int32_t x6824 = x6821;
int32_t x6825 = x6819;
int32_t x6826 = x6825;
int32_t x6827 = x6823;
int32_t x6828 = x6824;
for(int x6829=0; x6829 < 8; x6829++) {
int32_t x6830 = x6827;
int32_t x6831 = x6828;
int32_t x6832 = x6826;
int32_t x6833 = x6832;
int32_t x6834 = x6830;
int32_t x6835 = x6831;
for(int x6836=0; x6836 < 8; x6836++) {
int32_t x6837 = x6833;
int32_t x6838 = x6834;
float x6839 = x6744[x6838];
int32_t x6840 = x6835;
float x6841 = x6801[x6840];
float x6842 = x6839 / x6841;
x6811[x6837] = x6842;
x6833 += 1;
x6834 += 1;

}
x6826 += 8;
x6827 += 8;

}
x6819 += 64;
x6820 += 64;
x6821 += 1;

}
x6812 += 8192;
x6813 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x6862 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x6863 = 0;
int32_t x6864 = 0;
int32_t x6865 = 0;
for(int x6866=0; x6866 < 64; x6866++) {
int32_t x6867 = x6864;
int32_t x6868 = x6865;
int32_t x6869 = x6863;
int32_t x6870 = x6869;
int32_t x6871 = x6867;
int32_t x6872 = x6868;
for(int x6873=0; x6873 < 128; x6873++) {
int32_t x6874 = x6871;
int32_t x6875 = x6872;
int32_t x6876 = x6870;
int32_t x6877 = x6876;
int32_t x6878 = x6874;
int32_t x6879 = x6875;
for(int x6880=0; x6880 < 8; x6880++) {
int32_t x6881 = x6878;
int32_t x6882 = x6879;
int32_t x6883 = x6877;
int32_t x6884 = x6883;
int32_t x6885 = x6881;
int32_t x6886 = x6882;
for(int x6887=0; x6887 < 8; x6887++) {
int32_t x6888 = x6884;
int32_t x6889 = x6885;
float x6890 = x6811[x6889];
int32_t x6891 = x6886;
float x6892 = x38[x6891];
float x6893 = x6890 * x6892;
x6862[x6888] = x6893;
x6884 += 1;
x6885 += 1;

}
x6877 += 8;
x6878 += 8;

}
x6870 += 64;
x6871 += 64;
x6872 += 1;

}
x6863 += 8192;
x6864 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x6913 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x6914 = 0;
int32_t x6915 = 0;
int32_t x6916 = 0;
for(int x6917=0; x6917 < 64; x6917++) {
int32_t x6918 = x6915;
int32_t x6919 = x6916;
int32_t x6920 = x6914;
int32_t x6921 = x6920;
int32_t x6922 = x6918;
int32_t x6923 = x6919;
for(int x6924=0; x6924 < 128; x6924++) {
int32_t x6925 = x6922;
int32_t x6926 = x6923;
int32_t x6927 = x6921;
int32_t x6928 = x6927;
int32_t x6929 = x6925;
int32_t x6930 = x6926;
for(int x6931=0; x6931 < 8; x6931++) {
int32_t x6932 = x6929;
int32_t x6933 = x6930;
int32_t x6934 = x6928;
int32_t x6935 = x6934;
int32_t x6936 = x6932;
int32_t x6937 = x6933;
for(int x6938=0; x6938 < 8; x6938++) {
int32_t x6939 = x6935;
int32_t x6940 = x6936;
float x6941 = x6862[x6940];
int32_t x6942 = x6937;
float x6943 = x241[x6942];
float x6944 = x6941 + x6943;
x6913[x6939] = x6944;
x6935 += 1;
x6936 += 1;

}
x6928 += 8;
x6929 += 8;

}
x6921 += 64;
x6922 += 64;
x6923 += 1;

}
x6914 += 8192;
x6915 += 8192;

}
float* x6963 = (float*)myMalloc(524288 * sizeof(float));;
for(int x6964=0; x6964 < 524288; x6964++) {
float x6965 = x6913[x6964];
bool x6966 = x6965 < 0.0f;
if (x6966) {
x6963[x6964] = 0.0f;
} else {
float x6969 = x6913[x6964];
x6963[x6964] = x6969;
}

}
float* x6975 = (float*)myMalloc(524288 * sizeof(float));;
float* x6976 = (float*)myMalloc(4718592 * sizeof(float));;
for(int x6977=0; x6977 < 64; x6977++) {
int32_t x6978 = x6977 * 8192;
float* x6979 = x6963+x6978;
float* x6980 = x6975+x6978;
int32_t x6981 = x6977 * 73728;
float* x6982 = x6976+x6981;
for(int x6983=0; x6983 < 1152; x6983++) {
int32_t x6984 = x6983 / 9;
int32_t x6988 = x6984 * 3;
int32_t x6989 = x6988 * 3;
int32_t x6990 = x6989 * 8;
int32_t x6991 = x6990 * 8;
int32_t x6985 = x6983 % 9;
int32_t x6986 = x6985 / 3;
int32_t x6992 = x6986 * 3;
int32_t x6993 = x6992 * 8;
int32_t x6994 = x6993 * 8;
int32_t x6995 = x6991 + x6994;
int32_t x6987 = x6985 % 3;
int32_t x6996 = x6987 * 8;
int32_t x6997 = x6996 * 8;
int32_t x6998 = x6995 + x6997;
float* x6999 = x6982+x6998;
int32_t x7000 = x6984 * 8;
int32_t x7001 = x7000 * 8;
float* x7002 = x6979+x7001;
int32_t x7014 = 1 - x6987;
bool x7015 = x7014 > 0;
int32_t x7016;
if (x7015) {
x7016 = x7014;
} else {
x7016 = 0;
}
int32_t x7017 = 3 - x6987;
int32_t x7018 = x7017 - 1;
int32_t x7019 = 1 - x7018;
bool x7020 = x7019 > 0;
int32_t x7021;
if (x7020) {
x7021 = x7019;
} else {
x7021 = 0;
}
int32_t x7022 = 8 - x7021;
int32_t x7023 = x7022 - x7016;
bool x7024 = x7023 <= 0;
bool x7028 = x7016 > 0;
int32_t x7013 = -1 + x6987;
bool x7041 = x7021 > 0;
for(int x7003=0; x7003 < 8; x7003++) {
int32_t x7004 = x7003 - 1;
int32_t x7005 = x7004 + x6986;
bool x7006 = x7005 < 0;
bool x7007 = x7005 >= 8;
bool x7008 = x7006 || x7007;
if (x7008) {
int32_t x7009 = x7003 * 8;
float* x7010 = x6999+x7009;
memset(x7010, 0, 4 * 8);;
} else {
if (x7024) {
int32_t x7009 = x7003 * 8;
float* x7025 = x6999+x7009;
memset(x7025, 0, 4 * 8);;
} else {
int32_t x7009 = x7003 * 8;
if (x7028) {
float* x7029 = x6999+x7009;
memset(x7029, 0, 4 * x7016);;
} else {
}
// may have segfault here
int32_t x7034 = x7009 + x7016;
float* x7035 = x6999+x7034;
int32_t x7036 = x7005 * 8;
int32_t x7037 = x7036 + x7013;
int32_t x7038 = x7037 + x7016;
float* x7039 = x7002+x7038;
memcpy(x7035, x7039, 4 * x7023);;
if (x7041) {
int32_t x7042 = x7009 + 8;
int32_t x7043 = x7042 - x7021;
float* x7044 = x6999+x7043;
memset(x7044, 0, 4 * x7021);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,64,1152,1,x164,1152,x6982,64,1,x6980,64);

}
// resize to WrappedArray(-1, 1, 1)
float* x7060 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x7061 = 0;
int32_t x7062 = 0;
int32_t x7063 = 0;
for(int x7064=0; x7064 < 64; x7064++) {
int32_t x7065 = x7062;
int32_t x7066 = x7063;
int32_t x7067 = x7061;
int32_t x7068 = x7067;
int32_t x7069 = x7065;
int32_t x7070 = x7066;
for(int x7071=0; x7071 < 128; x7071++) {
int32_t x7072 = x7069;
int32_t x7073 = x7070;
int32_t x7074 = x7068;
int32_t x7075 = x7074;
int32_t x7076 = x7072;
int32_t x7077 = x7073;
for(int x7078=0; x7078 < 8; x7078++) {
int32_t x7079 = x7076;
int32_t x7080 = x7077;
int32_t x7081 = x7075;
int32_t x7082 = x7081;
int32_t x7083 = x7079;
int32_t x7084 = x7080;
for(int x7085=0; x7085 < 8; x7085++) {
int32_t x7086 = x7082;
int32_t x7087 = x7083;
float x7088 = x6975[x7087];
int32_t x7089 = x7084;
float x7090 = x267[x7089];
float x7091 = x7088 - x7090;
x7060[x7086] = x7091;
x7082 += 1;
x7083 += 1;

}
x7075 += 8;
x7076 += 8;

}
x7068 += 64;
x7069 += 64;
x7070 += 1;

}
x7061 += 8192;
x7062 += 8192;

}
float* x7110 = (float*)myMalloc(128 * sizeof(float));;
for(int x7111=0; x7111 < 128; x7111++) {
float x7112 = x147[x7111];
float x7113 = x7112 + 1.0E-5f;
x7110[x7111] = x7113;

}
float* x7117 = (float*)myMalloc(128 * sizeof(float));;
for(int x7118=0; x7118 < 128; x7118++) {
float x7119 = x7110[x7118];
double x7120 = (double)x7119;
double x7121 = sqrt(x7120);
float x7122 = (float)x7121;
x7117[x7118] = x7122;

}
// resize to WrappedArray(-1, 1, 1)
float* x7127 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x7128 = 0;
int32_t x7129 = 0;
int32_t x7130 = 0;
for(int x7131=0; x7131 < 64; x7131++) {
int32_t x7132 = x7129;
int32_t x7133 = x7130;
int32_t x7134 = x7128;
int32_t x7135 = x7134;
int32_t x7136 = x7132;
int32_t x7137 = x7133;
for(int x7138=0; x7138 < 128; x7138++) {
int32_t x7139 = x7136;
int32_t x7140 = x7137;
int32_t x7141 = x7135;
int32_t x7142 = x7141;
int32_t x7143 = x7139;
int32_t x7144 = x7140;
for(int x7145=0; x7145 < 8; x7145++) {
int32_t x7146 = x7143;
int32_t x7147 = x7144;
int32_t x7148 = x7142;
int32_t x7149 = x7148;
int32_t x7150 = x7146;
int32_t x7151 = x7147;
for(int x7152=0; x7152 < 8; x7152++) {
int32_t x7153 = x7149;
int32_t x7154 = x7150;
float x7155 = x7060[x7154];
int32_t x7156 = x7151;
float x7157 = x7117[x7156];
float x7158 = x7155 / x7157;
x7127[x7153] = x7158;
x7149 += 1;
x7150 += 1;

}
x7142 += 8;
x7143 += 8;

}
x7135 += 64;
x7136 += 64;
x7137 += 1;

}
x7128 += 8192;
x7129 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x7178 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x7179 = 0;
int32_t x7180 = 0;
int32_t x7181 = 0;
for(int x7182=0; x7182 < 64; x7182++) {
int32_t x7183 = x7180;
int32_t x7184 = x7181;
int32_t x7185 = x7179;
int32_t x7186 = x7185;
int32_t x7187 = x7183;
int32_t x7188 = x7184;
for(int x7189=0; x7189 < 128; x7189++) {
int32_t x7190 = x7187;
int32_t x7191 = x7188;
int32_t x7192 = x7186;
int32_t x7193 = x7192;
int32_t x7194 = x7190;
int32_t x7195 = x7191;
for(int x7196=0; x7196 < 8; x7196++) {
int32_t x7197 = x7194;
int32_t x7198 = x7195;
int32_t x7199 = x7193;
int32_t x7200 = x7199;
int32_t x7201 = x7197;
int32_t x7202 = x7198;
for(int x7203=0; x7203 < 8; x7203++) {
int32_t x7204 = x7200;
int32_t x7205 = x7201;
float x7206 = x7127[x7205];
int32_t x7207 = x7202;
float x7208 = x78[x7207];
float x7209 = x7206 * x7208;
x7178[x7204] = x7209;
x7200 += 1;
x7201 += 1;

}
x7193 += 8;
x7194 += 8;

}
x7186 += 64;
x7187 += 64;
x7188 += 1;

}
x7179 += 8192;
x7180 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x7229 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x7230 = 0;
int32_t x7231 = 0;
int32_t x7232 = 0;
for(int x7233=0; x7233 < 64; x7233++) {
int32_t x7234 = x7231;
int32_t x7235 = x7232;
int32_t x7236 = x7230;
int32_t x7237 = x7236;
int32_t x7238 = x7234;
int32_t x7239 = x7235;
for(int x7240=0; x7240 < 128; x7240++) {
int32_t x7241 = x7238;
int32_t x7242 = x7239;
int32_t x7243 = x7237;
int32_t x7244 = x7243;
int32_t x7245 = x7241;
int32_t x7246 = x7242;
for(int x7247=0; x7247 < 8; x7247++) {
int32_t x7248 = x7245;
int32_t x7249 = x7246;
int32_t x7250 = x7244;
int32_t x7251 = x7250;
int32_t x7252 = x7248;
int32_t x7253 = x7249;
for(int x7254=0; x7254 < 8; x7254++) {
int32_t x7255 = x7251;
int32_t x7256 = x7252;
float x7257 = x7178[x7256];
int32_t x7258 = x7253;
float x7259 = x37[x7258];
float x7260 = x7257 + x7259;
x7229[x7255] = x7260;
x7251 += 1;
x7252 += 1;

}
x7244 += 8;
x7245 += 8;

}
x7237 += 64;
x7238 += 64;
x7239 += 1;

}
x7230 += 8192;
x7231 += 8192;

}
float* x7279 = (float*)myMalloc(524288 * sizeof(float));;
for(int x7280=0; x7280 < 524288; x7280++) {
float x7281 = x7229[x7280];
bool x7282 = x7281 < 0.0f;
if (x7282) {
x7279[x7280] = 0.0f;
} else {
float x7285 = x7229[x7280];
x7279[x7280] = x7285;
}

}
float* x7291 = (float*)myMalloc(2097152 * sizeof(float));;
float* x7292 = (float*)myMalloc(524288 * sizeof(float));;
for(int x7293=0; x7293 < 64; x7293++) {
int32_t x7294 = x7293 * 8192;
float* x7295 = x7279+x7294;
int32_t x7296 = x7293 * 32768;
float* x7297 = x7291+x7296;
float* x7298 = x7292+x7294;
for(int x7299=0; x7299 < 128; x7299++) {
int32_t x7300 = x7299 / 1;
int32_t x7304 = x7300 * 8;
int32_t x7305 = x7304 * 8;
int32_t x7301 = x7299 % 1;
int32_t x7302 = x7301 / 1;
int32_t x7306 = x7302 * 8;
int32_t x7307 = x7306 * 8;
int32_t x7308 = x7305 + x7307;
int32_t x7303 = x7301 % 1;
int32_t x7309 = x7303 * 8;
int32_t x7310 = x7309 * 8;
int32_t x7311 = x7308 + x7310;
float* x7312 = x7298+x7311;
float* x7313 = x7295+x7305;
for(int x7314=0; x7314 < 8; x7314++) {
int32_t x7316 = x7314 * 8;
float* x7317 = x7312+x7316;
int32_t x7315 = x7314 + x7302;
int32_t x7318 = x7315 * 8;
int32_t x7319 = x7318 + x7303;
float* x7320 = x7313+x7319;
memcpy(x7317, x7320, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,64,128,1,x54,128,x7298,64,1,x7297,64);

}
// resize to WrappedArray(-1, 1, 1)
float* x7330 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x7331 = 0;
int32_t x7332 = 0;
int32_t x7333 = 0;
for(int x7334=0; x7334 < 64; x7334++) {
int32_t x7335 = x7332;
int32_t x7336 = x7333;
int32_t x7337 = x7331;
int32_t x7338 = x7337;
int32_t x7339 = x7335;
int32_t x7340 = x7336;
for(int x7341=0; x7341 < 512; x7341++) {
int32_t x7342 = x7339;
int32_t x7343 = x7340;
int32_t x7344 = x7338;
int32_t x7345 = x7344;
int32_t x7346 = x7342;
int32_t x7347 = x7343;
for(int x7348=0; x7348 < 8; x7348++) {
int32_t x7349 = x7346;
int32_t x7350 = x7347;
int32_t x7351 = x7345;
int32_t x7352 = x7351;
int32_t x7353 = x7349;
int32_t x7354 = x7350;
for(int x7355=0; x7355 < 8; x7355++) {
int32_t x7356 = x7352;
int32_t x7357 = x7353;
float x7358 = x7291[x7357];
int32_t x7359 = x7354;
float x7360 = x18[x7359];
float x7361 = x7358 - x7360;
x7330[x7356] = x7361;
x7352 += 1;
x7353 += 1;

}
x7345 += 8;
x7346 += 8;

}
x7338 += 64;
x7339 += 64;
x7340 += 1;

}
x7331 += 32768;
x7332 += 32768;

}
float* x7380 = (float*)myMalloc(512 * sizeof(float));;
for(int x7381=0; x7381 < 512; x7381++) {
float x7382 = x233[x7381];
float x7383 = x7382 + 1.0E-5f;
x7380[x7381] = x7383;

}
float* x7387 = (float*)myMalloc(512 * sizeof(float));;
for(int x7388=0; x7388 < 512; x7388++) {
float x7389 = x7380[x7388];
double x7390 = (double)x7389;
double x7391 = sqrt(x7390);
float x7392 = (float)x7391;
x7387[x7388] = x7392;

}
// resize to WrappedArray(-1, 1, 1)
float* x7397 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x7398 = 0;
int32_t x7399 = 0;
int32_t x7400 = 0;
for(int x7401=0; x7401 < 64; x7401++) {
int32_t x7402 = x7399;
int32_t x7403 = x7400;
int32_t x7404 = x7398;
int32_t x7405 = x7404;
int32_t x7406 = x7402;
int32_t x7407 = x7403;
for(int x7408=0; x7408 < 512; x7408++) {
int32_t x7409 = x7406;
int32_t x7410 = x7407;
int32_t x7411 = x7405;
int32_t x7412 = x7411;
int32_t x7413 = x7409;
int32_t x7414 = x7410;
for(int x7415=0; x7415 < 8; x7415++) {
int32_t x7416 = x7413;
int32_t x7417 = x7414;
int32_t x7418 = x7412;
int32_t x7419 = x7418;
int32_t x7420 = x7416;
int32_t x7421 = x7417;
for(int x7422=0; x7422 < 8; x7422++) {
int32_t x7423 = x7419;
int32_t x7424 = x7420;
float x7425 = x7330[x7424];
int32_t x7426 = x7421;
float x7427 = x7387[x7426];
float x7428 = x7425 / x7427;
x7397[x7423] = x7428;
x7419 += 1;
x7420 += 1;

}
x7412 += 8;
x7413 += 8;

}
x7405 += 64;
x7406 += 64;
x7407 += 1;

}
x7398 += 32768;
x7399 += 32768;

}
// resize to WrappedArray(-1, 1, 1)
float* x7448 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x7449 = 0;
int32_t x7450 = 0;
int32_t x7451 = 0;
for(int x7452=0; x7452 < 64; x7452++) {
int32_t x7453 = x7450;
int32_t x7454 = x7451;
int32_t x7455 = x7449;
int32_t x7456 = x7455;
int32_t x7457 = x7453;
int32_t x7458 = x7454;
for(int x7459=0; x7459 < 512; x7459++) {
int32_t x7460 = x7457;
int32_t x7461 = x7458;
int32_t x7462 = x7456;
int32_t x7463 = x7462;
int32_t x7464 = x7460;
int32_t x7465 = x7461;
for(int x7466=0; x7466 < 8; x7466++) {
int32_t x7467 = x7464;
int32_t x7468 = x7465;
int32_t x7469 = x7463;
int32_t x7470 = x7469;
int32_t x7471 = x7467;
int32_t x7472 = x7468;
for(int x7473=0; x7473 < 8; x7473++) {
int32_t x7474 = x7470;
int32_t x7475 = x7471;
float x7476 = x7397[x7475];
int32_t x7477 = x7472;
float x7478 = x155[x7477];
float x7479 = x7476 * x7478;
x7448[x7474] = x7479;
x7470 += 1;
x7471 += 1;

}
x7463 += 8;
x7464 += 8;

}
x7456 += 64;
x7457 += 64;
x7458 += 1;

}
x7449 += 32768;
x7450 += 32768;

}
// resize to WrappedArray(-1, 1, 1)
float* x7499 = (float*)myMalloc(2097152 * sizeof(float));;
int32_t x7500 = 0;
int32_t x7501 = 0;
int32_t x7502 = 0;
for(int x7503=0; x7503 < 64; x7503++) {
int32_t x7504 = x7501;
int32_t x7505 = x7502;
int32_t x7506 = x7500;
int32_t x7507 = x7506;
int32_t x7508 = x7504;
int32_t x7509 = x7505;
for(int x7510=0; x7510 < 512; x7510++) {
int32_t x7511 = x7508;
int32_t x7512 = x7509;
int32_t x7513 = x7507;
int32_t x7514 = x7513;
int32_t x7515 = x7511;
int32_t x7516 = x7512;
for(int x7517=0; x7517 < 8; x7517++) {
int32_t x7518 = x7515;
int32_t x7519 = x7516;
int32_t x7520 = x7514;
int32_t x7521 = x7520;
int32_t x7522 = x7518;
int32_t x7523 = x7519;
for(int x7524=0; x7524 < 8; x7524++) {
int32_t x7525 = x7521;
int32_t x7526 = x7522;
float x7527 = x7448[x7526];
int32_t x7528 = x7523;
float x7529 = x53[x7528];
float x7530 = x7527 + x7529;
x7499[x7525] = x7530;
x7521 += 1;
x7522 += 1;

}
x7514 += 8;
x7515 += 8;

}
x7507 += 64;
x7508 += 64;
x7509 += 1;

}
x7500 += 32768;
x7501 += 32768;

}
int32_t x7549 = 0;
int32_t x7550 = 0;
int32_t x7551 = 0;
for(int x7552=0; x7552 < 64; x7552++) {
int32_t x7553 = x7550;
int32_t x7554 = x7551;
int32_t x7555 = x7549;
int32_t x7556 = x7555;
int32_t x7557 = x7553;
int32_t x7558 = x7554;
for(int x7559=0; x7559 < 512; x7559++) {
int32_t x7560 = x7557;
int32_t x7561 = x7558;
int32_t x7562 = x7556;
int32_t x7563 = x7562;
int32_t x7564 = x7560;
int32_t x7565 = x7561;
for(int x7566=0; x7566 < 8; x7566++) {
int32_t x7567 = x7564;
int32_t x7568 = x7565;
int32_t x7569 = x7563;
int32_t x7570 = x7569;
int32_t x7571 = x7567;
int32_t x7572 = x7568;
for(int x7573=0; x7573 < 8; x7573++) {
int32_t x7574 = x7571;
float x7575 = x7499[x7574];
int32_t x7576 = x7572;
float x7577 = x6693[x7576];
float x7578 = x7575 + x7577;
x7499[x7574] = x7578;
x7570 += 1;
x7571 += 1;
x7572 += 1;

}
x7563 += 8;
x7564 += 8;
x7565 += 8;

}
x7556 += 64;
x7557 += 64;
x7558 += 64;

}
x7549 += 32768;
x7550 += 32768;
x7551 += 32768;

}
float* x7600 = (float*)myMalloc(2097152 * sizeof(float));;
for(int x7601=0; x7601 < 2097152; x7601++) {
float x7602 = x7499[x7601];
bool x7603 = x7602 < 0.0f;
if (x7603) {
x7600[x7601] = 0.0f;
} else {
float x7606 = x7499[x7601];
x7600[x7601] = x7606;
}

}
float* x7612 = (float*)myMalloc(1048576 * sizeof(float));;
float* x7613 = (float*)myMalloc(2097152 * sizeof(float));;
for(int x7614=0; x7614 < 64; x7614++) {
int32_t x7615 = x7614 * 32768;
float* x7616 = x7600+x7615;
int32_t x7617 = x7614 * 16384;
float* x7618 = x7612+x7617;
float* x7619 = x7613+x7615;
for(int x7620=0; x7620 < 512; x7620++) {
int32_t x7621 = x7620 / 1;
int32_t x7625 = x7621 * 8;
int32_t x7626 = x7625 * 8;
int32_t x7622 = x7620 % 1;
int32_t x7623 = x7622 / 1;
int32_t x7627 = x7623 * 8;
int32_t x7628 = x7627 * 8;
int32_t x7629 = x7626 + x7628;
int32_t x7624 = x7622 % 1;
int32_t x7630 = x7624 * 8;
int32_t x7631 = x7630 * 8;
int32_t x7632 = x7629 + x7631;
float* x7633 = x7619+x7632;
float* x7634 = x7616+x7626;
for(int x7635=0; x7635 < 8; x7635++) {
int32_t x7637 = x7635 * 8;
float* x7638 = x7633+x7637;
int32_t x7636 = x7635 + x7623;
int32_t x7639 = x7636 * 8;
int32_t x7640 = x7639 + x7624;
float* x7641 = x7634+x7640;
memcpy(x7638, x7641, 4 * 8);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,64,512,1,x179,512,x7619,64,1,x7618,64);

}
// resize to WrappedArray(-1, 1, 1)
float* x7651 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x7652 = 0;
int32_t x7653 = 0;
int32_t x7654 = 0;
for(int x7655=0; x7655 < 64; x7655++) {
int32_t x7656 = x7653;
int32_t x7657 = x7654;
int32_t x7658 = x7652;
int32_t x7659 = x7658;
int32_t x7660 = x7656;
int32_t x7661 = x7657;
for(int x7662=0; x7662 < 256; x7662++) {
int32_t x7663 = x7660;
int32_t x7664 = x7661;
int32_t x7665 = x7659;
int32_t x7666 = x7665;
int32_t x7667 = x7663;
int32_t x7668 = x7664;
for(int x7669=0; x7669 < 8; x7669++) {
int32_t x7670 = x7667;
int32_t x7671 = x7668;
int32_t x7672 = x7666;
int32_t x7673 = x7672;
int32_t x7674 = x7670;
int32_t x7675 = x7671;
for(int x7676=0; x7676 < 8; x7676++) {
int32_t x7677 = x7673;
int32_t x7678 = x7674;
float x7679 = x7612[x7678];
int32_t x7680 = x7675;
float x7681 = x130[x7680];
float x7682 = x7679 - x7681;
x7651[x7677] = x7682;
x7673 += 1;
x7674 += 1;

}
x7666 += 8;
x7667 += 8;

}
x7659 += 64;
x7660 += 64;
x7661 += 1;

}
x7652 += 16384;
x7653 += 16384;

}
float* x7701 = (float*)myMalloc(256 * sizeof(float));;
for(int x7702=0; x7702 < 256; x7702++) {
float x7703 = x197[x7702];
float x7704 = x7703 + 1.0E-5f;
x7701[x7702] = x7704;

}
float* x7708 = (float*)myMalloc(256 * sizeof(float));;
for(int x7709=0; x7709 < 256; x7709++) {
float x7710 = x7701[x7709];
double x7711 = (double)x7710;
double x7712 = sqrt(x7711);
float x7713 = (float)x7712;
x7708[x7709] = x7713;

}
// resize to WrappedArray(-1, 1, 1)
float* x7718 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x7719 = 0;
int32_t x7720 = 0;
int32_t x7721 = 0;
for(int x7722=0; x7722 < 64; x7722++) {
int32_t x7723 = x7720;
int32_t x7724 = x7721;
int32_t x7725 = x7719;
int32_t x7726 = x7725;
int32_t x7727 = x7723;
int32_t x7728 = x7724;
for(int x7729=0; x7729 < 256; x7729++) {
int32_t x7730 = x7727;
int32_t x7731 = x7728;
int32_t x7732 = x7726;
int32_t x7733 = x7732;
int32_t x7734 = x7730;
int32_t x7735 = x7731;
for(int x7736=0; x7736 < 8; x7736++) {
int32_t x7737 = x7734;
int32_t x7738 = x7735;
int32_t x7739 = x7733;
int32_t x7740 = x7739;
int32_t x7741 = x7737;
int32_t x7742 = x7738;
for(int x7743=0; x7743 < 8; x7743++) {
int32_t x7744 = x7740;
int32_t x7745 = x7741;
float x7746 = x7651[x7745];
int32_t x7747 = x7742;
float x7748 = x7708[x7747];
float x7749 = x7746 / x7748;
x7718[x7744] = x7749;
x7740 += 1;
x7741 += 1;

}
x7733 += 8;
x7734 += 8;

}
x7726 += 64;
x7727 += 64;
x7728 += 1;

}
x7719 += 16384;
x7720 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x7769 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x7770 = 0;
int32_t x7771 = 0;
int32_t x7772 = 0;
for(int x7773=0; x7773 < 64; x7773++) {
int32_t x7774 = x7771;
int32_t x7775 = x7772;
int32_t x7776 = x7770;
int32_t x7777 = x7776;
int32_t x7778 = x7774;
int32_t x7779 = x7775;
for(int x7780=0; x7780 < 256; x7780++) {
int32_t x7781 = x7778;
int32_t x7782 = x7779;
int32_t x7783 = x7777;
int32_t x7784 = x7783;
int32_t x7785 = x7781;
int32_t x7786 = x7782;
for(int x7787=0; x7787 < 8; x7787++) {
int32_t x7788 = x7785;
int32_t x7789 = x7786;
int32_t x7790 = x7784;
int32_t x7791 = x7790;
int32_t x7792 = x7788;
int32_t x7793 = x7789;
for(int x7794=0; x7794 < 8; x7794++) {
int32_t x7795 = x7791;
int32_t x7796 = x7792;
float x7797 = x7718[x7796];
int32_t x7798 = x7793;
float x7799 = x269[x7798];
float x7800 = x7797 * x7799;
x7769[x7795] = x7800;
x7791 += 1;
x7792 += 1;

}
x7784 += 8;
x7785 += 8;

}
x7777 += 64;
x7778 += 64;
x7779 += 1;

}
x7770 += 16384;
x7771 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x7820 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x7821 = 0;
int32_t x7822 = 0;
int32_t x7823 = 0;
for(int x7824=0; x7824 < 64; x7824++) {
int32_t x7825 = x7822;
int32_t x7826 = x7823;
int32_t x7827 = x7821;
int32_t x7828 = x7827;
int32_t x7829 = x7825;
int32_t x7830 = x7826;
for(int x7831=0; x7831 < 256; x7831++) {
int32_t x7832 = x7829;
int32_t x7833 = x7830;
int32_t x7834 = x7828;
int32_t x7835 = x7834;
int32_t x7836 = x7832;
int32_t x7837 = x7833;
for(int x7838=0; x7838 < 8; x7838++) {
int32_t x7839 = x7836;
int32_t x7840 = x7837;
int32_t x7841 = x7835;
int32_t x7842 = x7841;
int32_t x7843 = x7839;
int32_t x7844 = x7840;
for(int x7845=0; x7845 < 8; x7845++) {
int32_t x7846 = x7842;
int32_t x7847 = x7843;
float x7848 = x7769[x7847];
int32_t x7849 = x7844;
float x7850 = x20[x7849];
float x7851 = x7848 + x7850;
x7820[x7846] = x7851;
x7842 += 1;
x7843 += 1;

}
x7835 += 8;
x7836 += 8;

}
x7828 += 64;
x7829 += 64;
x7830 += 1;

}
x7821 += 16384;
x7822 += 16384;

}
float* x7870 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x7871=0; x7871 < 1048576; x7871++) {
float x7872 = x7820[x7871];
bool x7873 = x7872 < 0.0f;
if (x7873) {
x7870[x7871] = 0.0f;
} else {
float x7876 = x7820[x7871];
x7870[x7871] = x7876;
}

}
float* x7882 = (float*)myMalloc(262144 * sizeof(float));;
float* x7883 = (float*)myMalloc(2359296 * sizeof(float));;
for(int x7884=0; x7884 < 64; x7884++) {
int32_t x7885 = x7884 * 16384;
float* x7886 = x7870+x7885;
int32_t x7887 = x7884 * 4096;
float* x7888 = x7882+x7887;
int32_t x7889 = x7884 * 36864;
float* x7890 = x7883+x7889;
for(int x7892=0; x7892 < 2304; x7892++) {
int32_t x7893 = x7892 / 9;
int32_t x7897 = x7893 * 3;
int32_t x7898 = x7897 * 3;
int32_t x7899 = x7898 * 4;
int32_t x7900 = x7899 * 4;
int32_t x7894 = x7892 % 9;
int32_t x7895 = x7894 / 3;
int32_t x7901 = x7895 * 3;
int32_t x7902 = x7901 * 4;
int32_t x7903 = x7902 * 4;
int32_t x7904 = x7900 + x7903;
int32_t x7896 = x7894 % 3;
int32_t x7905 = x7896 * 4;
int32_t x7906 = x7905 * 4;
int32_t x7907 = x7904 + x7906;
float* x7908 = x7890+x7907;
int32_t x7909 = x7893 * 8;
int32_t x7910 = x7909 * 8;
float* x7911 = x7886+x7910;
for(int x7913=0; x7913 < 4; x7913++) {
int32_t x7914 = x7913 * 2;
int32_t x7915 = x7914 - 1;
int32_t x7916 = x7915 + x7895;
bool x7917 = x7916 < 0;
bool x7918 = x7916 >= 8;
bool x7919 = x7917 || x7918;
if (x7919) {
int32_t x7920 = x7913 * 4;
float* x7921 = x7908+x7920;
memset(x7921, 0, 4 * 4);;
} else {
int32_t x7920 = x7913 * 4;
int32_t x7936 = x7916 * 8;
for(int x7924=0; x7924 < 4; x7924++) {
int32_t x7925 = x7924 * 2;
int32_t x7926 = x7925 - 1;
int32_t x7927 = x7926 + x7896;
bool x7928 = x7927 < 0;
bool x7929 = x7927 >= 8;
bool x7930 = x7928 || x7929;
if (x7930) {
int32_t x7931 = x7920 + x7924;
float* x7932 = x7908+x7931;
memset(x7932, 0, 4 * 1);;
} else {
int32_t x7931 = x7920 + x7924;
float* x7935 = x7908+x7931;
int32_t x7937 = x7936 + x7927;
float* x7938 = x7911+x7937;
memcpy(x7935, x7938, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,16,2304,1,x174,2304,x7890,16,1,x7888,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x7954 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x7955 = 0;
int32_t x7956 = 0;
int32_t x7957 = 0;
for(int x7958=0; x7958 < 64; x7958++) {
int32_t x7959 = x7956;
int32_t x7960 = x7957;
int32_t x7961 = x7955;
int32_t x7962 = x7961;
int32_t x7963 = x7959;
int32_t x7964 = x7960;
for(int x7965=0; x7965 < 256; x7965++) {
int32_t x7966 = x7963;
int32_t x7967 = x7964;
int32_t x7968 = x7962;
int32_t x7969 = x7968;
int32_t x7970 = x7966;
int32_t x7971 = x7967;
for(int x7972=0; x7972 < 4; x7972++) {
int32_t x7973 = x7970;
int32_t x7974 = x7971;
int32_t x7975 = x7969;
int32_t x7976 = x7975;
int32_t x7977 = x7973;
int32_t x7978 = x7974;
for(int x7979=0; x7979 < 4; x7979++) {
int32_t x7980 = x7976;
int32_t x7981 = x7977;
float x7982 = x7882[x7981];
int32_t x7983 = x7978;
float x7984 = x228[x7983];
float x7985 = x7982 - x7984;
x7954[x7980] = x7985;
x7976 += 1;
x7977 += 1;

}
x7969 += 4;
x7970 += 4;

}
x7962 += 16;
x7963 += 16;
x7964 += 1;

}
x7955 += 4096;
x7956 += 4096;

}
float* x8004 = (float*)myMalloc(256 * sizeof(float));;
for(int x8005=0; x8005 < 256; x8005++) {
float x8006 = x98[x8005];
float x8007 = x8006 + 1.0E-5f;
x8004[x8005] = x8007;

}
float* x8011 = (float*)myMalloc(256 * sizeof(float));;
for(int x8012=0; x8012 < 256; x8012++) {
float x8013 = x8004[x8012];
double x8014 = (double)x8013;
double x8015 = sqrt(x8014);
float x8016 = (float)x8015;
x8011[x8012] = x8016;

}
// resize to WrappedArray(-1, 1, 1)
float* x8021 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x8022 = 0;
int32_t x8023 = 0;
int32_t x8024 = 0;
for(int x8025=0; x8025 < 64; x8025++) {
int32_t x8026 = x8023;
int32_t x8027 = x8024;
int32_t x8028 = x8022;
int32_t x8029 = x8028;
int32_t x8030 = x8026;
int32_t x8031 = x8027;
for(int x8032=0; x8032 < 256; x8032++) {
int32_t x8033 = x8030;
int32_t x8034 = x8031;
int32_t x8035 = x8029;
int32_t x8036 = x8035;
int32_t x8037 = x8033;
int32_t x8038 = x8034;
for(int x8039=0; x8039 < 4; x8039++) {
int32_t x8040 = x8037;
int32_t x8041 = x8038;
int32_t x8042 = x8036;
int32_t x8043 = x8042;
int32_t x8044 = x8040;
int32_t x8045 = x8041;
for(int x8046=0; x8046 < 4; x8046++) {
int32_t x8047 = x8043;
int32_t x8048 = x8044;
float x8049 = x7954[x8048];
int32_t x8050 = x8045;
float x8051 = x8011[x8050];
float x8052 = x8049 / x8051;
x8021[x8047] = x8052;
x8043 += 1;
x8044 += 1;

}
x8036 += 4;
x8037 += 4;

}
x8029 += 16;
x8030 += 16;
x8031 += 1;

}
x8022 += 4096;
x8023 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x8072 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x8073 = 0;
int32_t x8074 = 0;
int32_t x8075 = 0;
for(int x8076=0; x8076 < 64; x8076++) {
int32_t x8077 = x8074;
int32_t x8078 = x8075;
int32_t x8079 = x8073;
int32_t x8080 = x8079;
int32_t x8081 = x8077;
int32_t x8082 = x8078;
for(int x8083=0; x8083 < 256; x8083++) {
int32_t x8084 = x8081;
int32_t x8085 = x8082;
int32_t x8086 = x8080;
int32_t x8087 = x8086;
int32_t x8088 = x8084;
int32_t x8089 = x8085;
for(int x8090=0; x8090 < 4; x8090++) {
int32_t x8091 = x8088;
int32_t x8092 = x8089;
int32_t x8093 = x8087;
int32_t x8094 = x8093;
int32_t x8095 = x8091;
int32_t x8096 = x8092;
for(int x8097=0; x8097 < 4; x8097++) {
int32_t x8098 = x8094;
int32_t x8099 = x8095;
float x8100 = x8021[x8099];
int32_t x8101 = x8096;
float x8102 = x107[x8101];
float x8103 = x8100 * x8102;
x8072[x8098] = x8103;
x8094 += 1;
x8095 += 1;

}
x8087 += 4;
x8088 += 4;

}
x8080 += 16;
x8081 += 16;
x8082 += 1;

}
x8073 += 4096;
x8074 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x8123 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x8124 = 0;
int32_t x8125 = 0;
int32_t x8126 = 0;
for(int x8127=0; x8127 < 64; x8127++) {
int32_t x8128 = x8125;
int32_t x8129 = x8126;
int32_t x8130 = x8124;
int32_t x8131 = x8130;
int32_t x8132 = x8128;
int32_t x8133 = x8129;
for(int x8134=0; x8134 < 256; x8134++) {
int32_t x8135 = x8132;
int32_t x8136 = x8133;
int32_t x8137 = x8131;
int32_t x8138 = x8137;
int32_t x8139 = x8135;
int32_t x8140 = x8136;
for(int x8141=0; x8141 < 4; x8141++) {
int32_t x8142 = x8139;
int32_t x8143 = x8140;
int32_t x8144 = x8138;
int32_t x8145 = x8144;
int32_t x8146 = x8142;
int32_t x8147 = x8143;
for(int x8148=0; x8148 < 4; x8148++) {
int32_t x8149 = x8145;
int32_t x8150 = x8146;
float x8151 = x8072[x8150];
int32_t x8152 = x8147;
float x8153 = x15[x8152];
float x8154 = x8151 + x8153;
x8123[x8149] = x8154;
x8145 += 1;
x8146 += 1;

}
x8138 += 4;
x8139 += 4;

}
x8131 += 16;
x8132 += 16;
x8133 += 1;

}
x8124 += 4096;
x8125 += 4096;

}
float* x8173 = (float*)myMalloc(262144 * sizeof(float));;
for(int x8175=0; x8175 < 262144; x8175++) {
float x8176 = x8123[x8175];
bool x8177 = x8176 < 0.0f;
if (x8177) {
x8173[x8175] = 0.0f;
} else {
float x8180 = x8123[x8175];
x8173[x8175] = x8180;
}

}
float* x8186 = (float*)myMalloc(1048576 * sizeof(float));;
float* x8187 = (float*)myMalloc(262144 * sizeof(float));;
for(int x8188=0; x8188 < 64; x8188++) {
int32_t x8189 = x8188 * 4096;
float* x8190 = x8173+x8189;
int32_t x8191 = x8188 * 16384;
float* x8192 = x8186+x8191;
float* x8193 = x8187+x8189;
for(int x8194=0; x8194 < 256; x8194++) {
int32_t x8195 = x8194 / 1;
int32_t x8199 = x8195 * 4;
int32_t x8200 = x8199 * 4;
int32_t x8196 = x8194 % 1;
int32_t x8197 = x8196 / 1;
int32_t x8201 = x8197 * 4;
int32_t x8202 = x8201 * 4;
int32_t x8203 = x8200 + x8202;
int32_t x8198 = x8196 % 1;
int32_t x8204 = x8198 * 4;
int32_t x8205 = x8204 * 4;
int32_t x8206 = x8203 + x8205;
float* x8207 = x8193+x8206;
float* x8208 = x8190+x8200;
for(int x8209=0; x8209 < 4; x8209++) {
int32_t x8211 = x8209 * 4;
float* x8212 = x8207+x8211;
int32_t x8210 = x8209 + x8197;
int32_t x8213 = x8210 * 4;
int32_t x8214 = x8213 + x8198;
float* x8215 = x8208+x8214;
memcpy(x8212, x8215, 4 * 4);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,16,256,1,x268,256,x8193,16,1,x8192,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x8225 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x8226 = 0;
int32_t x8227 = 0;
int32_t x8228 = 0;
for(int x8229=0; x8229 < 64; x8229++) {
int32_t x8230 = x8227;
int32_t x8231 = x8228;
int32_t x8232 = x8226;
int32_t x8233 = x8232;
int32_t x8234 = x8230;
int32_t x8235 = x8231;
for(int x8237=0; x8237 < 1024; x8237++) {
int32_t x8238 = x8234;
int32_t x8239 = x8235;
int32_t x8240 = x8233;
int32_t x8241 = x8240;
int32_t x8242 = x8238;
int32_t x8243 = x8239;
for(int x8244=0; x8244 < 4; x8244++) {
int32_t x8245 = x8242;
int32_t x8246 = x8243;
int32_t x8247 = x8241;
int32_t x8248 = x8247;
int32_t x8249 = x8245;
int32_t x8250 = x8246;
for(int x8251=0; x8251 < 4; x8251++) {
int32_t x8252 = x8248;
int32_t x8253 = x8249;
float x8254 = x8186[x8253];
int32_t x8255 = x8250;
float x8256 = x215[x8255];
float x8257 = x8254 - x8256;
x8225[x8252] = x8257;
x8248 += 1;
x8249 += 1;

}
x8241 += 4;
x8242 += 4;

}
x8233 += 16;
x8234 += 16;
x8235 += 1;

}
x8226 += 16384;
x8227 += 16384;

}
float* x8276 = (float*)myMalloc(1024 * sizeof(float));;
for(int x8277=0; x8277 < 1024; x8277++) {
float x8278 = x266[x8277];
float x8279 = x8278 + 1.0E-5f;
x8276[x8277] = x8279;

}
float* x8283 = (float*)myMalloc(1024 * sizeof(float));;
for(int x8284=0; x8284 < 1024; x8284++) {
float x8285 = x8276[x8284];
double x8286 = (double)x8285;
double x8287 = sqrt(x8286);
float x8288 = (float)x8287;
x8283[x8284] = x8288;

}
// resize to WrappedArray(-1, 1, 1)
float* x8293 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x8294 = 0;
int32_t x8295 = 0;
int32_t x8296 = 0;
for(int x8297=0; x8297 < 64; x8297++) {
int32_t x8298 = x8295;
int32_t x8299 = x8296;
int32_t x8300 = x8294;
int32_t x8301 = x8300;
int32_t x8302 = x8298;
int32_t x8303 = x8299;
for(int x8304=0; x8304 < 1024; x8304++) {
int32_t x8305 = x8302;
int32_t x8306 = x8303;
int32_t x8307 = x8301;
int32_t x8308 = x8307;
int32_t x8309 = x8305;
int32_t x8310 = x8306;
for(int x8311=0; x8311 < 4; x8311++) {
int32_t x8312 = x8309;
int32_t x8313 = x8310;
int32_t x8314 = x8308;
int32_t x8315 = x8314;
int32_t x8316 = x8312;
int32_t x8317 = x8313;
for(int x8318=0; x8318 < 4; x8318++) {
int32_t x8319 = x8315;
int32_t x8320 = x8316;
float x8321 = x8225[x8320];
int32_t x8322 = x8317;
float x8323 = x8283[x8322];
float x8324 = x8321 / x8323;
x8293[x8319] = x8324;
x8315 += 1;
x8316 += 1;

}
x8308 += 4;
x8309 += 4;

}
x8301 += 16;
x8302 += 16;
x8303 += 1;

}
x8294 += 16384;
x8295 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x8344 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x8345 = 0;
int32_t x8346 = 0;
int32_t x8347 = 0;
for(int x8348=0; x8348 < 64; x8348++) {
int32_t x8349 = x8346;
int32_t x8350 = x8347;
int32_t x8351 = x8345;
int32_t x8352 = x8351;
int32_t x8353 = x8349;
int32_t x8354 = x8350;
for(int x8355=0; x8355 < 1024; x8355++) {
int32_t x8356 = x8353;
int32_t x8357 = x8354;
int32_t x8358 = x8352;
int32_t x8359 = x8358;
int32_t x8360 = x8356;
int32_t x8361 = x8357;
for(int x8362=0; x8362 < 4; x8362++) {
int32_t x8363 = x8360;
int32_t x8364 = x8361;
int32_t x8365 = x8359;
int32_t x8366 = x8365;
int32_t x8367 = x8363;
int32_t x8368 = x8364;
for(int x8369=0; x8369 < 4; x8369++) {
int32_t x8370 = x8366;
int32_t x8371 = x8367;
float x8372 = x8293[x8371];
int32_t x8373 = x8368;
float x8374 = x17[x8373];
float x8375 = x8372 * x8374;
x8344[x8370] = x8375;
x8366 += 1;
x8367 += 1;

}
x8359 += 4;
x8360 += 4;

}
x8352 += 16;
x8353 += 16;
x8354 += 1;

}
x8345 += 16384;
x8346 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x8395 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x8396 = 0;
int32_t x8397 = 0;
int32_t x8398 = 0;
for(int x8399=0; x8399 < 64; x8399++) {
int32_t x8400 = x8397;
int32_t x8401 = x8398;
int32_t x8402 = x8396;
int32_t x8403 = x8402;
int32_t x8404 = x8400;
int32_t x8405 = x8401;
for(int x8406=0; x8406 < 1024; x8406++) {
int32_t x8407 = x8404;
int32_t x8408 = x8405;
int32_t x8409 = x8403;
int32_t x8410 = x8409;
int32_t x8411 = x8407;
int32_t x8412 = x8408;
for(int x8413=0; x8413 < 4; x8413++) {
int32_t x8414 = x8411;
int32_t x8415 = x8412;
int32_t x8416 = x8410;
int32_t x8417 = x8416;
int32_t x8418 = x8414;
int32_t x8419 = x8415;
for(int x8420=0; x8420 < 4; x8420++) {
int32_t x8421 = x8417;
int32_t x8422 = x8418;
float x8423 = x8344[x8422];
int32_t x8424 = x8419;
float x8425 = x116[x8424];
float x8426 = x8423 + x8425;
x8395[x8421] = x8426;
x8417 += 1;
x8418 += 1;

}
x8410 += 4;
x8411 += 4;

}
x8403 += 16;
x8404 += 16;
x8405 += 1;

}
x8396 += 16384;
x8397 += 16384;

}
float* x8445 = (float*)myMalloc(1048576 * sizeof(float));;
float* x8446 = (float*)myMalloc(524288 * sizeof(float));;
for(int x8447=0; x8447 < 64; x8447++) {
int32_t x8448 = x8447 * 32768;
float* x8449 = x7600+x8448;
int32_t x8450 = x8447 * 16384;
float* x8451 = x8445+x8450;
int32_t x8452 = x8447 * 8192;
float* x8453 = x8446+x8452;
for(int x8454=0; x8454 < 512; x8454++) {
int32_t x8455 = x8454 / 1;
int32_t x8459 = x8455 * 4;
int32_t x8460 = x8459 * 4;
int32_t x8456 = x8454 % 1;
int32_t x8457 = x8456 / 1;
int32_t x8461 = x8457 * 4;
int32_t x8462 = x8461 * 4;
int32_t x8463 = x8460 + x8462;
int32_t x8458 = x8456 % 1;
int32_t x8464 = x8458 * 4;
int32_t x8465 = x8464 * 4;
int32_t x8466 = x8463 + x8465;
float* x8467 = x8453+x8466;
int32_t x8468 = x8455 * 8;
int32_t x8469 = x8468 * 8;
float* x8470 = x8449+x8469;
for(int x8471=0; x8471 < 4; x8471++) {
int32_t x8475 = x8471 * 4;
int32_t x8472 = x8471 * 2;
int32_t x8473 = x8472 + x8457;
int32_t x8478 = x8473 * 8;
int32_t x8479 = x8478 + x8458;
for(int x8474=0; x8474 < 4; x8474++) {
int32_t x8476 = x8475 + x8474;
float* x8477 = x8467+x8476;
int32_t x8480 = x8474 * 2;
int32_t x8481 = x8479 + x8480;
float* x8482 = x8470+x8481;
memcpy(x8477, x8482, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,16,512,1,x74,512,x8453,16,1,x8451,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x8494 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x8495 = 0;
int32_t x8496 = 0;
int32_t x8497 = 0;
for(int x8498=0; x8498 < 64; x8498++) {
int32_t x8499 = x8496;
int32_t x8500 = x8497;
int32_t x8501 = x8495;
int32_t x8502 = x8501;
int32_t x8503 = x8499;
int32_t x8504 = x8500;
for(int x8505=0; x8505 < 1024; x8505++) {
int32_t x8506 = x8503;
int32_t x8507 = x8504;
int32_t x8508 = x8502;
int32_t x8509 = x8508;
int32_t x8510 = x8506;
int32_t x8511 = x8507;
for(int x8512=0; x8512 < 4; x8512++) {
int32_t x8513 = x8510;
int32_t x8514 = x8511;
int32_t x8515 = x8509;
int32_t x8516 = x8515;
int32_t x8517 = x8513;
int32_t x8518 = x8514;
for(int x8519=0; x8519 < 4; x8519++) {
int32_t x8520 = x8516;
int32_t x8521 = x8517;
float x8522 = x8445[x8521];
int32_t x8523 = x8518;
float x8524 = x85[x8523];
float x8525 = x8522 - x8524;
x8494[x8520] = x8525;
x8516 += 1;
x8517 += 1;

}
x8509 += 4;
x8510 += 4;

}
x8502 += 16;
x8503 += 16;
x8504 += 1;

}
x8495 += 16384;
x8496 += 16384;

}
float* x8544 = (float*)myMalloc(1024 * sizeof(float));;
for(int x8545=0; x8545 < 1024; x8545++) {
float x8546 = x210[x8545];
float x8547 = x8546 + 1.0E-5f;
x8544[x8545] = x8547;

}
float* x8551 = (float*)myMalloc(1024 * sizeof(float));;
for(int x8552=0; x8552 < 1024; x8552++) {
float x8553 = x8544[x8552];
double x8554 = (double)x8553;
double x8555 = sqrt(x8554);
float x8556 = (float)x8555;
x8551[x8552] = x8556;

}
// resize to WrappedArray(-1, 1, 1)
float* x8561 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x8562 = 0;
int32_t x8563 = 0;
int32_t x8564 = 0;
for(int x8565=0; x8565 < 64; x8565++) {
int32_t x8566 = x8563;
int32_t x8567 = x8564;
int32_t x8568 = x8562;
int32_t x8569 = x8568;
int32_t x8570 = x8566;
int32_t x8571 = x8567;
for(int x8572=0; x8572 < 1024; x8572++) {
int32_t x8573 = x8570;
int32_t x8574 = x8571;
int32_t x8575 = x8569;
int32_t x8576 = x8575;
int32_t x8577 = x8573;
int32_t x8578 = x8574;
for(int x8579=0; x8579 < 4; x8579++) {
int32_t x8580 = x8577;
int32_t x8581 = x8578;
int32_t x8582 = x8576;
int32_t x8583 = x8582;
int32_t x8584 = x8580;
int32_t x8585 = x8581;
for(int x8586=0; x8586 < 4; x8586++) {
int32_t x8587 = x8583;
int32_t x8588 = x8584;
float x8589 = x8494[x8588];
int32_t x8590 = x8585;
float x8591 = x8551[x8590];
float x8592 = x8589 / x8591;
x8561[x8587] = x8592;
x8583 += 1;
x8584 += 1;

}
x8576 += 4;
x8577 += 4;

}
x8569 += 16;
x8570 += 16;
x8571 += 1;

}
x8562 += 16384;
x8563 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x8612 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x8613 = 0;
int32_t x8614 = 0;
int32_t x8615 = 0;
for(int x8616=0; x8616 < 64; x8616++) {
int32_t x8617 = x8614;
int32_t x8618 = x8615;
int32_t x8619 = x8613;
int32_t x8620 = x8619;
int32_t x8621 = x8617;
int32_t x8622 = x8618;
for(int x8623=0; x8623 < 1024; x8623++) {
int32_t x8624 = x8621;
int32_t x8625 = x8622;
int32_t x8626 = x8620;
int32_t x8627 = x8626;
int32_t x8628 = x8624;
int32_t x8629 = x8625;
for(int x8630=0; x8630 < 4; x8630++) {
int32_t x8631 = x8628;
int32_t x8632 = x8629;
int32_t x8633 = x8627;
int32_t x8634 = x8633;
int32_t x8635 = x8631;
int32_t x8636 = x8632;
for(int x8637=0; x8637 < 4; x8637++) {
int32_t x8638 = x8634;
int32_t x8639 = x8635;
float x8640 = x8561[x8639];
int32_t x8641 = x8636;
float x8642 = x28[x8641];
float x8643 = x8640 * x8642;
x8612[x8638] = x8643;
x8634 += 1;
x8635 += 1;

}
x8627 += 4;
x8628 += 4;

}
x8620 += 16;
x8621 += 16;
x8622 += 1;

}
x8613 += 16384;
x8614 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x8663 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x8664 = 0;
int32_t x8665 = 0;
int32_t x8666 = 0;
for(int x8667=0; x8667 < 64; x8667++) {
int32_t x8668 = x8665;
int32_t x8669 = x8666;
int32_t x8670 = x8664;
int32_t x8671 = x8670;
int32_t x8672 = x8668;
int32_t x8673 = x8669;
for(int x8674=0; x8674 < 1024; x8674++) {
int32_t x8675 = x8672;
int32_t x8676 = x8673;
int32_t x8677 = x8671;
int32_t x8678 = x8677;
int32_t x8679 = x8675;
int32_t x8680 = x8676;
for(int x8681=0; x8681 < 4; x8681++) {
int32_t x8682 = x8679;
int32_t x8683 = x8680;
int32_t x8684 = x8678;
int32_t x8685 = x8684;
int32_t x8686 = x8682;
int32_t x8687 = x8683;
for(int x8688=0; x8688 < 4; x8688++) {
int32_t x8689 = x8685;
int32_t x8690 = x8686;
float x8691 = x8612[x8690];
int32_t x8692 = x8687;
float x8693 = x219[x8692];
float x8694 = x8691 + x8693;
x8663[x8689] = x8694;
x8685 += 1;
x8686 += 1;

}
x8678 += 4;
x8679 += 4;

}
x8671 += 16;
x8672 += 16;
x8673 += 1;

}
x8664 += 16384;
x8665 += 16384;

}
int32_t x8713 = 0;
int32_t x8714 = 0;
int32_t x8715 = 0;
for(int x8716=0; x8716 < 64; x8716++) {
int32_t x8717 = x8714;
int32_t x8718 = x8715;
int32_t x8719 = x8713;
int32_t x8720 = x8719;
int32_t x8721 = x8717;
int32_t x8722 = x8718;
for(int x8723=0; x8723 < 1024; x8723++) {
int32_t x8724 = x8721;
int32_t x8725 = x8722;
int32_t x8726 = x8720;
int32_t x8727 = x8726;
int32_t x8728 = x8724;
int32_t x8729 = x8725;
for(int x8730=0; x8730 < 4; x8730++) {
int32_t x8731 = x8728;
int32_t x8732 = x8729;
int32_t x8733 = x8727;
int32_t x8734 = x8733;
int32_t x8735 = x8731;
int32_t x8736 = x8732;
for(int x8737=0; x8737 < 4; x8737++) {
int32_t x8738 = x8735;
float x8739 = x8395[x8738];
int32_t x8740 = x8736;
float x8741 = x8663[x8740];
float x8742 = x8739 + x8741;
x8395[x8738] = x8742;
x8734 += 1;
x8735 += 1;
x8736 += 1;

}
x8727 += 4;
x8728 += 4;
x8729 += 4;

}
x8720 += 16;
x8721 += 16;
x8722 += 16;

}
x8713 += 16384;
x8714 += 16384;
x8715 += 16384;

}
float* x8764 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x8765=0; x8765 < 1048576; x8765++) {
float x8766 = x8395[x8765];
bool x8767 = x8766 < 0.0f;
if (x8767) {
x8764[x8765] = 0.0f;
} else {
float x8770 = x8395[x8765];
x8764[x8765] = x8770;
}

}
float* x8776 = (float*)myMalloc(262144 * sizeof(float));;
float* x8777 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x8778=0; x8778 < 64; x8778++) {
int32_t x8779 = x8778 * 16384;
float* x8780 = x8764+x8779;
int32_t x8781 = x8778 * 4096;
float* x8782 = x8776+x8781;
float* x8783 = x8777+x8779;
for(int x8784=0; x8784 < 1024; x8784++) {
int32_t x8785 = x8784 / 1;
int32_t x8789 = x8785 * 4;
int32_t x8790 = x8789 * 4;
int32_t x8786 = x8784 % 1;
int32_t x8787 = x8786 / 1;
int32_t x8791 = x8787 * 4;
int32_t x8792 = x8791 * 4;
int32_t x8793 = x8790 + x8792;
int32_t x8788 = x8786 % 1;
int32_t x8794 = x8788 * 4;
int32_t x8795 = x8794 * 4;
int32_t x8796 = x8793 + x8795;
float* x8797 = x8783+x8796;
float* x8798 = x8780+x8790;
for(int x8799=0; x8799 < 4; x8799++) {
int32_t x8801 = x8799 * 4;
float* x8802 = x8797+x8801;
int32_t x8800 = x8799 + x8787;
int32_t x8803 = x8800 * 4;
int32_t x8804 = x8803 + x8788;
float* x8805 = x8798+x8804;
memcpy(x8802, x8805, 4 * 4);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,16,1024,1,x12,1024,x8783,16,1,x8782,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x8815 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x8816 = 0;
int32_t x8817 = 0;
int32_t x8818 = 0;
for(int x8819=0; x8819 < 64; x8819++) {
int32_t x8820 = x8817;
int32_t x8821 = x8818;
int32_t x8822 = x8816;
int32_t x8823 = x8822;
int32_t x8824 = x8820;
int32_t x8825 = x8821;
for(int x8826=0; x8826 < 256; x8826++) {
int32_t x8827 = x8824;
int32_t x8828 = x8825;
int32_t x8829 = x8823;
int32_t x8830 = x8829;
int32_t x8831 = x8827;
int32_t x8832 = x8828;
for(int x8833=0; x8833 < 4; x8833++) {
int32_t x8834 = x8831;
int32_t x8835 = x8832;
int32_t x8836 = x8830;
int32_t x8837 = x8836;
int32_t x8838 = x8834;
int32_t x8839 = x8835;
for(int x8840=0; x8840 < 4; x8840++) {
int32_t x8841 = x8837;
int32_t x8842 = x8838;
float x8843 = x8776[x8842];
int32_t x8844 = x8839;
float x8845 = x258[x8844];
float x8846 = x8843 - x8845;
x8815[x8841] = x8846;
x8837 += 1;
x8838 += 1;

}
x8830 += 4;
x8831 += 4;

}
x8823 += 16;
x8824 += 16;
x8825 += 1;

}
x8816 += 4096;
x8817 += 4096;

}
float* x8865 = (float*)myMalloc(256 * sizeof(float));;
for(int x8866=0; x8866 < 256; x8866++) {
float x8867 = x156[x8866];
float x8868 = x8867 + 1.0E-5f;
x8865[x8866] = x8868;

}
float* x8872 = (float*)myMalloc(256 * sizeof(float));;
for(int x8873=0; x8873 < 256; x8873++) {
float x8874 = x8865[x8873];
double x8875 = (double)x8874;
double x8876 = sqrt(x8875);
float x8877 = (float)x8876;
x8872[x8873] = x8877;

}
// resize to WrappedArray(-1, 1, 1)
float* x8882 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x8883 = 0;
int32_t x8884 = 0;
int32_t x8885 = 0;
for(int x8886=0; x8886 < 64; x8886++) {
int32_t x8887 = x8884;
int32_t x8888 = x8885;
int32_t x8889 = x8883;
int32_t x8890 = x8889;
int32_t x8891 = x8887;
int32_t x8892 = x8888;
for(int x8893=0; x8893 < 256; x8893++) {
int32_t x8894 = x8891;
int32_t x8895 = x8892;
int32_t x8896 = x8890;
int32_t x8897 = x8896;
int32_t x8898 = x8894;
int32_t x8899 = x8895;
for(int x8900=0; x8900 < 4; x8900++) {
int32_t x8901 = x8898;
int32_t x8902 = x8899;
int32_t x8903 = x8897;
int32_t x8904 = x8903;
int32_t x8905 = x8901;
int32_t x8906 = x8902;
for(int x8907=0; x8907 < 4; x8907++) {
int32_t x8908 = x8904;
int32_t x8909 = x8905;
float x8910 = x8815[x8909];
int32_t x8911 = x8906;
float x8912 = x8872[x8911];
float x8913 = x8910 / x8912;
x8882[x8908] = x8913;
x8904 += 1;
x8905 += 1;

}
x8897 += 4;
x8898 += 4;

}
x8890 += 16;
x8891 += 16;
x8892 += 1;

}
x8883 += 4096;
x8884 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x8933 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x8934 = 0;
int32_t x8935 = 0;
int32_t x8936 = 0;
for(int x8937=0; x8937 < 64; x8937++) {
int32_t x8938 = x8935;
int32_t x8939 = x8936;
int32_t x8940 = x8934;
int32_t x8941 = x8940;
int32_t x8942 = x8938;
int32_t x8943 = x8939;
for(int x8944=0; x8944 < 256; x8944++) {
int32_t x8945 = x8942;
int32_t x8946 = x8943;
int32_t x8947 = x8941;
int32_t x8948 = x8947;
int32_t x8949 = x8945;
int32_t x8950 = x8946;
for(int x8951=0; x8951 < 4; x8951++) {
int32_t x8952 = x8949;
int32_t x8953 = x8950;
int32_t x8954 = x8948;
int32_t x8955 = x8954;
int32_t x8956 = x8952;
int32_t x8957 = x8953;
for(int x8958=0; x8958 < 4; x8958++) {
int32_t x8959 = x8955;
int32_t x8960 = x8956;
float x8961 = x8882[x8960];
int32_t x8962 = x8957;
float x8963 = x29[x8962];
float x8964 = x8961 * x8963;
x8933[x8959] = x8964;
x8955 += 1;
x8956 += 1;

}
x8948 += 4;
x8949 += 4;

}
x8941 += 16;
x8942 += 16;
x8943 += 1;

}
x8934 += 4096;
x8935 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x8984 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x8985 = 0;
int32_t x8986 = 0;
int32_t x8987 = 0;
for(int x8988=0; x8988 < 64; x8988++) {
int32_t x8989 = x8986;
int32_t x8990 = x8987;
int32_t x8991 = x8985;
int32_t x8992 = x8991;
int32_t x8993 = x8989;
int32_t x8994 = x8990;
for(int x8995=0; x8995 < 256; x8995++) {
int32_t x8996 = x8993;
int32_t x8997 = x8994;
int32_t x8998 = x8992;
int32_t x8999 = x8998;
int32_t x9000 = x8996;
int32_t x9001 = x8997;
for(int x9002=0; x9002 < 4; x9002++) {
int32_t x9003 = x9000;
int32_t x9004 = x9001;
int32_t x9005 = x8999;
int32_t x9006 = x9005;
int32_t x9007 = x9003;
int32_t x9008 = x9004;
for(int x9009=0; x9009 < 4; x9009++) {
int32_t x9010 = x9006;
int32_t x9011 = x9007;
float x9012 = x8933[x9011];
int32_t x9013 = x9008;
float x9014 = x218[x9013];
float x9015 = x9012 + x9014;
x8984[x9010] = x9015;
x9006 += 1;
x9007 += 1;

}
x8999 += 4;
x9000 += 4;

}
x8992 += 16;
x8993 += 16;
x8994 += 1;

}
x8985 += 4096;
x8986 += 4096;

}
float* x9034 = (float*)myMalloc(262144 * sizeof(float));;
for(int x9035=0; x9035 < 262144; x9035++) {
float x9036 = x8984[x9035];
bool x9037 = x9036 < 0.0f;
if (x9037) {
x9034[x9035] = 0.0f;
} else {
float x9040 = x8984[x9035];
x9034[x9035] = x9040;
}

}
float* x9046 = (float*)myMalloc(262144 * sizeof(float));;
float* x9047 = (float*)myMalloc(2359296 * sizeof(float));;
for(int x9048=0; x9048 < 64; x9048++) {
int32_t x9049 = x9048 * 4096;
float* x9050 = x9034+x9049;
float* x9051 = x9046+x9049;
int32_t x9052 = x9048 * 36864;
float* x9053 = x9047+x9052;
for(int x9054=0; x9054 < 2304; x9054++) {
int32_t x9055 = x9054 / 9;
int32_t x9059 = x9055 * 3;
int32_t x9060 = x9059 * 3;
int32_t x9061 = x9060 * 4;
int32_t x9062 = x9061 * 4;
int32_t x9056 = x9054 % 9;
int32_t x9057 = x9056 / 3;
int32_t x9063 = x9057 * 3;
int32_t x9064 = x9063 * 4;
int32_t x9065 = x9064 * 4;
int32_t x9066 = x9062 + x9065;
int32_t x9058 = x9056 % 3;
int32_t x9067 = x9058 * 4;
int32_t x9068 = x9067 * 4;
int32_t x9069 = x9066 + x9068;
float* x9070 = x9053+x9069;
int32_t x9071 = x9055 * 4;
int32_t x9072 = x9071 * 4;
float* x9073 = x9050+x9072;
int32_t x9085 = 1 - x9058;
bool x9086 = x9085 > 0;
int32_t x9087;
if (x9086) {
x9087 = x9085;
} else {
x9087 = 0;
}
int32_t x9088 = 3 - x9058;
int32_t x9089 = x9088 - 1;
int32_t x9090 = 1 - x9089;
bool x9091 = x9090 > 0;
int32_t x9092;
if (x9091) {
x9092 = x9090;
} else {
x9092 = 0;
}
int32_t x9093 = 4 - x9092;
int32_t x9094 = x9093 - x9087;
bool x9095 = x9094 <= 0;
bool x9099 = x9087 > 0;
int32_t x9084 = -1 + x9058;
bool x9112 = x9092 > 0;
for(int x9074=0; x9074 < 4; x9074++) {
int32_t x9075 = x9074 - 1;
int32_t x9076 = x9075 + x9057;
bool x9077 = x9076 < 0;
bool x9078 = x9076 >= 4;
bool x9079 = x9077 || x9078;
if (x9079) {
int32_t x9080 = x9074 * 4;
float* x9081 = x9070+x9080;
memset(x9081, 0, 4 * 4);;
} else {
if (x9095) {
int32_t x9080 = x9074 * 4;
float* x9096 = x9070+x9080;
memset(x9096, 0, 4 * 4);;
} else {
int32_t x9080 = x9074 * 4;
if (x9099) {
float* x9100 = x9070+x9080;
memset(x9100, 0, 4 * x9087);;
} else {
}
// may have segfault here
int32_t x9105 = x9080 + x9087;
float* x9106 = x9070+x9105;
int32_t x9107 = x9076 * 4;
int32_t x9108 = x9107 + x9084;
int32_t x9109 = x9108 + x9087;
float* x9110 = x9073+x9109;
memcpy(x9106, x9110, 4 * x9094);;
if (x9112) {
int32_t x9113 = x9080 + 4;
int32_t x9114 = x9113 - x9092;
float* x9115 = x9070+x9114;
memset(x9115, 0, 4 * x9092);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,16,2304,1,x30,2304,x9053,16,1,x9051,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x9131 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x9132 = 0;
int32_t x9133 = 0;
int32_t x9134 = 0;
for(int x9135=0; x9135 < 64; x9135++) {
int32_t x9136 = x9133;
int32_t x9137 = x9134;
int32_t x9138 = x9132;
int32_t x9139 = x9138;
int32_t x9140 = x9136;
int32_t x9141 = x9137;
for(int x9142=0; x9142 < 256; x9142++) {
int32_t x9143 = x9140;
int32_t x9144 = x9141;
int32_t x9145 = x9139;
int32_t x9146 = x9145;
int32_t x9147 = x9143;
int32_t x9148 = x9144;
for(int x9149=0; x9149 < 4; x9149++) {
int32_t x9150 = x9147;
int32_t x9151 = x9148;
int32_t x9152 = x9146;
int32_t x9153 = x9152;
int32_t x9154 = x9150;
int32_t x9155 = x9151;
for(int x9156=0; x9156 < 4; x9156++) {
int32_t x9157 = x9153;
int32_t x9158 = x9154;
float x9159 = x9046[x9158];
int32_t x9160 = x9155;
float x9161 = x199[x9160];
float x9162 = x9159 - x9161;
x9131[x9157] = x9162;
x9153 += 1;
x9154 += 1;

}
x9146 += 4;
x9147 += 4;

}
x9139 += 16;
x9140 += 16;
x9141 += 1;

}
x9132 += 4096;
x9133 += 4096;

}
float* x9181 = (float*)myMalloc(256 * sizeof(float));;
for(int x9182=0; x9182 < 256; x9182++) {
float x9183 = x236[x9182];
float x9184 = x9183 + 1.0E-5f;
x9181[x9182] = x9184;

}
float* x9188 = (float*)myMalloc(256 * sizeof(float));;
for(int x9189=0; x9189 < 256; x9189++) {
float x9190 = x9181[x9189];
double x9191 = (double)x9190;
double x9192 = sqrt(x9191);
float x9193 = (float)x9192;
x9188[x9189] = x9193;

}
// resize to WrappedArray(-1, 1, 1)
float* x9198 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x9199 = 0;
int32_t x9200 = 0;
int32_t x9201 = 0;
for(int x9202=0; x9202 < 64; x9202++) {
int32_t x9203 = x9200;
int32_t x9204 = x9201;
int32_t x9205 = x9199;
int32_t x9206 = x9205;
int32_t x9207 = x9203;
int32_t x9208 = x9204;
for(int x9209=0; x9209 < 256; x9209++) {
int32_t x9210 = x9207;
int32_t x9211 = x9208;
int32_t x9212 = x9206;
int32_t x9213 = x9212;
int32_t x9214 = x9210;
int32_t x9215 = x9211;
for(int x9216=0; x9216 < 4; x9216++) {
int32_t x9217 = x9214;
int32_t x9218 = x9215;
int32_t x9219 = x9213;
int32_t x9220 = x9219;
int32_t x9221 = x9217;
int32_t x9222 = x9218;
for(int x9223=0; x9223 < 4; x9223++) {
int32_t x9224 = x9220;
int32_t x9225 = x9221;
float x9226 = x9131[x9225];
int32_t x9227 = x9222;
float x9228 = x9188[x9227];
float x9229 = x9226 / x9228;
x9198[x9224] = x9229;
x9220 += 1;
x9221 += 1;

}
x9213 += 4;
x9214 += 4;

}
x9206 += 16;
x9207 += 16;
x9208 += 1;

}
x9199 += 4096;
x9200 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x9249 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x9250 = 0;
int32_t x9251 = 0;
int32_t x9252 = 0;
for(int x9253=0; x9253 < 64; x9253++) {
int32_t x9254 = x9251;
int32_t x9255 = x9252;
int32_t x9256 = x9250;
int32_t x9257 = x9256;
int32_t x9258 = x9254;
int32_t x9259 = x9255;
for(int x9260=0; x9260 < 256; x9260++) {
int32_t x9261 = x9258;
int32_t x9262 = x9259;
int32_t x9263 = x9257;
int32_t x9264 = x9263;
int32_t x9265 = x9261;
int32_t x9266 = x9262;
for(int x9267=0; x9267 < 4; x9267++) {
int32_t x9268 = x9265;
int32_t x9269 = x9266;
int32_t x9270 = x9264;
int32_t x9271 = x9270;
int32_t x9272 = x9268;
int32_t x9273 = x9269;
for(int x9274=0; x9274 < 4; x9274++) {
int32_t x9275 = x9271;
int32_t x9276 = x9272;
float x9277 = x9198[x9276];
int32_t x9278 = x9273;
float x9279 = x270[x9278];
float x9280 = x9277 * x9279;
x9249[x9275] = x9280;
x9271 += 1;
x9272 += 1;

}
x9264 += 4;
x9265 += 4;

}
x9257 += 16;
x9258 += 16;
x9259 += 1;

}
x9250 += 4096;
x9251 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x9300 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x9301 = 0;
int32_t x9302 = 0;
int32_t x9303 = 0;
for(int x9304=0; x9304 < 64; x9304++) {
int32_t x9305 = x9302;
int32_t x9306 = x9303;
int32_t x9307 = x9301;
int32_t x9308 = x9307;
int32_t x9309 = x9305;
int32_t x9310 = x9306;
for(int x9311=0; x9311 < 256; x9311++) {
int32_t x9312 = x9309;
int32_t x9313 = x9310;
int32_t x9314 = x9308;
int32_t x9315 = x9314;
int32_t x9316 = x9312;
int32_t x9317 = x9313;
for(int x9318=0; x9318 < 4; x9318++) {
int32_t x9319 = x9316;
int32_t x9320 = x9317;
int32_t x9321 = x9315;
int32_t x9322 = x9321;
int32_t x9323 = x9319;
int32_t x9324 = x9320;
for(int x9325=0; x9325 < 4; x9325++) {
int32_t x9326 = x9322;
int32_t x9327 = x9323;
float x9328 = x9249[x9327];
int32_t x9329 = x9324;
float x9330 = x95[x9329];
float x9331 = x9328 + x9330;
x9300[x9326] = x9331;
x9322 += 1;
x9323 += 1;

}
x9315 += 4;
x9316 += 4;

}
x9308 += 16;
x9309 += 16;
x9310 += 1;

}
x9301 += 4096;
x9302 += 4096;

}
float* x9350 = (float*)myMalloc(262144 * sizeof(float));;
for(int x9351=0; x9351 < 262144; x9351++) {
float x9352 = x9300[x9351];
bool x9353 = x9352 < 0.0f;
if (x9353) {
x9350[x9351] = 0.0f;
} else {
float x9356 = x9300[x9351];
x9350[x9351] = x9356;
}

}
float* x9362 = (float*)myMalloc(1048576 * sizeof(float));;
float* x9363 = (float*)myMalloc(262144 * sizeof(float));;
for(int x9364=0; x9364 < 64; x9364++) {
int32_t x9365 = x9364 * 4096;
float* x9366 = x9350+x9365;
int32_t x9367 = x9364 * 16384;
float* x9368 = x9362+x9367;
float* x9369 = x9363+x9365;
for(int x9370=0; x9370 < 256; x9370++) {
int32_t x9371 = x9370 / 1;
int32_t x9375 = x9371 * 4;
int32_t x9376 = x9375 * 4;
int32_t x9372 = x9370 % 1;
int32_t x9373 = x9372 / 1;
int32_t x9377 = x9373 * 4;
int32_t x9378 = x9377 * 4;
int32_t x9379 = x9376 + x9378;
int32_t x9374 = x9372 % 1;
int32_t x9380 = x9374 * 4;
int32_t x9381 = x9380 * 4;
int32_t x9382 = x9379 + x9381;
float* x9383 = x9369+x9382;
float* x9384 = x9366+x9376;
for(int x9385=0; x9385 < 4; x9385++) {
int32_t x9387 = x9385 * 4;
float* x9388 = x9383+x9387;
int32_t x9386 = x9385 + x9373;
int32_t x9389 = x9386 * 4;
int32_t x9390 = x9389 + x9374;
float* x9391 = x9384+x9390;
memcpy(x9388, x9391, 4 * 4);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,16,256,1,x55,256,x9369,16,1,x9368,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x9401 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x9402 = 0;
int32_t x9403 = 0;
int32_t x9404 = 0;
for(int x9405=0; x9405 < 64; x9405++) {
int32_t x9406 = x9403;
int32_t x9407 = x9404;
int32_t x9408 = x9402;
int32_t x9409 = x9408;
int32_t x9410 = x9406;
int32_t x9411 = x9407;
for(int x9412=0; x9412 < 1024; x9412++) {
int32_t x9413 = x9410;
int32_t x9414 = x9411;
int32_t x9415 = x9409;
int32_t x9416 = x9415;
int32_t x9417 = x9413;
int32_t x9418 = x9414;
for(int x9419=0; x9419 < 4; x9419++) {
int32_t x9420 = x9417;
int32_t x9421 = x9418;
int32_t x9422 = x9416;
int32_t x9423 = x9422;
int32_t x9424 = x9420;
int32_t x9425 = x9421;
for(int x9426=0; x9426 < 4; x9426++) {
int32_t x9427 = x9423;
int32_t x9428 = x9424;
float x9429 = x9362[x9428];
int32_t x9430 = x9425;
float x9431 = x181[x9430];
float x9432 = x9429 - x9431;
x9401[x9427] = x9432;
x9423 += 1;
x9424 += 1;

}
x9416 += 4;
x9417 += 4;

}
x9409 += 16;
x9410 += 16;
x9411 += 1;

}
x9402 += 16384;
x9403 += 16384;

}
float* x9451 = (float*)myMalloc(1024 * sizeof(float));;
for(int x9452=0; x9452 < 1024; x9452++) {
float x9453 = x142[x9452];
float x9454 = x9453 + 1.0E-5f;
x9451[x9452] = x9454;

}
float* x9458 = (float*)myMalloc(1024 * sizeof(float));;
for(int x9459=0; x9459 < 1024; x9459++) {
float x9460 = x9451[x9459];
double x9461 = (double)x9460;
double x9462 = sqrt(x9461);
float x9463 = (float)x9462;
x9458[x9459] = x9463;

}
// resize to WrappedArray(-1, 1, 1)
float* x9468 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x9469 = 0;
int32_t x9470 = 0;
int32_t x9471 = 0;
for(int x9472=0; x9472 < 64; x9472++) {
int32_t x9473 = x9470;
int32_t x9474 = x9471;
int32_t x9475 = x9469;
int32_t x9476 = x9475;
int32_t x9477 = x9473;
int32_t x9478 = x9474;
for(int x9479=0; x9479 < 1024; x9479++) {
int32_t x9480 = x9477;
int32_t x9481 = x9478;
int32_t x9482 = x9476;
int32_t x9483 = x9482;
int32_t x9484 = x9480;
int32_t x9485 = x9481;
for(int x9486=0; x9486 < 4; x9486++) {
int32_t x9487 = x9484;
int32_t x9488 = x9485;
int32_t x9489 = x9483;
int32_t x9490 = x9489;
int32_t x9491 = x9487;
int32_t x9492 = x9488;
for(int x9493=0; x9493 < 4; x9493++) {
int32_t x9494 = x9490;
int32_t x9495 = x9491;
float x9496 = x9401[x9495];
int32_t x9497 = x9492;
float x9498 = x9458[x9497];
float x9499 = x9496 / x9498;
x9468[x9494] = x9499;
x9490 += 1;
x9491 += 1;

}
x9483 += 4;
x9484 += 4;

}
x9476 += 16;
x9477 += 16;
x9478 += 1;

}
x9469 += 16384;
x9470 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x9519 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x9520 = 0;
int32_t x9521 = 0;
int32_t x9522 = 0;
for(int x9523=0; x9523 < 64; x9523++) {
int32_t x9524 = x9521;
int32_t x9525 = x9522;
int32_t x9526 = x9520;
int32_t x9527 = x9526;
int32_t x9528 = x9524;
int32_t x9529 = x9525;
for(int x9530=0; x9530 < 1024; x9530++) {
int32_t x9531 = x9528;
int32_t x9532 = x9529;
int32_t x9533 = x9527;
int32_t x9534 = x9533;
int32_t x9535 = x9531;
int32_t x9536 = x9532;
for(int x9537=0; x9537 < 4; x9537++) {
int32_t x9538 = x9535;
int32_t x9539 = x9536;
int32_t x9540 = x9534;
int32_t x9541 = x9540;
int32_t x9542 = x9538;
int32_t x9543 = x9539;
for(int x9544=0; x9544 < 4; x9544++) {
int32_t x9545 = x9541;
int32_t x9546 = x9542;
float x9547 = x9468[x9546];
int32_t x9548 = x9543;
float x9549 = x19[x9548];
float x9550 = x9547 * x9549;
x9519[x9545] = x9550;
x9541 += 1;
x9542 += 1;

}
x9534 += 4;
x9535 += 4;

}
x9527 += 16;
x9528 += 16;
x9529 += 1;

}
x9520 += 16384;
x9521 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x9570 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x9571 = 0;
int32_t x9572 = 0;
int32_t x9573 = 0;
for(int x9574=0; x9574 < 64; x9574++) {
int32_t x9575 = x9572;
int32_t x9576 = x9573;
int32_t x9577 = x9571;
int32_t x9578 = x9577;
int32_t x9579 = x9575;
int32_t x9580 = x9576;
for(int x9581=0; x9581 < 1024; x9581++) {
int32_t x9582 = x9579;
int32_t x9583 = x9580;
int32_t x9584 = x9578;
int32_t x9585 = x9584;
int32_t x9586 = x9582;
int32_t x9587 = x9583;
for(int x9588=0; x9588 < 4; x9588++) {
int32_t x9589 = x9586;
int32_t x9590 = x9587;
int32_t x9591 = x9585;
int32_t x9592 = x9591;
int32_t x9593 = x9589;
int32_t x9594 = x9590;
for(int x9595=0; x9595 < 4; x9595++) {
int32_t x9596 = x9592;
int32_t x9597 = x9593;
float x9598 = x9519[x9597];
int32_t x9599 = x9594;
float x9600 = x231[x9599];
float x9601 = x9598 + x9600;
x9570[x9596] = x9601;
x9592 += 1;
x9593 += 1;

}
x9585 += 4;
x9586 += 4;

}
x9578 += 16;
x9579 += 16;
x9580 += 1;

}
x9571 += 16384;
x9572 += 16384;

}
int32_t x9620 = 0;
int32_t x9621 = 0;
int32_t x9622 = 0;
for(int x9623=0; x9623 < 64; x9623++) {
int32_t x9624 = x9621;
int32_t x9625 = x9622;
int32_t x9626 = x9620;
int32_t x9627 = x9626;
int32_t x9628 = x9624;
int32_t x9629 = x9625;
for(int x9630=0; x9630 < 1024; x9630++) {
int32_t x9631 = x9628;
int32_t x9632 = x9629;
int32_t x9633 = x9627;
int32_t x9634 = x9633;
int32_t x9635 = x9631;
int32_t x9636 = x9632;
for(int x9637=0; x9637 < 4; x9637++) {
int32_t x9638 = x9635;
int32_t x9639 = x9636;
int32_t x9640 = x9634;
int32_t x9641 = x9640;
int32_t x9642 = x9638;
int32_t x9643 = x9639;
for(int x9644=0; x9644 < 4; x9644++) {
int32_t x9645 = x9642;
float x9646 = x9570[x9645];
int32_t x9647 = x9643;
float x9648 = x8764[x9647];
float x9649 = x9646 + x9648;
x9570[x9645] = x9649;
x9641 += 1;
x9642 += 1;
x9643 += 1;

}
x9634 += 4;
x9635 += 4;
x9636 += 4;

}
x9627 += 16;
x9628 += 16;
x9629 += 16;

}
x9620 += 16384;
x9621 += 16384;
x9622 += 16384;

}
float* x9671 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x9672=0; x9672 < 1048576; x9672++) {
float x9673 = x9570[x9672];
bool x9674 = x9673 < 0.0f;
if (x9674) {
x9671[x9672] = 0.0f;
} else {
float x9677 = x9570[x9672];
x9671[x9672] = x9677;
}

}
float* x9683 = (float*)myMalloc(262144 * sizeof(float));;
float* x9684 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x9685=0; x9685 < 64; x9685++) {
int32_t x9686 = x9685 * 16384;
float* x9687 = x9671+x9686;
int32_t x9688 = x9685 * 4096;
float* x9689 = x9683+x9688;
float* x9690 = x9684+x9686;
for(int x9691=0; x9691 < 1024; x9691++) {
int32_t x9692 = x9691 / 1;
int32_t x9696 = x9692 * 4;
int32_t x9697 = x9696 * 4;
int32_t x9693 = x9691 % 1;
int32_t x9694 = x9693 / 1;
int32_t x9698 = x9694 * 4;
int32_t x9699 = x9698 * 4;
int32_t x9700 = x9697 + x9699;
int32_t x9695 = x9693 % 1;
int32_t x9701 = x9695 * 4;
int32_t x9702 = x9701 * 4;
int32_t x9703 = x9700 + x9702;
float* x9704 = x9690+x9703;
float* x9705 = x9687+x9697;
for(int x9706=0; x9706 < 4; x9706++) {
int32_t x9708 = x9706 * 4;
float* x9709 = x9704+x9708;
int32_t x9707 = x9706 + x9694;
int32_t x9710 = x9707 * 4;
int32_t x9711 = x9710 + x9695;
float* x9712 = x9705+x9711;
memcpy(x9709, x9712, 4 * 4);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,16,1024,1,x217,1024,x9690,16,1,x9689,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x9722 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x9723 = 0;
int32_t x9724 = 0;
int32_t x9725 = 0;
for(int x9726=0; x9726 < 64; x9726++) {
int32_t x9727 = x9724;
int32_t x9728 = x9725;
int32_t x9729 = x9723;
int32_t x9730 = x9729;
int32_t x9731 = x9727;
int32_t x9732 = x9728;
for(int x9733=0; x9733 < 256; x9733++) {
int32_t x9734 = x9731;
int32_t x9735 = x9732;
int32_t x9736 = x9730;
int32_t x9737 = x9736;
int32_t x9738 = x9734;
int32_t x9739 = x9735;
for(int x9740=0; x9740 < 4; x9740++) {
int32_t x9741 = x9738;
int32_t x9742 = x9739;
int32_t x9743 = x9737;
int32_t x9744 = x9743;
int32_t x9745 = x9741;
int32_t x9746 = x9742;
for(int x9747=0; x9747 < 4; x9747++) {
int32_t x9748 = x9744;
int32_t x9749 = x9745;
float x9750 = x9683[x9749];
int32_t x9751 = x9746;
float x9752 = x177[x9751];
float x9753 = x9750 - x9752;
x9722[x9748] = x9753;
x9744 += 1;
x9745 += 1;

}
x9737 += 4;
x9738 += 4;

}
x9730 += 16;
x9731 += 16;
x9732 += 1;

}
x9723 += 4096;
x9724 += 4096;

}
float* x9772 = (float*)myMalloc(256 * sizeof(float));;
for(int x9773=0; x9773 < 256; x9773++) {
float x9774 = x173[x9773];
float x9775 = x9774 + 1.0E-5f;
x9772[x9773] = x9775;

}
float* x9779 = (float*)myMalloc(256 * sizeof(float));;
for(int x9780=0; x9780 < 256; x9780++) {
float x9781 = x9772[x9780];
double x9782 = (double)x9781;
double x9783 = sqrt(x9782);
float x9784 = (float)x9783;
x9779[x9780] = x9784;

}
// resize to WrappedArray(-1, 1, 1)
float* x9789 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x9790 = 0;
int32_t x9791 = 0;
int32_t x9792 = 0;
for(int x9793=0; x9793 < 64; x9793++) {
int32_t x9794 = x9791;
int32_t x9795 = x9792;
int32_t x9796 = x9790;
int32_t x9797 = x9796;
int32_t x9798 = x9794;
int32_t x9799 = x9795;
for(int x9800=0; x9800 < 256; x9800++) {
int32_t x9801 = x9798;
int32_t x9802 = x9799;
int32_t x9803 = x9797;
int32_t x9804 = x9803;
int32_t x9805 = x9801;
int32_t x9806 = x9802;
for(int x9807=0; x9807 < 4; x9807++) {
int32_t x9808 = x9805;
int32_t x9809 = x9806;
int32_t x9810 = x9804;
int32_t x9811 = x9810;
int32_t x9812 = x9808;
int32_t x9813 = x9809;
for(int x9814=0; x9814 < 4; x9814++) {
int32_t x9815 = x9811;
int32_t x9816 = x9812;
float x9817 = x9722[x9816];
int32_t x9818 = x9813;
float x9819 = x9779[x9818];
float x9820 = x9817 / x9819;
x9789[x9815] = x9820;
x9811 += 1;
x9812 += 1;

}
x9804 += 4;
x9805 += 4;

}
x9797 += 16;
x9798 += 16;
x9799 += 1;

}
x9790 += 4096;
x9791 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x9840 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x9841 = 0;
int32_t x9842 = 0;
int32_t x9843 = 0;
for(int x9844=0; x9844 < 64; x9844++) {
int32_t x9845 = x9842;
int32_t x9846 = x9843;
int32_t x9847 = x9841;
int32_t x9848 = x9847;
int32_t x9849 = x9845;
int32_t x9850 = x9846;
for(int x9851=0; x9851 < 256; x9851++) {
int32_t x9852 = x9849;
int32_t x9853 = x9850;
int32_t x9854 = x9848;
int32_t x9855 = x9854;
int32_t x9856 = x9852;
int32_t x9857 = x9853;
for(int x9858=0; x9858 < 4; x9858++) {
int32_t x9859 = x9856;
int32_t x9860 = x9857;
int32_t x9861 = x9855;
int32_t x9862 = x9861;
int32_t x9863 = x9859;
int32_t x9864 = x9860;
for(int x9865=0; x9865 < 4; x9865++) {
int32_t x9866 = x9862;
int32_t x9867 = x9863;
float x9868 = x9789[x9867];
int32_t x9869 = x9864;
float x9870 = x128[x9869];
float x9871 = x9868 * x9870;
x9840[x9866] = x9871;
x9862 += 1;
x9863 += 1;

}
x9855 += 4;
x9856 += 4;

}
x9848 += 16;
x9849 += 16;
x9850 += 1;

}
x9841 += 4096;
x9842 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x9891 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x9892 = 0;
int32_t x9893 = 0;
int32_t x9894 = 0;
for(int x9895=0; x9895 < 64; x9895++) {
int32_t x9896 = x9893;
int32_t x9897 = x9894;
int32_t x9898 = x9892;
int32_t x9899 = x9898;
int32_t x9900 = x9896;
int32_t x9901 = x9897;
for(int x9902=0; x9902 < 256; x9902++) {
int32_t x9903 = x9900;
int32_t x9904 = x9901;
int32_t x9905 = x9899;
int32_t x9906 = x9905;
int32_t x9907 = x9903;
int32_t x9908 = x9904;
for(int x9909=0; x9909 < 4; x9909++) {
int32_t x9910 = x9907;
int32_t x9911 = x9908;
int32_t x9912 = x9906;
int32_t x9913 = x9912;
int32_t x9914 = x9910;
int32_t x9915 = x9911;
for(int x9916=0; x9916 < 4; x9916++) {
int32_t x9917 = x9913;
int32_t x9918 = x9914;
float x9919 = x9840[x9918];
int32_t x9920 = x9915;
float x9921 = x196[x9920];
float x9922 = x9919 + x9921;
x9891[x9917] = x9922;
x9913 += 1;
x9914 += 1;

}
x9906 += 4;
x9907 += 4;

}
x9899 += 16;
x9900 += 16;
x9901 += 1;

}
x9892 += 4096;
x9893 += 4096;

}
float* x9941 = (float*)myMalloc(262144 * sizeof(float));;
for(int x9942=0; x9942 < 262144; x9942++) {
float x9943 = x9891[x9942];
bool x9944 = x9943 < 0.0f;
if (x9944) {
x9941[x9942] = 0.0f;
} else {
float x9947 = x9891[x9942];
x9941[x9942] = x9947;
}

}
float* x9953 = (float*)myMalloc(262144 * sizeof(float));;
float* x9954 = (float*)myMalloc(2359296 * sizeof(float));;
for(int x9955=0; x9955 < 64; x9955++) {
int32_t x9956 = x9955 * 4096;
float* x9957 = x9941+x9956;
float* x9958 = x9953+x9956;
int32_t x9959 = x9955 * 36864;
float* x9960 = x9954+x9959;
for(int x9961=0; x9961 < 2304; x9961++) {
int32_t x9962 = x9961 / 9;
int32_t x9966 = x9962 * 3;
int32_t x9967 = x9966 * 3;
int32_t x9968 = x9967 * 4;
int32_t x9969 = x9968 * 4;
int32_t x9963 = x9961 % 9;
int32_t x9964 = x9963 / 3;
int32_t x9970 = x9964 * 3;
int32_t x9971 = x9970 * 4;
int32_t x9972 = x9971 * 4;
int32_t x9973 = x9969 + x9972;
int32_t x9965 = x9963 % 3;
int32_t x9974 = x9965 * 4;
int32_t x9975 = x9974 * 4;
int32_t x9976 = x9973 + x9975;
float* x9977 = x9960+x9976;
int32_t x9978 = x9962 * 4;
int32_t x9979 = x9978 * 4;
float* x9980 = x9957+x9979;
int32_t x9992 = 1 - x9965;
bool x9993 = x9992 > 0;
int32_t x9994;
if (x9993) {
x9994 = x9992;
} else {
x9994 = 0;
}
int32_t x9995 = 3 - x9965;
int32_t x9996 = x9995 - 1;
int32_t x9997 = 1 - x9996;
bool x9998 = x9997 > 0;
int32_t x9999;
if (x9998) {
x9999 = x9997;
} else {
x9999 = 0;
}
int32_t x10000 = 4 - x9999;
int32_t x10001 = x10000 - x9994;
bool x10002 = x10001 <= 0;
bool x10006 = x9994 > 0;
int32_t x9991 = -1 + x9965;
bool x10019 = x9999 > 0;
for(int x9981=0; x9981 < 4; x9981++) {
int32_t x9982 = x9981 - 1;
int32_t x9983 = x9982 + x9964;
bool x9984 = x9983 < 0;
bool x9985 = x9983 >= 4;
bool x9986 = x9984 || x9985;
if (x9986) {
int32_t x9987 = x9981 * 4;
float* x9988 = x9977+x9987;
memset(x9988, 0, 4 * 4);;
} else {
if (x10002) {
int32_t x9987 = x9981 * 4;
float* x10003 = x9977+x9987;
memset(x10003, 0, 4 * 4);;
} else {
int32_t x9987 = x9981 * 4;
if (x10006) {
float* x10007 = x9977+x9987;
memset(x10007, 0, 4 * x9994);;
} else {
}
// may have segfault here
int32_t x10012 = x9987 + x9994;
float* x10013 = x9977+x10012;
int32_t x10014 = x9983 * 4;
int32_t x10015 = x10014 + x9991;
int32_t x10016 = x10015 + x9994;
float* x10017 = x9980+x10016;
memcpy(x10013, x10017, 4 * x10001);;
if (x10019) {
int32_t x10020 = x9987 + 4;
int32_t x10021 = x10020 - x9999;
float* x10022 = x9977+x10021;
memset(x10022, 0, 4 * x9999);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,16,2304,1,x13,2304,x9960,16,1,x9958,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x10038 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x10039 = 0;
int32_t x10040 = 0;
int32_t x10041 = 0;
for(int x10042=0; x10042 < 64; x10042++) {
int32_t x10043 = x10040;
int32_t x10044 = x10041;
int32_t x10045 = x10039;
int32_t x10046 = x10045;
int32_t x10047 = x10043;
int32_t x10048 = x10044;
for(int x10049=0; x10049 < 256; x10049++) {
int32_t x10050 = x10047;
int32_t x10051 = x10048;
int32_t x10052 = x10046;
int32_t x10053 = x10052;
int32_t x10054 = x10050;
int32_t x10055 = x10051;
for(int x10056=0; x10056 < 4; x10056++) {
int32_t x10057 = x10054;
int32_t x10058 = x10055;
int32_t x10059 = x10053;
int32_t x10060 = x10059;
int32_t x10061 = x10057;
int32_t x10062 = x10058;
for(int x10063=0; x10063 < 4; x10063++) {
int32_t x10064 = x10060;
int32_t x10065 = x10061;
float x10066 = x9953[x10065];
int32_t x10067 = x10062;
float x10068 = x123[x10067];
float x10069 = x10066 - x10068;
x10038[x10064] = x10069;
x10060 += 1;
x10061 += 1;

}
x10053 += 4;
x10054 += 4;

}
x10046 += 16;
x10047 += 16;
x10048 += 1;

}
x10039 += 4096;
x10040 += 4096;

}
float* x10088 = (float*)myMalloc(256 * sizeof(float));;
for(int x10089=0; x10089 < 256; x10089++) {
float x10090 = x62[x10089];
float x10091 = x10090 + 1.0E-5f;
x10088[x10089] = x10091;

}
float* x10095 = (float*)myMalloc(256 * sizeof(float));;
for(int x10096=0; x10096 < 256; x10096++) {
float x10097 = x10088[x10096];
double x10098 = (double)x10097;
double x10099 = sqrt(x10098);
float x10100 = (float)x10099;
x10095[x10096] = x10100;

}
// resize to WrappedArray(-1, 1, 1)
float* x10105 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x10106 = 0;
int32_t x10107 = 0;
int32_t x10108 = 0;
for(int x10109=0; x10109 < 64; x10109++) {
int32_t x10110 = x10107;
int32_t x10111 = x10108;
int32_t x10112 = x10106;
int32_t x10113 = x10112;
int32_t x10114 = x10110;
int32_t x10115 = x10111;
for(int x10116=0; x10116 < 256; x10116++) {
int32_t x10117 = x10114;
int32_t x10118 = x10115;
int32_t x10119 = x10113;
int32_t x10120 = x10119;
int32_t x10121 = x10117;
int32_t x10122 = x10118;
for(int x10123=0; x10123 < 4; x10123++) {
int32_t x10124 = x10121;
int32_t x10125 = x10122;
int32_t x10126 = x10120;
int32_t x10127 = x10126;
int32_t x10128 = x10124;
int32_t x10129 = x10125;
for(int x10130=0; x10130 < 4; x10130++) {
int32_t x10131 = x10127;
int32_t x10132 = x10128;
float x10133 = x10038[x10132];
int32_t x10134 = x10129;
float x10135 = x10095[x10134];
float x10136 = x10133 / x10135;
x10105[x10131] = x10136;
x10127 += 1;
x10128 += 1;

}
x10120 += 4;
x10121 += 4;

}
x10113 += 16;
x10114 += 16;
x10115 += 1;

}
x10106 += 4096;
x10107 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x10156 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x10157 = 0;
int32_t x10158 = 0;
int32_t x10159 = 0;
for(int x10160=0; x10160 < 64; x10160++) {
int32_t x10161 = x10158;
int32_t x10162 = x10159;
int32_t x10163 = x10157;
int32_t x10164 = x10163;
int32_t x10165 = x10161;
int32_t x10166 = x10162;
for(int x10167=0; x10167 < 256; x10167++) {
int32_t x10168 = x10165;
int32_t x10169 = x10166;
int32_t x10170 = x10164;
int32_t x10171 = x10170;
int32_t x10172 = x10168;
int32_t x10173 = x10169;
for(int x10174=0; x10174 < 4; x10174++) {
int32_t x10175 = x10172;
int32_t x10176 = x10173;
int32_t x10177 = x10171;
int32_t x10178 = x10177;
int32_t x10179 = x10175;
int32_t x10180 = x10176;
for(int x10181=0; x10181 < 4; x10181++) {
int32_t x10182 = x10178;
int32_t x10183 = x10179;
float x10184 = x10105[x10183];
int32_t x10185 = x10180;
float x10186 = x227[x10185];
float x10187 = x10184 * x10186;
x10156[x10182] = x10187;
x10178 += 1;
x10179 += 1;

}
x10171 += 4;
x10172 += 4;

}
x10164 += 16;
x10165 += 16;
x10166 += 1;

}
x10157 += 4096;
x10158 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x10207 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x10208 = 0;
int32_t x10209 = 0;
int32_t x10210 = 0;
for(int x10211=0; x10211 < 64; x10211++) {
int32_t x10212 = x10209;
int32_t x10213 = x10210;
int32_t x10214 = x10208;
int32_t x10215 = x10214;
int32_t x10216 = x10212;
int32_t x10217 = x10213;
for(int x10218=0; x10218 < 256; x10218++) {
int32_t x10219 = x10216;
int32_t x10220 = x10217;
int32_t x10221 = x10215;
int32_t x10222 = x10221;
int32_t x10223 = x10219;
int32_t x10224 = x10220;
for(int x10225=0; x10225 < 4; x10225++) {
int32_t x10226 = x10223;
int32_t x10227 = x10224;
int32_t x10228 = x10222;
int32_t x10229 = x10228;
int32_t x10230 = x10226;
int32_t x10231 = x10227;
for(int x10232=0; x10232 < 4; x10232++) {
int32_t x10233 = x10229;
int32_t x10234 = x10230;
float x10235 = x10156[x10234];
int32_t x10236 = x10231;
float x10237 = x191[x10236];
float x10238 = x10235 + x10237;
x10207[x10233] = x10238;
x10229 += 1;
x10230 += 1;

}
x10222 += 4;
x10223 += 4;

}
x10215 += 16;
x10216 += 16;
x10217 += 1;

}
x10208 += 4096;
x10209 += 4096;

}
float* x10257 = (float*)myMalloc(262144 * sizeof(float));;
for(int x10258=0; x10258 < 262144; x10258++) {
float x10259 = x10207[x10258];
bool x10260 = x10259 < 0.0f;
if (x10260) {
x10257[x10258] = 0.0f;
} else {
float x10263 = x10207[x10258];
x10257[x10258] = x10263;
}

}
float* x10269 = (float*)myMalloc(1048576 * sizeof(float));;
float* x10270 = (float*)myMalloc(262144 * sizeof(float));;
for(int x10271=0; x10271 < 64; x10271++) {
int32_t x10272 = x10271 * 4096;
float* x10273 = x10257+x10272;
int32_t x10274 = x10271 * 16384;
float* x10275 = x10269+x10274;
float* x10276 = x10270+x10272;
for(int x10277=0; x10277 < 256; x10277++) {
int32_t x10278 = x10277 / 1;
int32_t x10282 = x10278 * 4;
int32_t x10283 = x10282 * 4;
int32_t x10279 = x10277 % 1;
int32_t x10280 = x10279 / 1;
int32_t x10284 = x10280 * 4;
int32_t x10285 = x10284 * 4;
int32_t x10286 = x10283 + x10285;
int32_t x10281 = x10279 % 1;
int32_t x10287 = x10281 * 4;
int32_t x10288 = x10287 * 4;
int32_t x10289 = x10286 + x10288;
float* x10290 = x10276+x10289;
float* x10291 = x10273+x10283;
for(int x10292=0; x10292 < 4; x10292++) {
int32_t x10294 = x10292 * 4;
float* x10295 = x10290+x10294;
int32_t x10293 = x10292 + x10280;
int32_t x10296 = x10293 * 4;
int32_t x10297 = x10296 + x10281;
float* x10298 = x10291+x10297;
memcpy(x10295, x10298, 4 * 4);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,16,256,1,x115,256,x10276,16,1,x10275,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x10308 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x10309 = 0;
int32_t x10310 = 0;
int32_t x10311 = 0;
for(int x10312=0; x10312 < 64; x10312++) {
int32_t x10313 = x10310;
int32_t x10314 = x10311;
int32_t x10315 = x10309;
int32_t x10316 = x10315;
int32_t x10317 = x10313;
int32_t x10318 = x10314;
for(int x10319=0; x10319 < 1024; x10319++) {
int32_t x10320 = x10317;
int32_t x10321 = x10318;
int32_t x10322 = x10316;
int32_t x10323 = x10322;
int32_t x10324 = x10320;
int32_t x10325 = x10321;
for(int x10326=0; x10326 < 4; x10326++) {
int32_t x10327 = x10324;
int32_t x10328 = x10325;
int32_t x10329 = x10323;
int32_t x10330 = x10329;
int32_t x10331 = x10327;
int32_t x10332 = x10328;
for(int x10333=0; x10333 < 4; x10333++) {
int32_t x10334 = x10330;
int32_t x10335 = x10331;
float x10336 = x10269[x10335];
int32_t x10337 = x10332;
float x10338 = x139[x10337];
float x10339 = x10336 - x10338;
x10308[x10334] = x10339;
x10330 += 1;
x10331 += 1;

}
x10323 += 4;
x10324 += 4;

}
x10316 += 16;
x10317 += 16;
x10318 += 1;

}
x10309 += 16384;
x10310 += 16384;

}
float* x10358 = (float*)myMalloc(1024 * sizeof(float));;
for(int x10359=0; x10359 < 1024; x10359++) {
float x10360 = x187[x10359];
float x10361 = x10360 + 1.0E-5f;
x10358[x10359] = x10361;

}
float* x10365 = (float*)myMalloc(1024 * sizeof(float));;
for(int x10366=0; x10366 < 1024; x10366++) {
float x10367 = x10358[x10366];
double x10368 = (double)x10367;
double x10369 = sqrt(x10368);
float x10370 = (float)x10369;
x10365[x10366] = x10370;

}
// resize to WrappedArray(-1, 1, 1)
float* x10375 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x10376 = 0;
int32_t x10377 = 0;
int32_t x10378 = 0;
for(int x10379=0; x10379 < 64; x10379++) {
int32_t x10380 = x10377;
int32_t x10381 = x10378;
int32_t x10382 = x10376;
int32_t x10383 = x10382;
int32_t x10384 = x10380;
int32_t x10385 = x10381;
for(int x10386=0; x10386 < 1024; x10386++) {
int32_t x10387 = x10384;
int32_t x10388 = x10385;
int32_t x10389 = x10383;
int32_t x10390 = x10389;
int32_t x10391 = x10387;
int32_t x10392 = x10388;
for(int x10393=0; x10393 < 4; x10393++) {
int32_t x10394 = x10391;
int32_t x10395 = x10392;
int32_t x10396 = x10390;
int32_t x10397 = x10396;
int32_t x10398 = x10394;
int32_t x10399 = x10395;
for(int x10400=0; x10400 < 4; x10400++) {
int32_t x10401 = x10397;
int32_t x10402 = x10398;
float x10403 = x10308[x10402];
int32_t x10404 = x10399;
float x10405 = x10365[x10404];
float x10406 = x10403 / x10405;
x10375[x10401] = x10406;
x10397 += 1;
x10398 += 1;

}
x10390 += 4;
x10391 += 4;

}
x10383 += 16;
x10384 += 16;
x10385 += 1;

}
x10376 += 16384;
x10377 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x10426 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x10427 = 0;
int32_t x10428 = 0;
int32_t x10429 = 0;
for(int x10430=0; x10430 < 64; x10430++) {
int32_t x10431 = x10428;
int32_t x10432 = x10429;
int32_t x10433 = x10427;
int32_t x10434 = x10433;
int32_t x10435 = x10431;
int32_t x10436 = x10432;
for(int x10437=0; x10437 < 1024; x10437++) {
int32_t x10438 = x10435;
int32_t x10439 = x10436;
int32_t x10440 = x10434;
int32_t x10441 = x10440;
int32_t x10442 = x10438;
int32_t x10443 = x10439;
for(int x10444=0; x10444 < 4; x10444++) {
int32_t x10445 = x10442;
int32_t x10446 = x10443;
int32_t x10447 = x10441;
int32_t x10448 = x10447;
int32_t x10449 = x10445;
int32_t x10450 = x10446;
for(int x10451=0; x10451 < 4; x10451++) {
int32_t x10452 = x10448;
int32_t x10453 = x10449;
float x10454 = x10375[x10453];
int32_t x10455 = x10450;
float x10456 = x262[x10455];
float x10457 = x10454 * x10456;
x10426[x10452] = x10457;
x10448 += 1;
x10449 += 1;

}
x10441 += 4;
x10442 += 4;

}
x10434 += 16;
x10435 += 16;
x10436 += 1;

}
x10427 += 16384;
x10428 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x10477 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x10478 = 0;
int32_t x10479 = 0;
int32_t x10480 = 0;
for(int x10481=0; x10481 < 64; x10481++) {
int32_t x10482 = x10479;
int32_t x10483 = x10480;
int32_t x10484 = x10478;
int32_t x10485 = x10484;
int32_t x10486 = x10482;
int32_t x10487 = x10483;
for(int x10488=0; x10488 < 1024; x10488++) {
int32_t x10489 = x10486;
int32_t x10490 = x10487;
int32_t x10491 = x10485;
int32_t x10492 = x10491;
int32_t x10493 = x10489;
int32_t x10494 = x10490;
for(int x10495=0; x10495 < 4; x10495++) {
int32_t x10496 = x10493;
int32_t x10497 = x10494;
int32_t x10498 = x10492;
int32_t x10499 = x10498;
int32_t x10500 = x10496;
int32_t x10501 = x10497;
for(int x10502=0; x10502 < 4; x10502++) {
int32_t x10503 = x10499;
int32_t x10504 = x10500;
float x10505 = x10426[x10504];
int32_t x10506 = x10501;
float x10507 = x56[x10506];
float x10508 = x10505 + x10507;
x10477[x10503] = x10508;
x10499 += 1;
x10500 += 1;

}
x10492 += 4;
x10493 += 4;

}
x10485 += 16;
x10486 += 16;
x10487 += 1;

}
x10478 += 16384;
x10479 += 16384;

}
int32_t x10527 = 0;
int32_t x10528 = 0;
int32_t x10529 = 0;
for(int x10530=0; x10530 < 64; x10530++) {
int32_t x10531 = x10528;
int32_t x10532 = x10529;
int32_t x10533 = x10527;
int32_t x10534 = x10533;
int32_t x10535 = x10531;
int32_t x10536 = x10532;
for(int x10537=0; x10537 < 1024; x10537++) {
int32_t x10538 = x10535;
int32_t x10539 = x10536;
int32_t x10540 = x10534;
int32_t x10541 = x10540;
int32_t x10542 = x10538;
int32_t x10543 = x10539;
for(int x10544=0; x10544 < 4; x10544++) {
int32_t x10545 = x10542;
int32_t x10546 = x10543;
int32_t x10547 = x10541;
int32_t x10548 = x10547;
int32_t x10549 = x10545;
int32_t x10550 = x10546;
for(int x10551=0; x10551 < 4; x10551++) {
int32_t x10552 = x10549;
float x10553 = x10477[x10552];
int32_t x10554 = x10550;
float x10555 = x9671[x10554];
float x10556 = x10553 + x10555;
x10477[x10552] = x10556;
x10548 += 1;
x10549 += 1;
x10550 += 1;

}
x10541 += 4;
x10542 += 4;
x10543 += 4;

}
x10534 += 16;
x10535 += 16;
x10536 += 16;

}
x10527 += 16384;
x10528 += 16384;
x10529 += 16384;

}
float* x10578 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x10579=0; x10579 < 1048576; x10579++) {
float x10580 = x10477[x10579];
bool x10581 = x10580 < 0.0f;
if (x10581) {
x10578[x10579] = 0.0f;
} else {
float x10584 = x10477[x10579];
x10578[x10579] = x10584;
}

}
float* x10590 = (float*)myMalloc(262144 * sizeof(float));;
float* x10591 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x10592=0; x10592 < 64; x10592++) {
int32_t x10593 = x10592 * 16384;
float* x10594 = x10578+x10593;
int32_t x10595 = x10592 * 4096;
float* x10596 = x10590+x10595;
float* x10597 = x10591+x10593;
for(int x10598=0; x10598 < 1024; x10598++) {
int32_t x10599 = x10598 / 1;
int32_t x10603 = x10599 * 4;
int32_t x10604 = x10603 * 4;
int32_t x10600 = x10598 % 1;
int32_t x10601 = x10600 / 1;
int32_t x10605 = x10601 * 4;
int32_t x10606 = x10605 * 4;
int32_t x10607 = x10604 + x10606;
int32_t x10602 = x10600 % 1;
int32_t x10608 = x10602 * 4;
int32_t x10609 = x10608 * 4;
int32_t x10610 = x10607 + x10609;
float* x10611 = x10597+x10610;
float* x10612 = x10594+x10604;
for(int x10613=0; x10613 < 4; x10613++) {
int32_t x10615 = x10613 * 4;
float* x10616 = x10611+x10615;
int32_t x10614 = x10613 + x10601;
int32_t x10617 = x10614 * 4;
int32_t x10618 = x10617 + x10602;
float* x10619 = x10612+x10618;
memcpy(x10616, x10619, 4 * 4);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,16,1024,1,x5,1024,x10597,16,1,x10596,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x10629 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x10630 = 0;
int32_t x10631 = 0;
int32_t x10632 = 0;
for(int x10633=0; x10633 < 64; x10633++) {
int32_t x10634 = x10631;
int32_t x10635 = x10632;
int32_t x10636 = x10630;
int32_t x10637 = x10636;
int32_t x10638 = x10634;
int32_t x10639 = x10635;
for(int x10640=0; x10640 < 256; x10640++) {
int32_t x10641 = x10638;
int32_t x10642 = x10639;
int32_t x10643 = x10637;
int32_t x10644 = x10643;
int32_t x10645 = x10641;
int32_t x10646 = x10642;
for(int x10647=0; x10647 < 4; x10647++) {
int32_t x10648 = x10645;
int32_t x10649 = x10646;
int32_t x10650 = x10644;
int32_t x10651 = x10650;
int32_t x10652 = x10648;
int32_t x10653 = x10649;
for(int x10654=0; x10654 < 4; x10654++) {
int32_t x10655 = x10651;
int32_t x10656 = x10652;
float x10657 = x10590[x10656];
int32_t x10658 = x10653;
float x10659 = x162[x10658];
float x10660 = x10657 - x10659;
x10629[x10655] = x10660;
x10651 += 1;
x10652 += 1;

}
x10644 += 4;
x10645 += 4;

}
x10637 += 16;
x10638 += 16;
x10639 += 1;

}
x10630 += 4096;
x10631 += 4096;

}
float* x10679 = (float*)myMalloc(256 * sizeof(float));;
for(int x10680=0; x10680 < 256; x10680++) {
float x10681 = x97[x10680];
float x10682 = x10681 + 1.0E-5f;
x10679[x10680] = x10682;

}
float* x10686 = (float*)myMalloc(256 * sizeof(float));;
for(int x10687=0; x10687 < 256; x10687++) {
float x10688 = x10679[x10687];
double x10689 = (double)x10688;
double x10690 = sqrt(x10689);
float x10691 = (float)x10690;
x10686[x10687] = x10691;

}
// resize to WrappedArray(-1, 1, 1)
float* x10696 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x10697 = 0;
int32_t x10698 = 0;
int32_t x10699 = 0;
for(int x10700=0; x10700 < 64; x10700++) {
int32_t x10701 = x10698;
int32_t x10702 = x10699;
int32_t x10703 = x10697;
int32_t x10704 = x10703;
int32_t x10705 = x10701;
int32_t x10706 = x10702;
for(int x10707=0; x10707 < 256; x10707++) {
int32_t x10708 = x10705;
int32_t x10709 = x10706;
int32_t x10710 = x10704;
int32_t x10711 = x10710;
int32_t x10712 = x10708;
int32_t x10713 = x10709;
for(int x10714=0; x10714 < 4; x10714++) {
int32_t x10715 = x10712;
int32_t x10716 = x10713;
int32_t x10717 = x10711;
int32_t x10718 = x10717;
int32_t x10719 = x10715;
int32_t x10720 = x10716;
for(int x10721=0; x10721 < 4; x10721++) {
int32_t x10722 = x10718;
int32_t x10723 = x10719;
float x10724 = x10629[x10723];
int32_t x10725 = x10720;
float x10726 = x10686[x10725];
float x10727 = x10724 / x10726;
x10696[x10722] = x10727;
x10718 += 1;
x10719 += 1;

}
x10711 += 4;
x10712 += 4;

}
x10704 += 16;
x10705 += 16;
x10706 += 1;

}
x10697 += 4096;
x10698 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x10747 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x10748 = 0;
int32_t x10749 = 0;
int32_t x10750 = 0;
for(int x10751=0; x10751 < 64; x10751++) {
int32_t x10752 = x10749;
int32_t x10753 = x10750;
int32_t x10754 = x10748;
int32_t x10755 = x10754;
int32_t x10756 = x10752;
int32_t x10757 = x10753;
for(int x10758=0; x10758 < 256; x10758++) {
int32_t x10759 = x10756;
int32_t x10760 = x10757;
int32_t x10761 = x10755;
int32_t x10762 = x10761;
int32_t x10763 = x10759;
int32_t x10764 = x10760;
for(int x10765=0; x10765 < 4; x10765++) {
int32_t x10766 = x10763;
int32_t x10767 = x10764;
int32_t x10768 = x10762;
int32_t x10769 = x10768;
int32_t x10770 = x10766;
int32_t x10771 = x10767;
for(int x10772=0; x10772 < 4; x10772++) {
int32_t x10773 = x10769;
int32_t x10774 = x10770;
float x10775 = x10696[x10774];
int32_t x10776 = x10771;
float x10777 = x91[x10776];
float x10778 = x10775 * x10777;
x10747[x10773] = x10778;
x10769 += 1;
x10770 += 1;

}
x10762 += 4;
x10763 += 4;

}
x10755 += 16;
x10756 += 16;
x10757 += 1;

}
x10748 += 4096;
x10749 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x10798 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x10799 = 0;
int32_t x10800 = 0;
int32_t x10801 = 0;
for(int x10802=0; x10802 < 64; x10802++) {
int32_t x10803 = x10800;
int32_t x10804 = x10801;
int32_t x10805 = x10799;
int32_t x10806 = x10805;
int32_t x10807 = x10803;
int32_t x10808 = x10804;
for(int x10809=0; x10809 < 256; x10809++) {
int32_t x10810 = x10807;
int32_t x10811 = x10808;
int32_t x10812 = x10806;
int32_t x10813 = x10812;
int32_t x10814 = x10810;
int32_t x10815 = x10811;
for(int x10816=0; x10816 < 4; x10816++) {
int32_t x10817 = x10814;
int32_t x10818 = x10815;
int32_t x10819 = x10813;
int32_t x10820 = x10819;
int32_t x10821 = x10817;
int32_t x10822 = x10818;
for(int x10823=0; x10823 < 4; x10823++) {
int32_t x10824 = x10820;
int32_t x10825 = x10821;
float x10826 = x10747[x10825];
int32_t x10827 = x10822;
float x10828 = x240[x10827];
float x10829 = x10826 + x10828;
x10798[x10824] = x10829;
x10820 += 1;
x10821 += 1;

}
x10813 += 4;
x10814 += 4;

}
x10806 += 16;
x10807 += 16;
x10808 += 1;

}
x10799 += 4096;
x10800 += 4096;

}
float* x10848 = (float*)myMalloc(262144 * sizeof(float));;
for(int x10849=0; x10849 < 262144; x10849++) {
float x10850 = x10798[x10849];
bool x10851 = x10850 < 0.0f;
if (x10851) {
x10848[x10849] = 0.0f;
} else {
float x10854 = x10798[x10849];
x10848[x10849] = x10854;
}

}
float* x10860 = (float*)myMalloc(262144 * sizeof(float));;
float* x10861 = (float*)myMalloc(2359296 * sizeof(float));;
for(int x10862=0; x10862 < 64; x10862++) {
int32_t x10863 = x10862 * 4096;
float* x10864 = x10848+x10863;
float* x10865 = x10860+x10863;
int32_t x10866 = x10862 * 36864;
float* x10867 = x10861+x10866;
for(int x10868=0; x10868 < 2304; x10868++) {
int32_t x10869 = x10868 / 9;
int32_t x10873 = x10869 * 3;
int32_t x10874 = x10873 * 3;
int32_t x10875 = x10874 * 4;
int32_t x10876 = x10875 * 4;
int32_t x10870 = x10868 % 9;
int32_t x10871 = x10870 / 3;
int32_t x10877 = x10871 * 3;
int32_t x10878 = x10877 * 4;
int32_t x10879 = x10878 * 4;
int32_t x10880 = x10876 + x10879;
int32_t x10872 = x10870 % 3;
int32_t x10881 = x10872 * 4;
int32_t x10882 = x10881 * 4;
int32_t x10883 = x10880 + x10882;
float* x10884 = x10867+x10883;
int32_t x10885 = x10869 * 4;
int32_t x10886 = x10885 * 4;
float* x10887 = x10864+x10886;
int32_t x10899 = 1 - x10872;
bool x10900 = x10899 > 0;
int32_t x10901;
if (x10900) {
x10901 = x10899;
} else {
x10901 = 0;
}
int32_t x10902 = 3 - x10872;
int32_t x10903 = x10902 - 1;
int32_t x10904 = 1 - x10903;
bool x10905 = x10904 > 0;
int32_t x10906;
if (x10905) {
x10906 = x10904;
} else {
x10906 = 0;
}
int32_t x10907 = 4 - x10906;
int32_t x10908 = x10907 - x10901;
bool x10909 = x10908 <= 0;
bool x10913 = x10901 > 0;
int32_t x10898 = -1 + x10872;
bool x10926 = x10906 > 0;
for(int x10888=0; x10888 < 4; x10888++) {
int32_t x10889 = x10888 - 1;
int32_t x10890 = x10889 + x10871;
bool x10891 = x10890 < 0;
bool x10892 = x10890 >= 4;
bool x10893 = x10891 || x10892;
if (x10893) {
int32_t x10894 = x10888 * 4;
float* x10895 = x10884+x10894;
memset(x10895, 0, 4 * 4);;
} else {
if (x10909) {
int32_t x10894 = x10888 * 4;
float* x10910 = x10884+x10894;
memset(x10910, 0, 4 * 4);;
} else {
int32_t x10894 = x10888 * 4;
if (x10913) {
float* x10914 = x10884+x10894;
memset(x10914, 0, 4 * x10901);;
} else {
}
// may have segfault here
int32_t x10919 = x10894 + x10901;
float* x10920 = x10884+x10919;
int32_t x10921 = x10890 * 4;
int32_t x10922 = x10921 + x10898;
int32_t x10923 = x10922 + x10901;
float* x10924 = x10887+x10923;
memcpy(x10920, x10924, 4 * x10908);;
if (x10926) {
int32_t x10927 = x10894 + 4;
int32_t x10928 = x10927 - x10906;
float* x10929 = x10884+x10928;
memset(x10929, 0, 4 * x10906);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,16,2304,1,x248,2304,x10867,16,1,x10865,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x10945 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x10946 = 0;
int32_t x10947 = 0;
int32_t x10948 = 0;
for(int x10949=0; x10949 < 64; x10949++) {
int32_t x10950 = x10947;
int32_t x10951 = x10948;
int32_t x10952 = x10946;
int32_t x10953 = x10952;
int32_t x10954 = x10950;
int32_t x10955 = x10951;
for(int x10956=0; x10956 < 256; x10956++) {
int32_t x10957 = x10954;
int32_t x10958 = x10955;
int32_t x10959 = x10953;
int32_t x10960 = x10959;
int32_t x10961 = x10957;
int32_t x10962 = x10958;
for(int x10963=0; x10963 < 4; x10963++) {
int32_t x10964 = x10961;
int32_t x10965 = x10962;
int32_t x10966 = x10960;
int32_t x10967 = x10966;
int32_t x10968 = x10964;
int32_t x10969 = x10965;
for(int x10970=0; x10970 < 4; x10970++) {
int32_t x10971 = x10967;
int32_t x10972 = x10968;
float x10973 = x10860[x10972];
int32_t x10974 = x10969;
float x10975 = x185[x10974];
float x10976 = x10973 - x10975;
x10945[x10971] = x10976;
x10967 += 1;
x10968 += 1;

}
x10960 += 4;
x10961 += 4;

}
x10953 += 16;
x10954 += 16;
x10955 += 1;

}
x10946 += 4096;
x10947 += 4096;

}
float* x10995 = (float*)myMalloc(256 * sizeof(float));;
for(int x10996=0; x10996 < 256; x10996++) {
float x10997 = x229[x10996];
float x10998 = x10997 + 1.0E-5f;
x10995[x10996] = x10998;

}
float* x11002 = (float*)myMalloc(256 * sizeof(float));;
for(int x11003=0; x11003 < 256; x11003++) {
float x11004 = x10995[x11003];
double x11005 = (double)x11004;
double x11006 = sqrt(x11005);
float x11007 = (float)x11006;
x11002[x11003] = x11007;

}
// resize to WrappedArray(-1, 1, 1)
float* x11012 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x11013 = 0;
int32_t x11014 = 0;
int32_t x11015 = 0;
for(int x11016=0; x11016 < 64; x11016++) {
int32_t x11017 = x11014;
int32_t x11018 = x11015;
int32_t x11019 = x11013;
int32_t x11020 = x11019;
int32_t x11021 = x11017;
int32_t x11022 = x11018;
for(int x11023=0; x11023 < 256; x11023++) {
int32_t x11024 = x11021;
int32_t x11025 = x11022;
int32_t x11026 = x11020;
int32_t x11027 = x11026;
int32_t x11028 = x11024;
int32_t x11029 = x11025;
for(int x11030=0; x11030 < 4; x11030++) {
int32_t x11031 = x11028;
int32_t x11032 = x11029;
int32_t x11033 = x11027;
int32_t x11034 = x11033;
int32_t x11035 = x11031;
int32_t x11036 = x11032;
for(int x11037=0; x11037 < 4; x11037++) {
int32_t x11038 = x11034;
int32_t x11039 = x11035;
float x11040 = x10945[x11039];
int32_t x11041 = x11036;
float x11042 = x11002[x11041];
float x11043 = x11040 / x11042;
x11012[x11038] = x11043;
x11034 += 1;
x11035 += 1;

}
x11027 += 4;
x11028 += 4;

}
x11020 += 16;
x11021 += 16;
x11022 += 1;

}
x11013 += 4096;
x11014 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x11063 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x11064 = 0;
int32_t x11065 = 0;
int32_t x11066 = 0;
for(int x11067=0; x11067 < 64; x11067++) {
int32_t x11068 = x11065;
int32_t x11069 = x11066;
int32_t x11070 = x11064;
int32_t x11071 = x11070;
int32_t x11072 = x11068;
int32_t x11073 = x11069;
for(int x11074=0; x11074 < 256; x11074++) {
int32_t x11075 = x11072;
int32_t x11076 = x11073;
int32_t x11077 = x11071;
int32_t x11078 = x11077;
int32_t x11079 = x11075;
int32_t x11080 = x11076;
for(int x11081=0; x11081 < 4; x11081++) {
int32_t x11082 = x11079;
int32_t x11083 = x11080;
int32_t x11084 = x11078;
int32_t x11085 = x11084;
int32_t x11086 = x11082;
int32_t x11087 = x11083;
for(int x11088=0; x11088 < 4; x11088++) {
int32_t x11089 = x11085;
int32_t x11090 = x11086;
float x11091 = x11012[x11090];
int32_t x11092 = x11087;
float x11093 = x73[x11092];
float x11094 = x11091 * x11093;
x11063[x11089] = x11094;
x11085 += 1;
x11086 += 1;

}
x11078 += 4;
x11079 += 4;

}
x11071 += 16;
x11072 += 16;
x11073 += 1;

}
x11064 += 4096;
x11065 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x11114 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x11115 = 0;
int32_t x11116 = 0;
int32_t x11117 = 0;
for(int x11118=0; x11118 < 64; x11118++) {
int32_t x11119 = x11116;
int32_t x11120 = x11117;
int32_t x11121 = x11115;
int32_t x11122 = x11121;
int32_t x11123 = x11119;
int32_t x11124 = x11120;
for(int x11125=0; x11125 < 256; x11125++) {
int32_t x11126 = x11123;
int32_t x11127 = x11124;
int32_t x11128 = x11122;
int32_t x11129 = x11128;
int32_t x11130 = x11126;
int32_t x11131 = x11127;
for(int x11132=0; x11132 < 4; x11132++) {
int32_t x11133 = x11130;
int32_t x11134 = x11131;
int32_t x11135 = x11129;
int32_t x11136 = x11135;
int32_t x11137 = x11133;
int32_t x11138 = x11134;
for(int x11139=0; x11139 < 4; x11139++) {
int32_t x11140 = x11136;
int32_t x11141 = x11137;
float x11142 = x11063[x11141];
int32_t x11143 = x11138;
float x11144 = x135[x11143];
float x11145 = x11142 + x11144;
x11114[x11140] = x11145;
x11136 += 1;
x11137 += 1;

}
x11129 += 4;
x11130 += 4;

}
x11122 += 16;
x11123 += 16;
x11124 += 1;

}
x11115 += 4096;
x11116 += 4096;

}
float* x11164 = (float*)myMalloc(262144 * sizeof(float));;
for(int x11165=0; x11165 < 262144; x11165++) {
float x11166 = x11114[x11165];
bool x11167 = x11166 < 0.0f;
if (x11167) {
x11164[x11165] = 0.0f;
} else {
float x11170 = x11114[x11165];
x11164[x11165] = x11170;
}

}
float* x11176 = (float*)myMalloc(1048576 * sizeof(float));;
float* x11177 = (float*)myMalloc(262144 * sizeof(float));;
for(int x11178=0; x11178 < 64; x11178++) {
int32_t x11179 = x11178 * 4096;
float* x11180 = x11164+x11179;
int32_t x11181 = x11178 * 16384;
float* x11182 = x11176+x11181;
float* x11183 = x11177+x11179;
for(int x11184=0; x11184 < 256; x11184++) {
int32_t x11185 = x11184 / 1;
int32_t x11189 = x11185 * 4;
int32_t x11190 = x11189 * 4;
int32_t x11186 = x11184 % 1;
int32_t x11187 = x11186 / 1;
int32_t x11191 = x11187 * 4;
int32_t x11192 = x11191 * 4;
int32_t x11193 = x11190 + x11192;
int32_t x11188 = x11186 % 1;
int32_t x11194 = x11188 * 4;
int32_t x11195 = x11194 * 4;
int32_t x11196 = x11193 + x11195;
float* x11197 = x11183+x11196;
float* x11198 = x11180+x11190;
for(int x11199=0; x11199 < 4; x11199++) {
int32_t x11201 = x11199 * 4;
float* x11202 = x11197+x11201;
int32_t x11200 = x11199 + x11187;
int32_t x11203 = x11200 * 4;
int32_t x11204 = x11203 + x11188;
float* x11205 = x11198+x11204;
memcpy(x11202, x11205, 4 * 4);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,16,256,1,x88,256,x11183,16,1,x11182,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x11215 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x11216 = 0;
int32_t x11217 = 0;
int32_t x11218 = 0;
for(int x11219=0; x11219 < 64; x11219++) {
int32_t x11220 = x11217;
int32_t x11221 = x11218;
int32_t x11222 = x11216;
int32_t x11223 = x11222;
int32_t x11224 = x11220;
int32_t x11225 = x11221;
for(int x11226=0; x11226 < 1024; x11226++) {
int32_t x11227 = x11224;
int32_t x11228 = x11225;
int32_t x11229 = x11223;
int32_t x11230 = x11229;
int32_t x11231 = x11227;
int32_t x11232 = x11228;
for(int x11233=0; x11233 < 4; x11233++) {
int32_t x11234 = x11231;
int32_t x11235 = x11232;
int32_t x11236 = x11230;
int32_t x11237 = x11236;
int32_t x11238 = x11234;
int32_t x11239 = x11235;
for(int x11240=0; x11240 < 4; x11240++) {
int32_t x11241 = x11237;
int32_t x11242 = x11238;
float x11243 = x11176[x11242];
int32_t x11244 = x11239;
float x11245 = x230[x11244];
float x11246 = x11243 - x11245;
x11215[x11241] = x11246;
x11237 += 1;
x11238 += 1;

}
x11230 += 4;
x11231 += 4;

}
x11223 += 16;
x11224 += 16;
x11225 += 1;

}
x11216 += 16384;
x11217 += 16384;

}
float* x11265 = (float*)myMalloc(1024 * sizeof(float));;
for(int x11266=0; x11266 < 1024; x11266++) {
float x11267 = x160[x11266];
float x11268 = x11267 + 1.0E-5f;
x11265[x11266] = x11268;

}
float* x11272 = (float*)myMalloc(1024 * sizeof(float));;
for(int x11273=0; x11273 < 1024; x11273++) {
float x11274 = x11265[x11273];
double x11275 = (double)x11274;
double x11276 = sqrt(x11275);
float x11277 = (float)x11276;
x11272[x11273] = x11277;

}
// resize to WrappedArray(-1, 1, 1)
float* x11282 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x11283 = 0;
int32_t x11284 = 0;
int32_t x11285 = 0;
for(int x11286=0; x11286 < 64; x11286++) {
int32_t x11287 = x11284;
int32_t x11288 = x11285;
int32_t x11289 = x11283;
int32_t x11290 = x11289;
int32_t x11291 = x11287;
int32_t x11292 = x11288;
for(int x11293=0; x11293 < 1024; x11293++) {
int32_t x11294 = x11291;
int32_t x11295 = x11292;
int32_t x11296 = x11290;
int32_t x11297 = x11296;
int32_t x11298 = x11294;
int32_t x11299 = x11295;
for(int x11300=0; x11300 < 4; x11300++) {
int32_t x11301 = x11298;
int32_t x11302 = x11299;
int32_t x11303 = x11297;
int32_t x11304 = x11303;
int32_t x11305 = x11301;
int32_t x11306 = x11302;
for(int x11307=0; x11307 < 4; x11307++) {
int32_t x11308 = x11304;
int32_t x11309 = x11305;
float x11310 = x11215[x11309];
int32_t x11311 = x11306;
float x11312 = x11272[x11311];
float x11313 = x11310 / x11312;
x11282[x11308] = x11313;
x11304 += 1;
x11305 += 1;

}
x11297 += 4;
x11298 += 4;

}
x11290 += 16;
x11291 += 16;
x11292 += 1;

}
x11283 += 16384;
x11284 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x11333 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x11334 = 0;
int32_t x11335 = 0;
int32_t x11336 = 0;
for(int x11337=0; x11337 < 64; x11337++) {
int32_t x11338 = x11335;
int32_t x11339 = x11336;
int32_t x11340 = x11334;
int32_t x11341 = x11340;
int32_t x11342 = x11338;
int32_t x11343 = x11339;
for(int x11344=0; x11344 < 1024; x11344++) {
int32_t x11345 = x11342;
int32_t x11346 = x11343;
int32_t x11347 = x11341;
int32_t x11348 = x11347;
int32_t x11349 = x11345;
int32_t x11350 = x11346;
for(int x11351=0; x11351 < 4; x11351++) {
int32_t x11352 = x11349;
int32_t x11353 = x11350;
int32_t x11354 = x11348;
int32_t x11355 = x11354;
int32_t x11356 = x11352;
int32_t x11357 = x11353;
for(int x11358=0; x11358 < 4; x11358++) {
int32_t x11359 = x11355;
int32_t x11360 = x11356;
float x11361 = x11282[x11360];
int32_t x11362 = x11357;
float x11363 = x237[x11362];
float x11364 = x11361 * x11363;
x11333[x11359] = x11364;
x11355 += 1;
x11356 += 1;

}
x11348 += 4;
x11349 += 4;

}
x11341 += 16;
x11342 += 16;
x11343 += 1;

}
x11334 += 16384;
x11335 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x11384 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x11385 = 0;
int32_t x11386 = 0;
int32_t x11387 = 0;
for(int x11388=0; x11388 < 64; x11388++) {
int32_t x11389 = x11386;
int32_t x11390 = x11387;
int32_t x11391 = x11385;
int32_t x11392 = x11391;
int32_t x11393 = x11389;
int32_t x11394 = x11390;
for(int x11395=0; x11395 < 1024; x11395++) {
int32_t x11396 = x11393;
int32_t x11397 = x11394;
int32_t x11398 = x11392;
int32_t x11399 = x11398;
int32_t x11400 = x11396;
int32_t x11401 = x11397;
for(int x11402=0; x11402 < 4; x11402++) {
int32_t x11403 = x11400;
int32_t x11404 = x11401;
int32_t x11405 = x11399;
int32_t x11406 = x11405;
int32_t x11407 = x11403;
int32_t x11408 = x11404;
for(int x11409=0; x11409 < 4; x11409++) {
int32_t x11410 = x11406;
int32_t x11411 = x11407;
float x11412 = x11333[x11411];
int32_t x11413 = x11408;
float x11414 = x145[x11413];
float x11415 = x11412 + x11414;
x11384[x11410] = x11415;
x11406 += 1;
x11407 += 1;

}
x11399 += 4;
x11400 += 4;

}
x11392 += 16;
x11393 += 16;
x11394 += 1;

}
x11385 += 16384;
x11386 += 16384;

}
int32_t x11434 = 0;
int32_t x11435 = 0;
int32_t x11436 = 0;
for(int x11437=0; x11437 < 64; x11437++) {
int32_t x11438 = x11435;
int32_t x11439 = x11436;
int32_t x11440 = x11434;
int32_t x11441 = x11440;
int32_t x11442 = x11438;
int32_t x11443 = x11439;
for(int x11444=0; x11444 < 1024; x11444++) {
int32_t x11445 = x11442;
int32_t x11446 = x11443;
int32_t x11447 = x11441;
int32_t x11448 = x11447;
int32_t x11449 = x11445;
int32_t x11450 = x11446;
for(int x11451=0; x11451 < 4; x11451++) {
int32_t x11452 = x11449;
int32_t x11453 = x11450;
int32_t x11454 = x11448;
int32_t x11455 = x11454;
int32_t x11456 = x11452;
int32_t x11457 = x11453;
for(int x11458=0; x11458 < 4; x11458++) {
int32_t x11459 = x11456;
float x11460 = x11384[x11459];
int32_t x11461 = x11457;
float x11462 = x10578[x11461];
float x11463 = x11460 + x11462;
x11384[x11459] = x11463;
x11455 += 1;
x11456 += 1;
x11457 += 1;

}
x11448 += 4;
x11449 += 4;
x11450 += 4;

}
x11441 += 16;
x11442 += 16;
x11443 += 16;

}
x11434 += 16384;
x11435 += 16384;
x11436 += 16384;

}
float* x11485 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x11486=0; x11486 < 1048576; x11486++) {
float x11487 = x11384[x11486];
bool x11488 = x11487 < 0.0f;
if (x11488) {
x11485[x11486] = 0.0f;
} else {
float x11491 = x11384[x11486];
x11485[x11486] = x11491;
}

}
float* x11497 = (float*)myMalloc(262144 * sizeof(float));;
float* x11498 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x11499=0; x11499 < 64; x11499++) {
int32_t x11500 = x11499 * 16384;
float* x11501 = x11485+x11500;
int32_t x11502 = x11499 * 4096;
float* x11503 = x11497+x11502;
float* x11504 = x11498+x11500;
for(int x11505=0; x11505 < 1024; x11505++) {
int32_t x11506 = x11505 / 1;
int32_t x11510 = x11506 * 4;
int32_t x11511 = x11510 * 4;
int32_t x11507 = x11505 % 1;
int32_t x11508 = x11507 / 1;
int32_t x11512 = x11508 * 4;
int32_t x11513 = x11512 * 4;
int32_t x11514 = x11511 + x11513;
int32_t x11509 = x11507 % 1;
int32_t x11515 = x11509 * 4;
int32_t x11516 = x11515 * 4;
int32_t x11517 = x11514 + x11516;
float* x11518 = x11504+x11517;
float* x11519 = x11501+x11511;
for(int x11520=0; x11520 < 4; x11520++) {
int32_t x11522 = x11520 * 4;
float* x11523 = x11518+x11522;
int32_t x11521 = x11520 + x11508;
int32_t x11524 = x11521 * 4;
int32_t x11525 = x11524 + x11509;
float* x11526 = x11519+x11525;
memcpy(x11523, x11526, 4 * 4);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,16,1024,1,x21,1024,x11504,16,1,x11503,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x11536 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x11537 = 0;
int32_t x11538 = 0;
int32_t x11539 = 0;
for(int x11540=0; x11540 < 64; x11540++) {
int32_t x11541 = x11538;
int32_t x11542 = x11539;
int32_t x11543 = x11537;
int32_t x11544 = x11543;
int32_t x11545 = x11541;
int32_t x11546 = x11542;
for(int x11547=0; x11547 < 256; x11547++) {
int32_t x11548 = x11545;
int32_t x11549 = x11546;
int32_t x11550 = x11544;
int32_t x11551 = x11550;
int32_t x11552 = x11548;
int32_t x11553 = x11549;
for(int x11554=0; x11554 < 4; x11554++) {
int32_t x11555 = x11552;
int32_t x11556 = x11553;
int32_t x11557 = x11551;
int32_t x11558 = x11557;
int32_t x11559 = x11555;
int32_t x11560 = x11556;
for(int x11561=0; x11561 < 4; x11561++) {
int32_t x11562 = x11558;
int32_t x11563 = x11559;
float x11564 = x11497[x11563];
int32_t x11565 = x11560;
float x11566 = x253[x11565];
float x11567 = x11564 - x11566;
x11536[x11562] = x11567;
x11558 += 1;
x11559 += 1;

}
x11551 += 4;
x11552 += 4;

}
x11544 += 16;
x11545 += 16;
x11546 += 1;

}
x11537 += 4096;
x11538 += 4096;

}
float* x11586 = (float*)myMalloc(256 * sizeof(float));;
for(int x11587=0; x11587 < 256; x11587++) {
float x11588 = x68[x11587];
float x11589 = x11588 + 1.0E-5f;
x11586[x11587] = x11589;

}
float* x11593 = (float*)myMalloc(256 * sizeof(float));;
for(int x11594=0; x11594 < 256; x11594++) {
float x11595 = x11586[x11594];
double x11596 = (double)x11595;
double x11597 = sqrt(x11596);
float x11598 = (float)x11597;
x11593[x11594] = x11598;

}
// resize to WrappedArray(-1, 1, 1)
float* x11603 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x11604 = 0;
int32_t x11605 = 0;
int32_t x11606 = 0;
for(int x11607=0; x11607 < 64; x11607++) {
int32_t x11608 = x11605;
int32_t x11609 = x11606;
int32_t x11610 = x11604;
int32_t x11611 = x11610;
int32_t x11612 = x11608;
int32_t x11613 = x11609;
for(int x11614=0; x11614 < 256; x11614++) {
int32_t x11615 = x11612;
int32_t x11616 = x11613;
int32_t x11617 = x11611;
int32_t x11618 = x11617;
int32_t x11619 = x11615;
int32_t x11620 = x11616;
for(int x11621=0; x11621 < 4; x11621++) {
int32_t x11622 = x11619;
int32_t x11623 = x11620;
int32_t x11624 = x11618;
int32_t x11625 = x11624;
int32_t x11626 = x11622;
int32_t x11627 = x11623;
for(int x11628=0; x11628 < 4; x11628++) {
int32_t x11629 = x11625;
int32_t x11630 = x11626;
float x11631 = x11536[x11630];
int32_t x11632 = x11627;
float x11633 = x11593[x11632];
float x11634 = x11631 / x11633;
x11603[x11629] = x11634;
x11625 += 1;
x11626 += 1;

}
x11618 += 4;
x11619 += 4;

}
x11611 += 16;
x11612 += 16;
x11613 += 1;

}
x11604 += 4096;
x11605 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x11654 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x11655 = 0;
int32_t x11656 = 0;
int32_t x11657 = 0;
for(int x11658=0; x11658 < 64; x11658++) {
int32_t x11659 = x11656;
int32_t x11660 = x11657;
int32_t x11661 = x11655;
int32_t x11662 = x11661;
int32_t x11663 = x11659;
int32_t x11664 = x11660;
for(int x11665=0; x11665 < 256; x11665++) {
int32_t x11666 = x11663;
int32_t x11667 = x11664;
int32_t x11668 = x11662;
int32_t x11669 = x11668;
int32_t x11670 = x11666;
int32_t x11671 = x11667;
for(int x11672=0; x11672 < 4; x11672++) {
int32_t x11673 = x11670;
int32_t x11674 = x11671;
int32_t x11675 = x11669;
int32_t x11676 = x11675;
int32_t x11677 = x11673;
int32_t x11678 = x11674;
for(int x11679=0; x11679 < 4; x11679++) {
int32_t x11680 = x11676;
int32_t x11681 = x11677;
float x11682 = x11603[x11681];
int32_t x11683 = x11678;
float x11684 = x76[x11683];
float x11685 = x11682 * x11684;
x11654[x11680] = x11685;
x11676 += 1;
x11677 += 1;

}
x11669 += 4;
x11670 += 4;

}
x11662 += 16;
x11663 += 16;
x11664 += 1;

}
x11655 += 4096;
x11656 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x11705 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x11706 = 0;
int32_t x11707 = 0;
int32_t x11708 = 0;
for(int x11709=0; x11709 < 64; x11709++) {
int32_t x11710 = x11707;
int32_t x11711 = x11708;
int32_t x11712 = x11706;
int32_t x11713 = x11712;
int32_t x11714 = x11710;
int32_t x11715 = x11711;
for(int x11716=0; x11716 < 256; x11716++) {
int32_t x11717 = x11714;
int32_t x11718 = x11715;
int32_t x11719 = x11713;
int32_t x11720 = x11719;
int32_t x11721 = x11717;
int32_t x11722 = x11718;
for(int x11723=0; x11723 < 4; x11723++) {
int32_t x11724 = x11721;
int32_t x11725 = x11722;
int32_t x11726 = x11720;
int32_t x11727 = x11726;
int32_t x11728 = x11724;
int32_t x11729 = x11725;
for(int x11730=0; x11730 < 4; x11730++) {
int32_t x11731 = x11727;
int32_t x11732 = x11728;
float x11733 = x11654[x11732];
int32_t x11734 = x11729;
float x11735 = x184[x11734];
float x11736 = x11733 + x11735;
x11705[x11731] = x11736;
x11727 += 1;
x11728 += 1;

}
x11720 += 4;
x11721 += 4;

}
x11713 += 16;
x11714 += 16;
x11715 += 1;

}
x11706 += 4096;
x11707 += 4096;

}
float* x11755 = (float*)myMalloc(262144 * sizeof(float));;
for(int x11756=0; x11756 < 262144; x11756++) {
float x11757 = x11705[x11756];
bool x11758 = x11757 < 0.0f;
if (x11758) {
x11755[x11756] = 0.0f;
} else {
float x11761 = x11705[x11756];
x11755[x11756] = x11761;
}

}
float* x11767 = (float*)myMalloc(262144 * sizeof(float));;
float* x11768 = (float*)myMalloc(2359296 * sizeof(float));;
for(int x11769=0; x11769 < 64; x11769++) {
int32_t x11770 = x11769 * 4096;
float* x11771 = x11755+x11770;
float* x11772 = x11767+x11770;
int32_t x11773 = x11769 * 36864;
float* x11774 = x11768+x11773;
for(int x11775=0; x11775 < 2304; x11775++) {
int32_t x11776 = x11775 / 9;
int32_t x11780 = x11776 * 3;
int32_t x11781 = x11780 * 3;
int32_t x11782 = x11781 * 4;
int32_t x11783 = x11782 * 4;
int32_t x11777 = x11775 % 9;
int32_t x11778 = x11777 / 3;
int32_t x11784 = x11778 * 3;
int32_t x11785 = x11784 * 4;
int32_t x11786 = x11785 * 4;
int32_t x11787 = x11783 + x11786;
int32_t x11779 = x11777 % 3;
int32_t x11788 = x11779 * 4;
int32_t x11789 = x11788 * 4;
int32_t x11790 = x11787 + x11789;
float* x11791 = x11774+x11790;
int32_t x11792 = x11776 * 4;
int32_t x11793 = x11792 * 4;
float* x11794 = x11771+x11793;
int32_t x11806 = 1 - x11779;
bool x11807 = x11806 > 0;
int32_t x11808;
if (x11807) {
x11808 = x11806;
} else {
x11808 = 0;
}
int32_t x11809 = 3 - x11779;
int32_t x11810 = x11809 - 1;
int32_t x11811 = 1 - x11810;
bool x11812 = x11811 > 0;
int32_t x11813;
if (x11812) {
x11813 = x11811;
} else {
x11813 = 0;
}
int32_t x11814 = 4 - x11813;
int32_t x11815 = x11814 - x11808;
bool x11816 = x11815 <= 0;
bool x11820 = x11808 > 0;
int32_t x11805 = -1 + x11779;
bool x11833 = x11813 > 0;
for(int x11795=0; x11795 < 4; x11795++) {
int32_t x11796 = x11795 - 1;
int32_t x11797 = x11796 + x11778;
bool x11798 = x11797 < 0;
bool x11799 = x11797 >= 4;
bool x11800 = x11798 || x11799;
if (x11800) {
int32_t x11801 = x11795 * 4;
float* x11802 = x11791+x11801;
memset(x11802, 0, 4 * 4);;
} else {
if (x11816) {
int32_t x11801 = x11795 * 4;
float* x11817 = x11791+x11801;
memset(x11817, 0, 4 * 4);;
} else {
int32_t x11801 = x11795 * 4;
if (x11820) {
float* x11821 = x11791+x11801;
memset(x11821, 0, 4 * x11808);;
} else {
}
// may have segfault here
int32_t x11826 = x11801 + x11808;
float* x11827 = x11791+x11826;
int32_t x11828 = x11797 * 4;
int32_t x11829 = x11828 + x11805;
int32_t x11830 = x11829 + x11808;
float* x11831 = x11794+x11830;
memcpy(x11827, x11831, 4 * x11815);;
if (x11833) {
int32_t x11834 = x11801 + 4;
int32_t x11835 = x11834 - x11813;
float* x11836 = x11791+x11835;
memset(x11836, 0, 4 * x11813);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,16,2304,1,x261,2304,x11774,16,1,x11772,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x11852 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x11853 = 0;
int32_t x11854 = 0;
int32_t x11855 = 0;
for(int x11856=0; x11856 < 64; x11856++) {
int32_t x11857 = x11854;
int32_t x11858 = x11855;
int32_t x11859 = x11853;
int32_t x11860 = x11859;
int32_t x11861 = x11857;
int32_t x11862 = x11858;
for(int x11863=0; x11863 < 256; x11863++) {
int32_t x11864 = x11861;
int32_t x11865 = x11862;
int32_t x11866 = x11860;
int32_t x11867 = x11866;
int32_t x11868 = x11864;
int32_t x11869 = x11865;
for(int x11870=0; x11870 < 4; x11870++) {
int32_t x11871 = x11868;
int32_t x11872 = x11869;
int32_t x11873 = x11867;
int32_t x11874 = x11873;
int32_t x11875 = x11871;
int32_t x11876 = x11872;
for(int x11877=0; x11877 < 4; x11877++) {
int32_t x11878 = x11874;
int32_t x11879 = x11875;
float x11880 = x11767[x11879];
int32_t x11881 = x11876;
float x11882 = x249[x11881];
float x11883 = x11880 - x11882;
x11852[x11878] = x11883;
x11874 += 1;
x11875 += 1;

}
x11867 += 4;
x11868 += 4;

}
x11860 += 16;
x11861 += 16;
x11862 += 1;

}
x11853 += 4096;
x11854 += 4096;

}
float* x11902 = (float*)myMalloc(256 * sizeof(float));;
for(int x11903=0; x11903 < 256; x11903++) {
float x11904 = x103[x11903];
float x11905 = x11904 + 1.0E-5f;
x11902[x11903] = x11905;

}
float* x11909 = (float*)myMalloc(256 * sizeof(float));;
for(int x11910=0; x11910 < 256; x11910++) {
float x11911 = x11902[x11910];
double x11912 = (double)x11911;
double x11913 = sqrt(x11912);
float x11914 = (float)x11913;
x11909[x11910] = x11914;

}
// resize to WrappedArray(-1, 1, 1)
float* x11919 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x11920 = 0;
int32_t x11921 = 0;
int32_t x11922 = 0;
for(int x11923=0; x11923 < 64; x11923++) {
int32_t x11924 = x11921;
int32_t x11925 = x11922;
int32_t x11926 = x11920;
int32_t x11927 = x11926;
int32_t x11928 = x11924;
int32_t x11929 = x11925;
for(int x11930=0; x11930 < 256; x11930++) {
int32_t x11931 = x11928;
int32_t x11932 = x11929;
int32_t x11933 = x11927;
int32_t x11934 = x11933;
int32_t x11935 = x11931;
int32_t x11936 = x11932;
for(int x11937=0; x11937 < 4; x11937++) {
int32_t x11938 = x11935;
int32_t x11939 = x11936;
int32_t x11940 = x11934;
int32_t x11941 = x11940;
int32_t x11942 = x11938;
int32_t x11943 = x11939;
for(int x11944=0; x11944 < 4; x11944++) {
int32_t x11945 = x11941;
int32_t x11946 = x11942;
float x11947 = x11852[x11946];
int32_t x11948 = x11943;
float x11949 = x11909[x11948];
float x11950 = x11947 / x11949;
x11919[x11945] = x11950;
x11941 += 1;
x11942 += 1;

}
x11934 += 4;
x11935 += 4;

}
x11927 += 16;
x11928 += 16;
x11929 += 1;

}
x11920 += 4096;
x11921 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x11970 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x11971 = 0;
int32_t x11972 = 0;
int32_t x11973 = 0;
for(int x11974=0; x11974 < 64; x11974++) {
int32_t x11975 = x11972;
int32_t x11976 = x11973;
int32_t x11977 = x11971;
int32_t x11978 = x11977;
int32_t x11979 = x11975;
int32_t x11980 = x11976;
for(int x11981=0; x11981 < 256; x11981++) {
int32_t x11982 = x11979;
int32_t x11983 = x11980;
int32_t x11984 = x11978;
int32_t x11985 = x11984;
int32_t x11986 = x11982;
int32_t x11987 = x11983;
for(int x11988=0; x11988 < 4; x11988++) {
int32_t x11989 = x11986;
int32_t x11990 = x11987;
int32_t x11991 = x11985;
int32_t x11992 = x11991;
int32_t x11993 = x11989;
int32_t x11994 = x11990;
for(int x11995=0; x11995 < 4; x11995++) {
int32_t x11996 = x11992;
int32_t x11997 = x11993;
float x11998 = x11919[x11997];
int32_t x11999 = x11994;
float x12000 = x167[x11999];
float x12001 = x11998 * x12000;
x11970[x11996] = x12001;
x11992 += 1;
x11993 += 1;

}
x11985 += 4;
x11986 += 4;

}
x11978 += 16;
x11979 += 16;
x11980 += 1;

}
x11971 += 4096;
x11972 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x12021 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x12022 = 0;
int32_t x12023 = 0;
int32_t x12024 = 0;
for(int x12025=0; x12025 < 64; x12025++) {
int32_t x12026 = x12023;
int32_t x12027 = x12024;
int32_t x12028 = x12022;
int32_t x12029 = x12028;
int32_t x12030 = x12026;
int32_t x12031 = x12027;
for(int x12032=0; x12032 < 256; x12032++) {
int32_t x12033 = x12030;
int32_t x12034 = x12031;
int32_t x12035 = x12029;
int32_t x12036 = x12035;
int32_t x12037 = x12033;
int32_t x12038 = x12034;
for(int x12039=0; x12039 < 4; x12039++) {
int32_t x12040 = x12037;
int32_t x12041 = x12038;
int32_t x12042 = x12036;
int32_t x12043 = x12042;
int32_t x12044 = x12040;
int32_t x12045 = x12041;
for(int x12046=0; x12046 < 4; x12046++) {
int32_t x12047 = x12043;
int32_t x12048 = x12044;
float x12049 = x11970[x12048];
int32_t x12050 = x12045;
float x12051 = x108[x12050];
float x12052 = x12049 + x12051;
x12021[x12047] = x12052;
x12043 += 1;
x12044 += 1;

}
x12036 += 4;
x12037 += 4;

}
x12029 += 16;
x12030 += 16;
x12031 += 1;

}
x12022 += 4096;
x12023 += 4096;

}
float* x12071 = (float*)myMalloc(262144 * sizeof(float));;
for(int x12072=0; x12072 < 262144; x12072++) {
float x12073 = x12021[x12072];
bool x12074 = x12073 < 0.0f;
if (x12074) {
x12071[x12072] = 0.0f;
} else {
float x12077 = x12021[x12072];
x12071[x12072] = x12077;
}

}
float* x12083 = (float*)myMalloc(1048576 * sizeof(float));;
float* x12084 = (float*)myMalloc(262144 * sizeof(float));;
for(int x12085=0; x12085 < 64; x12085++) {
int32_t x12086 = x12085 * 4096;
float* x12087 = x12071+x12086;
int32_t x12088 = x12085 * 16384;
float* x12089 = x12083+x12088;
float* x12090 = x12084+x12086;
for(int x12091=0; x12091 < 256; x12091++) {
int32_t x12092 = x12091 / 1;
int32_t x12096 = x12092 * 4;
int32_t x12097 = x12096 * 4;
int32_t x12093 = x12091 % 1;
int32_t x12094 = x12093 / 1;
int32_t x12098 = x12094 * 4;
int32_t x12099 = x12098 * 4;
int32_t x12100 = x12097 + x12099;
int32_t x12095 = x12093 % 1;
int32_t x12101 = x12095 * 4;
int32_t x12102 = x12101 * 4;
int32_t x12103 = x12100 + x12102;
float* x12104 = x12090+x12103;
float* x12105 = x12087+x12097;
for(int x12106=0; x12106 < 4; x12106++) {
int32_t x12108 = x12106 * 4;
float* x12109 = x12104+x12108;
int32_t x12107 = x12106 + x12094;
int32_t x12110 = x12107 * 4;
int32_t x12111 = x12110 + x12095;
float* x12112 = x12105+x12111;
memcpy(x12109, x12112, 4 * 4);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,16,256,1,x220,256,x12090,16,1,x12089,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x12122 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x12123 = 0;
int32_t x12124 = 0;
int32_t x12125 = 0;
for(int x12126=0; x12126 < 64; x12126++) {
int32_t x12127 = x12124;
int32_t x12128 = x12125;
int32_t x12129 = x12123;
int32_t x12130 = x12129;
int32_t x12131 = x12127;
int32_t x12132 = x12128;
for(int x12133=0; x12133 < 1024; x12133++) {
int32_t x12134 = x12131;
int32_t x12135 = x12132;
int32_t x12136 = x12130;
int32_t x12137 = x12136;
int32_t x12138 = x12134;
int32_t x12139 = x12135;
for(int x12140=0; x12140 < 4; x12140++) {
int32_t x12141 = x12138;
int32_t x12142 = x12139;
int32_t x12143 = x12137;
int32_t x12144 = x12143;
int32_t x12145 = x12141;
int32_t x12146 = x12142;
for(int x12147=0; x12147 < 4; x12147++) {
int32_t x12148 = x12144;
int32_t x12149 = x12145;
float x12150 = x12083[x12149];
int32_t x12151 = x12146;
float x12152 = x208[x12151];
float x12153 = x12150 - x12152;
x12122[x12148] = x12153;
x12144 += 1;
x12145 += 1;

}
x12137 += 4;
x12138 += 4;

}
x12130 += 16;
x12131 += 16;
x12132 += 1;

}
x12123 += 16384;
x12124 += 16384;

}
float* x12172 = (float*)myMalloc(1024 * sizeof(float));;
for(int x12173=0; x12173 < 1024; x12173++) {
float x12174 = x271[x12173];
float x12175 = x12174 + 1.0E-5f;
x12172[x12173] = x12175;

}
float* x12179 = (float*)myMalloc(1024 * sizeof(float));;
for(int x12180=0; x12180 < 1024; x12180++) {
float x12181 = x12172[x12180];
double x12182 = (double)x12181;
double x12183 = sqrt(x12182);
float x12184 = (float)x12183;
x12179[x12180] = x12184;

}
// resize to WrappedArray(-1, 1, 1)
float* x12189 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x12190 = 0;
int32_t x12191 = 0;
int32_t x12192 = 0;
for(int x12193=0; x12193 < 64; x12193++) {
int32_t x12194 = x12191;
int32_t x12195 = x12192;
int32_t x12196 = x12190;
int32_t x12197 = x12196;
int32_t x12198 = x12194;
int32_t x12199 = x12195;
for(int x12200=0; x12200 < 1024; x12200++) {
int32_t x12201 = x12198;
int32_t x12202 = x12199;
int32_t x12203 = x12197;
int32_t x12204 = x12203;
int32_t x12205 = x12201;
int32_t x12206 = x12202;
for(int x12207=0; x12207 < 4; x12207++) {
int32_t x12208 = x12205;
int32_t x12209 = x12206;
int32_t x12210 = x12204;
int32_t x12211 = x12210;
int32_t x12212 = x12208;
int32_t x12213 = x12209;
for(int x12214=0; x12214 < 4; x12214++) {
int32_t x12215 = x12211;
int32_t x12216 = x12212;
float x12217 = x12122[x12216];
int32_t x12218 = x12213;
float x12219 = x12179[x12218];
float x12220 = x12217 / x12219;
x12189[x12215] = x12220;
x12211 += 1;
x12212 += 1;

}
x12204 += 4;
x12205 += 4;

}
x12197 += 16;
x12198 += 16;
x12199 += 1;

}
x12190 += 16384;
x12191 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x12240 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x12241 = 0;
int32_t x12242 = 0;
int32_t x12243 = 0;
for(int x12244=0; x12244 < 64; x12244++) {
int32_t x12245 = x12242;
int32_t x12246 = x12243;
int32_t x12247 = x12241;
int32_t x12248 = x12247;
int32_t x12249 = x12245;
int32_t x12250 = x12246;
for(int x12251=0; x12251 < 1024; x12251++) {
int32_t x12252 = x12249;
int32_t x12253 = x12250;
int32_t x12254 = x12248;
int32_t x12255 = x12254;
int32_t x12256 = x12252;
int32_t x12257 = x12253;
for(int x12258=0; x12258 < 4; x12258++) {
int32_t x12259 = x12256;
int32_t x12260 = x12257;
int32_t x12261 = x12255;
int32_t x12262 = x12261;
int32_t x12263 = x12259;
int32_t x12264 = x12260;
for(int x12265=0; x12265 < 4; x12265++) {
int32_t x12266 = x12262;
int32_t x12267 = x12263;
float x12268 = x12189[x12267];
int32_t x12269 = x12264;
float x12270 = x58[x12269];
float x12271 = x12268 * x12270;
x12240[x12266] = x12271;
x12262 += 1;
x12263 += 1;

}
x12255 += 4;
x12256 += 4;

}
x12248 += 16;
x12249 += 16;
x12250 += 1;

}
x12241 += 16384;
x12242 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x12291 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x12292 = 0;
int32_t x12293 = 0;
int32_t x12294 = 0;
for(int x12295=0; x12295 < 64; x12295++) {
int32_t x12296 = x12293;
int32_t x12297 = x12294;
int32_t x12298 = x12292;
int32_t x12299 = x12298;
int32_t x12300 = x12296;
int32_t x12301 = x12297;
for(int x12302=0; x12302 < 1024; x12302++) {
int32_t x12303 = x12300;
int32_t x12304 = x12301;
int32_t x12305 = x12299;
int32_t x12306 = x12305;
int32_t x12307 = x12303;
int32_t x12308 = x12304;
for(int x12309=0; x12309 < 4; x12309++) {
int32_t x12310 = x12307;
int32_t x12311 = x12308;
int32_t x12312 = x12306;
int32_t x12313 = x12312;
int32_t x12314 = x12310;
int32_t x12315 = x12311;
for(int x12316=0; x12316 < 4; x12316++) {
int32_t x12317 = x12313;
int32_t x12318 = x12314;
float x12319 = x12240[x12318];
int32_t x12320 = x12315;
float x12321 = x119[x12320];
float x12322 = x12319 + x12321;
x12291[x12317] = x12322;
x12313 += 1;
x12314 += 1;

}
x12306 += 4;
x12307 += 4;

}
x12299 += 16;
x12300 += 16;
x12301 += 1;

}
x12292 += 16384;
x12293 += 16384;

}
int32_t x12341 = 0;
int32_t x12342 = 0;
int32_t x12343 = 0;
for(int x12344=0; x12344 < 64; x12344++) {
int32_t x12345 = x12342;
int32_t x12346 = x12343;
int32_t x12347 = x12341;
int32_t x12348 = x12347;
int32_t x12349 = x12345;
int32_t x12350 = x12346;
for(int x12351=0; x12351 < 1024; x12351++) {
int32_t x12352 = x12349;
int32_t x12353 = x12350;
int32_t x12354 = x12348;
int32_t x12355 = x12354;
int32_t x12356 = x12352;
int32_t x12357 = x12353;
for(int x12358=0; x12358 < 4; x12358++) {
int32_t x12359 = x12356;
int32_t x12360 = x12357;
int32_t x12361 = x12355;
int32_t x12362 = x12361;
int32_t x12363 = x12359;
int32_t x12364 = x12360;
for(int x12365=0; x12365 < 4; x12365++) {
int32_t x12366 = x12363;
float x12367 = x12291[x12366];
int32_t x12368 = x12364;
float x12369 = x11485[x12368];
float x12370 = x12367 + x12369;
x12291[x12366] = x12370;
x12362 += 1;
x12363 += 1;
x12364 += 1;

}
x12355 += 4;
x12356 += 4;
x12357 += 4;

}
x12348 += 16;
x12349 += 16;
x12350 += 16;

}
x12341 += 16384;
x12342 += 16384;
x12343 += 16384;

}
float* x12392 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x12393=0; x12393 < 1048576; x12393++) {
float x12394 = x12291[x12393];
bool x12395 = x12394 < 0.0f;
if (x12395) {
x12392[x12393] = 0.0f;
} else {
float x12398 = x12291[x12393];
x12392[x12393] = x12398;
}

}
float* x12404 = (float*)myMalloc(262144 * sizeof(float));;
float* x12405 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x12406=0; x12406 < 64; x12406++) {
int32_t x12407 = x12406 * 16384;
float* x12408 = x12392+x12407;
int32_t x12409 = x12406 * 4096;
float* x12410 = x12404+x12409;
float* x12411 = x12405+x12407;
for(int x12412=0; x12412 < 1024; x12412++) {
int32_t x12413 = x12412 / 1;
int32_t x12417 = x12413 * 4;
int32_t x12418 = x12417 * 4;
int32_t x12414 = x12412 % 1;
int32_t x12415 = x12414 / 1;
int32_t x12419 = x12415 * 4;
int32_t x12420 = x12419 * 4;
int32_t x12421 = x12418 + x12420;
int32_t x12416 = x12414 % 1;
int32_t x12422 = x12416 * 4;
int32_t x12423 = x12422 * 4;
int32_t x12424 = x12421 + x12423;
float* x12425 = x12411+x12424;
float* x12426 = x12408+x12418;
for(int x12427=0; x12427 < 4; x12427++) {
int32_t x12429 = x12427 * 4;
float* x12430 = x12425+x12429;
int32_t x12428 = x12427 + x12415;
int32_t x12431 = x12428 * 4;
int32_t x12432 = x12431 + x12416;
float* x12433 = x12426+x12432;
memcpy(x12430, x12433, 4 * 4);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,16,1024,1,x150,1024,x12411,16,1,x12410,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x12443 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x12444 = 0;
int32_t x12445 = 0;
int32_t x12446 = 0;
for(int x12447=0; x12447 < 64; x12447++) {
int32_t x12448 = x12445;
int32_t x12449 = x12446;
int32_t x12450 = x12444;
int32_t x12451 = x12450;
int32_t x12452 = x12448;
int32_t x12453 = x12449;
for(int x12454=0; x12454 < 256; x12454++) {
int32_t x12455 = x12452;
int32_t x12456 = x12453;
int32_t x12457 = x12451;
int32_t x12458 = x12457;
int32_t x12459 = x12455;
int32_t x12460 = x12456;
for(int x12461=0; x12461 < 4; x12461++) {
int32_t x12462 = x12459;
int32_t x12463 = x12460;
int32_t x12464 = x12458;
int32_t x12465 = x12464;
int32_t x12466 = x12462;
int32_t x12467 = x12463;
for(int x12468=0; x12468 < 4; x12468++) {
int32_t x12469 = x12465;
int32_t x12470 = x12466;
float x12471 = x12404[x12470];
int32_t x12472 = x12467;
float x12473 = x79[x12472];
float x12474 = x12471 - x12473;
x12443[x12469] = x12474;
x12465 += 1;
x12466 += 1;

}
x12458 += 4;
x12459 += 4;

}
x12451 += 16;
x12452 += 16;
x12453 += 1;

}
x12444 += 4096;
x12445 += 4096;

}
float* x12493 = (float*)myMalloc(256 * sizeof(float));;
for(int x12494=0; x12494 < 256; x12494++) {
float x12495 = x175[x12494];
float x12496 = x12495 + 1.0E-5f;
x12493[x12494] = x12496;

}
float* x12500 = (float*)myMalloc(256 * sizeof(float));;
for(int x12501=0; x12501 < 256; x12501++) {
float x12502 = x12493[x12501];
double x12503 = (double)x12502;
double x12504 = sqrt(x12503);
float x12505 = (float)x12504;
x12500[x12501] = x12505;

}
// resize to WrappedArray(-1, 1, 1)
float* x12510 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x12511 = 0;
int32_t x12512 = 0;
int32_t x12513 = 0;
for(int x12514=0; x12514 < 64; x12514++) {
int32_t x12515 = x12512;
int32_t x12516 = x12513;
int32_t x12517 = x12511;
int32_t x12518 = x12517;
int32_t x12519 = x12515;
int32_t x12520 = x12516;
for(int x12521=0; x12521 < 256; x12521++) {
int32_t x12522 = x12519;
int32_t x12523 = x12520;
int32_t x12524 = x12518;
int32_t x12525 = x12524;
int32_t x12526 = x12522;
int32_t x12527 = x12523;
for(int x12528=0; x12528 < 4; x12528++) {
int32_t x12529 = x12526;
int32_t x12530 = x12527;
int32_t x12531 = x12525;
int32_t x12532 = x12531;
int32_t x12533 = x12529;
int32_t x12534 = x12530;
for(int x12535=0; x12535 < 4; x12535++) {
int32_t x12536 = x12532;
int32_t x12537 = x12533;
float x12538 = x12443[x12537];
int32_t x12539 = x12534;
float x12540 = x12500[x12539];
float x12541 = x12538 / x12540;
x12510[x12536] = x12541;
x12532 += 1;
x12533 += 1;

}
x12525 += 4;
x12526 += 4;

}
x12518 += 16;
x12519 += 16;
x12520 += 1;

}
x12511 += 4096;
x12512 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x12561 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x12562 = 0;
int32_t x12563 = 0;
int32_t x12564 = 0;
for(int x12565=0; x12565 < 64; x12565++) {
int32_t x12566 = x12563;
int32_t x12567 = x12564;
int32_t x12568 = x12562;
int32_t x12569 = x12568;
int32_t x12570 = x12566;
int32_t x12571 = x12567;
for(int x12572=0; x12572 < 256; x12572++) {
int32_t x12573 = x12570;
int32_t x12574 = x12571;
int32_t x12575 = x12569;
int32_t x12576 = x12575;
int32_t x12577 = x12573;
int32_t x12578 = x12574;
for(int x12579=0; x12579 < 4; x12579++) {
int32_t x12580 = x12577;
int32_t x12581 = x12578;
int32_t x12582 = x12576;
int32_t x12583 = x12582;
int32_t x12584 = x12580;
int32_t x12585 = x12581;
for(int x12586=0; x12586 < 4; x12586++) {
int32_t x12587 = x12583;
int32_t x12588 = x12584;
float x12589 = x12510[x12588];
int32_t x12590 = x12585;
float x12591 = x84[x12590];
float x12592 = x12589 * x12591;
x12561[x12587] = x12592;
x12583 += 1;
x12584 += 1;

}
x12576 += 4;
x12577 += 4;

}
x12569 += 16;
x12570 += 16;
x12571 += 1;

}
x12562 += 4096;
x12563 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x12612 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x12613 = 0;
int32_t x12614 = 0;
int32_t x12615 = 0;
for(int x12616=0; x12616 < 64; x12616++) {
int32_t x12617 = x12614;
int32_t x12618 = x12615;
int32_t x12619 = x12613;
int32_t x12620 = x12619;
int32_t x12621 = x12617;
int32_t x12622 = x12618;
for(int x12623=0; x12623 < 256; x12623++) {
int32_t x12624 = x12621;
int32_t x12625 = x12622;
int32_t x12626 = x12620;
int32_t x12627 = x12626;
int32_t x12628 = x12624;
int32_t x12629 = x12625;
for(int x12630=0; x12630 < 4; x12630++) {
int32_t x12631 = x12628;
int32_t x12632 = x12629;
int32_t x12633 = x12627;
int32_t x12634 = x12633;
int32_t x12635 = x12631;
int32_t x12636 = x12632;
for(int x12637=0; x12637 < 4; x12637++) {
int32_t x12638 = x12634;
int32_t x12639 = x12635;
float x12640 = x12561[x12639];
int32_t x12641 = x12636;
float x12642 = x252[x12641];
float x12643 = x12640 + x12642;
x12612[x12638] = x12643;
x12634 += 1;
x12635 += 1;

}
x12627 += 4;
x12628 += 4;

}
x12620 += 16;
x12621 += 16;
x12622 += 1;

}
x12613 += 4096;
x12614 += 4096;

}
float* x12662 = (float*)myMalloc(262144 * sizeof(float));;
for(int x12663=0; x12663 < 262144; x12663++) {
float x12664 = x12612[x12663];
bool x12665 = x12664 < 0.0f;
if (x12665) {
x12662[x12663] = 0.0f;
} else {
float x12668 = x12612[x12663];
x12662[x12663] = x12668;
}

}
float* x12674 = (float*)myMalloc(262144 * sizeof(float));;
float* x12675 = (float*)myMalloc(2359296 * sizeof(float));;
for(int x12676=0; x12676 < 64; x12676++) {
int32_t x12677 = x12676 * 4096;
float* x12678 = x12662+x12677;
float* x12679 = x12674+x12677;
int32_t x12680 = x12676 * 36864;
float* x12681 = x12675+x12680;
for(int x12682=0; x12682 < 2304; x12682++) {
int32_t x12683 = x12682 / 9;
int32_t x12687 = x12683 * 3;
int32_t x12688 = x12687 * 3;
int32_t x12689 = x12688 * 4;
int32_t x12690 = x12689 * 4;
int32_t x12684 = x12682 % 9;
int32_t x12685 = x12684 / 3;
int32_t x12691 = x12685 * 3;
int32_t x12692 = x12691 * 4;
int32_t x12693 = x12692 * 4;
int32_t x12694 = x12690 + x12693;
int32_t x12686 = x12684 % 3;
int32_t x12695 = x12686 * 4;
int32_t x12696 = x12695 * 4;
int32_t x12697 = x12694 + x12696;
float* x12698 = x12681+x12697;
int32_t x12699 = x12683 * 4;
int32_t x12700 = x12699 * 4;
float* x12701 = x12678+x12700;
int32_t x12713 = 1 - x12686;
bool x12714 = x12713 > 0;
int32_t x12715;
if (x12714) {
x12715 = x12713;
} else {
x12715 = 0;
}
int32_t x12716 = 3 - x12686;
int32_t x12717 = x12716 - 1;
int32_t x12718 = 1 - x12717;
bool x12719 = x12718 > 0;
int32_t x12720;
if (x12719) {
x12720 = x12718;
} else {
x12720 = 0;
}
int32_t x12721 = 4 - x12720;
int32_t x12722 = x12721 - x12715;
bool x12723 = x12722 <= 0;
bool x12727 = x12715 > 0;
int32_t x12712 = -1 + x12686;
bool x12740 = x12720 > 0;
for(int x12702=0; x12702 < 4; x12702++) {
int32_t x12703 = x12702 - 1;
int32_t x12704 = x12703 + x12685;
bool x12705 = x12704 < 0;
bool x12706 = x12704 >= 4;
bool x12707 = x12705 || x12706;
if (x12707) {
int32_t x12708 = x12702 * 4;
float* x12709 = x12698+x12708;
memset(x12709, 0, 4 * 4);;
} else {
if (x12723) {
int32_t x12708 = x12702 * 4;
float* x12724 = x12698+x12708;
memset(x12724, 0, 4 * 4);;
} else {
int32_t x12708 = x12702 * 4;
if (x12727) {
float* x12728 = x12698+x12708;
memset(x12728, 0, 4 * x12715);;
} else {
}
// may have segfault here
int32_t x12733 = x12708 + x12715;
float* x12734 = x12698+x12733;
int32_t x12735 = x12704 * 4;
int32_t x12736 = x12735 + x12712;
int32_t x12737 = x12736 + x12715;
float* x12738 = x12701+x12737;
memcpy(x12734, x12738, 4 * x12722);;
if (x12740) {
int32_t x12741 = x12708 + 4;
int32_t x12742 = x12741 - x12720;
float* x12743 = x12698+x12742;
memset(x12743, 0, 4 * x12720);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,16,2304,1,x225,2304,x12681,16,1,x12679,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x12759 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x12760 = 0;
int32_t x12761 = 0;
int32_t x12762 = 0;
for(int x12763=0; x12763 < 64; x12763++) {
int32_t x12764 = x12761;
int32_t x12765 = x12762;
int32_t x12766 = x12760;
int32_t x12767 = x12766;
int32_t x12768 = x12764;
int32_t x12769 = x12765;
for(int x12770=0; x12770 < 256; x12770++) {
int32_t x12771 = x12768;
int32_t x12772 = x12769;
int32_t x12773 = x12767;
int32_t x12774 = x12773;
int32_t x12775 = x12771;
int32_t x12776 = x12772;
for(int x12777=0; x12777 < 4; x12777++) {
int32_t x12778 = x12775;
int32_t x12779 = x12776;
int32_t x12780 = x12774;
int32_t x12781 = x12780;
int32_t x12782 = x12778;
int32_t x12783 = x12779;
for(int x12784=0; x12784 < 4; x12784++) {
int32_t x12785 = x12781;
int32_t x12786 = x12782;
float x12787 = x12674[x12786];
int32_t x12788 = x12783;
float x12789 = x69[x12788];
float x12790 = x12787 - x12789;
x12759[x12785] = x12790;
x12781 += 1;
x12782 += 1;

}
x12774 += 4;
x12775 += 4;

}
x12767 += 16;
x12768 += 16;
x12769 += 1;

}
x12760 += 4096;
x12761 += 4096;

}
float* x12809 = (float*)myMalloc(256 * sizeof(float));;
for(int x12810=0; x12810 < 256; x12810++) {
float x12811 = x239[x12810];
float x12812 = x12811 + 1.0E-5f;
x12809[x12810] = x12812;

}
float* x12816 = (float*)myMalloc(256 * sizeof(float));;
for(int x12817=0; x12817 < 256; x12817++) {
float x12818 = x12809[x12817];
double x12819 = (double)x12818;
double x12820 = sqrt(x12819);
float x12821 = (float)x12820;
x12816[x12817] = x12821;

}
// resize to WrappedArray(-1, 1, 1)
float* x12826 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x12827 = 0;
int32_t x12828 = 0;
int32_t x12829 = 0;
for(int x12830=0; x12830 < 64; x12830++) {
int32_t x12831 = x12828;
int32_t x12832 = x12829;
int32_t x12833 = x12827;
int32_t x12834 = x12833;
int32_t x12835 = x12831;
int32_t x12836 = x12832;
for(int x12837=0; x12837 < 256; x12837++) {
int32_t x12838 = x12835;
int32_t x12839 = x12836;
int32_t x12840 = x12834;
int32_t x12841 = x12840;
int32_t x12842 = x12838;
int32_t x12843 = x12839;
for(int x12844=0; x12844 < 4; x12844++) {
int32_t x12845 = x12842;
int32_t x12846 = x12843;
int32_t x12847 = x12841;
int32_t x12848 = x12847;
int32_t x12849 = x12845;
int32_t x12850 = x12846;
for(int x12851=0; x12851 < 4; x12851++) {
int32_t x12852 = x12848;
int32_t x12853 = x12849;
float x12854 = x12759[x12853];
int32_t x12855 = x12850;
float x12856 = x12816[x12855];
float x12857 = x12854 / x12856;
x12826[x12852] = x12857;
x12848 += 1;
x12849 += 1;

}
x12841 += 4;
x12842 += 4;

}
x12834 += 16;
x12835 += 16;
x12836 += 1;

}
x12827 += 4096;
x12828 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x12877 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x12878 = 0;
int32_t x12879 = 0;
int32_t x12880 = 0;
for(int x12881=0; x12881 < 64; x12881++) {
int32_t x12882 = x12879;
int32_t x12883 = x12880;
int32_t x12884 = x12878;
int32_t x12885 = x12884;
int32_t x12886 = x12882;
int32_t x12887 = x12883;
for(int x12888=0; x12888 < 256; x12888++) {
int32_t x12889 = x12886;
int32_t x12890 = x12887;
int32_t x12891 = x12885;
int32_t x12892 = x12891;
int32_t x12893 = x12889;
int32_t x12894 = x12890;
for(int x12895=0; x12895 < 4; x12895++) {
int32_t x12896 = x12893;
int32_t x12897 = x12894;
int32_t x12898 = x12892;
int32_t x12899 = x12898;
int32_t x12900 = x12896;
int32_t x12901 = x12897;
for(int x12902=0; x12902 < 4; x12902++) {
int32_t x12903 = x12899;
int32_t x12904 = x12900;
float x12905 = x12826[x12904];
int32_t x12906 = x12901;
float x12907 = x140[x12906];
float x12908 = x12905 * x12907;
x12877[x12903] = x12908;
x12899 += 1;
x12900 += 1;

}
x12892 += 4;
x12893 += 4;

}
x12885 += 16;
x12886 += 16;
x12887 += 1;

}
x12878 += 4096;
x12879 += 4096;

}
// resize to WrappedArray(-1, 1, 1)
float* x12928 = (float*)myMalloc(262144 * sizeof(float));;
int32_t x12929 = 0;
int32_t x12930 = 0;
int32_t x12931 = 0;
for(int x12932=0; x12932 < 64; x12932++) {
int32_t x12933 = x12930;
int32_t x12934 = x12931;
int32_t x12935 = x12929;
int32_t x12936 = x12935;
int32_t x12937 = x12933;
int32_t x12938 = x12934;
for(int x12939=0; x12939 < 256; x12939++) {
int32_t x12940 = x12937;
int32_t x12941 = x12938;
int32_t x12942 = x12936;
int32_t x12943 = x12942;
int32_t x12944 = x12940;
int32_t x12945 = x12941;
for(int x12946=0; x12946 < 4; x12946++) {
int32_t x12947 = x12944;
int32_t x12948 = x12945;
int32_t x12949 = x12943;
int32_t x12950 = x12949;
int32_t x12951 = x12947;
int32_t x12952 = x12948;
for(int x12953=0; x12953 < 4; x12953++) {
int32_t x12954 = x12950;
int32_t x12955 = x12951;
float x12956 = x12877[x12955];
int32_t x12957 = x12952;
float x12958 = x188[x12957];
float x12959 = x12956 + x12958;
x12928[x12954] = x12959;
x12950 += 1;
x12951 += 1;

}
x12943 += 4;
x12944 += 4;

}
x12936 += 16;
x12937 += 16;
x12938 += 1;

}
x12929 += 4096;
x12930 += 4096;

}
float* x12978 = (float*)myMalloc(262144 * sizeof(float));;
for(int x12979=0; x12979 < 262144; x12979++) {
float x12980 = x12928[x12979];
bool x12981 = x12980 < 0.0f;
if (x12981) {
x12978[x12979] = 0.0f;
} else {
float x12984 = x12928[x12979];
x12978[x12979] = x12984;
}

}
float* x12990 = (float*)myMalloc(1048576 * sizeof(float));;
float* x12991 = (float*)myMalloc(262144 * sizeof(float));;
for(int x12992=0; x12992 < 64; x12992++) {
int32_t x12993 = x12992 * 4096;
float* x12994 = x12978+x12993;
int32_t x12995 = x12992 * 16384;
float* x12996 = x12990+x12995;
float* x12997 = x12991+x12993;
for(int x12998=0; x12998 < 256; x12998++) {
int32_t x12999 = x12998 / 1;
int32_t x13003 = x12999 * 4;
int32_t x13004 = x13003 * 4;
int32_t x13000 = x12998 % 1;
int32_t x13001 = x13000 / 1;
int32_t x13005 = x13001 * 4;
int32_t x13006 = x13005 * 4;
int32_t x13007 = x13004 + x13006;
int32_t x13002 = x13000 % 1;
int32_t x13008 = x13002 * 4;
int32_t x13009 = x13008 * 4;
int32_t x13010 = x13007 + x13009;
float* x13011 = x12997+x13010;
float* x13012 = x12994+x13004;
for(int x13013=0; x13013 < 4; x13013++) {
int32_t x13015 = x13013 * 4;
float* x13016 = x13011+x13015;
int32_t x13014 = x13013 + x13001;
int32_t x13017 = x13014 * 4;
int32_t x13018 = x13017 + x13002;
float* x13019 = x13012+x13018;
memcpy(x13016, x13019, 4 * 4);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,16,256,1,x96,256,x12997,16,1,x12996,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x13029 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x13030 = 0;
int32_t x13031 = 0;
int32_t x13032 = 0;
for(int x13033=0; x13033 < 64; x13033++) {
int32_t x13034 = x13031;
int32_t x13035 = x13032;
int32_t x13036 = x13030;
int32_t x13037 = x13036;
int32_t x13038 = x13034;
int32_t x13039 = x13035;
for(int x13040=0; x13040 < 1024; x13040++) {
int32_t x13041 = x13038;
int32_t x13042 = x13039;
int32_t x13043 = x13037;
int32_t x13044 = x13043;
int32_t x13045 = x13041;
int32_t x13046 = x13042;
for(int x13047=0; x13047 < 4; x13047++) {
int32_t x13048 = x13045;
int32_t x13049 = x13046;
int32_t x13050 = x13044;
int32_t x13051 = x13050;
int32_t x13052 = x13048;
int32_t x13053 = x13049;
for(int x13054=0; x13054 < 4; x13054++) {
int32_t x13055 = x13051;
int32_t x13056 = x13052;
float x13057 = x12990[x13056];
int32_t x13058 = x13053;
float x13059 = x121[x13058];
float x13060 = x13057 - x13059;
x13029[x13055] = x13060;
x13051 += 1;
x13052 += 1;

}
x13044 += 4;
x13045 += 4;

}
x13037 += 16;
x13038 += 16;
x13039 += 1;

}
x13030 += 16384;
x13031 += 16384;

}
float* x13079 = (float*)myMalloc(1024 * sizeof(float));;
for(int x13080=0; x13080 < 1024; x13080++) {
float x13081 = x182[x13080];
float x13082 = x13081 + 1.0E-5f;
x13079[x13080] = x13082;

}
float* x13086 = (float*)myMalloc(1024 * sizeof(float));;
for(int x13087=0; x13087 < 1024; x13087++) {
float x13088 = x13079[x13087];
double x13089 = (double)x13088;
double x13090 = sqrt(x13089);
float x13091 = (float)x13090;
x13086[x13087] = x13091;

}
// resize to WrappedArray(-1, 1, 1)
float* x13096 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x13097 = 0;
int32_t x13098 = 0;
int32_t x13099 = 0;
for(int x13100=0; x13100 < 64; x13100++) {
int32_t x13101 = x13098;
int32_t x13102 = x13099;
int32_t x13103 = x13097;
int32_t x13104 = x13103;
int32_t x13105 = x13101;
int32_t x13106 = x13102;
for(int x13107=0; x13107 < 1024; x13107++) {
int32_t x13108 = x13105;
int32_t x13109 = x13106;
int32_t x13110 = x13104;
int32_t x13111 = x13110;
int32_t x13112 = x13108;
int32_t x13113 = x13109;
for(int x13114=0; x13114 < 4; x13114++) {
int32_t x13115 = x13112;
int32_t x13116 = x13113;
int32_t x13117 = x13111;
int32_t x13118 = x13117;
int32_t x13119 = x13115;
int32_t x13120 = x13116;
for(int x13121=0; x13121 < 4; x13121++) {
int32_t x13122 = x13118;
int32_t x13123 = x13119;
float x13124 = x13029[x13123];
int32_t x13125 = x13120;
float x13126 = x13086[x13125];
float x13127 = x13124 / x13126;
x13096[x13122] = x13127;
x13118 += 1;
x13119 += 1;

}
x13111 += 4;
x13112 += 4;

}
x13104 += 16;
x13105 += 16;
x13106 += 1;

}
x13097 += 16384;
x13098 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x13147 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x13148 = 0;
int32_t x13149 = 0;
int32_t x13150 = 0;
for(int x13151=0; x13151 < 64; x13151++) {
int32_t x13152 = x13149;
int32_t x13153 = x13150;
int32_t x13154 = x13148;
int32_t x13155 = x13154;
int32_t x13156 = x13152;
int32_t x13157 = x13153;
for(int x13158=0; x13158 < 1024; x13158++) {
int32_t x13159 = x13156;
int32_t x13160 = x13157;
int32_t x13161 = x13155;
int32_t x13162 = x13161;
int32_t x13163 = x13159;
int32_t x13164 = x13160;
for(int x13165=0; x13165 < 4; x13165++) {
int32_t x13166 = x13163;
int32_t x13167 = x13164;
int32_t x13168 = x13162;
int32_t x13169 = x13168;
int32_t x13170 = x13166;
int32_t x13171 = x13167;
for(int x13172=0; x13172 < 4; x13172++) {
int32_t x13173 = x13169;
int32_t x13174 = x13170;
float x13175 = x13096[x13174];
int32_t x13176 = x13171;
float x13177 = x247[x13176];
float x13178 = x13175 * x13177;
x13147[x13173] = x13178;
x13169 += 1;
x13170 += 1;

}
x13162 += 4;
x13163 += 4;

}
x13155 += 16;
x13156 += 16;
x13157 += 1;

}
x13148 += 16384;
x13149 += 16384;

}
// resize to WrappedArray(-1, 1, 1)
float* x13198 = (float*)myMalloc(1048576 * sizeof(float));;
int32_t x13199 = 0;
int32_t x13200 = 0;
int32_t x13201 = 0;
for(int x13202=0; x13202 < 64; x13202++) {
int32_t x13203 = x13200;
int32_t x13204 = x13201;
int32_t x13205 = x13199;
int32_t x13206 = x13205;
int32_t x13207 = x13203;
int32_t x13208 = x13204;
for(int x13209=0; x13209 < 1024; x13209++) {
int32_t x13210 = x13207;
int32_t x13211 = x13208;
int32_t x13212 = x13206;
int32_t x13213 = x13212;
int32_t x13214 = x13210;
int32_t x13215 = x13211;
for(int x13216=0; x13216 < 4; x13216++) {
int32_t x13217 = x13214;
int32_t x13218 = x13215;
int32_t x13219 = x13213;
int32_t x13220 = x13219;
int32_t x13221 = x13217;
int32_t x13222 = x13218;
for(int x13223=0; x13223 < 4; x13223++) {
int32_t x13224 = x13220;
int32_t x13225 = x13221;
float x13226 = x13147[x13225];
int32_t x13227 = x13222;
float x13228 = x92[x13227];
float x13229 = x13226 + x13228;
x13198[x13224] = x13229;
x13220 += 1;
x13221 += 1;

}
x13213 += 4;
x13214 += 4;

}
x13206 += 16;
x13207 += 16;
x13208 += 1;

}
x13199 += 16384;
x13200 += 16384;

}
int32_t x13248 = 0;
int32_t x13249 = 0;
int32_t x13250 = 0;
for(int x13251=0; x13251 < 64; x13251++) {
int32_t x13252 = x13249;
int32_t x13253 = x13250;
int32_t x13254 = x13248;
int32_t x13255 = x13254;
int32_t x13256 = x13252;
int32_t x13257 = x13253;
for(int x13258=0; x13258 < 1024; x13258++) {
int32_t x13259 = x13256;
int32_t x13260 = x13257;
int32_t x13261 = x13255;
int32_t x13262 = x13261;
int32_t x13263 = x13259;
int32_t x13264 = x13260;
for(int x13265=0; x13265 < 4; x13265++) {
int32_t x13266 = x13263;
int32_t x13267 = x13264;
int32_t x13268 = x13262;
int32_t x13269 = x13268;
int32_t x13270 = x13266;
int32_t x13271 = x13267;
for(int x13272=0; x13272 < 4; x13272++) {
int32_t x13273 = x13270;
float x13274 = x13198[x13273];
int32_t x13275 = x13271;
float x13276 = x12392[x13275];
float x13277 = x13274 + x13276;
x13198[x13273] = x13277;
x13269 += 1;
x13270 += 1;
x13271 += 1;

}
x13262 += 4;
x13263 += 4;
x13264 += 4;

}
x13255 += 16;
x13256 += 16;
x13257 += 16;

}
x13248 += 16384;
x13249 += 16384;
x13250 += 16384;

}
float* x13299 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x13300=0; x13300 < 1048576; x13300++) {
float x13301 = x13198[x13300];
bool x13302 = x13301 < 0.0f;
if (x13302) {
x13299[x13300] = 0.0f;
} else {
float x13305 = x13198[x13300];
x13299[x13300] = x13305;
}

}
float* x13311 = (float*)myMalloc(524288 * sizeof(float));;
float* x13312 = (float*)myMalloc(1048576 * sizeof(float));;
for(int x13313=0; x13313 < 64; x13313++) {
int32_t x13314 = x13313 * 16384;
float* x13315 = x13299+x13314;
int32_t x13316 = x13313 * 8192;
float* x13317 = x13311+x13316;
float* x13318 = x13312+x13314;
for(int x13319=0; x13319 < 1024; x13319++) {
int32_t x13320 = x13319 / 1;
int32_t x13324 = x13320 * 4;
int32_t x13325 = x13324 * 4;
int32_t x13321 = x13319 % 1;
int32_t x13322 = x13321 / 1;
int32_t x13326 = x13322 * 4;
int32_t x13327 = x13326 * 4;
int32_t x13328 = x13325 + x13327;
int32_t x13323 = x13321 % 1;
int32_t x13329 = x13323 * 4;
int32_t x13330 = x13329 * 4;
int32_t x13331 = x13328 + x13330;
float* x13332 = x13318+x13331;
float* x13333 = x13315+x13325;
for(int x13334=0; x13334 < 4; x13334++) {
int32_t x13336 = x13334 * 4;
float* x13337 = x13332+x13336;
int32_t x13335 = x13334 + x13322;
int32_t x13338 = x13335 * 4;
int32_t x13339 = x13338 + x13323;
float* x13340 = x13333+x13339;
memcpy(x13337, x13340, 4 * 4);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,16,1024,1,x138,1024,x13318,16,1,x13317,16);

}
// resize to WrappedArray(-1, 1, 1)
float* x13350 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x13351 = 0;
int32_t x13352 = 0;
int32_t x13353 = 0;
for(int x13354=0; x13354 < 64; x13354++) {
int32_t x13355 = x13352;
int32_t x13356 = x13353;
int32_t x13357 = x13351;
int32_t x13358 = x13357;
int32_t x13359 = x13355;
int32_t x13360 = x13356;
for(int x13361=0; x13361 < 512; x13361++) {
int32_t x13362 = x13359;
int32_t x13363 = x13360;
int32_t x13364 = x13358;
int32_t x13365 = x13364;
int32_t x13366 = x13362;
int32_t x13367 = x13363;
for(int x13368=0; x13368 < 4; x13368++) {
int32_t x13369 = x13366;
int32_t x13370 = x13367;
int32_t x13371 = x13365;
int32_t x13372 = x13371;
int32_t x13373 = x13369;
int32_t x13374 = x13370;
for(int x13375=0; x13375 < 4; x13375++) {
int32_t x13376 = x13372;
int32_t x13377 = x13373;
float x13378 = x13311[x13377];
int32_t x13379 = x13374;
float x13380 = x66[x13379];
float x13381 = x13378 - x13380;
x13350[x13376] = x13381;
x13372 += 1;
x13373 += 1;

}
x13365 += 4;
x13366 += 4;

}
x13358 += 16;
x13359 += 16;
x13360 += 1;

}
x13351 += 8192;
x13352 += 8192;

}
float* x13400 = (float*)myMalloc(512 * sizeof(float));;
for(int x13401=0; x13401 < 512; x13401++) {
float x13402 = x120[x13401];
float x13403 = x13402 + 1.0E-5f;
x13400[x13401] = x13403;

}
float* x13407 = (float*)myMalloc(512 * sizeof(float));;
for(int x13408=0; x13408 < 512; x13408++) {
float x13409 = x13400[x13408];
double x13410 = (double)x13409;
double x13411 = sqrt(x13410);
float x13412 = (float)x13411;
x13407[x13408] = x13412;

}
// resize to WrappedArray(-1, 1, 1)
float* x13417 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x13418 = 0;
int32_t x13419 = 0;
int32_t x13420 = 0;
for(int x13421=0; x13421 < 64; x13421++) {
int32_t x13422 = x13419;
int32_t x13423 = x13420;
int32_t x13424 = x13418;
int32_t x13425 = x13424;
int32_t x13426 = x13422;
int32_t x13427 = x13423;
for(int x13428=0; x13428 < 512; x13428++) {
int32_t x13429 = x13426;
int32_t x13430 = x13427;
int32_t x13431 = x13425;
int32_t x13432 = x13431;
int32_t x13433 = x13429;
int32_t x13434 = x13430;
for(int x13435=0; x13435 < 4; x13435++) {
int32_t x13436 = x13433;
int32_t x13437 = x13434;
int32_t x13438 = x13432;
int32_t x13439 = x13438;
int32_t x13440 = x13436;
int32_t x13441 = x13437;
for(int x13442=0; x13442 < 4; x13442++) {
int32_t x13443 = x13439;
int32_t x13444 = x13440;
float x13445 = x13350[x13444];
int32_t x13446 = x13441;
float x13447 = x13407[x13446];
float x13448 = x13445 / x13447;
x13417[x13443] = x13448;
x13439 += 1;
x13440 += 1;

}
x13432 += 4;
x13433 += 4;

}
x13425 += 16;
x13426 += 16;
x13427 += 1;

}
x13418 += 8192;
x13419 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x13468 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x13469 = 0;
int32_t x13470 = 0;
int32_t x13471 = 0;
for(int x13472=0; x13472 < 64; x13472++) {
int32_t x13473 = x13470;
int32_t x13474 = x13471;
int32_t x13475 = x13469;
int32_t x13476 = x13475;
int32_t x13477 = x13473;
int32_t x13478 = x13474;
for(int x13479=0; x13479 < 512; x13479++) {
int32_t x13480 = x13477;
int32_t x13481 = x13478;
int32_t x13482 = x13476;
int32_t x13483 = x13482;
int32_t x13484 = x13480;
int32_t x13485 = x13481;
for(int x13486=0; x13486 < 4; x13486++) {
int32_t x13487 = x13484;
int32_t x13488 = x13485;
int32_t x13489 = x13483;
int32_t x13490 = x13489;
int32_t x13491 = x13487;
int32_t x13492 = x13488;
for(int x13493=0; x13493 < 4; x13493++) {
int32_t x13494 = x13490;
int32_t x13495 = x13491;
float x13496 = x13417[x13495];
int32_t x13497 = x13492;
float x13498 = x200[x13497];
float x13499 = x13496 * x13498;
x13468[x13494] = x13499;
x13490 += 1;
x13491 += 1;

}
x13483 += 4;
x13484 += 4;

}
x13476 += 16;
x13477 += 16;
x13478 += 1;

}
x13469 += 8192;
x13470 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x13519 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x13520 = 0;
int32_t x13521 = 0;
int32_t x13522 = 0;
for(int x13523=0; x13523 < 64; x13523++) {
int32_t x13524 = x13521;
int32_t x13525 = x13522;
int32_t x13526 = x13520;
int32_t x13527 = x13526;
int32_t x13528 = x13524;
int32_t x13529 = x13525;
for(int x13530=0; x13530 < 512; x13530++) {
int32_t x13531 = x13528;
int32_t x13532 = x13529;
int32_t x13533 = x13527;
int32_t x13534 = x13533;
int32_t x13535 = x13531;
int32_t x13536 = x13532;
for(int x13537=0; x13537 < 4; x13537++) {
int32_t x13538 = x13535;
int32_t x13539 = x13536;
int32_t x13540 = x13534;
int32_t x13541 = x13540;
int32_t x13542 = x13538;
int32_t x13543 = x13539;
for(int x13544=0; x13544 < 4; x13544++) {
int32_t x13545 = x13541;
int32_t x13546 = x13542;
float x13547 = x13468[x13546];
int32_t x13548 = x13543;
float x13549 = x223[x13548];
float x13550 = x13547 + x13549;
x13519[x13545] = x13550;
x13541 += 1;
x13542 += 1;

}
x13534 += 4;
x13535 += 4;

}
x13527 += 16;
x13528 += 16;
x13529 += 1;

}
x13520 += 8192;
x13521 += 8192;

}
float* x13569 = (float*)myMalloc(524288 * sizeof(float));;
for(int x13570=0; x13570 < 524288; x13570++) {
float x13571 = x13519[x13570];
bool x13572 = x13571 < 0.0f;
if (x13572) {
x13569[x13570] = 0.0f;
} else {
float x13575 = x13519[x13570];
x13569[x13570] = x13575;
}

}
float* x13581 = (float*)myMalloc(131072 * sizeof(float));;
float* x13582 = (float*)myMalloc(1179648 * sizeof(float));;
for(int x13583=0; x13583 < 64; x13583++) {
int32_t x13584 = x13583 * 8192;
float* x13585 = x13569+x13584;
int32_t x13586 = x13583 * 2048;
float* x13587 = x13581+x13586;
int32_t x13588 = x13583 * 18432;
float* x13589 = x13582+x13588;
for(int x13591=0; x13591 < 4608; x13591++) {
int32_t x13592 = x13591 / 9;
int32_t x13596 = x13592 * 3;
int32_t x13597 = x13596 * 3;
int32_t x13598 = x13597 * 2;
int32_t x13599 = x13598 * 2;
int32_t x13593 = x13591 % 9;
int32_t x13594 = x13593 / 3;
int32_t x13600 = x13594 * 3;
int32_t x13601 = x13600 * 2;
int32_t x13602 = x13601 * 2;
int32_t x13603 = x13599 + x13602;
int32_t x13595 = x13593 % 3;
int32_t x13604 = x13595 * 2;
int32_t x13605 = x13604 * 2;
int32_t x13606 = x13603 + x13605;
float* x13607 = x13589+x13606;
int32_t x13608 = x13592 * 4;
int32_t x13609 = x13608 * 4;
float* x13610 = x13585+x13609;
for(int x13612=0; x13612 < 2; x13612++) {
int32_t x13613 = x13612 * 2;
int32_t x13614 = x13613 - 1;
int32_t x13615 = x13614 + x13594;
bool x13616 = x13615 < 0;
bool x13617 = x13615 >= 4;
bool x13618 = x13616 || x13617;
if (x13618) {
float* x13619 = x13607+x13613;
memset(x13619, 0, 4 * 2);;
} else {
int32_t x13634 = x13615 * 4;
for(int x13622=0; x13622 < 2; x13622++) {
int32_t x13623 = x13622 * 2;
int32_t x13624 = x13623 - 1;
int32_t x13625 = x13624 + x13595;
bool x13626 = x13625 < 0;
bool x13627 = x13625 >= 4;
bool x13628 = x13626 || x13627;
if (x13628) {
int32_t x13629 = x13613 + x13622;
float* x13630 = x13607+x13629;
memset(x13630, 0, 4 * 1);;
} else {
int32_t x13629 = x13613 + x13622;
float* x13633 = x13607+x13629;
int32_t x13635 = x13634 + x13625;
float* x13636 = x13610+x13635;
memcpy(x13633, x13636, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,4,4608,1,x33,4608,x13589,4,1,x13587,4);

}
// resize to WrappedArray(-1, 1, 1)
float* x13652 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x13653 = 0;
int32_t x13654 = 0;
int32_t x13655 = 0;
for(int x13656=0; x13656 < 64; x13656++) {
int32_t x13657 = x13654;
int32_t x13658 = x13655;
int32_t x13659 = x13653;
int32_t x13660 = x13659;
int32_t x13661 = x13657;
int32_t x13662 = x13658;
for(int x13663=0; x13663 < 512; x13663++) {
int32_t x13664 = x13661;
int32_t x13665 = x13662;
int32_t x13666 = x13660;
int32_t x13667 = x13666;
int32_t x13668 = x13664;
int32_t x13669 = x13665;
int32_t x13670 = x13668;
int32_t x13671 = x13669;
int32_t x13672 = x13667;
int32_t x13673 = x13672;
int32_t x13674 = x13670;
int32_t x13675 = x13671;
int32_t x13676 = x13673;
int32_t x13677 = x13674;
float x13678 = x13581[x13677];
int32_t x13679 = x13675;
float x13680 = x112[x13679];
float x13681 = x13678 - x13680;
x13652[x13676] = x13681;
x13673 += 1;
x13674 += 1;
int32_t x13685 = x13673;
int32_t x13686 = x13674;
float x13687 = x13581[x13686];
float x13688 = x112[x13679];
float x13689 = x13687 - x13688;
x13652[x13685] = x13689;
x13673 += 1;
x13674 += 1;
x13667 += 2;
x13668 += 2;
int32_t x13695 = x13668;
int32_t x13696 = x13667;
int32_t x13697 = x13696;
int32_t x13698 = x13695;
int32_t x13699 = x13671;
int32_t x13700 = x13697;
int32_t x13701 = x13698;
float x13702 = x13581[x13701];
int32_t x13703 = x13699;
float x13704 = x112[x13703];
float x13705 = x13702 - x13704;
x13652[x13700] = x13705;
x13697 += 1;
x13698 += 1;
int32_t x13709 = x13697;
int32_t x13710 = x13698;
float x13711 = x13581[x13710];
float x13712 = x112[x13703];
float x13713 = x13711 - x13712;
x13652[x13709] = x13713;
x13697 += 1;
x13698 += 1;
x13667 += 2;
x13668 += 2;
x13660 += 4;
x13661 += 4;
x13662 += 1;

}
x13653 += 2048;
x13654 += 2048;

}
float* x13728 = (float*)myMalloc(512 * sizeof(float));;
for(int x13729=0; x13729 < 512; x13729++) {
float x13730 = x49[x13729];
float x13731 = x13730 + 1.0E-5f;
x13728[x13729] = x13731;

}
float* x13735 = (float*)myMalloc(512 * sizeof(float));;
for(int x13736=0; x13736 < 512; x13736++) {
float x13737 = x13728[x13736];
double x13738 = (double)x13737;
double x13739 = sqrt(x13738);
float x13740 = (float)x13739;
x13735[x13736] = x13740;

}
// resize to WrappedArray(-1, 1, 1)
float* x13745 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x13746 = 0;
int32_t x13747 = 0;
int32_t x13748 = 0;
for(int x13749=0; x13749 < 64; x13749++) {
int32_t x13750 = x13747;
int32_t x13751 = x13748;
int32_t x13752 = x13746;
int32_t x13753 = x13752;
int32_t x13754 = x13750;
int32_t x13755 = x13751;
for(int x13756=0; x13756 < 512; x13756++) {
int32_t x13757 = x13754;
int32_t x13758 = x13755;
int32_t x13759 = x13753;
int32_t x13760 = x13759;
int32_t x13761 = x13757;
int32_t x13762 = x13758;
int32_t x13763 = x13761;
int32_t x13764 = x13762;
int32_t x13765 = x13760;
int32_t x13766 = x13765;
int32_t x13767 = x13763;
int32_t x13768 = x13764;
int32_t x13769 = x13766;
int32_t x13770 = x13767;
float x13771 = x13652[x13770];
int32_t x13772 = x13768;
float x13773 = x13735[x13772];
float x13774 = x13771 / x13773;
x13745[x13769] = x13774;
x13766 += 1;
x13767 += 1;
int32_t x13778 = x13766;
int32_t x13779 = x13767;
float x13780 = x13652[x13779];
float x13781 = x13735[x13772];
float x13782 = x13780 / x13781;
x13745[x13778] = x13782;
x13766 += 1;
x13767 += 1;
x13760 += 2;
x13761 += 2;
int32_t x13788 = x13761;
int32_t x13789 = x13760;
int32_t x13790 = x13789;
int32_t x13791 = x13788;
int32_t x13792 = x13764;
int32_t x13793 = x13790;
int32_t x13794 = x13791;
float x13795 = x13652[x13794];
int32_t x13796 = x13792;
float x13797 = x13735[x13796];
float x13798 = x13795 / x13797;
x13745[x13793] = x13798;
x13790 += 1;
x13791 += 1;
int32_t x13802 = x13790;
int32_t x13803 = x13791;
float x13804 = x13652[x13803];
float x13805 = x13735[x13796];
float x13806 = x13804 / x13805;
x13745[x13802] = x13806;
x13790 += 1;
x13791 += 1;
x13760 += 2;
x13761 += 2;
x13753 += 4;
x13754 += 4;
x13755 += 1;

}
x13746 += 2048;
x13747 += 2048;

}
// resize to WrappedArray(-1, 1, 1)
float* x13822 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x13823 = 0;
int32_t x13824 = 0;
int32_t x13825 = 0;
for(int x13826=0; x13826 < 64; x13826++) {
int32_t x13827 = x13824;
int32_t x13828 = x13825;
int32_t x13829 = x13823;
int32_t x13830 = x13829;
int32_t x13831 = x13827;
int32_t x13832 = x13828;
for(int x13833=0; x13833 < 512; x13833++) {
int32_t x13834 = x13831;
int32_t x13835 = x13832;
int32_t x13836 = x13830;
int32_t x13837 = x13836;
int32_t x13838 = x13834;
int32_t x13839 = x13835;
int32_t x13840 = x13838;
int32_t x13841 = x13839;
int32_t x13842 = x13837;
int32_t x13843 = x13842;
int32_t x13844 = x13840;
int32_t x13845 = x13841;
int32_t x13846 = x13843;
int32_t x13847 = x13844;
float x13848 = x13745[x13847];
int32_t x13849 = x13845;
float x13850 = x204[x13849];
float x13851 = x13848 * x13850;
x13822[x13846] = x13851;
x13843 += 1;
x13844 += 1;
int32_t x13855 = x13843;
int32_t x13856 = x13844;
float x13857 = x13745[x13856];
float x13858 = x204[x13849];
float x13859 = x13857 * x13858;
x13822[x13855] = x13859;
x13843 += 1;
x13844 += 1;
x13837 += 2;
x13838 += 2;
int32_t x13865 = x13838;
int32_t x13866 = x13837;
int32_t x13867 = x13866;
int32_t x13868 = x13865;
int32_t x13869 = x13841;
int32_t x13870 = x13867;
int32_t x13871 = x13868;
float x13872 = x13745[x13871];
int32_t x13873 = x13869;
float x13874 = x204[x13873];
float x13875 = x13872 * x13874;
x13822[x13870] = x13875;
x13867 += 1;
x13868 += 1;
int32_t x13879 = x13867;
int32_t x13880 = x13868;
float x13881 = x13745[x13880];
float x13882 = x204[x13873];
float x13883 = x13881 * x13882;
x13822[x13879] = x13883;
x13867 += 1;
x13868 += 1;
x13837 += 2;
x13838 += 2;
x13830 += 4;
x13831 += 4;
x13832 += 1;

}
x13823 += 2048;
x13824 += 2048;

}
// resize to WrappedArray(-1, 1, 1)
float* x13899 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x13900 = 0;
int32_t x13901 = 0;
int32_t x13902 = 0;
for(int x13903=0; x13903 < 64; x13903++) {
int32_t x13904 = x13901;
int32_t x13905 = x13902;
int32_t x13906 = x13900;
int32_t x13907 = x13906;
int32_t x13908 = x13904;
int32_t x13909 = x13905;
for(int x13910=0; x13910 < 512; x13910++) {
int32_t x13911 = x13908;
int32_t x13912 = x13909;
int32_t x13913 = x13907;
int32_t x13914 = x13913;
int32_t x13915 = x13911;
int32_t x13916 = x13912;
int32_t x13917 = x13915;
int32_t x13918 = x13916;
int32_t x13919 = x13914;
int32_t x13920 = x13919;
int32_t x13921 = x13917;
int32_t x13922 = x13918;
int32_t x13923 = x13920;
int32_t x13924 = x13921;
float x13925 = x13822[x13924];
int32_t x13926 = x13922;
float x13927 = x158[x13926];
float x13928 = x13925 + x13927;
x13899[x13923] = x13928;
x13920 += 1;
x13921 += 1;
int32_t x13932 = x13920;
int32_t x13933 = x13921;
float x13934 = x13822[x13933];
float x13935 = x158[x13926];
float x13936 = x13934 + x13935;
x13899[x13932] = x13936;
x13920 += 1;
x13921 += 1;
x13914 += 2;
x13915 += 2;
int32_t x13942 = x13915;
int32_t x13943 = x13914;
int32_t x13944 = x13943;
int32_t x13945 = x13942;
int32_t x13946 = x13918;
int32_t x13947 = x13944;
int32_t x13948 = x13945;
float x13949 = x13822[x13948];
int32_t x13950 = x13946;
float x13951 = x158[x13950];
float x13952 = x13949 + x13951;
x13899[x13947] = x13952;
x13944 += 1;
x13945 += 1;
int32_t x13956 = x13944;
int32_t x13957 = x13945;
float x13958 = x13822[x13957];
float x13959 = x158[x13950];
float x13960 = x13958 + x13959;
x13899[x13956] = x13960;
x13944 += 1;
x13945 += 1;
x13914 += 2;
x13915 += 2;
x13907 += 4;
x13908 += 4;
x13909 += 1;

}
x13900 += 2048;
x13901 += 2048;

}
float* x13975 = (float*)myMalloc(131072 * sizeof(float));;
for(int x13977=0; x13977 < 131072; x13977++) {
float x13978 = x13899[x13977];
bool x13979 = x13978 < 0.0f;
if (x13979) {
x13975[x13977] = 0.0f;
} else {
float x13982 = x13899[x13977];
x13975[x13977] = x13982;
}

}
float* x13988 = (float*)myMalloc(524288 * sizeof(float));;
float* x13989 = (float*)myMalloc(131072 * sizeof(float));;
for(int x13990=0; x13990 < 64; x13990++) {
int32_t x13991 = x13990 * 2048;
float* x13992 = x13975+x13991;
int32_t x13993 = x13990 * 8192;
float* x13994 = x13988+x13993;
float* x13995 = x13989+x13991;
for(int x13996=0; x13996 < 512; x13996++) {
int32_t x13997 = x13996 / 1;
int32_t x14001 = x13997 * 2;
int32_t x14002 = x14001 * 2;
int32_t x13998 = x13996 % 1;
int32_t x13999 = x13998 / 1;
int32_t x14003 = x13999 * 2;
int32_t x14004 = x14003 * 2;
int32_t x14005 = x14002 + x14004;
int32_t x14000 = x13998 % 1;
int32_t x14006 = x14000 * 2;
int32_t x14007 = x14006 * 2;
int32_t x14008 = x14005 + x14007;
float* x14009 = x13995+x14008;
float* x14010 = x13992+x14002;
for(int x14011=0; x14011 < 2; x14011++) {
int32_t x14013 = x14011 * 2;
float* x14014 = x14009+x14013;
int32_t x14012 = x14011 + x13999;
int32_t x14015 = x14012 * 2;
int32_t x14016 = x14015 + x14000;
float* x14017 = x14010+x14016;
memcpy(x14014, x14017, 4 * 2);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,4,512,1,x211,512,x13995,4,1,x13994,4);

}
// resize to WrappedArray(-1, 1, 1)
float* x14027 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x14028 = 0;
int32_t x14029 = 0;
int32_t x14030 = 0;
for(int x14031=0; x14031 < 64; x14031++) {
int32_t x14032 = x14029;
int32_t x14033 = x14030;
int32_t x14034 = x14028;
int32_t x14035 = x14034;
int32_t x14036 = x14032;
int32_t x14037 = x14033;
for(int x14039=0; x14039 < 2048; x14039++) {
int32_t x14040 = x14036;
int32_t x14041 = x14037;
int32_t x14042 = x14035;
int32_t x14043 = x14042;
int32_t x14044 = x14040;
int32_t x14045 = x14041;
int32_t x14046 = x14044;
int32_t x14047 = x14045;
int32_t x14048 = x14043;
int32_t x14049 = x14048;
int32_t x14050 = x14046;
int32_t x14051 = x14047;
int32_t x14052 = x14049;
int32_t x14053 = x14050;
float x14054 = x13988[x14053];
int32_t x14055 = x14051;
float x14056 = x114[x14055];
float x14057 = x14054 - x14056;
x14027[x14052] = x14057;
x14049 += 1;
x14050 += 1;
int32_t x14061 = x14049;
int32_t x14062 = x14050;
float x14063 = x13988[x14062];
float x14064 = x114[x14055];
float x14065 = x14063 - x14064;
x14027[x14061] = x14065;
x14049 += 1;
x14050 += 1;
x14043 += 2;
x14044 += 2;
int32_t x14071 = x14044;
int32_t x14072 = x14043;
int32_t x14073 = x14072;
int32_t x14074 = x14071;
int32_t x14075 = x14047;
int32_t x14076 = x14073;
int32_t x14077 = x14074;
float x14078 = x13988[x14077];
int32_t x14079 = x14075;
float x14080 = x114[x14079];
float x14081 = x14078 - x14080;
x14027[x14076] = x14081;
x14073 += 1;
x14074 += 1;
int32_t x14085 = x14073;
int32_t x14086 = x14074;
float x14087 = x13988[x14086];
float x14088 = x114[x14079];
float x14089 = x14087 - x14088;
x14027[x14085] = x14089;
x14073 += 1;
x14074 += 1;
x14043 += 2;
x14044 += 2;
x14035 += 4;
x14036 += 4;
x14037 += 1;

}
x14028 += 8192;
x14029 += 8192;

}
float* x14104 = (float*)myMalloc(2048 * sizeof(float));;
for(int x14105=0; x14105 < 2048; x14105++) {
float x14106 = x192[x14105];
float x14107 = x14106 + 1.0E-5f;
x14104[x14105] = x14107;

}
float* x14111 = (float*)myMalloc(2048 * sizeof(float));;
for(int x14112=0; x14112 < 2048; x14112++) {
float x14113 = x14104[x14112];
double x14114 = (double)x14113;
double x14115 = sqrt(x14114);
float x14116 = (float)x14115;
x14111[x14112] = x14116;

}
// resize to WrappedArray(-1, 1, 1)
float* x14121 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x14122 = 0;
int32_t x14123 = 0;
int32_t x14124 = 0;
for(int x14125=0; x14125 < 64; x14125++) {
int32_t x14126 = x14123;
int32_t x14127 = x14124;
int32_t x14128 = x14122;
int32_t x14129 = x14128;
int32_t x14130 = x14126;
int32_t x14131 = x14127;
for(int x14132=0; x14132 < 2048; x14132++) {
int32_t x14133 = x14130;
int32_t x14134 = x14131;
int32_t x14135 = x14129;
int32_t x14136 = x14135;
int32_t x14137 = x14133;
int32_t x14138 = x14134;
int32_t x14139 = x14137;
int32_t x14140 = x14138;
int32_t x14141 = x14136;
int32_t x14142 = x14141;
int32_t x14143 = x14139;
int32_t x14144 = x14140;
int32_t x14145 = x14142;
int32_t x14146 = x14143;
float x14147 = x14027[x14146];
int32_t x14148 = x14144;
float x14149 = x14111[x14148];
float x14150 = x14147 / x14149;
x14121[x14145] = x14150;
x14142 += 1;
x14143 += 1;
int32_t x14154 = x14142;
int32_t x14155 = x14143;
float x14156 = x14027[x14155];
float x14157 = x14111[x14148];
float x14158 = x14156 / x14157;
x14121[x14154] = x14158;
x14142 += 1;
x14143 += 1;
x14136 += 2;
x14137 += 2;
int32_t x14164 = x14137;
int32_t x14165 = x14136;
int32_t x14166 = x14165;
int32_t x14167 = x14164;
int32_t x14168 = x14140;
int32_t x14169 = x14166;
int32_t x14170 = x14167;
float x14171 = x14027[x14170];
int32_t x14172 = x14168;
float x14173 = x14111[x14172];
float x14174 = x14171 / x14173;
x14121[x14169] = x14174;
x14166 += 1;
x14167 += 1;
int32_t x14178 = x14166;
int32_t x14179 = x14167;
float x14180 = x14027[x14179];
float x14181 = x14111[x14172];
float x14182 = x14180 / x14181;
x14121[x14178] = x14182;
x14166 += 1;
x14167 += 1;
x14136 += 2;
x14137 += 2;
x14129 += 4;
x14130 += 4;
x14131 += 1;

}
x14122 += 8192;
x14123 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x14198 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x14199 = 0;
int32_t x14200 = 0;
int32_t x14201 = 0;
for(int x14202=0; x14202 < 64; x14202++) {
int32_t x14203 = x14200;
int32_t x14204 = x14201;
int32_t x14205 = x14199;
int32_t x14206 = x14205;
int32_t x14207 = x14203;
int32_t x14208 = x14204;
for(int x14209=0; x14209 < 2048; x14209++) {
int32_t x14210 = x14207;
int32_t x14211 = x14208;
int32_t x14212 = x14206;
int32_t x14213 = x14212;
int32_t x14214 = x14210;
int32_t x14215 = x14211;
int32_t x14216 = x14214;
int32_t x14217 = x14215;
int32_t x14218 = x14213;
int32_t x14219 = x14218;
int32_t x14220 = x14216;
int32_t x14221 = x14217;
int32_t x14222 = x14219;
int32_t x14223 = x14220;
float x14224 = x14121[x14223];
int32_t x14225 = x14221;
float x14226 = x238[x14225];
float x14227 = x14224 * x14226;
x14198[x14222] = x14227;
x14219 += 1;
x14220 += 1;
int32_t x14231 = x14219;
int32_t x14232 = x14220;
float x14233 = x14121[x14232];
float x14234 = x238[x14225];
float x14235 = x14233 * x14234;
x14198[x14231] = x14235;
x14219 += 1;
x14220 += 1;
x14213 += 2;
x14214 += 2;
int32_t x14241 = x14214;
int32_t x14242 = x14213;
int32_t x14243 = x14242;
int32_t x14244 = x14241;
int32_t x14245 = x14217;
int32_t x14246 = x14243;
int32_t x14247 = x14244;
float x14248 = x14121[x14247];
int32_t x14249 = x14245;
float x14250 = x238[x14249];
float x14251 = x14248 * x14250;
x14198[x14246] = x14251;
x14243 += 1;
x14244 += 1;
int32_t x14255 = x14243;
int32_t x14256 = x14244;
float x14257 = x14121[x14256];
float x14258 = x238[x14249];
float x14259 = x14257 * x14258;
x14198[x14255] = x14259;
x14243 += 1;
x14244 += 1;
x14213 += 2;
x14214 += 2;
x14206 += 4;
x14207 += 4;
x14208 += 1;

}
x14199 += 8192;
x14200 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x14275 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x14276 = 0;
int32_t x14277 = 0;
int32_t x14278 = 0;
for(int x14279=0; x14279 < 64; x14279++) {
int32_t x14280 = x14277;
int32_t x14281 = x14278;
int32_t x14282 = x14276;
int32_t x14283 = x14282;
int32_t x14284 = x14280;
int32_t x14285 = x14281;
for(int x14286=0; x14286 < 2048; x14286++) {
int32_t x14287 = x14284;
int32_t x14288 = x14285;
int32_t x14289 = x14283;
int32_t x14290 = x14289;
int32_t x14291 = x14287;
int32_t x14292 = x14288;
int32_t x14293 = x14291;
int32_t x14294 = x14292;
int32_t x14295 = x14290;
int32_t x14296 = x14295;
int32_t x14297 = x14293;
int32_t x14298 = x14294;
int32_t x14299 = x14296;
int32_t x14300 = x14297;
float x14301 = x14198[x14300];
int32_t x14302 = x14298;
float x14303 = x61[x14302];
float x14304 = x14301 + x14303;
x14275[x14299] = x14304;
x14296 += 1;
x14297 += 1;
int32_t x14308 = x14296;
int32_t x14309 = x14297;
float x14310 = x14198[x14309];
float x14311 = x61[x14302];
float x14312 = x14310 + x14311;
x14275[x14308] = x14312;
x14296 += 1;
x14297 += 1;
x14290 += 2;
x14291 += 2;
int32_t x14318 = x14291;
int32_t x14319 = x14290;
int32_t x14320 = x14319;
int32_t x14321 = x14318;
int32_t x14322 = x14294;
int32_t x14323 = x14320;
int32_t x14324 = x14321;
float x14325 = x14198[x14324];
int32_t x14326 = x14322;
float x14327 = x61[x14326];
float x14328 = x14325 + x14327;
x14275[x14323] = x14328;
x14320 += 1;
x14321 += 1;
int32_t x14332 = x14320;
int32_t x14333 = x14321;
float x14334 = x14198[x14333];
float x14335 = x61[x14326];
float x14336 = x14334 + x14335;
x14275[x14332] = x14336;
x14320 += 1;
x14321 += 1;
x14290 += 2;
x14291 += 2;
x14283 += 4;
x14284 += 4;
x14285 += 1;

}
x14276 += 8192;
x14277 += 8192;

}
float* x14351 = (float*)myMalloc(524288 * sizeof(float));;
float* x14352 = (float*)myMalloc(262144 * sizeof(float));;
for(int x14353=0; x14353 < 64; x14353++) {
int32_t x14354 = x14353 * 16384;
float* x14355 = x13299+x14354;
int32_t x14356 = x14353 * 8192;
float* x14357 = x14351+x14356;
int32_t x14358 = x14353 * 4096;
float* x14359 = x14352+x14358;
for(int x14360=0; x14360 < 1024; x14360++) {
int32_t x14361 = x14360 / 1;
int32_t x14365 = x14361 * 2;
int32_t x14366 = x14365 * 2;
int32_t x14362 = x14360 % 1;
int32_t x14363 = x14362 / 1;
int32_t x14367 = x14363 * 2;
int32_t x14368 = x14367 * 2;
int32_t x14369 = x14366 + x14368;
int32_t x14364 = x14362 % 1;
int32_t x14370 = x14364 * 2;
int32_t x14371 = x14370 * 2;
int32_t x14372 = x14369 + x14371;
float* x14373 = x14359+x14372;
int32_t x14374 = x14361 * 4;
int32_t x14375 = x14374 * 4;
float* x14376 = x14355+x14375;
for(int x14377=0; x14377 < 2; x14377++) {
int32_t x14378 = x14377 * 2;
int32_t x14379 = x14378 + x14363;
int32_t x14383 = x14379 * 4;
int32_t x14384 = x14383 + x14364;
for(int x14380=0; x14380 < 2; x14380++) {
int32_t x14381 = x14378 + x14380;
float* x14382 = x14373+x14381;
int32_t x14385 = x14380 * 2;
int32_t x14386 = x14384 + x14385;
float* x14387 = x14376+x14386;
memcpy(x14382, x14387, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,4,1024,1,x213,1024,x14359,4,1,x14357,4);

}
// resize to WrappedArray(-1, 1, 1)
float* x14399 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x14400 = 0;
int32_t x14401 = 0;
int32_t x14402 = 0;
for(int x14403=0; x14403 < 64; x14403++) {
int32_t x14404 = x14401;
int32_t x14405 = x14402;
int32_t x14406 = x14400;
int32_t x14407 = x14406;
int32_t x14408 = x14404;
int32_t x14409 = x14405;
for(int x14410=0; x14410 < 2048; x14410++) {
int32_t x14411 = x14408;
int32_t x14412 = x14409;
int32_t x14413 = x14407;
int32_t x14414 = x14413;
int32_t x14415 = x14411;
int32_t x14416 = x14412;
int32_t x14417 = x14415;
int32_t x14418 = x14416;
int32_t x14419 = x14414;
int32_t x14420 = x14419;
int32_t x14421 = x14417;
int32_t x14422 = x14418;
int32_t x14423 = x14420;
int32_t x14424 = x14421;
float x14425 = x14351[x14424];
int32_t x14426 = x14422;
float x14427 = x63[x14426];
float x14428 = x14425 - x14427;
x14399[x14423] = x14428;
x14420 += 1;
x14421 += 1;
int32_t x14432 = x14420;
int32_t x14433 = x14421;
float x14434 = x14351[x14433];
float x14435 = x63[x14426];
float x14436 = x14434 - x14435;
x14399[x14432] = x14436;
x14420 += 1;
x14421 += 1;
x14414 += 2;
x14415 += 2;
int32_t x14442 = x14415;
int32_t x14443 = x14414;
int32_t x14444 = x14443;
int32_t x14445 = x14442;
int32_t x14446 = x14418;
int32_t x14447 = x14444;
int32_t x14448 = x14445;
float x14449 = x14351[x14448];
int32_t x14450 = x14446;
float x14451 = x63[x14450];
float x14452 = x14449 - x14451;
x14399[x14447] = x14452;
x14444 += 1;
x14445 += 1;
int32_t x14456 = x14444;
int32_t x14457 = x14445;
float x14458 = x14351[x14457];
float x14459 = x63[x14450];
float x14460 = x14458 - x14459;
x14399[x14456] = x14460;
x14444 += 1;
x14445 += 1;
x14414 += 2;
x14415 += 2;
x14407 += 4;
x14408 += 4;
x14409 += 1;

}
x14400 += 8192;
x14401 += 8192;

}
float* x14475 = (float*)myMalloc(2048 * sizeof(float));;
for(int x14476=0; x14476 < 2048; x14476++) {
float x14477 = x124[x14476];
float x14478 = x14477 + 1.0E-5f;
x14475[x14476] = x14478;

}
float* x14482 = (float*)myMalloc(2048 * sizeof(float));;
for(int x14483=0; x14483 < 2048; x14483++) {
float x14484 = x14475[x14483];
double x14485 = (double)x14484;
double x14486 = sqrt(x14485);
float x14487 = (float)x14486;
x14482[x14483] = x14487;

}
// resize to WrappedArray(-1, 1, 1)
float* x14492 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x14493 = 0;
int32_t x14494 = 0;
int32_t x14495 = 0;
for(int x14496=0; x14496 < 64; x14496++) {
int32_t x14497 = x14494;
int32_t x14498 = x14495;
int32_t x14499 = x14493;
int32_t x14500 = x14499;
int32_t x14501 = x14497;
int32_t x14502 = x14498;
for(int x14503=0; x14503 < 2048; x14503++) {
int32_t x14504 = x14501;
int32_t x14505 = x14502;
int32_t x14506 = x14500;
int32_t x14507 = x14506;
int32_t x14508 = x14504;
int32_t x14509 = x14505;
int32_t x14510 = x14508;
int32_t x14511 = x14509;
int32_t x14512 = x14507;
int32_t x14513 = x14512;
int32_t x14514 = x14510;
int32_t x14515 = x14511;
int32_t x14516 = x14513;
int32_t x14517 = x14514;
float x14518 = x14399[x14517];
int32_t x14519 = x14515;
float x14520 = x14482[x14519];
float x14521 = x14518 / x14520;
x14492[x14516] = x14521;
x14513 += 1;
x14514 += 1;
int32_t x14525 = x14513;
int32_t x14526 = x14514;
float x14527 = x14399[x14526];
float x14528 = x14482[x14519];
float x14529 = x14527 / x14528;
x14492[x14525] = x14529;
x14513 += 1;
x14514 += 1;
x14507 += 2;
x14508 += 2;
int32_t x14535 = x14508;
int32_t x14536 = x14507;
int32_t x14537 = x14536;
int32_t x14538 = x14535;
int32_t x14539 = x14511;
int32_t x14540 = x14537;
int32_t x14541 = x14538;
float x14542 = x14399[x14541];
int32_t x14543 = x14539;
float x14544 = x14482[x14543];
float x14545 = x14542 / x14544;
x14492[x14540] = x14545;
x14537 += 1;
x14538 += 1;
int32_t x14549 = x14537;
int32_t x14550 = x14538;
float x14551 = x14399[x14550];
float x14552 = x14482[x14543];
float x14553 = x14551 / x14552;
x14492[x14549] = x14553;
x14537 += 1;
x14538 += 1;
x14507 += 2;
x14508 += 2;
x14500 += 4;
x14501 += 4;
x14502 += 1;

}
x14493 += 8192;
x14494 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x14569 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x14570 = 0;
int32_t x14571 = 0;
int32_t x14572 = 0;
for(int x14573=0; x14573 < 64; x14573++) {
int32_t x14574 = x14571;
int32_t x14575 = x14572;
int32_t x14576 = x14570;
int32_t x14577 = x14576;
int32_t x14578 = x14574;
int32_t x14579 = x14575;
for(int x14580=0; x14580 < 2048; x14580++) {
int32_t x14581 = x14578;
int32_t x14582 = x14579;
int32_t x14583 = x14577;
int32_t x14584 = x14583;
int32_t x14585 = x14581;
int32_t x14586 = x14582;
int32_t x14587 = x14585;
int32_t x14588 = x14586;
int32_t x14589 = x14584;
int32_t x14590 = x14589;
int32_t x14591 = x14587;
int32_t x14592 = x14588;
int32_t x14593 = x14590;
int32_t x14594 = x14591;
float x14595 = x14492[x14594];
int32_t x14596 = x14592;
float x14597 = x172[x14596];
float x14598 = x14595 * x14597;
x14569[x14593] = x14598;
x14590 += 1;
x14591 += 1;
int32_t x14602 = x14590;
int32_t x14603 = x14591;
float x14604 = x14492[x14603];
float x14605 = x172[x14596];
float x14606 = x14604 * x14605;
x14569[x14602] = x14606;
x14590 += 1;
x14591 += 1;
x14584 += 2;
x14585 += 2;
int32_t x14612 = x14585;
int32_t x14613 = x14584;
int32_t x14614 = x14613;
int32_t x14615 = x14612;
int32_t x14616 = x14588;
int32_t x14617 = x14614;
int32_t x14618 = x14615;
float x14619 = x14492[x14618];
int32_t x14620 = x14616;
float x14621 = x172[x14620];
float x14622 = x14619 * x14621;
x14569[x14617] = x14622;
x14614 += 1;
x14615 += 1;
int32_t x14626 = x14614;
int32_t x14627 = x14615;
float x14628 = x14492[x14627];
float x14629 = x172[x14620];
float x14630 = x14628 * x14629;
x14569[x14626] = x14630;
x14614 += 1;
x14615 += 1;
x14584 += 2;
x14585 += 2;
x14577 += 4;
x14578 += 4;
x14579 += 1;

}
x14570 += 8192;
x14571 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x14646 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x14647 = 0;
int32_t x14648 = 0;
int32_t x14649 = 0;
for(int x14650=0; x14650 < 64; x14650++) {
int32_t x14651 = x14648;
int32_t x14652 = x14649;
int32_t x14653 = x14647;
int32_t x14654 = x14653;
int32_t x14655 = x14651;
int32_t x14656 = x14652;
for(int x14657=0; x14657 < 2048; x14657++) {
int32_t x14658 = x14655;
int32_t x14659 = x14656;
int32_t x14660 = x14654;
int32_t x14661 = x14660;
int32_t x14662 = x14658;
int32_t x14663 = x14659;
int32_t x14664 = x14662;
int32_t x14665 = x14663;
int32_t x14666 = x14661;
int32_t x14667 = x14666;
int32_t x14668 = x14664;
int32_t x14669 = x14665;
int32_t x14670 = x14667;
int32_t x14671 = x14668;
float x14672 = x14569[x14671];
int32_t x14673 = x14669;
float x14674 = x106[x14673];
float x14675 = x14672 + x14674;
x14646[x14670] = x14675;
x14667 += 1;
x14668 += 1;
int32_t x14679 = x14667;
int32_t x14680 = x14668;
float x14681 = x14569[x14680];
float x14682 = x106[x14673];
float x14683 = x14681 + x14682;
x14646[x14679] = x14683;
x14667 += 1;
x14668 += 1;
x14661 += 2;
x14662 += 2;
int32_t x14689 = x14662;
int32_t x14690 = x14661;
int32_t x14691 = x14690;
int32_t x14692 = x14689;
int32_t x14693 = x14665;
int32_t x14694 = x14691;
int32_t x14695 = x14692;
float x14696 = x14569[x14695];
int32_t x14697 = x14693;
float x14698 = x106[x14697];
float x14699 = x14696 + x14698;
x14646[x14694] = x14699;
x14691 += 1;
x14692 += 1;
int32_t x14703 = x14691;
int32_t x14704 = x14692;
float x14705 = x14569[x14704];
float x14706 = x106[x14697];
float x14707 = x14705 + x14706;
x14646[x14703] = x14707;
x14691 += 1;
x14692 += 1;
x14661 += 2;
x14662 += 2;
x14654 += 4;
x14655 += 4;
x14656 += 1;

}
x14647 += 8192;
x14648 += 8192;

}
int32_t x14722 = 0;
int32_t x14723 = 0;
int32_t x14724 = 0;
for(int x14725=0; x14725 < 64; x14725++) {
int32_t x14726 = x14723;
int32_t x14727 = x14724;
int32_t x14728 = x14722;
int32_t x14729 = x14728;
int32_t x14730 = x14726;
int32_t x14731 = x14727;
for(int x14732=0; x14732 < 2048; x14732++) {
int32_t x14733 = x14730;
int32_t x14734 = x14731;
int32_t x14735 = x14729;
int32_t x14736 = x14735;
int32_t x14737 = x14733;
int32_t x14738 = x14734;
int32_t x14739 = x14737;
int32_t x14740 = x14738;
int32_t x14741 = x14736;
int32_t x14742 = x14741;
int32_t x14743 = x14739;
int32_t x14744 = x14740;
int32_t x14745 = x14743;
float x14746 = x14275[x14745];
int32_t x14747 = x14744;
float x14748 = x14646[x14747];
float x14749 = x14746 + x14748;
x14275[x14745] = x14749;
x14742 += 1;
x14743 += 1;
x14744 += 1;
int32_t x14754 = x14743;
float x14755 = x14275[x14754];
int32_t x14756 = x14744;
float x14757 = x14646[x14756];
float x14758 = x14755 + x14757;
x14275[x14754] = x14758;
x14742 += 1;
x14743 += 1;
x14744 += 1;
x14736 += 2;
x14737 += 2;
x14738 += 2;
int32_t x14766 = x14737;
int32_t x14767 = x14738;
int32_t x14768 = x14736;
int32_t x14769 = x14768;
int32_t x14770 = x14766;
int32_t x14771 = x14767;
int32_t x14772 = x14770;
float x14773 = x14275[x14772];
int32_t x14774 = x14771;
float x14775 = x14646[x14774];
float x14776 = x14773 + x14775;
x14275[x14772] = x14776;
x14769 += 1;
x14770 += 1;
x14771 += 1;
int32_t x14781 = x14770;
float x14782 = x14275[x14781];
int32_t x14783 = x14771;
float x14784 = x14646[x14783];
float x14785 = x14782 + x14784;
x14275[x14781] = x14785;
x14769 += 1;
x14770 += 1;
x14771 += 1;
x14736 += 2;
x14737 += 2;
x14738 += 2;
x14729 += 4;
x14730 += 4;
x14731 += 4;

}
x14722 += 8192;
x14723 += 8192;
x14724 += 8192;

}
float* x14803 = (float*)myMalloc(524288 * sizeof(float));;
for(int x14804=0; x14804 < 524288; x14804++) {
float x14805 = x14275[x14804];
bool x14806 = x14805 < 0.0f;
if (x14806) {
x14803[x14804] = 0.0f;
} else {
float x14809 = x14275[x14804];
x14803[x14804] = x14809;
}

}
float* x14815 = (float*)myMalloc(131072 * sizeof(float));;
float* x14816 = (float*)myMalloc(524288 * sizeof(float));;
for(int x14817=0; x14817 < 64; x14817++) {
int32_t x14818 = x14817 * 8192;
float* x14819 = x14803+x14818;
int32_t x14820 = x14817 * 2048;
float* x14821 = x14815+x14820;
float* x14822 = x14816+x14818;
for(int x14823=0; x14823 < 2048; x14823++) {
int32_t x14824 = x14823 / 1;
int32_t x14828 = x14824 * 2;
int32_t x14829 = x14828 * 2;
int32_t x14825 = x14823 % 1;
int32_t x14826 = x14825 / 1;
int32_t x14830 = x14826 * 2;
int32_t x14831 = x14830 * 2;
int32_t x14832 = x14829 + x14831;
int32_t x14827 = x14825 % 1;
int32_t x14833 = x14827 * 2;
int32_t x14834 = x14833 * 2;
int32_t x14835 = x14832 + x14834;
float* x14836 = x14822+x14835;
float* x14837 = x14819+x14829;
for(int x14838=0; x14838 < 2; x14838++) {
int32_t x14840 = x14838 * 2;
float* x14841 = x14836+x14840;
int32_t x14839 = x14838 + x14826;
int32_t x14842 = x14839 * 2;
int32_t x14843 = x14842 + x14827;
float* x14844 = x14837+x14843;
memcpy(x14841, x14844, 4 * 2);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,4,2048,1,x214,2048,x14822,4,1,x14821,4);

}
// resize to WrappedArray(-1, 1, 1)
float* x14854 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x14855 = 0;
int32_t x14856 = 0;
int32_t x14857 = 0;
for(int x14858=0; x14858 < 64; x14858++) {
int32_t x14859 = x14856;
int32_t x14860 = x14857;
int32_t x14861 = x14855;
int32_t x14862 = x14861;
int32_t x14863 = x14859;
int32_t x14864 = x14860;
for(int x14865=0; x14865 < 512; x14865++) {
int32_t x14866 = x14863;
int32_t x14867 = x14864;
int32_t x14868 = x14862;
int32_t x14869 = x14868;
int32_t x14870 = x14866;
int32_t x14871 = x14867;
int32_t x14872 = x14870;
int32_t x14873 = x14871;
int32_t x14874 = x14869;
int32_t x14875 = x14874;
int32_t x14876 = x14872;
int32_t x14877 = x14873;
int32_t x14878 = x14875;
int32_t x14879 = x14876;
float x14880 = x14815[x14879];
int32_t x14881 = x14877;
float x14882 = x153[x14881];
float x14883 = x14880 - x14882;
x14854[x14878] = x14883;
x14875 += 1;
x14876 += 1;
int32_t x14887 = x14875;
int32_t x14888 = x14876;
float x14889 = x14815[x14888];
float x14890 = x153[x14881];
float x14891 = x14889 - x14890;
x14854[x14887] = x14891;
x14875 += 1;
x14876 += 1;
x14869 += 2;
x14870 += 2;
int32_t x14897 = x14870;
int32_t x14898 = x14869;
int32_t x14899 = x14898;
int32_t x14900 = x14897;
int32_t x14901 = x14873;
int32_t x14902 = x14899;
int32_t x14903 = x14900;
float x14904 = x14815[x14903];
int32_t x14905 = x14901;
float x14906 = x153[x14905];
float x14907 = x14904 - x14906;
x14854[x14902] = x14907;
x14899 += 1;
x14900 += 1;
int32_t x14911 = x14899;
int32_t x14912 = x14900;
float x14913 = x14815[x14912];
float x14914 = x153[x14905];
float x14915 = x14913 - x14914;
x14854[x14911] = x14915;
x14899 += 1;
x14900 += 1;
x14869 += 2;
x14870 += 2;
x14862 += 4;
x14863 += 4;
x14864 += 1;

}
x14855 += 2048;
x14856 += 2048;

}
float* x14930 = (float*)myMalloc(512 * sizeof(float));;
for(int x14931=0; x14931 < 512; x14931++) {
float x14932 = x64[x14931];
float x14933 = x14932 + 1.0E-5f;
x14930[x14931] = x14933;

}
float* x14937 = (float*)myMalloc(512 * sizeof(float));;
for(int x14938=0; x14938 < 512; x14938++) {
float x14939 = x14930[x14938];
double x14940 = (double)x14939;
double x14941 = sqrt(x14940);
float x14942 = (float)x14941;
x14937[x14938] = x14942;

}
// resize to WrappedArray(-1, 1, 1)
float* x14947 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x14948 = 0;
int32_t x14949 = 0;
int32_t x14950 = 0;
for(int x14951=0; x14951 < 64; x14951++) {
int32_t x14952 = x14949;
int32_t x14953 = x14950;
int32_t x14954 = x14948;
int32_t x14955 = x14954;
int32_t x14956 = x14952;
int32_t x14957 = x14953;
for(int x14958=0; x14958 < 512; x14958++) {
int32_t x14959 = x14956;
int32_t x14960 = x14957;
int32_t x14961 = x14955;
int32_t x14962 = x14961;
int32_t x14963 = x14959;
int32_t x14964 = x14960;
int32_t x14965 = x14963;
int32_t x14966 = x14964;
int32_t x14967 = x14962;
int32_t x14968 = x14967;
int32_t x14969 = x14965;
int32_t x14970 = x14966;
int32_t x14971 = x14968;
int32_t x14972 = x14969;
float x14973 = x14854[x14972];
int32_t x14974 = x14970;
float x14975 = x14937[x14974];
float x14976 = x14973 / x14975;
x14947[x14971] = x14976;
x14968 += 1;
x14969 += 1;
int32_t x14980 = x14968;
int32_t x14981 = x14969;
float x14982 = x14854[x14981];
float x14983 = x14937[x14974];
float x14984 = x14982 / x14983;
x14947[x14980] = x14984;
x14968 += 1;
x14969 += 1;
x14962 += 2;
x14963 += 2;
int32_t x14990 = x14963;
int32_t x14991 = x14962;
int32_t x14992 = x14991;
int32_t x14993 = x14990;
int32_t x14994 = x14966;
int32_t x14995 = x14992;
int32_t x14996 = x14993;
float x14997 = x14854[x14996];
int32_t x14998 = x14994;
float x14999 = x14937[x14998];
float x15000 = x14997 / x14999;
x14947[x14995] = x15000;
x14992 += 1;
x14993 += 1;
int32_t x15004 = x14992;
int32_t x15005 = x14993;
float x15006 = x14854[x15005];
float x15007 = x14937[x14998];
float x15008 = x15006 / x15007;
x14947[x15004] = x15008;
x14992 += 1;
x14993 += 1;
x14962 += 2;
x14963 += 2;
x14955 += 4;
x14956 += 4;
x14957 += 1;

}
x14948 += 2048;
x14949 += 2048;

}
// resize to WrappedArray(-1, 1, 1)
float* x15024 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x15025 = 0;
int32_t x15026 = 0;
int32_t x15027 = 0;
for(int x15028=0; x15028 < 64; x15028++) {
int32_t x15029 = x15026;
int32_t x15030 = x15027;
int32_t x15031 = x15025;
int32_t x15032 = x15031;
int32_t x15033 = x15029;
int32_t x15034 = x15030;
for(int x15035=0; x15035 < 512; x15035++) {
int32_t x15036 = x15033;
int32_t x15037 = x15034;
int32_t x15038 = x15032;
int32_t x15039 = x15038;
int32_t x15040 = x15036;
int32_t x15041 = x15037;
int32_t x15042 = x15040;
int32_t x15043 = x15041;
int32_t x15044 = x15039;
int32_t x15045 = x15044;
int32_t x15046 = x15042;
int32_t x15047 = x15043;
int32_t x15048 = x15045;
int32_t x15049 = x15046;
float x15050 = x14947[x15049];
int32_t x15051 = x15047;
float x15052 = x45[x15051];
float x15053 = x15050 * x15052;
x15024[x15048] = x15053;
x15045 += 1;
x15046 += 1;
int32_t x15057 = x15045;
int32_t x15058 = x15046;
float x15059 = x14947[x15058];
float x15060 = x45[x15051];
float x15061 = x15059 * x15060;
x15024[x15057] = x15061;
x15045 += 1;
x15046 += 1;
x15039 += 2;
x15040 += 2;
int32_t x15067 = x15040;
int32_t x15068 = x15039;
int32_t x15069 = x15068;
int32_t x15070 = x15067;
int32_t x15071 = x15043;
int32_t x15072 = x15069;
int32_t x15073 = x15070;
float x15074 = x14947[x15073];
int32_t x15075 = x15071;
float x15076 = x45[x15075];
float x15077 = x15074 * x15076;
x15024[x15072] = x15077;
x15069 += 1;
x15070 += 1;
int32_t x15081 = x15069;
int32_t x15082 = x15070;
float x15083 = x14947[x15082];
float x15084 = x45[x15075];
float x15085 = x15083 * x15084;
x15024[x15081] = x15085;
x15069 += 1;
x15070 += 1;
x15039 += 2;
x15040 += 2;
x15032 += 4;
x15033 += 4;
x15034 += 1;

}
x15025 += 2048;
x15026 += 2048;

}
// resize to WrappedArray(-1, 1, 1)
float* x15101 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x15102 = 0;
int32_t x15103 = 0;
int32_t x15104 = 0;
for(int x15105=0; x15105 < 64; x15105++) {
int32_t x15106 = x15103;
int32_t x15107 = x15104;
int32_t x15108 = x15102;
int32_t x15109 = x15108;
int32_t x15110 = x15106;
int32_t x15111 = x15107;
for(int x15112=0; x15112 < 512; x15112++) {
int32_t x15113 = x15110;
int32_t x15114 = x15111;
int32_t x15115 = x15109;
int32_t x15116 = x15115;
int32_t x15117 = x15113;
int32_t x15118 = x15114;
int32_t x15119 = x15117;
int32_t x15120 = x15118;
int32_t x15121 = x15116;
int32_t x15122 = x15121;
int32_t x15123 = x15119;
int32_t x15124 = x15120;
int32_t x15125 = x15122;
int32_t x15126 = x15123;
float x15127 = x15024[x15126];
int32_t x15128 = x15124;
float x15129 = x136[x15128];
float x15130 = x15127 + x15129;
x15101[x15125] = x15130;
x15122 += 1;
x15123 += 1;
int32_t x15134 = x15122;
int32_t x15135 = x15123;
float x15136 = x15024[x15135];
float x15137 = x136[x15128];
float x15138 = x15136 + x15137;
x15101[x15134] = x15138;
x15122 += 1;
x15123 += 1;
x15116 += 2;
x15117 += 2;
int32_t x15144 = x15117;
int32_t x15145 = x15116;
int32_t x15146 = x15145;
int32_t x15147 = x15144;
int32_t x15148 = x15120;
int32_t x15149 = x15146;
int32_t x15150 = x15147;
float x15151 = x15024[x15150];
int32_t x15152 = x15148;
float x15153 = x136[x15152];
float x15154 = x15151 + x15153;
x15101[x15149] = x15154;
x15146 += 1;
x15147 += 1;
int32_t x15158 = x15146;
int32_t x15159 = x15147;
float x15160 = x15024[x15159];
float x15161 = x136[x15152];
float x15162 = x15160 + x15161;
x15101[x15158] = x15162;
x15146 += 1;
x15147 += 1;
x15116 += 2;
x15117 += 2;
x15109 += 4;
x15110 += 4;
x15111 += 1;

}
x15102 += 2048;
x15103 += 2048;

}
float* x15177 = (float*)myMalloc(131072 * sizeof(float));;
for(int x15178=0; x15178 < 131072; x15178++) {
float x15179 = x15101[x15178];
bool x15180 = x15179 < 0.0f;
if (x15180) {
x15177[x15178] = 0.0f;
} else {
float x15183 = x15101[x15178];
x15177[x15178] = x15183;
}

}
float* x15189 = (float*)myMalloc(131072 * sizeof(float));;
float* x15190 = (float*)myMalloc(1179648 * sizeof(float));;
for(int x15191=0; x15191 < 64; x15191++) {
int32_t x15192 = x15191 * 2048;
float* x15193 = x15177+x15192;
float* x15194 = x15189+x15192;
int32_t x15195 = x15191 * 18432;
float* x15196 = x15190+x15195;
for(int x15197=0; x15197 < 4608; x15197++) {
int32_t x15198 = x15197 / 9;
int32_t x15202 = x15198 * 3;
int32_t x15203 = x15202 * 3;
int32_t x15204 = x15203 * 2;
int32_t x15205 = x15204 * 2;
int32_t x15199 = x15197 % 9;
int32_t x15200 = x15199 / 3;
int32_t x15206 = x15200 * 3;
int32_t x15207 = x15206 * 2;
int32_t x15208 = x15207 * 2;
int32_t x15209 = x15205 + x15208;
int32_t x15201 = x15199 % 3;
int32_t x15210 = x15201 * 2;
int32_t x15211 = x15210 * 2;
int32_t x15212 = x15209 + x15211;
float* x15213 = x15196+x15212;
int32_t x15214 = x15198 * 2;
int32_t x15215 = x15214 * 2;
float* x15216 = x15193+x15215;
int32_t x15228 = 1 - x15201;
bool x15229 = x15228 > 0;
int32_t x15230;
if (x15229) {
x15230 = x15228;
} else {
x15230 = 0;
}
int32_t x15231 = 3 - x15201;
int32_t x15232 = x15231 - 1;
int32_t x15233 = 1 - x15232;
bool x15234 = x15233 > 0;
int32_t x15235;
if (x15234) {
x15235 = x15233;
} else {
x15235 = 0;
}
int32_t x15236 = 2 - x15235;
int32_t x15237 = x15236 - x15230;
bool x15238 = x15237 <= 0;
bool x15242 = x15230 > 0;
int32_t x15227 = -1 + x15201;
bool x15255 = x15235 > 0;
for(int x15217=0; x15217 < 2; x15217++) {
int32_t x15218 = x15217 - 1;
int32_t x15219 = x15218 + x15200;
bool x15220 = x15219 < 0;
bool x15221 = x15219 >= 2;
bool x15222 = x15220 || x15221;
if (x15222) {
int32_t x15223 = x15217 * 2;
float* x15224 = x15213+x15223;
memset(x15224, 0, 4 * 2);;
} else {
if (x15238) {
int32_t x15223 = x15217 * 2;
float* x15239 = x15213+x15223;
memset(x15239, 0, 4 * 2);;
} else {
int32_t x15223 = x15217 * 2;
if (x15242) {
float* x15243 = x15213+x15223;
memset(x15243, 0, 4 * x15230);;
} else {
}
// may have segfault here
int32_t x15248 = x15223 + x15230;
float* x15249 = x15213+x15248;
int32_t x15250 = x15219 * 2;
int32_t x15251 = x15250 + x15227;
int32_t x15252 = x15251 + x15230;
float* x15253 = x15216+x15252;
memcpy(x15249, x15253, 4 * x15237);;
if (x15255) {
int32_t x15256 = x15223 + 2;
int32_t x15257 = x15256 - x15235;
float* x15258 = x15213+x15257;
memset(x15258, 0, 4 * x15235);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,4,4608,1,x154,4608,x15196,4,1,x15194,4);

}
// resize to WrappedArray(-1, 1, 1)
float* x15274 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x15275 = 0;
int32_t x15276 = 0;
int32_t x15277 = 0;
for(int x15278=0; x15278 < 64; x15278++) {
int32_t x15279 = x15276;
int32_t x15280 = x15277;
int32_t x15281 = x15275;
int32_t x15282 = x15281;
int32_t x15283 = x15279;
int32_t x15284 = x15280;
for(int x15285=0; x15285 < 512; x15285++) {
int32_t x15286 = x15283;
int32_t x15287 = x15284;
int32_t x15288 = x15282;
int32_t x15289 = x15288;
int32_t x15290 = x15286;
int32_t x15291 = x15287;
int32_t x15292 = x15290;
int32_t x15293 = x15291;
int32_t x15294 = x15289;
int32_t x15295 = x15294;
int32_t x15296 = x15292;
int32_t x15297 = x15293;
int32_t x15298 = x15295;
int32_t x15299 = x15296;
float x15300 = x15189[x15299];
int32_t x15301 = x15297;
float x15302 = x137[x15301];
float x15303 = x15300 - x15302;
x15274[x15298] = x15303;
x15295 += 1;
x15296 += 1;
int32_t x15307 = x15295;
int32_t x15308 = x15296;
float x15309 = x15189[x15308];
float x15310 = x137[x15301];
float x15311 = x15309 - x15310;
x15274[x15307] = x15311;
x15295 += 1;
x15296 += 1;
x15289 += 2;
x15290 += 2;
int32_t x15317 = x15290;
int32_t x15318 = x15289;
int32_t x15319 = x15318;
int32_t x15320 = x15317;
int32_t x15321 = x15293;
int32_t x15322 = x15319;
int32_t x15323 = x15320;
float x15324 = x15189[x15323];
int32_t x15325 = x15321;
float x15326 = x137[x15325];
float x15327 = x15324 - x15326;
x15274[x15322] = x15327;
x15319 += 1;
x15320 += 1;
int32_t x15331 = x15319;
int32_t x15332 = x15320;
float x15333 = x15189[x15332];
float x15334 = x137[x15325];
float x15335 = x15333 - x15334;
x15274[x15331] = x15335;
x15319 += 1;
x15320 += 1;
x15289 += 2;
x15290 += 2;
x15282 += 4;
x15283 += 4;
x15284 += 1;

}
x15275 += 2048;
x15276 += 2048;

}
float* x15350 = (float*)myMalloc(512 * sizeof(float));;
for(int x15351=0; x15351 < 512; x15351++) {
float x15352 = x194[x15351];
float x15353 = x15352 + 1.0E-5f;
x15350[x15351] = x15353;

}
float* x15357 = (float*)myMalloc(512 * sizeof(float));;
for(int x15358=0; x15358 < 512; x15358++) {
float x15359 = x15350[x15358];
double x15360 = (double)x15359;
double x15361 = sqrt(x15360);
float x15362 = (float)x15361;
x15357[x15358] = x15362;

}
// resize to WrappedArray(-1, 1, 1)
float* x15367 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x15368 = 0;
int32_t x15369 = 0;
int32_t x15370 = 0;
for(int x15371=0; x15371 < 64; x15371++) {
int32_t x15372 = x15369;
int32_t x15373 = x15370;
int32_t x15374 = x15368;
int32_t x15375 = x15374;
int32_t x15376 = x15372;
int32_t x15377 = x15373;
for(int x15378=0; x15378 < 512; x15378++) {
int32_t x15379 = x15376;
int32_t x15380 = x15377;
int32_t x15381 = x15375;
int32_t x15382 = x15381;
int32_t x15383 = x15379;
int32_t x15384 = x15380;
int32_t x15385 = x15383;
int32_t x15386 = x15384;
int32_t x15387 = x15382;
int32_t x15388 = x15387;
int32_t x15389 = x15385;
int32_t x15390 = x15386;
int32_t x15391 = x15388;
int32_t x15392 = x15389;
float x15393 = x15274[x15392];
int32_t x15394 = x15390;
float x15395 = x15357[x15394];
float x15396 = x15393 / x15395;
x15367[x15391] = x15396;
x15388 += 1;
x15389 += 1;
int32_t x15400 = x15388;
int32_t x15401 = x15389;
float x15402 = x15274[x15401];
float x15403 = x15357[x15394];
float x15404 = x15402 / x15403;
x15367[x15400] = x15404;
x15388 += 1;
x15389 += 1;
x15382 += 2;
x15383 += 2;
int32_t x15410 = x15383;
int32_t x15411 = x15382;
int32_t x15412 = x15411;
int32_t x15413 = x15410;
int32_t x15414 = x15386;
int32_t x15415 = x15412;
int32_t x15416 = x15413;
float x15417 = x15274[x15416];
int32_t x15418 = x15414;
float x15419 = x15357[x15418];
float x15420 = x15417 / x15419;
x15367[x15415] = x15420;
x15412 += 1;
x15413 += 1;
int32_t x15424 = x15412;
int32_t x15425 = x15413;
float x15426 = x15274[x15425];
float x15427 = x15357[x15418];
float x15428 = x15426 / x15427;
x15367[x15424] = x15428;
x15412 += 1;
x15413 += 1;
x15382 += 2;
x15383 += 2;
x15375 += 4;
x15376 += 4;
x15377 += 1;

}
x15368 += 2048;
x15369 += 2048;

}
// resize to WrappedArray(-1, 1, 1)
float* x15444 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x15445 = 0;
int32_t x15446 = 0;
int32_t x15447 = 0;
for(int x15448=0; x15448 < 64; x15448++) {
int32_t x15449 = x15446;
int32_t x15450 = x15447;
int32_t x15451 = x15445;
int32_t x15452 = x15451;
int32_t x15453 = x15449;
int32_t x15454 = x15450;
for(int x15455=0; x15455 < 512; x15455++) {
int32_t x15456 = x15453;
int32_t x15457 = x15454;
int32_t x15458 = x15452;
int32_t x15459 = x15458;
int32_t x15460 = x15456;
int32_t x15461 = x15457;
int32_t x15462 = x15460;
int32_t x15463 = x15461;
int32_t x15464 = x15459;
int32_t x15465 = x15464;
int32_t x15466 = x15462;
int32_t x15467 = x15463;
int32_t x15468 = x15465;
int32_t x15469 = x15466;
float x15470 = x15367[x15469];
int32_t x15471 = x15467;
float x15472 = x159[x15471];
float x15473 = x15470 * x15472;
x15444[x15468] = x15473;
x15465 += 1;
x15466 += 1;
int32_t x15477 = x15465;
int32_t x15478 = x15466;
float x15479 = x15367[x15478];
float x15480 = x159[x15471];
float x15481 = x15479 * x15480;
x15444[x15477] = x15481;
x15465 += 1;
x15466 += 1;
x15459 += 2;
x15460 += 2;
int32_t x15487 = x15460;
int32_t x15488 = x15459;
int32_t x15489 = x15488;
int32_t x15490 = x15487;
int32_t x15491 = x15463;
int32_t x15492 = x15489;
int32_t x15493 = x15490;
float x15494 = x15367[x15493];
int32_t x15495 = x15491;
float x15496 = x159[x15495];
float x15497 = x15494 * x15496;
x15444[x15492] = x15497;
x15489 += 1;
x15490 += 1;
int32_t x15501 = x15489;
int32_t x15502 = x15490;
float x15503 = x15367[x15502];
float x15504 = x159[x15495];
float x15505 = x15503 * x15504;
x15444[x15501] = x15505;
x15489 += 1;
x15490 += 1;
x15459 += 2;
x15460 += 2;
x15452 += 4;
x15453 += 4;
x15454 += 1;

}
x15445 += 2048;
x15446 += 2048;

}
// resize to WrappedArray(-1, 1, 1)
float* x15521 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x15522 = 0;
int32_t x15523 = 0;
int32_t x15524 = 0;
for(int x15525=0; x15525 < 64; x15525++) {
int32_t x15526 = x15523;
int32_t x15527 = x15524;
int32_t x15528 = x15522;
int32_t x15529 = x15528;
int32_t x15530 = x15526;
int32_t x15531 = x15527;
for(int x15532=0; x15532 < 512; x15532++) {
int32_t x15533 = x15530;
int32_t x15534 = x15531;
int32_t x15535 = x15529;
int32_t x15536 = x15535;
int32_t x15537 = x15533;
int32_t x15538 = x15534;
int32_t x15539 = x15537;
int32_t x15540 = x15538;
int32_t x15541 = x15536;
int32_t x15542 = x15541;
int32_t x15543 = x15539;
int32_t x15544 = x15540;
int32_t x15545 = x15542;
int32_t x15546 = x15543;
float x15547 = x15444[x15546];
int32_t x15548 = x15544;
float x15549 = x65[x15548];
float x15550 = x15547 + x15549;
x15521[x15545] = x15550;
x15542 += 1;
x15543 += 1;
int32_t x15554 = x15542;
int32_t x15555 = x15543;
float x15556 = x15444[x15555];
float x15557 = x65[x15548];
float x15558 = x15556 + x15557;
x15521[x15554] = x15558;
x15542 += 1;
x15543 += 1;
x15536 += 2;
x15537 += 2;
int32_t x15564 = x15537;
int32_t x15565 = x15536;
int32_t x15566 = x15565;
int32_t x15567 = x15564;
int32_t x15568 = x15540;
int32_t x15569 = x15566;
int32_t x15570 = x15567;
float x15571 = x15444[x15570];
int32_t x15572 = x15568;
float x15573 = x65[x15572];
float x15574 = x15571 + x15573;
x15521[x15569] = x15574;
x15566 += 1;
x15567 += 1;
int32_t x15578 = x15566;
int32_t x15579 = x15567;
float x15580 = x15444[x15579];
float x15581 = x65[x15572];
float x15582 = x15580 + x15581;
x15521[x15578] = x15582;
x15566 += 1;
x15567 += 1;
x15536 += 2;
x15537 += 2;
x15529 += 4;
x15530 += 4;
x15531 += 1;

}
x15522 += 2048;
x15523 += 2048;

}
float* x15597 = (float*)myMalloc(131072 * sizeof(float));;
for(int x15598=0; x15598 < 131072; x15598++) {
float x15599 = x15521[x15598];
bool x15600 = x15599 < 0.0f;
if (x15600) {
x15597[x15598] = 0.0f;
} else {
float x15603 = x15521[x15598];
x15597[x15598] = x15603;
}

}
float* x15609 = (float*)myMalloc(524288 * sizeof(float));;
float* x15610 = (float*)myMalloc(131072 * sizeof(float));;
for(int x15611=0; x15611 < 64; x15611++) {
int32_t x15612 = x15611 * 2048;
float* x15613 = x15597+x15612;
int32_t x15614 = x15611 * 8192;
float* x15615 = x15609+x15614;
float* x15616 = x15610+x15612;
for(int x15617=0; x15617 < 512; x15617++) {
int32_t x15618 = x15617 / 1;
int32_t x15622 = x15618 * 2;
int32_t x15623 = x15622 * 2;
int32_t x15619 = x15617 % 1;
int32_t x15620 = x15619 / 1;
int32_t x15624 = x15620 * 2;
int32_t x15625 = x15624 * 2;
int32_t x15626 = x15623 + x15625;
int32_t x15621 = x15619 % 1;
int32_t x15627 = x15621 * 2;
int32_t x15628 = x15627 * 2;
int32_t x15629 = x15626 + x15628;
float* x15630 = x15616+x15629;
float* x15631 = x15613+x15623;
for(int x15632=0; x15632 < 2; x15632++) {
int32_t x15634 = x15632 * 2;
float* x15635 = x15630+x15634;
int32_t x15633 = x15632 + x15620;
int32_t x15636 = x15633 * 2;
int32_t x15637 = x15636 + x15621;
float* x15638 = x15631+x15637;
memcpy(x15635, x15638, 4 * 2);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,4,512,1,x46,512,x15616,4,1,x15615,4);

}
// resize to WrappedArray(-1, 1, 1)
float* x15648 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x15649 = 0;
int32_t x15650 = 0;
int32_t x15651 = 0;
for(int x15652=0; x15652 < 64; x15652++) {
int32_t x15653 = x15650;
int32_t x15654 = x15651;
int32_t x15655 = x15649;
int32_t x15656 = x15655;
int32_t x15657 = x15653;
int32_t x15658 = x15654;
for(int x15659=0; x15659 < 2048; x15659++) {
int32_t x15660 = x15657;
int32_t x15661 = x15658;
int32_t x15662 = x15656;
int32_t x15663 = x15662;
int32_t x15664 = x15660;
int32_t x15665 = x15661;
int32_t x15666 = x15664;
int32_t x15667 = x15665;
int32_t x15668 = x15663;
int32_t x15669 = x15668;
int32_t x15670 = x15666;
int32_t x15671 = x15667;
int32_t x15672 = x15669;
int32_t x15673 = x15670;
float x15674 = x15609[x15673];
int32_t x15675 = x15671;
float x15676 = x67[x15675];
float x15677 = x15674 - x15676;
x15648[x15672] = x15677;
x15669 += 1;
x15670 += 1;
int32_t x15681 = x15669;
int32_t x15682 = x15670;
float x15683 = x15609[x15682];
float x15684 = x67[x15675];
float x15685 = x15683 - x15684;
x15648[x15681] = x15685;
x15669 += 1;
x15670 += 1;
x15663 += 2;
x15664 += 2;
int32_t x15691 = x15664;
int32_t x15692 = x15663;
int32_t x15693 = x15692;
int32_t x15694 = x15691;
int32_t x15695 = x15667;
int32_t x15696 = x15693;
int32_t x15697 = x15694;
float x15698 = x15609[x15697];
int32_t x15699 = x15695;
float x15700 = x67[x15699];
float x15701 = x15698 - x15700;
x15648[x15696] = x15701;
x15693 += 1;
x15694 += 1;
int32_t x15705 = x15693;
int32_t x15706 = x15694;
float x15707 = x15609[x15706];
float x15708 = x67[x15699];
float x15709 = x15707 - x15708;
x15648[x15705] = x15709;
x15693 += 1;
x15694 += 1;
x15663 += 2;
x15664 += 2;
x15656 += 4;
x15657 += 4;
x15658 += 1;

}
x15649 += 8192;
x15650 += 8192;

}
float* x15724 = (float*)myMalloc(2048 * sizeof(float));;
for(int x15725=0; x15725 < 2048; x15725++) {
float x15726 = x244[x15725];
float x15727 = x15726 + 1.0E-5f;
x15724[x15725] = x15727;

}
float* x15731 = (float*)myMalloc(2048 * sizeof(float));;
for(int x15732=0; x15732 < 2048; x15732++) {
float x15733 = x15724[x15732];
double x15734 = (double)x15733;
double x15735 = sqrt(x15734);
float x15736 = (float)x15735;
x15731[x15732] = x15736;

}
// resize to WrappedArray(-1, 1, 1)
float* x15741 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x15742 = 0;
int32_t x15743 = 0;
int32_t x15744 = 0;
for(int x15745=0; x15745 < 64; x15745++) {
int32_t x15746 = x15743;
int32_t x15747 = x15744;
int32_t x15748 = x15742;
int32_t x15749 = x15748;
int32_t x15750 = x15746;
int32_t x15751 = x15747;
for(int x15752=0; x15752 < 2048; x15752++) {
int32_t x15753 = x15750;
int32_t x15754 = x15751;
int32_t x15755 = x15749;
int32_t x15756 = x15755;
int32_t x15757 = x15753;
int32_t x15758 = x15754;
int32_t x15759 = x15757;
int32_t x15760 = x15758;
int32_t x15761 = x15756;
int32_t x15762 = x15761;
int32_t x15763 = x15759;
int32_t x15764 = x15760;
int32_t x15765 = x15762;
int32_t x15766 = x15763;
float x15767 = x15648[x15766];
int32_t x15768 = x15764;
float x15769 = x15731[x15768];
float x15770 = x15767 / x15769;
x15741[x15765] = x15770;
x15762 += 1;
x15763 += 1;
int32_t x15774 = x15762;
int32_t x15775 = x15763;
float x15776 = x15648[x15775];
float x15777 = x15731[x15768];
float x15778 = x15776 / x15777;
x15741[x15774] = x15778;
x15762 += 1;
x15763 += 1;
x15756 += 2;
x15757 += 2;
int32_t x15784 = x15757;
int32_t x15785 = x15756;
int32_t x15786 = x15785;
int32_t x15787 = x15784;
int32_t x15788 = x15760;
int32_t x15789 = x15786;
int32_t x15790 = x15787;
float x15791 = x15648[x15790];
int32_t x15792 = x15788;
float x15793 = x15731[x15792];
float x15794 = x15791 / x15793;
x15741[x15789] = x15794;
x15786 += 1;
x15787 += 1;
int32_t x15798 = x15786;
int32_t x15799 = x15787;
float x15800 = x15648[x15799];
float x15801 = x15731[x15792];
float x15802 = x15800 / x15801;
x15741[x15798] = x15802;
x15786 += 1;
x15787 += 1;
x15756 += 2;
x15757 += 2;
x15749 += 4;
x15750 += 4;
x15751 += 1;

}
x15742 += 8192;
x15743 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x15818 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x15819 = 0;
int32_t x15820 = 0;
int32_t x15821 = 0;
for(int x15822=0; x15822 < 64; x15822++) {
int32_t x15823 = x15820;
int32_t x15824 = x15821;
int32_t x15825 = x15819;
int32_t x15826 = x15825;
int32_t x15827 = x15823;
int32_t x15828 = x15824;
for(int x15829=0; x15829 < 2048; x15829++) {
int32_t x15830 = x15827;
int32_t x15831 = x15828;
int32_t x15832 = x15826;
int32_t x15833 = x15832;
int32_t x15834 = x15830;
int32_t x15835 = x15831;
int32_t x15836 = x15834;
int32_t x15837 = x15835;
int32_t x15838 = x15833;
int32_t x15839 = x15838;
int32_t x15840 = x15836;
int32_t x15841 = x15837;
int32_t x15842 = x15839;
int32_t x15843 = x15840;
float x15844 = x15741[x15843];
int32_t x15845 = x15841;
float x15846 = x93[x15845];
float x15847 = x15844 * x15846;
x15818[x15842] = x15847;
x15839 += 1;
x15840 += 1;
int32_t x15851 = x15839;
int32_t x15852 = x15840;
float x15853 = x15741[x15852];
float x15854 = x93[x15845];
float x15855 = x15853 * x15854;
x15818[x15851] = x15855;
x15839 += 1;
x15840 += 1;
x15833 += 2;
x15834 += 2;
int32_t x15861 = x15834;
int32_t x15862 = x15833;
int32_t x15863 = x15862;
int32_t x15864 = x15861;
int32_t x15865 = x15837;
int32_t x15866 = x15863;
int32_t x15867 = x15864;
float x15868 = x15741[x15867];
int32_t x15869 = x15865;
float x15870 = x93[x15869];
float x15871 = x15868 * x15870;
x15818[x15866] = x15871;
x15863 += 1;
x15864 += 1;
int32_t x15875 = x15863;
int32_t x15876 = x15864;
float x15877 = x15741[x15876];
float x15878 = x93[x15869];
float x15879 = x15877 * x15878;
x15818[x15875] = x15879;
x15863 += 1;
x15864 += 1;
x15833 += 2;
x15834 += 2;
x15826 += 4;
x15827 += 4;
x15828 += 1;

}
x15819 += 8192;
x15820 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x15895 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x15896 = 0;
int32_t x15897 = 0;
int32_t x15898 = 0;
for(int x15899=0; x15899 < 64; x15899++) {
int32_t x15900 = x15897;
int32_t x15901 = x15898;
int32_t x15902 = x15896;
int32_t x15903 = x15902;
int32_t x15904 = x15900;
int32_t x15905 = x15901;
for(int x15906=0; x15906 < 2048; x15906++) {
int32_t x15907 = x15904;
int32_t x15908 = x15905;
int32_t x15909 = x15903;
int32_t x15910 = x15909;
int32_t x15911 = x15907;
int32_t x15912 = x15908;
int32_t x15913 = x15911;
int32_t x15914 = x15912;
int32_t x15915 = x15910;
int32_t x15916 = x15915;
int32_t x15917 = x15913;
int32_t x15918 = x15914;
int32_t x15919 = x15916;
int32_t x15920 = x15917;
float x15921 = x15818[x15920];
int32_t x15922 = x15918;
float x15923 = x143[x15922];
float x15924 = x15921 + x15923;
x15895[x15919] = x15924;
x15916 += 1;
x15917 += 1;
int32_t x15928 = x15916;
int32_t x15929 = x15917;
float x15930 = x15818[x15929];
float x15931 = x143[x15922];
float x15932 = x15930 + x15931;
x15895[x15928] = x15932;
x15916 += 1;
x15917 += 1;
x15910 += 2;
x15911 += 2;
int32_t x15938 = x15911;
int32_t x15939 = x15910;
int32_t x15940 = x15939;
int32_t x15941 = x15938;
int32_t x15942 = x15914;
int32_t x15943 = x15940;
int32_t x15944 = x15941;
float x15945 = x15818[x15944];
int32_t x15946 = x15942;
float x15947 = x143[x15946];
float x15948 = x15945 + x15947;
x15895[x15943] = x15948;
x15940 += 1;
x15941 += 1;
int32_t x15952 = x15940;
int32_t x15953 = x15941;
float x15954 = x15818[x15953];
float x15955 = x143[x15946];
float x15956 = x15954 + x15955;
x15895[x15952] = x15956;
x15940 += 1;
x15941 += 1;
x15910 += 2;
x15911 += 2;
x15903 += 4;
x15904 += 4;
x15905 += 1;

}
x15896 += 8192;
x15897 += 8192;

}
int32_t x15971 = 0;
int32_t x15972 = 0;
int32_t x15973 = 0;
for(int x15974=0; x15974 < 64; x15974++) {
int32_t x15975 = x15972;
int32_t x15976 = x15973;
int32_t x15977 = x15971;
int32_t x15978 = x15977;
int32_t x15979 = x15975;
int32_t x15980 = x15976;
for(int x15981=0; x15981 < 2048; x15981++) {
int32_t x15982 = x15979;
int32_t x15983 = x15980;
int32_t x15984 = x15978;
int32_t x15985 = x15984;
int32_t x15986 = x15982;
int32_t x15987 = x15983;
int32_t x15988 = x15986;
int32_t x15989 = x15987;
int32_t x15990 = x15985;
int32_t x15991 = x15990;
int32_t x15992 = x15988;
int32_t x15993 = x15989;
int32_t x15994 = x15992;
float x15995 = x15895[x15994];
int32_t x15996 = x15993;
float x15997 = x14803[x15996];
float x15998 = x15995 + x15997;
x15895[x15994] = x15998;
x15991 += 1;
x15992 += 1;
x15993 += 1;
int32_t x16003 = x15992;
float x16004 = x15895[x16003];
int32_t x16005 = x15993;
float x16006 = x14803[x16005];
float x16007 = x16004 + x16006;
x15895[x16003] = x16007;
x15991 += 1;
x15992 += 1;
x15993 += 1;
x15985 += 2;
x15986 += 2;
x15987 += 2;
int32_t x16015 = x15986;
int32_t x16016 = x15987;
int32_t x16017 = x15985;
int32_t x16018 = x16017;
int32_t x16019 = x16015;
int32_t x16020 = x16016;
int32_t x16021 = x16019;
float x16022 = x15895[x16021];
int32_t x16023 = x16020;
float x16024 = x14803[x16023];
float x16025 = x16022 + x16024;
x15895[x16021] = x16025;
x16018 += 1;
x16019 += 1;
x16020 += 1;
int32_t x16030 = x16019;
float x16031 = x15895[x16030];
int32_t x16032 = x16020;
float x16033 = x14803[x16032];
float x16034 = x16031 + x16033;
x15895[x16030] = x16034;
x16018 += 1;
x16019 += 1;
x16020 += 1;
x15985 += 2;
x15986 += 2;
x15987 += 2;
x15978 += 4;
x15979 += 4;
x15980 += 4;

}
x15971 += 8192;
x15972 += 8192;
x15973 += 8192;

}
float* x16052 = (float*)myMalloc(524288 * sizeof(float));;
for(int x16053=0; x16053 < 524288; x16053++) {
float x16054 = x15895[x16053];
bool x16055 = x16054 < 0.0f;
if (x16055) {
x16052[x16053] = 0.0f;
} else {
float x16058 = x15895[x16053];
x16052[x16053] = x16058;
}

}
float* x16064 = (float*)myMalloc(131072 * sizeof(float));;
float* x16065 = (float*)myMalloc(524288 * sizeof(float));;
for(int x16066=0; x16066 < 64; x16066++) {
int32_t x16067 = x16066 * 8192;
float* x16068 = x16052+x16067;
int32_t x16069 = x16066 * 2048;
float* x16070 = x16064+x16069;
float* x16071 = x16065+x16067;
for(int x16072=0; x16072 < 2048; x16072++) {
int32_t x16073 = x16072 / 1;
int32_t x16077 = x16073 * 2;
int32_t x16078 = x16077 * 2;
int32_t x16074 = x16072 % 1;
int32_t x16075 = x16074 / 1;
int32_t x16079 = x16075 * 2;
int32_t x16080 = x16079 * 2;
int32_t x16081 = x16078 + x16080;
int32_t x16076 = x16074 % 1;
int32_t x16082 = x16076 * 2;
int32_t x16083 = x16082 * 2;
int32_t x16084 = x16081 + x16083;
float* x16085 = x16071+x16084;
float* x16086 = x16068+x16078;
for(int x16087=0; x16087 < 2; x16087++) {
int32_t x16089 = x16087 * 2;
float* x16090 = x16085+x16089;
int32_t x16088 = x16087 + x16075;
int32_t x16091 = x16088 * 2;
int32_t x16092 = x16091 + x16076;
float* x16093 = x16086+x16092;
memcpy(x16090, x16093, 4 * 2);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,4,2048,1,x264,2048,x16071,4,1,x16070,4);

}
// resize to WrappedArray(-1, 1, 1)
float* x16103 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x16104 = 0;
int32_t x16105 = 0;
int32_t x16106 = 0;
for(int x16107=0; x16107 < 64; x16107++) {
int32_t x16108 = x16105;
int32_t x16109 = x16106;
int32_t x16110 = x16104;
int32_t x16111 = x16110;
int32_t x16112 = x16108;
int32_t x16113 = x16109;
for(int x16114=0; x16114 < 512; x16114++) {
int32_t x16115 = x16112;
int32_t x16116 = x16113;
int32_t x16117 = x16111;
int32_t x16118 = x16117;
int32_t x16119 = x16115;
int32_t x16120 = x16116;
int32_t x16121 = x16119;
int32_t x16122 = x16120;
int32_t x16123 = x16118;
int32_t x16124 = x16123;
int32_t x16125 = x16121;
int32_t x16126 = x16122;
int32_t x16127 = x16124;
int32_t x16128 = x16125;
float x16129 = x16064[x16128];
int32_t x16130 = x16126;
float x16131 = x212[x16130];
float x16132 = x16129 - x16131;
x16103[x16127] = x16132;
x16124 += 1;
x16125 += 1;
int32_t x16136 = x16124;
int32_t x16137 = x16125;
float x16138 = x16064[x16137];
float x16139 = x212[x16130];
float x16140 = x16138 - x16139;
x16103[x16136] = x16140;
x16124 += 1;
x16125 += 1;
x16118 += 2;
x16119 += 2;
int32_t x16146 = x16119;
int32_t x16147 = x16118;
int32_t x16148 = x16147;
int32_t x16149 = x16146;
int32_t x16150 = x16122;
int32_t x16151 = x16148;
int32_t x16152 = x16149;
float x16153 = x16064[x16152];
int32_t x16154 = x16150;
float x16155 = x212[x16154];
float x16156 = x16153 - x16155;
x16103[x16151] = x16156;
x16148 += 1;
x16149 += 1;
int32_t x16160 = x16148;
int32_t x16161 = x16149;
float x16162 = x16064[x16161];
float x16163 = x212[x16154];
float x16164 = x16162 - x16163;
x16103[x16160] = x16164;
x16148 += 1;
x16149 += 1;
x16118 += 2;
x16119 += 2;
x16111 += 4;
x16112 += 4;
x16113 += 1;

}
x16104 += 2048;
x16105 += 2048;

}
float* x16179 = (float*)myMalloc(512 * sizeof(float));;
for(int x16180=0; x16180 < 512; x16180++) {
float x16181 = x254[x16180];
float x16182 = x16181 + 1.0E-5f;
x16179[x16180] = x16182;

}
float* x16186 = (float*)myMalloc(512 * sizeof(float));;
for(int x16187=0; x16187 < 512; x16187++) {
float x16188 = x16179[x16187];
double x16189 = (double)x16188;
double x16190 = sqrt(x16189);
float x16191 = (float)x16190;
x16186[x16187] = x16191;

}
// resize to WrappedArray(-1, 1, 1)
float* x16196 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x16197 = 0;
int32_t x16198 = 0;
int32_t x16199 = 0;
for(int x16200=0; x16200 < 64; x16200++) {
int32_t x16201 = x16198;
int32_t x16202 = x16199;
int32_t x16203 = x16197;
int32_t x16204 = x16203;
int32_t x16205 = x16201;
int32_t x16206 = x16202;
for(int x16207=0; x16207 < 512; x16207++) {
int32_t x16208 = x16205;
int32_t x16209 = x16206;
int32_t x16210 = x16204;
int32_t x16211 = x16210;
int32_t x16212 = x16208;
int32_t x16213 = x16209;
int32_t x16214 = x16212;
int32_t x16215 = x16213;
int32_t x16216 = x16211;
int32_t x16217 = x16216;
int32_t x16218 = x16214;
int32_t x16219 = x16215;
int32_t x16220 = x16217;
int32_t x16221 = x16218;
float x16222 = x16103[x16221];
int32_t x16223 = x16219;
float x16224 = x16186[x16223];
float x16225 = x16222 / x16224;
x16196[x16220] = x16225;
x16217 += 1;
x16218 += 1;
int32_t x16229 = x16217;
int32_t x16230 = x16218;
float x16231 = x16103[x16230];
float x16232 = x16186[x16223];
float x16233 = x16231 / x16232;
x16196[x16229] = x16233;
x16217 += 1;
x16218 += 1;
x16211 += 2;
x16212 += 2;
int32_t x16239 = x16212;
int32_t x16240 = x16211;
int32_t x16241 = x16240;
int32_t x16242 = x16239;
int32_t x16243 = x16215;
int32_t x16244 = x16241;
int32_t x16245 = x16242;
float x16246 = x16103[x16245];
int32_t x16247 = x16243;
float x16248 = x16186[x16247];
float x16249 = x16246 / x16248;
x16196[x16244] = x16249;
x16241 += 1;
x16242 += 1;
int32_t x16253 = x16241;
int32_t x16254 = x16242;
float x16255 = x16103[x16254];
float x16256 = x16186[x16247];
float x16257 = x16255 / x16256;
x16196[x16253] = x16257;
x16241 += 1;
x16242 += 1;
x16211 += 2;
x16212 += 2;
x16204 += 4;
x16205 += 4;
x16206 += 1;

}
x16197 += 2048;
x16198 += 2048;

}
// resize to WrappedArray(-1, 1, 1)
float* x16273 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x16274 = 0;
int32_t x16275 = 0;
int32_t x16276 = 0;
for(int x16277=0; x16277 < 64; x16277++) {
int32_t x16278 = x16275;
int32_t x16279 = x16276;
int32_t x16280 = x16274;
int32_t x16281 = x16280;
int32_t x16282 = x16278;
int32_t x16283 = x16279;
for(int x16284=0; x16284 < 512; x16284++) {
int32_t x16285 = x16282;
int32_t x16286 = x16283;
int32_t x16287 = x16281;
int32_t x16288 = x16287;
int32_t x16289 = x16285;
int32_t x16290 = x16286;
int32_t x16291 = x16289;
int32_t x16292 = x16290;
int32_t x16293 = x16288;
int32_t x16294 = x16293;
int32_t x16295 = x16291;
int32_t x16296 = x16292;
int32_t x16297 = x16294;
int32_t x16298 = x16295;
float x16299 = x16196[x16298];
int32_t x16300 = x16296;
float x16301 = x14[x16300];
float x16302 = x16299 * x16301;
x16273[x16297] = x16302;
x16294 += 1;
x16295 += 1;
int32_t x16306 = x16294;
int32_t x16307 = x16295;
float x16308 = x16196[x16307];
float x16309 = x14[x16300];
float x16310 = x16308 * x16309;
x16273[x16306] = x16310;
x16294 += 1;
x16295 += 1;
x16288 += 2;
x16289 += 2;
int32_t x16316 = x16289;
int32_t x16317 = x16288;
int32_t x16318 = x16317;
int32_t x16319 = x16316;
int32_t x16320 = x16292;
int32_t x16321 = x16318;
int32_t x16322 = x16319;
float x16323 = x16196[x16322];
int32_t x16324 = x16320;
float x16325 = x14[x16324];
float x16326 = x16323 * x16325;
x16273[x16321] = x16326;
x16318 += 1;
x16319 += 1;
int32_t x16330 = x16318;
int32_t x16331 = x16319;
float x16332 = x16196[x16331];
float x16333 = x14[x16324];
float x16334 = x16332 * x16333;
x16273[x16330] = x16334;
x16318 += 1;
x16319 += 1;
x16288 += 2;
x16289 += 2;
x16281 += 4;
x16282 += 4;
x16283 += 1;

}
x16274 += 2048;
x16275 += 2048;

}
// resize to WrappedArray(-1, 1, 1)
float* x16350 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x16351 = 0;
int32_t x16352 = 0;
int32_t x16353 = 0;
for(int x16354=0; x16354 < 64; x16354++) {
int32_t x16355 = x16352;
int32_t x16356 = x16353;
int32_t x16357 = x16351;
int32_t x16358 = x16357;
int32_t x16359 = x16355;
int32_t x16360 = x16356;
for(int x16361=0; x16361 < 512; x16361++) {
int32_t x16362 = x16359;
int32_t x16363 = x16360;
int32_t x16364 = x16358;
int32_t x16365 = x16364;
int32_t x16366 = x16362;
int32_t x16367 = x16363;
int32_t x16368 = x16366;
int32_t x16369 = x16367;
int32_t x16370 = x16365;
int32_t x16371 = x16370;
int32_t x16372 = x16368;
int32_t x16373 = x16369;
int32_t x16374 = x16371;
int32_t x16375 = x16372;
float x16376 = x16273[x16375];
int32_t x16377 = x16373;
float x16378 = x77[x16377];
float x16379 = x16376 + x16378;
x16350[x16374] = x16379;
x16371 += 1;
x16372 += 1;
int32_t x16383 = x16371;
int32_t x16384 = x16372;
float x16385 = x16273[x16384];
float x16386 = x77[x16377];
float x16387 = x16385 + x16386;
x16350[x16383] = x16387;
x16371 += 1;
x16372 += 1;
x16365 += 2;
x16366 += 2;
int32_t x16393 = x16366;
int32_t x16394 = x16365;
int32_t x16395 = x16394;
int32_t x16396 = x16393;
int32_t x16397 = x16369;
int32_t x16398 = x16395;
int32_t x16399 = x16396;
float x16400 = x16273[x16399];
int32_t x16401 = x16397;
float x16402 = x77[x16401];
float x16403 = x16400 + x16402;
x16350[x16398] = x16403;
x16395 += 1;
x16396 += 1;
int32_t x16407 = x16395;
int32_t x16408 = x16396;
float x16409 = x16273[x16408];
float x16410 = x77[x16401];
float x16411 = x16409 + x16410;
x16350[x16407] = x16411;
x16395 += 1;
x16396 += 1;
x16365 += 2;
x16366 += 2;
x16358 += 4;
x16359 += 4;
x16360 += 1;

}
x16351 += 2048;
x16352 += 2048;

}
float* x16426 = (float*)myMalloc(131072 * sizeof(float));;
for(int x16427=0; x16427 < 131072; x16427++) {
float x16428 = x16350[x16427];
bool x16429 = x16428 < 0.0f;
if (x16429) {
x16426[x16427] = 0.0f;
} else {
float x16432 = x16350[x16427];
x16426[x16427] = x16432;
}

}
float* x16438 = (float*)myMalloc(131072 * sizeof(float));;
float* x16439 = (float*)myMalloc(1179648 * sizeof(float));;
for(int x16440=0; x16440 < 64; x16440++) {
int32_t x16441 = x16440 * 2048;
float* x16442 = x16426+x16441;
float* x16443 = x16438+x16441;
int32_t x16444 = x16440 * 18432;
float* x16445 = x16439+x16444;
for(int x16446=0; x16446 < 4608; x16446++) {
int32_t x16447 = x16446 / 9;
int32_t x16451 = x16447 * 3;
int32_t x16452 = x16451 * 3;
int32_t x16453 = x16452 * 2;
int32_t x16454 = x16453 * 2;
int32_t x16448 = x16446 % 9;
int32_t x16449 = x16448 / 3;
int32_t x16455 = x16449 * 3;
int32_t x16456 = x16455 * 2;
int32_t x16457 = x16456 * 2;
int32_t x16458 = x16454 + x16457;
int32_t x16450 = x16448 % 3;
int32_t x16459 = x16450 * 2;
int32_t x16460 = x16459 * 2;
int32_t x16461 = x16458 + x16460;
float* x16462 = x16445+x16461;
int32_t x16463 = x16447 * 2;
int32_t x16464 = x16463 * 2;
float* x16465 = x16442+x16464;
int32_t x16477 = 1 - x16450;
bool x16478 = x16477 > 0;
int32_t x16479;
if (x16478) {
x16479 = x16477;
} else {
x16479 = 0;
}
int32_t x16480 = 3 - x16450;
int32_t x16481 = x16480 - 1;
int32_t x16482 = 1 - x16481;
bool x16483 = x16482 > 0;
int32_t x16484;
if (x16483) {
x16484 = x16482;
} else {
x16484 = 0;
}
int32_t x16485 = 2 - x16484;
int32_t x16486 = x16485 - x16479;
bool x16487 = x16486 <= 0;
bool x16491 = x16479 > 0;
int32_t x16476 = -1 + x16450;
bool x16504 = x16484 > 0;
for(int x16466=0; x16466 < 2; x16466++) {
int32_t x16467 = x16466 - 1;
int32_t x16468 = x16467 + x16449;
bool x16469 = x16468 < 0;
bool x16470 = x16468 >= 2;
bool x16471 = x16469 || x16470;
if (x16471) {
int32_t x16472 = x16466 * 2;
float* x16473 = x16462+x16472;
memset(x16473, 0, 4 * 2);;
} else {
if (x16487) {
int32_t x16472 = x16466 * 2;
float* x16488 = x16462+x16472;
memset(x16488, 0, 4 * 2);;
} else {
int32_t x16472 = x16466 * 2;
if (x16491) {
float* x16492 = x16462+x16472;
memset(x16492, 0, 4 * x16479);;
} else {
}
// may have segfault here
int32_t x16497 = x16472 + x16479;
float* x16498 = x16462+x16497;
int32_t x16499 = x16468 * 2;
int32_t x16500 = x16499 + x16476;
int32_t x16501 = x16500 + x16479;
float* x16502 = x16465+x16501;
memcpy(x16498, x16502, 4 * x16486);;
if (x16504) {
int32_t x16505 = x16472 + 2;
int32_t x16506 = x16505 - x16484;
float* x16507 = x16462+x16506;
memset(x16507, 0, 4 * x16484);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,4,4608,1,x27,4608,x16445,4,1,x16443,4);

}
// resize to WrappedArray(-1, 1, 1)
float* x16523 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x16524 = 0;
int32_t x16525 = 0;
int32_t x16526 = 0;
for(int x16527=0; x16527 < 64; x16527++) {
int32_t x16528 = x16525;
int32_t x16529 = x16526;
int32_t x16530 = x16524;
int32_t x16531 = x16530;
int32_t x16532 = x16528;
int32_t x16533 = x16529;
for(int x16534=0; x16534 < 512; x16534++) {
int32_t x16535 = x16532;
int32_t x16536 = x16533;
int32_t x16537 = x16531;
int32_t x16538 = x16537;
int32_t x16539 = x16535;
int32_t x16540 = x16536;
int32_t x16541 = x16539;
int32_t x16542 = x16540;
int32_t x16543 = x16538;
int32_t x16544 = x16543;
int32_t x16545 = x16541;
int32_t x16546 = x16542;
int32_t x16547 = x16544;
int32_t x16548 = x16545;
float x16549 = x16438[x16548];
int32_t x16550 = x16546;
float x16551 = x11[x16550];
float x16552 = x16549 - x16551;
x16523[x16547] = x16552;
x16544 += 1;
x16545 += 1;
int32_t x16556 = x16544;
int32_t x16557 = x16545;
float x16558 = x16438[x16557];
float x16559 = x11[x16550];
float x16560 = x16558 - x16559;
x16523[x16556] = x16560;
x16544 += 1;
x16545 += 1;
x16538 += 2;
x16539 += 2;
int32_t x16566 = x16539;
int32_t x16567 = x16538;
int32_t x16568 = x16567;
int32_t x16569 = x16566;
int32_t x16570 = x16542;
int32_t x16571 = x16568;
int32_t x16572 = x16569;
float x16573 = x16438[x16572];
int32_t x16574 = x16570;
float x16575 = x11[x16574];
float x16576 = x16573 - x16575;
x16523[x16571] = x16576;
x16568 += 1;
x16569 += 1;
int32_t x16580 = x16568;
int32_t x16581 = x16569;
float x16582 = x16438[x16581];
float x16583 = x11[x16574];
float x16584 = x16582 - x16583;
x16523[x16580] = x16584;
x16568 += 1;
x16569 += 1;
x16538 += 2;
x16539 += 2;
x16531 += 4;
x16532 += 4;
x16533 += 1;

}
x16524 += 2048;
x16525 += 2048;

}
float* x16599 = (float*)myMalloc(512 * sizeof(float));;
for(int x16600=0; x16600 < 512; x16600++) {
float x16601 = x201[x16600];
float x16602 = x16601 + 1.0E-5f;
x16599[x16600] = x16602;

}
float* x16606 = (float*)myMalloc(512 * sizeof(float));;
for(int x16607=0; x16607 < 512; x16607++) {
float x16608 = x16599[x16607];
double x16609 = (double)x16608;
double x16610 = sqrt(x16609);
float x16611 = (float)x16610;
x16606[x16607] = x16611;

}
// resize to WrappedArray(-1, 1, 1)
float* x16616 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x16617 = 0;
int32_t x16618 = 0;
int32_t x16619 = 0;
for(int x16620=0; x16620 < 64; x16620++) {
int32_t x16621 = x16618;
int32_t x16622 = x16619;
int32_t x16623 = x16617;
int32_t x16624 = x16623;
int32_t x16625 = x16621;
int32_t x16626 = x16622;
for(int x16627=0; x16627 < 512; x16627++) {
int32_t x16628 = x16625;
int32_t x16629 = x16626;
int32_t x16630 = x16624;
int32_t x16631 = x16630;
int32_t x16632 = x16628;
int32_t x16633 = x16629;
int32_t x16634 = x16632;
int32_t x16635 = x16633;
int32_t x16636 = x16631;
int32_t x16637 = x16636;
int32_t x16638 = x16634;
int32_t x16639 = x16635;
int32_t x16640 = x16637;
int32_t x16641 = x16638;
float x16642 = x16523[x16641];
int32_t x16643 = x16639;
float x16644 = x16606[x16643];
float x16645 = x16642 / x16644;
x16616[x16640] = x16645;
x16637 += 1;
x16638 += 1;
int32_t x16649 = x16637;
int32_t x16650 = x16638;
float x16651 = x16523[x16650];
float x16652 = x16606[x16643];
float x16653 = x16651 / x16652;
x16616[x16649] = x16653;
x16637 += 1;
x16638 += 1;
x16631 += 2;
x16632 += 2;
int32_t x16659 = x16632;
int32_t x16660 = x16631;
int32_t x16661 = x16660;
int32_t x16662 = x16659;
int32_t x16663 = x16635;
int32_t x16664 = x16661;
int32_t x16665 = x16662;
float x16666 = x16523[x16665];
int32_t x16667 = x16663;
float x16668 = x16606[x16667];
float x16669 = x16666 / x16668;
x16616[x16664] = x16669;
x16661 += 1;
x16662 += 1;
int32_t x16673 = x16661;
int32_t x16674 = x16662;
float x16675 = x16523[x16674];
float x16676 = x16606[x16667];
float x16677 = x16675 / x16676;
x16616[x16673] = x16677;
x16661 += 1;
x16662 += 1;
x16631 += 2;
x16632 += 2;
x16624 += 4;
x16625 += 4;
x16626 += 1;

}
x16617 += 2048;
x16618 += 2048;

}
// resize to WrappedArray(-1, 1, 1)
float* x16693 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x16694 = 0;
int32_t x16695 = 0;
int32_t x16696 = 0;
for(int x16697=0; x16697 < 64; x16697++) {
int32_t x16698 = x16695;
int32_t x16699 = x16696;
int32_t x16700 = x16694;
int32_t x16701 = x16700;
int32_t x16702 = x16698;
int32_t x16703 = x16699;
for(int x16704=0; x16704 < 512; x16704++) {
int32_t x16705 = x16702;
int32_t x16706 = x16703;
int32_t x16707 = x16701;
int32_t x16708 = x16707;
int32_t x16709 = x16705;
int32_t x16710 = x16706;
int32_t x16711 = x16709;
int32_t x16712 = x16710;
int32_t x16713 = x16708;
int32_t x16714 = x16713;
int32_t x16715 = x16711;
int32_t x16716 = x16712;
int32_t x16717 = x16714;
int32_t x16718 = x16715;
float x16719 = x16616[x16718];
int32_t x16720 = x16716;
float x16721 = x193[x16720];
float x16722 = x16719 * x16721;
x16693[x16717] = x16722;
x16714 += 1;
x16715 += 1;
int32_t x16726 = x16714;
int32_t x16727 = x16715;
float x16728 = x16616[x16727];
float x16729 = x193[x16720];
float x16730 = x16728 * x16729;
x16693[x16726] = x16730;
x16714 += 1;
x16715 += 1;
x16708 += 2;
x16709 += 2;
int32_t x16736 = x16709;
int32_t x16737 = x16708;
int32_t x16738 = x16737;
int32_t x16739 = x16736;
int32_t x16740 = x16712;
int32_t x16741 = x16738;
int32_t x16742 = x16739;
float x16743 = x16616[x16742];
int32_t x16744 = x16740;
float x16745 = x193[x16744];
float x16746 = x16743 * x16745;
x16693[x16741] = x16746;
x16738 += 1;
x16739 += 1;
int32_t x16750 = x16738;
int32_t x16751 = x16739;
float x16752 = x16616[x16751];
float x16753 = x193[x16744];
float x16754 = x16752 * x16753;
x16693[x16750] = x16754;
x16738 += 1;
x16739 += 1;
x16708 += 2;
x16709 += 2;
x16701 += 4;
x16702 += 4;
x16703 += 1;

}
x16694 += 2048;
x16695 += 2048;

}
// resize to WrappedArray(-1, 1, 1)
float* x16770 = (float*)myMalloc(131072 * sizeof(float));;
int32_t x16771 = 0;
int32_t x16772 = 0;
int32_t x16773 = 0;
for(int x16774=0; x16774 < 64; x16774++) {
int32_t x16775 = x16772;
int32_t x16776 = x16773;
int32_t x16777 = x16771;
int32_t x16778 = x16777;
int32_t x16779 = x16775;
int32_t x16780 = x16776;
for(int x16781=0; x16781 < 512; x16781++) {
int32_t x16782 = x16779;
int32_t x16783 = x16780;
int32_t x16784 = x16778;
int32_t x16785 = x16784;
int32_t x16786 = x16782;
int32_t x16787 = x16783;
int32_t x16788 = x16786;
int32_t x16789 = x16787;
int32_t x16790 = x16785;
int32_t x16791 = x16790;
int32_t x16792 = x16788;
int32_t x16793 = x16789;
int32_t x16794 = x16791;
int32_t x16795 = x16792;
float x16796 = x16693[x16795];
int32_t x16797 = x16793;
float x16798 = x168[x16797];
float x16799 = x16796 + x16798;
x16770[x16794] = x16799;
x16791 += 1;
x16792 += 1;
int32_t x16803 = x16791;
int32_t x16804 = x16792;
float x16805 = x16693[x16804];
float x16806 = x168[x16797];
float x16807 = x16805 + x16806;
x16770[x16803] = x16807;
x16791 += 1;
x16792 += 1;
x16785 += 2;
x16786 += 2;
int32_t x16813 = x16786;
int32_t x16814 = x16785;
int32_t x16815 = x16814;
int32_t x16816 = x16813;
int32_t x16817 = x16789;
int32_t x16818 = x16815;
int32_t x16819 = x16816;
float x16820 = x16693[x16819];
int32_t x16821 = x16817;
float x16822 = x168[x16821];
float x16823 = x16820 + x16822;
x16770[x16818] = x16823;
x16815 += 1;
x16816 += 1;
int32_t x16827 = x16815;
int32_t x16828 = x16816;
float x16829 = x16693[x16828];
float x16830 = x168[x16821];
float x16831 = x16829 + x16830;
x16770[x16827] = x16831;
x16815 += 1;
x16816 += 1;
x16785 += 2;
x16786 += 2;
x16778 += 4;
x16779 += 4;
x16780 += 1;

}
x16771 += 2048;
x16772 += 2048;

}
float* x16846 = (float*)myMalloc(131072 * sizeof(float));;
for(int x16847=0; x16847 < 131072; x16847++) {
float x16848 = x16770[x16847];
bool x16849 = x16848 < 0.0f;
if (x16849) {
x16846[x16847] = 0.0f;
} else {
float x16852 = x16770[x16847];
x16846[x16847] = x16852;
}

}
float* x16858 = (float*)myMalloc(524288 * sizeof(float));;
float* x16859 = (float*)myMalloc(131072 * sizeof(float));;
for(int x16860=0; x16860 < 64; x16860++) {
int32_t x16861 = x16860 * 2048;
float* x16862 = x16846+x16861;
int32_t x16863 = x16860 * 8192;
float* x16864 = x16858+x16863;
float* x16865 = x16859+x16861;
for(int x16866=0; x16866 < 512; x16866++) {
int32_t x16867 = x16866 / 1;
int32_t x16871 = x16867 * 2;
int32_t x16872 = x16871 * 2;
int32_t x16868 = x16866 % 1;
int32_t x16869 = x16868 / 1;
int32_t x16873 = x16869 * 2;
int32_t x16874 = x16873 * 2;
int32_t x16875 = x16872 + x16874;
int32_t x16870 = x16868 % 1;
int32_t x16876 = x16870 * 2;
int32_t x16877 = x16876 * 2;
int32_t x16878 = x16875 + x16877;
float* x16879 = x16865+x16878;
float* x16880 = x16862+x16872;
for(int x16881=0; x16881 < 2; x16881++) {
int32_t x16883 = x16881 * 2;
float* x16884 = x16879+x16883;
int32_t x16882 = x16881 + x16869;
int32_t x16885 = x16882 * 2;
int32_t x16886 = x16885 + x16870;
float* x16887 = x16880+x16886;
memcpy(x16884, x16887, 4 * 2);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,4,512,1,x32,512,x16865,4,1,x16864,4);

}
// resize to WrappedArray(-1, 1, 1)
float* x16897 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x16898 = 0;
int32_t x16899 = 0;
int32_t x16900 = 0;
for(int x16901=0; x16901 < 64; x16901++) {
int32_t x16902 = x16899;
int32_t x16903 = x16900;
int32_t x16904 = x16898;
int32_t x16905 = x16904;
int32_t x16906 = x16902;
int32_t x16907 = x16903;
for(int x16908=0; x16908 < 2048; x16908++) {
int32_t x16909 = x16906;
int32_t x16910 = x16907;
int32_t x16911 = x16905;
int32_t x16912 = x16911;
int32_t x16913 = x16909;
int32_t x16914 = x16910;
int32_t x16915 = x16913;
int32_t x16916 = x16914;
int32_t x16917 = x16912;
int32_t x16918 = x16917;
int32_t x16919 = x16915;
int32_t x16920 = x16916;
int32_t x16921 = x16918;
int32_t x16922 = x16919;
float x16923 = x16858[x16922];
int32_t x16924 = x16920;
float x16925 = x259[x16924];
float x16926 = x16923 - x16925;
x16897[x16921] = x16926;
x16918 += 1;
x16919 += 1;
int32_t x16930 = x16918;
int32_t x16931 = x16919;
float x16932 = x16858[x16931];
float x16933 = x259[x16924];
float x16934 = x16932 - x16933;
x16897[x16930] = x16934;
x16918 += 1;
x16919 += 1;
x16912 += 2;
x16913 += 2;
int32_t x16940 = x16913;
int32_t x16941 = x16912;
int32_t x16942 = x16941;
int32_t x16943 = x16940;
int32_t x16944 = x16916;
int32_t x16945 = x16942;
int32_t x16946 = x16943;
float x16947 = x16858[x16946];
int32_t x16948 = x16944;
float x16949 = x259[x16948];
float x16950 = x16947 - x16949;
x16897[x16945] = x16950;
x16942 += 1;
x16943 += 1;
int32_t x16954 = x16942;
int32_t x16955 = x16943;
float x16956 = x16858[x16955];
float x16957 = x259[x16948];
float x16958 = x16956 - x16957;
x16897[x16954] = x16958;
x16942 += 1;
x16943 += 1;
x16912 += 2;
x16913 += 2;
x16905 += 4;
x16906 += 4;
x16907 += 1;

}
x16898 += 8192;
x16899 += 8192;

}
float* x16973 = (float*)myMalloc(2048 * sizeof(float));;
for(int x16974=0; x16974 < 2048; x16974++) {
float x16975 = x122[x16974];
float x16976 = x16975 + 1.0E-5f;
x16973[x16974] = x16976;

}
float* x16980 = (float*)myMalloc(2048 * sizeof(float));;
for(int x16981=0; x16981 < 2048; x16981++) {
float x16982 = x16973[x16981];
double x16983 = (double)x16982;
double x16984 = sqrt(x16983);
float x16985 = (float)x16984;
x16980[x16981] = x16985;

}
// resize to WrappedArray(-1, 1, 1)
float* x16990 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x16991 = 0;
int32_t x16992 = 0;
int32_t x16993 = 0;
for(int x16994=0; x16994 < 64; x16994++) {
int32_t x16995 = x16992;
int32_t x16996 = x16993;
int32_t x16997 = x16991;
int32_t x16998 = x16997;
int32_t x16999 = x16995;
int32_t x17000 = x16996;
for(int x17001=0; x17001 < 2048; x17001++) {
int32_t x17002 = x16999;
int32_t x17003 = x17000;
int32_t x17004 = x16998;
int32_t x17005 = x17004;
int32_t x17006 = x17002;
int32_t x17007 = x17003;
int32_t x17008 = x17006;
int32_t x17009 = x17007;
int32_t x17010 = x17005;
int32_t x17011 = x17010;
int32_t x17012 = x17008;
int32_t x17013 = x17009;
int32_t x17014 = x17011;
int32_t x17015 = x17012;
float x17016 = x16897[x17015];
int32_t x17017 = x17013;
float x17018 = x16980[x17017];
float x17019 = x17016 / x17018;
x16990[x17014] = x17019;
x17011 += 1;
x17012 += 1;
int32_t x17023 = x17011;
int32_t x17024 = x17012;
float x17025 = x16897[x17024];
float x17026 = x16980[x17017];
float x17027 = x17025 / x17026;
x16990[x17023] = x17027;
x17011 += 1;
x17012 += 1;
x17005 += 2;
x17006 += 2;
int32_t x17033 = x17006;
int32_t x17034 = x17005;
int32_t x17035 = x17034;
int32_t x17036 = x17033;
int32_t x17037 = x17009;
int32_t x17038 = x17035;
int32_t x17039 = x17036;
float x17040 = x16897[x17039];
int32_t x17041 = x17037;
float x17042 = x16980[x17041];
float x17043 = x17040 / x17042;
x16990[x17038] = x17043;
x17035 += 1;
x17036 += 1;
int32_t x17047 = x17035;
int32_t x17048 = x17036;
float x17049 = x16897[x17048];
float x17050 = x16980[x17041];
float x17051 = x17049 / x17050;
x16990[x17047] = x17051;
x17035 += 1;
x17036 += 1;
x17005 += 2;
x17006 += 2;
x16998 += 4;
x16999 += 4;
x17000 += 1;

}
x16991 += 8192;
x16992 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x17067 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x17068 = 0;
int32_t x17069 = 0;
int32_t x17070 = 0;
for(int x17071=0; x17071 < 64; x17071++) {
int32_t x17072 = x17069;
int32_t x17073 = x17070;
int32_t x17074 = x17068;
int32_t x17075 = x17074;
int32_t x17076 = x17072;
int32_t x17077 = x17073;
for(int x17078=0; x17078 < 2048; x17078++) {
int32_t x17079 = x17076;
int32_t x17080 = x17077;
int32_t x17081 = x17075;
int32_t x17082 = x17081;
int32_t x17083 = x17079;
int32_t x17084 = x17080;
int32_t x17085 = x17083;
int32_t x17086 = x17084;
int32_t x17087 = x17082;
int32_t x17088 = x17087;
int32_t x17089 = x17085;
int32_t x17090 = x17086;
int32_t x17091 = x17088;
int32_t x17092 = x17089;
float x17093 = x16990[x17092];
int32_t x17094 = x17090;
float x17095 = x102[x17094];
float x17096 = x17093 * x17095;
x17067[x17091] = x17096;
x17088 += 1;
x17089 += 1;
int32_t x17100 = x17088;
int32_t x17101 = x17089;
float x17102 = x16990[x17101];
float x17103 = x102[x17094];
float x17104 = x17102 * x17103;
x17067[x17100] = x17104;
x17088 += 1;
x17089 += 1;
x17082 += 2;
x17083 += 2;
int32_t x17110 = x17083;
int32_t x17111 = x17082;
int32_t x17112 = x17111;
int32_t x17113 = x17110;
int32_t x17114 = x17086;
int32_t x17115 = x17112;
int32_t x17116 = x17113;
float x17117 = x16990[x17116];
int32_t x17118 = x17114;
float x17119 = x102[x17118];
float x17120 = x17117 * x17119;
x17067[x17115] = x17120;
x17112 += 1;
x17113 += 1;
int32_t x17124 = x17112;
int32_t x17125 = x17113;
float x17126 = x16990[x17125];
float x17127 = x102[x17118];
float x17128 = x17126 * x17127;
x17067[x17124] = x17128;
x17112 += 1;
x17113 += 1;
x17082 += 2;
x17083 += 2;
x17075 += 4;
x17076 += 4;
x17077 += 1;

}
x17068 += 8192;
x17069 += 8192;

}
// resize to WrappedArray(-1, 1, 1)
float* x17144 = (float*)myMalloc(524288 * sizeof(float));;
int32_t x17145 = 0;
int32_t x17146 = 0;
int32_t x17147 = 0;
for(int x17148=0; x17148 < 64; x17148++) {
int32_t x17149 = x17146;
int32_t x17150 = x17147;
int32_t x17151 = x17145;
int32_t x17152 = x17151;
int32_t x17153 = x17149;
int32_t x17154 = x17150;
for(int x17155=0; x17155 < 2048; x17155++) {
int32_t x17156 = x17153;
int32_t x17157 = x17154;
int32_t x17158 = x17152;
int32_t x17159 = x17158;
int32_t x17160 = x17156;
int32_t x17161 = x17157;
int32_t x17162 = x17160;
int32_t x17163 = x17161;
int32_t x17164 = x17159;
int32_t x17165 = x17164;
int32_t x17166 = x17162;
int32_t x17167 = x17163;
int32_t x17168 = x17165;
int32_t x17169 = x17166;
float x17170 = x17067[x17169];
int32_t x17171 = x17167;
float x17172 = x180[x17171];
float x17173 = x17170 + x17172;
x17144[x17168] = x17173;
x17165 += 1;
x17166 += 1;
int32_t x17177 = x17165;
int32_t x17178 = x17166;
float x17179 = x17067[x17178];
float x17180 = x180[x17171];
float x17181 = x17179 + x17180;
x17144[x17177] = x17181;
x17165 += 1;
x17166 += 1;
x17159 += 2;
x17160 += 2;
int32_t x17187 = x17160;
int32_t x17188 = x17159;
int32_t x17189 = x17188;
int32_t x17190 = x17187;
int32_t x17191 = x17163;
int32_t x17192 = x17189;
int32_t x17193 = x17190;
float x17194 = x17067[x17193];
int32_t x17195 = x17191;
float x17196 = x180[x17195];
float x17197 = x17194 + x17196;
x17144[x17192] = x17197;
x17189 += 1;
x17190 += 1;
int32_t x17201 = x17189;
int32_t x17202 = x17190;
float x17203 = x17067[x17202];
float x17204 = x180[x17195];
float x17205 = x17203 + x17204;
x17144[x17201] = x17205;
x17189 += 1;
x17190 += 1;
x17159 += 2;
x17160 += 2;
x17152 += 4;
x17153 += 4;
x17154 += 1;

}
x17145 += 8192;
x17146 += 8192;

}
int32_t x17220 = 0;
int32_t x17221 = 0;
int32_t x17222 = 0;
for(int x17223=0; x17223 < 64; x17223++) {
int32_t x17224 = x17221;
int32_t x17225 = x17222;
int32_t x17226 = x17220;
int32_t x17227 = x17226;
int32_t x17228 = x17224;
int32_t x17229 = x17225;
for(int x17230=0; x17230 < 2048; x17230++) {
int32_t x17231 = x17228;
int32_t x17232 = x17229;
int32_t x17233 = x17227;
int32_t x17234 = x17233;
int32_t x17235 = x17231;
int32_t x17236 = x17232;
int32_t x17237 = x17235;
int32_t x17238 = x17236;
int32_t x17239 = x17234;
int32_t x17240 = x17239;
int32_t x17241 = x17237;
int32_t x17242 = x17238;
int32_t x17243 = x17241;
float x17244 = x17144[x17243];
int32_t x17245 = x17242;
float x17246 = x16052[x17245];
float x17247 = x17244 + x17246;
x17144[x17243] = x17247;
x17240 += 1;
x17241 += 1;
x17242 += 1;
int32_t x17252 = x17241;
float x17253 = x17144[x17252];
int32_t x17254 = x17242;
float x17255 = x16052[x17254];
float x17256 = x17253 + x17255;
x17144[x17252] = x17256;
x17240 += 1;
x17241 += 1;
x17242 += 1;
x17234 += 2;
x17235 += 2;
x17236 += 2;
int32_t x17264 = x17235;
int32_t x17265 = x17236;
int32_t x17266 = x17234;
int32_t x17267 = x17266;
int32_t x17268 = x17264;
int32_t x17269 = x17265;
int32_t x17270 = x17268;
float x17271 = x17144[x17270];
int32_t x17272 = x17269;
float x17273 = x16052[x17272];
float x17274 = x17271 + x17273;
x17144[x17270] = x17274;
x17267 += 1;
x17268 += 1;
x17269 += 1;
int32_t x17279 = x17268;
float x17280 = x17144[x17279];
int32_t x17281 = x17269;
float x17282 = x16052[x17281];
float x17283 = x17280 + x17282;
x17144[x17279] = x17283;
x17267 += 1;
x17268 += 1;
x17269 += 1;
x17234 += 2;
x17235 += 2;
x17236 += 2;
x17227 += 4;
x17228 += 4;
x17229 += 4;

}
x17220 += 8192;
x17221 += 8192;
x17222 += 8192;

}
float* x17301 = (float*)myMalloc(524288 * sizeof(float));;
for(int x17302=0; x17302 < 524288; x17302++) {
float x17303 = x17144[x17302];
bool x17304 = x17303 < 0.0f;
if (x17304) {
x17301[x17302] = 0.0f;
} else {
float x17307 = x17144[x17302];
x17301[x17302] = x17307;
}

}
float* x17313 = (float*)myMalloc(131072 * sizeof(float));;
for(int x17314=0; x17314 < 64; x17314++) {
int32_t x17315 = x17314 * 8192;
float* x17316 = x17301+x17315;
int32_t x17317 = x17314 * 2048;
float* x17318 = x17313+x17317;
for(int x17319=0; x17319 < 2048; x17319++) {
float x17321 = 0.0f;
int32_t x17320 = x17319 * 4;
float x17322 = x17316[x17320];
x17321 += x17322;
int32_t x17324 = x17320 + 1;
float x17325 = x17316[x17324];
x17321 += x17325;
int32_t x17327 = x17320 + 2;
float x17328 = x17316[x17327];
x17321 += x17328;
int32_t x17330 = x17327 + 1;
float x17331 = x17316[x17330];
x17321 += x17331;
float x17333 = x17321;
float x17334 = x17333 / 4.0f;
x17318[x17319] = x17334;

}

}
// resize to WrappedArray(64, 2048)
// gemm: WrappedArray(64, 2048), Vector(10, 2048)
float* x17342 = (float*)myMalloc(640 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 64,10,2048,1.0,x17313,2048,x226,2048,0,x17342,10);
int32_t x17344 = 0;
int32_t x17345 = 0;
int32_t x17346 = 0;
for(int x17347=0; x17347 < 64; x17347++) {
int32_t x17348 = x17345;
int32_t x17349 = x17346;
int32_t x17350 = x17344;
int32_t x17351 = x17350;
int32_t x17352 = x17348;
int32_t x17353 = x17349;
for(int x17354=0; x17354 < 10; x17354++) {
int32_t x17355 = x17352;
float x17356 = x17342[x17355];
int32_t x17357 = x17353;
float x17358 = x47[x17357];
float x17359 = x17356 + x17358;
x17342[x17355] = x17359;
x17351 += 1;
x17352 += 1;
x17353 += 1;

}
x17344 += 10;
x17345 += 10;

}
printf("output (size 64 x 10)\n");
float x17371 = 0.0f;
for(int x17373=0; x17373 < 640; x17373++) {
float x17374 = x17371;
float x17375 = x17342[x17373];
float x17376 = fabs(x17375);
float x17377 = fabs(x17374);
bool x17378 = x17376 > x17377;
float x17379;
if (x17378) {
x17379 = x17375;
} else {
x17379 = x17374;
}
x17371 = x17379;

}
float x17383 = x17371;
printf("Max Abs: %.5f || ",x17383);
for(int x17385=0; x17385 < 10; x17385++) {
float x17386 = x17342[x17385];
printf("%.5f ",x17386);

}
printf("\n");
assert(false && "stop");

}
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

