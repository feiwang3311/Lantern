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
int32_t x273 = open("../../cifar10_data/cifar-10-batches-bin/data_batch_1.bin",0);
int64_t x274 = fsize(x273);
int64_t x276 = x274 / 3073LL;
int32_t x277 = (int32_t)x276;
int32_t x278 = x277 * 3072;
float* x279 = (float*)myMalloc(x278 * sizeof(float));;
int* x280 = (int32_t*)myMalloc(x277 * sizeof(int32_t));;
char* x275 = (char*)mmap(0, x274, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x273, 0);
for(int x282=0; x282 < x277; x282++) {
int32_t x283 = x282 * 3073;
char x284 = x275[x283];
int32_t x285 = (int32_t)(unsigned char)x284;
x280[x282] = x285;
int32_t x291 = x283 + 1;
int32_t x289 = x282 * 3072;
for(int x288=0; x288 < 3072; x288++) {
int32_t x292 = x291 + x288;
char x293 = x275[x292];
int32_t x290 = x289 + x288;
float x294 = (float)(unsigned char)x293;
float x295 = x294 / 255.0f;
x279[x290] = x295;

}

}
int32_t x301 = x277 / 64;
int32_t x330 = 31 / 1;
int32_t x331 = x330 + 1;
int32_t x335 = 4096 * x331;
int32_t x336 = x335 * x331;
int32_t x332 = x331 * x331;
int32_t x340 = 1728 * x332;
int32_t x333 = 64 * x332;
int32_t x338 = 27 * x332;
int32_t x3 = open("/u/data/u99/wang603/TiarkMlEnv/Lantern/src/out/PLDI19evaluation/resnet50/resnet50.onnx.bin",0);
int64_t x4 = fsize(x3);
float* x5 = (float*)mmap(0, x4, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x3, 0);
float* x152 = x5+0;
bool x428 = -7 > 0;
float* x40 = x5+1856;
bool x448 = 64 > 1;
bool x452 = 1 > 1;
float* x110 = x5+1920;
bool x474 = 1 > 0;
bool x476;
if (x474) {
bool x475 = -1 > 0;
x476 = x475;
} else {
x476 = false;
}
bool x477;
if (x476) {
x477 = x474;
} else {
x477 = false;
}
bool x478;
if (x477) {
x478 = x428;
} else {
x478 = false;
}
bool x520 = -7 > 1;
bool x542 = -1 > 1;
float* x206 = x5+1728;
float* x251 = x5+1792;
bool x716 = 1 >= 2;
bool x718;
if (x716) {
bool x717 = -7 >= 2;
x718 = x717;
} else {
x718 = false;
}
int32_t x725 = -9 / 2;
int32_t x726 = x725 + 1;
int32_t x723 = -1 / 2;
int32_t x724 = x723 + 1;
int32_t x729 = -1 * x724;
int32_t x730 = x729 * x726;
int32_t x727 = x724 * x726;
int32_t x728 = -1 * x727;
int32_t x828 = x725 / 1;
int32_t x829 = x828 + 1;
int32_t x826 = x723 / 1;
int32_t x827 = x826 + 1;
int32_t x832 = 64 * x827;
int32_t x833 = x832 * x829;
int32_t x830 = x827 * x829;
int32_t x835 = -1 * x830;
int32_t x831 = 64 * x830;
float* x233 = x5+1984;
float* x114 = x5+6208;
float* x51 = x5+6272;
float* x26 = x5+6080;
float* x53 = x5+6144;
int32_t x1151 = -8 / 1;
int32_t x1152 = x1151 + 1;
int32_t x1149 = 0 / 1;
int32_t x1150 = x1149 + 1;
int32_t x1155 = 64 * x1150;
int32_t x1156 = x1155 * x1152;
int32_t x1153 = x1150 * x1152;
int32_t x1158 = -9 * x1153;
int32_t x1154 = 64 * x1153;
float* x90 = x5+6336;
float* x105 = x5+43328;
float* x158 = x5+43392;
float* x164 = x5+43200;
float* x49 = x5+43264;
int32_t x1517 = 256 * x1150;
int32_t x1518 = x1517 * x1152;
int32_t x1520 = -1 * x1153;
int32_t x1516 = 256 * x1153;
float* x32 = x5+43456;
float* x71 = x5+60352;
float* x36 = x5+60608;
float* x199 = x5+59840;
float* x126 = x5+60096;
int32_t x1822 = 256 * x827;
int32_t x1823 = x1822 * x829;
int32_t x1821 = 256 * x830;
float* x162 = x5+60864;
float* x264 = x5+77760;
float* x243 = x5+78016;
float* x76 = x5+77248;
float* x203 = x5+77504;
float* x171 = x5+78272;
float* x10 = x5+94784;
float* x102 = x5+94848;
float* x142 = x5+94656;
float* x60 = x5+94720;
float* x83 = x5+94912;
float* x44 = x5+131904;
float* x244 = x5+131968;
float* x208 = x5+131776;
float* x153 = x5+131840;
float* x130 = x5+132032;
float* x91 = x5+148928;
float* x166 = x5+149184;
float* x58 = x5+148416;
float* x7 = x5+148672;
float* x150 = x5+149440;
float* x257 = x5+165952;
float* x187 = x5+166016;
float* x81 = x5+165824;
float* x24 = x5+165888;
float* x73 = x5+166080;
float* x179 = x5+203072;
float* x118 = x5+203136;
float* x72 = x5+202944;
float* x135 = x5+203008;
float* x87 = x5+203200;
float* x184 = x5+220096;
float* x133 = x5+220352;
float* x37 = x5+219584;
float* x247 = x5+219840;
int32_t x4299 = 128 * x1150;
int32_t x4300 = x4299 * x1152;
int32_t x4298 = 128 * x1153;
float* x11 = x5+220608;
float* x204 = x5+253632;
float* x134 = x5+253760;
float* x84 = x5+253376;
float* x172 = x5+253504;
int32_t x4616 = -8 / 2;
int32_t x4617 = x4616 + 1;
int32_t x4614 = 0 / 2;
int32_t x4615 = x4614 + 1;
int32_t x4620 = 128 * x4615;
int32_t x4621 = x4620 * x4617;
int32_t x4618 = x4615 * x4617;
int32_t x4623 = -9 * x4618;
int32_t x4619 = 128 * x4618;
float* x27 = x5+253888;
float* x128 = x5+401600;
float* x43 = x5+401728;
float* x252 = x5+401344;
float* x190 = x5+401472;
int32_t x4966 = 512 * x1150;
int32_t x4967 = x4966 * x1152;
int32_t x4965 = 512 * x1153;
float* x106 = x5+401856;
float* x149 = x5+468416;
float* x101 = x5+468928;
float* x145 = x5+467392;
float* x210 = x5+467904;
int32_t x5270 = 512 * x4615;
int32_t x5271 = x5270 * x4617;
int32_t x5273 = -1 * x4618;
int32_t x5269 = 512 * x4618;
float* x258 = x5+469440;
float* x42 = x5+601536;
float* x23 = x5+602048;
float* x207 = x5+600512;
float* x119 = x5+601024;
float* x256 = x5+602560;
float* x100 = x5+668352;
float* x177 = x5+668480;
float* x222 = x5+668096;
float* x17 = x5+668224;
float* x235 = x5+668608;
float* x35 = x5+816320;
float* x225 = x5+816448;
float* x8 = x5+816064;
float* x95 = x5+816192;
float* x111 = x5+816576;
float* x147 = x5+883136;
float* x88 = x5+883648;
float* x52 = x5+882112;
float* x246 = x5+882624;
float* x196 = x5+884160;
float* x112 = x5+949952;
float* x9 = x5+950080;
float* x45 = x5+949696;
float* x170 = x5+949824;
float* x191 = x5+950208;
float* x217 = x5+1097920;
float* x266 = x5+1098048;
float* x127 = x5+1097664;
float* x61 = x5+1097792;
float* x41 = x5+1098176;
float* x25 = x5+1164736;
float* x223 = x5+1165248;
float* x167 = x5+1163712;
float* x82 = x5+1164224;
float* x132 = x5+1165760;
float* x236 = x5+1231552;
float* x261 = x5+1231680;
float* x39 = x5+1231296;
float* x242 = x5+1231424;
float* x165 = x5+1231808;
float* x268 = x5+1379520;
float* x148 = x5+1379648;
float* x79 = x5+1379264;
float* x38 = x5+1379392;
float* x55 = x5+1379776;
float* x19 = x5+1446336;
float* x234 = x5+1446848;
float* x156 = x5+1445312;
float* x54 = x5+1445824;
float* x180 = x5+1447360;
float* x131 = x5+1578944;
float* x198 = x5+1579200;
float* x270 = x5+1578432;
float* x21 = x5+1578688;
int32_t x9113 = 256 * x4615;
int32_t x9114 = x9113 * x4617;
int32_t x9112 = 256 * x4618;
float* x175 = x5+1579456;
float* x229 = x5+2169792;
float* x99 = x5+2170048;
float* x108 = x5+2169280;
float* x16 = x5+2169536;
int32_t x9456 = 1024 * x1150;
int32_t x9457 = x9456 * x1152;
int32_t x9455 = 1024 * x1153;
float* x269 = x5+2170304;
float* x216 = x5+2434496;
float* x267 = x5+2435520;
float* x18 = x5+2432448;
float* x117 = x5+2433472;
int32_t x9760 = 1024 * x4615;
int32_t x9761 = x9760 * x4617;
int32_t x9759 = 1024 * x4618;
float* x75 = x5+2436544;
float* x86 = x5+2962880;
float* x211 = x5+2963904;
float* x29 = x5+2960832;
float* x220 = x5+2961856;
float* x13 = x5+2964928;
float* x259 = x5+3227584;
float* x157 = x5+3227840;
float* x30 = x5+3227072;
float* x219 = x5+3227328;
float* x31 = x5+3228096;
float* x200 = x5+3818432;
float* x237 = x5+3818688;
float* x271 = x5+3817920;
float* x96 = x5+3818176;
float* x56 = x5+3818944;
float* x182 = x5+4083136;
float* x143 = x5+4084160;
float* x20 = x5+4081088;
float* x232 = x5+4082112;
float* x218 = x5+4085184;
float* x178 = x5+4347840;
float* x174 = x5+4348096;
float* x129 = x5+4347328;
float* x197 = x5+4347584;
float* x14 = x5+4348352;
float* x124 = x5+4938688;
float* x63 = x5+4938944;
float* x228 = x5+4938176;
float* x192 = x5+4938432;
float* x116 = x5+4939200;
float* x140 = x5+5203392;
float* x188 = x5+5204416;
float* x263 = x5+5201344;
float* x57 = x5+5202368;
float* x6 = x5+5205440;
float* x163 = x5+5468096;
float* x98 = x5+5468352;
float* x92 = x5+5467584;
float* x241 = x5+5467840;
float* x249 = x5+5468608;
float* x186 = x5+6058944;
float* x230 = x5+6059200;
float* x74 = x5+6058432;
float* x136 = x5+6058688;
float* x89 = x5+6059456;
float* x231 = x5+6323648;
float* x161 = x5+6324672;
float* x238 = x5+6321600;
float* x146 = x5+6322624;
float* x22 = x5+6325696;
float* x254 = x5+6588352;
float* x69 = x5+6588608;
float* x77 = x5+6587840;
float* x185 = x5+6588096;
float* x262 = x5+6588864;
float* x250 = x5+7179200;
float* x104 = x5+7179456;
float* x168 = x5+7178688;
float* x109 = x5+7178944;
float* x221 = x5+7179712;
float* x209 = x5+7443904;
float* x272 = x5+7444928;
float* x59 = x5+7441856;
float* x120 = x5+7442880;
float* x151 = x5+7445952;
float* x80 = x5+7708608;
float* x176 = x5+7708864;
float* x85 = x5+7708096;
float* x253 = x5+7708352;
float* x226 = x5+7709120;
float* x70 = x5+8299456;
float* x240 = x5+8299712;
float* x141 = x5+8298944;
float* x189 = x5+8299200;
float* x97 = x5+8299968;
float* x122 = x5+8564160;
float* x183 = x5+8565184;
float* x248 = x5+8562112;
float* x93 = x5+8563136;
float* x139 = x5+8566208;
float* x67 = x5+9091520;
float* x121 = x5+9092032;
float* x201 = x5+9090496;
float* x224 = x5+9091008;
float* x34 = x5+9092544;
float* x113 = x5+11452864;
float* x50 = x5+11453376;
float* x205 = x5+11451840;
float* x159 = x5+11452352;
int32_t x16036 = 2048 * x1150;
int32_t x16037 = x16036 * x1152;
int32_t x16035 = 2048 * x1153;
float* x212 = x5+11453888;
float* x115 = x5+12506560;
float* x193 = x5+12508608;
float* x239 = x5+12502464;
float* x62 = x5+12504512;
int32_t x16340 = 2048 * x4615;
int32_t x16341 = x16340 * x4617;
int32_t x16339 = 2048 * x4618;
float* x214 = x5+12510656;
float* x64 = x5+14611904;
float* x125 = x5+14613952;
float* x173 = x5+14607808;
float* x107 = x5+14609856;
float* x215 = x5+14616000;
float* x154 = x5+15665600;
float* x65 = x5+15666112;
float* x46 = x5+15664576;
float* x137 = x5+15665088;
float* x155 = x5+15666624;
float* x138 = x5+18026944;
float* x195 = x5+18027456;
float* x160 = x5+18025920;
float* x66 = x5+18026432;
float* x47 = x5+18027968;
float* x68 = x5+19080640;
float* x245 = x5+19082688;
float* x94 = x5+19076544;
float* x144 = x5+19078592;
float* x265 = x5+19084736;
float* x213 = x5+20134336;
float* x255 = x5+20134848;
float* x15 = x5+20133312;
float* x78 = x5+20133824;
float* x28 = x5+20135360;
float* x12 = x5+22495680;
float* x202 = x5+22496192;
float* x194 = x5+22494656;
float* x169 = x5+22495168;
float* x33 = x5+22496704;
float* x260 = x5+23549376;
float* x123 = x5+23551424;
float* x103 = x5+23545280;
float* x181 = x5+23547328;
int32_t x18827 = -9 / 1;
int32_t x18828 = x18827 + 1;
int32_t x18825 = -1 / 1;
int32_t x18826 = x18825 + 1;
int32_t x18831 = -1 * x18826;
int32_t x18832 = x18831 * x18828;
int32_t x18829 = x18826 * x18828;
int32_t x18830 = -1 * x18829;
float* x227 = x5+23553472;
float* x48 = x5+23573952;
bool x18896 = 10 > 1;
for(int x303=0; x303 < x301; x303++) {
int32_t x304 = x303 * 64;
int32_t x305 = x304 * 3072;
float* x306 = x279+x305;
int* x307 = x280+x304;
printf("input (size Const(64) x Const(3) x Const(32) x Const(32))\n");
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
float* x337 = (float*)myMalloc(x336 * sizeof(float));;
float* x341 = (float*)myMalloc(x340 * sizeof(float));;
for(int x343=0; x343 < 64; x343++) {
int32_t x344 = x343 * 3072;
float* x345 = x306+x344;
int32_t x346 = x343 * x333;
float* x347 = x337+x346;
int32_t x348 = x343 * x338;
float* x349 = x341+x348;
for(int x351=0; x351 < 27; x351++) {
int32_t x352 = x351 / 9;
int32_t x356 = x352 * 3;
int32_t x357 = x356 * 3;
int32_t x358 = x357 * x331;
int32_t x359 = x358 * x331;
int32_t x353 = x351 % 9;
int32_t x354 = x353 / 3;
int32_t x360 = x354 * 3;
int32_t x361 = x360 * x331;
int32_t x362 = x361 * x331;
int32_t x363 = x359 + x362;
int32_t x355 = x353 % 3;
int32_t x364 = x355 * x331;
int32_t x365 = x364 * x331;
int32_t x366 = x363 + x365;
float* x367 = x349+x366;
int32_t x368 = x352 * 32;
int32_t x369 = x368 * 32;
float* x370 = x345+x369;
int32_t x383 = 1 - x355;
bool x384 = x383 > 0;
int32_t x385;
if (x384) {
x385 = x383;
} else {
x385 = 0;
}
int32_t x386 = 3 - x355;
int32_t x387 = x386 - 1;
int32_t x388 = 1 - x387;
bool x389 = x388 > 0;
int32_t x390;
if (x389) {
x390 = x388;
} else {
x390 = 0;
}
int32_t x391 = x331 - x390;
int32_t x392 = x391 - x385;
bool x393 = x392 <= 0;
bool x397 = x385 > 0;
int32_t x382 = -1 + x355;
bool x410 = x390 > 0;
for(int x372=0; x372 < x331; x372++) {
int32_t x373 = x372 - 1;
int32_t x374 = x373 + x354;
bool x375 = x374 < 0;
bool x376 = x374 >= 32;
bool x377 = x375 || x376;
if (x377) {
int32_t x378 = x372 * x331;
float* x379 = x367+x378;
memset(x379, 0, 4 * x331);;
} else {
if (x393) {
int32_t x378 = x372 * x331;
float* x394 = x367+x378;
memset(x394, 0, 4 * x331);;
} else {
int32_t x378 = x372 * x331;
if (x397) {
float* x398 = x367+x378;
memset(x398, 0, 4 * x385);;
} else {
}
// may have segfault here
int32_t x403 = x378 + x385;
float* x404 = x367+x403;
int32_t x405 = x374 * 32;
int32_t x406 = x405 + x382;
int32_t x407 = x406 + x385;
float* x408 = x370+x407;
memcpy(x404, x408, 4 * x392);;
if (x410) {
int32_t x411 = x378 + x331;
int32_t x412 = x411 - x390;
float* x413 = x367+x412;
memset(x413, 0, 4 * x390);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x332,27,1,x152,27,x349,x332,1,x347,x332);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(64) x Const(64) x Sym(331) x Sym(331)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x434 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x435 = 0;
int32_t x436 = 0;
int32_t x437 = 0;
for(int x439=0; x439 < -7; x439++) {
int32_t x440 = x435;
int32_t x441 = x436;
float x442 = x337[x441];
int32_t x443 = x437;
float x444 = x40[x443];
float x445 = x442 - x444;
x434[x440] = x445;
x435 += 1;
if (x448) {
x436 += x333;
} else {
}
if (x452) {
x437 += -1;
} else {
}

}
float* x458 = (float*)myMalloc(64 * sizeof(float));;
for(int x459=0; x459 < 64; x459++) {
float x460 = x110[x459];
float x461 = x460 + 1.0E-5f;
x458[x459] = x461;

}
float* x465 = (float*)myMalloc(64 * sizeof(float));;
for(int x466=0; x466 < 64; x466++) {
float x467 = x458[x466];
double x468 = (double)x467;
double x469 = sqrt(x468);
float x470 = (float)x469;
x465[x466] = x470;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x484 = (float*)myMalloc(7 * sizeof(float));;
int32_t x485 = 0;
int32_t x486 = 0;
int32_t x487 = 0;
for(int x489=0; x489 < 1; x489++) {
int32_t x490 = x486;
int32_t x491 = x487;
int32_t x492 = x485;
int32_t x493 = x492;
int32_t x494 = x490;
int32_t x495 = x491;
for(int x497=0; x497 < -1; x497++) {
int32_t x498 = x494;
int32_t x499 = x495;
int32_t x500 = x493;
int32_t x501 = x500;
int32_t x502 = x498;
int32_t x503 = x499;
for(int x504=0; x504 < 1; x504++) {
int32_t x505 = x502;
int32_t x506 = x503;
int32_t x507 = x501;
int32_t x508 = x507;
int32_t x509 = x505;
int32_t x510 = x506;
for(int x511=0; x511 < -7; x511++) {
int32_t x512 = x508;
int32_t x513 = x509;
float x514 = x434[x513];
int32_t x515 = x510;
float x516 = x465[x515];
float x517 = x514 / x516;
x484[x512] = x517;
x508 += 1;
if (x520) {
x509 += 1;
} else {
}
if (x452) {
x510 += 1;
} else {
}

}
x501 += -7;
if (x452) {
x502 += -7;
} else {
}
if (x452) {
x503 += 1;
} else {
}

}
x493 += -7;
if (x452) {
x494 += -7;
} else {
}
if (x542) {
x495 += 1;
} else {
}

}
x485 += 7;
if (x452) {
x486 += -7;
} else {
}
if (x452) {
x487 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x561 = (float*)myMalloc(7 * sizeof(float));;
int32_t x562 = 0;
int32_t x563 = 0;
int32_t x564 = 0;
for(int x565=0; x565 < 1; x565++) {
int32_t x566 = x563;
int32_t x567 = x564;
int32_t x568 = x562;
int32_t x569 = x568;
int32_t x570 = x566;
int32_t x571 = x567;
for(int x572=0; x572 < -1; x572++) {
int32_t x573 = x570;
int32_t x574 = x571;
int32_t x575 = x569;
int32_t x576 = x575;
int32_t x577 = x573;
int32_t x578 = x574;
for(int x579=0; x579 < 1; x579++) {
int32_t x580 = x577;
int32_t x581 = x578;
int32_t x582 = x576;
int32_t x583 = x582;
int32_t x584 = x580;
int32_t x585 = x581;
for(int x586=0; x586 < -7; x586++) {
int32_t x587 = x583;
int32_t x588 = x584;
float x589 = x484[x588];
int32_t x590 = x585;
float x591 = x206[x590];
float x592 = x589 * x591;
x561[x587] = x592;
x583 += 1;
if (x520) {
x584 += 1;
} else {
}
if (x452) {
x585 += 1;
} else {
}

}
x576 += -7;
if (x452) {
x577 += -7;
} else {
}
if (x452) {
x578 += 1;
} else {
}

}
x569 += -7;
if (x542) {
x570 += -7;
} else {
}
if (x542) {
x571 += 1;
} else {
}

}
x562 += 7;
if (x452) {
x563 += 7;
} else {
}
if (x452) {
x564 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x634 = (float*)myMalloc(7 * sizeof(float));;
int32_t x635 = 0;
int32_t x636 = 0;
int32_t x637 = 0;
for(int x638=0; x638 < 1; x638++) {
int32_t x639 = x636;
int32_t x640 = x637;
int32_t x641 = x635;
int32_t x642 = x641;
int32_t x643 = x639;
int32_t x644 = x640;
for(int x645=0; x645 < -1; x645++) {
int32_t x646 = x643;
int32_t x647 = x644;
int32_t x648 = x642;
int32_t x649 = x648;
int32_t x650 = x646;
int32_t x651 = x647;
for(int x652=0; x652 < 1; x652++) {
int32_t x653 = x650;
int32_t x654 = x651;
int32_t x655 = x649;
int32_t x656 = x655;
int32_t x657 = x653;
int32_t x658 = x654;
for(int x659=0; x659 < -7; x659++) {
int32_t x660 = x656;
int32_t x661 = x657;
float x662 = x561[x661];
int32_t x663 = x658;
float x664 = x251[x663];
float x665 = x662 + x664;
x634[x660] = x665;
x656 += 1;
if (x520) {
x657 += 1;
} else {
}
if (x452) {
x658 += 1;
} else {
}

}
x649 += -7;
if (x452) {
x650 += -7;
} else {
}
if (x452) {
x651 += 1;
} else {
}

}
x642 += -7;
if (x542) {
x643 += -7;
} else {
}
if (x542) {
x644 += 1;
} else {
}

}
x635 += 7;
if (x452) {
x636 += 7;
} else {
}
if (x452) {
x637 += -1;
} else {
}

}
float* x703 = (float*)myMalloc(7 * sizeof(float));;
for(int x705=0; x705 < 7; x705++) {
float x706 = x634[x705];
bool x707 = x706 < 0.0f;
if (x707) {
x703[x705] = 0.0f;
} else {
float x710 = x634[x705];
x703[x705] = x710;
}

}
if (x718) {
} else {
assert(false && "Image too small for maxPool_k:  x Const(1) x Const(-1) x Const(1) x Const(-7)|(2,2)");
}
float* x731 = (float*)myMalloc(x730 * sizeof(float));;
for(int x733=0; x733 < x730; x733++) {
x731[x733] = -3.4028235E38f;

}
int* x737 = (int32_t*)myMalloc(x728 * sizeof(int32_t));;
for(int x738=0; x738 < 1; x738++) {
int32_t x739 = x738 * 7;
float* x740 = x703+x739;
int32_t x741 = x738 * x728;
float* x742 = x731+x741;
int* x743 = x737+x741;
int32_t x744 = 0;
int32_t x745 = 0;
for(int x746=0; x746 < -1; x746++) {
int32_t x747 = x744;
int32_t x748 = x747;
int32_t x749 = x745;
int32_t x750 = x749;
for(int x752=0; x752 < x724; x752++) {
int32_t x753 = x748;
int32_t x754 = x753;
int32_t x755 = x750;
int32_t x756 = x755;
for(int x758=0; x758 < x726; x758++) {
int32_t x759 = x756;
int32_t x760 = x759;
int32_t x761 = x760;
int32_t x762 = x761;
int32_t x763 = x762;
float x764 = x740[x763];
int32_t x765 = x754;
float x766 = x742[x765];
bool x767 = x764 > x766;
if (x767) {
float x768 = x740[x763];
x742[x765] = x768;
int32_t x770 = x763 + x739;
x743[x765] = x770;
} else {
}
x762 += 1;
int32_t x775 = x762;
float x776 = x740[x775];
float x777 = x742[x765];
bool x778 = x776 > x777;
if (x778) {
float x779 = x740[x775];
x742[x765] = x779;
int32_t x781 = x775 + x739;
x743[x765] = x781;
} else {
}
x762 += 1;
x760 += -7;
int32_t x787 = x760;
int32_t x788 = x787;
int32_t x789 = x788;
float x790 = x740[x789];
float x791 = x742[x765];
bool x792 = x790 > x791;
if (x792) {
float x793 = x740[x789];
x742[x765] = x793;
int32_t x795 = x789 + x739;
x743[x765] = x795;
} else {
}
x788 += 1;
int32_t x800 = x788;
float x801 = x740[x800];
float x802 = x742[x765];
bool x803 = x801 > x802;
if (x803) {
float x804 = x740[x800];
x742[x765] = x804;
int32_t x806 = x800 + x739;
x743[x765] = x806;
} else {
}
x788 += 1;
x760 += -7;
x754 += 1;
x756 += 2;

}
x748 += x726;
x750 += -14;

}
x744 += x727;
x745 += -7;

}

}
float* x834 = (float*)myMalloc(x833 * sizeof(float));;
float* x836 = (float*)myMalloc(x835 * sizeof(float));;
for(int x837=0; x837 < 1; x837++) {
int32_t x838 = x837 * x728;
float* x839 = x731+x838;
int32_t x840 = x837 * x831;
float* x841 = x834+x840;
int32_t x842 = x837 * x835;
float* x843 = x836+x842;
for(int x844=0; x844 < -1; x844++) {
int32_t x845 = x844 / 1;
int32_t x849 = x845 * x827;
int32_t x850 = x849 * x829;
int32_t x846 = x844 % 1;
int32_t x847 = x846 / 1;
int32_t x851 = x847 * x827;
int32_t x852 = x851 * x829;
int32_t x853 = x850 + x852;
int32_t x848 = x846 % 1;
int32_t x854 = x848 * x829;
int32_t x855 = x854 * x829;
int32_t x856 = x853 + x855;
float* x857 = x843+x856;
int32_t x858 = x845 * x724;
int32_t x859 = x858 * x726;
float* x860 = x839+x859;
for(int x862=0; x862 < x827; x862++) {
int32_t x864 = x862 * x829;
float* x865 = x857+x864;
int32_t x863 = x862 + x847;
int32_t x866 = x863 * x726;
int32_t x867 = x866 + x848;
float* x868 = x860+x867;
memcpy(x865, x868, 4 * x829);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x830,-1,1,x233,-1,x843,x830,1,x841,x830);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(64) x Sym(827) x Sym(829)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x881 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x882 = 0;
int32_t x883 = 0;
int32_t x884 = 0;
for(int x885=0; x885 < -7; x885++) {
int32_t x886 = x882;
int32_t x887 = x883;
float x888 = x834[x887];
int32_t x889 = x884;
float x890 = x114[x889];
float x891 = x888 - x890;
x881[x886] = x891;
x882 += 1;
if (x452) {
x883 += x831;
} else {
}
if (x452) {
x884 += -1;
} else {
}

}
float* x902 = (float*)myMalloc(64 * sizeof(float));;
for(int x903=0; x903 < 64; x903++) {
float x904 = x51[x903];
float x905 = x904 + 1.0E-5f;
x902[x903] = x905;

}
float* x909 = (float*)myMalloc(64 * sizeof(float));;
for(int x910=0; x910 < 64; x910++) {
float x911 = x902[x910];
double x912 = (double)x911;
double x913 = sqrt(x912);
float x914 = (float)x913;
x909[x910] = x914;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x922 = (float*)myMalloc(7 * sizeof(float));;
int32_t x923 = 0;
int32_t x924 = 0;
int32_t x925 = 0;
for(int x926=0; x926 < 1; x926++) {
int32_t x927 = x924;
int32_t x928 = x925;
int32_t x929 = x923;
int32_t x930 = x929;
int32_t x931 = x927;
int32_t x932 = x928;
for(int x933=0; x933 < -1; x933++) {
int32_t x934 = x931;
int32_t x935 = x932;
int32_t x936 = x930;
int32_t x937 = x936;
int32_t x938 = x934;
int32_t x939 = x935;
for(int x940=0; x940 < 1; x940++) {
int32_t x941 = x938;
int32_t x942 = x939;
int32_t x943 = x937;
int32_t x944 = x943;
int32_t x945 = x941;
int32_t x946 = x942;
for(int x947=0; x947 < -7; x947++) {
int32_t x948 = x944;
int32_t x949 = x945;
float x950 = x881[x949];
int32_t x951 = x946;
float x952 = x909[x951];
float x953 = x950 / x952;
x922[x948] = x953;
x944 += 1;
if (x520) {
x945 += 1;
} else {
}
if (x452) {
x946 += 1;
} else {
}

}
x937 += -7;
if (x452) {
x938 += -7;
} else {
}
if (x452) {
x939 += 1;
} else {
}

}
x930 += -7;
if (x452) {
x931 += -7;
} else {
}
if (x542) {
x932 += 1;
} else {
}

}
x923 += 7;
if (x452) {
x924 += -7;
} else {
}
if (x452) {
x925 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x995 = (float*)myMalloc(7 * sizeof(float));;
int32_t x996 = 0;
int32_t x997 = 0;
int32_t x998 = 0;
for(int x999=0; x999 < 1; x999++) {
int32_t x1000 = x997;
int32_t x1001 = x998;
int32_t x1002 = x996;
int32_t x1003 = x1002;
int32_t x1004 = x1000;
int32_t x1005 = x1001;
for(int x1006=0; x1006 < -1; x1006++) {
int32_t x1007 = x1004;
int32_t x1008 = x1005;
int32_t x1009 = x1003;
int32_t x1010 = x1009;
int32_t x1011 = x1007;
int32_t x1012 = x1008;
for(int x1013=0; x1013 < 1; x1013++) {
int32_t x1014 = x1011;
int32_t x1015 = x1012;
int32_t x1016 = x1010;
int32_t x1017 = x1016;
int32_t x1018 = x1014;
int32_t x1019 = x1015;
for(int x1020=0; x1020 < -7; x1020++) {
int32_t x1021 = x1017;
int32_t x1022 = x1018;
float x1023 = x922[x1022];
int32_t x1024 = x1019;
float x1025 = x26[x1024];
float x1026 = x1023 * x1025;
x995[x1021] = x1026;
x1017 += 1;
if (x520) {
x1018 += 1;
} else {
}
if (x452) {
x1019 += 1;
} else {
}

}
x1010 += -7;
if (x452) {
x1011 += -7;
} else {
}
if (x452) {
x1012 += 1;
} else {
}

}
x1003 += -7;
if (x542) {
x1004 += -7;
} else {
}
if (x542) {
x1005 += 1;
} else {
}

}
x996 += 7;
if (x452) {
x997 += 7;
} else {
}
if (x452) {
x998 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x1068 = (float*)myMalloc(7 * sizeof(float));;
int32_t x1069 = 0;
int32_t x1070 = 0;
int32_t x1071 = 0;
for(int x1072=0; x1072 < 1; x1072++) {
int32_t x1073 = x1070;
int32_t x1074 = x1071;
int32_t x1075 = x1069;
int32_t x1076 = x1075;
int32_t x1077 = x1073;
int32_t x1078 = x1074;
for(int x1079=0; x1079 < -1; x1079++) {
int32_t x1080 = x1077;
int32_t x1081 = x1078;
int32_t x1082 = x1076;
int32_t x1083 = x1082;
int32_t x1084 = x1080;
int32_t x1085 = x1081;
for(int x1086=0; x1086 < 1; x1086++) {
int32_t x1087 = x1084;
int32_t x1088 = x1085;
int32_t x1089 = x1083;
int32_t x1090 = x1089;
int32_t x1091 = x1087;
int32_t x1092 = x1088;
for(int x1093=0; x1093 < -7; x1093++) {
int32_t x1094 = x1090;
int32_t x1095 = x1091;
float x1096 = x995[x1095];
int32_t x1097 = x1092;
float x1098 = x53[x1097];
float x1099 = x1096 + x1098;
x1068[x1094] = x1099;
x1090 += 1;
if (x520) {
x1091 += 1;
} else {
}
if (x452) {
x1092 += 1;
} else {
}

}
x1083 += -7;
if (x452) {
x1084 += -7;
} else {
}
if (x452) {
x1085 += 1;
} else {
}

}
x1076 += -7;
if (x542) {
x1077 += -7;
} else {
}
if (x542) {
x1078 += 1;
} else {
}

}
x1069 += 7;
if (x452) {
x1070 += 7;
} else {
}
if (x452) {
x1071 += -1;
} else {
}

}
float* x1137 = (float*)myMalloc(7 * sizeof(float));;
for(int x1138=0; x1138 < 7; x1138++) {
float x1139 = x1068[x1138];
bool x1140 = x1139 < 0.0f;
if (x1140) {
x1137[x1138] = 0.0f;
} else {
float x1143 = x1068[x1138];
x1137[x1138] = x1143;
}

}
float* x1157 = (float*)myMalloc(x1156 * sizeof(float));;
float* x1159 = (float*)myMalloc(x1158 * sizeof(float));;
for(int x1160=0; x1160 < 1; x1160++) {
int32_t x1161 = x1160 * 7;
float* x1162 = x1137+x1161;
int32_t x1163 = x1160 * x1154;
float* x1164 = x1157+x1163;
int32_t x1165 = x1160 * x1158;
float* x1166 = x1159+x1165;
for(int x1168=0; x1168 < -9; x1168++) {
int32_t x1169 = x1168 / 9;
int32_t x1173 = x1169 * 3;
int32_t x1174 = x1173 * 3;
int32_t x1175 = x1174 * x1150;
int32_t x1176 = x1175 * x1152;
int32_t x1170 = x1168 % 9;
int32_t x1171 = x1170 / 3;
int32_t x1177 = x1171 * 3;
int32_t x1178 = x1177 * x1150;
int32_t x1179 = x1178 * x1152;
int32_t x1180 = x1176 + x1179;
int32_t x1172 = x1170 % 3;
int32_t x1181 = x1172 * x1152;
int32_t x1182 = x1181 * x1152;
int32_t x1183 = x1180 + x1182;
float* x1184 = x1166+x1183;
int32_t x1185 = x1169 * -7;
float* x1186 = x1162+x1185;
int32_t x1199 = 1 - x1172;
bool x1200 = x1199 > 0;
int32_t x1201;
if (x1200) {
x1201 = x1199;
} else {
x1201 = 0;
}
int32_t x1202 = 3 - x1172;
int32_t x1203 = x1202 - 1;
int32_t x1204 = 1 - x1203;
bool x1205 = x1204 > 0;
int32_t x1206;
if (x1205) {
x1206 = x1204;
} else {
x1206 = 0;
}
int32_t x1207 = x1152 - x1206;
int32_t x1208 = x1207 - x1201;
bool x1209 = x1208 <= 0;
bool x1213 = x1201 > 0;
int32_t x1198 = -1 + x1172;
bool x1226 = x1206 > 0;
for(int x1188=0; x1188 < x1150; x1188++) {
int32_t x1189 = x1188 - 1;
int32_t x1190 = x1189 + x1171;
bool x1191 = x1190 < 0;
bool x1192 = x1190 >= 1;
bool x1193 = x1191 || x1192;
if (x1193) {
int32_t x1194 = x1188 * x1152;
float* x1195 = x1184+x1194;
memset(x1195, 0, 4 * x1152);;
} else {
if (x1209) {
int32_t x1194 = x1188 * x1152;
float* x1210 = x1184+x1194;
memset(x1210, 0, 4 * x1152);;
} else {
int32_t x1194 = x1188 * x1152;
if (x1213) {
float* x1214 = x1184+x1194;
memset(x1214, 0, 4 * x1201);;
} else {
}
// may have segfault here
int32_t x1219 = x1194 + x1201;
float* x1220 = x1184+x1219;
int32_t x1221 = x1190 * -7;
int32_t x1222 = x1221 + x1198;
int32_t x1223 = x1222 + x1201;
float* x1224 = x1186+x1223;
memcpy(x1220, x1224, 4 * x1208);;
if (x1226) {
int32_t x1227 = x1194 + x1152;
int32_t x1228 = x1227 - x1206;
float* x1229 = x1184+x1228;
memset(x1229, 0, 4 * x1206);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x1153,-9,1,x90,-9,x1166,x1153,1,x1164,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(64) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x1248 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x1249 = 0;
int32_t x1250 = 0;
int32_t x1251 = 0;
for(int x1252=0; x1252 < -7; x1252++) {
int32_t x1253 = x1249;
int32_t x1254 = x1250;
float x1255 = x1157[x1254];
int32_t x1256 = x1251;
float x1257 = x105[x1256];
float x1258 = x1255 - x1257;
x1248[x1253] = x1258;
x1249 += 1;
if (x452) {
x1250 += x1154;
} else {
}
if (x452) {
x1251 += -1;
} else {
}

}
float* x1269 = (float*)myMalloc(64 * sizeof(float));;
for(int x1270=0; x1270 < 64; x1270++) {
float x1271 = x158[x1270];
float x1272 = x1271 + 1.0E-5f;
x1269[x1270] = x1272;

}
float* x1276 = (float*)myMalloc(64 * sizeof(float));;
for(int x1277=0; x1277 < 64; x1277++) {
float x1278 = x1269[x1277];
double x1279 = (double)x1278;
double x1280 = sqrt(x1279);
float x1281 = (float)x1280;
x1276[x1277] = x1281;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x1289 = (float*)myMalloc(7 * sizeof(float));;
int32_t x1290 = 0;
int32_t x1291 = 0;
int32_t x1292 = 0;
for(int x1293=0; x1293 < 1; x1293++) {
int32_t x1294 = x1291;
int32_t x1295 = x1292;
int32_t x1296 = x1290;
int32_t x1297 = x1296;
int32_t x1298 = x1294;
int32_t x1299 = x1295;
for(int x1300=0; x1300 < -1; x1300++) {
int32_t x1301 = x1298;
int32_t x1302 = x1299;
int32_t x1303 = x1297;
int32_t x1304 = x1303;
int32_t x1305 = x1301;
int32_t x1306 = x1302;
for(int x1307=0; x1307 < 1; x1307++) {
int32_t x1308 = x1305;
int32_t x1309 = x1306;
int32_t x1310 = x1304;
int32_t x1311 = x1310;
int32_t x1312 = x1308;
int32_t x1313 = x1309;
for(int x1314=0; x1314 < -7; x1314++) {
int32_t x1315 = x1311;
int32_t x1316 = x1312;
float x1317 = x1248[x1316];
int32_t x1318 = x1313;
float x1319 = x1276[x1318];
float x1320 = x1317 / x1319;
x1289[x1315] = x1320;
x1311 += 1;
if (x520) {
x1312 += 1;
} else {
}
if (x452) {
x1313 += 1;
} else {
}

}
x1304 += -7;
if (x452) {
x1305 += -7;
} else {
}
if (x452) {
x1306 += 1;
} else {
}

}
x1297 += -7;
if (x452) {
x1298 += -7;
} else {
}
if (x542) {
x1299 += 1;
} else {
}

}
x1290 += 7;
if (x452) {
x1291 += -7;
} else {
}
if (x452) {
x1292 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x1362 = (float*)myMalloc(7 * sizeof(float));;
int32_t x1363 = 0;
int32_t x1364 = 0;
int32_t x1365 = 0;
for(int x1366=0; x1366 < 1; x1366++) {
int32_t x1367 = x1364;
int32_t x1368 = x1365;
int32_t x1369 = x1363;
int32_t x1370 = x1369;
int32_t x1371 = x1367;
int32_t x1372 = x1368;
for(int x1373=0; x1373 < -1; x1373++) {
int32_t x1374 = x1371;
int32_t x1375 = x1372;
int32_t x1376 = x1370;
int32_t x1377 = x1376;
int32_t x1378 = x1374;
int32_t x1379 = x1375;
for(int x1380=0; x1380 < 1; x1380++) {
int32_t x1381 = x1378;
int32_t x1382 = x1379;
int32_t x1383 = x1377;
int32_t x1384 = x1383;
int32_t x1385 = x1381;
int32_t x1386 = x1382;
for(int x1387=0; x1387 < -7; x1387++) {
int32_t x1388 = x1384;
int32_t x1389 = x1385;
float x1390 = x1289[x1389];
int32_t x1391 = x1386;
float x1392 = x164[x1391];
float x1393 = x1390 * x1392;
x1362[x1388] = x1393;
x1384 += 1;
if (x520) {
x1385 += 1;
} else {
}
if (x452) {
x1386 += 1;
} else {
}

}
x1377 += -7;
if (x452) {
x1378 += -7;
} else {
}
if (x452) {
x1379 += 1;
} else {
}

}
x1370 += -7;
if (x542) {
x1371 += -7;
} else {
}
if (x542) {
x1372 += 1;
} else {
}

}
x1363 += 7;
if (x452) {
x1364 += 7;
} else {
}
if (x452) {
x1365 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x1435 = (float*)myMalloc(7 * sizeof(float));;
int32_t x1436 = 0;
int32_t x1437 = 0;
int32_t x1438 = 0;
for(int x1439=0; x1439 < 1; x1439++) {
int32_t x1440 = x1437;
int32_t x1441 = x1438;
int32_t x1442 = x1436;
int32_t x1443 = x1442;
int32_t x1444 = x1440;
int32_t x1445 = x1441;
for(int x1446=0; x1446 < -1; x1446++) {
int32_t x1447 = x1444;
int32_t x1448 = x1445;
int32_t x1449 = x1443;
int32_t x1450 = x1449;
int32_t x1451 = x1447;
int32_t x1452 = x1448;
for(int x1453=0; x1453 < 1; x1453++) {
int32_t x1454 = x1451;
int32_t x1455 = x1452;
int32_t x1456 = x1450;
int32_t x1457 = x1456;
int32_t x1458 = x1454;
int32_t x1459 = x1455;
for(int x1460=0; x1460 < -7; x1460++) {
int32_t x1461 = x1457;
int32_t x1462 = x1458;
float x1463 = x1362[x1462];
int32_t x1464 = x1459;
float x1465 = x49[x1464];
float x1466 = x1463 + x1465;
x1435[x1461] = x1466;
x1457 += 1;
if (x520) {
x1458 += 1;
} else {
}
if (x452) {
x1459 += 1;
} else {
}

}
x1450 += -7;
if (x452) {
x1451 += -7;
} else {
}
if (x452) {
x1452 += 1;
} else {
}

}
x1443 += -7;
if (x542) {
x1444 += -7;
} else {
}
if (x542) {
x1445 += 1;
} else {
}

}
x1436 += 7;
if (x452) {
x1437 += 7;
} else {
}
if (x452) {
x1438 += -1;
} else {
}

}
float* x1504 = (float*)myMalloc(7 * sizeof(float));;
for(int x1505=0; x1505 < 7; x1505++) {
float x1506 = x1435[x1505];
bool x1507 = x1506 < 0.0f;
if (x1507) {
x1504[x1505] = 0.0f;
} else {
float x1510 = x1435[x1505];
x1504[x1505] = x1510;
}

}
float* x1519 = (float*)myMalloc(x1518 * sizeof(float));;
float* x1521 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x1522=0; x1522 < 1; x1522++) {
int32_t x1523 = x1522 * 7;
float* x1524 = x1504+x1523;
int32_t x1525 = x1522 * x1516;
float* x1526 = x1519+x1525;
int32_t x1527 = x1522 * x1520;
float* x1528 = x1521+x1527;
for(int x1529=0; x1529 < -1; x1529++) {
int32_t x1530 = x1529 / 1;
int32_t x1534 = x1530 * x1150;
int32_t x1535 = x1534 * x1152;
int32_t x1531 = x1529 % 1;
int32_t x1532 = x1531 / 1;
int32_t x1536 = x1532 * x1150;
int32_t x1537 = x1536 * x1152;
int32_t x1538 = x1535 + x1537;
int32_t x1533 = x1531 % 1;
int32_t x1539 = x1533 * x1152;
int32_t x1540 = x1539 * x1152;
int32_t x1541 = x1538 + x1540;
float* x1542 = x1528+x1541;
int32_t x1543 = x1530 * -7;
float* x1544 = x1524+x1543;
for(int x1545=0; x1545 < x1150; x1545++) {
int32_t x1547 = x1545 * x1152;
float* x1548 = x1542+x1547;
int32_t x1546 = x1545 + x1532;
int32_t x1549 = x1546 * -7;
int32_t x1550 = x1549 + x1533;
float* x1551 = x1544+x1550;
memcpy(x1548, x1551, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1153,-1,1,x32,-1,x1528,x1153,1,x1526,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x1564 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x1565 = 0;
int32_t x1566 = 0;
int32_t x1567 = 0;
for(int x1568=0; x1568 < -7; x1568++) {
int32_t x1569 = x1565;
int32_t x1570 = x1566;
float x1571 = x1519[x1570];
int32_t x1572 = x1567;
float x1573 = x71[x1572];
float x1574 = x1571 - x1573;
x1564[x1569] = x1574;
x1565 += 1;
if (x452) {
x1566 += x1516;
} else {
}
if (x452) {
x1567 += -1;
} else {
}

}
float* x1585 = (float*)myMalloc(256 * sizeof(float));;
for(int x1587=0; x1587 < 256; x1587++) {
float x1588 = x36[x1587];
float x1589 = x1588 + 1.0E-5f;
x1585[x1587] = x1589;

}
float* x1593 = (float*)myMalloc(256 * sizeof(float));;
for(int x1594=0; x1594 < 256; x1594++) {
float x1595 = x1585[x1594];
double x1596 = (double)x1595;
double x1597 = sqrt(x1596);
float x1598 = (float)x1597;
x1593[x1594] = x1598;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x1606 = (float*)myMalloc(7 * sizeof(float));;
int32_t x1607 = 0;
int32_t x1608 = 0;
int32_t x1609 = 0;
for(int x1610=0; x1610 < 1; x1610++) {
int32_t x1611 = x1608;
int32_t x1612 = x1609;
int32_t x1613 = x1607;
int32_t x1614 = x1613;
int32_t x1615 = x1611;
int32_t x1616 = x1612;
for(int x1617=0; x1617 < -1; x1617++) {
int32_t x1618 = x1615;
int32_t x1619 = x1616;
int32_t x1620 = x1614;
int32_t x1621 = x1620;
int32_t x1622 = x1618;
int32_t x1623 = x1619;
for(int x1624=0; x1624 < 1; x1624++) {
int32_t x1625 = x1622;
int32_t x1626 = x1623;
int32_t x1627 = x1621;
int32_t x1628 = x1627;
int32_t x1629 = x1625;
int32_t x1630 = x1626;
for(int x1631=0; x1631 < -7; x1631++) {
int32_t x1632 = x1628;
int32_t x1633 = x1629;
float x1634 = x1564[x1633];
int32_t x1635 = x1630;
float x1636 = x1593[x1635];
float x1637 = x1634 / x1636;
x1606[x1632] = x1637;
x1628 += 1;
if (x520) {
x1629 += 1;
} else {
}
if (x452) {
x1630 += 1;
} else {
}

}
x1621 += -7;
if (x452) {
x1622 += -7;
} else {
}
if (x452) {
x1623 += 1;
} else {
}

}
x1614 += -7;
if (x452) {
x1615 += -7;
} else {
}
if (x542) {
x1616 += 1;
} else {
}

}
x1607 += 7;
if (x452) {
x1608 += -7;
} else {
}
if (x452) {
x1609 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x1679 = (float*)myMalloc(7 * sizeof(float));;
int32_t x1680 = 0;
int32_t x1681 = 0;
int32_t x1682 = 0;
for(int x1683=0; x1683 < 1; x1683++) {
int32_t x1684 = x1681;
int32_t x1685 = x1682;
int32_t x1686 = x1680;
int32_t x1687 = x1686;
int32_t x1688 = x1684;
int32_t x1689 = x1685;
for(int x1690=0; x1690 < -1; x1690++) {
int32_t x1691 = x1688;
int32_t x1692 = x1689;
int32_t x1693 = x1687;
int32_t x1694 = x1693;
int32_t x1695 = x1691;
int32_t x1696 = x1692;
for(int x1697=0; x1697 < 1; x1697++) {
int32_t x1698 = x1695;
int32_t x1699 = x1696;
int32_t x1700 = x1694;
int32_t x1701 = x1700;
int32_t x1702 = x1698;
int32_t x1703 = x1699;
for(int x1704=0; x1704 < -7; x1704++) {
int32_t x1705 = x1701;
int32_t x1706 = x1702;
float x1707 = x1606[x1706];
int32_t x1708 = x1703;
float x1709 = x199[x1708];
float x1710 = x1707 * x1709;
x1679[x1705] = x1710;
x1701 += 1;
if (x520) {
x1702 += 1;
} else {
}
if (x452) {
x1703 += 1;
} else {
}

}
x1694 += -7;
if (x452) {
x1695 += -7;
} else {
}
if (x452) {
x1696 += 1;
} else {
}

}
x1687 += -7;
if (x542) {
x1688 += -7;
} else {
}
if (x542) {
x1689 += 1;
} else {
}

}
x1680 += 7;
if (x452) {
x1681 += 7;
} else {
}
if (x452) {
x1682 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x1752 = (float*)myMalloc(7 * sizeof(float));;
int32_t x1753 = 0;
int32_t x1754 = 0;
int32_t x1755 = 0;
for(int x1756=0; x1756 < 1; x1756++) {
int32_t x1757 = x1754;
int32_t x1758 = x1755;
int32_t x1759 = x1753;
int32_t x1760 = x1759;
int32_t x1761 = x1757;
int32_t x1762 = x1758;
for(int x1763=0; x1763 < -1; x1763++) {
int32_t x1764 = x1761;
int32_t x1765 = x1762;
int32_t x1766 = x1760;
int32_t x1767 = x1766;
int32_t x1768 = x1764;
int32_t x1769 = x1765;
for(int x1770=0; x1770 < 1; x1770++) {
int32_t x1771 = x1768;
int32_t x1772 = x1769;
int32_t x1773 = x1767;
int32_t x1774 = x1773;
int32_t x1775 = x1771;
int32_t x1776 = x1772;
for(int x1777=0; x1777 < -7; x1777++) {
int32_t x1778 = x1774;
int32_t x1779 = x1775;
float x1780 = x1679[x1779];
int32_t x1781 = x1776;
float x1782 = x126[x1781];
float x1783 = x1780 + x1782;
x1752[x1778] = x1783;
x1774 += 1;
if (x520) {
x1775 += 1;
} else {
}
if (x452) {
x1776 += 1;
} else {
}

}
x1767 += -7;
if (x452) {
x1768 += -7;
} else {
}
if (x452) {
x1769 += 1;
} else {
}

}
x1760 += -7;
if (x542) {
x1761 += -7;
} else {
}
if (x542) {
x1762 += 1;
} else {
}

}
x1753 += 7;
if (x452) {
x1754 += 7;
} else {
}
if (x452) {
x1755 += -1;
} else {
}

}
float* x1824 = (float*)myMalloc(x1823 * sizeof(float));;
float* x1825 = (float*)myMalloc(x835 * sizeof(float));;
for(int x1826=0; x1826 < 1; x1826++) {
int32_t x1827 = x1826 * x728;
float* x1828 = x731+x1827;
int32_t x1829 = x1826 * x1821;
float* x1830 = x1824+x1829;
int32_t x1831 = x1826 * x835;
float* x1832 = x1825+x1831;
for(int x1833=0; x1833 < -1; x1833++) {
int32_t x1834 = x1833 / 1;
int32_t x1838 = x1834 * x827;
int32_t x1839 = x1838 * x829;
int32_t x1835 = x1833 % 1;
int32_t x1836 = x1835 / 1;
int32_t x1840 = x1836 * x827;
int32_t x1841 = x1840 * x829;
int32_t x1842 = x1839 + x1841;
int32_t x1837 = x1835 % 1;
int32_t x1843 = x1837 * x829;
int32_t x1844 = x1843 * x829;
int32_t x1845 = x1842 + x1844;
float* x1846 = x1832+x1845;
int32_t x1847 = x1834 * x724;
int32_t x1848 = x1847 * x726;
float* x1849 = x1828+x1848;
for(int x1850=0; x1850 < x827; x1850++) {
int32_t x1852 = x1850 * x829;
float* x1853 = x1846+x1852;
int32_t x1851 = x1850 + x1836;
int32_t x1854 = x1851 * x726;
int32_t x1855 = x1854 + x1837;
float* x1856 = x1849+x1855;
memcpy(x1853, x1856, 4 * x829);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x830,-1,1,x162,-1,x1832,x830,1,x1830,x830);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(827) x Sym(829)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x1869 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x1870 = 0;
int32_t x1871 = 0;
int32_t x1872 = 0;
for(int x1873=0; x1873 < -7; x1873++) {
int32_t x1874 = x1870;
int32_t x1875 = x1871;
float x1876 = x1824[x1875];
int32_t x1877 = x1872;
float x1878 = x264[x1877];
float x1879 = x1876 - x1878;
x1869[x1874] = x1879;
x1870 += 1;
if (x452) {
x1871 += x1821;
} else {
}
if (x452) {
x1872 += -1;
} else {
}

}
float* x1890 = (float*)myMalloc(256 * sizeof(float));;
for(int x1891=0; x1891 < 256; x1891++) {
float x1892 = x243[x1891];
float x1893 = x1892 + 1.0E-5f;
x1890[x1891] = x1893;

}
float* x1897 = (float*)myMalloc(256 * sizeof(float));;
for(int x1898=0; x1898 < 256; x1898++) {
float x1899 = x1890[x1898];
double x1900 = (double)x1899;
double x1901 = sqrt(x1900);
float x1902 = (float)x1901;
x1897[x1898] = x1902;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x1910 = (float*)myMalloc(7 * sizeof(float));;
int32_t x1911 = 0;
int32_t x1912 = 0;
int32_t x1913 = 0;
for(int x1914=0; x1914 < 1; x1914++) {
int32_t x1915 = x1912;
int32_t x1916 = x1913;
int32_t x1917 = x1911;
int32_t x1918 = x1917;
int32_t x1919 = x1915;
int32_t x1920 = x1916;
for(int x1921=0; x1921 < -1; x1921++) {
int32_t x1922 = x1919;
int32_t x1923 = x1920;
int32_t x1924 = x1918;
int32_t x1925 = x1924;
int32_t x1926 = x1922;
int32_t x1927 = x1923;
for(int x1928=0; x1928 < 1; x1928++) {
int32_t x1929 = x1926;
int32_t x1930 = x1927;
int32_t x1931 = x1925;
int32_t x1932 = x1931;
int32_t x1933 = x1929;
int32_t x1934 = x1930;
for(int x1935=0; x1935 < -7; x1935++) {
int32_t x1936 = x1932;
int32_t x1937 = x1933;
float x1938 = x1869[x1937];
int32_t x1939 = x1934;
float x1940 = x1897[x1939];
float x1941 = x1938 / x1940;
x1910[x1936] = x1941;
x1932 += 1;
if (x520) {
x1933 += 1;
} else {
}
if (x452) {
x1934 += 1;
} else {
}

}
x1925 += -7;
if (x452) {
x1926 += -7;
} else {
}
if (x452) {
x1927 += 1;
} else {
}

}
x1918 += -7;
if (x452) {
x1919 += -7;
} else {
}
if (x542) {
x1920 += 1;
} else {
}

}
x1911 += 7;
if (x452) {
x1912 += -7;
} else {
}
if (x452) {
x1913 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x1983 = (float*)myMalloc(7 * sizeof(float));;
int32_t x1984 = 0;
int32_t x1985 = 0;
int32_t x1986 = 0;
for(int x1987=0; x1987 < 1; x1987++) {
int32_t x1988 = x1985;
int32_t x1989 = x1986;
int32_t x1990 = x1984;
int32_t x1991 = x1990;
int32_t x1992 = x1988;
int32_t x1993 = x1989;
for(int x1994=0; x1994 < -1; x1994++) {
int32_t x1995 = x1992;
int32_t x1996 = x1993;
int32_t x1997 = x1991;
int32_t x1998 = x1997;
int32_t x1999 = x1995;
int32_t x2000 = x1996;
for(int x2001=0; x2001 < 1; x2001++) {
int32_t x2002 = x1999;
int32_t x2003 = x2000;
int32_t x2004 = x1998;
int32_t x2005 = x2004;
int32_t x2006 = x2002;
int32_t x2007 = x2003;
for(int x2008=0; x2008 < -7; x2008++) {
int32_t x2009 = x2005;
int32_t x2010 = x2006;
float x2011 = x1910[x2010];
int32_t x2012 = x2007;
float x2013 = x76[x2012];
float x2014 = x2011 * x2013;
x1983[x2009] = x2014;
x2005 += 1;
if (x520) {
x2006 += 1;
} else {
}
if (x452) {
x2007 += 1;
} else {
}

}
x1998 += -7;
if (x452) {
x1999 += -7;
} else {
}
if (x452) {
x2000 += 1;
} else {
}

}
x1991 += -7;
if (x542) {
x1992 += -7;
} else {
}
if (x542) {
x1993 += 1;
} else {
}

}
x1984 += 7;
if (x452) {
x1985 += 7;
} else {
}
if (x452) {
x1986 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x2056 = (float*)myMalloc(7 * sizeof(float));;
int32_t x2057 = 0;
int32_t x2058 = 0;
int32_t x2059 = 0;
for(int x2060=0; x2060 < 1; x2060++) {
int32_t x2061 = x2058;
int32_t x2062 = x2059;
int32_t x2063 = x2057;
int32_t x2064 = x2063;
int32_t x2065 = x2061;
int32_t x2066 = x2062;
for(int x2067=0; x2067 < -1; x2067++) {
int32_t x2068 = x2065;
int32_t x2069 = x2066;
int32_t x2070 = x2064;
int32_t x2071 = x2070;
int32_t x2072 = x2068;
int32_t x2073 = x2069;
for(int x2074=0; x2074 < 1; x2074++) {
int32_t x2075 = x2072;
int32_t x2076 = x2073;
int32_t x2077 = x2071;
int32_t x2078 = x2077;
int32_t x2079 = x2075;
int32_t x2080 = x2076;
for(int x2081=0; x2081 < -7; x2081++) {
int32_t x2082 = x2078;
int32_t x2083 = x2079;
float x2084 = x1983[x2083];
int32_t x2085 = x2080;
float x2086 = x203[x2085];
float x2087 = x2084 + x2086;
x2056[x2082] = x2087;
x2078 += 1;
if (x520) {
x2079 += 1;
} else {
}
if (x452) {
x2080 += 1;
} else {
}

}
x2071 += -7;
if (x452) {
x2072 += -7;
} else {
}
if (x452) {
x2073 += 1;
} else {
}

}
x2064 += -7;
if (x542) {
x2065 += -7;
} else {
}
if (x542) {
x2066 += 1;
} else {
}

}
x2057 += 7;
if (x452) {
x2058 += 7;
} else {
}
if (x452) {
x2059 += -1;
} else {
}

}
int32_t x2125 = 0;
int32_t x2126 = 0;
int32_t x2127 = 0;
for(int x2128=0; x2128 < 1; x2128++) {
int32_t x2129 = x2126;
int32_t x2130 = x2127;
int32_t x2131 = x2125;
int32_t x2132 = x2131;
int32_t x2133 = x2129;
int32_t x2134 = x2130;
for(int x2135=0; x2135 < -1; x2135++) {
int32_t x2136 = x2133;
int32_t x2137 = x2134;
int32_t x2138 = x2132;
int32_t x2139 = x2138;
int32_t x2140 = x2136;
int32_t x2141 = x2137;
for(int x2142=0; x2142 < 1; x2142++) {
int32_t x2143 = x2140;
int32_t x2144 = x2141;
int32_t x2145 = x2139;
int32_t x2146 = x2145;
int32_t x2147 = x2143;
int32_t x2148 = x2144;
for(int x2149=0; x2149 < -7; x2149++) {
int32_t x2150 = x2147;
float x2151 = x1752[x2150];
int32_t x2152 = x2148;
float x2153 = x2056[x2152];
float x2154 = x2151 + x2153;
x1752[x2150] = x2154;
x2146 += 1;
if (x520) {
x2147 += 1;
} else {
}
if (x520) {
x2148 += 1;
} else {
}

}
x2139 += -7;
if (x452) {
x2140 += -7;
} else {
}
if (x452) {
x2141 += -7;
} else {
}

}
x2132 += -7;
if (x542) {
x2133 += -7;
} else {
}
if (x542) {
x2134 += -7;
} else {
}

}
x2125 += 7;
if (x452) {
x2126 += 7;
} else {
}
if (x452) {
x2127 += 7;
} else {
}

}
float* x2192 = (float*)myMalloc(7 * sizeof(float));;
for(int x2193=0; x2193 < 7; x2193++) {
float x2194 = x1752[x2193];
bool x2195 = x2194 < 0.0f;
if (x2195) {
x2192[x2193] = 0.0f;
} else {
float x2198 = x1752[x2193];
x2192[x2193] = x2198;
}

}
float* x2204 = (float*)myMalloc(x1156 * sizeof(float));;
float* x2205 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x2206=0; x2206 < 1; x2206++) {
int32_t x2207 = x2206 * 7;
float* x2208 = x2192+x2207;
int32_t x2209 = x2206 * x1154;
float* x2210 = x2204+x2209;
int32_t x2211 = x2206 * x1520;
float* x2212 = x2205+x2211;
for(int x2213=0; x2213 < -1; x2213++) {
int32_t x2214 = x2213 / 1;
int32_t x2218 = x2214 * x1150;
int32_t x2219 = x2218 * x1152;
int32_t x2215 = x2213 % 1;
int32_t x2216 = x2215 / 1;
int32_t x2220 = x2216 * x1150;
int32_t x2221 = x2220 * x1152;
int32_t x2222 = x2219 + x2221;
int32_t x2217 = x2215 % 1;
int32_t x2223 = x2217 * x1152;
int32_t x2224 = x2223 * x1152;
int32_t x2225 = x2222 + x2224;
float* x2226 = x2212+x2225;
int32_t x2227 = x2214 * -7;
float* x2228 = x2208+x2227;
for(int x2229=0; x2229 < x1150; x2229++) {
int32_t x2231 = x2229 * x1152;
float* x2232 = x2226+x2231;
int32_t x2230 = x2229 + x2216;
int32_t x2233 = x2230 * -7;
int32_t x2234 = x2233 + x2217;
float* x2235 = x2228+x2234;
memcpy(x2232, x2235, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x1153,-1,1,x171,-1,x2212,x1153,1,x2210,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(64) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x2248 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x2249 = 0;
int32_t x2250 = 0;
int32_t x2251 = 0;
for(int x2252=0; x2252 < -7; x2252++) {
int32_t x2253 = x2249;
int32_t x2254 = x2250;
float x2255 = x2204[x2254];
int32_t x2256 = x2251;
float x2257 = x10[x2256];
float x2258 = x2255 - x2257;
x2248[x2253] = x2258;
x2249 += 1;
if (x452) {
x2250 += x1154;
} else {
}
if (x452) {
x2251 += -1;
} else {
}

}
float* x2269 = (float*)myMalloc(64 * sizeof(float));;
for(int x2270=0; x2270 < 64; x2270++) {
float x2271 = x102[x2270];
float x2272 = x2271 + 1.0E-5f;
x2269[x2270] = x2272;

}
float* x2276 = (float*)myMalloc(64 * sizeof(float));;
for(int x2277=0; x2277 < 64; x2277++) {
float x2278 = x2269[x2277];
double x2279 = (double)x2278;
double x2280 = sqrt(x2279);
float x2281 = (float)x2280;
x2276[x2277] = x2281;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x2289 = (float*)myMalloc(7 * sizeof(float));;
int32_t x2290 = 0;
int32_t x2291 = 0;
int32_t x2292 = 0;
for(int x2293=0; x2293 < 1; x2293++) {
int32_t x2294 = x2291;
int32_t x2295 = x2292;
int32_t x2296 = x2290;
int32_t x2297 = x2296;
int32_t x2298 = x2294;
int32_t x2299 = x2295;
for(int x2300=0; x2300 < -1; x2300++) {
int32_t x2301 = x2298;
int32_t x2302 = x2299;
int32_t x2303 = x2297;
int32_t x2304 = x2303;
int32_t x2305 = x2301;
int32_t x2306 = x2302;
for(int x2307=0; x2307 < 1; x2307++) {
int32_t x2308 = x2305;
int32_t x2309 = x2306;
int32_t x2310 = x2304;
int32_t x2311 = x2310;
int32_t x2312 = x2308;
int32_t x2313 = x2309;
for(int x2314=0; x2314 < -7; x2314++) {
int32_t x2315 = x2311;
int32_t x2316 = x2312;
float x2317 = x2248[x2316];
int32_t x2318 = x2313;
float x2319 = x2276[x2318];
float x2320 = x2317 / x2319;
x2289[x2315] = x2320;
x2311 += 1;
if (x520) {
x2312 += 1;
} else {
}
if (x452) {
x2313 += 1;
} else {
}

}
x2304 += -7;
if (x452) {
x2305 += -7;
} else {
}
if (x452) {
x2306 += 1;
} else {
}

}
x2297 += -7;
if (x452) {
x2298 += -7;
} else {
}
if (x542) {
x2299 += 1;
} else {
}

}
x2290 += 7;
if (x452) {
x2291 += -7;
} else {
}
if (x452) {
x2292 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x2362 = (float*)myMalloc(7 * sizeof(float));;
int32_t x2363 = 0;
int32_t x2364 = 0;
int32_t x2365 = 0;
for(int x2366=0; x2366 < 1; x2366++) {
int32_t x2367 = x2364;
int32_t x2368 = x2365;
int32_t x2369 = x2363;
int32_t x2370 = x2369;
int32_t x2371 = x2367;
int32_t x2372 = x2368;
for(int x2373=0; x2373 < -1; x2373++) {
int32_t x2374 = x2371;
int32_t x2375 = x2372;
int32_t x2376 = x2370;
int32_t x2377 = x2376;
int32_t x2378 = x2374;
int32_t x2379 = x2375;
for(int x2380=0; x2380 < 1; x2380++) {
int32_t x2381 = x2378;
int32_t x2382 = x2379;
int32_t x2383 = x2377;
int32_t x2384 = x2383;
int32_t x2385 = x2381;
int32_t x2386 = x2382;
for(int x2387=0; x2387 < -7; x2387++) {
int32_t x2388 = x2384;
int32_t x2389 = x2385;
float x2390 = x2289[x2389];
int32_t x2391 = x2386;
float x2392 = x142[x2391];
float x2393 = x2390 * x2392;
x2362[x2388] = x2393;
x2384 += 1;
if (x520) {
x2385 += 1;
} else {
}
if (x452) {
x2386 += 1;
} else {
}

}
x2377 += -7;
if (x452) {
x2378 += -7;
} else {
}
if (x452) {
x2379 += 1;
} else {
}

}
x2370 += -7;
if (x542) {
x2371 += -7;
} else {
}
if (x542) {
x2372 += 1;
} else {
}

}
x2363 += 7;
if (x452) {
x2364 += 7;
} else {
}
if (x452) {
x2365 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x2435 = (float*)myMalloc(7 * sizeof(float));;
int32_t x2436 = 0;
int32_t x2437 = 0;
int32_t x2438 = 0;
for(int x2439=0; x2439 < 1; x2439++) {
int32_t x2440 = x2437;
int32_t x2441 = x2438;
int32_t x2442 = x2436;
int32_t x2443 = x2442;
int32_t x2444 = x2440;
int32_t x2445 = x2441;
for(int x2446=0; x2446 < -1; x2446++) {
int32_t x2447 = x2444;
int32_t x2448 = x2445;
int32_t x2449 = x2443;
int32_t x2450 = x2449;
int32_t x2451 = x2447;
int32_t x2452 = x2448;
for(int x2453=0; x2453 < 1; x2453++) {
int32_t x2454 = x2451;
int32_t x2455 = x2452;
int32_t x2456 = x2450;
int32_t x2457 = x2456;
int32_t x2458 = x2454;
int32_t x2459 = x2455;
for(int x2460=0; x2460 < -7; x2460++) {
int32_t x2461 = x2457;
int32_t x2462 = x2458;
float x2463 = x2362[x2462];
int32_t x2464 = x2459;
float x2465 = x60[x2464];
float x2466 = x2463 + x2465;
x2435[x2461] = x2466;
x2457 += 1;
if (x520) {
x2458 += 1;
} else {
}
if (x452) {
x2459 += 1;
} else {
}

}
x2450 += -7;
if (x452) {
x2451 += -7;
} else {
}
if (x452) {
x2452 += 1;
} else {
}

}
x2443 += -7;
if (x542) {
x2444 += -7;
} else {
}
if (x542) {
x2445 += 1;
} else {
}

}
x2436 += 7;
if (x452) {
x2437 += 7;
} else {
}
if (x452) {
x2438 += -1;
} else {
}

}
float* x2504 = (float*)myMalloc(7 * sizeof(float));;
for(int x2505=0; x2505 < 7; x2505++) {
float x2506 = x2435[x2505];
bool x2507 = x2506 < 0.0f;
if (x2507) {
x2504[x2505] = 0.0f;
} else {
float x2510 = x2435[x2505];
x2504[x2505] = x2510;
}

}
float* x2516 = (float*)myMalloc(x1156 * sizeof(float));;
float* x2517 = (float*)myMalloc(x1158 * sizeof(float));;
for(int x2518=0; x2518 < 1; x2518++) {
int32_t x2519 = x2518 * 7;
float* x2520 = x2504+x2519;
int32_t x2521 = x2518 * x1154;
float* x2522 = x2516+x2521;
int32_t x2523 = x2518 * x1158;
float* x2524 = x2517+x2523;
for(int x2525=0; x2525 < -9; x2525++) {
int32_t x2526 = x2525 / 9;
int32_t x2530 = x2526 * 3;
int32_t x2531 = x2530 * 3;
int32_t x2532 = x2531 * x1150;
int32_t x2533 = x2532 * x1152;
int32_t x2527 = x2525 % 9;
int32_t x2528 = x2527 / 3;
int32_t x2534 = x2528 * 3;
int32_t x2535 = x2534 * x1150;
int32_t x2536 = x2535 * x1152;
int32_t x2537 = x2533 + x2536;
int32_t x2529 = x2527 % 3;
int32_t x2538 = x2529 * x1152;
int32_t x2539 = x2538 * x1152;
int32_t x2540 = x2537 + x2539;
float* x2541 = x2524+x2540;
int32_t x2542 = x2526 * -7;
float* x2543 = x2520+x2542;
int32_t x2555 = 1 - x2529;
bool x2556 = x2555 > 0;
int32_t x2557;
if (x2556) {
x2557 = x2555;
} else {
x2557 = 0;
}
int32_t x2558 = 3 - x2529;
int32_t x2559 = x2558 - 1;
int32_t x2560 = 1 - x2559;
bool x2561 = x2560 > 0;
int32_t x2562;
if (x2561) {
x2562 = x2560;
} else {
x2562 = 0;
}
int32_t x2563 = x1152 - x2562;
int32_t x2564 = x2563 - x2557;
bool x2565 = x2564 <= 0;
bool x2569 = x2557 > 0;
int32_t x2554 = -1 + x2529;
bool x2582 = x2562 > 0;
for(int x2544=0; x2544 < x1150; x2544++) {
int32_t x2545 = x2544 - 1;
int32_t x2546 = x2545 + x2528;
bool x2547 = x2546 < 0;
bool x2548 = x2546 >= 1;
bool x2549 = x2547 || x2548;
if (x2549) {
int32_t x2550 = x2544 * x1152;
float* x2551 = x2541+x2550;
memset(x2551, 0, 4 * x1152);;
} else {
if (x2565) {
int32_t x2550 = x2544 * x1152;
float* x2566 = x2541+x2550;
memset(x2566, 0, 4 * x1152);;
} else {
int32_t x2550 = x2544 * x1152;
if (x2569) {
float* x2570 = x2541+x2550;
memset(x2570, 0, 4 * x2557);;
} else {
}
// may have segfault here
int32_t x2575 = x2550 + x2557;
float* x2576 = x2541+x2575;
int32_t x2577 = x2546 * -7;
int32_t x2578 = x2577 + x2554;
int32_t x2579 = x2578 + x2557;
float* x2580 = x2543+x2579;
memcpy(x2576, x2580, 4 * x2564);;
if (x2582) {
int32_t x2583 = x2550 + x1152;
int32_t x2584 = x2583 - x2562;
float* x2585 = x2541+x2584;
memset(x2585, 0, 4 * x2562);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x1153,-9,1,x83,-9,x2524,x1153,1,x2522,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(64) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x2604 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x2605 = 0;
int32_t x2606 = 0;
int32_t x2607 = 0;
for(int x2608=0; x2608 < -7; x2608++) {
int32_t x2609 = x2605;
int32_t x2610 = x2606;
float x2611 = x2516[x2610];
int32_t x2612 = x2607;
float x2613 = x44[x2612];
float x2614 = x2611 - x2613;
x2604[x2609] = x2614;
x2605 += 1;
if (x452) {
x2606 += x1154;
} else {
}
if (x452) {
x2607 += -1;
} else {
}

}
float* x2625 = (float*)myMalloc(64 * sizeof(float));;
for(int x2626=0; x2626 < 64; x2626++) {
float x2627 = x244[x2626];
float x2628 = x2627 + 1.0E-5f;
x2625[x2626] = x2628;

}
float* x2632 = (float*)myMalloc(64 * sizeof(float));;
for(int x2633=0; x2633 < 64; x2633++) {
float x2634 = x2625[x2633];
double x2635 = (double)x2634;
double x2636 = sqrt(x2635);
float x2637 = (float)x2636;
x2632[x2633] = x2637;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x2645 = (float*)myMalloc(7 * sizeof(float));;
int32_t x2646 = 0;
int32_t x2647 = 0;
int32_t x2648 = 0;
for(int x2649=0; x2649 < 1; x2649++) {
int32_t x2650 = x2647;
int32_t x2651 = x2648;
int32_t x2652 = x2646;
int32_t x2653 = x2652;
int32_t x2654 = x2650;
int32_t x2655 = x2651;
for(int x2656=0; x2656 < -1; x2656++) {
int32_t x2657 = x2654;
int32_t x2658 = x2655;
int32_t x2659 = x2653;
int32_t x2660 = x2659;
int32_t x2661 = x2657;
int32_t x2662 = x2658;
for(int x2663=0; x2663 < 1; x2663++) {
int32_t x2664 = x2661;
int32_t x2665 = x2662;
int32_t x2666 = x2660;
int32_t x2667 = x2666;
int32_t x2668 = x2664;
int32_t x2669 = x2665;
for(int x2670=0; x2670 < -7; x2670++) {
int32_t x2671 = x2667;
int32_t x2672 = x2668;
float x2673 = x2604[x2672];
int32_t x2674 = x2669;
float x2675 = x2632[x2674];
float x2676 = x2673 / x2675;
x2645[x2671] = x2676;
x2667 += 1;
if (x520) {
x2668 += 1;
} else {
}
if (x452) {
x2669 += 1;
} else {
}

}
x2660 += -7;
if (x452) {
x2661 += -7;
} else {
}
if (x452) {
x2662 += 1;
} else {
}

}
x2653 += -7;
if (x452) {
x2654 += -7;
} else {
}
if (x542) {
x2655 += 1;
} else {
}

}
x2646 += 7;
if (x452) {
x2647 += -7;
} else {
}
if (x452) {
x2648 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x2718 = (float*)myMalloc(7 * sizeof(float));;
int32_t x2719 = 0;
int32_t x2720 = 0;
int32_t x2721 = 0;
for(int x2722=0; x2722 < 1; x2722++) {
int32_t x2723 = x2720;
int32_t x2724 = x2721;
int32_t x2725 = x2719;
int32_t x2726 = x2725;
int32_t x2727 = x2723;
int32_t x2728 = x2724;
for(int x2729=0; x2729 < -1; x2729++) {
int32_t x2730 = x2727;
int32_t x2731 = x2728;
int32_t x2732 = x2726;
int32_t x2733 = x2732;
int32_t x2734 = x2730;
int32_t x2735 = x2731;
for(int x2736=0; x2736 < 1; x2736++) {
int32_t x2737 = x2734;
int32_t x2738 = x2735;
int32_t x2739 = x2733;
int32_t x2740 = x2739;
int32_t x2741 = x2737;
int32_t x2742 = x2738;
for(int x2743=0; x2743 < -7; x2743++) {
int32_t x2744 = x2740;
int32_t x2745 = x2741;
float x2746 = x2645[x2745];
int32_t x2747 = x2742;
float x2748 = x208[x2747];
float x2749 = x2746 * x2748;
x2718[x2744] = x2749;
x2740 += 1;
if (x520) {
x2741 += 1;
} else {
}
if (x452) {
x2742 += 1;
} else {
}

}
x2733 += -7;
if (x452) {
x2734 += -7;
} else {
}
if (x452) {
x2735 += 1;
} else {
}

}
x2726 += -7;
if (x542) {
x2727 += -7;
} else {
}
if (x542) {
x2728 += 1;
} else {
}

}
x2719 += 7;
if (x452) {
x2720 += 7;
} else {
}
if (x452) {
x2721 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x2791 = (float*)myMalloc(7 * sizeof(float));;
int32_t x2792 = 0;
int32_t x2793 = 0;
int32_t x2794 = 0;
for(int x2795=0; x2795 < 1; x2795++) {
int32_t x2796 = x2793;
int32_t x2797 = x2794;
int32_t x2798 = x2792;
int32_t x2799 = x2798;
int32_t x2800 = x2796;
int32_t x2801 = x2797;
for(int x2802=0; x2802 < -1; x2802++) {
int32_t x2803 = x2800;
int32_t x2804 = x2801;
int32_t x2805 = x2799;
int32_t x2806 = x2805;
int32_t x2807 = x2803;
int32_t x2808 = x2804;
for(int x2809=0; x2809 < 1; x2809++) {
int32_t x2810 = x2807;
int32_t x2811 = x2808;
int32_t x2812 = x2806;
int32_t x2813 = x2812;
int32_t x2814 = x2810;
int32_t x2815 = x2811;
for(int x2816=0; x2816 < -7; x2816++) {
int32_t x2817 = x2813;
int32_t x2818 = x2814;
float x2819 = x2718[x2818];
int32_t x2820 = x2815;
float x2821 = x153[x2820];
float x2822 = x2819 + x2821;
x2791[x2817] = x2822;
x2813 += 1;
if (x520) {
x2814 += 1;
} else {
}
if (x452) {
x2815 += 1;
} else {
}

}
x2806 += -7;
if (x452) {
x2807 += -7;
} else {
}
if (x452) {
x2808 += 1;
} else {
}

}
x2799 += -7;
if (x542) {
x2800 += -7;
} else {
}
if (x542) {
x2801 += 1;
} else {
}

}
x2792 += 7;
if (x452) {
x2793 += 7;
} else {
}
if (x452) {
x2794 += -1;
} else {
}

}
float* x2860 = (float*)myMalloc(7 * sizeof(float));;
for(int x2861=0; x2861 < 7; x2861++) {
float x2862 = x2791[x2861];
bool x2863 = x2862 < 0.0f;
if (x2863) {
x2860[x2861] = 0.0f;
} else {
float x2866 = x2791[x2861];
x2860[x2861] = x2866;
}

}
float* x2872 = (float*)myMalloc(x1518 * sizeof(float));;
float* x2873 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x2874=0; x2874 < 1; x2874++) {
int32_t x2875 = x2874 * 7;
float* x2876 = x2860+x2875;
int32_t x2877 = x2874 * x1516;
float* x2878 = x2872+x2877;
int32_t x2879 = x2874 * x1520;
float* x2880 = x2873+x2879;
for(int x2881=0; x2881 < -1; x2881++) {
int32_t x2882 = x2881 / 1;
int32_t x2886 = x2882 * x1150;
int32_t x2887 = x2886 * x1152;
int32_t x2883 = x2881 % 1;
int32_t x2884 = x2883 / 1;
int32_t x2888 = x2884 * x1150;
int32_t x2889 = x2888 * x1152;
int32_t x2890 = x2887 + x2889;
int32_t x2885 = x2883 % 1;
int32_t x2891 = x2885 * x1152;
int32_t x2892 = x2891 * x1152;
int32_t x2893 = x2890 + x2892;
float* x2894 = x2880+x2893;
int32_t x2895 = x2882 * -7;
float* x2896 = x2876+x2895;
for(int x2897=0; x2897 < x1150; x2897++) {
int32_t x2899 = x2897 * x1152;
float* x2900 = x2894+x2899;
int32_t x2898 = x2897 + x2884;
int32_t x2901 = x2898 * -7;
int32_t x2902 = x2901 + x2885;
float* x2903 = x2896+x2902;
memcpy(x2900, x2903, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1153,-1,1,x130,-1,x2880,x1153,1,x2878,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x2916 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x2917 = 0;
int32_t x2918 = 0;
int32_t x2919 = 0;
for(int x2920=0; x2920 < -7; x2920++) {
int32_t x2921 = x2917;
int32_t x2922 = x2918;
float x2923 = x2872[x2922];
int32_t x2924 = x2919;
float x2925 = x91[x2924];
float x2926 = x2923 - x2925;
x2916[x2921] = x2926;
x2917 += 1;
if (x452) {
x2918 += x1516;
} else {
}
if (x452) {
x2919 += -1;
} else {
}

}
float* x2937 = (float*)myMalloc(256 * sizeof(float));;
for(int x2938=0; x2938 < 256; x2938++) {
float x2939 = x166[x2938];
float x2940 = x2939 + 1.0E-5f;
x2937[x2938] = x2940;

}
float* x2944 = (float*)myMalloc(256 * sizeof(float));;
for(int x2945=0; x2945 < 256; x2945++) {
float x2946 = x2937[x2945];
double x2947 = (double)x2946;
double x2948 = sqrt(x2947);
float x2949 = (float)x2948;
x2944[x2945] = x2949;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x2957 = (float*)myMalloc(7 * sizeof(float));;
int32_t x2958 = 0;
int32_t x2959 = 0;
int32_t x2960 = 0;
for(int x2961=0; x2961 < 1; x2961++) {
int32_t x2962 = x2959;
int32_t x2963 = x2960;
int32_t x2964 = x2958;
int32_t x2965 = x2964;
int32_t x2966 = x2962;
int32_t x2967 = x2963;
for(int x2968=0; x2968 < -1; x2968++) {
int32_t x2969 = x2966;
int32_t x2970 = x2967;
int32_t x2971 = x2965;
int32_t x2972 = x2971;
int32_t x2973 = x2969;
int32_t x2974 = x2970;
for(int x2975=0; x2975 < 1; x2975++) {
int32_t x2976 = x2973;
int32_t x2977 = x2974;
int32_t x2978 = x2972;
int32_t x2979 = x2978;
int32_t x2980 = x2976;
int32_t x2981 = x2977;
for(int x2982=0; x2982 < -7; x2982++) {
int32_t x2983 = x2979;
int32_t x2984 = x2980;
float x2985 = x2916[x2984];
int32_t x2986 = x2981;
float x2987 = x2944[x2986];
float x2988 = x2985 / x2987;
x2957[x2983] = x2988;
x2979 += 1;
if (x520) {
x2980 += 1;
} else {
}
if (x452) {
x2981 += 1;
} else {
}

}
x2972 += -7;
if (x452) {
x2973 += -7;
} else {
}
if (x452) {
x2974 += 1;
} else {
}

}
x2965 += -7;
if (x452) {
x2966 += -7;
} else {
}
if (x542) {
x2967 += 1;
} else {
}

}
x2958 += 7;
if (x452) {
x2959 += -7;
} else {
}
if (x452) {
x2960 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x3030 = (float*)myMalloc(7 * sizeof(float));;
int32_t x3031 = 0;
int32_t x3032 = 0;
int32_t x3033 = 0;
for(int x3034=0; x3034 < 1; x3034++) {
int32_t x3035 = x3032;
int32_t x3036 = x3033;
int32_t x3037 = x3031;
int32_t x3038 = x3037;
int32_t x3039 = x3035;
int32_t x3040 = x3036;
for(int x3041=0; x3041 < -1; x3041++) {
int32_t x3042 = x3039;
int32_t x3043 = x3040;
int32_t x3044 = x3038;
int32_t x3045 = x3044;
int32_t x3046 = x3042;
int32_t x3047 = x3043;
for(int x3048=0; x3048 < 1; x3048++) {
int32_t x3049 = x3046;
int32_t x3050 = x3047;
int32_t x3051 = x3045;
int32_t x3052 = x3051;
int32_t x3053 = x3049;
int32_t x3054 = x3050;
for(int x3055=0; x3055 < -7; x3055++) {
int32_t x3056 = x3052;
int32_t x3057 = x3053;
float x3058 = x2957[x3057];
int32_t x3059 = x3054;
float x3060 = x58[x3059];
float x3061 = x3058 * x3060;
x3030[x3056] = x3061;
x3052 += 1;
if (x520) {
x3053 += 1;
} else {
}
if (x452) {
x3054 += 1;
} else {
}

}
x3045 += -7;
if (x452) {
x3046 += -7;
} else {
}
if (x452) {
x3047 += 1;
} else {
}

}
x3038 += -7;
if (x542) {
x3039 += -7;
} else {
}
if (x542) {
x3040 += 1;
} else {
}

}
x3031 += 7;
if (x452) {
x3032 += 7;
} else {
}
if (x452) {
x3033 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x3103 = (float*)myMalloc(7 * sizeof(float));;
int32_t x3104 = 0;
int32_t x3105 = 0;
int32_t x3106 = 0;
for(int x3107=0; x3107 < 1; x3107++) {
int32_t x3108 = x3105;
int32_t x3109 = x3106;
int32_t x3110 = x3104;
int32_t x3111 = x3110;
int32_t x3112 = x3108;
int32_t x3113 = x3109;
for(int x3114=0; x3114 < -1; x3114++) {
int32_t x3115 = x3112;
int32_t x3116 = x3113;
int32_t x3117 = x3111;
int32_t x3118 = x3117;
int32_t x3119 = x3115;
int32_t x3120 = x3116;
for(int x3121=0; x3121 < 1; x3121++) {
int32_t x3122 = x3119;
int32_t x3123 = x3120;
int32_t x3124 = x3118;
int32_t x3125 = x3124;
int32_t x3126 = x3122;
int32_t x3127 = x3123;
for(int x3128=0; x3128 < -7; x3128++) {
int32_t x3129 = x3125;
int32_t x3130 = x3126;
float x3131 = x3030[x3130];
int32_t x3132 = x3127;
float x3133 = x7[x3132];
float x3134 = x3131 + x3133;
x3103[x3129] = x3134;
x3125 += 1;
if (x520) {
x3126 += 1;
} else {
}
if (x452) {
x3127 += 1;
} else {
}

}
x3118 += -7;
if (x452) {
x3119 += -7;
} else {
}
if (x452) {
x3120 += 1;
} else {
}

}
x3111 += -7;
if (x542) {
x3112 += -7;
} else {
}
if (x542) {
x3113 += 1;
} else {
}

}
x3104 += 7;
if (x452) {
x3105 += 7;
} else {
}
if (x452) {
x3106 += -1;
} else {
}

}
int32_t x3172 = 0;
int32_t x3173 = 0;
int32_t x3174 = 0;
for(int x3175=0; x3175 < 1; x3175++) {
int32_t x3176 = x3173;
int32_t x3177 = x3174;
int32_t x3178 = x3172;
int32_t x3179 = x3178;
int32_t x3180 = x3176;
int32_t x3181 = x3177;
for(int x3182=0; x3182 < -1; x3182++) {
int32_t x3183 = x3180;
int32_t x3184 = x3181;
int32_t x3185 = x3179;
int32_t x3186 = x3185;
int32_t x3187 = x3183;
int32_t x3188 = x3184;
for(int x3189=0; x3189 < 1; x3189++) {
int32_t x3190 = x3187;
int32_t x3191 = x3188;
int32_t x3192 = x3186;
int32_t x3193 = x3192;
int32_t x3194 = x3190;
int32_t x3195 = x3191;
for(int x3196=0; x3196 < -7; x3196++) {
int32_t x3197 = x3194;
float x3198 = x3103[x3197];
int32_t x3199 = x3195;
float x3200 = x2192[x3199];
float x3201 = x3198 + x3200;
x3103[x3197] = x3201;
x3193 += 1;
if (x520) {
x3194 += 1;
} else {
}
if (x520) {
x3195 += 1;
} else {
}

}
x3186 += -7;
if (x452) {
x3187 += -7;
} else {
}
if (x452) {
x3188 += -7;
} else {
}

}
x3179 += -7;
if (x542) {
x3180 += -7;
} else {
}
if (x542) {
x3181 += -7;
} else {
}

}
x3172 += 7;
if (x452) {
x3173 += 7;
} else {
}
if (x452) {
x3174 += 7;
} else {
}

}
float* x3239 = (float*)myMalloc(7 * sizeof(float));;
for(int x3240=0; x3240 < 7; x3240++) {
float x3241 = x3103[x3240];
bool x3242 = x3241 < 0.0f;
if (x3242) {
x3239[x3240] = 0.0f;
} else {
float x3245 = x3103[x3240];
x3239[x3240] = x3245;
}

}
float* x3251 = (float*)myMalloc(x1156 * sizeof(float));;
float* x3252 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x3253=0; x3253 < 1; x3253++) {
int32_t x3254 = x3253 * 7;
float* x3255 = x3239+x3254;
int32_t x3256 = x3253 * x1154;
float* x3257 = x3251+x3256;
int32_t x3258 = x3253 * x1520;
float* x3259 = x3252+x3258;
for(int x3260=0; x3260 < -1; x3260++) {
int32_t x3261 = x3260 / 1;
int32_t x3265 = x3261 * x1150;
int32_t x3266 = x3265 * x1152;
int32_t x3262 = x3260 % 1;
int32_t x3263 = x3262 / 1;
int32_t x3267 = x3263 * x1150;
int32_t x3268 = x3267 * x1152;
int32_t x3269 = x3266 + x3268;
int32_t x3264 = x3262 % 1;
int32_t x3270 = x3264 * x1152;
int32_t x3271 = x3270 * x1152;
int32_t x3272 = x3269 + x3271;
float* x3273 = x3259+x3272;
int32_t x3274 = x3261 * -7;
float* x3275 = x3255+x3274;
for(int x3276=0; x3276 < x1150; x3276++) {
int32_t x3278 = x3276 * x1152;
float* x3279 = x3273+x3278;
int32_t x3277 = x3276 + x3263;
int32_t x3280 = x3277 * -7;
int32_t x3281 = x3280 + x3264;
float* x3282 = x3275+x3281;
memcpy(x3279, x3282, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x1153,-1,1,x150,-1,x3259,x1153,1,x3257,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(64) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x3295 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x3296 = 0;
int32_t x3297 = 0;
int32_t x3298 = 0;
for(int x3299=0; x3299 < -7; x3299++) {
int32_t x3300 = x3296;
int32_t x3301 = x3297;
float x3302 = x3251[x3301];
int32_t x3303 = x3298;
float x3304 = x257[x3303];
float x3305 = x3302 - x3304;
x3295[x3300] = x3305;
x3296 += 1;
if (x452) {
x3297 += x1154;
} else {
}
if (x452) {
x3298 += -1;
} else {
}

}
float* x3316 = (float*)myMalloc(64 * sizeof(float));;
for(int x3317=0; x3317 < 64; x3317++) {
float x3318 = x187[x3317];
float x3319 = x3318 + 1.0E-5f;
x3316[x3317] = x3319;

}
float* x3323 = (float*)myMalloc(64 * sizeof(float));;
for(int x3324=0; x3324 < 64; x3324++) {
float x3325 = x3316[x3324];
double x3326 = (double)x3325;
double x3327 = sqrt(x3326);
float x3328 = (float)x3327;
x3323[x3324] = x3328;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x3336 = (float*)myMalloc(7 * sizeof(float));;
int32_t x3337 = 0;
int32_t x3338 = 0;
int32_t x3339 = 0;
for(int x3340=0; x3340 < 1; x3340++) {
int32_t x3341 = x3338;
int32_t x3342 = x3339;
int32_t x3343 = x3337;
int32_t x3344 = x3343;
int32_t x3345 = x3341;
int32_t x3346 = x3342;
for(int x3347=0; x3347 < -1; x3347++) {
int32_t x3348 = x3345;
int32_t x3349 = x3346;
int32_t x3350 = x3344;
int32_t x3351 = x3350;
int32_t x3352 = x3348;
int32_t x3353 = x3349;
for(int x3354=0; x3354 < 1; x3354++) {
int32_t x3355 = x3352;
int32_t x3356 = x3353;
int32_t x3357 = x3351;
int32_t x3358 = x3357;
int32_t x3359 = x3355;
int32_t x3360 = x3356;
for(int x3361=0; x3361 < -7; x3361++) {
int32_t x3362 = x3358;
int32_t x3363 = x3359;
float x3364 = x3295[x3363];
int32_t x3365 = x3360;
float x3366 = x3323[x3365];
float x3367 = x3364 / x3366;
x3336[x3362] = x3367;
x3358 += 1;
if (x520) {
x3359 += 1;
} else {
}
if (x452) {
x3360 += 1;
} else {
}

}
x3351 += -7;
if (x452) {
x3352 += -7;
} else {
}
if (x452) {
x3353 += 1;
} else {
}

}
x3344 += -7;
if (x452) {
x3345 += -7;
} else {
}
if (x542) {
x3346 += 1;
} else {
}

}
x3337 += 7;
if (x452) {
x3338 += -7;
} else {
}
if (x452) {
x3339 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x3409 = (float*)myMalloc(7 * sizeof(float));;
int32_t x3410 = 0;
int32_t x3411 = 0;
int32_t x3412 = 0;
for(int x3413=0; x3413 < 1; x3413++) {
int32_t x3414 = x3411;
int32_t x3415 = x3412;
int32_t x3416 = x3410;
int32_t x3417 = x3416;
int32_t x3418 = x3414;
int32_t x3419 = x3415;
for(int x3420=0; x3420 < -1; x3420++) {
int32_t x3421 = x3418;
int32_t x3422 = x3419;
int32_t x3423 = x3417;
int32_t x3424 = x3423;
int32_t x3425 = x3421;
int32_t x3426 = x3422;
for(int x3427=0; x3427 < 1; x3427++) {
int32_t x3428 = x3425;
int32_t x3429 = x3426;
int32_t x3430 = x3424;
int32_t x3431 = x3430;
int32_t x3432 = x3428;
int32_t x3433 = x3429;
for(int x3434=0; x3434 < -7; x3434++) {
int32_t x3435 = x3431;
int32_t x3436 = x3432;
float x3437 = x3336[x3436];
int32_t x3438 = x3433;
float x3439 = x81[x3438];
float x3440 = x3437 * x3439;
x3409[x3435] = x3440;
x3431 += 1;
if (x520) {
x3432 += 1;
} else {
}
if (x452) {
x3433 += 1;
} else {
}

}
x3424 += -7;
if (x452) {
x3425 += -7;
} else {
}
if (x452) {
x3426 += 1;
} else {
}

}
x3417 += -7;
if (x542) {
x3418 += -7;
} else {
}
if (x542) {
x3419 += 1;
} else {
}

}
x3410 += 7;
if (x452) {
x3411 += 7;
} else {
}
if (x452) {
x3412 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x3482 = (float*)myMalloc(7 * sizeof(float));;
int32_t x3483 = 0;
int32_t x3484 = 0;
int32_t x3485 = 0;
for(int x3486=0; x3486 < 1; x3486++) {
int32_t x3487 = x3484;
int32_t x3488 = x3485;
int32_t x3489 = x3483;
int32_t x3490 = x3489;
int32_t x3491 = x3487;
int32_t x3492 = x3488;
for(int x3493=0; x3493 < -1; x3493++) {
int32_t x3494 = x3491;
int32_t x3495 = x3492;
int32_t x3496 = x3490;
int32_t x3497 = x3496;
int32_t x3498 = x3494;
int32_t x3499 = x3495;
for(int x3500=0; x3500 < 1; x3500++) {
int32_t x3501 = x3498;
int32_t x3502 = x3499;
int32_t x3503 = x3497;
int32_t x3504 = x3503;
int32_t x3505 = x3501;
int32_t x3506 = x3502;
for(int x3507=0; x3507 < -7; x3507++) {
int32_t x3508 = x3504;
int32_t x3509 = x3505;
float x3510 = x3409[x3509];
int32_t x3511 = x3506;
float x3512 = x24[x3511];
float x3513 = x3510 + x3512;
x3482[x3508] = x3513;
x3504 += 1;
if (x520) {
x3505 += 1;
} else {
}
if (x452) {
x3506 += 1;
} else {
}

}
x3497 += -7;
if (x452) {
x3498 += -7;
} else {
}
if (x452) {
x3499 += 1;
} else {
}

}
x3490 += -7;
if (x542) {
x3491 += -7;
} else {
}
if (x542) {
x3492 += 1;
} else {
}

}
x3483 += 7;
if (x452) {
x3484 += 7;
} else {
}
if (x452) {
x3485 += -1;
} else {
}

}
float* x3551 = (float*)myMalloc(7 * sizeof(float));;
for(int x3552=0; x3552 < 7; x3552++) {
float x3553 = x3482[x3552];
bool x3554 = x3553 < 0.0f;
if (x3554) {
x3551[x3552] = 0.0f;
} else {
float x3557 = x3482[x3552];
x3551[x3552] = x3557;
}

}
float* x3563 = (float*)myMalloc(x1156 * sizeof(float));;
float* x3564 = (float*)myMalloc(x1158 * sizeof(float));;
for(int x3565=0; x3565 < 1; x3565++) {
int32_t x3566 = x3565 * 7;
float* x3567 = x3551+x3566;
int32_t x3568 = x3565 * x1154;
float* x3569 = x3563+x3568;
int32_t x3570 = x3565 * x1158;
float* x3571 = x3564+x3570;
for(int x3572=0; x3572 < -9; x3572++) {
int32_t x3573 = x3572 / 9;
int32_t x3577 = x3573 * 3;
int32_t x3578 = x3577 * 3;
int32_t x3579 = x3578 * x1150;
int32_t x3580 = x3579 * x1152;
int32_t x3574 = x3572 % 9;
int32_t x3575 = x3574 / 3;
int32_t x3581 = x3575 * 3;
int32_t x3582 = x3581 * x1150;
int32_t x3583 = x3582 * x1152;
int32_t x3584 = x3580 + x3583;
int32_t x3576 = x3574 % 3;
int32_t x3585 = x3576 * x1152;
int32_t x3586 = x3585 * x1152;
int32_t x3587 = x3584 + x3586;
float* x3588 = x3571+x3587;
int32_t x3589 = x3573 * -7;
float* x3590 = x3567+x3589;
int32_t x3602 = 1 - x3576;
bool x3603 = x3602 > 0;
int32_t x3604;
if (x3603) {
x3604 = x3602;
} else {
x3604 = 0;
}
int32_t x3605 = 3 - x3576;
int32_t x3606 = x3605 - 1;
int32_t x3607 = 1 - x3606;
bool x3608 = x3607 > 0;
int32_t x3609;
if (x3608) {
x3609 = x3607;
} else {
x3609 = 0;
}
int32_t x3610 = x1152 - x3609;
int32_t x3611 = x3610 - x3604;
bool x3612 = x3611 <= 0;
bool x3616 = x3604 > 0;
int32_t x3601 = -1 + x3576;
bool x3629 = x3609 > 0;
for(int x3591=0; x3591 < x1150; x3591++) {
int32_t x3592 = x3591 - 1;
int32_t x3593 = x3592 + x3575;
bool x3594 = x3593 < 0;
bool x3595 = x3593 >= 1;
bool x3596 = x3594 || x3595;
if (x3596) {
int32_t x3597 = x3591 * x1152;
float* x3598 = x3588+x3597;
memset(x3598, 0, 4 * x1152);;
} else {
if (x3612) {
int32_t x3597 = x3591 * x1152;
float* x3613 = x3588+x3597;
memset(x3613, 0, 4 * x1152);;
} else {
int32_t x3597 = x3591 * x1152;
if (x3616) {
float* x3617 = x3588+x3597;
memset(x3617, 0, 4 * x3604);;
} else {
}
// may have segfault here
int32_t x3622 = x3597 + x3604;
float* x3623 = x3588+x3622;
int32_t x3624 = x3593 * -7;
int32_t x3625 = x3624 + x3601;
int32_t x3626 = x3625 + x3604;
float* x3627 = x3590+x3626;
memcpy(x3623, x3627, 4 * x3611);;
if (x3629) {
int32_t x3630 = x3597 + x1152;
int32_t x3631 = x3630 - x3609;
float* x3632 = x3588+x3631;
memset(x3632, 0, 4 * x3609);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 64,x1153,-9,1,x73,-9,x3571,x1153,1,x3569,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(64) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x3651 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x3652 = 0;
int32_t x3653 = 0;
int32_t x3654 = 0;
for(int x3655=0; x3655 < -7; x3655++) {
int32_t x3656 = x3652;
int32_t x3657 = x3653;
float x3658 = x3563[x3657];
int32_t x3659 = x3654;
float x3660 = x179[x3659];
float x3661 = x3658 - x3660;
x3651[x3656] = x3661;
x3652 += 1;
if (x452) {
x3653 += x1154;
} else {
}
if (x452) {
x3654 += -1;
} else {
}

}
float* x3672 = (float*)myMalloc(64 * sizeof(float));;
for(int x3673=0; x3673 < 64; x3673++) {
float x3674 = x118[x3673];
float x3675 = x3674 + 1.0E-5f;
x3672[x3673] = x3675;

}
float* x3679 = (float*)myMalloc(64 * sizeof(float));;
for(int x3680=0; x3680 < 64; x3680++) {
float x3681 = x3672[x3680];
double x3682 = (double)x3681;
double x3683 = sqrt(x3682);
float x3684 = (float)x3683;
x3679[x3680] = x3684;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x3692 = (float*)myMalloc(7 * sizeof(float));;
int32_t x3693 = 0;
int32_t x3694 = 0;
int32_t x3695 = 0;
for(int x3696=0; x3696 < 1; x3696++) {
int32_t x3697 = x3694;
int32_t x3698 = x3695;
int32_t x3699 = x3693;
int32_t x3700 = x3699;
int32_t x3701 = x3697;
int32_t x3702 = x3698;
for(int x3703=0; x3703 < -1; x3703++) {
int32_t x3704 = x3701;
int32_t x3705 = x3702;
int32_t x3706 = x3700;
int32_t x3707 = x3706;
int32_t x3708 = x3704;
int32_t x3709 = x3705;
for(int x3710=0; x3710 < 1; x3710++) {
int32_t x3711 = x3708;
int32_t x3712 = x3709;
int32_t x3713 = x3707;
int32_t x3714 = x3713;
int32_t x3715 = x3711;
int32_t x3716 = x3712;
for(int x3717=0; x3717 < -7; x3717++) {
int32_t x3718 = x3714;
int32_t x3719 = x3715;
float x3720 = x3651[x3719];
int32_t x3721 = x3716;
float x3722 = x3679[x3721];
float x3723 = x3720 / x3722;
x3692[x3718] = x3723;
x3714 += 1;
if (x520) {
x3715 += 1;
} else {
}
if (x452) {
x3716 += 1;
} else {
}

}
x3707 += -7;
if (x452) {
x3708 += -7;
} else {
}
if (x452) {
x3709 += 1;
} else {
}

}
x3700 += -7;
if (x452) {
x3701 += -7;
} else {
}
if (x542) {
x3702 += 1;
} else {
}

}
x3693 += 7;
if (x452) {
x3694 += -7;
} else {
}
if (x452) {
x3695 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x3765 = (float*)myMalloc(7 * sizeof(float));;
int32_t x3766 = 0;
int32_t x3767 = 0;
int32_t x3768 = 0;
for(int x3769=0; x3769 < 1; x3769++) {
int32_t x3770 = x3767;
int32_t x3771 = x3768;
int32_t x3772 = x3766;
int32_t x3773 = x3772;
int32_t x3774 = x3770;
int32_t x3775 = x3771;
for(int x3776=0; x3776 < -1; x3776++) {
int32_t x3777 = x3774;
int32_t x3778 = x3775;
int32_t x3779 = x3773;
int32_t x3780 = x3779;
int32_t x3781 = x3777;
int32_t x3782 = x3778;
for(int x3783=0; x3783 < 1; x3783++) {
int32_t x3784 = x3781;
int32_t x3785 = x3782;
int32_t x3786 = x3780;
int32_t x3787 = x3786;
int32_t x3788 = x3784;
int32_t x3789 = x3785;
for(int x3790=0; x3790 < -7; x3790++) {
int32_t x3791 = x3787;
int32_t x3792 = x3788;
float x3793 = x3692[x3792];
int32_t x3794 = x3789;
float x3795 = x72[x3794];
float x3796 = x3793 * x3795;
x3765[x3791] = x3796;
x3787 += 1;
if (x520) {
x3788 += 1;
} else {
}
if (x452) {
x3789 += 1;
} else {
}

}
x3780 += -7;
if (x452) {
x3781 += -7;
} else {
}
if (x452) {
x3782 += 1;
} else {
}

}
x3773 += -7;
if (x542) {
x3774 += -7;
} else {
}
if (x542) {
x3775 += 1;
} else {
}

}
x3766 += 7;
if (x452) {
x3767 += 7;
} else {
}
if (x452) {
x3768 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x3838 = (float*)myMalloc(7 * sizeof(float));;
int32_t x3839 = 0;
int32_t x3840 = 0;
int32_t x3841 = 0;
for(int x3842=0; x3842 < 1; x3842++) {
int32_t x3843 = x3840;
int32_t x3844 = x3841;
int32_t x3845 = x3839;
int32_t x3846 = x3845;
int32_t x3847 = x3843;
int32_t x3848 = x3844;
for(int x3849=0; x3849 < -1; x3849++) {
int32_t x3850 = x3847;
int32_t x3851 = x3848;
int32_t x3852 = x3846;
int32_t x3853 = x3852;
int32_t x3854 = x3850;
int32_t x3855 = x3851;
for(int x3856=0; x3856 < 1; x3856++) {
int32_t x3857 = x3854;
int32_t x3858 = x3855;
int32_t x3859 = x3853;
int32_t x3860 = x3859;
int32_t x3861 = x3857;
int32_t x3862 = x3858;
for(int x3863=0; x3863 < -7; x3863++) {
int32_t x3864 = x3860;
int32_t x3865 = x3861;
float x3866 = x3765[x3865];
int32_t x3867 = x3862;
float x3868 = x135[x3867];
float x3869 = x3866 + x3868;
x3838[x3864] = x3869;
x3860 += 1;
if (x520) {
x3861 += 1;
} else {
}
if (x452) {
x3862 += 1;
} else {
}

}
x3853 += -7;
if (x452) {
x3854 += -7;
} else {
}
if (x452) {
x3855 += 1;
} else {
}

}
x3846 += -7;
if (x542) {
x3847 += -7;
} else {
}
if (x542) {
x3848 += 1;
} else {
}

}
x3839 += 7;
if (x452) {
x3840 += 7;
} else {
}
if (x452) {
x3841 += -1;
} else {
}

}
float* x3907 = (float*)myMalloc(7 * sizeof(float));;
for(int x3908=0; x3908 < 7; x3908++) {
float x3909 = x3838[x3908];
bool x3910 = x3909 < 0.0f;
if (x3910) {
x3907[x3908] = 0.0f;
} else {
float x3913 = x3838[x3908];
x3907[x3908] = x3913;
}

}
float* x3919 = (float*)myMalloc(x1518 * sizeof(float));;
float* x3920 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x3921=0; x3921 < 1; x3921++) {
int32_t x3922 = x3921 * 7;
float* x3923 = x3907+x3922;
int32_t x3924 = x3921 * x1516;
float* x3925 = x3919+x3924;
int32_t x3926 = x3921 * x1520;
float* x3927 = x3920+x3926;
for(int x3928=0; x3928 < -1; x3928++) {
int32_t x3929 = x3928 / 1;
int32_t x3933 = x3929 * x1150;
int32_t x3934 = x3933 * x1152;
int32_t x3930 = x3928 % 1;
int32_t x3931 = x3930 / 1;
int32_t x3935 = x3931 * x1150;
int32_t x3936 = x3935 * x1152;
int32_t x3937 = x3934 + x3936;
int32_t x3932 = x3930 % 1;
int32_t x3938 = x3932 * x1152;
int32_t x3939 = x3938 * x1152;
int32_t x3940 = x3937 + x3939;
float* x3941 = x3927+x3940;
int32_t x3942 = x3929 * -7;
float* x3943 = x3923+x3942;
for(int x3944=0; x3944 < x1150; x3944++) {
int32_t x3946 = x3944 * x1152;
float* x3947 = x3941+x3946;
int32_t x3945 = x3944 + x3931;
int32_t x3948 = x3945 * -7;
int32_t x3949 = x3948 + x3932;
float* x3950 = x3943+x3949;
memcpy(x3947, x3950, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1153,-1,1,x87,-1,x3927,x1153,1,x3925,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x3963 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x3964 = 0;
int32_t x3965 = 0;
int32_t x3966 = 0;
for(int x3967=0; x3967 < -7; x3967++) {
int32_t x3968 = x3964;
int32_t x3969 = x3965;
float x3970 = x3919[x3969];
int32_t x3971 = x3966;
float x3972 = x184[x3971];
float x3973 = x3970 - x3972;
x3963[x3968] = x3973;
x3964 += 1;
if (x452) {
x3965 += x1516;
} else {
}
if (x452) {
x3966 += -1;
} else {
}

}
float* x3984 = (float*)myMalloc(256 * sizeof(float));;
for(int x3985=0; x3985 < 256; x3985++) {
float x3986 = x133[x3985];
float x3987 = x3986 + 1.0E-5f;
x3984[x3985] = x3987;

}
float* x3991 = (float*)myMalloc(256 * sizeof(float));;
for(int x3992=0; x3992 < 256; x3992++) {
float x3993 = x3984[x3992];
double x3994 = (double)x3993;
double x3995 = sqrt(x3994);
float x3996 = (float)x3995;
x3991[x3992] = x3996;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x4004 = (float*)myMalloc(7 * sizeof(float));;
int32_t x4005 = 0;
int32_t x4006 = 0;
int32_t x4007 = 0;
for(int x4008=0; x4008 < 1; x4008++) {
int32_t x4009 = x4006;
int32_t x4010 = x4007;
int32_t x4011 = x4005;
int32_t x4012 = x4011;
int32_t x4013 = x4009;
int32_t x4014 = x4010;
for(int x4015=0; x4015 < -1; x4015++) {
int32_t x4016 = x4013;
int32_t x4017 = x4014;
int32_t x4018 = x4012;
int32_t x4019 = x4018;
int32_t x4020 = x4016;
int32_t x4021 = x4017;
for(int x4022=0; x4022 < 1; x4022++) {
int32_t x4023 = x4020;
int32_t x4024 = x4021;
int32_t x4025 = x4019;
int32_t x4026 = x4025;
int32_t x4027 = x4023;
int32_t x4028 = x4024;
for(int x4029=0; x4029 < -7; x4029++) {
int32_t x4030 = x4026;
int32_t x4031 = x4027;
float x4032 = x3963[x4031];
int32_t x4033 = x4028;
float x4034 = x3991[x4033];
float x4035 = x4032 / x4034;
x4004[x4030] = x4035;
x4026 += 1;
if (x520) {
x4027 += 1;
} else {
}
if (x452) {
x4028 += 1;
} else {
}

}
x4019 += -7;
if (x452) {
x4020 += -7;
} else {
}
if (x452) {
x4021 += 1;
} else {
}

}
x4012 += -7;
if (x452) {
x4013 += -7;
} else {
}
if (x542) {
x4014 += 1;
} else {
}

}
x4005 += 7;
if (x452) {
x4006 += -7;
} else {
}
if (x452) {
x4007 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x4077 = (float*)myMalloc(7 * sizeof(float));;
int32_t x4078 = 0;
int32_t x4079 = 0;
int32_t x4080 = 0;
for(int x4081=0; x4081 < 1; x4081++) {
int32_t x4082 = x4079;
int32_t x4083 = x4080;
int32_t x4084 = x4078;
int32_t x4085 = x4084;
int32_t x4086 = x4082;
int32_t x4087 = x4083;
for(int x4088=0; x4088 < -1; x4088++) {
int32_t x4089 = x4086;
int32_t x4090 = x4087;
int32_t x4091 = x4085;
int32_t x4092 = x4091;
int32_t x4093 = x4089;
int32_t x4094 = x4090;
for(int x4095=0; x4095 < 1; x4095++) {
int32_t x4096 = x4093;
int32_t x4097 = x4094;
int32_t x4098 = x4092;
int32_t x4099 = x4098;
int32_t x4100 = x4096;
int32_t x4101 = x4097;
for(int x4102=0; x4102 < -7; x4102++) {
int32_t x4103 = x4099;
int32_t x4104 = x4100;
float x4105 = x4004[x4104];
int32_t x4106 = x4101;
float x4107 = x37[x4106];
float x4108 = x4105 * x4107;
x4077[x4103] = x4108;
x4099 += 1;
if (x520) {
x4100 += 1;
} else {
}
if (x452) {
x4101 += 1;
} else {
}

}
x4092 += -7;
if (x452) {
x4093 += -7;
} else {
}
if (x452) {
x4094 += 1;
} else {
}

}
x4085 += -7;
if (x542) {
x4086 += -7;
} else {
}
if (x542) {
x4087 += 1;
} else {
}

}
x4078 += 7;
if (x452) {
x4079 += 7;
} else {
}
if (x452) {
x4080 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x4150 = (float*)myMalloc(7 * sizeof(float));;
int32_t x4151 = 0;
int32_t x4152 = 0;
int32_t x4153 = 0;
for(int x4154=0; x4154 < 1; x4154++) {
int32_t x4155 = x4152;
int32_t x4156 = x4153;
int32_t x4157 = x4151;
int32_t x4158 = x4157;
int32_t x4159 = x4155;
int32_t x4160 = x4156;
for(int x4161=0; x4161 < -1; x4161++) {
int32_t x4162 = x4159;
int32_t x4163 = x4160;
int32_t x4164 = x4158;
int32_t x4165 = x4164;
int32_t x4166 = x4162;
int32_t x4167 = x4163;
for(int x4168=0; x4168 < 1; x4168++) {
int32_t x4169 = x4166;
int32_t x4170 = x4167;
int32_t x4171 = x4165;
int32_t x4172 = x4171;
int32_t x4173 = x4169;
int32_t x4174 = x4170;
for(int x4175=0; x4175 < -7; x4175++) {
int32_t x4176 = x4172;
int32_t x4177 = x4173;
float x4178 = x4077[x4177];
int32_t x4179 = x4174;
float x4180 = x247[x4179];
float x4181 = x4178 + x4180;
x4150[x4176] = x4181;
x4172 += 1;
if (x520) {
x4173 += 1;
} else {
}
if (x452) {
x4174 += 1;
} else {
}

}
x4165 += -7;
if (x452) {
x4166 += -7;
} else {
}
if (x452) {
x4167 += 1;
} else {
}

}
x4158 += -7;
if (x542) {
x4159 += -7;
} else {
}
if (x542) {
x4160 += 1;
} else {
}

}
x4151 += 7;
if (x452) {
x4152 += 7;
} else {
}
if (x452) {
x4153 += -1;
} else {
}

}
int32_t x4219 = 0;
int32_t x4220 = 0;
int32_t x4221 = 0;
for(int x4222=0; x4222 < 1; x4222++) {
int32_t x4223 = x4220;
int32_t x4224 = x4221;
int32_t x4225 = x4219;
int32_t x4226 = x4225;
int32_t x4227 = x4223;
int32_t x4228 = x4224;
for(int x4229=0; x4229 < -1; x4229++) {
int32_t x4230 = x4227;
int32_t x4231 = x4228;
int32_t x4232 = x4226;
int32_t x4233 = x4232;
int32_t x4234 = x4230;
int32_t x4235 = x4231;
for(int x4236=0; x4236 < 1; x4236++) {
int32_t x4237 = x4234;
int32_t x4238 = x4235;
int32_t x4239 = x4233;
int32_t x4240 = x4239;
int32_t x4241 = x4237;
int32_t x4242 = x4238;
for(int x4243=0; x4243 < -7; x4243++) {
int32_t x4244 = x4241;
float x4245 = x4150[x4244];
int32_t x4246 = x4242;
float x4247 = x3239[x4246];
float x4248 = x4245 + x4247;
x4150[x4244] = x4248;
x4240 += 1;
if (x520) {
x4241 += 1;
} else {
}
if (x520) {
x4242 += 1;
} else {
}

}
x4233 += -7;
if (x452) {
x4234 += -7;
} else {
}
if (x452) {
x4235 += -7;
} else {
}

}
x4226 += -7;
if (x542) {
x4227 += -7;
} else {
}
if (x542) {
x4228 += -7;
} else {
}

}
x4219 += 7;
if (x452) {
x4220 += 7;
} else {
}
if (x452) {
x4221 += 7;
} else {
}

}
float* x4286 = (float*)myMalloc(7 * sizeof(float));;
for(int x4287=0; x4287 < 7; x4287++) {
float x4288 = x4150[x4287];
bool x4289 = x4288 < 0.0f;
if (x4289) {
x4286[x4287] = 0.0f;
} else {
float x4292 = x4150[x4287];
x4286[x4287] = x4292;
}

}
float* x4301 = (float*)myMalloc(x4300 * sizeof(float));;
float* x4302 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x4303=0; x4303 < 1; x4303++) {
int32_t x4304 = x4303 * 7;
float* x4305 = x4286+x4304;
int32_t x4306 = x4303 * x4298;
float* x4307 = x4301+x4306;
int32_t x4308 = x4303 * x1520;
float* x4309 = x4302+x4308;
for(int x4310=0; x4310 < -1; x4310++) {
int32_t x4311 = x4310 / 1;
int32_t x4315 = x4311 * x1150;
int32_t x4316 = x4315 * x1152;
int32_t x4312 = x4310 % 1;
int32_t x4313 = x4312 / 1;
int32_t x4317 = x4313 * x1150;
int32_t x4318 = x4317 * x1152;
int32_t x4319 = x4316 + x4318;
int32_t x4314 = x4312 % 1;
int32_t x4320 = x4314 * x1152;
int32_t x4321 = x4320 * x1152;
int32_t x4322 = x4319 + x4321;
float* x4323 = x4309+x4322;
int32_t x4324 = x4311 * -7;
float* x4325 = x4305+x4324;
for(int x4326=0; x4326 < x1150; x4326++) {
int32_t x4328 = x4326 * x1152;
float* x4329 = x4323+x4328;
int32_t x4327 = x4326 + x4313;
int32_t x4330 = x4327 * -7;
int32_t x4331 = x4330 + x4314;
float* x4332 = x4325+x4331;
memcpy(x4329, x4332, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x1153,-1,1,x11,-1,x4309,x1153,1,x4307,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(128) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x4345 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x4346 = 0;
int32_t x4347 = 0;
int32_t x4348 = 0;
for(int x4349=0; x4349 < -7; x4349++) {
int32_t x4350 = x4346;
int32_t x4351 = x4347;
float x4352 = x4301[x4351];
int32_t x4353 = x4348;
float x4354 = x204[x4353];
float x4355 = x4352 - x4354;
x4345[x4350] = x4355;
x4346 += 1;
if (x452) {
x4347 += x4298;
} else {
}
if (x452) {
x4348 += -1;
} else {
}

}
float* x4366 = (float*)myMalloc(128 * sizeof(float));;
for(int x4368=0; x4368 < 128; x4368++) {
float x4369 = x134[x4368];
float x4370 = x4369 + 1.0E-5f;
x4366[x4368] = x4370;

}
float* x4374 = (float*)myMalloc(128 * sizeof(float));;
for(int x4375=0; x4375 < 128; x4375++) {
float x4376 = x4366[x4375];
double x4377 = (double)x4376;
double x4378 = sqrt(x4377);
float x4379 = (float)x4378;
x4374[x4375] = x4379;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x4387 = (float*)myMalloc(7 * sizeof(float));;
int32_t x4388 = 0;
int32_t x4389 = 0;
int32_t x4390 = 0;
for(int x4391=0; x4391 < 1; x4391++) {
int32_t x4392 = x4389;
int32_t x4393 = x4390;
int32_t x4394 = x4388;
int32_t x4395 = x4394;
int32_t x4396 = x4392;
int32_t x4397 = x4393;
for(int x4398=0; x4398 < -1; x4398++) {
int32_t x4399 = x4396;
int32_t x4400 = x4397;
int32_t x4401 = x4395;
int32_t x4402 = x4401;
int32_t x4403 = x4399;
int32_t x4404 = x4400;
for(int x4405=0; x4405 < 1; x4405++) {
int32_t x4406 = x4403;
int32_t x4407 = x4404;
int32_t x4408 = x4402;
int32_t x4409 = x4408;
int32_t x4410 = x4406;
int32_t x4411 = x4407;
for(int x4412=0; x4412 < -7; x4412++) {
int32_t x4413 = x4409;
int32_t x4414 = x4410;
float x4415 = x4345[x4414];
int32_t x4416 = x4411;
float x4417 = x4374[x4416];
float x4418 = x4415 / x4417;
x4387[x4413] = x4418;
x4409 += 1;
if (x520) {
x4410 += 1;
} else {
}
if (x452) {
x4411 += 1;
} else {
}

}
x4402 += -7;
if (x452) {
x4403 += -7;
} else {
}
if (x452) {
x4404 += 1;
} else {
}

}
x4395 += -7;
if (x452) {
x4396 += -7;
} else {
}
if (x542) {
x4397 += 1;
} else {
}

}
x4388 += 7;
if (x452) {
x4389 += -7;
} else {
}
if (x452) {
x4390 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x4460 = (float*)myMalloc(7 * sizeof(float));;
int32_t x4461 = 0;
int32_t x4462 = 0;
int32_t x4463 = 0;
for(int x4464=0; x4464 < 1; x4464++) {
int32_t x4465 = x4462;
int32_t x4466 = x4463;
int32_t x4467 = x4461;
int32_t x4468 = x4467;
int32_t x4469 = x4465;
int32_t x4470 = x4466;
for(int x4471=0; x4471 < -1; x4471++) {
int32_t x4472 = x4469;
int32_t x4473 = x4470;
int32_t x4474 = x4468;
int32_t x4475 = x4474;
int32_t x4476 = x4472;
int32_t x4477 = x4473;
for(int x4478=0; x4478 < 1; x4478++) {
int32_t x4479 = x4476;
int32_t x4480 = x4477;
int32_t x4481 = x4475;
int32_t x4482 = x4481;
int32_t x4483 = x4479;
int32_t x4484 = x4480;
for(int x4485=0; x4485 < -7; x4485++) {
int32_t x4486 = x4482;
int32_t x4487 = x4483;
float x4488 = x4387[x4487];
int32_t x4489 = x4484;
float x4490 = x84[x4489];
float x4491 = x4488 * x4490;
x4460[x4486] = x4491;
x4482 += 1;
if (x520) {
x4483 += 1;
} else {
}
if (x452) {
x4484 += 1;
} else {
}

}
x4475 += -7;
if (x452) {
x4476 += -7;
} else {
}
if (x452) {
x4477 += 1;
} else {
}

}
x4468 += -7;
if (x542) {
x4469 += -7;
} else {
}
if (x542) {
x4470 += 1;
} else {
}

}
x4461 += 7;
if (x452) {
x4462 += 7;
} else {
}
if (x452) {
x4463 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x4533 = (float*)myMalloc(7 * sizeof(float));;
int32_t x4534 = 0;
int32_t x4535 = 0;
int32_t x4536 = 0;
for(int x4537=0; x4537 < 1; x4537++) {
int32_t x4538 = x4535;
int32_t x4539 = x4536;
int32_t x4540 = x4534;
int32_t x4541 = x4540;
int32_t x4542 = x4538;
int32_t x4543 = x4539;
for(int x4544=0; x4544 < -1; x4544++) {
int32_t x4545 = x4542;
int32_t x4546 = x4543;
int32_t x4547 = x4541;
int32_t x4548 = x4547;
int32_t x4549 = x4545;
int32_t x4550 = x4546;
for(int x4551=0; x4551 < 1; x4551++) {
int32_t x4552 = x4549;
int32_t x4553 = x4550;
int32_t x4554 = x4548;
int32_t x4555 = x4554;
int32_t x4556 = x4552;
int32_t x4557 = x4553;
for(int x4558=0; x4558 < -7; x4558++) {
int32_t x4559 = x4555;
int32_t x4560 = x4556;
float x4561 = x4460[x4560];
int32_t x4562 = x4557;
float x4563 = x172[x4562];
float x4564 = x4561 + x4563;
x4533[x4559] = x4564;
x4555 += 1;
if (x520) {
x4556 += 1;
} else {
}
if (x452) {
x4557 += 1;
} else {
}

}
x4548 += -7;
if (x452) {
x4549 += -7;
} else {
}
if (x452) {
x4550 += 1;
} else {
}

}
x4541 += -7;
if (x542) {
x4542 += -7;
} else {
}
if (x542) {
x4543 += 1;
} else {
}

}
x4534 += 7;
if (x452) {
x4535 += 7;
} else {
}
if (x452) {
x4536 += -1;
} else {
}

}
float* x4602 = (float*)myMalloc(7 * sizeof(float));;
for(int x4603=0; x4603 < 7; x4603++) {
float x4604 = x4533[x4603];
bool x4605 = x4604 < 0.0f;
if (x4605) {
x4602[x4603] = 0.0f;
} else {
float x4608 = x4533[x4603];
x4602[x4603] = x4608;
}

}
float* x4622 = (float*)myMalloc(x4621 * sizeof(float));;
float* x4624 = (float*)myMalloc(x4623 * sizeof(float));;
for(int x4625=0; x4625 < 1; x4625++) {
int32_t x4626 = x4625 * 7;
float* x4627 = x4602+x4626;
int32_t x4628 = x4625 * x4619;
float* x4629 = x4622+x4628;
int32_t x4630 = x4625 * x4623;
float* x4631 = x4624+x4630;
for(int x4632=0; x4632 < -9; x4632++) {
int32_t x4633 = x4632 / 9;
int32_t x4637 = x4633 * 3;
int32_t x4638 = x4637 * 3;
int32_t x4639 = x4638 * x4615;
int32_t x4640 = x4639 * x4617;
int32_t x4634 = x4632 % 9;
int32_t x4635 = x4634 / 3;
int32_t x4641 = x4635 * 3;
int32_t x4642 = x4641 * x4615;
int32_t x4643 = x4642 * x4617;
int32_t x4644 = x4640 + x4643;
int32_t x4636 = x4634 % 3;
int32_t x4645 = x4636 * x4617;
int32_t x4646 = x4645 * x4617;
int32_t x4647 = x4644 + x4646;
float* x4648 = x4631+x4647;
int32_t x4649 = x4633 * -7;
float* x4650 = x4627+x4649;
for(int x4652=0; x4652 < x4615; x4652++) {
int32_t x4653 = x4652 * 2;
int32_t x4654 = x4653 - 1;
int32_t x4655 = x4654 + x4635;
bool x4656 = x4655 < 0;
bool x4657 = x4655 >= 1;
bool x4658 = x4656 || x4657;
if (x4658) {
int32_t x4659 = x4652 * x4617;
float* x4660 = x4648+x4659;
memset(x4660, 0, 4 * x4617);;
} else {
int32_t x4659 = x4652 * x4617;
int32_t x4676 = x4655 * -7;
for(int x4664=0; x4664 < x4617; x4664++) {
int32_t x4665 = x4664 * 2;
int32_t x4666 = x4665 - 1;
int32_t x4667 = x4666 + x4636;
bool x4668 = x4667 < 0;
bool x4669 = x4667 >= -7;
bool x4670 = x4668 || x4669;
if (x4670) {
int32_t x4671 = x4659 + x4664;
float* x4672 = x4648+x4671;
memset(x4672, 0, 4 * 1);;
} else {
int32_t x4671 = x4659 + x4664;
float* x4675 = x4648+x4671;
int32_t x4677 = x4676 + x4667;
float* x4678 = x4650+x4677;
memcpy(x4675, x4678, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x4618,-9,1,x27,-9,x4631,x4618,1,x4629,x4618);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(128) x Sym(4615) x Sym(4617)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x4697 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x4698 = 0;
int32_t x4699 = 0;
int32_t x4700 = 0;
for(int x4701=0; x4701 < -7; x4701++) {
int32_t x4702 = x4698;
int32_t x4703 = x4699;
float x4704 = x4622[x4703];
int32_t x4705 = x4700;
float x4706 = x128[x4705];
float x4707 = x4704 - x4706;
x4697[x4702] = x4707;
x4698 += 1;
if (x452) {
x4699 += x4619;
} else {
}
if (x452) {
x4700 += -1;
} else {
}

}
float* x4718 = (float*)myMalloc(128 * sizeof(float));;
for(int x4719=0; x4719 < 128; x4719++) {
float x4720 = x43[x4719];
float x4721 = x4720 + 1.0E-5f;
x4718[x4719] = x4721;

}
float* x4725 = (float*)myMalloc(128 * sizeof(float));;
for(int x4726=0; x4726 < 128; x4726++) {
float x4727 = x4718[x4726];
double x4728 = (double)x4727;
double x4729 = sqrt(x4728);
float x4730 = (float)x4729;
x4725[x4726] = x4730;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x4738 = (float*)myMalloc(7 * sizeof(float));;
int32_t x4739 = 0;
int32_t x4740 = 0;
int32_t x4741 = 0;
for(int x4742=0; x4742 < 1; x4742++) {
int32_t x4743 = x4740;
int32_t x4744 = x4741;
int32_t x4745 = x4739;
int32_t x4746 = x4745;
int32_t x4747 = x4743;
int32_t x4748 = x4744;
for(int x4749=0; x4749 < -1; x4749++) {
int32_t x4750 = x4747;
int32_t x4751 = x4748;
int32_t x4752 = x4746;
int32_t x4753 = x4752;
int32_t x4754 = x4750;
int32_t x4755 = x4751;
for(int x4756=0; x4756 < 1; x4756++) {
int32_t x4757 = x4754;
int32_t x4758 = x4755;
int32_t x4759 = x4753;
int32_t x4760 = x4759;
int32_t x4761 = x4757;
int32_t x4762 = x4758;
for(int x4763=0; x4763 < -7; x4763++) {
int32_t x4764 = x4760;
int32_t x4765 = x4761;
float x4766 = x4697[x4765];
int32_t x4767 = x4762;
float x4768 = x4725[x4767];
float x4769 = x4766 / x4768;
x4738[x4764] = x4769;
x4760 += 1;
if (x520) {
x4761 += 1;
} else {
}
if (x452) {
x4762 += 1;
} else {
}

}
x4753 += -7;
if (x452) {
x4754 += -7;
} else {
}
if (x452) {
x4755 += 1;
} else {
}

}
x4746 += -7;
if (x452) {
x4747 += -7;
} else {
}
if (x542) {
x4748 += 1;
} else {
}

}
x4739 += 7;
if (x452) {
x4740 += -7;
} else {
}
if (x452) {
x4741 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x4811 = (float*)myMalloc(7 * sizeof(float));;
int32_t x4812 = 0;
int32_t x4813 = 0;
int32_t x4814 = 0;
for(int x4815=0; x4815 < 1; x4815++) {
int32_t x4816 = x4813;
int32_t x4817 = x4814;
int32_t x4818 = x4812;
int32_t x4819 = x4818;
int32_t x4820 = x4816;
int32_t x4821 = x4817;
for(int x4822=0; x4822 < -1; x4822++) {
int32_t x4823 = x4820;
int32_t x4824 = x4821;
int32_t x4825 = x4819;
int32_t x4826 = x4825;
int32_t x4827 = x4823;
int32_t x4828 = x4824;
for(int x4829=0; x4829 < 1; x4829++) {
int32_t x4830 = x4827;
int32_t x4831 = x4828;
int32_t x4832 = x4826;
int32_t x4833 = x4832;
int32_t x4834 = x4830;
int32_t x4835 = x4831;
for(int x4836=0; x4836 < -7; x4836++) {
int32_t x4837 = x4833;
int32_t x4838 = x4834;
float x4839 = x4738[x4838];
int32_t x4840 = x4835;
float x4841 = x252[x4840];
float x4842 = x4839 * x4841;
x4811[x4837] = x4842;
x4833 += 1;
if (x520) {
x4834 += 1;
} else {
}
if (x452) {
x4835 += 1;
} else {
}

}
x4826 += -7;
if (x452) {
x4827 += -7;
} else {
}
if (x452) {
x4828 += 1;
} else {
}

}
x4819 += -7;
if (x542) {
x4820 += -7;
} else {
}
if (x542) {
x4821 += 1;
} else {
}

}
x4812 += 7;
if (x452) {
x4813 += 7;
} else {
}
if (x452) {
x4814 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x4884 = (float*)myMalloc(7 * sizeof(float));;
int32_t x4885 = 0;
int32_t x4886 = 0;
int32_t x4887 = 0;
for(int x4888=0; x4888 < 1; x4888++) {
int32_t x4889 = x4886;
int32_t x4890 = x4887;
int32_t x4891 = x4885;
int32_t x4892 = x4891;
int32_t x4893 = x4889;
int32_t x4894 = x4890;
for(int x4895=0; x4895 < -1; x4895++) {
int32_t x4896 = x4893;
int32_t x4897 = x4894;
int32_t x4898 = x4892;
int32_t x4899 = x4898;
int32_t x4900 = x4896;
int32_t x4901 = x4897;
for(int x4902=0; x4902 < 1; x4902++) {
int32_t x4903 = x4900;
int32_t x4904 = x4901;
int32_t x4905 = x4899;
int32_t x4906 = x4905;
int32_t x4907 = x4903;
int32_t x4908 = x4904;
for(int x4909=0; x4909 < -7; x4909++) {
int32_t x4910 = x4906;
int32_t x4911 = x4907;
float x4912 = x4811[x4911];
int32_t x4913 = x4908;
float x4914 = x190[x4913];
float x4915 = x4912 + x4914;
x4884[x4910] = x4915;
x4906 += 1;
if (x520) {
x4907 += 1;
} else {
}
if (x452) {
x4908 += 1;
} else {
}

}
x4899 += -7;
if (x452) {
x4900 += -7;
} else {
}
if (x452) {
x4901 += 1;
} else {
}

}
x4892 += -7;
if (x542) {
x4893 += -7;
} else {
}
if (x542) {
x4894 += 1;
} else {
}

}
x4885 += 7;
if (x452) {
x4886 += 7;
} else {
}
if (x452) {
x4887 += -1;
} else {
}

}
float* x4953 = (float*)myMalloc(7 * sizeof(float));;
for(int x4954=0; x4954 < 7; x4954++) {
float x4955 = x4884[x4954];
bool x4956 = x4955 < 0.0f;
if (x4956) {
x4953[x4954] = 0.0f;
} else {
float x4959 = x4884[x4954];
x4953[x4954] = x4959;
}

}
float* x4968 = (float*)myMalloc(x4967 * sizeof(float));;
float* x4969 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x4970=0; x4970 < 1; x4970++) {
int32_t x4971 = x4970 * 7;
float* x4972 = x4953+x4971;
int32_t x4973 = x4970 * x4965;
float* x4974 = x4968+x4973;
int32_t x4975 = x4970 * x1520;
float* x4976 = x4969+x4975;
for(int x4977=0; x4977 < -1; x4977++) {
int32_t x4978 = x4977 / 1;
int32_t x4982 = x4978 * x1150;
int32_t x4983 = x4982 * x1152;
int32_t x4979 = x4977 % 1;
int32_t x4980 = x4979 / 1;
int32_t x4984 = x4980 * x1150;
int32_t x4985 = x4984 * x1152;
int32_t x4986 = x4983 + x4985;
int32_t x4981 = x4979 % 1;
int32_t x4987 = x4981 * x1152;
int32_t x4988 = x4987 * x1152;
int32_t x4989 = x4986 + x4988;
float* x4990 = x4976+x4989;
int32_t x4991 = x4978 * -7;
float* x4992 = x4972+x4991;
for(int x4993=0; x4993 < x1150; x4993++) {
int32_t x4995 = x4993 * x1152;
float* x4996 = x4990+x4995;
int32_t x4994 = x4993 + x4980;
int32_t x4997 = x4994 * -7;
int32_t x4998 = x4997 + x4981;
float* x4999 = x4992+x4998;
memcpy(x4996, x4999, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x1153,-1,1,x106,-1,x4976,x1153,1,x4974,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(512) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x5012 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x5013 = 0;
int32_t x5014 = 0;
int32_t x5015 = 0;
for(int x5016=0; x5016 < -7; x5016++) {
int32_t x5017 = x5013;
int32_t x5018 = x5014;
float x5019 = x4968[x5018];
int32_t x5020 = x5015;
float x5021 = x149[x5020];
float x5022 = x5019 - x5021;
x5012[x5017] = x5022;
x5013 += 1;
if (x452) {
x5014 += x4965;
} else {
}
if (x452) {
x5015 += -1;
} else {
}

}
float* x5033 = (float*)myMalloc(512 * sizeof(float));;
for(int x5035=0; x5035 < 512; x5035++) {
float x5036 = x101[x5035];
float x5037 = x5036 + 1.0E-5f;
x5033[x5035] = x5037;

}
float* x5041 = (float*)myMalloc(512 * sizeof(float));;
for(int x5042=0; x5042 < 512; x5042++) {
float x5043 = x5033[x5042];
double x5044 = (double)x5043;
double x5045 = sqrt(x5044);
float x5046 = (float)x5045;
x5041[x5042] = x5046;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x5054 = (float*)myMalloc(7 * sizeof(float));;
int32_t x5055 = 0;
int32_t x5056 = 0;
int32_t x5057 = 0;
for(int x5058=0; x5058 < 1; x5058++) {
int32_t x5059 = x5056;
int32_t x5060 = x5057;
int32_t x5061 = x5055;
int32_t x5062 = x5061;
int32_t x5063 = x5059;
int32_t x5064 = x5060;
for(int x5065=0; x5065 < -1; x5065++) {
int32_t x5066 = x5063;
int32_t x5067 = x5064;
int32_t x5068 = x5062;
int32_t x5069 = x5068;
int32_t x5070 = x5066;
int32_t x5071 = x5067;
for(int x5072=0; x5072 < 1; x5072++) {
int32_t x5073 = x5070;
int32_t x5074 = x5071;
int32_t x5075 = x5069;
int32_t x5076 = x5075;
int32_t x5077 = x5073;
int32_t x5078 = x5074;
for(int x5079=0; x5079 < -7; x5079++) {
int32_t x5080 = x5076;
int32_t x5081 = x5077;
float x5082 = x5012[x5081];
int32_t x5083 = x5078;
float x5084 = x5041[x5083];
float x5085 = x5082 / x5084;
x5054[x5080] = x5085;
x5076 += 1;
if (x520) {
x5077 += 1;
} else {
}
if (x452) {
x5078 += 1;
} else {
}

}
x5069 += -7;
if (x452) {
x5070 += -7;
} else {
}
if (x452) {
x5071 += 1;
} else {
}

}
x5062 += -7;
if (x452) {
x5063 += -7;
} else {
}
if (x542) {
x5064 += 1;
} else {
}

}
x5055 += 7;
if (x452) {
x5056 += -7;
} else {
}
if (x452) {
x5057 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x5127 = (float*)myMalloc(7 * sizeof(float));;
int32_t x5128 = 0;
int32_t x5129 = 0;
int32_t x5130 = 0;
for(int x5131=0; x5131 < 1; x5131++) {
int32_t x5132 = x5129;
int32_t x5133 = x5130;
int32_t x5134 = x5128;
int32_t x5135 = x5134;
int32_t x5136 = x5132;
int32_t x5137 = x5133;
for(int x5138=0; x5138 < -1; x5138++) {
int32_t x5139 = x5136;
int32_t x5140 = x5137;
int32_t x5141 = x5135;
int32_t x5142 = x5141;
int32_t x5143 = x5139;
int32_t x5144 = x5140;
for(int x5145=0; x5145 < 1; x5145++) {
int32_t x5146 = x5143;
int32_t x5147 = x5144;
int32_t x5148 = x5142;
int32_t x5149 = x5148;
int32_t x5150 = x5146;
int32_t x5151 = x5147;
for(int x5152=0; x5152 < -7; x5152++) {
int32_t x5153 = x5149;
int32_t x5154 = x5150;
float x5155 = x5054[x5154];
int32_t x5156 = x5151;
float x5157 = x145[x5156];
float x5158 = x5155 * x5157;
x5127[x5153] = x5158;
x5149 += 1;
if (x520) {
x5150 += 1;
} else {
}
if (x452) {
x5151 += 1;
} else {
}

}
x5142 += -7;
if (x452) {
x5143 += -7;
} else {
}
if (x452) {
x5144 += 1;
} else {
}

}
x5135 += -7;
if (x542) {
x5136 += -7;
} else {
}
if (x542) {
x5137 += 1;
} else {
}

}
x5128 += 7;
if (x452) {
x5129 += 7;
} else {
}
if (x452) {
x5130 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x5200 = (float*)myMalloc(7 * sizeof(float));;
int32_t x5201 = 0;
int32_t x5202 = 0;
int32_t x5203 = 0;
for(int x5204=0; x5204 < 1; x5204++) {
int32_t x5205 = x5202;
int32_t x5206 = x5203;
int32_t x5207 = x5201;
int32_t x5208 = x5207;
int32_t x5209 = x5205;
int32_t x5210 = x5206;
for(int x5211=0; x5211 < -1; x5211++) {
int32_t x5212 = x5209;
int32_t x5213 = x5210;
int32_t x5214 = x5208;
int32_t x5215 = x5214;
int32_t x5216 = x5212;
int32_t x5217 = x5213;
for(int x5218=0; x5218 < 1; x5218++) {
int32_t x5219 = x5216;
int32_t x5220 = x5217;
int32_t x5221 = x5215;
int32_t x5222 = x5221;
int32_t x5223 = x5219;
int32_t x5224 = x5220;
for(int x5225=0; x5225 < -7; x5225++) {
int32_t x5226 = x5222;
int32_t x5227 = x5223;
float x5228 = x5127[x5227];
int32_t x5229 = x5224;
float x5230 = x210[x5229];
float x5231 = x5228 + x5230;
x5200[x5226] = x5231;
x5222 += 1;
if (x520) {
x5223 += 1;
} else {
}
if (x452) {
x5224 += 1;
} else {
}

}
x5215 += -7;
if (x452) {
x5216 += -7;
} else {
}
if (x452) {
x5217 += 1;
} else {
}

}
x5208 += -7;
if (x542) {
x5209 += -7;
} else {
}
if (x542) {
x5210 += 1;
} else {
}

}
x5201 += 7;
if (x452) {
x5202 += 7;
} else {
}
if (x452) {
x5203 += -1;
} else {
}

}
float* x5272 = (float*)myMalloc(x5271 * sizeof(float));;
float* x5274 = (float*)myMalloc(x5273 * sizeof(float));;
for(int x5275=0; x5275 < 1; x5275++) {
int32_t x5276 = x5275 * 7;
float* x5277 = x4286+x5276;
int32_t x5278 = x5275 * x5269;
float* x5279 = x5272+x5278;
int32_t x5280 = x5275 * x5273;
float* x5281 = x5274+x5280;
for(int x5282=0; x5282 < -1; x5282++) {
int32_t x5283 = x5282 / 1;
int32_t x5287 = x5283 * x4615;
int32_t x5288 = x5287 * x4617;
int32_t x5284 = x5282 % 1;
int32_t x5285 = x5284 / 1;
int32_t x5289 = x5285 * x4615;
int32_t x5290 = x5289 * x4617;
int32_t x5291 = x5288 + x5290;
int32_t x5286 = x5284 % 1;
int32_t x5292 = x5286 * x4617;
int32_t x5293 = x5292 * x4617;
int32_t x5294 = x5291 + x5293;
float* x5295 = x5281+x5294;
int32_t x5296 = x5283 * -7;
float* x5297 = x5277+x5296;
for(int x5298=0; x5298 < x4615; x5298++) {
int32_t x5302 = x5298 * x4617;
int32_t x5299 = x5298 * 2;
int32_t x5300 = x5299 + x5285;
int32_t x5305 = x5300 * -7;
int32_t x5306 = x5305 + x5286;
for(int x5301=0; x5301 < x4617; x5301++) {
int32_t x5303 = x5302 + x5301;
float* x5304 = x5295+x5303;
int32_t x5307 = x5301 * 2;
int32_t x5308 = x5306 + x5307;
float* x5309 = x5297+x5308;
memcpy(x5304, x5309, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x4618,-1,1,x258,-1,x5281,x4618,1,x5279,x4618);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(512) x Sym(4615) x Sym(4617)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x5324 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x5325 = 0;
int32_t x5326 = 0;
int32_t x5327 = 0;
for(int x5328=0; x5328 < -7; x5328++) {
int32_t x5329 = x5325;
int32_t x5330 = x5326;
float x5331 = x5272[x5330];
int32_t x5332 = x5327;
float x5333 = x42[x5332];
float x5334 = x5331 - x5333;
x5324[x5329] = x5334;
x5325 += 1;
if (x452) {
x5326 += x5269;
} else {
}
if (x452) {
x5327 += -1;
} else {
}

}
float* x5345 = (float*)myMalloc(512 * sizeof(float));;
for(int x5346=0; x5346 < 512; x5346++) {
float x5347 = x23[x5346];
float x5348 = x5347 + 1.0E-5f;
x5345[x5346] = x5348;

}
float* x5352 = (float*)myMalloc(512 * sizeof(float));;
for(int x5353=0; x5353 < 512; x5353++) {
float x5354 = x5345[x5353];
double x5355 = (double)x5354;
double x5356 = sqrt(x5355);
float x5357 = (float)x5356;
x5352[x5353] = x5357;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x5365 = (float*)myMalloc(7 * sizeof(float));;
int32_t x5366 = 0;
int32_t x5367 = 0;
int32_t x5368 = 0;
for(int x5369=0; x5369 < 1; x5369++) {
int32_t x5370 = x5367;
int32_t x5371 = x5368;
int32_t x5372 = x5366;
int32_t x5373 = x5372;
int32_t x5374 = x5370;
int32_t x5375 = x5371;
for(int x5376=0; x5376 < -1; x5376++) {
int32_t x5377 = x5374;
int32_t x5378 = x5375;
int32_t x5379 = x5373;
int32_t x5380 = x5379;
int32_t x5381 = x5377;
int32_t x5382 = x5378;
for(int x5383=0; x5383 < 1; x5383++) {
int32_t x5384 = x5381;
int32_t x5385 = x5382;
int32_t x5386 = x5380;
int32_t x5387 = x5386;
int32_t x5388 = x5384;
int32_t x5389 = x5385;
for(int x5390=0; x5390 < -7; x5390++) {
int32_t x5391 = x5387;
int32_t x5392 = x5388;
float x5393 = x5324[x5392];
int32_t x5394 = x5389;
float x5395 = x5352[x5394];
float x5396 = x5393 / x5395;
x5365[x5391] = x5396;
x5387 += 1;
if (x520) {
x5388 += 1;
} else {
}
if (x452) {
x5389 += 1;
} else {
}

}
x5380 += -7;
if (x452) {
x5381 += -7;
} else {
}
if (x452) {
x5382 += 1;
} else {
}

}
x5373 += -7;
if (x452) {
x5374 += -7;
} else {
}
if (x542) {
x5375 += 1;
} else {
}

}
x5366 += 7;
if (x452) {
x5367 += -7;
} else {
}
if (x452) {
x5368 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x5438 = (float*)myMalloc(7 * sizeof(float));;
int32_t x5439 = 0;
int32_t x5440 = 0;
int32_t x5441 = 0;
for(int x5442=0; x5442 < 1; x5442++) {
int32_t x5443 = x5440;
int32_t x5444 = x5441;
int32_t x5445 = x5439;
int32_t x5446 = x5445;
int32_t x5447 = x5443;
int32_t x5448 = x5444;
for(int x5449=0; x5449 < -1; x5449++) {
int32_t x5450 = x5447;
int32_t x5451 = x5448;
int32_t x5452 = x5446;
int32_t x5453 = x5452;
int32_t x5454 = x5450;
int32_t x5455 = x5451;
for(int x5456=0; x5456 < 1; x5456++) {
int32_t x5457 = x5454;
int32_t x5458 = x5455;
int32_t x5459 = x5453;
int32_t x5460 = x5459;
int32_t x5461 = x5457;
int32_t x5462 = x5458;
for(int x5463=0; x5463 < -7; x5463++) {
int32_t x5464 = x5460;
int32_t x5465 = x5461;
float x5466 = x5365[x5465];
int32_t x5467 = x5462;
float x5468 = x207[x5467];
float x5469 = x5466 * x5468;
x5438[x5464] = x5469;
x5460 += 1;
if (x520) {
x5461 += 1;
} else {
}
if (x452) {
x5462 += 1;
} else {
}

}
x5453 += -7;
if (x452) {
x5454 += -7;
} else {
}
if (x452) {
x5455 += 1;
} else {
}

}
x5446 += -7;
if (x542) {
x5447 += -7;
} else {
}
if (x542) {
x5448 += 1;
} else {
}

}
x5439 += 7;
if (x452) {
x5440 += 7;
} else {
}
if (x452) {
x5441 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x5511 = (float*)myMalloc(7 * sizeof(float));;
int32_t x5512 = 0;
int32_t x5513 = 0;
int32_t x5514 = 0;
for(int x5515=0; x5515 < 1; x5515++) {
int32_t x5516 = x5513;
int32_t x5517 = x5514;
int32_t x5518 = x5512;
int32_t x5519 = x5518;
int32_t x5520 = x5516;
int32_t x5521 = x5517;
for(int x5522=0; x5522 < -1; x5522++) {
int32_t x5523 = x5520;
int32_t x5524 = x5521;
int32_t x5525 = x5519;
int32_t x5526 = x5525;
int32_t x5527 = x5523;
int32_t x5528 = x5524;
for(int x5529=0; x5529 < 1; x5529++) {
int32_t x5530 = x5527;
int32_t x5531 = x5528;
int32_t x5532 = x5526;
int32_t x5533 = x5532;
int32_t x5534 = x5530;
int32_t x5535 = x5531;
for(int x5536=0; x5536 < -7; x5536++) {
int32_t x5537 = x5533;
int32_t x5538 = x5534;
float x5539 = x5438[x5538];
int32_t x5540 = x5535;
float x5541 = x119[x5540];
float x5542 = x5539 + x5541;
x5511[x5537] = x5542;
x5533 += 1;
if (x520) {
x5534 += 1;
} else {
}
if (x452) {
x5535 += 1;
} else {
}

}
x5526 += -7;
if (x452) {
x5527 += -7;
} else {
}
if (x452) {
x5528 += 1;
} else {
}

}
x5519 += -7;
if (x542) {
x5520 += -7;
} else {
}
if (x542) {
x5521 += 1;
} else {
}

}
x5512 += 7;
if (x452) {
x5513 += 7;
} else {
}
if (x452) {
x5514 += -1;
} else {
}

}
int32_t x5580 = 0;
int32_t x5581 = 0;
int32_t x5582 = 0;
for(int x5583=0; x5583 < 1; x5583++) {
int32_t x5584 = x5581;
int32_t x5585 = x5582;
int32_t x5586 = x5580;
int32_t x5587 = x5586;
int32_t x5588 = x5584;
int32_t x5589 = x5585;
for(int x5590=0; x5590 < -1; x5590++) {
int32_t x5591 = x5588;
int32_t x5592 = x5589;
int32_t x5593 = x5587;
int32_t x5594 = x5593;
int32_t x5595 = x5591;
int32_t x5596 = x5592;
for(int x5597=0; x5597 < 1; x5597++) {
int32_t x5598 = x5595;
int32_t x5599 = x5596;
int32_t x5600 = x5594;
int32_t x5601 = x5600;
int32_t x5602 = x5598;
int32_t x5603 = x5599;
for(int x5604=0; x5604 < -7; x5604++) {
int32_t x5605 = x5602;
float x5606 = x5200[x5605];
int32_t x5607 = x5603;
float x5608 = x5511[x5607];
float x5609 = x5606 + x5608;
x5200[x5605] = x5609;
x5601 += 1;
if (x520) {
x5602 += 1;
} else {
}
if (x520) {
x5603 += 1;
} else {
}

}
x5594 += -7;
if (x452) {
x5595 += -7;
} else {
}
if (x452) {
x5596 += -7;
} else {
}

}
x5587 += -7;
if (x542) {
x5588 += -7;
} else {
}
if (x542) {
x5589 += -7;
} else {
}

}
x5580 += 7;
if (x452) {
x5581 += 7;
} else {
}
if (x452) {
x5582 += 7;
} else {
}

}
float* x5647 = (float*)myMalloc(7 * sizeof(float));;
for(int x5648=0; x5648 < 7; x5648++) {
float x5649 = x5200[x5648];
bool x5650 = x5649 < 0.0f;
if (x5650) {
x5647[x5648] = 0.0f;
} else {
float x5653 = x5200[x5648];
x5647[x5648] = x5653;
}

}
float* x5659 = (float*)myMalloc(x4300 * sizeof(float));;
float* x5660 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x5661=0; x5661 < 1; x5661++) {
int32_t x5662 = x5661 * 7;
float* x5663 = x5647+x5662;
int32_t x5664 = x5661 * x4298;
float* x5665 = x5659+x5664;
int32_t x5666 = x5661 * x1520;
float* x5667 = x5660+x5666;
for(int x5668=0; x5668 < -1; x5668++) {
int32_t x5669 = x5668 / 1;
int32_t x5673 = x5669 * x1150;
int32_t x5674 = x5673 * x1152;
int32_t x5670 = x5668 % 1;
int32_t x5671 = x5670 / 1;
int32_t x5675 = x5671 * x1150;
int32_t x5676 = x5675 * x1152;
int32_t x5677 = x5674 + x5676;
int32_t x5672 = x5670 % 1;
int32_t x5678 = x5672 * x1152;
int32_t x5679 = x5678 * x1152;
int32_t x5680 = x5677 + x5679;
float* x5681 = x5667+x5680;
int32_t x5682 = x5669 * -7;
float* x5683 = x5663+x5682;
for(int x5684=0; x5684 < x1150; x5684++) {
int32_t x5686 = x5684 * x1152;
float* x5687 = x5681+x5686;
int32_t x5685 = x5684 + x5671;
int32_t x5688 = x5685 * -7;
int32_t x5689 = x5688 + x5672;
float* x5690 = x5683+x5689;
memcpy(x5687, x5690, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x1153,-1,1,x256,-1,x5667,x1153,1,x5665,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(128) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x5703 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x5704 = 0;
int32_t x5705 = 0;
int32_t x5706 = 0;
for(int x5707=0; x5707 < -7; x5707++) {
int32_t x5708 = x5704;
int32_t x5709 = x5705;
float x5710 = x5659[x5709];
int32_t x5711 = x5706;
float x5712 = x100[x5711];
float x5713 = x5710 - x5712;
x5703[x5708] = x5713;
x5704 += 1;
if (x452) {
x5705 += x4298;
} else {
}
if (x452) {
x5706 += -1;
} else {
}

}
float* x5724 = (float*)myMalloc(128 * sizeof(float));;
for(int x5725=0; x5725 < 128; x5725++) {
float x5726 = x177[x5725];
float x5727 = x5726 + 1.0E-5f;
x5724[x5725] = x5727;

}
float* x5731 = (float*)myMalloc(128 * sizeof(float));;
for(int x5732=0; x5732 < 128; x5732++) {
float x5733 = x5724[x5732];
double x5734 = (double)x5733;
double x5735 = sqrt(x5734);
float x5736 = (float)x5735;
x5731[x5732] = x5736;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x5744 = (float*)myMalloc(7 * sizeof(float));;
int32_t x5745 = 0;
int32_t x5746 = 0;
int32_t x5747 = 0;
for(int x5748=0; x5748 < 1; x5748++) {
int32_t x5749 = x5746;
int32_t x5750 = x5747;
int32_t x5751 = x5745;
int32_t x5752 = x5751;
int32_t x5753 = x5749;
int32_t x5754 = x5750;
for(int x5755=0; x5755 < -1; x5755++) {
int32_t x5756 = x5753;
int32_t x5757 = x5754;
int32_t x5758 = x5752;
int32_t x5759 = x5758;
int32_t x5760 = x5756;
int32_t x5761 = x5757;
for(int x5762=0; x5762 < 1; x5762++) {
int32_t x5763 = x5760;
int32_t x5764 = x5761;
int32_t x5765 = x5759;
int32_t x5766 = x5765;
int32_t x5767 = x5763;
int32_t x5768 = x5764;
for(int x5769=0; x5769 < -7; x5769++) {
int32_t x5770 = x5766;
int32_t x5771 = x5767;
float x5772 = x5703[x5771];
int32_t x5773 = x5768;
float x5774 = x5731[x5773];
float x5775 = x5772 / x5774;
x5744[x5770] = x5775;
x5766 += 1;
if (x520) {
x5767 += 1;
} else {
}
if (x452) {
x5768 += 1;
} else {
}

}
x5759 += -7;
if (x452) {
x5760 += -7;
} else {
}
if (x452) {
x5761 += 1;
} else {
}

}
x5752 += -7;
if (x452) {
x5753 += -7;
} else {
}
if (x542) {
x5754 += 1;
} else {
}

}
x5745 += 7;
if (x452) {
x5746 += -7;
} else {
}
if (x452) {
x5747 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x5817 = (float*)myMalloc(7 * sizeof(float));;
int32_t x5818 = 0;
int32_t x5819 = 0;
int32_t x5820 = 0;
for(int x5821=0; x5821 < 1; x5821++) {
int32_t x5822 = x5819;
int32_t x5823 = x5820;
int32_t x5824 = x5818;
int32_t x5825 = x5824;
int32_t x5826 = x5822;
int32_t x5827 = x5823;
for(int x5828=0; x5828 < -1; x5828++) {
int32_t x5829 = x5826;
int32_t x5830 = x5827;
int32_t x5831 = x5825;
int32_t x5832 = x5831;
int32_t x5833 = x5829;
int32_t x5834 = x5830;
for(int x5835=0; x5835 < 1; x5835++) {
int32_t x5836 = x5833;
int32_t x5837 = x5834;
int32_t x5838 = x5832;
int32_t x5839 = x5838;
int32_t x5840 = x5836;
int32_t x5841 = x5837;
for(int x5842=0; x5842 < -7; x5842++) {
int32_t x5843 = x5839;
int32_t x5844 = x5840;
float x5845 = x5744[x5844];
int32_t x5846 = x5841;
float x5847 = x222[x5846];
float x5848 = x5845 * x5847;
x5817[x5843] = x5848;
x5839 += 1;
if (x520) {
x5840 += 1;
} else {
}
if (x452) {
x5841 += 1;
} else {
}

}
x5832 += -7;
if (x452) {
x5833 += -7;
} else {
}
if (x452) {
x5834 += 1;
} else {
}

}
x5825 += -7;
if (x542) {
x5826 += -7;
} else {
}
if (x542) {
x5827 += 1;
} else {
}

}
x5818 += 7;
if (x452) {
x5819 += 7;
} else {
}
if (x452) {
x5820 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x5890 = (float*)myMalloc(7 * sizeof(float));;
int32_t x5891 = 0;
int32_t x5892 = 0;
int32_t x5893 = 0;
for(int x5894=0; x5894 < 1; x5894++) {
int32_t x5895 = x5892;
int32_t x5896 = x5893;
int32_t x5897 = x5891;
int32_t x5898 = x5897;
int32_t x5899 = x5895;
int32_t x5900 = x5896;
for(int x5901=0; x5901 < -1; x5901++) {
int32_t x5902 = x5899;
int32_t x5903 = x5900;
int32_t x5904 = x5898;
int32_t x5905 = x5904;
int32_t x5906 = x5902;
int32_t x5907 = x5903;
for(int x5908=0; x5908 < 1; x5908++) {
int32_t x5909 = x5906;
int32_t x5910 = x5907;
int32_t x5911 = x5905;
int32_t x5912 = x5911;
int32_t x5913 = x5909;
int32_t x5914 = x5910;
for(int x5915=0; x5915 < -7; x5915++) {
int32_t x5916 = x5912;
int32_t x5917 = x5913;
float x5918 = x5817[x5917];
int32_t x5919 = x5914;
float x5920 = x17[x5919];
float x5921 = x5918 + x5920;
x5890[x5916] = x5921;
x5912 += 1;
if (x520) {
x5913 += 1;
} else {
}
if (x452) {
x5914 += 1;
} else {
}

}
x5905 += -7;
if (x452) {
x5906 += -7;
} else {
}
if (x452) {
x5907 += 1;
} else {
}

}
x5898 += -7;
if (x542) {
x5899 += -7;
} else {
}
if (x542) {
x5900 += 1;
} else {
}

}
x5891 += 7;
if (x452) {
x5892 += 7;
} else {
}
if (x452) {
x5893 += -1;
} else {
}

}
float* x5959 = (float*)myMalloc(7 * sizeof(float));;
for(int x5960=0; x5960 < 7; x5960++) {
float x5961 = x5890[x5960];
bool x5962 = x5961 < 0.0f;
if (x5962) {
x5959[x5960] = 0.0f;
} else {
float x5965 = x5890[x5960];
x5959[x5960] = x5965;
}

}
float* x5971 = (float*)myMalloc(x4300 * sizeof(float));;
float* x5972 = (float*)myMalloc(x1158 * sizeof(float));;
for(int x5973=0; x5973 < 1; x5973++) {
int32_t x5974 = x5973 * 7;
float* x5975 = x5959+x5974;
int32_t x5976 = x5973 * x4298;
float* x5977 = x5971+x5976;
int32_t x5978 = x5973 * x1158;
float* x5979 = x5972+x5978;
for(int x5980=0; x5980 < -9; x5980++) {
int32_t x5981 = x5980 / 9;
int32_t x5985 = x5981 * 3;
int32_t x5986 = x5985 * 3;
int32_t x5987 = x5986 * x1150;
int32_t x5988 = x5987 * x1152;
int32_t x5982 = x5980 % 9;
int32_t x5983 = x5982 / 3;
int32_t x5989 = x5983 * 3;
int32_t x5990 = x5989 * x1150;
int32_t x5991 = x5990 * x1152;
int32_t x5992 = x5988 + x5991;
int32_t x5984 = x5982 % 3;
int32_t x5993 = x5984 * x1152;
int32_t x5994 = x5993 * x1152;
int32_t x5995 = x5992 + x5994;
float* x5996 = x5979+x5995;
int32_t x5997 = x5981 * -7;
float* x5998 = x5975+x5997;
int32_t x6010 = 1 - x5984;
bool x6011 = x6010 > 0;
int32_t x6012;
if (x6011) {
x6012 = x6010;
} else {
x6012 = 0;
}
int32_t x6013 = 3 - x5984;
int32_t x6014 = x6013 - 1;
int32_t x6015 = 1 - x6014;
bool x6016 = x6015 > 0;
int32_t x6017;
if (x6016) {
x6017 = x6015;
} else {
x6017 = 0;
}
int32_t x6018 = x1152 - x6017;
int32_t x6019 = x6018 - x6012;
bool x6020 = x6019 <= 0;
bool x6024 = x6012 > 0;
int32_t x6009 = -1 + x5984;
bool x6037 = x6017 > 0;
for(int x5999=0; x5999 < x1150; x5999++) {
int32_t x6000 = x5999 - 1;
int32_t x6001 = x6000 + x5983;
bool x6002 = x6001 < 0;
bool x6003 = x6001 >= 1;
bool x6004 = x6002 || x6003;
if (x6004) {
int32_t x6005 = x5999 * x1152;
float* x6006 = x5996+x6005;
memset(x6006, 0, 4 * x1152);;
} else {
if (x6020) {
int32_t x6005 = x5999 * x1152;
float* x6021 = x5996+x6005;
memset(x6021, 0, 4 * x1152);;
} else {
int32_t x6005 = x5999 * x1152;
if (x6024) {
float* x6025 = x5996+x6005;
memset(x6025, 0, 4 * x6012);;
} else {
}
// may have segfault here
int32_t x6030 = x6005 + x6012;
float* x6031 = x5996+x6030;
int32_t x6032 = x6001 * -7;
int32_t x6033 = x6032 + x6009;
int32_t x6034 = x6033 + x6012;
float* x6035 = x5998+x6034;
memcpy(x6031, x6035, 4 * x6019);;
if (x6037) {
int32_t x6038 = x6005 + x1152;
int32_t x6039 = x6038 - x6017;
float* x6040 = x5996+x6039;
memset(x6040, 0, 4 * x6017);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x1153,-9,1,x235,-9,x5979,x1153,1,x5977,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(128) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x6059 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x6060 = 0;
int32_t x6061 = 0;
int32_t x6062 = 0;
for(int x6063=0; x6063 < -7; x6063++) {
int32_t x6064 = x6060;
int32_t x6065 = x6061;
float x6066 = x5971[x6065];
int32_t x6067 = x6062;
float x6068 = x35[x6067];
float x6069 = x6066 - x6068;
x6059[x6064] = x6069;
x6060 += 1;
if (x452) {
x6061 += x4298;
} else {
}
if (x452) {
x6062 += -1;
} else {
}

}
float* x6080 = (float*)myMalloc(128 * sizeof(float));;
for(int x6081=0; x6081 < 128; x6081++) {
float x6082 = x225[x6081];
float x6083 = x6082 + 1.0E-5f;
x6080[x6081] = x6083;

}
float* x6087 = (float*)myMalloc(128 * sizeof(float));;
for(int x6088=0; x6088 < 128; x6088++) {
float x6089 = x6080[x6088];
double x6090 = (double)x6089;
double x6091 = sqrt(x6090);
float x6092 = (float)x6091;
x6087[x6088] = x6092;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x6100 = (float*)myMalloc(7 * sizeof(float));;
int32_t x6101 = 0;
int32_t x6102 = 0;
int32_t x6103 = 0;
for(int x6104=0; x6104 < 1; x6104++) {
int32_t x6105 = x6102;
int32_t x6106 = x6103;
int32_t x6107 = x6101;
int32_t x6108 = x6107;
int32_t x6109 = x6105;
int32_t x6110 = x6106;
for(int x6111=0; x6111 < -1; x6111++) {
int32_t x6112 = x6109;
int32_t x6113 = x6110;
int32_t x6114 = x6108;
int32_t x6115 = x6114;
int32_t x6116 = x6112;
int32_t x6117 = x6113;
for(int x6118=0; x6118 < 1; x6118++) {
int32_t x6119 = x6116;
int32_t x6120 = x6117;
int32_t x6121 = x6115;
int32_t x6122 = x6121;
int32_t x6123 = x6119;
int32_t x6124 = x6120;
for(int x6125=0; x6125 < -7; x6125++) {
int32_t x6126 = x6122;
int32_t x6127 = x6123;
float x6128 = x6059[x6127];
int32_t x6129 = x6124;
float x6130 = x6087[x6129];
float x6131 = x6128 / x6130;
x6100[x6126] = x6131;
x6122 += 1;
if (x520) {
x6123 += 1;
} else {
}
if (x452) {
x6124 += 1;
} else {
}

}
x6115 += -7;
if (x452) {
x6116 += -7;
} else {
}
if (x452) {
x6117 += 1;
} else {
}

}
x6108 += -7;
if (x452) {
x6109 += -7;
} else {
}
if (x542) {
x6110 += 1;
} else {
}

}
x6101 += 7;
if (x452) {
x6102 += -7;
} else {
}
if (x452) {
x6103 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x6173 = (float*)myMalloc(7 * sizeof(float));;
int32_t x6174 = 0;
int32_t x6175 = 0;
int32_t x6176 = 0;
for(int x6177=0; x6177 < 1; x6177++) {
int32_t x6178 = x6175;
int32_t x6179 = x6176;
int32_t x6180 = x6174;
int32_t x6181 = x6180;
int32_t x6182 = x6178;
int32_t x6183 = x6179;
for(int x6184=0; x6184 < -1; x6184++) {
int32_t x6185 = x6182;
int32_t x6186 = x6183;
int32_t x6187 = x6181;
int32_t x6188 = x6187;
int32_t x6189 = x6185;
int32_t x6190 = x6186;
for(int x6191=0; x6191 < 1; x6191++) {
int32_t x6192 = x6189;
int32_t x6193 = x6190;
int32_t x6194 = x6188;
int32_t x6195 = x6194;
int32_t x6196 = x6192;
int32_t x6197 = x6193;
for(int x6198=0; x6198 < -7; x6198++) {
int32_t x6199 = x6195;
int32_t x6200 = x6196;
float x6201 = x6100[x6200];
int32_t x6202 = x6197;
float x6203 = x8[x6202];
float x6204 = x6201 * x6203;
x6173[x6199] = x6204;
x6195 += 1;
if (x520) {
x6196 += 1;
} else {
}
if (x452) {
x6197 += 1;
} else {
}

}
x6188 += -7;
if (x452) {
x6189 += -7;
} else {
}
if (x452) {
x6190 += 1;
} else {
}

}
x6181 += -7;
if (x542) {
x6182 += -7;
} else {
}
if (x542) {
x6183 += 1;
} else {
}

}
x6174 += 7;
if (x452) {
x6175 += 7;
} else {
}
if (x452) {
x6176 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x6246 = (float*)myMalloc(7 * sizeof(float));;
int32_t x6247 = 0;
int32_t x6248 = 0;
int32_t x6249 = 0;
for(int x6250=0; x6250 < 1; x6250++) {
int32_t x6251 = x6248;
int32_t x6252 = x6249;
int32_t x6253 = x6247;
int32_t x6254 = x6253;
int32_t x6255 = x6251;
int32_t x6256 = x6252;
for(int x6257=0; x6257 < -1; x6257++) {
int32_t x6258 = x6255;
int32_t x6259 = x6256;
int32_t x6260 = x6254;
int32_t x6261 = x6260;
int32_t x6262 = x6258;
int32_t x6263 = x6259;
for(int x6264=0; x6264 < 1; x6264++) {
int32_t x6265 = x6262;
int32_t x6266 = x6263;
int32_t x6267 = x6261;
int32_t x6268 = x6267;
int32_t x6269 = x6265;
int32_t x6270 = x6266;
for(int x6271=0; x6271 < -7; x6271++) {
int32_t x6272 = x6268;
int32_t x6273 = x6269;
float x6274 = x6173[x6273];
int32_t x6275 = x6270;
float x6276 = x95[x6275];
float x6277 = x6274 + x6276;
x6246[x6272] = x6277;
x6268 += 1;
if (x520) {
x6269 += 1;
} else {
}
if (x452) {
x6270 += 1;
} else {
}

}
x6261 += -7;
if (x452) {
x6262 += -7;
} else {
}
if (x452) {
x6263 += 1;
} else {
}

}
x6254 += -7;
if (x542) {
x6255 += -7;
} else {
}
if (x542) {
x6256 += 1;
} else {
}

}
x6247 += 7;
if (x452) {
x6248 += 7;
} else {
}
if (x452) {
x6249 += -1;
} else {
}

}
float* x6315 = (float*)myMalloc(7 * sizeof(float));;
for(int x6316=0; x6316 < 7; x6316++) {
float x6317 = x6246[x6316];
bool x6318 = x6317 < 0.0f;
if (x6318) {
x6315[x6316] = 0.0f;
} else {
float x6321 = x6246[x6316];
x6315[x6316] = x6321;
}

}
float* x6327 = (float*)myMalloc(x4967 * sizeof(float));;
float* x6328 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x6329=0; x6329 < 1; x6329++) {
int32_t x6330 = x6329 * 7;
float* x6331 = x6315+x6330;
int32_t x6332 = x6329 * x4965;
float* x6333 = x6327+x6332;
int32_t x6334 = x6329 * x1520;
float* x6335 = x6328+x6334;
for(int x6336=0; x6336 < -1; x6336++) {
int32_t x6337 = x6336 / 1;
int32_t x6341 = x6337 * x1150;
int32_t x6342 = x6341 * x1152;
int32_t x6338 = x6336 % 1;
int32_t x6339 = x6338 / 1;
int32_t x6343 = x6339 * x1150;
int32_t x6344 = x6343 * x1152;
int32_t x6345 = x6342 + x6344;
int32_t x6340 = x6338 % 1;
int32_t x6346 = x6340 * x1152;
int32_t x6347 = x6346 * x1152;
int32_t x6348 = x6345 + x6347;
float* x6349 = x6335+x6348;
int32_t x6350 = x6337 * -7;
float* x6351 = x6331+x6350;
for(int x6352=0; x6352 < x1150; x6352++) {
int32_t x6354 = x6352 * x1152;
float* x6355 = x6349+x6354;
int32_t x6353 = x6352 + x6339;
int32_t x6356 = x6353 * -7;
int32_t x6357 = x6356 + x6340;
float* x6358 = x6351+x6357;
memcpy(x6355, x6358, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x1153,-1,1,x111,-1,x6335,x1153,1,x6333,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(512) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x6371 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x6372 = 0;
int32_t x6373 = 0;
int32_t x6374 = 0;
for(int x6375=0; x6375 < -7; x6375++) {
int32_t x6376 = x6372;
int32_t x6377 = x6373;
float x6378 = x6327[x6377];
int32_t x6379 = x6374;
float x6380 = x147[x6379];
float x6381 = x6378 - x6380;
x6371[x6376] = x6381;
x6372 += 1;
if (x452) {
x6373 += x4965;
} else {
}
if (x452) {
x6374 += -1;
} else {
}

}
float* x6392 = (float*)myMalloc(512 * sizeof(float));;
for(int x6393=0; x6393 < 512; x6393++) {
float x6394 = x88[x6393];
float x6395 = x6394 + 1.0E-5f;
x6392[x6393] = x6395;

}
float* x6399 = (float*)myMalloc(512 * sizeof(float));;
for(int x6400=0; x6400 < 512; x6400++) {
float x6401 = x6392[x6400];
double x6402 = (double)x6401;
double x6403 = sqrt(x6402);
float x6404 = (float)x6403;
x6399[x6400] = x6404;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x6412 = (float*)myMalloc(7 * sizeof(float));;
int32_t x6413 = 0;
int32_t x6414 = 0;
int32_t x6415 = 0;
for(int x6416=0; x6416 < 1; x6416++) {
int32_t x6417 = x6414;
int32_t x6418 = x6415;
int32_t x6419 = x6413;
int32_t x6420 = x6419;
int32_t x6421 = x6417;
int32_t x6422 = x6418;
for(int x6423=0; x6423 < -1; x6423++) {
int32_t x6424 = x6421;
int32_t x6425 = x6422;
int32_t x6426 = x6420;
int32_t x6427 = x6426;
int32_t x6428 = x6424;
int32_t x6429 = x6425;
for(int x6430=0; x6430 < 1; x6430++) {
int32_t x6431 = x6428;
int32_t x6432 = x6429;
int32_t x6433 = x6427;
int32_t x6434 = x6433;
int32_t x6435 = x6431;
int32_t x6436 = x6432;
for(int x6437=0; x6437 < -7; x6437++) {
int32_t x6438 = x6434;
int32_t x6439 = x6435;
float x6440 = x6371[x6439];
int32_t x6441 = x6436;
float x6442 = x6399[x6441];
float x6443 = x6440 / x6442;
x6412[x6438] = x6443;
x6434 += 1;
if (x520) {
x6435 += 1;
} else {
}
if (x452) {
x6436 += 1;
} else {
}

}
x6427 += -7;
if (x452) {
x6428 += -7;
} else {
}
if (x452) {
x6429 += 1;
} else {
}

}
x6420 += -7;
if (x452) {
x6421 += -7;
} else {
}
if (x542) {
x6422 += 1;
} else {
}

}
x6413 += 7;
if (x452) {
x6414 += -7;
} else {
}
if (x452) {
x6415 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x6485 = (float*)myMalloc(7 * sizeof(float));;
int32_t x6486 = 0;
int32_t x6487 = 0;
int32_t x6488 = 0;
for(int x6489=0; x6489 < 1; x6489++) {
int32_t x6490 = x6487;
int32_t x6491 = x6488;
int32_t x6492 = x6486;
int32_t x6493 = x6492;
int32_t x6494 = x6490;
int32_t x6495 = x6491;
for(int x6496=0; x6496 < -1; x6496++) {
int32_t x6497 = x6494;
int32_t x6498 = x6495;
int32_t x6499 = x6493;
int32_t x6500 = x6499;
int32_t x6501 = x6497;
int32_t x6502 = x6498;
for(int x6503=0; x6503 < 1; x6503++) {
int32_t x6504 = x6501;
int32_t x6505 = x6502;
int32_t x6506 = x6500;
int32_t x6507 = x6506;
int32_t x6508 = x6504;
int32_t x6509 = x6505;
for(int x6510=0; x6510 < -7; x6510++) {
int32_t x6511 = x6507;
int32_t x6512 = x6508;
float x6513 = x6412[x6512];
int32_t x6514 = x6509;
float x6515 = x52[x6514];
float x6516 = x6513 * x6515;
x6485[x6511] = x6516;
x6507 += 1;
if (x520) {
x6508 += 1;
} else {
}
if (x452) {
x6509 += 1;
} else {
}

}
x6500 += -7;
if (x452) {
x6501 += -7;
} else {
}
if (x452) {
x6502 += 1;
} else {
}

}
x6493 += -7;
if (x542) {
x6494 += -7;
} else {
}
if (x542) {
x6495 += 1;
} else {
}

}
x6486 += 7;
if (x452) {
x6487 += 7;
} else {
}
if (x452) {
x6488 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x6558 = (float*)myMalloc(7 * sizeof(float));;
int32_t x6559 = 0;
int32_t x6560 = 0;
int32_t x6561 = 0;
for(int x6562=0; x6562 < 1; x6562++) {
int32_t x6563 = x6560;
int32_t x6564 = x6561;
int32_t x6565 = x6559;
int32_t x6566 = x6565;
int32_t x6567 = x6563;
int32_t x6568 = x6564;
for(int x6569=0; x6569 < -1; x6569++) {
int32_t x6570 = x6567;
int32_t x6571 = x6568;
int32_t x6572 = x6566;
int32_t x6573 = x6572;
int32_t x6574 = x6570;
int32_t x6575 = x6571;
for(int x6576=0; x6576 < 1; x6576++) {
int32_t x6577 = x6574;
int32_t x6578 = x6575;
int32_t x6579 = x6573;
int32_t x6580 = x6579;
int32_t x6581 = x6577;
int32_t x6582 = x6578;
for(int x6583=0; x6583 < -7; x6583++) {
int32_t x6584 = x6580;
int32_t x6585 = x6581;
float x6586 = x6485[x6585];
int32_t x6587 = x6582;
float x6588 = x246[x6587];
float x6589 = x6586 + x6588;
x6558[x6584] = x6589;
x6580 += 1;
if (x520) {
x6581 += 1;
} else {
}
if (x452) {
x6582 += 1;
} else {
}

}
x6573 += -7;
if (x452) {
x6574 += -7;
} else {
}
if (x452) {
x6575 += 1;
} else {
}

}
x6566 += -7;
if (x542) {
x6567 += -7;
} else {
}
if (x542) {
x6568 += 1;
} else {
}

}
x6559 += 7;
if (x452) {
x6560 += 7;
} else {
}
if (x452) {
x6561 += -1;
} else {
}

}
int32_t x6627 = 0;
int32_t x6628 = 0;
int32_t x6629 = 0;
for(int x6630=0; x6630 < 1; x6630++) {
int32_t x6631 = x6628;
int32_t x6632 = x6629;
int32_t x6633 = x6627;
int32_t x6634 = x6633;
int32_t x6635 = x6631;
int32_t x6636 = x6632;
for(int x6637=0; x6637 < -1; x6637++) {
int32_t x6638 = x6635;
int32_t x6639 = x6636;
int32_t x6640 = x6634;
int32_t x6641 = x6640;
int32_t x6642 = x6638;
int32_t x6643 = x6639;
for(int x6644=0; x6644 < 1; x6644++) {
int32_t x6645 = x6642;
int32_t x6646 = x6643;
int32_t x6647 = x6641;
int32_t x6648 = x6647;
int32_t x6649 = x6645;
int32_t x6650 = x6646;
for(int x6651=0; x6651 < -7; x6651++) {
int32_t x6652 = x6649;
float x6653 = x6558[x6652];
int32_t x6654 = x6650;
float x6655 = x5647[x6654];
float x6656 = x6653 + x6655;
x6558[x6652] = x6656;
x6648 += 1;
if (x520) {
x6649 += 1;
} else {
}
if (x520) {
x6650 += 1;
} else {
}

}
x6641 += -7;
if (x452) {
x6642 += -7;
} else {
}
if (x452) {
x6643 += -7;
} else {
}

}
x6634 += -7;
if (x542) {
x6635 += -7;
} else {
}
if (x542) {
x6636 += -7;
} else {
}

}
x6627 += 7;
if (x452) {
x6628 += 7;
} else {
}
if (x452) {
x6629 += 7;
} else {
}

}
float* x6694 = (float*)myMalloc(7 * sizeof(float));;
for(int x6695=0; x6695 < 7; x6695++) {
float x6696 = x6558[x6695];
bool x6697 = x6696 < 0.0f;
if (x6697) {
x6694[x6695] = 0.0f;
} else {
float x6700 = x6558[x6695];
x6694[x6695] = x6700;
}

}
float* x6706 = (float*)myMalloc(x4300 * sizeof(float));;
float* x6707 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x6708=0; x6708 < 1; x6708++) {
int32_t x6709 = x6708 * 7;
float* x6710 = x6694+x6709;
int32_t x6711 = x6708 * x4298;
float* x6712 = x6706+x6711;
int32_t x6713 = x6708 * x1520;
float* x6714 = x6707+x6713;
for(int x6715=0; x6715 < -1; x6715++) {
int32_t x6716 = x6715 / 1;
int32_t x6720 = x6716 * x1150;
int32_t x6721 = x6720 * x1152;
int32_t x6717 = x6715 % 1;
int32_t x6718 = x6717 / 1;
int32_t x6722 = x6718 * x1150;
int32_t x6723 = x6722 * x1152;
int32_t x6724 = x6721 + x6723;
int32_t x6719 = x6717 % 1;
int32_t x6725 = x6719 * x1152;
int32_t x6726 = x6725 * x1152;
int32_t x6727 = x6724 + x6726;
float* x6728 = x6714+x6727;
int32_t x6729 = x6716 * -7;
float* x6730 = x6710+x6729;
for(int x6731=0; x6731 < x1150; x6731++) {
int32_t x6733 = x6731 * x1152;
float* x6734 = x6728+x6733;
int32_t x6732 = x6731 + x6718;
int32_t x6735 = x6732 * -7;
int32_t x6736 = x6735 + x6719;
float* x6737 = x6730+x6736;
memcpy(x6734, x6737, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x1153,-1,1,x196,-1,x6714,x1153,1,x6712,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(128) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x6750 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x6751 = 0;
int32_t x6752 = 0;
int32_t x6753 = 0;
for(int x6754=0; x6754 < -7; x6754++) {
int32_t x6755 = x6751;
int32_t x6756 = x6752;
float x6757 = x6706[x6756];
int32_t x6758 = x6753;
float x6759 = x112[x6758];
float x6760 = x6757 - x6759;
x6750[x6755] = x6760;
x6751 += 1;
if (x452) {
x6752 += x4298;
} else {
}
if (x452) {
x6753 += -1;
} else {
}

}
float* x6771 = (float*)myMalloc(128 * sizeof(float));;
for(int x6772=0; x6772 < 128; x6772++) {
float x6773 = x9[x6772];
float x6774 = x6773 + 1.0E-5f;
x6771[x6772] = x6774;

}
float* x6778 = (float*)myMalloc(128 * sizeof(float));;
for(int x6779=0; x6779 < 128; x6779++) {
float x6780 = x6771[x6779];
double x6781 = (double)x6780;
double x6782 = sqrt(x6781);
float x6783 = (float)x6782;
x6778[x6779] = x6783;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x6791 = (float*)myMalloc(7 * sizeof(float));;
int32_t x6792 = 0;
int32_t x6793 = 0;
int32_t x6794 = 0;
for(int x6795=0; x6795 < 1; x6795++) {
int32_t x6796 = x6793;
int32_t x6797 = x6794;
int32_t x6798 = x6792;
int32_t x6799 = x6798;
int32_t x6800 = x6796;
int32_t x6801 = x6797;
for(int x6802=0; x6802 < -1; x6802++) {
int32_t x6803 = x6800;
int32_t x6804 = x6801;
int32_t x6805 = x6799;
int32_t x6806 = x6805;
int32_t x6807 = x6803;
int32_t x6808 = x6804;
for(int x6809=0; x6809 < 1; x6809++) {
int32_t x6810 = x6807;
int32_t x6811 = x6808;
int32_t x6812 = x6806;
int32_t x6813 = x6812;
int32_t x6814 = x6810;
int32_t x6815 = x6811;
for(int x6816=0; x6816 < -7; x6816++) {
int32_t x6817 = x6813;
int32_t x6818 = x6814;
float x6819 = x6750[x6818];
int32_t x6820 = x6815;
float x6821 = x6778[x6820];
float x6822 = x6819 / x6821;
x6791[x6817] = x6822;
x6813 += 1;
if (x520) {
x6814 += 1;
} else {
}
if (x452) {
x6815 += 1;
} else {
}

}
x6806 += -7;
if (x452) {
x6807 += -7;
} else {
}
if (x452) {
x6808 += 1;
} else {
}

}
x6799 += -7;
if (x452) {
x6800 += -7;
} else {
}
if (x542) {
x6801 += 1;
} else {
}

}
x6792 += 7;
if (x452) {
x6793 += -7;
} else {
}
if (x452) {
x6794 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x6864 = (float*)myMalloc(7 * sizeof(float));;
int32_t x6865 = 0;
int32_t x6866 = 0;
int32_t x6867 = 0;
for(int x6868=0; x6868 < 1; x6868++) {
int32_t x6869 = x6866;
int32_t x6870 = x6867;
int32_t x6871 = x6865;
int32_t x6872 = x6871;
int32_t x6873 = x6869;
int32_t x6874 = x6870;
for(int x6875=0; x6875 < -1; x6875++) {
int32_t x6876 = x6873;
int32_t x6877 = x6874;
int32_t x6878 = x6872;
int32_t x6879 = x6878;
int32_t x6880 = x6876;
int32_t x6881 = x6877;
for(int x6882=0; x6882 < 1; x6882++) {
int32_t x6883 = x6880;
int32_t x6884 = x6881;
int32_t x6885 = x6879;
int32_t x6886 = x6885;
int32_t x6887 = x6883;
int32_t x6888 = x6884;
for(int x6889=0; x6889 < -7; x6889++) {
int32_t x6890 = x6886;
int32_t x6891 = x6887;
float x6892 = x6791[x6891];
int32_t x6893 = x6888;
float x6894 = x45[x6893];
float x6895 = x6892 * x6894;
x6864[x6890] = x6895;
x6886 += 1;
if (x520) {
x6887 += 1;
} else {
}
if (x452) {
x6888 += 1;
} else {
}

}
x6879 += -7;
if (x452) {
x6880 += -7;
} else {
}
if (x452) {
x6881 += 1;
} else {
}

}
x6872 += -7;
if (x542) {
x6873 += -7;
} else {
}
if (x542) {
x6874 += 1;
} else {
}

}
x6865 += 7;
if (x452) {
x6866 += 7;
} else {
}
if (x452) {
x6867 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x6937 = (float*)myMalloc(7 * sizeof(float));;
int32_t x6938 = 0;
int32_t x6939 = 0;
int32_t x6940 = 0;
for(int x6941=0; x6941 < 1; x6941++) {
int32_t x6942 = x6939;
int32_t x6943 = x6940;
int32_t x6944 = x6938;
int32_t x6945 = x6944;
int32_t x6946 = x6942;
int32_t x6947 = x6943;
for(int x6948=0; x6948 < -1; x6948++) {
int32_t x6949 = x6946;
int32_t x6950 = x6947;
int32_t x6951 = x6945;
int32_t x6952 = x6951;
int32_t x6953 = x6949;
int32_t x6954 = x6950;
for(int x6955=0; x6955 < 1; x6955++) {
int32_t x6956 = x6953;
int32_t x6957 = x6954;
int32_t x6958 = x6952;
int32_t x6959 = x6958;
int32_t x6960 = x6956;
int32_t x6961 = x6957;
for(int x6962=0; x6962 < -7; x6962++) {
int32_t x6963 = x6959;
int32_t x6964 = x6960;
float x6965 = x6864[x6964];
int32_t x6966 = x6961;
float x6967 = x170[x6966];
float x6968 = x6965 + x6967;
x6937[x6963] = x6968;
x6959 += 1;
if (x520) {
x6960 += 1;
} else {
}
if (x452) {
x6961 += 1;
} else {
}

}
x6952 += -7;
if (x452) {
x6953 += -7;
} else {
}
if (x452) {
x6954 += 1;
} else {
}

}
x6945 += -7;
if (x542) {
x6946 += -7;
} else {
}
if (x542) {
x6947 += 1;
} else {
}

}
x6938 += 7;
if (x452) {
x6939 += 7;
} else {
}
if (x452) {
x6940 += -1;
} else {
}

}
float* x7006 = (float*)myMalloc(7 * sizeof(float));;
for(int x7007=0; x7007 < 7; x7007++) {
float x7008 = x6937[x7007];
bool x7009 = x7008 < 0.0f;
if (x7009) {
x7006[x7007] = 0.0f;
} else {
float x7012 = x6937[x7007];
x7006[x7007] = x7012;
}

}
float* x7018 = (float*)myMalloc(x4300 * sizeof(float));;
float* x7019 = (float*)myMalloc(x1158 * sizeof(float));;
for(int x7020=0; x7020 < 1; x7020++) {
int32_t x7021 = x7020 * 7;
float* x7022 = x7006+x7021;
int32_t x7023 = x7020 * x4298;
float* x7024 = x7018+x7023;
int32_t x7025 = x7020 * x1158;
float* x7026 = x7019+x7025;
for(int x7027=0; x7027 < -9; x7027++) {
int32_t x7028 = x7027 / 9;
int32_t x7032 = x7028 * 3;
int32_t x7033 = x7032 * 3;
int32_t x7034 = x7033 * x1150;
int32_t x7035 = x7034 * x1152;
int32_t x7029 = x7027 % 9;
int32_t x7030 = x7029 / 3;
int32_t x7036 = x7030 * 3;
int32_t x7037 = x7036 * x1150;
int32_t x7038 = x7037 * x1152;
int32_t x7039 = x7035 + x7038;
int32_t x7031 = x7029 % 3;
int32_t x7040 = x7031 * x1152;
int32_t x7041 = x7040 * x1152;
int32_t x7042 = x7039 + x7041;
float* x7043 = x7026+x7042;
int32_t x7044 = x7028 * -7;
float* x7045 = x7022+x7044;
int32_t x7057 = 1 - x7031;
bool x7058 = x7057 > 0;
int32_t x7059;
if (x7058) {
x7059 = x7057;
} else {
x7059 = 0;
}
int32_t x7060 = 3 - x7031;
int32_t x7061 = x7060 - 1;
int32_t x7062 = 1 - x7061;
bool x7063 = x7062 > 0;
int32_t x7064;
if (x7063) {
x7064 = x7062;
} else {
x7064 = 0;
}
int32_t x7065 = x1152 - x7064;
int32_t x7066 = x7065 - x7059;
bool x7067 = x7066 <= 0;
bool x7071 = x7059 > 0;
int32_t x7056 = -1 + x7031;
bool x7084 = x7064 > 0;
for(int x7046=0; x7046 < x1150; x7046++) {
int32_t x7047 = x7046 - 1;
int32_t x7048 = x7047 + x7030;
bool x7049 = x7048 < 0;
bool x7050 = x7048 >= 1;
bool x7051 = x7049 || x7050;
if (x7051) {
int32_t x7052 = x7046 * x1152;
float* x7053 = x7043+x7052;
memset(x7053, 0, 4 * x1152);;
} else {
if (x7067) {
int32_t x7052 = x7046 * x1152;
float* x7068 = x7043+x7052;
memset(x7068, 0, 4 * x1152);;
} else {
int32_t x7052 = x7046 * x1152;
if (x7071) {
float* x7072 = x7043+x7052;
memset(x7072, 0, 4 * x7059);;
} else {
}
// may have segfault here
int32_t x7077 = x7052 + x7059;
float* x7078 = x7043+x7077;
int32_t x7079 = x7048 * -7;
int32_t x7080 = x7079 + x7056;
int32_t x7081 = x7080 + x7059;
float* x7082 = x7045+x7081;
memcpy(x7078, x7082, 4 * x7066);;
if (x7084) {
int32_t x7085 = x7052 + x1152;
int32_t x7086 = x7085 - x7064;
float* x7087 = x7043+x7086;
memset(x7087, 0, 4 * x7064);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x1153,-9,1,x191,-9,x7026,x1153,1,x7024,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(128) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x7106 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x7107 = 0;
int32_t x7108 = 0;
int32_t x7109 = 0;
for(int x7110=0; x7110 < -7; x7110++) {
int32_t x7111 = x7107;
int32_t x7112 = x7108;
float x7113 = x7018[x7112];
int32_t x7114 = x7109;
float x7115 = x217[x7114];
float x7116 = x7113 - x7115;
x7106[x7111] = x7116;
x7107 += 1;
if (x452) {
x7108 += x4298;
} else {
}
if (x452) {
x7109 += -1;
} else {
}

}
float* x7127 = (float*)myMalloc(128 * sizeof(float));;
for(int x7128=0; x7128 < 128; x7128++) {
float x7129 = x266[x7128];
float x7130 = x7129 + 1.0E-5f;
x7127[x7128] = x7130;

}
float* x7134 = (float*)myMalloc(128 * sizeof(float));;
for(int x7135=0; x7135 < 128; x7135++) {
float x7136 = x7127[x7135];
double x7137 = (double)x7136;
double x7138 = sqrt(x7137);
float x7139 = (float)x7138;
x7134[x7135] = x7139;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x7147 = (float*)myMalloc(7 * sizeof(float));;
int32_t x7148 = 0;
int32_t x7149 = 0;
int32_t x7150 = 0;
for(int x7151=0; x7151 < 1; x7151++) {
int32_t x7152 = x7149;
int32_t x7153 = x7150;
int32_t x7154 = x7148;
int32_t x7155 = x7154;
int32_t x7156 = x7152;
int32_t x7157 = x7153;
for(int x7158=0; x7158 < -1; x7158++) {
int32_t x7159 = x7156;
int32_t x7160 = x7157;
int32_t x7161 = x7155;
int32_t x7162 = x7161;
int32_t x7163 = x7159;
int32_t x7164 = x7160;
for(int x7165=0; x7165 < 1; x7165++) {
int32_t x7166 = x7163;
int32_t x7167 = x7164;
int32_t x7168 = x7162;
int32_t x7169 = x7168;
int32_t x7170 = x7166;
int32_t x7171 = x7167;
for(int x7172=0; x7172 < -7; x7172++) {
int32_t x7173 = x7169;
int32_t x7174 = x7170;
float x7175 = x7106[x7174];
int32_t x7176 = x7171;
float x7177 = x7134[x7176];
float x7178 = x7175 / x7177;
x7147[x7173] = x7178;
x7169 += 1;
if (x520) {
x7170 += 1;
} else {
}
if (x452) {
x7171 += 1;
} else {
}

}
x7162 += -7;
if (x452) {
x7163 += -7;
} else {
}
if (x452) {
x7164 += 1;
} else {
}

}
x7155 += -7;
if (x452) {
x7156 += -7;
} else {
}
if (x542) {
x7157 += 1;
} else {
}

}
x7148 += 7;
if (x452) {
x7149 += -7;
} else {
}
if (x452) {
x7150 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x7220 = (float*)myMalloc(7 * sizeof(float));;
int32_t x7221 = 0;
int32_t x7222 = 0;
int32_t x7223 = 0;
for(int x7224=0; x7224 < 1; x7224++) {
int32_t x7225 = x7222;
int32_t x7226 = x7223;
int32_t x7227 = x7221;
int32_t x7228 = x7227;
int32_t x7229 = x7225;
int32_t x7230 = x7226;
for(int x7231=0; x7231 < -1; x7231++) {
int32_t x7232 = x7229;
int32_t x7233 = x7230;
int32_t x7234 = x7228;
int32_t x7235 = x7234;
int32_t x7236 = x7232;
int32_t x7237 = x7233;
for(int x7238=0; x7238 < 1; x7238++) {
int32_t x7239 = x7236;
int32_t x7240 = x7237;
int32_t x7241 = x7235;
int32_t x7242 = x7241;
int32_t x7243 = x7239;
int32_t x7244 = x7240;
for(int x7245=0; x7245 < -7; x7245++) {
int32_t x7246 = x7242;
int32_t x7247 = x7243;
float x7248 = x7147[x7247];
int32_t x7249 = x7244;
float x7250 = x127[x7249];
float x7251 = x7248 * x7250;
x7220[x7246] = x7251;
x7242 += 1;
if (x520) {
x7243 += 1;
} else {
}
if (x452) {
x7244 += 1;
} else {
}

}
x7235 += -7;
if (x452) {
x7236 += -7;
} else {
}
if (x452) {
x7237 += 1;
} else {
}

}
x7228 += -7;
if (x542) {
x7229 += -7;
} else {
}
if (x542) {
x7230 += 1;
} else {
}

}
x7221 += 7;
if (x452) {
x7222 += 7;
} else {
}
if (x452) {
x7223 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x7293 = (float*)myMalloc(7 * sizeof(float));;
int32_t x7294 = 0;
int32_t x7295 = 0;
int32_t x7296 = 0;
for(int x7297=0; x7297 < 1; x7297++) {
int32_t x7298 = x7295;
int32_t x7299 = x7296;
int32_t x7300 = x7294;
int32_t x7301 = x7300;
int32_t x7302 = x7298;
int32_t x7303 = x7299;
for(int x7304=0; x7304 < -1; x7304++) {
int32_t x7305 = x7302;
int32_t x7306 = x7303;
int32_t x7307 = x7301;
int32_t x7308 = x7307;
int32_t x7309 = x7305;
int32_t x7310 = x7306;
for(int x7311=0; x7311 < 1; x7311++) {
int32_t x7312 = x7309;
int32_t x7313 = x7310;
int32_t x7314 = x7308;
int32_t x7315 = x7314;
int32_t x7316 = x7312;
int32_t x7317 = x7313;
for(int x7318=0; x7318 < -7; x7318++) {
int32_t x7319 = x7315;
int32_t x7320 = x7316;
float x7321 = x7220[x7320];
int32_t x7322 = x7317;
float x7323 = x61[x7322];
float x7324 = x7321 + x7323;
x7293[x7319] = x7324;
x7315 += 1;
if (x520) {
x7316 += 1;
} else {
}
if (x452) {
x7317 += 1;
} else {
}

}
x7308 += -7;
if (x452) {
x7309 += -7;
} else {
}
if (x452) {
x7310 += 1;
} else {
}

}
x7301 += -7;
if (x542) {
x7302 += -7;
} else {
}
if (x542) {
x7303 += 1;
} else {
}

}
x7294 += 7;
if (x452) {
x7295 += 7;
} else {
}
if (x452) {
x7296 += -1;
} else {
}

}
float* x7362 = (float*)myMalloc(7 * sizeof(float));;
for(int x7363=0; x7363 < 7; x7363++) {
float x7364 = x7293[x7363];
bool x7365 = x7364 < 0.0f;
if (x7365) {
x7362[x7363] = 0.0f;
} else {
float x7368 = x7293[x7363];
x7362[x7363] = x7368;
}

}
float* x7374 = (float*)myMalloc(x4967 * sizeof(float));;
float* x7375 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x7376=0; x7376 < 1; x7376++) {
int32_t x7377 = x7376 * 7;
float* x7378 = x7362+x7377;
int32_t x7379 = x7376 * x4965;
float* x7380 = x7374+x7379;
int32_t x7381 = x7376 * x1520;
float* x7382 = x7375+x7381;
for(int x7383=0; x7383 < -1; x7383++) {
int32_t x7384 = x7383 / 1;
int32_t x7388 = x7384 * x1150;
int32_t x7389 = x7388 * x1152;
int32_t x7385 = x7383 % 1;
int32_t x7386 = x7385 / 1;
int32_t x7390 = x7386 * x1150;
int32_t x7391 = x7390 * x1152;
int32_t x7392 = x7389 + x7391;
int32_t x7387 = x7385 % 1;
int32_t x7393 = x7387 * x1152;
int32_t x7394 = x7393 * x1152;
int32_t x7395 = x7392 + x7394;
float* x7396 = x7382+x7395;
int32_t x7397 = x7384 * -7;
float* x7398 = x7378+x7397;
for(int x7399=0; x7399 < x1150; x7399++) {
int32_t x7401 = x7399 * x1152;
float* x7402 = x7396+x7401;
int32_t x7400 = x7399 + x7386;
int32_t x7403 = x7400 * -7;
int32_t x7404 = x7403 + x7387;
float* x7405 = x7398+x7404;
memcpy(x7402, x7405, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x1153,-1,1,x41,-1,x7382,x1153,1,x7380,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(512) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x7418 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x7419 = 0;
int32_t x7420 = 0;
int32_t x7421 = 0;
for(int x7422=0; x7422 < -7; x7422++) {
int32_t x7423 = x7419;
int32_t x7424 = x7420;
float x7425 = x7374[x7424];
int32_t x7426 = x7421;
float x7427 = x25[x7426];
float x7428 = x7425 - x7427;
x7418[x7423] = x7428;
x7419 += 1;
if (x452) {
x7420 += x4965;
} else {
}
if (x452) {
x7421 += -1;
} else {
}

}
float* x7439 = (float*)myMalloc(512 * sizeof(float));;
for(int x7440=0; x7440 < 512; x7440++) {
float x7441 = x223[x7440];
float x7442 = x7441 + 1.0E-5f;
x7439[x7440] = x7442;

}
float* x7446 = (float*)myMalloc(512 * sizeof(float));;
for(int x7447=0; x7447 < 512; x7447++) {
float x7448 = x7439[x7447];
double x7449 = (double)x7448;
double x7450 = sqrt(x7449);
float x7451 = (float)x7450;
x7446[x7447] = x7451;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x7459 = (float*)myMalloc(7 * sizeof(float));;
int32_t x7460 = 0;
int32_t x7461 = 0;
int32_t x7462 = 0;
for(int x7463=0; x7463 < 1; x7463++) {
int32_t x7464 = x7461;
int32_t x7465 = x7462;
int32_t x7466 = x7460;
int32_t x7467 = x7466;
int32_t x7468 = x7464;
int32_t x7469 = x7465;
for(int x7470=0; x7470 < -1; x7470++) {
int32_t x7471 = x7468;
int32_t x7472 = x7469;
int32_t x7473 = x7467;
int32_t x7474 = x7473;
int32_t x7475 = x7471;
int32_t x7476 = x7472;
for(int x7477=0; x7477 < 1; x7477++) {
int32_t x7478 = x7475;
int32_t x7479 = x7476;
int32_t x7480 = x7474;
int32_t x7481 = x7480;
int32_t x7482 = x7478;
int32_t x7483 = x7479;
for(int x7484=0; x7484 < -7; x7484++) {
int32_t x7485 = x7481;
int32_t x7486 = x7482;
float x7487 = x7418[x7486];
int32_t x7488 = x7483;
float x7489 = x7446[x7488];
float x7490 = x7487 / x7489;
x7459[x7485] = x7490;
x7481 += 1;
if (x520) {
x7482 += 1;
} else {
}
if (x452) {
x7483 += 1;
} else {
}

}
x7474 += -7;
if (x452) {
x7475 += -7;
} else {
}
if (x452) {
x7476 += 1;
} else {
}

}
x7467 += -7;
if (x452) {
x7468 += -7;
} else {
}
if (x542) {
x7469 += 1;
} else {
}

}
x7460 += 7;
if (x452) {
x7461 += -7;
} else {
}
if (x452) {
x7462 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x7532 = (float*)myMalloc(7 * sizeof(float));;
int32_t x7533 = 0;
int32_t x7534 = 0;
int32_t x7535 = 0;
for(int x7536=0; x7536 < 1; x7536++) {
int32_t x7537 = x7534;
int32_t x7538 = x7535;
int32_t x7539 = x7533;
int32_t x7540 = x7539;
int32_t x7541 = x7537;
int32_t x7542 = x7538;
for(int x7543=0; x7543 < -1; x7543++) {
int32_t x7544 = x7541;
int32_t x7545 = x7542;
int32_t x7546 = x7540;
int32_t x7547 = x7546;
int32_t x7548 = x7544;
int32_t x7549 = x7545;
for(int x7550=0; x7550 < 1; x7550++) {
int32_t x7551 = x7548;
int32_t x7552 = x7549;
int32_t x7553 = x7547;
int32_t x7554 = x7553;
int32_t x7555 = x7551;
int32_t x7556 = x7552;
for(int x7557=0; x7557 < -7; x7557++) {
int32_t x7558 = x7554;
int32_t x7559 = x7555;
float x7560 = x7459[x7559];
int32_t x7561 = x7556;
float x7562 = x167[x7561];
float x7563 = x7560 * x7562;
x7532[x7558] = x7563;
x7554 += 1;
if (x520) {
x7555 += 1;
} else {
}
if (x452) {
x7556 += 1;
} else {
}

}
x7547 += -7;
if (x452) {
x7548 += -7;
} else {
}
if (x452) {
x7549 += 1;
} else {
}

}
x7540 += -7;
if (x542) {
x7541 += -7;
} else {
}
if (x542) {
x7542 += 1;
} else {
}

}
x7533 += 7;
if (x452) {
x7534 += 7;
} else {
}
if (x452) {
x7535 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x7605 = (float*)myMalloc(7 * sizeof(float));;
int32_t x7606 = 0;
int32_t x7607 = 0;
int32_t x7608 = 0;
for(int x7609=0; x7609 < 1; x7609++) {
int32_t x7610 = x7607;
int32_t x7611 = x7608;
int32_t x7612 = x7606;
int32_t x7613 = x7612;
int32_t x7614 = x7610;
int32_t x7615 = x7611;
for(int x7616=0; x7616 < -1; x7616++) {
int32_t x7617 = x7614;
int32_t x7618 = x7615;
int32_t x7619 = x7613;
int32_t x7620 = x7619;
int32_t x7621 = x7617;
int32_t x7622 = x7618;
for(int x7623=0; x7623 < 1; x7623++) {
int32_t x7624 = x7621;
int32_t x7625 = x7622;
int32_t x7626 = x7620;
int32_t x7627 = x7626;
int32_t x7628 = x7624;
int32_t x7629 = x7625;
for(int x7630=0; x7630 < -7; x7630++) {
int32_t x7631 = x7627;
int32_t x7632 = x7628;
float x7633 = x7532[x7632];
int32_t x7634 = x7629;
float x7635 = x82[x7634];
float x7636 = x7633 + x7635;
x7605[x7631] = x7636;
x7627 += 1;
if (x520) {
x7628 += 1;
} else {
}
if (x452) {
x7629 += 1;
} else {
}

}
x7620 += -7;
if (x452) {
x7621 += -7;
} else {
}
if (x452) {
x7622 += 1;
} else {
}

}
x7613 += -7;
if (x542) {
x7614 += -7;
} else {
}
if (x542) {
x7615 += 1;
} else {
}

}
x7606 += 7;
if (x452) {
x7607 += 7;
} else {
}
if (x452) {
x7608 += -1;
} else {
}

}
int32_t x7674 = 0;
int32_t x7675 = 0;
int32_t x7676 = 0;
for(int x7677=0; x7677 < 1; x7677++) {
int32_t x7678 = x7675;
int32_t x7679 = x7676;
int32_t x7680 = x7674;
int32_t x7681 = x7680;
int32_t x7682 = x7678;
int32_t x7683 = x7679;
for(int x7684=0; x7684 < -1; x7684++) {
int32_t x7685 = x7682;
int32_t x7686 = x7683;
int32_t x7687 = x7681;
int32_t x7688 = x7687;
int32_t x7689 = x7685;
int32_t x7690 = x7686;
for(int x7691=0; x7691 < 1; x7691++) {
int32_t x7692 = x7689;
int32_t x7693 = x7690;
int32_t x7694 = x7688;
int32_t x7695 = x7694;
int32_t x7696 = x7692;
int32_t x7697 = x7693;
for(int x7698=0; x7698 < -7; x7698++) {
int32_t x7699 = x7696;
float x7700 = x7605[x7699];
int32_t x7701 = x7697;
float x7702 = x6694[x7701];
float x7703 = x7700 + x7702;
x7605[x7699] = x7703;
x7695 += 1;
if (x520) {
x7696 += 1;
} else {
}
if (x520) {
x7697 += 1;
} else {
}

}
x7688 += -7;
if (x452) {
x7689 += -7;
} else {
}
if (x452) {
x7690 += -7;
} else {
}

}
x7681 += -7;
if (x542) {
x7682 += -7;
} else {
}
if (x542) {
x7683 += -7;
} else {
}

}
x7674 += 7;
if (x452) {
x7675 += 7;
} else {
}
if (x452) {
x7676 += 7;
} else {
}

}
float* x7741 = (float*)myMalloc(7 * sizeof(float));;
for(int x7742=0; x7742 < 7; x7742++) {
float x7743 = x7605[x7742];
bool x7744 = x7743 < 0.0f;
if (x7744) {
x7741[x7742] = 0.0f;
} else {
float x7747 = x7605[x7742];
x7741[x7742] = x7747;
}

}
float* x7753 = (float*)myMalloc(x4300 * sizeof(float));;
float* x7754 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x7755=0; x7755 < 1; x7755++) {
int32_t x7756 = x7755 * 7;
float* x7757 = x7741+x7756;
int32_t x7758 = x7755 * x4298;
float* x7759 = x7753+x7758;
int32_t x7760 = x7755 * x1520;
float* x7761 = x7754+x7760;
for(int x7762=0; x7762 < -1; x7762++) {
int32_t x7763 = x7762 / 1;
int32_t x7767 = x7763 * x1150;
int32_t x7768 = x7767 * x1152;
int32_t x7764 = x7762 % 1;
int32_t x7765 = x7764 / 1;
int32_t x7769 = x7765 * x1150;
int32_t x7770 = x7769 * x1152;
int32_t x7771 = x7768 + x7770;
int32_t x7766 = x7764 % 1;
int32_t x7772 = x7766 * x1152;
int32_t x7773 = x7772 * x1152;
int32_t x7774 = x7771 + x7773;
float* x7775 = x7761+x7774;
int32_t x7776 = x7763 * -7;
float* x7777 = x7757+x7776;
for(int x7778=0; x7778 < x1150; x7778++) {
int32_t x7780 = x7778 * x1152;
float* x7781 = x7775+x7780;
int32_t x7779 = x7778 + x7765;
int32_t x7782 = x7779 * -7;
int32_t x7783 = x7782 + x7766;
float* x7784 = x7777+x7783;
memcpy(x7781, x7784, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x1153,-1,1,x132,-1,x7761,x1153,1,x7759,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(128) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x7797 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x7798 = 0;
int32_t x7799 = 0;
int32_t x7800 = 0;
for(int x7801=0; x7801 < -7; x7801++) {
int32_t x7802 = x7798;
int32_t x7803 = x7799;
float x7804 = x7753[x7803];
int32_t x7805 = x7800;
float x7806 = x236[x7805];
float x7807 = x7804 - x7806;
x7797[x7802] = x7807;
x7798 += 1;
if (x452) {
x7799 += x4298;
} else {
}
if (x452) {
x7800 += -1;
} else {
}

}
float* x7818 = (float*)myMalloc(128 * sizeof(float));;
for(int x7819=0; x7819 < 128; x7819++) {
float x7820 = x261[x7819];
float x7821 = x7820 + 1.0E-5f;
x7818[x7819] = x7821;

}
float* x7825 = (float*)myMalloc(128 * sizeof(float));;
for(int x7826=0; x7826 < 128; x7826++) {
float x7827 = x7818[x7826];
double x7828 = (double)x7827;
double x7829 = sqrt(x7828);
float x7830 = (float)x7829;
x7825[x7826] = x7830;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x7838 = (float*)myMalloc(7 * sizeof(float));;
int32_t x7839 = 0;
int32_t x7840 = 0;
int32_t x7841 = 0;
for(int x7842=0; x7842 < 1; x7842++) {
int32_t x7843 = x7840;
int32_t x7844 = x7841;
int32_t x7845 = x7839;
int32_t x7846 = x7845;
int32_t x7847 = x7843;
int32_t x7848 = x7844;
for(int x7849=0; x7849 < -1; x7849++) {
int32_t x7850 = x7847;
int32_t x7851 = x7848;
int32_t x7852 = x7846;
int32_t x7853 = x7852;
int32_t x7854 = x7850;
int32_t x7855 = x7851;
for(int x7856=0; x7856 < 1; x7856++) {
int32_t x7857 = x7854;
int32_t x7858 = x7855;
int32_t x7859 = x7853;
int32_t x7860 = x7859;
int32_t x7861 = x7857;
int32_t x7862 = x7858;
for(int x7863=0; x7863 < -7; x7863++) {
int32_t x7864 = x7860;
int32_t x7865 = x7861;
float x7866 = x7797[x7865];
int32_t x7867 = x7862;
float x7868 = x7825[x7867];
float x7869 = x7866 / x7868;
x7838[x7864] = x7869;
x7860 += 1;
if (x520) {
x7861 += 1;
} else {
}
if (x452) {
x7862 += 1;
} else {
}

}
x7853 += -7;
if (x452) {
x7854 += -7;
} else {
}
if (x452) {
x7855 += 1;
} else {
}

}
x7846 += -7;
if (x452) {
x7847 += -7;
} else {
}
if (x542) {
x7848 += 1;
} else {
}

}
x7839 += 7;
if (x452) {
x7840 += -7;
} else {
}
if (x452) {
x7841 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x7911 = (float*)myMalloc(7 * sizeof(float));;
int32_t x7912 = 0;
int32_t x7913 = 0;
int32_t x7914 = 0;
for(int x7915=0; x7915 < 1; x7915++) {
int32_t x7916 = x7913;
int32_t x7917 = x7914;
int32_t x7918 = x7912;
int32_t x7919 = x7918;
int32_t x7920 = x7916;
int32_t x7921 = x7917;
for(int x7922=0; x7922 < -1; x7922++) {
int32_t x7923 = x7920;
int32_t x7924 = x7921;
int32_t x7925 = x7919;
int32_t x7926 = x7925;
int32_t x7927 = x7923;
int32_t x7928 = x7924;
for(int x7929=0; x7929 < 1; x7929++) {
int32_t x7930 = x7927;
int32_t x7931 = x7928;
int32_t x7932 = x7926;
int32_t x7933 = x7932;
int32_t x7934 = x7930;
int32_t x7935 = x7931;
for(int x7936=0; x7936 < -7; x7936++) {
int32_t x7937 = x7933;
int32_t x7938 = x7934;
float x7939 = x7838[x7938];
int32_t x7940 = x7935;
float x7941 = x39[x7940];
float x7942 = x7939 * x7941;
x7911[x7937] = x7942;
x7933 += 1;
if (x520) {
x7934 += 1;
} else {
}
if (x452) {
x7935 += 1;
} else {
}

}
x7926 += -7;
if (x452) {
x7927 += -7;
} else {
}
if (x452) {
x7928 += 1;
} else {
}

}
x7919 += -7;
if (x542) {
x7920 += -7;
} else {
}
if (x542) {
x7921 += 1;
} else {
}

}
x7912 += 7;
if (x452) {
x7913 += 7;
} else {
}
if (x452) {
x7914 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x7984 = (float*)myMalloc(7 * sizeof(float));;
int32_t x7985 = 0;
int32_t x7986 = 0;
int32_t x7987 = 0;
for(int x7988=0; x7988 < 1; x7988++) {
int32_t x7989 = x7986;
int32_t x7990 = x7987;
int32_t x7991 = x7985;
int32_t x7992 = x7991;
int32_t x7993 = x7989;
int32_t x7994 = x7990;
for(int x7995=0; x7995 < -1; x7995++) {
int32_t x7996 = x7993;
int32_t x7997 = x7994;
int32_t x7998 = x7992;
int32_t x7999 = x7998;
int32_t x8000 = x7996;
int32_t x8001 = x7997;
for(int x8002=0; x8002 < 1; x8002++) {
int32_t x8003 = x8000;
int32_t x8004 = x8001;
int32_t x8005 = x7999;
int32_t x8006 = x8005;
int32_t x8007 = x8003;
int32_t x8008 = x8004;
for(int x8009=0; x8009 < -7; x8009++) {
int32_t x8010 = x8006;
int32_t x8011 = x8007;
float x8012 = x7911[x8011];
int32_t x8013 = x8008;
float x8014 = x242[x8013];
float x8015 = x8012 + x8014;
x7984[x8010] = x8015;
x8006 += 1;
if (x520) {
x8007 += 1;
} else {
}
if (x452) {
x8008 += 1;
} else {
}

}
x7999 += -7;
if (x452) {
x8000 += -7;
} else {
}
if (x452) {
x8001 += 1;
} else {
}

}
x7992 += -7;
if (x542) {
x7993 += -7;
} else {
}
if (x542) {
x7994 += 1;
} else {
}

}
x7985 += 7;
if (x452) {
x7986 += 7;
} else {
}
if (x452) {
x7987 += -1;
} else {
}

}
float* x8053 = (float*)myMalloc(7 * sizeof(float));;
for(int x8054=0; x8054 < 7; x8054++) {
float x8055 = x7984[x8054];
bool x8056 = x8055 < 0.0f;
if (x8056) {
x8053[x8054] = 0.0f;
} else {
float x8059 = x7984[x8054];
x8053[x8054] = x8059;
}

}
float* x8065 = (float*)myMalloc(x4300 * sizeof(float));;
float* x8066 = (float*)myMalloc(x1158 * sizeof(float));;
for(int x8067=0; x8067 < 1; x8067++) {
int32_t x8068 = x8067 * 7;
float* x8069 = x8053+x8068;
int32_t x8070 = x8067 * x4298;
float* x8071 = x8065+x8070;
int32_t x8072 = x8067 * x1158;
float* x8073 = x8066+x8072;
for(int x8074=0; x8074 < -9; x8074++) {
int32_t x8075 = x8074 / 9;
int32_t x8079 = x8075 * 3;
int32_t x8080 = x8079 * 3;
int32_t x8081 = x8080 * x1150;
int32_t x8082 = x8081 * x1152;
int32_t x8076 = x8074 % 9;
int32_t x8077 = x8076 / 3;
int32_t x8083 = x8077 * 3;
int32_t x8084 = x8083 * x1150;
int32_t x8085 = x8084 * x1152;
int32_t x8086 = x8082 + x8085;
int32_t x8078 = x8076 % 3;
int32_t x8087 = x8078 * x1152;
int32_t x8088 = x8087 * x1152;
int32_t x8089 = x8086 + x8088;
float* x8090 = x8073+x8089;
int32_t x8091 = x8075 * -7;
float* x8092 = x8069+x8091;
int32_t x8104 = 1 - x8078;
bool x8105 = x8104 > 0;
int32_t x8106;
if (x8105) {
x8106 = x8104;
} else {
x8106 = 0;
}
int32_t x8107 = 3 - x8078;
int32_t x8108 = x8107 - 1;
int32_t x8109 = 1 - x8108;
bool x8110 = x8109 > 0;
int32_t x8111;
if (x8110) {
x8111 = x8109;
} else {
x8111 = 0;
}
int32_t x8112 = x1152 - x8111;
int32_t x8113 = x8112 - x8106;
bool x8114 = x8113 <= 0;
bool x8118 = x8106 > 0;
int32_t x8103 = -1 + x8078;
bool x8131 = x8111 > 0;
for(int x8093=0; x8093 < x1150; x8093++) {
int32_t x8094 = x8093 - 1;
int32_t x8095 = x8094 + x8077;
bool x8096 = x8095 < 0;
bool x8097 = x8095 >= 1;
bool x8098 = x8096 || x8097;
if (x8098) {
int32_t x8099 = x8093 * x1152;
float* x8100 = x8090+x8099;
memset(x8100, 0, 4 * x1152);;
} else {
if (x8114) {
int32_t x8099 = x8093 * x1152;
float* x8115 = x8090+x8099;
memset(x8115, 0, 4 * x1152);;
} else {
int32_t x8099 = x8093 * x1152;
if (x8118) {
float* x8119 = x8090+x8099;
memset(x8119, 0, 4 * x8106);;
} else {
}
// may have segfault here
int32_t x8124 = x8099 + x8106;
float* x8125 = x8090+x8124;
int32_t x8126 = x8095 * -7;
int32_t x8127 = x8126 + x8103;
int32_t x8128 = x8127 + x8106;
float* x8129 = x8092+x8128;
memcpy(x8125, x8129, 4 * x8113);;
if (x8131) {
int32_t x8132 = x8099 + x1152;
int32_t x8133 = x8132 - x8111;
float* x8134 = x8090+x8133;
memset(x8134, 0, 4 * x8111);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 128,x1153,-9,1,x165,-9,x8073,x1153,1,x8071,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(128) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x8153 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x8154 = 0;
int32_t x8155 = 0;
int32_t x8156 = 0;
for(int x8157=0; x8157 < -7; x8157++) {
int32_t x8158 = x8154;
int32_t x8159 = x8155;
float x8160 = x8065[x8159];
int32_t x8161 = x8156;
float x8162 = x268[x8161];
float x8163 = x8160 - x8162;
x8153[x8158] = x8163;
x8154 += 1;
if (x452) {
x8155 += x4298;
} else {
}
if (x452) {
x8156 += -1;
} else {
}

}
float* x8174 = (float*)myMalloc(128 * sizeof(float));;
for(int x8175=0; x8175 < 128; x8175++) {
float x8176 = x148[x8175];
float x8177 = x8176 + 1.0E-5f;
x8174[x8175] = x8177;

}
float* x8181 = (float*)myMalloc(128 * sizeof(float));;
for(int x8182=0; x8182 < 128; x8182++) {
float x8183 = x8174[x8182];
double x8184 = (double)x8183;
double x8185 = sqrt(x8184);
float x8186 = (float)x8185;
x8181[x8182] = x8186;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x8194 = (float*)myMalloc(7 * sizeof(float));;
int32_t x8195 = 0;
int32_t x8196 = 0;
int32_t x8197 = 0;
for(int x8198=0; x8198 < 1; x8198++) {
int32_t x8199 = x8196;
int32_t x8200 = x8197;
int32_t x8201 = x8195;
int32_t x8202 = x8201;
int32_t x8203 = x8199;
int32_t x8204 = x8200;
for(int x8205=0; x8205 < -1; x8205++) {
int32_t x8206 = x8203;
int32_t x8207 = x8204;
int32_t x8208 = x8202;
int32_t x8209 = x8208;
int32_t x8210 = x8206;
int32_t x8211 = x8207;
for(int x8212=0; x8212 < 1; x8212++) {
int32_t x8213 = x8210;
int32_t x8214 = x8211;
int32_t x8215 = x8209;
int32_t x8216 = x8215;
int32_t x8217 = x8213;
int32_t x8218 = x8214;
for(int x8219=0; x8219 < -7; x8219++) {
int32_t x8220 = x8216;
int32_t x8221 = x8217;
float x8222 = x8153[x8221];
int32_t x8223 = x8218;
float x8224 = x8181[x8223];
float x8225 = x8222 / x8224;
x8194[x8220] = x8225;
x8216 += 1;
if (x520) {
x8217 += 1;
} else {
}
if (x452) {
x8218 += 1;
} else {
}

}
x8209 += -7;
if (x452) {
x8210 += -7;
} else {
}
if (x452) {
x8211 += 1;
} else {
}

}
x8202 += -7;
if (x452) {
x8203 += -7;
} else {
}
if (x542) {
x8204 += 1;
} else {
}

}
x8195 += 7;
if (x452) {
x8196 += -7;
} else {
}
if (x452) {
x8197 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x8267 = (float*)myMalloc(7 * sizeof(float));;
int32_t x8268 = 0;
int32_t x8269 = 0;
int32_t x8270 = 0;
for(int x8271=0; x8271 < 1; x8271++) {
int32_t x8272 = x8269;
int32_t x8273 = x8270;
int32_t x8274 = x8268;
int32_t x8275 = x8274;
int32_t x8276 = x8272;
int32_t x8277 = x8273;
for(int x8278=0; x8278 < -1; x8278++) {
int32_t x8279 = x8276;
int32_t x8280 = x8277;
int32_t x8281 = x8275;
int32_t x8282 = x8281;
int32_t x8283 = x8279;
int32_t x8284 = x8280;
for(int x8285=0; x8285 < 1; x8285++) {
int32_t x8286 = x8283;
int32_t x8287 = x8284;
int32_t x8288 = x8282;
int32_t x8289 = x8288;
int32_t x8290 = x8286;
int32_t x8291 = x8287;
for(int x8292=0; x8292 < -7; x8292++) {
int32_t x8293 = x8289;
int32_t x8294 = x8290;
float x8295 = x8194[x8294];
int32_t x8296 = x8291;
float x8297 = x79[x8296];
float x8298 = x8295 * x8297;
x8267[x8293] = x8298;
x8289 += 1;
if (x520) {
x8290 += 1;
} else {
}
if (x452) {
x8291 += 1;
} else {
}

}
x8282 += -7;
if (x452) {
x8283 += -7;
} else {
}
if (x452) {
x8284 += 1;
} else {
}

}
x8275 += -7;
if (x542) {
x8276 += -7;
} else {
}
if (x542) {
x8277 += 1;
} else {
}

}
x8268 += 7;
if (x452) {
x8269 += 7;
} else {
}
if (x452) {
x8270 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x8340 = (float*)myMalloc(7 * sizeof(float));;
int32_t x8341 = 0;
int32_t x8342 = 0;
int32_t x8343 = 0;
for(int x8344=0; x8344 < 1; x8344++) {
int32_t x8345 = x8342;
int32_t x8346 = x8343;
int32_t x8347 = x8341;
int32_t x8348 = x8347;
int32_t x8349 = x8345;
int32_t x8350 = x8346;
for(int x8351=0; x8351 < -1; x8351++) {
int32_t x8352 = x8349;
int32_t x8353 = x8350;
int32_t x8354 = x8348;
int32_t x8355 = x8354;
int32_t x8356 = x8352;
int32_t x8357 = x8353;
for(int x8358=0; x8358 < 1; x8358++) {
int32_t x8359 = x8356;
int32_t x8360 = x8357;
int32_t x8361 = x8355;
int32_t x8362 = x8361;
int32_t x8363 = x8359;
int32_t x8364 = x8360;
for(int x8365=0; x8365 < -7; x8365++) {
int32_t x8366 = x8362;
int32_t x8367 = x8363;
float x8368 = x8267[x8367];
int32_t x8369 = x8364;
float x8370 = x38[x8369];
float x8371 = x8368 + x8370;
x8340[x8366] = x8371;
x8362 += 1;
if (x520) {
x8363 += 1;
} else {
}
if (x452) {
x8364 += 1;
} else {
}

}
x8355 += -7;
if (x452) {
x8356 += -7;
} else {
}
if (x452) {
x8357 += 1;
} else {
}

}
x8348 += -7;
if (x542) {
x8349 += -7;
} else {
}
if (x542) {
x8350 += 1;
} else {
}

}
x8341 += 7;
if (x452) {
x8342 += 7;
} else {
}
if (x452) {
x8343 += -1;
} else {
}

}
float* x8409 = (float*)myMalloc(7 * sizeof(float));;
for(int x8410=0; x8410 < 7; x8410++) {
float x8411 = x8340[x8410];
bool x8412 = x8411 < 0.0f;
if (x8412) {
x8409[x8410] = 0.0f;
} else {
float x8415 = x8340[x8410];
x8409[x8410] = x8415;
}

}
float* x8421 = (float*)myMalloc(x4967 * sizeof(float));;
float* x8422 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x8423=0; x8423 < 1; x8423++) {
int32_t x8424 = x8423 * 7;
float* x8425 = x8409+x8424;
int32_t x8426 = x8423 * x4965;
float* x8427 = x8421+x8426;
int32_t x8428 = x8423 * x1520;
float* x8429 = x8422+x8428;
for(int x8430=0; x8430 < -1; x8430++) {
int32_t x8431 = x8430 / 1;
int32_t x8435 = x8431 * x1150;
int32_t x8436 = x8435 * x1152;
int32_t x8432 = x8430 % 1;
int32_t x8433 = x8432 / 1;
int32_t x8437 = x8433 * x1150;
int32_t x8438 = x8437 * x1152;
int32_t x8439 = x8436 + x8438;
int32_t x8434 = x8432 % 1;
int32_t x8440 = x8434 * x1152;
int32_t x8441 = x8440 * x1152;
int32_t x8442 = x8439 + x8441;
float* x8443 = x8429+x8442;
int32_t x8444 = x8431 * -7;
float* x8445 = x8425+x8444;
for(int x8446=0; x8446 < x1150; x8446++) {
int32_t x8448 = x8446 * x1152;
float* x8449 = x8443+x8448;
int32_t x8447 = x8446 + x8433;
int32_t x8450 = x8447 * -7;
int32_t x8451 = x8450 + x8434;
float* x8452 = x8445+x8451;
memcpy(x8449, x8452, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x1153,-1,1,x55,-1,x8429,x1153,1,x8427,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(512) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x8465 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x8466 = 0;
int32_t x8467 = 0;
int32_t x8468 = 0;
for(int x8469=0; x8469 < -7; x8469++) {
int32_t x8470 = x8466;
int32_t x8471 = x8467;
float x8472 = x8421[x8471];
int32_t x8473 = x8468;
float x8474 = x19[x8473];
float x8475 = x8472 - x8474;
x8465[x8470] = x8475;
x8466 += 1;
if (x452) {
x8467 += x4965;
} else {
}
if (x452) {
x8468 += -1;
} else {
}

}
float* x8486 = (float*)myMalloc(512 * sizeof(float));;
for(int x8487=0; x8487 < 512; x8487++) {
float x8488 = x234[x8487];
float x8489 = x8488 + 1.0E-5f;
x8486[x8487] = x8489;

}
float* x8493 = (float*)myMalloc(512 * sizeof(float));;
for(int x8494=0; x8494 < 512; x8494++) {
float x8495 = x8486[x8494];
double x8496 = (double)x8495;
double x8497 = sqrt(x8496);
float x8498 = (float)x8497;
x8493[x8494] = x8498;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x8506 = (float*)myMalloc(7 * sizeof(float));;
int32_t x8507 = 0;
int32_t x8508 = 0;
int32_t x8509 = 0;
for(int x8510=0; x8510 < 1; x8510++) {
int32_t x8511 = x8508;
int32_t x8512 = x8509;
int32_t x8513 = x8507;
int32_t x8514 = x8513;
int32_t x8515 = x8511;
int32_t x8516 = x8512;
for(int x8517=0; x8517 < -1; x8517++) {
int32_t x8518 = x8515;
int32_t x8519 = x8516;
int32_t x8520 = x8514;
int32_t x8521 = x8520;
int32_t x8522 = x8518;
int32_t x8523 = x8519;
for(int x8524=0; x8524 < 1; x8524++) {
int32_t x8525 = x8522;
int32_t x8526 = x8523;
int32_t x8527 = x8521;
int32_t x8528 = x8527;
int32_t x8529 = x8525;
int32_t x8530 = x8526;
for(int x8531=0; x8531 < -7; x8531++) {
int32_t x8532 = x8528;
int32_t x8533 = x8529;
float x8534 = x8465[x8533];
int32_t x8535 = x8530;
float x8536 = x8493[x8535];
float x8537 = x8534 / x8536;
x8506[x8532] = x8537;
x8528 += 1;
if (x520) {
x8529 += 1;
} else {
}
if (x452) {
x8530 += 1;
} else {
}

}
x8521 += -7;
if (x452) {
x8522 += -7;
} else {
}
if (x452) {
x8523 += 1;
} else {
}

}
x8514 += -7;
if (x452) {
x8515 += -7;
} else {
}
if (x542) {
x8516 += 1;
} else {
}

}
x8507 += 7;
if (x452) {
x8508 += -7;
} else {
}
if (x452) {
x8509 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x8579 = (float*)myMalloc(7 * sizeof(float));;
int32_t x8580 = 0;
int32_t x8581 = 0;
int32_t x8582 = 0;
for(int x8583=0; x8583 < 1; x8583++) {
int32_t x8584 = x8581;
int32_t x8585 = x8582;
int32_t x8586 = x8580;
int32_t x8587 = x8586;
int32_t x8588 = x8584;
int32_t x8589 = x8585;
for(int x8590=0; x8590 < -1; x8590++) {
int32_t x8591 = x8588;
int32_t x8592 = x8589;
int32_t x8593 = x8587;
int32_t x8594 = x8593;
int32_t x8595 = x8591;
int32_t x8596 = x8592;
for(int x8597=0; x8597 < 1; x8597++) {
int32_t x8598 = x8595;
int32_t x8599 = x8596;
int32_t x8600 = x8594;
int32_t x8601 = x8600;
int32_t x8602 = x8598;
int32_t x8603 = x8599;
for(int x8604=0; x8604 < -7; x8604++) {
int32_t x8605 = x8601;
int32_t x8606 = x8602;
float x8607 = x8506[x8606];
int32_t x8608 = x8603;
float x8609 = x156[x8608];
float x8610 = x8607 * x8609;
x8579[x8605] = x8610;
x8601 += 1;
if (x520) {
x8602 += 1;
} else {
}
if (x452) {
x8603 += 1;
} else {
}

}
x8594 += -7;
if (x452) {
x8595 += -7;
} else {
}
if (x452) {
x8596 += 1;
} else {
}

}
x8587 += -7;
if (x542) {
x8588 += -7;
} else {
}
if (x542) {
x8589 += 1;
} else {
}

}
x8580 += 7;
if (x452) {
x8581 += 7;
} else {
}
if (x452) {
x8582 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x8652 = (float*)myMalloc(7 * sizeof(float));;
int32_t x8653 = 0;
int32_t x8654 = 0;
int32_t x8655 = 0;
for(int x8656=0; x8656 < 1; x8656++) {
int32_t x8657 = x8654;
int32_t x8658 = x8655;
int32_t x8659 = x8653;
int32_t x8660 = x8659;
int32_t x8661 = x8657;
int32_t x8662 = x8658;
for(int x8663=0; x8663 < -1; x8663++) {
int32_t x8664 = x8661;
int32_t x8665 = x8662;
int32_t x8666 = x8660;
int32_t x8667 = x8666;
int32_t x8668 = x8664;
int32_t x8669 = x8665;
for(int x8670=0; x8670 < 1; x8670++) {
int32_t x8671 = x8668;
int32_t x8672 = x8669;
int32_t x8673 = x8667;
int32_t x8674 = x8673;
int32_t x8675 = x8671;
int32_t x8676 = x8672;
for(int x8677=0; x8677 < -7; x8677++) {
int32_t x8678 = x8674;
int32_t x8679 = x8675;
float x8680 = x8579[x8679];
int32_t x8681 = x8676;
float x8682 = x54[x8681];
float x8683 = x8680 + x8682;
x8652[x8678] = x8683;
x8674 += 1;
if (x520) {
x8675 += 1;
} else {
}
if (x452) {
x8676 += 1;
} else {
}

}
x8667 += -7;
if (x452) {
x8668 += -7;
} else {
}
if (x452) {
x8669 += 1;
} else {
}

}
x8660 += -7;
if (x542) {
x8661 += -7;
} else {
}
if (x542) {
x8662 += 1;
} else {
}

}
x8653 += 7;
if (x452) {
x8654 += 7;
} else {
}
if (x452) {
x8655 += -1;
} else {
}

}
int32_t x8721 = 0;
int32_t x8722 = 0;
int32_t x8723 = 0;
for(int x8724=0; x8724 < 1; x8724++) {
int32_t x8725 = x8722;
int32_t x8726 = x8723;
int32_t x8727 = x8721;
int32_t x8728 = x8727;
int32_t x8729 = x8725;
int32_t x8730 = x8726;
for(int x8731=0; x8731 < -1; x8731++) {
int32_t x8732 = x8729;
int32_t x8733 = x8730;
int32_t x8734 = x8728;
int32_t x8735 = x8734;
int32_t x8736 = x8732;
int32_t x8737 = x8733;
for(int x8738=0; x8738 < 1; x8738++) {
int32_t x8739 = x8736;
int32_t x8740 = x8737;
int32_t x8741 = x8735;
int32_t x8742 = x8741;
int32_t x8743 = x8739;
int32_t x8744 = x8740;
for(int x8745=0; x8745 < -7; x8745++) {
int32_t x8746 = x8743;
float x8747 = x8652[x8746];
int32_t x8748 = x8744;
float x8749 = x7741[x8748];
float x8750 = x8747 + x8749;
x8652[x8746] = x8750;
x8742 += 1;
if (x520) {
x8743 += 1;
} else {
}
if (x520) {
x8744 += 1;
} else {
}

}
x8735 += -7;
if (x452) {
x8736 += -7;
} else {
}
if (x452) {
x8737 += -7;
} else {
}

}
x8728 += -7;
if (x542) {
x8729 += -7;
} else {
}
if (x542) {
x8730 += -7;
} else {
}

}
x8721 += 7;
if (x452) {
x8722 += 7;
} else {
}
if (x452) {
x8723 += 7;
} else {
}

}
float* x8788 = (float*)myMalloc(7 * sizeof(float));;
for(int x8789=0; x8789 < 7; x8789++) {
float x8790 = x8652[x8789];
bool x8791 = x8790 < 0.0f;
if (x8791) {
x8788[x8789] = 0.0f;
} else {
float x8794 = x8652[x8789];
x8788[x8789] = x8794;
}

}
float* x8800 = (float*)myMalloc(x1518 * sizeof(float));;
float* x8801 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x8802=0; x8802 < 1; x8802++) {
int32_t x8803 = x8802 * 7;
float* x8804 = x8788+x8803;
int32_t x8805 = x8802 * x1516;
float* x8806 = x8800+x8805;
int32_t x8807 = x8802 * x1520;
float* x8808 = x8801+x8807;
for(int x8809=0; x8809 < -1; x8809++) {
int32_t x8810 = x8809 / 1;
int32_t x8814 = x8810 * x1150;
int32_t x8815 = x8814 * x1152;
int32_t x8811 = x8809 % 1;
int32_t x8812 = x8811 / 1;
int32_t x8816 = x8812 * x1150;
int32_t x8817 = x8816 * x1152;
int32_t x8818 = x8815 + x8817;
int32_t x8813 = x8811 % 1;
int32_t x8819 = x8813 * x1152;
int32_t x8820 = x8819 * x1152;
int32_t x8821 = x8818 + x8820;
float* x8822 = x8808+x8821;
int32_t x8823 = x8810 * -7;
float* x8824 = x8804+x8823;
for(int x8825=0; x8825 < x1150; x8825++) {
int32_t x8827 = x8825 * x1152;
float* x8828 = x8822+x8827;
int32_t x8826 = x8825 + x8812;
int32_t x8829 = x8826 * -7;
int32_t x8830 = x8829 + x8813;
float* x8831 = x8824+x8830;
memcpy(x8828, x8831, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1153,-1,1,x180,-1,x8808,x1153,1,x8806,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x8844 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x8845 = 0;
int32_t x8846 = 0;
int32_t x8847 = 0;
for(int x8848=0; x8848 < -7; x8848++) {
int32_t x8849 = x8845;
int32_t x8850 = x8846;
float x8851 = x8800[x8850];
int32_t x8852 = x8847;
float x8853 = x131[x8852];
float x8854 = x8851 - x8853;
x8844[x8849] = x8854;
x8845 += 1;
if (x452) {
x8846 += x1516;
} else {
}
if (x452) {
x8847 += -1;
} else {
}

}
float* x8865 = (float*)myMalloc(256 * sizeof(float));;
for(int x8866=0; x8866 < 256; x8866++) {
float x8867 = x198[x8866];
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
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x8885 = (float*)myMalloc(7 * sizeof(float));;
int32_t x8886 = 0;
int32_t x8887 = 0;
int32_t x8888 = 0;
for(int x8889=0; x8889 < 1; x8889++) {
int32_t x8890 = x8887;
int32_t x8891 = x8888;
int32_t x8892 = x8886;
int32_t x8893 = x8892;
int32_t x8894 = x8890;
int32_t x8895 = x8891;
for(int x8896=0; x8896 < -1; x8896++) {
int32_t x8897 = x8894;
int32_t x8898 = x8895;
int32_t x8899 = x8893;
int32_t x8900 = x8899;
int32_t x8901 = x8897;
int32_t x8902 = x8898;
for(int x8903=0; x8903 < 1; x8903++) {
int32_t x8904 = x8901;
int32_t x8905 = x8902;
int32_t x8906 = x8900;
int32_t x8907 = x8906;
int32_t x8908 = x8904;
int32_t x8909 = x8905;
for(int x8910=0; x8910 < -7; x8910++) {
int32_t x8911 = x8907;
int32_t x8912 = x8908;
float x8913 = x8844[x8912];
int32_t x8914 = x8909;
float x8915 = x8872[x8914];
float x8916 = x8913 / x8915;
x8885[x8911] = x8916;
x8907 += 1;
if (x520) {
x8908 += 1;
} else {
}
if (x452) {
x8909 += 1;
} else {
}

}
x8900 += -7;
if (x452) {
x8901 += -7;
} else {
}
if (x452) {
x8902 += 1;
} else {
}

}
x8893 += -7;
if (x452) {
x8894 += -7;
} else {
}
if (x542) {
x8895 += 1;
} else {
}

}
x8886 += 7;
if (x452) {
x8887 += -7;
} else {
}
if (x452) {
x8888 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x8958 = (float*)myMalloc(7 * sizeof(float));;
int32_t x8959 = 0;
int32_t x8960 = 0;
int32_t x8961 = 0;
for(int x8962=0; x8962 < 1; x8962++) {
int32_t x8963 = x8960;
int32_t x8964 = x8961;
int32_t x8965 = x8959;
int32_t x8966 = x8965;
int32_t x8967 = x8963;
int32_t x8968 = x8964;
for(int x8969=0; x8969 < -1; x8969++) {
int32_t x8970 = x8967;
int32_t x8971 = x8968;
int32_t x8972 = x8966;
int32_t x8973 = x8972;
int32_t x8974 = x8970;
int32_t x8975 = x8971;
for(int x8976=0; x8976 < 1; x8976++) {
int32_t x8977 = x8974;
int32_t x8978 = x8975;
int32_t x8979 = x8973;
int32_t x8980 = x8979;
int32_t x8981 = x8977;
int32_t x8982 = x8978;
for(int x8983=0; x8983 < -7; x8983++) {
int32_t x8984 = x8980;
int32_t x8985 = x8981;
float x8986 = x8885[x8985];
int32_t x8987 = x8982;
float x8988 = x270[x8987];
float x8989 = x8986 * x8988;
x8958[x8984] = x8989;
x8980 += 1;
if (x520) {
x8981 += 1;
} else {
}
if (x452) {
x8982 += 1;
} else {
}

}
x8973 += -7;
if (x452) {
x8974 += -7;
} else {
}
if (x452) {
x8975 += 1;
} else {
}

}
x8966 += -7;
if (x542) {
x8967 += -7;
} else {
}
if (x542) {
x8968 += 1;
} else {
}

}
x8959 += 7;
if (x452) {
x8960 += 7;
} else {
}
if (x452) {
x8961 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x9031 = (float*)myMalloc(7 * sizeof(float));;
int32_t x9032 = 0;
int32_t x9033 = 0;
int32_t x9034 = 0;
for(int x9035=0; x9035 < 1; x9035++) {
int32_t x9036 = x9033;
int32_t x9037 = x9034;
int32_t x9038 = x9032;
int32_t x9039 = x9038;
int32_t x9040 = x9036;
int32_t x9041 = x9037;
for(int x9042=0; x9042 < -1; x9042++) {
int32_t x9043 = x9040;
int32_t x9044 = x9041;
int32_t x9045 = x9039;
int32_t x9046 = x9045;
int32_t x9047 = x9043;
int32_t x9048 = x9044;
for(int x9049=0; x9049 < 1; x9049++) {
int32_t x9050 = x9047;
int32_t x9051 = x9048;
int32_t x9052 = x9046;
int32_t x9053 = x9052;
int32_t x9054 = x9050;
int32_t x9055 = x9051;
for(int x9056=0; x9056 < -7; x9056++) {
int32_t x9057 = x9053;
int32_t x9058 = x9054;
float x9059 = x8958[x9058];
int32_t x9060 = x9055;
float x9061 = x21[x9060];
float x9062 = x9059 + x9061;
x9031[x9057] = x9062;
x9053 += 1;
if (x520) {
x9054 += 1;
} else {
}
if (x452) {
x9055 += 1;
} else {
}

}
x9046 += -7;
if (x452) {
x9047 += -7;
} else {
}
if (x452) {
x9048 += 1;
} else {
}

}
x9039 += -7;
if (x542) {
x9040 += -7;
} else {
}
if (x542) {
x9041 += 1;
} else {
}

}
x9032 += 7;
if (x452) {
x9033 += 7;
} else {
}
if (x452) {
x9034 += -1;
} else {
}

}
float* x9100 = (float*)myMalloc(7 * sizeof(float));;
for(int x9101=0; x9101 < 7; x9101++) {
float x9102 = x9031[x9101];
bool x9103 = x9102 < 0.0f;
if (x9103) {
x9100[x9101] = 0.0f;
} else {
float x9106 = x9031[x9101];
x9100[x9101] = x9106;
}

}
float* x9115 = (float*)myMalloc(x9114 * sizeof(float));;
float* x9116 = (float*)myMalloc(x4623 * sizeof(float));;
for(int x9117=0; x9117 < 1; x9117++) {
int32_t x9118 = x9117 * 7;
float* x9119 = x9100+x9118;
int32_t x9120 = x9117 * x9112;
float* x9121 = x9115+x9120;
int32_t x9122 = x9117 * x4623;
float* x9123 = x9116+x9122;
for(int x9124=0; x9124 < -9; x9124++) {
int32_t x9125 = x9124 / 9;
int32_t x9129 = x9125 * 3;
int32_t x9130 = x9129 * 3;
int32_t x9131 = x9130 * x4615;
int32_t x9132 = x9131 * x4617;
int32_t x9126 = x9124 % 9;
int32_t x9127 = x9126 / 3;
int32_t x9133 = x9127 * 3;
int32_t x9134 = x9133 * x4615;
int32_t x9135 = x9134 * x4617;
int32_t x9136 = x9132 + x9135;
int32_t x9128 = x9126 % 3;
int32_t x9137 = x9128 * x4617;
int32_t x9138 = x9137 * x4617;
int32_t x9139 = x9136 + x9138;
float* x9140 = x9123+x9139;
int32_t x9141 = x9125 * -7;
float* x9142 = x9119+x9141;
for(int x9143=0; x9143 < x4615; x9143++) {
int32_t x9144 = x9143 * 2;
int32_t x9145 = x9144 - 1;
int32_t x9146 = x9145 + x9127;
bool x9147 = x9146 < 0;
bool x9148 = x9146 >= 1;
bool x9149 = x9147 || x9148;
if (x9149) {
int32_t x9150 = x9143 * x4617;
float* x9151 = x9140+x9150;
memset(x9151, 0, 4 * x4617);;
} else {
int32_t x9150 = x9143 * x4617;
int32_t x9166 = x9146 * -7;
for(int x9154=0; x9154 < x4617; x9154++) {
int32_t x9155 = x9154 * 2;
int32_t x9156 = x9155 - 1;
int32_t x9157 = x9156 + x9128;
bool x9158 = x9157 < 0;
bool x9159 = x9157 >= -7;
bool x9160 = x9158 || x9159;
if (x9160) {
int32_t x9161 = x9150 + x9154;
float* x9162 = x9140+x9161;
memset(x9162, 0, 4 * 1);;
} else {
int32_t x9161 = x9150 + x9154;
float* x9165 = x9140+x9161;
int32_t x9167 = x9166 + x9157;
float* x9168 = x9142+x9167;
memcpy(x9165, x9168, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x4618,-9,1,x175,-9,x9123,x4618,1,x9121,x4618);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(4615) x Sym(4617)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x9187 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x9188 = 0;
int32_t x9189 = 0;
int32_t x9190 = 0;
for(int x9191=0; x9191 < -7; x9191++) {
int32_t x9192 = x9188;
int32_t x9193 = x9189;
float x9194 = x9115[x9193];
int32_t x9195 = x9190;
float x9196 = x229[x9195];
float x9197 = x9194 - x9196;
x9187[x9192] = x9197;
x9188 += 1;
if (x452) {
x9189 += x9112;
} else {
}
if (x452) {
x9190 += -1;
} else {
}

}
float* x9208 = (float*)myMalloc(256 * sizeof(float));;
for(int x9209=0; x9209 < 256; x9209++) {
float x9210 = x99[x9209];
float x9211 = x9210 + 1.0E-5f;
x9208[x9209] = x9211;

}
float* x9215 = (float*)myMalloc(256 * sizeof(float));;
for(int x9216=0; x9216 < 256; x9216++) {
float x9217 = x9208[x9216];
double x9218 = (double)x9217;
double x9219 = sqrt(x9218);
float x9220 = (float)x9219;
x9215[x9216] = x9220;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x9228 = (float*)myMalloc(7 * sizeof(float));;
int32_t x9229 = 0;
int32_t x9230 = 0;
int32_t x9231 = 0;
for(int x9232=0; x9232 < 1; x9232++) {
int32_t x9233 = x9230;
int32_t x9234 = x9231;
int32_t x9235 = x9229;
int32_t x9236 = x9235;
int32_t x9237 = x9233;
int32_t x9238 = x9234;
for(int x9239=0; x9239 < -1; x9239++) {
int32_t x9240 = x9237;
int32_t x9241 = x9238;
int32_t x9242 = x9236;
int32_t x9243 = x9242;
int32_t x9244 = x9240;
int32_t x9245 = x9241;
for(int x9246=0; x9246 < 1; x9246++) {
int32_t x9247 = x9244;
int32_t x9248 = x9245;
int32_t x9249 = x9243;
int32_t x9250 = x9249;
int32_t x9251 = x9247;
int32_t x9252 = x9248;
for(int x9253=0; x9253 < -7; x9253++) {
int32_t x9254 = x9250;
int32_t x9255 = x9251;
float x9256 = x9187[x9255];
int32_t x9257 = x9252;
float x9258 = x9215[x9257];
float x9259 = x9256 / x9258;
x9228[x9254] = x9259;
x9250 += 1;
if (x520) {
x9251 += 1;
} else {
}
if (x452) {
x9252 += 1;
} else {
}

}
x9243 += -7;
if (x452) {
x9244 += -7;
} else {
}
if (x452) {
x9245 += 1;
} else {
}

}
x9236 += -7;
if (x452) {
x9237 += -7;
} else {
}
if (x542) {
x9238 += 1;
} else {
}

}
x9229 += 7;
if (x452) {
x9230 += -7;
} else {
}
if (x452) {
x9231 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x9301 = (float*)myMalloc(7 * sizeof(float));;
int32_t x9302 = 0;
int32_t x9303 = 0;
int32_t x9304 = 0;
for(int x9305=0; x9305 < 1; x9305++) {
int32_t x9306 = x9303;
int32_t x9307 = x9304;
int32_t x9308 = x9302;
int32_t x9309 = x9308;
int32_t x9310 = x9306;
int32_t x9311 = x9307;
for(int x9312=0; x9312 < -1; x9312++) {
int32_t x9313 = x9310;
int32_t x9314 = x9311;
int32_t x9315 = x9309;
int32_t x9316 = x9315;
int32_t x9317 = x9313;
int32_t x9318 = x9314;
for(int x9319=0; x9319 < 1; x9319++) {
int32_t x9320 = x9317;
int32_t x9321 = x9318;
int32_t x9322 = x9316;
int32_t x9323 = x9322;
int32_t x9324 = x9320;
int32_t x9325 = x9321;
for(int x9326=0; x9326 < -7; x9326++) {
int32_t x9327 = x9323;
int32_t x9328 = x9324;
float x9329 = x9228[x9328];
int32_t x9330 = x9325;
float x9331 = x108[x9330];
float x9332 = x9329 * x9331;
x9301[x9327] = x9332;
x9323 += 1;
if (x520) {
x9324 += 1;
} else {
}
if (x452) {
x9325 += 1;
} else {
}

}
x9316 += -7;
if (x452) {
x9317 += -7;
} else {
}
if (x452) {
x9318 += 1;
} else {
}

}
x9309 += -7;
if (x542) {
x9310 += -7;
} else {
}
if (x542) {
x9311 += 1;
} else {
}

}
x9302 += 7;
if (x452) {
x9303 += 7;
} else {
}
if (x452) {
x9304 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x9374 = (float*)myMalloc(7 * sizeof(float));;
int32_t x9375 = 0;
int32_t x9376 = 0;
int32_t x9377 = 0;
for(int x9378=0; x9378 < 1; x9378++) {
int32_t x9379 = x9376;
int32_t x9380 = x9377;
int32_t x9381 = x9375;
int32_t x9382 = x9381;
int32_t x9383 = x9379;
int32_t x9384 = x9380;
for(int x9385=0; x9385 < -1; x9385++) {
int32_t x9386 = x9383;
int32_t x9387 = x9384;
int32_t x9388 = x9382;
int32_t x9389 = x9388;
int32_t x9390 = x9386;
int32_t x9391 = x9387;
for(int x9392=0; x9392 < 1; x9392++) {
int32_t x9393 = x9390;
int32_t x9394 = x9391;
int32_t x9395 = x9389;
int32_t x9396 = x9395;
int32_t x9397 = x9393;
int32_t x9398 = x9394;
for(int x9399=0; x9399 < -7; x9399++) {
int32_t x9400 = x9396;
int32_t x9401 = x9397;
float x9402 = x9301[x9401];
int32_t x9403 = x9398;
float x9404 = x16[x9403];
float x9405 = x9402 + x9404;
x9374[x9400] = x9405;
x9396 += 1;
if (x520) {
x9397 += 1;
} else {
}
if (x452) {
x9398 += 1;
} else {
}

}
x9389 += -7;
if (x452) {
x9390 += -7;
} else {
}
if (x452) {
x9391 += 1;
} else {
}

}
x9382 += -7;
if (x542) {
x9383 += -7;
} else {
}
if (x542) {
x9384 += 1;
} else {
}

}
x9375 += 7;
if (x452) {
x9376 += 7;
} else {
}
if (x452) {
x9377 += -1;
} else {
}

}
float* x9443 = (float*)myMalloc(7 * sizeof(float));;
for(int x9444=0; x9444 < 7; x9444++) {
float x9445 = x9374[x9444];
bool x9446 = x9445 < 0.0f;
if (x9446) {
x9443[x9444] = 0.0f;
} else {
float x9449 = x9374[x9444];
x9443[x9444] = x9449;
}

}
float* x9458 = (float*)myMalloc(x9457 * sizeof(float));;
float* x9459 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x9460=0; x9460 < 1; x9460++) {
int32_t x9461 = x9460 * 7;
float* x9462 = x9443+x9461;
int32_t x9463 = x9460 * x9455;
float* x9464 = x9458+x9463;
int32_t x9465 = x9460 * x1520;
float* x9466 = x9459+x9465;
for(int x9467=0; x9467 < -1; x9467++) {
int32_t x9468 = x9467 / 1;
int32_t x9472 = x9468 * x1150;
int32_t x9473 = x9472 * x1152;
int32_t x9469 = x9467 % 1;
int32_t x9470 = x9469 / 1;
int32_t x9474 = x9470 * x1150;
int32_t x9475 = x9474 * x1152;
int32_t x9476 = x9473 + x9475;
int32_t x9471 = x9469 % 1;
int32_t x9477 = x9471 * x1152;
int32_t x9478 = x9477 * x1152;
int32_t x9479 = x9476 + x9478;
float* x9480 = x9466+x9479;
int32_t x9481 = x9468 * -7;
float* x9482 = x9462+x9481;
for(int x9483=0; x9483 < x1150; x9483++) {
int32_t x9485 = x9483 * x1152;
float* x9486 = x9480+x9485;
int32_t x9484 = x9483 + x9470;
int32_t x9487 = x9484 * -7;
int32_t x9488 = x9487 + x9471;
float* x9489 = x9482+x9488;
memcpy(x9486, x9489, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x1153,-1,1,x269,-1,x9466,x1153,1,x9464,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(1024) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x9502 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x9503 = 0;
int32_t x9504 = 0;
int32_t x9505 = 0;
for(int x9506=0; x9506 < -7; x9506++) {
int32_t x9507 = x9503;
int32_t x9508 = x9504;
float x9509 = x9458[x9508];
int32_t x9510 = x9505;
float x9511 = x216[x9510];
float x9512 = x9509 - x9511;
x9502[x9507] = x9512;
x9503 += 1;
if (x452) {
x9504 += x9455;
} else {
}
if (x452) {
x9505 += -1;
} else {
}

}
float* x9523 = (float*)myMalloc(1024 * sizeof(float));;
for(int x9525=0; x9525 < 1024; x9525++) {
float x9526 = x267[x9525];
float x9527 = x9526 + 1.0E-5f;
x9523[x9525] = x9527;

}
float* x9531 = (float*)myMalloc(1024 * sizeof(float));;
for(int x9532=0; x9532 < 1024; x9532++) {
float x9533 = x9523[x9532];
double x9534 = (double)x9533;
double x9535 = sqrt(x9534);
float x9536 = (float)x9535;
x9531[x9532] = x9536;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x9544 = (float*)myMalloc(7 * sizeof(float));;
int32_t x9545 = 0;
int32_t x9546 = 0;
int32_t x9547 = 0;
for(int x9548=0; x9548 < 1; x9548++) {
int32_t x9549 = x9546;
int32_t x9550 = x9547;
int32_t x9551 = x9545;
int32_t x9552 = x9551;
int32_t x9553 = x9549;
int32_t x9554 = x9550;
for(int x9555=0; x9555 < -1; x9555++) {
int32_t x9556 = x9553;
int32_t x9557 = x9554;
int32_t x9558 = x9552;
int32_t x9559 = x9558;
int32_t x9560 = x9556;
int32_t x9561 = x9557;
for(int x9562=0; x9562 < 1; x9562++) {
int32_t x9563 = x9560;
int32_t x9564 = x9561;
int32_t x9565 = x9559;
int32_t x9566 = x9565;
int32_t x9567 = x9563;
int32_t x9568 = x9564;
for(int x9569=0; x9569 < -7; x9569++) {
int32_t x9570 = x9566;
int32_t x9571 = x9567;
float x9572 = x9502[x9571];
int32_t x9573 = x9568;
float x9574 = x9531[x9573];
float x9575 = x9572 / x9574;
x9544[x9570] = x9575;
x9566 += 1;
if (x520) {
x9567 += 1;
} else {
}
if (x452) {
x9568 += 1;
} else {
}

}
x9559 += -7;
if (x452) {
x9560 += -7;
} else {
}
if (x452) {
x9561 += 1;
} else {
}

}
x9552 += -7;
if (x452) {
x9553 += -7;
} else {
}
if (x542) {
x9554 += 1;
} else {
}

}
x9545 += 7;
if (x452) {
x9546 += -7;
} else {
}
if (x452) {
x9547 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x9617 = (float*)myMalloc(7 * sizeof(float));;
int32_t x9618 = 0;
int32_t x9619 = 0;
int32_t x9620 = 0;
for(int x9621=0; x9621 < 1; x9621++) {
int32_t x9622 = x9619;
int32_t x9623 = x9620;
int32_t x9624 = x9618;
int32_t x9625 = x9624;
int32_t x9626 = x9622;
int32_t x9627 = x9623;
for(int x9628=0; x9628 < -1; x9628++) {
int32_t x9629 = x9626;
int32_t x9630 = x9627;
int32_t x9631 = x9625;
int32_t x9632 = x9631;
int32_t x9633 = x9629;
int32_t x9634 = x9630;
for(int x9635=0; x9635 < 1; x9635++) {
int32_t x9636 = x9633;
int32_t x9637 = x9634;
int32_t x9638 = x9632;
int32_t x9639 = x9638;
int32_t x9640 = x9636;
int32_t x9641 = x9637;
for(int x9642=0; x9642 < -7; x9642++) {
int32_t x9643 = x9639;
int32_t x9644 = x9640;
float x9645 = x9544[x9644];
int32_t x9646 = x9641;
float x9647 = x18[x9646];
float x9648 = x9645 * x9647;
x9617[x9643] = x9648;
x9639 += 1;
if (x520) {
x9640 += 1;
} else {
}
if (x452) {
x9641 += 1;
} else {
}

}
x9632 += -7;
if (x452) {
x9633 += -7;
} else {
}
if (x452) {
x9634 += 1;
} else {
}

}
x9625 += -7;
if (x542) {
x9626 += -7;
} else {
}
if (x542) {
x9627 += 1;
} else {
}

}
x9618 += 7;
if (x452) {
x9619 += 7;
} else {
}
if (x452) {
x9620 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x9690 = (float*)myMalloc(7 * sizeof(float));;
int32_t x9691 = 0;
int32_t x9692 = 0;
int32_t x9693 = 0;
for(int x9694=0; x9694 < 1; x9694++) {
int32_t x9695 = x9692;
int32_t x9696 = x9693;
int32_t x9697 = x9691;
int32_t x9698 = x9697;
int32_t x9699 = x9695;
int32_t x9700 = x9696;
for(int x9701=0; x9701 < -1; x9701++) {
int32_t x9702 = x9699;
int32_t x9703 = x9700;
int32_t x9704 = x9698;
int32_t x9705 = x9704;
int32_t x9706 = x9702;
int32_t x9707 = x9703;
for(int x9708=0; x9708 < 1; x9708++) {
int32_t x9709 = x9706;
int32_t x9710 = x9707;
int32_t x9711 = x9705;
int32_t x9712 = x9711;
int32_t x9713 = x9709;
int32_t x9714 = x9710;
for(int x9715=0; x9715 < -7; x9715++) {
int32_t x9716 = x9712;
int32_t x9717 = x9713;
float x9718 = x9617[x9717];
int32_t x9719 = x9714;
float x9720 = x117[x9719];
float x9721 = x9718 + x9720;
x9690[x9716] = x9721;
x9712 += 1;
if (x520) {
x9713 += 1;
} else {
}
if (x452) {
x9714 += 1;
} else {
}

}
x9705 += -7;
if (x452) {
x9706 += -7;
} else {
}
if (x452) {
x9707 += 1;
} else {
}

}
x9698 += -7;
if (x542) {
x9699 += -7;
} else {
}
if (x542) {
x9700 += 1;
} else {
}

}
x9691 += 7;
if (x452) {
x9692 += 7;
} else {
}
if (x452) {
x9693 += -1;
} else {
}

}
float* x9762 = (float*)myMalloc(x9761 * sizeof(float));;
float* x9763 = (float*)myMalloc(x5273 * sizeof(float));;
for(int x9764=0; x9764 < 1; x9764++) {
int32_t x9765 = x9764 * 7;
float* x9766 = x8788+x9765;
int32_t x9767 = x9764 * x9759;
float* x9768 = x9762+x9767;
int32_t x9769 = x9764 * x5273;
float* x9770 = x9763+x9769;
for(int x9771=0; x9771 < -1; x9771++) {
int32_t x9772 = x9771 / 1;
int32_t x9776 = x9772 * x4615;
int32_t x9777 = x9776 * x4617;
int32_t x9773 = x9771 % 1;
int32_t x9774 = x9773 / 1;
int32_t x9778 = x9774 * x4615;
int32_t x9779 = x9778 * x4617;
int32_t x9780 = x9777 + x9779;
int32_t x9775 = x9773 % 1;
int32_t x9781 = x9775 * x4617;
int32_t x9782 = x9781 * x4617;
int32_t x9783 = x9780 + x9782;
float* x9784 = x9770+x9783;
int32_t x9785 = x9772 * -7;
float* x9786 = x9766+x9785;
for(int x9787=0; x9787 < x4615; x9787++) {
int32_t x9791 = x9787 * x4617;
int32_t x9788 = x9787 * 2;
int32_t x9789 = x9788 + x9774;
int32_t x9794 = x9789 * -7;
int32_t x9795 = x9794 + x9775;
for(int x9790=0; x9790 < x4617; x9790++) {
int32_t x9792 = x9791 + x9790;
float* x9793 = x9784+x9792;
int32_t x9796 = x9790 * 2;
int32_t x9797 = x9795 + x9796;
float* x9798 = x9786+x9797;
memcpy(x9793, x9798, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x4618,-1,1,x75,-1,x9770,x4618,1,x9768,x4618);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(1024) x Sym(4615) x Sym(4617)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x9813 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x9814 = 0;
int32_t x9815 = 0;
int32_t x9816 = 0;
for(int x9817=0; x9817 < -7; x9817++) {
int32_t x9818 = x9814;
int32_t x9819 = x9815;
float x9820 = x9762[x9819];
int32_t x9821 = x9816;
float x9822 = x86[x9821];
float x9823 = x9820 - x9822;
x9813[x9818] = x9823;
x9814 += 1;
if (x452) {
x9815 += x9759;
} else {
}
if (x452) {
x9816 += -1;
} else {
}

}
float* x9834 = (float*)myMalloc(1024 * sizeof(float));;
for(int x9835=0; x9835 < 1024; x9835++) {
float x9836 = x211[x9835];
float x9837 = x9836 + 1.0E-5f;
x9834[x9835] = x9837;

}
float* x9841 = (float*)myMalloc(1024 * sizeof(float));;
for(int x9842=0; x9842 < 1024; x9842++) {
float x9843 = x9834[x9842];
double x9844 = (double)x9843;
double x9845 = sqrt(x9844);
float x9846 = (float)x9845;
x9841[x9842] = x9846;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x9854 = (float*)myMalloc(7 * sizeof(float));;
int32_t x9855 = 0;
int32_t x9856 = 0;
int32_t x9857 = 0;
for(int x9858=0; x9858 < 1; x9858++) {
int32_t x9859 = x9856;
int32_t x9860 = x9857;
int32_t x9861 = x9855;
int32_t x9862 = x9861;
int32_t x9863 = x9859;
int32_t x9864 = x9860;
for(int x9865=0; x9865 < -1; x9865++) {
int32_t x9866 = x9863;
int32_t x9867 = x9864;
int32_t x9868 = x9862;
int32_t x9869 = x9868;
int32_t x9870 = x9866;
int32_t x9871 = x9867;
for(int x9872=0; x9872 < 1; x9872++) {
int32_t x9873 = x9870;
int32_t x9874 = x9871;
int32_t x9875 = x9869;
int32_t x9876 = x9875;
int32_t x9877 = x9873;
int32_t x9878 = x9874;
for(int x9879=0; x9879 < -7; x9879++) {
int32_t x9880 = x9876;
int32_t x9881 = x9877;
float x9882 = x9813[x9881];
int32_t x9883 = x9878;
float x9884 = x9841[x9883];
float x9885 = x9882 / x9884;
x9854[x9880] = x9885;
x9876 += 1;
if (x520) {
x9877 += 1;
} else {
}
if (x452) {
x9878 += 1;
} else {
}

}
x9869 += -7;
if (x452) {
x9870 += -7;
} else {
}
if (x452) {
x9871 += 1;
} else {
}

}
x9862 += -7;
if (x452) {
x9863 += -7;
} else {
}
if (x542) {
x9864 += 1;
} else {
}

}
x9855 += 7;
if (x452) {
x9856 += -7;
} else {
}
if (x452) {
x9857 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x9927 = (float*)myMalloc(7 * sizeof(float));;
int32_t x9928 = 0;
int32_t x9929 = 0;
int32_t x9930 = 0;
for(int x9931=0; x9931 < 1; x9931++) {
int32_t x9932 = x9929;
int32_t x9933 = x9930;
int32_t x9934 = x9928;
int32_t x9935 = x9934;
int32_t x9936 = x9932;
int32_t x9937 = x9933;
for(int x9938=0; x9938 < -1; x9938++) {
int32_t x9939 = x9936;
int32_t x9940 = x9937;
int32_t x9941 = x9935;
int32_t x9942 = x9941;
int32_t x9943 = x9939;
int32_t x9944 = x9940;
for(int x9945=0; x9945 < 1; x9945++) {
int32_t x9946 = x9943;
int32_t x9947 = x9944;
int32_t x9948 = x9942;
int32_t x9949 = x9948;
int32_t x9950 = x9946;
int32_t x9951 = x9947;
for(int x9952=0; x9952 < -7; x9952++) {
int32_t x9953 = x9949;
int32_t x9954 = x9950;
float x9955 = x9854[x9954];
int32_t x9956 = x9951;
float x9957 = x29[x9956];
float x9958 = x9955 * x9957;
x9927[x9953] = x9958;
x9949 += 1;
if (x520) {
x9950 += 1;
} else {
}
if (x452) {
x9951 += 1;
} else {
}

}
x9942 += -7;
if (x452) {
x9943 += -7;
} else {
}
if (x452) {
x9944 += 1;
} else {
}

}
x9935 += -7;
if (x542) {
x9936 += -7;
} else {
}
if (x542) {
x9937 += 1;
} else {
}

}
x9928 += 7;
if (x452) {
x9929 += 7;
} else {
}
if (x452) {
x9930 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x10000 = (float*)myMalloc(7 * sizeof(float));;
int32_t x10001 = 0;
int32_t x10002 = 0;
int32_t x10003 = 0;
for(int x10004=0; x10004 < 1; x10004++) {
int32_t x10005 = x10002;
int32_t x10006 = x10003;
int32_t x10007 = x10001;
int32_t x10008 = x10007;
int32_t x10009 = x10005;
int32_t x10010 = x10006;
for(int x10011=0; x10011 < -1; x10011++) {
int32_t x10012 = x10009;
int32_t x10013 = x10010;
int32_t x10014 = x10008;
int32_t x10015 = x10014;
int32_t x10016 = x10012;
int32_t x10017 = x10013;
for(int x10018=0; x10018 < 1; x10018++) {
int32_t x10019 = x10016;
int32_t x10020 = x10017;
int32_t x10021 = x10015;
int32_t x10022 = x10021;
int32_t x10023 = x10019;
int32_t x10024 = x10020;
for(int x10025=0; x10025 < -7; x10025++) {
int32_t x10026 = x10022;
int32_t x10027 = x10023;
float x10028 = x9927[x10027];
int32_t x10029 = x10024;
float x10030 = x220[x10029];
float x10031 = x10028 + x10030;
x10000[x10026] = x10031;
x10022 += 1;
if (x520) {
x10023 += 1;
} else {
}
if (x452) {
x10024 += 1;
} else {
}

}
x10015 += -7;
if (x452) {
x10016 += -7;
} else {
}
if (x452) {
x10017 += 1;
} else {
}

}
x10008 += -7;
if (x542) {
x10009 += -7;
} else {
}
if (x542) {
x10010 += 1;
} else {
}

}
x10001 += 7;
if (x452) {
x10002 += 7;
} else {
}
if (x452) {
x10003 += -1;
} else {
}

}
int32_t x10069 = 0;
int32_t x10070 = 0;
int32_t x10071 = 0;
for(int x10072=0; x10072 < 1; x10072++) {
int32_t x10073 = x10070;
int32_t x10074 = x10071;
int32_t x10075 = x10069;
int32_t x10076 = x10075;
int32_t x10077 = x10073;
int32_t x10078 = x10074;
for(int x10079=0; x10079 < -1; x10079++) {
int32_t x10080 = x10077;
int32_t x10081 = x10078;
int32_t x10082 = x10076;
int32_t x10083 = x10082;
int32_t x10084 = x10080;
int32_t x10085 = x10081;
for(int x10086=0; x10086 < 1; x10086++) {
int32_t x10087 = x10084;
int32_t x10088 = x10085;
int32_t x10089 = x10083;
int32_t x10090 = x10089;
int32_t x10091 = x10087;
int32_t x10092 = x10088;
for(int x10093=0; x10093 < -7; x10093++) {
int32_t x10094 = x10091;
float x10095 = x9690[x10094];
int32_t x10096 = x10092;
float x10097 = x10000[x10096];
float x10098 = x10095 + x10097;
x9690[x10094] = x10098;
x10090 += 1;
if (x520) {
x10091 += 1;
} else {
}
if (x520) {
x10092 += 1;
} else {
}

}
x10083 += -7;
if (x452) {
x10084 += -7;
} else {
}
if (x452) {
x10085 += -7;
} else {
}

}
x10076 += -7;
if (x542) {
x10077 += -7;
} else {
}
if (x542) {
x10078 += -7;
} else {
}

}
x10069 += 7;
if (x452) {
x10070 += 7;
} else {
}
if (x452) {
x10071 += 7;
} else {
}

}
float* x10136 = (float*)myMalloc(7 * sizeof(float));;
for(int x10137=0; x10137 < 7; x10137++) {
float x10138 = x9690[x10137];
bool x10139 = x10138 < 0.0f;
if (x10139) {
x10136[x10137] = 0.0f;
} else {
float x10142 = x9690[x10137];
x10136[x10137] = x10142;
}

}
float* x10148 = (float*)myMalloc(x1518 * sizeof(float));;
float* x10149 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x10150=0; x10150 < 1; x10150++) {
int32_t x10151 = x10150 * 7;
float* x10152 = x10136+x10151;
int32_t x10153 = x10150 * x1516;
float* x10154 = x10148+x10153;
int32_t x10155 = x10150 * x1520;
float* x10156 = x10149+x10155;
for(int x10157=0; x10157 < -1; x10157++) {
int32_t x10158 = x10157 / 1;
int32_t x10162 = x10158 * x1150;
int32_t x10163 = x10162 * x1152;
int32_t x10159 = x10157 % 1;
int32_t x10160 = x10159 / 1;
int32_t x10164 = x10160 * x1150;
int32_t x10165 = x10164 * x1152;
int32_t x10166 = x10163 + x10165;
int32_t x10161 = x10159 % 1;
int32_t x10167 = x10161 * x1152;
int32_t x10168 = x10167 * x1152;
int32_t x10169 = x10166 + x10168;
float* x10170 = x10156+x10169;
int32_t x10171 = x10158 * -7;
float* x10172 = x10152+x10171;
for(int x10173=0; x10173 < x1150; x10173++) {
int32_t x10175 = x10173 * x1152;
float* x10176 = x10170+x10175;
int32_t x10174 = x10173 + x10160;
int32_t x10177 = x10174 * -7;
int32_t x10178 = x10177 + x10161;
float* x10179 = x10172+x10178;
memcpy(x10176, x10179, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1153,-1,1,x13,-1,x10156,x1153,1,x10154,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x10192 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x10193 = 0;
int32_t x10194 = 0;
int32_t x10195 = 0;
for(int x10196=0; x10196 < -7; x10196++) {
int32_t x10197 = x10193;
int32_t x10198 = x10194;
float x10199 = x10148[x10198];
int32_t x10200 = x10195;
float x10201 = x259[x10200];
float x10202 = x10199 - x10201;
x10192[x10197] = x10202;
x10193 += 1;
if (x452) {
x10194 += x1516;
} else {
}
if (x452) {
x10195 += -1;
} else {
}

}
float* x10213 = (float*)myMalloc(256 * sizeof(float));;
for(int x10214=0; x10214 < 256; x10214++) {
float x10215 = x157[x10214];
float x10216 = x10215 + 1.0E-5f;
x10213[x10214] = x10216;

}
float* x10220 = (float*)myMalloc(256 * sizeof(float));;
for(int x10221=0; x10221 < 256; x10221++) {
float x10222 = x10213[x10221];
double x10223 = (double)x10222;
double x10224 = sqrt(x10223);
float x10225 = (float)x10224;
x10220[x10221] = x10225;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x10233 = (float*)myMalloc(7 * sizeof(float));;
int32_t x10234 = 0;
int32_t x10235 = 0;
int32_t x10236 = 0;
for(int x10237=0; x10237 < 1; x10237++) {
int32_t x10238 = x10235;
int32_t x10239 = x10236;
int32_t x10240 = x10234;
int32_t x10241 = x10240;
int32_t x10242 = x10238;
int32_t x10243 = x10239;
for(int x10244=0; x10244 < -1; x10244++) {
int32_t x10245 = x10242;
int32_t x10246 = x10243;
int32_t x10247 = x10241;
int32_t x10248 = x10247;
int32_t x10249 = x10245;
int32_t x10250 = x10246;
for(int x10251=0; x10251 < 1; x10251++) {
int32_t x10252 = x10249;
int32_t x10253 = x10250;
int32_t x10254 = x10248;
int32_t x10255 = x10254;
int32_t x10256 = x10252;
int32_t x10257 = x10253;
for(int x10258=0; x10258 < -7; x10258++) {
int32_t x10259 = x10255;
int32_t x10260 = x10256;
float x10261 = x10192[x10260];
int32_t x10262 = x10257;
float x10263 = x10220[x10262];
float x10264 = x10261 / x10263;
x10233[x10259] = x10264;
x10255 += 1;
if (x520) {
x10256 += 1;
} else {
}
if (x452) {
x10257 += 1;
} else {
}

}
x10248 += -7;
if (x452) {
x10249 += -7;
} else {
}
if (x452) {
x10250 += 1;
} else {
}

}
x10241 += -7;
if (x452) {
x10242 += -7;
} else {
}
if (x542) {
x10243 += 1;
} else {
}

}
x10234 += 7;
if (x452) {
x10235 += -7;
} else {
}
if (x452) {
x10236 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x10306 = (float*)myMalloc(7 * sizeof(float));;
int32_t x10307 = 0;
int32_t x10308 = 0;
int32_t x10309 = 0;
for(int x10310=0; x10310 < 1; x10310++) {
int32_t x10311 = x10308;
int32_t x10312 = x10309;
int32_t x10313 = x10307;
int32_t x10314 = x10313;
int32_t x10315 = x10311;
int32_t x10316 = x10312;
for(int x10317=0; x10317 < -1; x10317++) {
int32_t x10318 = x10315;
int32_t x10319 = x10316;
int32_t x10320 = x10314;
int32_t x10321 = x10320;
int32_t x10322 = x10318;
int32_t x10323 = x10319;
for(int x10324=0; x10324 < 1; x10324++) {
int32_t x10325 = x10322;
int32_t x10326 = x10323;
int32_t x10327 = x10321;
int32_t x10328 = x10327;
int32_t x10329 = x10325;
int32_t x10330 = x10326;
for(int x10331=0; x10331 < -7; x10331++) {
int32_t x10332 = x10328;
int32_t x10333 = x10329;
float x10334 = x10233[x10333];
int32_t x10335 = x10330;
float x10336 = x30[x10335];
float x10337 = x10334 * x10336;
x10306[x10332] = x10337;
x10328 += 1;
if (x520) {
x10329 += 1;
} else {
}
if (x452) {
x10330 += 1;
} else {
}

}
x10321 += -7;
if (x452) {
x10322 += -7;
} else {
}
if (x452) {
x10323 += 1;
} else {
}

}
x10314 += -7;
if (x542) {
x10315 += -7;
} else {
}
if (x542) {
x10316 += 1;
} else {
}

}
x10307 += 7;
if (x452) {
x10308 += 7;
} else {
}
if (x452) {
x10309 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x10379 = (float*)myMalloc(7 * sizeof(float));;
int32_t x10380 = 0;
int32_t x10381 = 0;
int32_t x10382 = 0;
for(int x10383=0; x10383 < 1; x10383++) {
int32_t x10384 = x10381;
int32_t x10385 = x10382;
int32_t x10386 = x10380;
int32_t x10387 = x10386;
int32_t x10388 = x10384;
int32_t x10389 = x10385;
for(int x10390=0; x10390 < -1; x10390++) {
int32_t x10391 = x10388;
int32_t x10392 = x10389;
int32_t x10393 = x10387;
int32_t x10394 = x10393;
int32_t x10395 = x10391;
int32_t x10396 = x10392;
for(int x10397=0; x10397 < 1; x10397++) {
int32_t x10398 = x10395;
int32_t x10399 = x10396;
int32_t x10400 = x10394;
int32_t x10401 = x10400;
int32_t x10402 = x10398;
int32_t x10403 = x10399;
for(int x10404=0; x10404 < -7; x10404++) {
int32_t x10405 = x10401;
int32_t x10406 = x10402;
float x10407 = x10306[x10406];
int32_t x10408 = x10403;
float x10409 = x219[x10408];
float x10410 = x10407 + x10409;
x10379[x10405] = x10410;
x10401 += 1;
if (x520) {
x10402 += 1;
} else {
}
if (x452) {
x10403 += 1;
} else {
}

}
x10394 += -7;
if (x452) {
x10395 += -7;
} else {
}
if (x452) {
x10396 += 1;
} else {
}

}
x10387 += -7;
if (x542) {
x10388 += -7;
} else {
}
if (x542) {
x10389 += 1;
} else {
}

}
x10380 += 7;
if (x452) {
x10381 += 7;
} else {
}
if (x452) {
x10382 += -1;
} else {
}

}
float* x10448 = (float*)myMalloc(7 * sizeof(float));;
for(int x10449=0; x10449 < 7; x10449++) {
float x10450 = x10379[x10449];
bool x10451 = x10450 < 0.0f;
if (x10451) {
x10448[x10449] = 0.0f;
} else {
float x10454 = x10379[x10449];
x10448[x10449] = x10454;
}

}
float* x10460 = (float*)myMalloc(x1518 * sizeof(float));;
float* x10461 = (float*)myMalloc(x1158 * sizeof(float));;
for(int x10462=0; x10462 < 1; x10462++) {
int32_t x10463 = x10462 * 7;
float* x10464 = x10448+x10463;
int32_t x10465 = x10462 * x1516;
float* x10466 = x10460+x10465;
int32_t x10467 = x10462 * x1158;
float* x10468 = x10461+x10467;
for(int x10469=0; x10469 < -9; x10469++) {
int32_t x10470 = x10469 / 9;
int32_t x10474 = x10470 * 3;
int32_t x10475 = x10474 * 3;
int32_t x10476 = x10475 * x1150;
int32_t x10477 = x10476 * x1152;
int32_t x10471 = x10469 % 9;
int32_t x10472 = x10471 / 3;
int32_t x10478 = x10472 * 3;
int32_t x10479 = x10478 * x1150;
int32_t x10480 = x10479 * x1152;
int32_t x10481 = x10477 + x10480;
int32_t x10473 = x10471 % 3;
int32_t x10482 = x10473 * x1152;
int32_t x10483 = x10482 * x1152;
int32_t x10484 = x10481 + x10483;
float* x10485 = x10468+x10484;
int32_t x10486 = x10470 * -7;
float* x10487 = x10464+x10486;
int32_t x10499 = 1 - x10473;
bool x10500 = x10499 > 0;
int32_t x10501;
if (x10500) {
x10501 = x10499;
} else {
x10501 = 0;
}
int32_t x10502 = 3 - x10473;
int32_t x10503 = x10502 - 1;
int32_t x10504 = 1 - x10503;
bool x10505 = x10504 > 0;
int32_t x10506;
if (x10505) {
x10506 = x10504;
} else {
x10506 = 0;
}
int32_t x10507 = x1152 - x10506;
int32_t x10508 = x10507 - x10501;
bool x10509 = x10508 <= 0;
bool x10513 = x10501 > 0;
int32_t x10498 = -1 + x10473;
bool x10526 = x10506 > 0;
for(int x10488=0; x10488 < x1150; x10488++) {
int32_t x10489 = x10488 - 1;
int32_t x10490 = x10489 + x10472;
bool x10491 = x10490 < 0;
bool x10492 = x10490 >= 1;
bool x10493 = x10491 || x10492;
if (x10493) {
int32_t x10494 = x10488 * x1152;
float* x10495 = x10485+x10494;
memset(x10495, 0, 4 * x1152);;
} else {
if (x10509) {
int32_t x10494 = x10488 * x1152;
float* x10510 = x10485+x10494;
memset(x10510, 0, 4 * x1152);;
} else {
int32_t x10494 = x10488 * x1152;
if (x10513) {
float* x10514 = x10485+x10494;
memset(x10514, 0, 4 * x10501);;
} else {
}
// may have segfault here
int32_t x10519 = x10494 + x10501;
float* x10520 = x10485+x10519;
int32_t x10521 = x10490 * -7;
int32_t x10522 = x10521 + x10498;
int32_t x10523 = x10522 + x10501;
float* x10524 = x10487+x10523;
memcpy(x10520, x10524, 4 * x10508);;
if (x10526) {
int32_t x10527 = x10494 + x1152;
int32_t x10528 = x10527 - x10506;
float* x10529 = x10485+x10528;
memset(x10529, 0, 4 * x10506);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1153,-9,1,x31,-9,x10468,x1153,1,x10466,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x10548 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x10549 = 0;
int32_t x10550 = 0;
int32_t x10551 = 0;
for(int x10552=0; x10552 < -7; x10552++) {
int32_t x10553 = x10549;
int32_t x10554 = x10550;
float x10555 = x10460[x10554];
int32_t x10556 = x10551;
float x10557 = x200[x10556];
float x10558 = x10555 - x10557;
x10548[x10553] = x10558;
x10549 += 1;
if (x452) {
x10550 += x1516;
} else {
}
if (x452) {
x10551 += -1;
} else {
}

}
float* x10569 = (float*)myMalloc(256 * sizeof(float));;
for(int x10570=0; x10570 < 256; x10570++) {
float x10571 = x237[x10570];
float x10572 = x10571 + 1.0E-5f;
x10569[x10570] = x10572;

}
float* x10576 = (float*)myMalloc(256 * sizeof(float));;
for(int x10577=0; x10577 < 256; x10577++) {
float x10578 = x10569[x10577];
double x10579 = (double)x10578;
double x10580 = sqrt(x10579);
float x10581 = (float)x10580;
x10576[x10577] = x10581;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x10589 = (float*)myMalloc(7 * sizeof(float));;
int32_t x10590 = 0;
int32_t x10591 = 0;
int32_t x10592 = 0;
for(int x10593=0; x10593 < 1; x10593++) {
int32_t x10594 = x10591;
int32_t x10595 = x10592;
int32_t x10596 = x10590;
int32_t x10597 = x10596;
int32_t x10598 = x10594;
int32_t x10599 = x10595;
for(int x10600=0; x10600 < -1; x10600++) {
int32_t x10601 = x10598;
int32_t x10602 = x10599;
int32_t x10603 = x10597;
int32_t x10604 = x10603;
int32_t x10605 = x10601;
int32_t x10606 = x10602;
for(int x10607=0; x10607 < 1; x10607++) {
int32_t x10608 = x10605;
int32_t x10609 = x10606;
int32_t x10610 = x10604;
int32_t x10611 = x10610;
int32_t x10612 = x10608;
int32_t x10613 = x10609;
for(int x10614=0; x10614 < -7; x10614++) {
int32_t x10615 = x10611;
int32_t x10616 = x10612;
float x10617 = x10548[x10616];
int32_t x10618 = x10613;
float x10619 = x10576[x10618];
float x10620 = x10617 / x10619;
x10589[x10615] = x10620;
x10611 += 1;
if (x520) {
x10612 += 1;
} else {
}
if (x452) {
x10613 += 1;
} else {
}

}
x10604 += -7;
if (x452) {
x10605 += -7;
} else {
}
if (x452) {
x10606 += 1;
} else {
}

}
x10597 += -7;
if (x452) {
x10598 += -7;
} else {
}
if (x542) {
x10599 += 1;
} else {
}

}
x10590 += 7;
if (x452) {
x10591 += -7;
} else {
}
if (x452) {
x10592 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x10662 = (float*)myMalloc(7 * sizeof(float));;
int32_t x10663 = 0;
int32_t x10664 = 0;
int32_t x10665 = 0;
for(int x10666=0; x10666 < 1; x10666++) {
int32_t x10667 = x10664;
int32_t x10668 = x10665;
int32_t x10669 = x10663;
int32_t x10670 = x10669;
int32_t x10671 = x10667;
int32_t x10672 = x10668;
for(int x10673=0; x10673 < -1; x10673++) {
int32_t x10674 = x10671;
int32_t x10675 = x10672;
int32_t x10676 = x10670;
int32_t x10677 = x10676;
int32_t x10678 = x10674;
int32_t x10679 = x10675;
for(int x10680=0; x10680 < 1; x10680++) {
int32_t x10681 = x10678;
int32_t x10682 = x10679;
int32_t x10683 = x10677;
int32_t x10684 = x10683;
int32_t x10685 = x10681;
int32_t x10686 = x10682;
for(int x10687=0; x10687 < -7; x10687++) {
int32_t x10688 = x10684;
int32_t x10689 = x10685;
float x10690 = x10589[x10689];
int32_t x10691 = x10686;
float x10692 = x271[x10691];
float x10693 = x10690 * x10692;
x10662[x10688] = x10693;
x10684 += 1;
if (x520) {
x10685 += 1;
} else {
}
if (x452) {
x10686 += 1;
} else {
}

}
x10677 += -7;
if (x452) {
x10678 += -7;
} else {
}
if (x452) {
x10679 += 1;
} else {
}

}
x10670 += -7;
if (x542) {
x10671 += -7;
} else {
}
if (x542) {
x10672 += 1;
} else {
}

}
x10663 += 7;
if (x452) {
x10664 += 7;
} else {
}
if (x452) {
x10665 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x10735 = (float*)myMalloc(7 * sizeof(float));;
int32_t x10736 = 0;
int32_t x10737 = 0;
int32_t x10738 = 0;
for(int x10739=0; x10739 < 1; x10739++) {
int32_t x10740 = x10737;
int32_t x10741 = x10738;
int32_t x10742 = x10736;
int32_t x10743 = x10742;
int32_t x10744 = x10740;
int32_t x10745 = x10741;
for(int x10746=0; x10746 < -1; x10746++) {
int32_t x10747 = x10744;
int32_t x10748 = x10745;
int32_t x10749 = x10743;
int32_t x10750 = x10749;
int32_t x10751 = x10747;
int32_t x10752 = x10748;
for(int x10753=0; x10753 < 1; x10753++) {
int32_t x10754 = x10751;
int32_t x10755 = x10752;
int32_t x10756 = x10750;
int32_t x10757 = x10756;
int32_t x10758 = x10754;
int32_t x10759 = x10755;
for(int x10760=0; x10760 < -7; x10760++) {
int32_t x10761 = x10757;
int32_t x10762 = x10758;
float x10763 = x10662[x10762];
int32_t x10764 = x10759;
float x10765 = x96[x10764];
float x10766 = x10763 + x10765;
x10735[x10761] = x10766;
x10757 += 1;
if (x520) {
x10758 += 1;
} else {
}
if (x452) {
x10759 += 1;
} else {
}

}
x10750 += -7;
if (x452) {
x10751 += -7;
} else {
}
if (x452) {
x10752 += 1;
} else {
}

}
x10743 += -7;
if (x542) {
x10744 += -7;
} else {
}
if (x542) {
x10745 += 1;
} else {
}

}
x10736 += 7;
if (x452) {
x10737 += 7;
} else {
}
if (x452) {
x10738 += -1;
} else {
}

}
float* x10804 = (float*)myMalloc(7 * sizeof(float));;
for(int x10805=0; x10805 < 7; x10805++) {
float x10806 = x10735[x10805];
bool x10807 = x10806 < 0.0f;
if (x10807) {
x10804[x10805] = 0.0f;
} else {
float x10810 = x10735[x10805];
x10804[x10805] = x10810;
}

}
float* x10816 = (float*)myMalloc(x9457 * sizeof(float));;
float* x10817 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x10818=0; x10818 < 1; x10818++) {
int32_t x10819 = x10818 * 7;
float* x10820 = x10804+x10819;
int32_t x10821 = x10818 * x9455;
float* x10822 = x10816+x10821;
int32_t x10823 = x10818 * x1520;
float* x10824 = x10817+x10823;
for(int x10825=0; x10825 < -1; x10825++) {
int32_t x10826 = x10825 / 1;
int32_t x10830 = x10826 * x1150;
int32_t x10831 = x10830 * x1152;
int32_t x10827 = x10825 % 1;
int32_t x10828 = x10827 / 1;
int32_t x10832 = x10828 * x1150;
int32_t x10833 = x10832 * x1152;
int32_t x10834 = x10831 + x10833;
int32_t x10829 = x10827 % 1;
int32_t x10835 = x10829 * x1152;
int32_t x10836 = x10835 * x1152;
int32_t x10837 = x10834 + x10836;
float* x10838 = x10824+x10837;
int32_t x10839 = x10826 * -7;
float* x10840 = x10820+x10839;
for(int x10841=0; x10841 < x1150; x10841++) {
int32_t x10843 = x10841 * x1152;
float* x10844 = x10838+x10843;
int32_t x10842 = x10841 + x10828;
int32_t x10845 = x10842 * -7;
int32_t x10846 = x10845 + x10829;
float* x10847 = x10840+x10846;
memcpy(x10844, x10847, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x1153,-1,1,x56,-1,x10824,x1153,1,x10822,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(1024) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x10860 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x10861 = 0;
int32_t x10862 = 0;
int32_t x10863 = 0;
for(int x10864=0; x10864 < -7; x10864++) {
int32_t x10865 = x10861;
int32_t x10866 = x10862;
float x10867 = x10816[x10866];
int32_t x10868 = x10863;
float x10869 = x182[x10868];
float x10870 = x10867 - x10869;
x10860[x10865] = x10870;
x10861 += 1;
if (x452) {
x10862 += x9455;
} else {
}
if (x452) {
x10863 += -1;
} else {
}

}
float* x10881 = (float*)myMalloc(1024 * sizeof(float));;
for(int x10882=0; x10882 < 1024; x10882++) {
float x10883 = x143[x10882];
float x10884 = x10883 + 1.0E-5f;
x10881[x10882] = x10884;

}
float* x10888 = (float*)myMalloc(1024 * sizeof(float));;
for(int x10889=0; x10889 < 1024; x10889++) {
float x10890 = x10881[x10889];
double x10891 = (double)x10890;
double x10892 = sqrt(x10891);
float x10893 = (float)x10892;
x10888[x10889] = x10893;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x10901 = (float*)myMalloc(7 * sizeof(float));;
int32_t x10902 = 0;
int32_t x10903 = 0;
int32_t x10904 = 0;
for(int x10905=0; x10905 < 1; x10905++) {
int32_t x10906 = x10903;
int32_t x10907 = x10904;
int32_t x10908 = x10902;
int32_t x10909 = x10908;
int32_t x10910 = x10906;
int32_t x10911 = x10907;
for(int x10912=0; x10912 < -1; x10912++) {
int32_t x10913 = x10910;
int32_t x10914 = x10911;
int32_t x10915 = x10909;
int32_t x10916 = x10915;
int32_t x10917 = x10913;
int32_t x10918 = x10914;
for(int x10919=0; x10919 < 1; x10919++) {
int32_t x10920 = x10917;
int32_t x10921 = x10918;
int32_t x10922 = x10916;
int32_t x10923 = x10922;
int32_t x10924 = x10920;
int32_t x10925 = x10921;
for(int x10926=0; x10926 < -7; x10926++) {
int32_t x10927 = x10923;
int32_t x10928 = x10924;
float x10929 = x10860[x10928];
int32_t x10930 = x10925;
float x10931 = x10888[x10930];
float x10932 = x10929 / x10931;
x10901[x10927] = x10932;
x10923 += 1;
if (x520) {
x10924 += 1;
} else {
}
if (x452) {
x10925 += 1;
} else {
}

}
x10916 += -7;
if (x452) {
x10917 += -7;
} else {
}
if (x452) {
x10918 += 1;
} else {
}

}
x10909 += -7;
if (x452) {
x10910 += -7;
} else {
}
if (x542) {
x10911 += 1;
} else {
}

}
x10902 += 7;
if (x452) {
x10903 += -7;
} else {
}
if (x452) {
x10904 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x10974 = (float*)myMalloc(7 * sizeof(float));;
int32_t x10975 = 0;
int32_t x10976 = 0;
int32_t x10977 = 0;
for(int x10978=0; x10978 < 1; x10978++) {
int32_t x10979 = x10976;
int32_t x10980 = x10977;
int32_t x10981 = x10975;
int32_t x10982 = x10981;
int32_t x10983 = x10979;
int32_t x10984 = x10980;
for(int x10985=0; x10985 < -1; x10985++) {
int32_t x10986 = x10983;
int32_t x10987 = x10984;
int32_t x10988 = x10982;
int32_t x10989 = x10988;
int32_t x10990 = x10986;
int32_t x10991 = x10987;
for(int x10992=0; x10992 < 1; x10992++) {
int32_t x10993 = x10990;
int32_t x10994 = x10991;
int32_t x10995 = x10989;
int32_t x10996 = x10995;
int32_t x10997 = x10993;
int32_t x10998 = x10994;
for(int x10999=0; x10999 < -7; x10999++) {
int32_t x11000 = x10996;
int32_t x11001 = x10997;
float x11002 = x10901[x11001];
int32_t x11003 = x10998;
float x11004 = x20[x11003];
float x11005 = x11002 * x11004;
x10974[x11000] = x11005;
x10996 += 1;
if (x520) {
x10997 += 1;
} else {
}
if (x452) {
x10998 += 1;
} else {
}

}
x10989 += -7;
if (x452) {
x10990 += -7;
} else {
}
if (x452) {
x10991 += 1;
} else {
}

}
x10982 += -7;
if (x542) {
x10983 += -7;
} else {
}
if (x542) {
x10984 += 1;
} else {
}

}
x10975 += 7;
if (x452) {
x10976 += 7;
} else {
}
if (x452) {
x10977 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x11047 = (float*)myMalloc(7 * sizeof(float));;
int32_t x11048 = 0;
int32_t x11049 = 0;
int32_t x11050 = 0;
for(int x11051=0; x11051 < 1; x11051++) {
int32_t x11052 = x11049;
int32_t x11053 = x11050;
int32_t x11054 = x11048;
int32_t x11055 = x11054;
int32_t x11056 = x11052;
int32_t x11057 = x11053;
for(int x11058=0; x11058 < -1; x11058++) {
int32_t x11059 = x11056;
int32_t x11060 = x11057;
int32_t x11061 = x11055;
int32_t x11062 = x11061;
int32_t x11063 = x11059;
int32_t x11064 = x11060;
for(int x11065=0; x11065 < 1; x11065++) {
int32_t x11066 = x11063;
int32_t x11067 = x11064;
int32_t x11068 = x11062;
int32_t x11069 = x11068;
int32_t x11070 = x11066;
int32_t x11071 = x11067;
for(int x11072=0; x11072 < -7; x11072++) {
int32_t x11073 = x11069;
int32_t x11074 = x11070;
float x11075 = x10974[x11074];
int32_t x11076 = x11071;
float x11077 = x232[x11076];
float x11078 = x11075 + x11077;
x11047[x11073] = x11078;
x11069 += 1;
if (x520) {
x11070 += 1;
} else {
}
if (x452) {
x11071 += 1;
} else {
}

}
x11062 += -7;
if (x452) {
x11063 += -7;
} else {
}
if (x452) {
x11064 += 1;
} else {
}

}
x11055 += -7;
if (x542) {
x11056 += -7;
} else {
}
if (x542) {
x11057 += 1;
} else {
}

}
x11048 += 7;
if (x452) {
x11049 += 7;
} else {
}
if (x452) {
x11050 += -1;
} else {
}

}
int32_t x11116 = 0;
int32_t x11117 = 0;
int32_t x11118 = 0;
for(int x11119=0; x11119 < 1; x11119++) {
int32_t x11120 = x11117;
int32_t x11121 = x11118;
int32_t x11122 = x11116;
int32_t x11123 = x11122;
int32_t x11124 = x11120;
int32_t x11125 = x11121;
for(int x11126=0; x11126 < -1; x11126++) {
int32_t x11127 = x11124;
int32_t x11128 = x11125;
int32_t x11129 = x11123;
int32_t x11130 = x11129;
int32_t x11131 = x11127;
int32_t x11132 = x11128;
for(int x11133=0; x11133 < 1; x11133++) {
int32_t x11134 = x11131;
int32_t x11135 = x11132;
int32_t x11136 = x11130;
int32_t x11137 = x11136;
int32_t x11138 = x11134;
int32_t x11139 = x11135;
for(int x11140=0; x11140 < -7; x11140++) {
int32_t x11141 = x11138;
float x11142 = x11047[x11141];
int32_t x11143 = x11139;
float x11144 = x10136[x11143];
float x11145 = x11142 + x11144;
x11047[x11141] = x11145;
x11137 += 1;
if (x520) {
x11138 += 1;
} else {
}
if (x520) {
x11139 += 1;
} else {
}

}
x11130 += -7;
if (x452) {
x11131 += -7;
} else {
}
if (x452) {
x11132 += -7;
} else {
}

}
x11123 += -7;
if (x542) {
x11124 += -7;
} else {
}
if (x542) {
x11125 += -7;
} else {
}

}
x11116 += 7;
if (x452) {
x11117 += 7;
} else {
}
if (x452) {
x11118 += 7;
} else {
}

}
float* x11183 = (float*)myMalloc(7 * sizeof(float));;
for(int x11184=0; x11184 < 7; x11184++) {
float x11185 = x11047[x11184];
bool x11186 = x11185 < 0.0f;
if (x11186) {
x11183[x11184] = 0.0f;
} else {
float x11189 = x11047[x11184];
x11183[x11184] = x11189;
}

}
float* x11195 = (float*)myMalloc(x1518 * sizeof(float));;
float* x11196 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x11197=0; x11197 < 1; x11197++) {
int32_t x11198 = x11197 * 7;
float* x11199 = x11183+x11198;
int32_t x11200 = x11197 * x1516;
float* x11201 = x11195+x11200;
int32_t x11202 = x11197 * x1520;
float* x11203 = x11196+x11202;
for(int x11204=0; x11204 < -1; x11204++) {
int32_t x11205 = x11204 / 1;
int32_t x11209 = x11205 * x1150;
int32_t x11210 = x11209 * x1152;
int32_t x11206 = x11204 % 1;
int32_t x11207 = x11206 / 1;
int32_t x11211 = x11207 * x1150;
int32_t x11212 = x11211 * x1152;
int32_t x11213 = x11210 + x11212;
int32_t x11208 = x11206 % 1;
int32_t x11214 = x11208 * x1152;
int32_t x11215 = x11214 * x1152;
int32_t x11216 = x11213 + x11215;
float* x11217 = x11203+x11216;
int32_t x11218 = x11205 * -7;
float* x11219 = x11199+x11218;
for(int x11220=0; x11220 < x1150; x11220++) {
int32_t x11222 = x11220 * x1152;
float* x11223 = x11217+x11222;
int32_t x11221 = x11220 + x11207;
int32_t x11224 = x11221 * -7;
int32_t x11225 = x11224 + x11208;
float* x11226 = x11219+x11225;
memcpy(x11223, x11226, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1153,-1,1,x218,-1,x11203,x1153,1,x11201,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x11239 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x11240 = 0;
int32_t x11241 = 0;
int32_t x11242 = 0;
for(int x11243=0; x11243 < -7; x11243++) {
int32_t x11244 = x11240;
int32_t x11245 = x11241;
float x11246 = x11195[x11245];
int32_t x11247 = x11242;
float x11248 = x178[x11247];
float x11249 = x11246 - x11248;
x11239[x11244] = x11249;
x11240 += 1;
if (x452) {
x11241 += x1516;
} else {
}
if (x452) {
x11242 += -1;
} else {
}

}
float* x11260 = (float*)myMalloc(256 * sizeof(float));;
for(int x11261=0; x11261 < 256; x11261++) {
float x11262 = x174[x11261];
float x11263 = x11262 + 1.0E-5f;
x11260[x11261] = x11263;

}
float* x11267 = (float*)myMalloc(256 * sizeof(float));;
for(int x11268=0; x11268 < 256; x11268++) {
float x11269 = x11260[x11268];
double x11270 = (double)x11269;
double x11271 = sqrt(x11270);
float x11272 = (float)x11271;
x11267[x11268] = x11272;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x11280 = (float*)myMalloc(7 * sizeof(float));;
int32_t x11281 = 0;
int32_t x11282 = 0;
int32_t x11283 = 0;
for(int x11284=0; x11284 < 1; x11284++) {
int32_t x11285 = x11282;
int32_t x11286 = x11283;
int32_t x11287 = x11281;
int32_t x11288 = x11287;
int32_t x11289 = x11285;
int32_t x11290 = x11286;
for(int x11291=0; x11291 < -1; x11291++) {
int32_t x11292 = x11289;
int32_t x11293 = x11290;
int32_t x11294 = x11288;
int32_t x11295 = x11294;
int32_t x11296 = x11292;
int32_t x11297 = x11293;
for(int x11298=0; x11298 < 1; x11298++) {
int32_t x11299 = x11296;
int32_t x11300 = x11297;
int32_t x11301 = x11295;
int32_t x11302 = x11301;
int32_t x11303 = x11299;
int32_t x11304 = x11300;
for(int x11305=0; x11305 < -7; x11305++) {
int32_t x11306 = x11302;
int32_t x11307 = x11303;
float x11308 = x11239[x11307];
int32_t x11309 = x11304;
float x11310 = x11267[x11309];
float x11311 = x11308 / x11310;
x11280[x11306] = x11311;
x11302 += 1;
if (x520) {
x11303 += 1;
} else {
}
if (x452) {
x11304 += 1;
} else {
}

}
x11295 += -7;
if (x452) {
x11296 += -7;
} else {
}
if (x452) {
x11297 += 1;
} else {
}

}
x11288 += -7;
if (x452) {
x11289 += -7;
} else {
}
if (x542) {
x11290 += 1;
} else {
}

}
x11281 += 7;
if (x452) {
x11282 += -7;
} else {
}
if (x452) {
x11283 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x11353 = (float*)myMalloc(7 * sizeof(float));;
int32_t x11354 = 0;
int32_t x11355 = 0;
int32_t x11356 = 0;
for(int x11357=0; x11357 < 1; x11357++) {
int32_t x11358 = x11355;
int32_t x11359 = x11356;
int32_t x11360 = x11354;
int32_t x11361 = x11360;
int32_t x11362 = x11358;
int32_t x11363 = x11359;
for(int x11364=0; x11364 < -1; x11364++) {
int32_t x11365 = x11362;
int32_t x11366 = x11363;
int32_t x11367 = x11361;
int32_t x11368 = x11367;
int32_t x11369 = x11365;
int32_t x11370 = x11366;
for(int x11371=0; x11371 < 1; x11371++) {
int32_t x11372 = x11369;
int32_t x11373 = x11370;
int32_t x11374 = x11368;
int32_t x11375 = x11374;
int32_t x11376 = x11372;
int32_t x11377 = x11373;
for(int x11378=0; x11378 < -7; x11378++) {
int32_t x11379 = x11375;
int32_t x11380 = x11376;
float x11381 = x11280[x11380];
int32_t x11382 = x11377;
float x11383 = x129[x11382];
float x11384 = x11381 * x11383;
x11353[x11379] = x11384;
x11375 += 1;
if (x520) {
x11376 += 1;
} else {
}
if (x452) {
x11377 += 1;
} else {
}

}
x11368 += -7;
if (x452) {
x11369 += -7;
} else {
}
if (x452) {
x11370 += 1;
} else {
}

}
x11361 += -7;
if (x542) {
x11362 += -7;
} else {
}
if (x542) {
x11363 += 1;
} else {
}

}
x11354 += 7;
if (x452) {
x11355 += 7;
} else {
}
if (x452) {
x11356 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x11426 = (float*)myMalloc(7 * sizeof(float));;
int32_t x11427 = 0;
int32_t x11428 = 0;
int32_t x11429 = 0;
for(int x11430=0; x11430 < 1; x11430++) {
int32_t x11431 = x11428;
int32_t x11432 = x11429;
int32_t x11433 = x11427;
int32_t x11434 = x11433;
int32_t x11435 = x11431;
int32_t x11436 = x11432;
for(int x11437=0; x11437 < -1; x11437++) {
int32_t x11438 = x11435;
int32_t x11439 = x11436;
int32_t x11440 = x11434;
int32_t x11441 = x11440;
int32_t x11442 = x11438;
int32_t x11443 = x11439;
for(int x11444=0; x11444 < 1; x11444++) {
int32_t x11445 = x11442;
int32_t x11446 = x11443;
int32_t x11447 = x11441;
int32_t x11448 = x11447;
int32_t x11449 = x11445;
int32_t x11450 = x11446;
for(int x11451=0; x11451 < -7; x11451++) {
int32_t x11452 = x11448;
int32_t x11453 = x11449;
float x11454 = x11353[x11453];
int32_t x11455 = x11450;
float x11456 = x197[x11455];
float x11457 = x11454 + x11456;
x11426[x11452] = x11457;
x11448 += 1;
if (x520) {
x11449 += 1;
} else {
}
if (x452) {
x11450 += 1;
} else {
}

}
x11441 += -7;
if (x452) {
x11442 += -7;
} else {
}
if (x452) {
x11443 += 1;
} else {
}

}
x11434 += -7;
if (x542) {
x11435 += -7;
} else {
}
if (x542) {
x11436 += 1;
} else {
}

}
x11427 += 7;
if (x452) {
x11428 += 7;
} else {
}
if (x452) {
x11429 += -1;
} else {
}

}
float* x11495 = (float*)myMalloc(7 * sizeof(float));;
for(int x11496=0; x11496 < 7; x11496++) {
float x11497 = x11426[x11496];
bool x11498 = x11497 < 0.0f;
if (x11498) {
x11495[x11496] = 0.0f;
} else {
float x11501 = x11426[x11496];
x11495[x11496] = x11501;
}

}
float* x11507 = (float*)myMalloc(x1518 * sizeof(float));;
float* x11508 = (float*)myMalloc(x1158 * sizeof(float));;
for(int x11509=0; x11509 < 1; x11509++) {
int32_t x11510 = x11509 * 7;
float* x11511 = x11495+x11510;
int32_t x11512 = x11509 * x1516;
float* x11513 = x11507+x11512;
int32_t x11514 = x11509 * x1158;
float* x11515 = x11508+x11514;
for(int x11516=0; x11516 < -9; x11516++) {
int32_t x11517 = x11516 / 9;
int32_t x11521 = x11517 * 3;
int32_t x11522 = x11521 * 3;
int32_t x11523 = x11522 * x1150;
int32_t x11524 = x11523 * x1152;
int32_t x11518 = x11516 % 9;
int32_t x11519 = x11518 / 3;
int32_t x11525 = x11519 * 3;
int32_t x11526 = x11525 * x1150;
int32_t x11527 = x11526 * x1152;
int32_t x11528 = x11524 + x11527;
int32_t x11520 = x11518 % 3;
int32_t x11529 = x11520 * x1152;
int32_t x11530 = x11529 * x1152;
int32_t x11531 = x11528 + x11530;
float* x11532 = x11515+x11531;
int32_t x11533 = x11517 * -7;
float* x11534 = x11511+x11533;
int32_t x11546 = 1 - x11520;
bool x11547 = x11546 > 0;
int32_t x11548;
if (x11547) {
x11548 = x11546;
} else {
x11548 = 0;
}
int32_t x11549 = 3 - x11520;
int32_t x11550 = x11549 - 1;
int32_t x11551 = 1 - x11550;
bool x11552 = x11551 > 0;
int32_t x11553;
if (x11552) {
x11553 = x11551;
} else {
x11553 = 0;
}
int32_t x11554 = x1152 - x11553;
int32_t x11555 = x11554 - x11548;
bool x11556 = x11555 <= 0;
bool x11560 = x11548 > 0;
int32_t x11545 = -1 + x11520;
bool x11573 = x11553 > 0;
for(int x11535=0; x11535 < x1150; x11535++) {
int32_t x11536 = x11535 - 1;
int32_t x11537 = x11536 + x11519;
bool x11538 = x11537 < 0;
bool x11539 = x11537 >= 1;
bool x11540 = x11538 || x11539;
if (x11540) {
int32_t x11541 = x11535 * x1152;
float* x11542 = x11532+x11541;
memset(x11542, 0, 4 * x1152);;
} else {
if (x11556) {
int32_t x11541 = x11535 * x1152;
float* x11557 = x11532+x11541;
memset(x11557, 0, 4 * x1152);;
} else {
int32_t x11541 = x11535 * x1152;
if (x11560) {
float* x11561 = x11532+x11541;
memset(x11561, 0, 4 * x11548);;
} else {
}
// may have segfault here
int32_t x11566 = x11541 + x11548;
float* x11567 = x11532+x11566;
int32_t x11568 = x11537 * -7;
int32_t x11569 = x11568 + x11545;
int32_t x11570 = x11569 + x11548;
float* x11571 = x11534+x11570;
memcpy(x11567, x11571, 4 * x11555);;
if (x11573) {
int32_t x11574 = x11541 + x1152;
int32_t x11575 = x11574 - x11553;
float* x11576 = x11532+x11575;
memset(x11576, 0, 4 * x11553);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1153,-9,1,x14,-9,x11515,x1153,1,x11513,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x11595 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x11596 = 0;
int32_t x11597 = 0;
int32_t x11598 = 0;
for(int x11599=0; x11599 < -7; x11599++) {
int32_t x11600 = x11596;
int32_t x11601 = x11597;
float x11602 = x11507[x11601];
int32_t x11603 = x11598;
float x11604 = x124[x11603];
float x11605 = x11602 - x11604;
x11595[x11600] = x11605;
x11596 += 1;
if (x452) {
x11597 += x1516;
} else {
}
if (x452) {
x11598 += -1;
} else {
}

}
float* x11616 = (float*)myMalloc(256 * sizeof(float));;
for(int x11617=0; x11617 < 256; x11617++) {
float x11618 = x63[x11617];
float x11619 = x11618 + 1.0E-5f;
x11616[x11617] = x11619;

}
float* x11623 = (float*)myMalloc(256 * sizeof(float));;
for(int x11624=0; x11624 < 256; x11624++) {
float x11625 = x11616[x11624];
double x11626 = (double)x11625;
double x11627 = sqrt(x11626);
float x11628 = (float)x11627;
x11623[x11624] = x11628;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x11636 = (float*)myMalloc(7 * sizeof(float));;
int32_t x11637 = 0;
int32_t x11638 = 0;
int32_t x11639 = 0;
for(int x11640=0; x11640 < 1; x11640++) {
int32_t x11641 = x11638;
int32_t x11642 = x11639;
int32_t x11643 = x11637;
int32_t x11644 = x11643;
int32_t x11645 = x11641;
int32_t x11646 = x11642;
for(int x11647=0; x11647 < -1; x11647++) {
int32_t x11648 = x11645;
int32_t x11649 = x11646;
int32_t x11650 = x11644;
int32_t x11651 = x11650;
int32_t x11652 = x11648;
int32_t x11653 = x11649;
for(int x11654=0; x11654 < 1; x11654++) {
int32_t x11655 = x11652;
int32_t x11656 = x11653;
int32_t x11657 = x11651;
int32_t x11658 = x11657;
int32_t x11659 = x11655;
int32_t x11660 = x11656;
for(int x11661=0; x11661 < -7; x11661++) {
int32_t x11662 = x11658;
int32_t x11663 = x11659;
float x11664 = x11595[x11663];
int32_t x11665 = x11660;
float x11666 = x11623[x11665];
float x11667 = x11664 / x11666;
x11636[x11662] = x11667;
x11658 += 1;
if (x520) {
x11659 += 1;
} else {
}
if (x452) {
x11660 += 1;
} else {
}

}
x11651 += -7;
if (x452) {
x11652 += -7;
} else {
}
if (x452) {
x11653 += 1;
} else {
}

}
x11644 += -7;
if (x452) {
x11645 += -7;
} else {
}
if (x542) {
x11646 += 1;
} else {
}

}
x11637 += 7;
if (x452) {
x11638 += -7;
} else {
}
if (x452) {
x11639 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x11709 = (float*)myMalloc(7 * sizeof(float));;
int32_t x11710 = 0;
int32_t x11711 = 0;
int32_t x11712 = 0;
for(int x11713=0; x11713 < 1; x11713++) {
int32_t x11714 = x11711;
int32_t x11715 = x11712;
int32_t x11716 = x11710;
int32_t x11717 = x11716;
int32_t x11718 = x11714;
int32_t x11719 = x11715;
for(int x11720=0; x11720 < -1; x11720++) {
int32_t x11721 = x11718;
int32_t x11722 = x11719;
int32_t x11723 = x11717;
int32_t x11724 = x11723;
int32_t x11725 = x11721;
int32_t x11726 = x11722;
for(int x11727=0; x11727 < 1; x11727++) {
int32_t x11728 = x11725;
int32_t x11729 = x11726;
int32_t x11730 = x11724;
int32_t x11731 = x11730;
int32_t x11732 = x11728;
int32_t x11733 = x11729;
for(int x11734=0; x11734 < -7; x11734++) {
int32_t x11735 = x11731;
int32_t x11736 = x11732;
float x11737 = x11636[x11736];
int32_t x11738 = x11733;
float x11739 = x228[x11738];
float x11740 = x11737 * x11739;
x11709[x11735] = x11740;
x11731 += 1;
if (x520) {
x11732 += 1;
} else {
}
if (x452) {
x11733 += 1;
} else {
}

}
x11724 += -7;
if (x452) {
x11725 += -7;
} else {
}
if (x452) {
x11726 += 1;
} else {
}

}
x11717 += -7;
if (x542) {
x11718 += -7;
} else {
}
if (x542) {
x11719 += 1;
} else {
}

}
x11710 += 7;
if (x452) {
x11711 += 7;
} else {
}
if (x452) {
x11712 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x11782 = (float*)myMalloc(7 * sizeof(float));;
int32_t x11783 = 0;
int32_t x11784 = 0;
int32_t x11785 = 0;
for(int x11786=0; x11786 < 1; x11786++) {
int32_t x11787 = x11784;
int32_t x11788 = x11785;
int32_t x11789 = x11783;
int32_t x11790 = x11789;
int32_t x11791 = x11787;
int32_t x11792 = x11788;
for(int x11793=0; x11793 < -1; x11793++) {
int32_t x11794 = x11791;
int32_t x11795 = x11792;
int32_t x11796 = x11790;
int32_t x11797 = x11796;
int32_t x11798 = x11794;
int32_t x11799 = x11795;
for(int x11800=0; x11800 < 1; x11800++) {
int32_t x11801 = x11798;
int32_t x11802 = x11799;
int32_t x11803 = x11797;
int32_t x11804 = x11803;
int32_t x11805 = x11801;
int32_t x11806 = x11802;
for(int x11807=0; x11807 < -7; x11807++) {
int32_t x11808 = x11804;
int32_t x11809 = x11805;
float x11810 = x11709[x11809];
int32_t x11811 = x11806;
float x11812 = x192[x11811];
float x11813 = x11810 + x11812;
x11782[x11808] = x11813;
x11804 += 1;
if (x520) {
x11805 += 1;
} else {
}
if (x452) {
x11806 += 1;
} else {
}

}
x11797 += -7;
if (x452) {
x11798 += -7;
} else {
}
if (x452) {
x11799 += 1;
} else {
}

}
x11790 += -7;
if (x542) {
x11791 += -7;
} else {
}
if (x542) {
x11792 += 1;
} else {
}

}
x11783 += 7;
if (x452) {
x11784 += 7;
} else {
}
if (x452) {
x11785 += -1;
} else {
}

}
float* x11851 = (float*)myMalloc(7 * sizeof(float));;
for(int x11852=0; x11852 < 7; x11852++) {
float x11853 = x11782[x11852];
bool x11854 = x11853 < 0.0f;
if (x11854) {
x11851[x11852] = 0.0f;
} else {
float x11857 = x11782[x11852];
x11851[x11852] = x11857;
}

}
float* x11863 = (float*)myMalloc(x9457 * sizeof(float));;
float* x11864 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x11865=0; x11865 < 1; x11865++) {
int32_t x11866 = x11865 * 7;
float* x11867 = x11851+x11866;
int32_t x11868 = x11865 * x9455;
float* x11869 = x11863+x11868;
int32_t x11870 = x11865 * x1520;
float* x11871 = x11864+x11870;
for(int x11872=0; x11872 < -1; x11872++) {
int32_t x11873 = x11872 / 1;
int32_t x11877 = x11873 * x1150;
int32_t x11878 = x11877 * x1152;
int32_t x11874 = x11872 % 1;
int32_t x11875 = x11874 / 1;
int32_t x11879 = x11875 * x1150;
int32_t x11880 = x11879 * x1152;
int32_t x11881 = x11878 + x11880;
int32_t x11876 = x11874 % 1;
int32_t x11882 = x11876 * x1152;
int32_t x11883 = x11882 * x1152;
int32_t x11884 = x11881 + x11883;
float* x11885 = x11871+x11884;
int32_t x11886 = x11873 * -7;
float* x11887 = x11867+x11886;
for(int x11888=0; x11888 < x1150; x11888++) {
int32_t x11890 = x11888 * x1152;
float* x11891 = x11885+x11890;
int32_t x11889 = x11888 + x11875;
int32_t x11892 = x11889 * -7;
int32_t x11893 = x11892 + x11876;
float* x11894 = x11887+x11893;
memcpy(x11891, x11894, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x1153,-1,1,x116,-1,x11871,x1153,1,x11869,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(1024) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x11907 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x11908 = 0;
int32_t x11909 = 0;
int32_t x11910 = 0;
for(int x11911=0; x11911 < -7; x11911++) {
int32_t x11912 = x11908;
int32_t x11913 = x11909;
float x11914 = x11863[x11913];
int32_t x11915 = x11910;
float x11916 = x140[x11915];
float x11917 = x11914 - x11916;
x11907[x11912] = x11917;
x11908 += 1;
if (x452) {
x11909 += x9455;
} else {
}
if (x452) {
x11910 += -1;
} else {
}

}
float* x11928 = (float*)myMalloc(1024 * sizeof(float));;
for(int x11929=0; x11929 < 1024; x11929++) {
float x11930 = x188[x11929];
float x11931 = x11930 + 1.0E-5f;
x11928[x11929] = x11931;

}
float* x11935 = (float*)myMalloc(1024 * sizeof(float));;
for(int x11936=0; x11936 < 1024; x11936++) {
float x11937 = x11928[x11936];
double x11938 = (double)x11937;
double x11939 = sqrt(x11938);
float x11940 = (float)x11939;
x11935[x11936] = x11940;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x11948 = (float*)myMalloc(7 * sizeof(float));;
int32_t x11949 = 0;
int32_t x11950 = 0;
int32_t x11951 = 0;
for(int x11952=0; x11952 < 1; x11952++) {
int32_t x11953 = x11950;
int32_t x11954 = x11951;
int32_t x11955 = x11949;
int32_t x11956 = x11955;
int32_t x11957 = x11953;
int32_t x11958 = x11954;
for(int x11959=0; x11959 < -1; x11959++) {
int32_t x11960 = x11957;
int32_t x11961 = x11958;
int32_t x11962 = x11956;
int32_t x11963 = x11962;
int32_t x11964 = x11960;
int32_t x11965 = x11961;
for(int x11966=0; x11966 < 1; x11966++) {
int32_t x11967 = x11964;
int32_t x11968 = x11965;
int32_t x11969 = x11963;
int32_t x11970 = x11969;
int32_t x11971 = x11967;
int32_t x11972 = x11968;
for(int x11973=0; x11973 < -7; x11973++) {
int32_t x11974 = x11970;
int32_t x11975 = x11971;
float x11976 = x11907[x11975];
int32_t x11977 = x11972;
float x11978 = x11935[x11977];
float x11979 = x11976 / x11978;
x11948[x11974] = x11979;
x11970 += 1;
if (x520) {
x11971 += 1;
} else {
}
if (x452) {
x11972 += 1;
} else {
}

}
x11963 += -7;
if (x452) {
x11964 += -7;
} else {
}
if (x452) {
x11965 += 1;
} else {
}

}
x11956 += -7;
if (x452) {
x11957 += -7;
} else {
}
if (x542) {
x11958 += 1;
} else {
}

}
x11949 += 7;
if (x452) {
x11950 += -7;
} else {
}
if (x452) {
x11951 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x12021 = (float*)myMalloc(7 * sizeof(float));;
int32_t x12022 = 0;
int32_t x12023 = 0;
int32_t x12024 = 0;
for(int x12025=0; x12025 < 1; x12025++) {
int32_t x12026 = x12023;
int32_t x12027 = x12024;
int32_t x12028 = x12022;
int32_t x12029 = x12028;
int32_t x12030 = x12026;
int32_t x12031 = x12027;
for(int x12032=0; x12032 < -1; x12032++) {
int32_t x12033 = x12030;
int32_t x12034 = x12031;
int32_t x12035 = x12029;
int32_t x12036 = x12035;
int32_t x12037 = x12033;
int32_t x12038 = x12034;
for(int x12039=0; x12039 < 1; x12039++) {
int32_t x12040 = x12037;
int32_t x12041 = x12038;
int32_t x12042 = x12036;
int32_t x12043 = x12042;
int32_t x12044 = x12040;
int32_t x12045 = x12041;
for(int x12046=0; x12046 < -7; x12046++) {
int32_t x12047 = x12043;
int32_t x12048 = x12044;
float x12049 = x11948[x12048];
int32_t x12050 = x12045;
float x12051 = x263[x12050];
float x12052 = x12049 * x12051;
x12021[x12047] = x12052;
x12043 += 1;
if (x520) {
x12044 += 1;
} else {
}
if (x452) {
x12045 += 1;
} else {
}

}
x12036 += -7;
if (x452) {
x12037 += -7;
} else {
}
if (x452) {
x12038 += 1;
} else {
}

}
x12029 += -7;
if (x542) {
x12030 += -7;
} else {
}
if (x542) {
x12031 += 1;
} else {
}

}
x12022 += 7;
if (x452) {
x12023 += 7;
} else {
}
if (x452) {
x12024 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x12094 = (float*)myMalloc(7 * sizeof(float));;
int32_t x12095 = 0;
int32_t x12096 = 0;
int32_t x12097 = 0;
for(int x12098=0; x12098 < 1; x12098++) {
int32_t x12099 = x12096;
int32_t x12100 = x12097;
int32_t x12101 = x12095;
int32_t x12102 = x12101;
int32_t x12103 = x12099;
int32_t x12104 = x12100;
for(int x12105=0; x12105 < -1; x12105++) {
int32_t x12106 = x12103;
int32_t x12107 = x12104;
int32_t x12108 = x12102;
int32_t x12109 = x12108;
int32_t x12110 = x12106;
int32_t x12111 = x12107;
for(int x12112=0; x12112 < 1; x12112++) {
int32_t x12113 = x12110;
int32_t x12114 = x12111;
int32_t x12115 = x12109;
int32_t x12116 = x12115;
int32_t x12117 = x12113;
int32_t x12118 = x12114;
for(int x12119=0; x12119 < -7; x12119++) {
int32_t x12120 = x12116;
int32_t x12121 = x12117;
float x12122 = x12021[x12121];
int32_t x12123 = x12118;
float x12124 = x57[x12123];
float x12125 = x12122 + x12124;
x12094[x12120] = x12125;
x12116 += 1;
if (x520) {
x12117 += 1;
} else {
}
if (x452) {
x12118 += 1;
} else {
}

}
x12109 += -7;
if (x452) {
x12110 += -7;
} else {
}
if (x452) {
x12111 += 1;
} else {
}

}
x12102 += -7;
if (x542) {
x12103 += -7;
} else {
}
if (x542) {
x12104 += 1;
} else {
}

}
x12095 += 7;
if (x452) {
x12096 += 7;
} else {
}
if (x452) {
x12097 += -1;
} else {
}

}
int32_t x12163 = 0;
int32_t x12164 = 0;
int32_t x12165 = 0;
for(int x12166=0; x12166 < 1; x12166++) {
int32_t x12167 = x12164;
int32_t x12168 = x12165;
int32_t x12169 = x12163;
int32_t x12170 = x12169;
int32_t x12171 = x12167;
int32_t x12172 = x12168;
for(int x12173=0; x12173 < -1; x12173++) {
int32_t x12174 = x12171;
int32_t x12175 = x12172;
int32_t x12176 = x12170;
int32_t x12177 = x12176;
int32_t x12178 = x12174;
int32_t x12179 = x12175;
for(int x12180=0; x12180 < 1; x12180++) {
int32_t x12181 = x12178;
int32_t x12182 = x12179;
int32_t x12183 = x12177;
int32_t x12184 = x12183;
int32_t x12185 = x12181;
int32_t x12186 = x12182;
for(int x12187=0; x12187 < -7; x12187++) {
int32_t x12188 = x12185;
float x12189 = x12094[x12188];
int32_t x12190 = x12186;
float x12191 = x11183[x12190];
float x12192 = x12189 + x12191;
x12094[x12188] = x12192;
x12184 += 1;
if (x520) {
x12185 += 1;
} else {
}
if (x520) {
x12186 += 1;
} else {
}

}
x12177 += -7;
if (x452) {
x12178 += -7;
} else {
}
if (x452) {
x12179 += -7;
} else {
}

}
x12170 += -7;
if (x542) {
x12171 += -7;
} else {
}
if (x542) {
x12172 += -7;
} else {
}

}
x12163 += 7;
if (x452) {
x12164 += 7;
} else {
}
if (x452) {
x12165 += 7;
} else {
}

}
float* x12230 = (float*)myMalloc(7 * sizeof(float));;
for(int x12231=0; x12231 < 7; x12231++) {
float x12232 = x12094[x12231];
bool x12233 = x12232 < 0.0f;
if (x12233) {
x12230[x12231] = 0.0f;
} else {
float x12236 = x12094[x12231];
x12230[x12231] = x12236;
}

}
float* x12242 = (float*)myMalloc(x1518 * sizeof(float));;
float* x12243 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x12244=0; x12244 < 1; x12244++) {
int32_t x12245 = x12244 * 7;
float* x12246 = x12230+x12245;
int32_t x12247 = x12244 * x1516;
float* x12248 = x12242+x12247;
int32_t x12249 = x12244 * x1520;
float* x12250 = x12243+x12249;
for(int x12251=0; x12251 < -1; x12251++) {
int32_t x12252 = x12251 / 1;
int32_t x12256 = x12252 * x1150;
int32_t x12257 = x12256 * x1152;
int32_t x12253 = x12251 % 1;
int32_t x12254 = x12253 / 1;
int32_t x12258 = x12254 * x1150;
int32_t x12259 = x12258 * x1152;
int32_t x12260 = x12257 + x12259;
int32_t x12255 = x12253 % 1;
int32_t x12261 = x12255 * x1152;
int32_t x12262 = x12261 * x1152;
int32_t x12263 = x12260 + x12262;
float* x12264 = x12250+x12263;
int32_t x12265 = x12252 * -7;
float* x12266 = x12246+x12265;
for(int x12267=0; x12267 < x1150; x12267++) {
int32_t x12269 = x12267 * x1152;
float* x12270 = x12264+x12269;
int32_t x12268 = x12267 + x12254;
int32_t x12271 = x12268 * -7;
int32_t x12272 = x12271 + x12255;
float* x12273 = x12266+x12272;
memcpy(x12270, x12273, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1153,-1,1,x6,-1,x12250,x1153,1,x12248,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x12286 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x12287 = 0;
int32_t x12288 = 0;
int32_t x12289 = 0;
for(int x12290=0; x12290 < -7; x12290++) {
int32_t x12291 = x12287;
int32_t x12292 = x12288;
float x12293 = x12242[x12292];
int32_t x12294 = x12289;
float x12295 = x163[x12294];
float x12296 = x12293 - x12295;
x12286[x12291] = x12296;
x12287 += 1;
if (x452) {
x12288 += x1516;
} else {
}
if (x452) {
x12289 += -1;
} else {
}

}
float* x12307 = (float*)myMalloc(256 * sizeof(float));;
for(int x12308=0; x12308 < 256; x12308++) {
float x12309 = x98[x12308];
float x12310 = x12309 + 1.0E-5f;
x12307[x12308] = x12310;

}
float* x12314 = (float*)myMalloc(256 * sizeof(float));;
for(int x12315=0; x12315 < 256; x12315++) {
float x12316 = x12307[x12315];
double x12317 = (double)x12316;
double x12318 = sqrt(x12317);
float x12319 = (float)x12318;
x12314[x12315] = x12319;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x12327 = (float*)myMalloc(7 * sizeof(float));;
int32_t x12328 = 0;
int32_t x12329 = 0;
int32_t x12330 = 0;
for(int x12331=0; x12331 < 1; x12331++) {
int32_t x12332 = x12329;
int32_t x12333 = x12330;
int32_t x12334 = x12328;
int32_t x12335 = x12334;
int32_t x12336 = x12332;
int32_t x12337 = x12333;
for(int x12338=0; x12338 < -1; x12338++) {
int32_t x12339 = x12336;
int32_t x12340 = x12337;
int32_t x12341 = x12335;
int32_t x12342 = x12341;
int32_t x12343 = x12339;
int32_t x12344 = x12340;
for(int x12345=0; x12345 < 1; x12345++) {
int32_t x12346 = x12343;
int32_t x12347 = x12344;
int32_t x12348 = x12342;
int32_t x12349 = x12348;
int32_t x12350 = x12346;
int32_t x12351 = x12347;
for(int x12352=0; x12352 < -7; x12352++) {
int32_t x12353 = x12349;
int32_t x12354 = x12350;
float x12355 = x12286[x12354];
int32_t x12356 = x12351;
float x12357 = x12314[x12356];
float x12358 = x12355 / x12357;
x12327[x12353] = x12358;
x12349 += 1;
if (x520) {
x12350 += 1;
} else {
}
if (x452) {
x12351 += 1;
} else {
}

}
x12342 += -7;
if (x452) {
x12343 += -7;
} else {
}
if (x452) {
x12344 += 1;
} else {
}

}
x12335 += -7;
if (x452) {
x12336 += -7;
} else {
}
if (x542) {
x12337 += 1;
} else {
}

}
x12328 += 7;
if (x452) {
x12329 += -7;
} else {
}
if (x452) {
x12330 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x12400 = (float*)myMalloc(7 * sizeof(float));;
int32_t x12401 = 0;
int32_t x12402 = 0;
int32_t x12403 = 0;
for(int x12404=0; x12404 < 1; x12404++) {
int32_t x12405 = x12402;
int32_t x12406 = x12403;
int32_t x12407 = x12401;
int32_t x12408 = x12407;
int32_t x12409 = x12405;
int32_t x12410 = x12406;
for(int x12411=0; x12411 < -1; x12411++) {
int32_t x12412 = x12409;
int32_t x12413 = x12410;
int32_t x12414 = x12408;
int32_t x12415 = x12414;
int32_t x12416 = x12412;
int32_t x12417 = x12413;
for(int x12418=0; x12418 < 1; x12418++) {
int32_t x12419 = x12416;
int32_t x12420 = x12417;
int32_t x12421 = x12415;
int32_t x12422 = x12421;
int32_t x12423 = x12419;
int32_t x12424 = x12420;
for(int x12425=0; x12425 < -7; x12425++) {
int32_t x12426 = x12422;
int32_t x12427 = x12423;
float x12428 = x12327[x12427];
int32_t x12429 = x12424;
float x12430 = x92[x12429];
float x12431 = x12428 * x12430;
x12400[x12426] = x12431;
x12422 += 1;
if (x520) {
x12423 += 1;
} else {
}
if (x452) {
x12424 += 1;
} else {
}

}
x12415 += -7;
if (x452) {
x12416 += -7;
} else {
}
if (x452) {
x12417 += 1;
} else {
}

}
x12408 += -7;
if (x542) {
x12409 += -7;
} else {
}
if (x542) {
x12410 += 1;
} else {
}

}
x12401 += 7;
if (x452) {
x12402 += 7;
} else {
}
if (x452) {
x12403 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x12473 = (float*)myMalloc(7 * sizeof(float));;
int32_t x12474 = 0;
int32_t x12475 = 0;
int32_t x12476 = 0;
for(int x12477=0; x12477 < 1; x12477++) {
int32_t x12478 = x12475;
int32_t x12479 = x12476;
int32_t x12480 = x12474;
int32_t x12481 = x12480;
int32_t x12482 = x12478;
int32_t x12483 = x12479;
for(int x12484=0; x12484 < -1; x12484++) {
int32_t x12485 = x12482;
int32_t x12486 = x12483;
int32_t x12487 = x12481;
int32_t x12488 = x12487;
int32_t x12489 = x12485;
int32_t x12490 = x12486;
for(int x12491=0; x12491 < 1; x12491++) {
int32_t x12492 = x12489;
int32_t x12493 = x12490;
int32_t x12494 = x12488;
int32_t x12495 = x12494;
int32_t x12496 = x12492;
int32_t x12497 = x12493;
for(int x12498=0; x12498 < -7; x12498++) {
int32_t x12499 = x12495;
int32_t x12500 = x12496;
float x12501 = x12400[x12500];
int32_t x12502 = x12497;
float x12503 = x241[x12502];
float x12504 = x12501 + x12503;
x12473[x12499] = x12504;
x12495 += 1;
if (x520) {
x12496 += 1;
} else {
}
if (x452) {
x12497 += 1;
} else {
}

}
x12488 += -7;
if (x452) {
x12489 += -7;
} else {
}
if (x452) {
x12490 += 1;
} else {
}

}
x12481 += -7;
if (x542) {
x12482 += -7;
} else {
}
if (x542) {
x12483 += 1;
} else {
}

}
x12474 += 7;
if (x452) {
x12475 += 7;
} else {
}
if (x452) {
x12476 += -1;
} else {
}

}
float* x12542 = (float*)myMalloc(7 * sizeof(float));;
for(int x12543=0; x12543 < 7; x12543++) {
float x12544 = x12473[x12543];
bool x12545 = x12544 < 0.0f;
if (x12545) {
x12542[x12543] = 0.0f;
} else {
float x12548 = x12473[x12543];
x12542[x12543] = x12548;
}

}
float* x12554 = (float*)myMalloc(x1518 * sizeof(float));;
float* x12555 = (float*)myMalloc(x1158 * sizeof(float));;
for(int x12556=0; x12556 < 1; x12556++) {
int32_t x12557 = x12556 * 7;
float* x12558 = x12542+x12557;
int32_t x12559 = x12556 * x1516;
float* x12560 = x12554+x12559;
int32_t x12561 = x12556 * x1158;
float* x12562 = x12555+x12561;
for(int x12563=0; x12563 < -9; x12563++) {
int32_t x12564 = x12563 / 9;
int32_t x12568 = x12564 * 3;
int32_t x12569 = x12568 * 3;
int32_t x12570 = x12569 * x1150;
int32_t x12571 = x12570 * x1152;
int32_t x12565 = x12563 % 9;
int32_t x12566 = x12565 / 3;
int32_t x12572 = x12566 * 3;
int32_t x12573 = x12572 * x1150;
int32_t x12574 = x12573 * x1152;
int32_t x12575 = x12571 + x12574;
int32_t x12567 = x12565 % 3;
int32_t x12576 = x12567 * x1152;
int32_t x12577 = x12576 * x1152;
int32_t x12578 = x12575 + x12577;
float* x12579 = x12562+x12578;
int32_t x12580 = x12564 * -7;
float* x12581 = x12558+x12580;
int32_t x12593 = 1 - x12567;
bool x12594 = x12593 > 0;
int32_t x12595;
if (x12594) {
x12595 = x12593;
} else {
x12595 = 0;
}
int32_t x12596 = 3 - x12567;
int32_t x12597 = x12596 - 1;
int32_t x12598 = 1 - x12597;
bool x12599 = x12598 > 0;
int32_t x12600;
if (x12599) {
x12600 = x12598;
} else {
x12600 = 0;
}
int32_t x12601 = x1152 - x12600;
int32_t x12602 = x12601 - x12595;
bool x12603 = x12602 <= 0;
bool x12607 = x12595 > 0;
int32_t x12592 = -1 + x12567;
bool x12620 = x12600 > 0;
for(int x12582=0; x12582 < x1150; x12582++) {
int32_t x12583 = x12582 - 1;
int32_t x12584 = x12583 + x12566;
bool x12585 = x12584 < 0;
bool x12586 = x12584 >= 1;
bool x12587 = x12585 || x12586;
if (x12587) {
int32_t x12588 = x12582 * x1152;
float* x12589 = x12579+x12588;
memset(x12589, 0, 4 * x1152);;
} else {
if (x12603) {
int32_t x12588 = x12582 * x1152;
float* x12604 = x12579+x12588;
memset(x12604, 0, 4 * x1152);;
} else {
int32_t x12588 = x12582 * x1152;
if (x12607) {
float* x12608 = x12579+x12588;
memset(x12608, 0, 4 * x12595);;
} else {
}
// may have segfault here
int32_t x12613 = x12588 + x12595;
float* x12614 = x12579+x12613;
int32_t x12615 = x12584 * -7;
int32_t x12616 = x12615 + x12592;
int32_t x12617 = x12616 + x12595;
float* x12618 = x12581+x12617;
memcpy(x12614, x12618, 4 * x12602);;
if (x12620) {
int32_t x12621 = x12588 + x1152;
int32_t x12622 = x12621 - x12600;
float* x12623 = x12579+x12622;
memset(x12623, 0, 4 * x12600);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1153,-9,1,x249,-9,x12562,x1153,1,x12560,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x12642 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x12643 = 0;
int32_t x12644 = 0;
int32_t x12645 = 0;
for(int x12646=0; x12646 < -7; x12646++) {
int32_t x12647 = x12643;
int32_t x12648 = x12644;
float x12649 = x12554[x12648];
int32_t x12650 = x12645;
float x12651 = x186[x12650];
float x12652 = x12649 - x12651;
x12642[x12647] = x12652;
x12643 += 1;
if (x452) {
x12644 += x1516;
} else {
}
if (x452) {
x12645 += -1;
} else {
}

}
float* x12663 = (float*)myMalloc(256 * sizeof(float));;
for(int x12664=0; x12664 < 256; x12664++) {
float x12665 = x230[x12664];
float x12666 = x12665 + 1.0E-5f;
x12663[x12664] = x12666;

}
float* x12670 = (float*)myMalloc(256 * sizeof(float));;
for(int x12671=0; x12671 < 256; x12671++) {
float x12672 = x12663[x12671];
double x12673 = (double)x12672;
double x12674 = sqrt(x12673);
float x12675 = (float)x12674;
x12670[x12671] = x12675;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x12683 = (float*)myMalloc(7 * sizeof(float));;
int32_t x12684 = 0;
int32_t x12685 = 0;
int32_t x12686 = 0;
for(int x12687=0; x12687 < 1; x12687++) {
int32_t x12688 = x12685;
int32_t x12689 = x12686;
int32_t x12690 = x12684;
int32_t x12691 = x12690;
int32_t x12692 = x12688;
int32_t x12693 = x12689;
for(int x12694=0; x12694 < -1; x12694++) {
int32_t x12695 = x12692;
int32_t x12696 = x12693;
int32_t x12697 = x12691;
int32_t x12698 = x12697;
int32_t x12699 = x12695;
int32_t x12700 = x12696;
for(int x12701=0; x12701 < 1; x12701++) {
int32_t x12702 = x12699;
int32_t x12703 = x12700;
int32_t x12704 = x12698;
int32_t x12705 = x12704;
int32_t x12706 = x12702;
int32_t x12707 = x12703;
for(int x12708=0; x12708 < -7; x12708++) {
int32_t x12709 = x12705;
int32_t x12710 = x12706;
float x12711 = x12642[x12710];
int32_t x12712 = x12707;
float x12713 = x12670[x12712];
float x12714 = x12711 / x12713;
x12683[x12709] = x12714;
x12705 += 1;
if (x520) {
x12706 += 1;
} else {
}
if (x452) {
x12707 += 1;
} else {
}

}
x12698 += -7;
if (x452) {
x12699 += -7;
} else {
}
if (x452) {
x12700 += 1;
} else {
}

}
x12691 += -7;
if (x452) {
x12692 += -7;
} else {
}
if (x542) {
x12693 += 1;
} else {
}

}
x12684 += 7;
if (x452) {
x12685 += -7;
} else {
}
if (x452) {
x12686 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x12756 = (float*)myMalloc(7 * sizeof(float));;
int32_t x12757 = 0;
int32_t x12758 = 0;
int32_t x12759 = 0;
for(int x12760=0; x12760 < 1; x12760++) {
int32_t x12761 = x12758;
int32_t x12762 = x12759;
int32_t x12763 = x12757;
int32_t x12764 = x12763;
int32_t x12765 = x12761;
int32_t x12766 = x12762;
for(int x12767=0; x12767 < -1; x12767++) {
int32_t x12768 = x12765;
int32_t x12769 = x12766;
int32_t x12770 = x12764;
int32_t x12771 = x12770;
int32_t x12772 = x12768;
int32_t x12773 = x12769;
for(int x12774=0; x12774 < 1; x12774++) {
int32_t x12775 = x12772;
int32_t x12776 = x12773;
int32_t x12777 = x12771;
int32_t x12778 = x12777;
int32_t x12779 = x12775;
int32_t x12780 = x12776;
for(int x12781=0; x12781 < -7; x12781++) {
int32_t x12782 = x12778;
int32_t x12783 = x12779;
float x12784 = x12683[x12783];
int32_t x12785 = x12780;
float x12786 = x74[x12785];
float x12787 = x12784 * x12786;
x12756[x12782] = x12787;
x12778 += 1;
if (x520) {
x12779 += 1;
} else {
}
if (x452) {
x12780 += 1;
} else {
}

}
x12771 += -7;
if (x452) {
x12772 += -7;
} else {
}
if (x452) {
x12773 += 1;
} else {
}

}
x12764 += -7;
if (x542) {
x12765 += -7;
} else {
}
if (x542) {
x12766 += 1;
} else {
}

}
x12757 += 7;
if (x452) {
x12758 += 7;
} else {
}
if (x452) {
x12759 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x12829 = (float*)myMalloc(7 * sizeof(float));;
int32_t x12830 = 0;
int32_t x12831 = 0;
int32_t x12832 = 0;
for(int x12833=0; x12833 < 1; x12833++) {
int32_t x12834 = x12831;
int32_t x12835 = x12832;
int32_t x12836 = x12830;
int32_t x12837 = x12836;
int32_t x12838 = x12834;
int32_t x12839 = x12835;
for(int x12840=0; x12840 < -1; x12840++) {
int32_t x12841 = x12838;
int32_t x12842 = x12839;
int32_t x12843 = x12837;
int32_t x12844 = x12843;
int32_t x12845 = x12841;
int32_t x12846 = x12842;
for(int x12847=0; x12847 < 1; x12847++) {
int32_t x12848 = x12845;
int32_t x12849 = x12846;
int32_t x12850 = x12844;
int32_t x12851 = x12850;
int32_t x12852 = x12848;
int32_t x12853 = x12849;
for(int x12854=0; x12854 < -7; x12854++) {
int32_t x12855 = x12851;
int32_t x12856 = x12852;
float x12857 = x12756[x12856];
int32_t x12858 = x12853;
float x12859 = x136[x12858];
float x12860 = x12857 + x12859;
x12829[x12855] = x12860;
x12851 += 1;
if (x520) {
x12852 += 1;
} else {
}
if (x452) {
x12853 += 1;
} else {
}

}
x12844 += -7;
if (x452) {
x12845 += -7;
} else {
}
if (x452) {
x12846 += 1;
} else {
}

}
x12837 += -7;
if (x542) {
x12838 += -7;
} else {
}
if (x542) {
x12839 += 1;
} else {
}

}
x12830 += 7;
if (x452) {
x12831 += 7;
} else {
}
if (x452) {
x12832 += -1;
} else {
}

}
float* x12898 = (float*)myMalloc(7 * sizeof(float));;
for(int x12899=0; x12899 < 7; x12899++) {
float x12900 = x12829[x12899];
bool x12901 = x12900 < 0.0f;
if (x12901) {
x12898[x12899] = 0.0f;
} else {
float x12904 = x12829[x12899];
x12898[x12899] = x12904;
}

}
float* x12910 = (float*)myMalloc(x9457 * sizeof(float));;
float* x12911 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x12912=0; x12912 < 1; x12912++) {
int32_t x12913 = x12912 * 7;
float* x12914 = x12898+x12913;
int32_t x12915 = x12912 * x9455;
float* x12916 = x12910+x12915;
int32_t x12917 = x12912 * x1520;
float* x12918 = x12911+x12917;
for(int x12919=0; x12919 < -1; x12919++) {
int32_t x12920 = x12919 / 1;
int32_t x12924 = x12920 * x1150;
int32_t x12925 = x12924 * x1152;
int32_t x12921 = x12919 % 1;
int32_t x12922 = x12921 / 1;
int32_t x12926 = x12922 * x1150;
int32_t x12927 = x12926 * x1152;
int32_t x12928 = x12925 + x12927;
int32_t x12923 = x12921 % 1;
int32_t x12929 = x12923 * x1152;
int32_t x12930 = x12929 * x1152;
int32_t x12931 = x12928 + x12930;
float* x12932 = x12918+x12931;
int32_t x12933 = x12920 * -7;
float* x12934 = x12914+x12933;
for(int x12935=0; x12935 < x1150; x12935++) {
int32_t x12937 = x12935 * x1152;
float* x12938 = x12932+x12937;
int32_t x12936 = x12935 + x12922;
int32_t x12939 = x12936 * -7;
int32_t x12940 = x12939 + x12923;
float* x12941 = x12934+x12940;
memcpy(x12938, x12941, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x1153,-1,1,x89,-1,x12918,x1153,1,x12916,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(1024) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x12954 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x12955 = 0;
int32_t x12956 = 0;
int32_t x12957 = 0;
for(int x12958=0; x12958 < -7; x12958++) {
int32_t x12959 = x12955;
int32_t x12960 = x12956;
float x12961 = x12910[x12960];
int32_t x12962 = x12957;
float x12963 = x231[x12962];
float x12964 = x12961 - x12963;
x12954[x12959] = x12964;
x12955 += 1;
if (x452) {
x12956 += x9455;
} else {
}
if (x452) {
x12957 += -1;
} else {
}

}
float* x12975 = (float*)myMalloc(1024 * sizeof(float));;
for(int x12976=0; x12976 < 1024; x12976++) {
float x12977 = x161[x12976];
float x12978 = x12977 + 1.0E-5f;
x12975[x12976] = x12978;

}
float* x12982 = (float*)myMalloc(1024 * sizeof(float));;
for(int x12983=0; x12983 < 1024; x12983++) {
float x12984 = x12975[x12983];
double x12985 = (double)x12984;
double x12986 = sqrt(x12985);
float x12987 = (float)x12986;
x12982[x12983] = x12987;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x12995 = (float*)myMalloc(7 * sizeof(float));;
int32_t x12996 = 0;
int32_t x12997 = 0;
int32_t x12998 = 0;
for(int x12999=0; x12999 < 1; x12999++) {
int32_t x13000 = x12997;
int32_t x13001 = x12998;
int32_t x13002 = x12996;
int32_t x13003 = x13002;
int32_t x13004 = x13000;
int32_t x13005 = x13001;
for(int x13006=0; x13006 < -1; x13006++) {
int32_t x13007 = x13004;
int32_t x13008 = x13005;
int32_t x13009 = x13003;
int32_t x13010 = x13009;
int32_t x13011 = x13007;
int32_t x13012 = x13008;
for(int x13013=0; x13013 < 1; x13013++) {
int32_t x13014 = x13011;
int32_t x13015 = x13012;
int32_t x13016 = x13010;
int32_t x13017 = x13016;
int32_t x13018 = x13014;
int32_t x13019 = x13015;
for(int x13020=0; x13020 < -7; x13020++) {
int32_t x13021 = x13017;
int32_t x13022 = x13018;
float x13023 = x12954[x13022];
int32_t x13024 = x13019;
float x13025 = x12982[x13024];
float x13026 = x13023 / x13025;
x12995[x13021] = x13026;
x13017 += 1;
if (x520) {
x13018 += 1;
} else {
}
if (x452) {
x13019 += 1;
} else {
}

}
x13010 += -7;
if (x452) {
x13011 += -7;
} else {
}
if (x452) {
x13012 += 1;
} else {
}

}
x13003 += -7;
if (x452) {
x13004 += -7;
} else {
}
if (x542) {
x13005 += 1;
} else {
}

}
x12996 += 7;
if (x452) {
x12997 += -7;
} else {
}
if (x452) {
x12998 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x13068 = (float*)myMalloc(7 * sizeof(float));;
int32_t x13069 = 0;
int32_t x13070 = 0;
int32_t x13071 = 0;
for(int x13072=0; x13072 < 1; x13072++) {
int32_t x13073 = x13070;
int32_t x13074 = x13071;
int32_t x13075 = x13069;
int32_t x13076 = x13075;
int32_t x13077 = x13073;
int32_t x13078 = x13074;
for(int x13079=0; x13079 < -1; x13079++) {
int32_t x13080 = x13077;
int32_t x13081 = x13078;
int32_t x13082 = x13076;
int32_t x13083 = x13082;
int32_t x13084 = x13080;
int32_t x13085 = x13081;
for(int x13086=0; x13086 < 1; x13086++) {
int32_t x13087 = x13084;
int32_t x13088 = x13085;
int32_t x13089 = x13083;
int32_t x13090 = x13089;
int32_t x13091 = x13087;
int32_t x13092 = x13088;
for(int x13093=0; x13093 < -7; x13093++) {
int32_t x13094 = x13090;
int32_t x13095 = x13091;
float x13096 = x12995[x13095];
int32_t x13097 = x13092;
float x13098 = x238[x13097];
float x13099 = x13096 * x13098;
x13068[x13094] = x13099;
x13090 += 1;
if (x520) {
x13091 += 1;
} else {
}
if (x452) {
x13092 += 1;
} else {
}

}
x13083 += -7;
if (x452) {
x13084 += -7;
} else {
}
if (x452) {
x13085 += 1;
} else {
}

}
x13076 += -7;
if (x542) {
x13077 += -7;
} else {
}
if (x542) {
x13078 += 1;
} else {
}

}
x13069 += 7;
if (x452) {
x13070 += 7;
} else {
}
if (x452) {
x13071 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x13141 = (float*)myMalloc(7 * sizeof(float));;
int32_t x13142 = 0;
int32_t x13143 = 0;
int32_t x13144 = 0;
for(int x13145=0; x13145 < 1; x13145++) {
int32_t x13146 = x13143;
int32_t x13147 = x13144;
int32_t x13148 = x13142;
int32_t x13149 = x13148;
int32_t x13150 = x13146;
int32_t x13151 = x13147;
for(int x13152=0; x13152 < -1; x13152++) {
int32_t x13153 = x13150;
int32_t x13154 = x13151;
int32_t x13155 = x13149;
int32_t x13156 = x13155;
int32_t x13157 = x13153;
int32_t x13158 = x13154;
for(int x13159=0; x13159 < 1; x13159++) {
int32_t x13160 = x13157;
int32_t x13161 = x13158;
int32_t x13162 = x13156;
int32_t x13163 = x13162;
int32_t x13164 = x13160;
int32_t x13165 = x13161;
for(int x13166=0; x13166 < -7; x13166++) {
int32_t x13167 = x13163;
int32_t x13168 = x13164;
float x13169 = x13068[x13168];
int32_t x13170 = x13165;
float x13171 = x146[x13170];
float x13172 = x13169 + x13171;
x13141[x13167] = x13172;
x13163 += 1;
if (x520) {
x13164 += 1;
} else {
}
if (x452) {
x13165 += 1;
} else {
}

}
x13156 += -7;
if (x452) {
x13157 += -7;
} else {
}
if (x452) {
x13158 += 1;
} else {
}

}
x13149 += -7;
if (x542) {
x13150 += -7;
} else {
}
if (x542) {
x13151 += 1;
} else {
}

}
x13142 += 7;
if (x452) {
x13143 += 7;
} else {
}
if (x452) {
x13144 += -1;
} else {
}

}
int32_t x13210 = 0;
int32_t x13211 = 0;
int32_t x13212 = 0;
for(int x13213=0; x13213 < 1; x13213++) {
int32_t x13214 = x13211;
int32_t x13215 = x13212;
int32_t x13216 = x13210;
int32_t x13217 = x13216;
int32_t x13218 = x13214;
int32_t x13219 = x13215;
for(int x13220=0; x13220 < -1; x13220++) {
int32_t x13221 = x13218;
int32_t x13222 = x13219;
int32_t x13223 = x13217;
int32_t x13224 = x13223;
int32_t x13225 = x13221;
int32_t x13226 = x13222;
for(int x13227=0; x13227 < 1; x13227++) {
int32_t x13228 = x13225;
int32_t x13229 = x13226;
int32_t x13230 = x13224;
int32_t x13231 = x13230;
int32_t x13232 = x13228;
int32_t x13233 = x13229;
for(int x13234=0; x13234 < -7; x13234++) {
int32_t x13235 = x13232;
float x13236 = x13141[x13235];
int32_t x13237 = x13233;
float x13238 = x12230[x13237];
float x13239 = x13236 + x13238;
x13141[x13235] = x13239;
x13231 += 1;
if (x520) {
x13232 += 1;
} else {
}
if (x520) {
x13233 += 1;
} else {
}

}
x13224 += -7;
if (x452) {
x13225 += -7;
} else {
}
if (x452) {
x13226 += -7;
} else {
}

}
x13217 += -7;
if (x542) {
x13218 += -7;
} else {
}
if (x542) {
x13219 += -7;
} else {
}

}
x13210 += 7;
if (x452) {
x13211 += 7;
} else {
}
if (x452) {
x13212 += 7;
} else {
}

}
float* x13277 = (float*)myMalloc(7 * sizeof(float));;
for(int x13278=0; x13278 < 7; x13278++) {
float x13279 = x13141[x13278];
bool x13280 = x13279 < 0.0f;
if (x13280) {
x13277[x13278] = 0.0f;
} else {
float x13283 = x13141[x13278];
x13277[x13278] = x13283;
}

}
float* x13289 = (float*)myMalloc(x1518 * sizeof(float));;
float* x13290 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x13291=0; x13291 < 1; x13291++) {
int32_t x13292 = x13291 * 7;
float* x13293 = x13277+x13292;
int32_t x13294 = x13291 * x1516;
float* x13295 = x13289+x13294;
int32_t x13296 = x13291 * x1520;
float* x13297 = x13290+x13296;
for(int x13298=0; x13298 < -1; x13298++) {
int32_t x13299 = x13298 / 1;
int32_t x13303 = x13299 * x1150;
int32_t x13304 = x13303 * x1152;
int32_t x13300 = x13298 % 1;
int32_t x13301 = x13300 / 1;
int32_t x13305 = x13301 * x1150;
int32_t x13306 = x13305 * x1152;
int32_t x13307 = x13304 + x13306;
int32_t x13302 = x13300 % 1;
int32_t x13308 = x13302 * x1152;
int32_t x13309 = x13308 * x1152;
int32_t x13310 = x13307 + x13309;
float* x13311 = x13297+x13310;
int32_t x13312 = x13299 * -7;
float* x13313 = x13293+x13312;
for(int x13314=0; x13314 < x1150; x13314++) {
int32_t x13316 = x13314 * x1152;
float* x13317 = x13311+x13316;
int32_t x13315 = x13314 + x13301;
int32_t x13318 = x13315 * -7;
int32_t x13319 = x13318 + x13302;
float* x13320 = x13313+x13319;
memcpy(x13317, x13320, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1153,-1,1,x22,-1,x13297,x1153,1,x13295,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x13333 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x13334 = 0;
int32_t x13335 = 0;
int32_t x13336 = 0;
for(int x13337=0; x13337 < -7; x13337++) {
int32_t x13338 = x13334;
int32_t x13339 = x13335;
float x13340 = x13289[x13339];
int32_t x13341 = x13336;
float x13342 = x254[x13341];
float x13343 = x13340 - x13342;
x13333[x13338] = x13343;
x13334 += 1;
if (x452) {
x13335 += x1516;
} else {
}
if (x452) {
x13336 += -1;
} else {
}

}
float* x13354 = (float*)myMalloc(256 * sizeof(float));;
for(int x13355=0; x13355 < 256; x13355++) {
float x13356 = x69[x13355];
float x13357 = x13356 + 1.0E-5f;
x13354[x13355] = x13357;

}
float* x13361 = (float*)myMalloc(256 * sizeof(float));;
for(int x13362=0; x13362 < 256; x13362++) {
float x13363 = x13354[x13362];
double x13364 = (double)x13363;
double x13365 = sqrt(x13364);
float x13366 = (float)x13365;
x13361[x13362] = x13366;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x13374 = (float*)myMalloc(7 * sizeof(float));;
int32_t x13375 = 0;
int32_t x13376 = 0;
int32_t x13377 = 0;
for(int x13378=0; x13378 < 1; x13378++) {
int32_t x13379 = x13376;
int32_t x13380 = x13377;
int32_t x13381 = x13375;
int32_t x13382 = x13381;
int32_t x13383 = x13379;
int32_t x13384 = x13380;
for(int x13385=0; x13385 < -1; x13385++) {
int32_t x13386 = x13383;
int32_t x13387 = x13384;
int32_t x13388 = x13382;
int32_t x13389 = x13388;
int32_t x13390 = x13386;
int32_t x13391 = x13387;
for(int x13392=0; x13392 < 1; x13392++) {
int32_t x13393 = x13390;
int32_t x13394 = x13391;
int32_t x13395 = x13389;
int32_t x13396 = x13395;
int32_t x13397 = x13393;
int32_t x13398 = x13394;
for(int x13399=0; x13399 < -7; x13399++) {
int32_t x13400 = x13396;
int32_t x13401 = x13397;
float x13402 = x13333[x13401];
int32_t x13403 = x13398;
float x13404 = x13361[x13403];
float x13405 = x13402 / x13404;
x13374[x13400] = x13405;
x13396 += 1;
if (x520) {
x13397 += 1;
} else {
}
if (x452) {
x13398 += 1;
} else {
}

}
x13389 += -7;
if (x452) {
x13390 += -7;
} else {
}
if (x452) {
x13391 += 1;
} else {
}

}
x13382 += -7;
if (x452) {
x13383 += -7;
} else {
}
if (x542) {
x13384 += 1;
} else {
}

}
x13375 += 7;
if (x452) {
x13376 += -7;
} else {
}
if (x452) {
x13377 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x13447 = (float*)myMalloc(7 * sizeof(float));;
int32_t x13448 = 0;
int32_t x13449 = 0;
int32_t x13450 = 0;
for(int x13451=0; x13451 < 1; x13451++) {
int32_t x13452 = x13449;
int32_t x13453 = x13450;
int32_t x13454 = x13448;
int32_t x13455 = x13454;
int32_t x13456 = x13452;
int32_t x13457 = x13453;
for(int x13458=0; x13458 < -1; x13458++) {
int32_t x13459 = x13456;
int32_t x13460 = x13457;
int32_t x13461 = x13455;
int32_t x13462 = x13461;
int32_t x13463 = x13459;
int32_t x13464 = x13460;
for(int x13465=0; x13465 < 1; x13465++) {
int32_t x13466 = x13463;
int32_t x13467 = x13464;
int32_t x13468 = x13462;
int32_t x13469 = x13468;
int32_t x13470 = x13466;
int32_t x13471 = x13467;
for(int x13472=0; x13472 < -7; x13472++) {
int32_t x13473 = x13469;
int32_t x13474 = x13470;
float x13475 = x13374[x13474];
int32_t x13476 = x13471;
float x13477 = x77[x13476];
float x13478 = x13475 * x13477;
x13447[x13473] = x13478;
x13469 += 1;
if (x520) {
x13470 += 1;
} else {
}
if (x452) {
x13471 += 1;
} else {
}

}
x13462 += -7;
if (x452) {
x13463 += -7;
} else {
}
if (x452) {
x13464 += 1;
} else {
}

}
x13455 += -7;
if (x542) {
x13456 += -7;
} else {
}
if (x542) {
x13457 += 1;
} else {
}

}
x13448 += 7;
if (x452) {
x13449 += 7;
} else {
}
if (x452) {
x13450 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x13520 = (float*)myMalloc(7 * sizeof(float));;
int32_t x13521 = 0;
int32_t x13522 = 0;
int32_t x13523 = 0;
for(int x13524=0; x13524 < 1; x13524++) {
int32_t x13525 = x13522;
int32_t x13526 = x13523;
int32_t x13527 = x13521;
int32_t x13528 = x13527;
int32_t x13529 = x13525;
int32_t x13530 = x13526;
for(int x13531=0; x13531 < -1; x13531++) {
int32_t x13532 = x13529;
int32_t x13533 = x13530;
int32_t x13534 = x13528;
int32_t x13535 = x13534;
int32_t x13536 = x13532;
int32_t x13537 = x13533;
for(int x13538=0; x13538 < 1; x13538++) {
int32_t x13539 = x13536;
int32_t x13540 = x13537;
int32_t x13541 = x13535;
int32_t x13542 = x13541;
int32_t x13543 = x13539;
int32_t x13544 = x13540;
for(int x13545=0; x13545 < -7; x13545++) {
int32_t x13546 = x13542;
int32_t x13547 = x13543;
float x13548 = x13447[x13547];
int32_t x13549 = x13544;
float x13550 = x185[x13549];
float x13551 = x13548 + x13550;
x13520[x13546] = x13551;
x13542 += 1;
if (x520) {
x13543 += 1;
} else {
}
if (x452) {
x13544 += 1;
} else {
}

}
x13535 += -7;
if (x452) {
x13536 += -7;
} else {
}
if (x452) {
x13537 += 1;
} else {
}

}
x13528 += -7;
if (x542) {
x13529 += -7;
} else {
}
if (x542) {
x13530 += 1;
} else {
}

}
x13521 += 7;
if (x452) {
x13522 += 7;
} else {
}
if (x452) {
x13523 += -1;
} else {
}

}
float* x13589 = (float*)myMalloc(7 * sizeof(float));;
for(int x13590=0; x13590 < 7; x13590++) {
float x13591 = x13520[x13590];
bool x13592 = x13591 < 0.0f;
if (x13592) {
x13589[x13590] = 0.0f;
} else {
float x13595 = x13520[x13590];
x13589[x13590] = x13595;
}

}
float* x13601 = (float*)myMalloc(x1518 * sizeof(float));;
float* x13602 = (float*)myMalloc(x1158 * sizeof(float));;
for(int x13603=0; x13603 < 1; x13603++) {
int32_t x13604 = x13603 * 7;
float* x13605 = x13589+x13604;
int32_t x13606 = x13603 * x1516;
float* x13607 = x13601+x13606;
int32_t x13608 = x13603 * x1158;
float* x13609 = x13602+x13608;
for(int x13610=0; x13610 < -9; x13610++) {
int32_t x13611 = x13610 / 9;
int32_t x13615 = x13611 * 3;
int32_t x13616 = x13615 * 3;
int32_t x13617 = x13616 * x1150;
int32_t x13618 = x13617 * x1152;
int32_t x13612 = x13610 % 9;
int32_t x13613 = x13612 / 3;
int32_t x13619 = x13613 * 3;
int32_t x13620 = x13619 * x1150;
int32_t x13621 = x13620 * x1152;
int32_t x13622 = x13618 + x13621;
int32_t x13614 = x13612 % 3;
int32_t x13623 = x13614 * x1152;
int32_t x13624 = x13623 * x1152;
int32_t x13625 = x13622 + x13624;
float* x13626 = x13609+x13625;
int32_t x13627 = x13611 * -7;
float* x13628 = x13605+x13627;
int32_t x13640 = 1 - x13614;
bool x13641 = x13640 > 0;
int32_t x13642;
if (x13641) {
x13642 = x13640;
} else {
x13642 = 0;
}
int32_t x13643 = 3 - x13614;
int32_t x13644 = x13643 - 1;
int32_t x13645 = 1 - x13644;
bool x13646 = x13645 > 0;
int32_t x13647;
if (x13646) {
x13647 = x13645;
} else {
x13647 = 0;
}
int32_t x13648 = x1152 - x13647;
int32_t x13649 = x13648 - x13642;
bool x13650 = x13649 <= 0;
bool x13654 = x13642 > 0;
int32_t x13639 = -1 + x13614;
bool x13667 = x13647 > 0;
for(int x13629=0; x13629 < x1150; x13629++) {
int32_t x13630 = x13629 - 1;
int32_t x13631 = x13630 + x13613;
bool x13632 = x13631 < 0;
bool x13633 = x13631 >= 1;
bool x13634 = x13632 || x13633;
if (x13634) {
int32_t x13635 = x13629 * x1152;
float* x13636 = x13626+x13635;
memset(x13636, 0, 4 * x1152);;
} else {
if (x13650) {
int32_t x13635 = x13629 * x1152;
float* x13651 = x13626+x13635;
memset(x13651, 0, 4 * x1152);;
} else {
int32_t x13635 = x13629 * x1152;
if (x13654) {
float* x13655 = x13626+x13635;
memset(x13655, 0, 4 * x13642);;
} else {
}
// may have segfault here
int32_t x13660 = x13635 + x13642;
float* x13661 = x13626+x13660;
int32_t x13662 = x13631 * -7;
int32_t x13663 = x13662 + x13639;
int32_t x13664 = x13663 + x13642;
float* x13665 = x13628+x13664;
memcpy(x13661, x13665, 4 * x13649);;
if (x13667) {
int32_t x13668 = x13635 + x1152;
int32_t x13669 = x13668 - x13647;
float* x13670 = x13626+x13669;
memset(x13670, 0, 4 * x13647);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1153,-9,1,x262,-9,x13609,x1153,1,x13607,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x13689 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x13690 = 0;
int32_t x13691 = 0;
int32_t x13692 = 0;
for(int x13693=0; x13693 < -7; x13693++) {
int32_t x13694 = x13690;
int32_t x13695 = x13691;
float x13696 = x13601[x13695];
int32_t x13697 = x13692;
float x13698 = x250[x13697];
float x13699 = x13696 - x13698;
x13689[x13694] = x13699;
x13690 += 1;
if (x452) {
x13691 += x1516;
} else {
}
if (x452) {
x13692 += -1;
} else {
}

}
float* x13710 = (float*)myMalloc(256 * sizeof(float));;
for(int x13711=0; x13711 < 256; x13711++) {
float x13712 = x104[x13711];
float x13713 = x13712 + 1.0E-5f;
x13710[x13711] = x13713;

}
float* x13717 = (float*)myMalloc(256 * sizeof(float));;
for(int x13718=0; x13718 < 256; x13718++) {
float x13719 = x13710[x13718];
double x13720 = (double)x13719;
double x13721 = sqrt(x13720);
float x13722 = (float)x13721;
x13717[x13718] = x13722;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x13730 = (float*)myMalloc(7 * sizeof(float));;
int32_t x13731 = 0;
int32_t x13732 = 0;
int32_t x13733 = 0;
for(int x13734=0; x13734 < 1; x13734++) {
int32_t x13735 = x13732;
int32_t x13736 = x13733;
int32_t x13737 = x13731;
int32_t x13738 = x13737;
int32_t x13739 = x13735;
int32_t x13740 = x13736;
for(int x13741=0; x13741 < -1; x13741++) {
int32_t x13742 = x13739;
int32_t x13743 = x13740;
int32_t x13744 = x13738;
int32_t x13745 = x13744;
int32_t x13746 = x13742;
int32_t x13747 = x13743;
for(int x13748=0; x13748 < 1; x13748++) {
int32_t x13749 = x13746;
int32_t x13750 = x13747;
int32_t x13751 = x13745;
int32_t x13752 = x13751;
int32_t x13753 = x13749;
int32_t x13754 = x13750;
for(int x13755=0; x13755 < -7; x13755++) {
int32_t x13756 = x13752;
int32_t x13757 = x13753;
float x13758 = x13689[x13757];
int32_t x13759 = x13754;
float x13760 = x13717[x13759];
float x13761 = x13758 / x13760;
x13730[x13756] = x13761;
x13752 += 1;
if (x520) {
x13753 += 1;
} else {
}
if (x452) {
x13754 += 1;
} else {
}

}
x13745 += -7;
if (x452) {
x13746 += -7;
} else {
}
if (x452) {
x13747 += 1;
} else {
}

}
x13738 += -7;
if (x452) {
x13739 += -7;
} else {
}
if (x542) {
x13740 += 1;
} else {
}

}
x13731 += 7;
if (x452) {
x13732 += -7;
} else {
}
if (x452) {
x13733 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x13803 = (float*)myMalloc(7 * sizeof(float));;
int32_t x13804 = 0;
int32_t x13805 = 0;
int32_t x13806 = 0;
for(int x13807=0; x13807 < 1; x13807++) {
int32_t x13808 = x13805;
int32_t x13809 = x13806;
int32_t x13810 = x13804;
int32_t x13811 = x13810;
int32_t x13812 = x13808;
int32_t x13813 = x13809;
for(int x13814=0; x13814 < -1; x13814++) {
int32_t x13815 = x13812;
int32_t x13816 = x13813;
int32_t x13817 = x13811;
int32_t x13818 = x13817;
int32_t x13819 = x13815;
int32_t x13820 = x13816;
for(int x13821=0; x13821 < 1; x13821++) {
int32_t x13822 = x13819;
int32_t x13823 = x13820;
int32_t x13824 = x13818;
int32_t x13825 = x13824;
int32_t x13826 = x13822;
int32_t x13827 = x13823;
for(int x13828=0; x13828 < -7; x13828++) {
int32_t x13829 = x13825;
int32_t x13830 = x13826;
float x13831 = x13730[x13830];
int32_t x13832 = x13827;
float x13833 = x168[x13832];
float x13834 = x13831 * x13833;
x13803[x13829] = x13834;
x13825 += 1;
if (x520) {
x13826 += 1;
} else {
}
if (x452) {
x13827 += 1;
} else {
}

}
x13818 += -7;
if (x452) {
x13819 += -7;
} else {
}
if (x452) {
x13820 += 1;
} else {
}

}
x13811 += -7;
if (x542) {
x13812 += -7;
} else {
}
if (x542) {
x13813 += 1;
} else {
}

}
x13804 += 7;
if (x452) {
x13805 += 7;
} else {
}
if (x452) {
x13806 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x13876 = (float*)myMalloc(7 * sizeof(float));;
int32_t x13877 = 0;
int32_t x13878 = 0;
int32_t x13879 = 0;
for(int x13880=0; x13880 < 1; x13880++) {
int32_t x13881 = x13878;
int32_t x13882 = x13879;
int32_t x13883 = x13877;
int32_t x13884 = x13883;
int32_t x13885 = x13881;
int32_t x13886 = x13882;
for(int x13887=0; x13887 < -1; x13887++) {
int32_t x13888 = x13885;
int32_t x13889 = x13886;
int32_t x13890 = x13884;
int32_t x13891 = x13890;
int32_t x13892 = x13888;
int32_t x13893 = x13889;
for(int x13894=0; x13894 < 1; x13894++) {
int32_t x13895 = x13892;
int32_t x13896 = x13893;
int32_t x13897 = x13891;
int32_t x13898 = x13897;
int32_t x13899 = x13895;
int32_t x13900 = x13896;
for(int x13901=0; x13901 < -7; x13901++) {
int32_t x13902 = x13898;
int32_t x13903 = x13899;
float x13904 = x13803[x13903];
int32_t x13905 = x13900;
float x13906 = x109[x13905];
float x13907 = x13904 + x13906;
x13876[x13902] = x13907;
x13898 += 1;
if (x520) {
x13899 += 1;
} else {
}
if (x452) {
x13900 += 1;
} else {
}

}
x13891 += -7;
if (x452) {
x13892 += -7;
} else {
}
if (x452) {
x13893 += 1;
} else {
}

}
x13884 += -7;
if (x542) {
x13885 += -7;
} else {
}
if (x542) {
x13886 += 1;
} else {
}

}
x13877 += 7;
if (x452) {
x13878 += 7;
} else {
}
if (x452) {
x13879 += -1;
} else {
}

}
float* x13945 = (float*)myMalloc(7 * sizeof(float));;
for(int x13946=0; x13946 < 7; x13946++) {
float x13947 = x13876[x13946];
bool x13948 = x13947 < 0.0f;
if (x13948) {
x13945[x13946] = 0.0f;
} else {
float x13951 = x13876[x13946];
x13945[x13946] = x13951;
}

}
float* x13957 = (float*)myMalloc(x9457 * sizeof(float));;
float* x13958 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x13959=0; x13959 < 1; x13959++) {
int32_t x13960 = x13959 * 7;
float* x13961 = x13945+x13960;
int32_t x13962 = x13959 * x9455;
float* x13963 = x13957+x13962;
int32_t x13964 = x13959 * x1520;
float* x13965 = x13958+x13964;
for(int x13966=0; x13966 < -1; x13966++) {
int32_t x13967 = x13966 / 1;
int32_t x13971 = x13967 * x1150;
int32_t x13972 = x13971 * x1152;
int32_t x13968 = x13966 % 1;
int32_t x13969 = x13968 / 1;
int32_t x13973 = x13969 * x1150;
int32_t x13974 = x13973 * x1152;
int32_t x13975 = x13972 + x13974;
int32_t x13970 = x13968 % 1;
int32_t x13976 = x13970 * x1152;
int32_t x13977 = x13976 * x1152;
int32_t x13978 = x13975 + x13977;
float* x13979 = x13965+x13978;
int32_t x13980 = x13967 * -7;
float* x13981 = x13961+x13980;
for(int x13982=0; x13982 < x1150; x13982++) {
int32_t x13984 = x13982 * x1152;
float* x13985 = x13979+x13984;
int32_t x13983 = x13982 + x13969;
int32_t x13986 = x13983 * -7;
int32_t x13987 = x13986 + x13970;
float* x13988 = x13981+x13987;
memcpy(x13985, x13988, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x1153,-1,1,x221,-1,x13965,x1153,1,x13963,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(1024) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x14001 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x14002 = 0;
int32_t x14003 = 0;
int32_t x14004 = 0;
for(int x14005=0; x14005 < -7; x14005++) {
int32_t x14006 = x14002;
int32_t x14007 = x14003;
float x14008 = x13957[x14007];
int32_t x14009 = x14004;
float x14010 = x209[x14009];
float x14011 = x14008 - x14010;
x14001[x14006] = x14011;
x14002 += 1;
if (x452) {
x14003 += x9455;
} else {
}
if (x452) {
x14004 += -1;
} else {
}

}
float* x14022 = (float*)myMalloc(1024 * sizeof(float));;
for(int x14023=0; x14023 < 1024; x14023++) {
float x14024 = x272[x14023];
float x14025 = x14024 + 1.0E-5f;
x14022[x14023] = x14025;

}
float* x14029 = (float*)myMalloc(1024 * sizeof(float));;
for(int x14030=0; x14030 < 1024; x14030++) {
float x14031 = x14022[x14030];
double x14032 = (double)x14031;
double x14033 = sqrt(x14032);
float x14034 = (float)x14033;
x14029[x14030] = x14034;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x14042 = (float*)myMalloc(7 * sizeof(float));;
int32_t x14043 = 0;
int32_t x14044 = 0;
int32_t x14045 = 0;
for(int x14046=0; x14046 < 1; x14046++) {
int32_t x14047 = x14044;
int32_t x14048 = x14045;
int32_t x14049 = x14043;
int32_t x14050 = x14049;
int32_t x14051 = x14047;
int32_t x14052 = x14048;
for(int x14053=0; x14053 < -1; x14053++) {
int32_t x14054 = x14051;
int32_t x14055 = x14052;
int32_t x14056 = x14050;
int32_t x14057 = x14056;
int32_t x14058 = x14054;
int32_t x14059 = x14055;
for(int x14060=0; x14060 < 1; x14060++) {
int32_t x14061 = x14058;
int32_t x14062 = x14059;
int32_t x14063 = x14057;
int32_t x14064 = x14063;
int32_t x14065 = x14061;
int32_t x14066 = x14062;
for(int x14067=0; x14067 < -7; x14067++) {
int32_t x14068 = x14064;
int32_t x14069 = x14065;
float x14070 = x14001[x14069];
int32_t x14071 = x14066;
float x14072 = x14029[x14071];
float x14073 = x14070 / x14072;
x14042[x14068] = x14073;
x14064 += 1;
if (x520) {
x14065 += 1;
} else {
}
if (x452) {
x14066 += 1;
} else {
}

}
x14057 += -7;
if (x452) {
x14058 += -7;
} else {
}
if (x452) {
x14059 += 1;
} else {
}

}
x14050 += -7;
if (x452) {
x14051 += -7;
} else {
}
if (x542) {
x14052 += 1;
} else {
}

}
x14043 += 7;
if (x452) {
x14044 += -7;
} else {
}
if (x452) {
x14045 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x14115 = (float*)myMalloc(7 * sizeof(float));;
int32_t x14116 = 0;
int32_t x14117 = 0;
int32_t x14118 = 0;
for(int x14119=0; x14119 < 1; x14119++) {
int32_t x14120 = x14117;
int32_t x14121 = x14118;
int32_t x14122 = x14116;
int32_t x14123 = x14122;
int32_t x14124 = x14120;
int32_t x14125 = x14121;
for(int x14126=0; x14126 < -1; x14126++) {
int32_t x14127 = x14124;
int32_t x14128 = x14125;
int32_t x14129 = x14123;
int32_t x14130 = x14129;
int32_t x14131 = x14127;
int32_t x14132 = x14128;
for(int x14133=0; x14133 < 1; x14133++) {
int32_t x14134 = x14131;
int32_t x14135 = x14132;
int32_t x14136 = x14130;
int32_t x14137 = x14136;
int32_t x14138 = x14134;
int32_t x14139 = x14135;
for(int x14140=0; x14140 < -7; x14140++) {
int32_t x14141 = x14137;
int32_t x14142 = x14138;
float x14143 = x14042[x14142];
int32_t x14144 = x14139;
float x14145 = x59[x14144];
float x14146 = x14143 * x14145;
x14115[x14141] = x14146;
x14137 += 1;
if (x520) {
x14138 += 1;
} else {
}
if (x452) {
x14139 += 1;
} else {
}

}
x14130 += -7;
if (x452) {
x14131 += -7;
} else {
}
if (x452) {
x14132 += 1;
} else {
}

}
x14123 += -7;
if (x542) {
x14124 += -7;
} else {
}
if (x542) {
x14125 += 1;
} else {
}

}
x14116 += 7;
if (x452) {
x14117 += 7;
} else {
}
if (x452) {
x14118 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x14188 = (float*)myMalloc(7 * sizeof(float));;
int32_t x14189 = 0;
int32_t x14190 = 0;
int32_t x14191 = 0;
for(int x14192=0; x14192 < 1; x14192++) {
int32_t x14193 = x14190;
int32_t x14194 = x14191;
int32_t x14195 = x14189;
int32_t x14196 = x14195;
int32_t x14197 = x14193;
int32_t x14198 = x14194;
for(int x14199=0; x14199 < -1; x14199++) {
int32_t x14200 = x14197;
int32_t x14201 = x14198;
int32_t x14202 = x14196;
int32_t x14203 = x14202;
int32_t x14204 = x14200;
int32_t x14205 = x14201;
for(int x14206=0; x14206 < 1; x14206++) {
int32_t x14207 = x14204;
int32_t x14208 = x14205;
int32_t x14209 = x14203;
int32_t x14210 = x14209;
int32_t x14211 = x14207;
int32_t x14212 = x14208;
for(int x14213=0; x14213 < -7; x14213++) {
int32_t x14214 = x14210;
int32_t x14215 = x14211;
float x14216 = x14115[x14215];
int32_t x14217 = x14212;
float x14218 = x120[x14217];
float x14219 = x14216 + x14218;
x14188[x14214] = x14219;
x14210 += 1;
if (x520) {
x14211 += 1;
} else {
}
if (x452) {
x14212 += 1;
} else {
}

}
x14203 += -7;
if (x452) {
x14204 += -7;
} else {
}
if (x452) {
x14205 += 1;
} else {
}

}
x14196 += -7;
if (x542) {
x14197 += -7;
} else {
}
if (x542) {
x14198 += 1;
} else {
}

}
x14189 += 7;
if (x452) {
x14190 += 7;
} else {
}
if (x452) {
x14191 += -1;
} else {
}

}
int32_t x14257 = 0;
int32_t x14258 = 0;
int32_t x14259 = 0;
for(int x14260=0; x14260 < 1; x14260++) {
int32_t x14261 = x14258;
int32_t x14262 = x14259;
int32_t x14263 = x14257;
int32_t x14264 = x14263;
int32_t x14265 = x14261;
int32_t x14266 = x14262;
for(int x14267=0; x14267 < -1; x14267++) {
int32_t x14268 = x14265;
int32_t x14269 = x14266;
int32_t x14270 = x14264;
int32_t x14271 = x14270;
int32_t x14272 = x14268;
int32_t x14273 = x14269;
for(int x14274=0; x14274 < 1; x14274++) {
int32_t x14275 = x14272;
int32_t x14276 = x14273;
int32_t x14277 = x14271;
int32_t x14278 = x14277;
int32_t x14279 = x14275;
int32_t x14280 = x14276;
for(int x14281=0; x14281 < -7; x14281++) {
int32_t x14282 = x14279;
float x14283 = x14188[x14282];
int32_t x14284 = x14280;
float x14285 = x13277[x14284];
float x14286 = x14283 + x14285;
x14188[x14282] = x14286;
x14278 += 1;
if (x520) {
x14279 += 1;
} else {
}
if (x520) {
x14280 += 1;
} else {
}

}
x14271 += -7;
if (x452) {
x14272 += -7;
} else {
}
if (x452) {
x14273 += -7;
} else {
}

}
x14264 += -7;
if (x542) {
x14265 += -7;
} else {
}
if (x542) {
x14266 += -7;
} else {
}

}
x14257 += 7;
if (x452) {
x14258 += 7;
} else {
}
if (x452) {
x14259 += 7;
} else {
}

}
float* x14324 = (float*)myMalloc(7 * sizeof(float));;
for(int x14325=0; x14325 < 7; x14325++) {
float x14326 = x14188[x14325];
bool x14327 = x14326 < 0.0f;
if (x14327) {
x14324[x14325] = 0.0f;
} else {
float x14330 = x14188[x14325];
x14324[x14325] = x14330;
}

}
float* x14336 = (float*)myMalloc(x1518 * sizeof(float));;
float* x14337 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x14338=0; x14338 < 1; x14338++) {
int32_t x14339 = x14338 * 7;
float* x14340 = x14324+x14339;
int32_t x14341 = x14338 * x1516;
float* x14342 = x14336+x14341;
int32_t x14343 = x14338 * x1520;
float* x14344 = x14337+x14343;
for(int x14345=0; x14345 < -1; x14345++) {
int32_t x14346 = x14345 / 1;
int32_t x14350 = x14346 * x1150;
int32_t x14351 = x14350 * x1152;
int32_t x14347 = x14345 % 1;
int32_t x14348 = x14347 / 1;
int32_t x14352 = x14348 * x1150;
int32_t x14353 = x14352 * x1152;
int32_t x14354 = x14351 + x14353;
int32_t x14349 = x14347 % 1;
int32_t x14355 = x14349 * x1152;
int32_t x14356 = x14355 * x1152;
int32_t x14357 = x14354 + x14356;
float* x14358 = x14344+x14357;
int32_t x14359 = x14346 * -7;
float* x14360 = x14340+x14359;
for(int x14361=0; x14361 < x1150; x14361++) {
int32_t x14363 = x14361 * x1152;
float* x14364 = x14358+x14363;
int32_t x14362 = x14361 + x14348;
int32_t x14365 = x14362 * -7;
int32_t x14366 = x14365 + x14349;
float* x14367 = x14360+x14366;
memcpy(x14364, x14367, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1153,-1,1,x151,-1,x14344,x1153,1,x14342,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x14380 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x14381 = 0;
int32_t x14382 = 0;
int32_t x14383 = 0;
for(int x14384=0; x14384 < -7; x14384++) {
int32_t x14385 = x14381;
int32_t x14386 = x14382;
float x14387 = x14336[x14386];
int32_t x14388 = x14383;
float x14389 = x80[x14388];
float x14390 = x14387 - x14389;
x14380[x14385] = x14390;
x14381 += 1;
if (x452) {
x14382 += x1516;
} else {
}
if (x452) {
x14383 += -1;
} else {
}

}
float* x14401 = (float*)myMalloc(256 * sizeof(float));;
for(int x14402=0; x14402 < 256; x14402++) {
float x14403 = x176[x14402];
float x14404 = x14403 + 1.0E-5f;
x14401[x14402] = x14404;

}
float* x14408 = (float*)myMalloc(256 * sizeof(float));;
for(int x14409=0; x14409 < 256; x14409++) {
float x14410 = x14401[x14409];
double x14411 = (double)x14410;
double x14412 = sqrt(x14411);
float x14413 = (float)x14412;
x14408[x14409] = x14413;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x14421 = (float*)myMalloc(7 * sizeof(float));;
int32_t x14422 = 0;
int32_t x14423 = 0;
int32_t x14424 = 0;
for(int x14425=0; x14425 < 1; x14425++) {
int32_t x14426 = x14423;
int32_t x14427 = x14424;
int32_t x14428 = x14422;
int32_t x14429 = x14428;
int32_t x14430 = x14426;
int32_t x14431 = x14427;
for(int x14432=0; x14432 < -1; x14432++) {
int32_t x14433 = x14430;
int32_t x14434 = x14431;
int32_t x14435 = x14429;
int32_t x14436 = x14435;
int32_t x14437 = x14433;
int32_t x14438 = x14434;
for(int x14439=0; x14439 < 1; x14439++) {
int32_t x14440 = x14437;
int32_t x14441 = x14438;
int32_t x14442 = x14436;
int32_t x14443 = x14442;
int32_t x14444 = x14440;
int32_t x14445 = x14441;
for(int x14446=0; x14446 < -7; x14446++) {
int32_t x14447 = x14443;
int32_t x14448 = x14444;
float x14449 = x14380[x14448];
int32_t x14450 = x14445;
float x14451 = x14408[x14450];
float x14452 = x14449 / x14451;
x14421[x14447] = x14452;
x14443 += 1;
if (x520) {
x14444 += 1;
} else {
}
if (x452) {
x14445 += 1;
} else {
}

}
x14436 += -7;
if (x452) {
x14437 += -7;
} else {
}
if (x452) {
x14438 += 1;
} else {
}

}
x14429 += -7;
if (x452) {
x14430 += -7;
} else {
}
if (x542) {
x14431 += 1;
} else {
}

}
x14422 += 7;
if (x452) {
x14423 += -7;
} else {
}
if (x452) {
x14424 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x14494 = (float*)myMalloc(7 * sizeof(float));;
int32_t x14495 = 0;
int32_t x14496 = 0;
int32_t x14497 = 0;
for(int x14498=0; x14498 < 1; x14498++) {
int32_t x14499 = x14496;
int32_t x14500 = x14497;
int32_t x14501 = x14495;
int32_t x14502 = x14501;
int32_t x14503 = x14499;
int32_t x14504 = x14500;
for(int x14505=0; x14505 < -1; x14505++) {
int32_t x14506 = x14503;
int32_t x14507 = x14504;
int32_t x14508 = x14502;
int32_t x14509 = x14508;
int32_t x14510 = x14506;
int32_t x14511 = x14507;
for(int x14512=0; x14512 < 1; x14512++) {
int32_t x14513 = x14510;
int32_t x14514 = x14511;
int32_t x14515 = x14509;
int32_t x14516 = x14515;
int32_t x14517 = x14513;
int32_t x14518 = x14514;
for(int x14519=0; x14519 < -7; x14519++) {
int32_t x14520 = x14516;
int32_t x14521 = x14517;
float x14522 = x14421[x14521];
int32_t x14523 = x14518;
float x14524 = x85[x14523];
float x14525 = x14522 * x14524;
x14494[x14520] = x14525;
x14516 += 1;
if (x520) {
x14517 += 1;
} else {
}
if (x452) {
x14518 += 1;
} else {
}

}
x14509 += -7;
if (x452) {
x14510 += -7;
} else {
}
if (x452) {
x14511 += 1;
} else {
}

}
x14502 += -7;
if (x542) {
x14503 += -7;
} else {
}
if (x542) {
x14504 += 1;
} else {
}

}
x14495 += 7;
if (x452) {
x14496 += 7;
} else {
}
if (x452) {
x14497 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x14567 = (float*)myMalloc(7 * sizeof(float));;
int32_t x14568 = 0;
int32_t x14569 = 0;
int32_t x14570 = 0;
for(int x14571=0; x14571 < 1; x14571++) {
int32_t x14572 = x14569;
int32_t x14573 = x14570;
int32_t x14574 = x14568;
int32_t x14575 = x14574;
int32_t x14576 = x14572;
int32_t x14577 = x14573;
for(int x14578=0; x14578 < -1; x14578++) {
int32_t x14579 = x14576;
int32_t x14580 = x14577;
int32_t x14581 = x14575;
int32_t x14582 = x14581;
int32_t x14583 = x14579;
int32_t x14584 = x14580;
for(int x14585=0; x14585 < 1; x14585++) {
int32_t x14586 = x14583;
int32_t x14587 = x14584;
int32_t x14588 = x14582;
int32_t x14589 = x14588;
int32_t x14590 = x14586;
int32_t x14591 = x14587;
for(int x14592=0; x14592 < -7; x14592++) {
int32_t x14593 = x14589;
int32_t x14594 = x14590;
float x14595 = x14494[x14594];
int32_t x14596 = x14591;
float x14597 = x253[x14596];
float x14598 = x14595 + x14597;
x14567[x14593] = x14598;
x14589 += 1;
if (x520) {
x14590 += 1;
} else {
}
if (x452) {
x14591 += 1;
} else {
}

}
x14582 += -7;
if (x452) {
x14583 += -7;
} else {
}
if (x452) {
x14584 += 1;
} else {
}

}
x14575 += -7;
if (x542) {
x14576 += -7;
} else {
}
if (x542) {
x14577 += 1;
} else {
}

}
x14568 += 7;
if (x452) {
x14569 += 7;
} else {
}
if (x452) {
x14570 += -1;
} else {
}

}
float* x14636 = (float*)myMalloc(7 * sizeof(float));;
for(int x14637=0; x14637 < 7; x14637++) {
float x14638 = x14567[x14637];
bool x14639 = x14638 < 0.0f;
if (x14639) {
x14636[x14637] = 0.0f;
} else {
float x14642 = x14567[x14637];
x14636[x14637] = x14642;
}

}
float* x14648 = (float*)myMalloc(x1518 * sizeof(float));;
float* x14649 = (float*)myMalloc(x1158 * sizeof(float));;
for(int x14650=0; x14650 < 1; x14650++) {
int32_t x14651 = x14650 * 7;
float* x14652 = x14636+x14651;
int32_t x14653 = x14650 * x1516;
float* x14654 = x14648+x14653;
int32_t x14655 = x14650 * x1158;
float* x14656 = x14649+x14655;
for(int x14657=0; x14657 < -9; x14657++) {
int32_t x14658 = x14657 / 9;
int32_t x14662 = x14658 * 3;
int32_t x14663 = x14662 * 3;
int32_t x14664 = x14663 * x1150;
int32_t x14665 = x14664 * x1152;
int32_t x14659 = x14657 % 9;
int32_t x14660 = x14659 / 3;
int32_t x14666 = x14660 * 3;
int32_t x14667 = x14666 * x1150;
int32_t x14668 = x14667 * x1152;
int32_t x14669 = x14665 + x14668;
int32_t x14661 = x14659 % 3;
int32_t x14670 = x14661 * x1152;
int32_t x14671 = x14670 * x1152;
int32_t x14672 = x14669 + x14671;
float* x14673 = x14656+x14672;
int32_t x14674 = x14658 * -7;
float* x14675 = x14652+x14674;
int32_t x14687 = 1 - x14661;
bool x14688 = x14687 > 0;
int32_t x14689;
if (x14688) {
x14689 = x14687;
} else {
x14689 = 0;
}
int32_t x14690 = 3 - x14661;
int32_t x14691 = x14690 - 1;
int32_t x14692 = 1 - x14691;
bool x14693 = x14692 > 0;
int32_t x14694;
if (x14693) {
x14694 = x14692;
} else {
x14694 = 0;
}
int32_t x14695 = x1152 - x14694;
int32_t x14696 = x14695 - x14689;
bool x14697 = x14696 <= 0;
bool x14701 = x14689 > 0;
int32_t x14686 = -1 + x14661;
bool x14714 = x14694 > 0;
for(int x14676=0; x14676 < x1150; x14676++) {
int32_t x14677 = x14676 - 1;
int32_t x14678 = x14677 + x14660;
bool x14679 = x14678 < 0;
bool x14680 = x14678 >= 1;
bool x14681 = x14679 || x14680;
if (x14681) {
int32_t x14682 = x14676 * x1152;
float* x14683 = x14673+x14682;
memset(x14683, 0, 4 * x1152);;
} else {
if (x14697) {
int32_t x14682 = x14676 * x1152;
float* x14698 = x14673+x14682;
memset(x14698, 0, 4 * x1152);;
} else {
int32_t x14682 = x14676 * x1152;
if (x14701) {
float* x14702 = x14673+x14682;
memset(x14702, 0, 4 * x14689);;
} else {
}
// may have segfault here
int32_t x14707 = x14682 + x14689;
float* x14708 = x14673+x14707;
int32_t x14709 = x14678 * -7;
int32_t x14710 = x14709 + x14686;
int32_t x14711 = x14710 + x14689;
float* x14712 = x14675+x14711;
memcpy(x14708, x14712, 4 * x14696);;
if (x14714) {
int32_t x14715 = x14682 + x1152;
int32_t x14716 = x14715 - x14694;
float* x14717 = x14673+x14716;
memset(x14717, 0, 4 * x14694);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 256,x1153,-9,1,x226,-9,x14656,x1153,1,x14654,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(256) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x14736 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x14737 = 0;
int32_t x14738 = 0;
int32_t x14739 = 0;
for(int x14740=0; x14740 < -7; x14740++) {
int32_t x14741 = x14737;
int32_t x14742 = x14738;
float x14743 = x14648[x14742];
int32_t x14744 = x14739;
float x14745 = x70[x14744];
float x14746 = x14743 - x14745;
x14736[x14741] = x14746;
x14737 += 1;
if (x452) {
x14738 += x1516;
} else {
}
if (x452) {
x14739 += -1;
} else {
}

}
float* x14757 = (float*)myMalloc(256 * sizeof(float));;
for(int x14758=0; x14758 < 256; x14758++) {
float x14759 = x240[x14758];
float x14760 = x14759 + 1.0E-5f;
x14757[x14758] = x14760;

}
float* x14764 = (float*)myMalloc(256 * sizeof(float));;
for(int x14765=0; x14765 < 256; x14765++) {
float x14766 = x14757[x14765];
double x14767 = (double)x14766;
double x14768 = sqrt(x14767);
float x14769 = (float)x14768;
x14764[x14765] = x14769;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x14777 = (float*)myMalloc(7 * sizeof(float));;
int32_t x14778 = 0;
int32_t x14779 = 0;
int32_t x14780 = 0;
for(int x14781=0; x14781 < 1; x14781++) {
int32_t x14782 = x14779;
int32_t x14783 = x14780;
int32_t x14784 = x14778;
int32_t x14785 = x14784;
int32_t x14786 = x14782;
int32_t x14787 = x14783;
for(int x14788=0; x14788 < -1; x14788++) {
int32_t x14789 = x14786;
int32_t x14790 = x14787;
int32_t x14791 = x14785;
int32_t x14792 = x14791;
int32_t x14793 = x14789;
int32_t x14794 = x14790;
for(int x14795=0; x14795 < 1; x14795++) {
int32_t x14796 = x14793;
int32_t x14797 = x14794;
int32_t x14798 = x14792;
int32_t x14799 = x14798;
int32_t x14800 = x14796;
int32_t x14801 = x14797;
for(int x14802=0; x14802 < -7; x14802++) {
int32_t x14803 = x14799;
int32_t x14804 = x14800;
float x14805 = x14736[x14804];
int32_t x14806 = x14801;
float x14807 = x14764[x14806];
float x14808 = x14805 / x14807;
x14777[x14803] = x14808;
x14799 += 1;
if (x520) {
x14800 += 1;
} else {
}
if (x452) {
x14801 += 1;
} else {
}

}
x14792 += -7;
if (x452) {
x14793 += -7;
} else {
}
if (x452) {
x14794 += 1;
} else {
}

}
x14785 += -7;
if (x452) {
x14786 += -7;
} else {
}
if (x542) {
x14787 += 1;
} else {
}

}
x14778 += 7;
if (x452) {
x14779 += -7;
} else {
}
if (x452) {
x14780 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x14850 = (float*)myMalloc(7 * sizeof(float));;
int32_t x14851 = 0;
int32_t x14852 = 0;
int32_t x14853 = 0;
for(int x14854=0; x14854 < 1; x14854++) {
int32_t x14855 = x14852;
int32_t x14856 = x14853;
int32_t x14857 = x14851;
int32_t x14858 = x14857;
int32_t x14859 = x14855;
int32_t x14860 = x14856;
for(int x14861=0; x14861 < -1; x14861++) {
int32_t x14862 = x14859;
int32_t x14863 = x14860;
int32_t x14864 = x14858;
int32_t x14865 = x14864;
int32_t x14866 = x14862;
int32_t x14867 = x14863;
for(int x14868=0; x14868 < 1; x14868++) {
int32_t x14869 = x14866;
int32_t x14870 = x14867;
int32_t x14871 = x14865;
int32_t x14872 = x14871;
int32_t x14873 = x14869;
int32_t x14874 = x14870;
for(int x14875=0; x14875 < -7; x14875++) {
int32_t x14876 = x14872;
int32_t x14877 = x14873;
float x14878 = x14777[x14877];
int32_t x14879 = x14874;
float x14880 = x141[x14879];
float x14881 = x14878 * x14880;
x14850[x14876] = x14881;
x14872 += 1;
if (x520) {
x14873 += 1;
} else {
}
if (x452) {
x14874 += 1;
} else {
}

}
x14865 += -7;
if (x452) {
x14866 += -7;
} else {
}
if (x452) {
x14867 += 1;
} else {
}

}
x14858 += -7;
if (x542) {
x14859 += -7;
} else {
}
if (x542) {
x14860 += 1;
} else {
}

}
x14851 += 7;
if (x452) {
x14852 += 7;
} else {
}
if (x452) {
x14853 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x14923 = (float*)myMalloc(7 * sizeof(float));;
int32_t x14924 = 0;
int32_t x14925 = 0;
int32_t x14926 = 0;
for(int x14927=0; x14927 < 1; x14927++) {
int32_t x14928 = x14925;
int32_t x14929 = x14926;
int32_t x14930 = x14924;
int32_t x14931 = x14930;
int32_t x14932 = x14928;
int32_t x14933 = x14929;
for(int x14934=0; x14934 < -1; x14934++) {
int32_t x14935 = x14932;
int32_t x14936 = x14933;
int32_t x14937 = x14931;
int32_t x14938 = x14937;
int32_t x14939 = x14935;
int32_t x14940 = x14936;
for(int x14941=0; x14941 < 1; x14941++) {
int32_t x14942 = x14939;
int32_t x14943 = x14940;
int32_t x14944 = x14938;
int32_t x14945 = x14944;
int32_t x14946 = x14942;
int32_t x14947 = x14943;
for(int x14948=0; x14948 < -7; x14948++) {
int32_t x14949 = x14945;
int32_t x14950 = x14946;
float x14951 = x14850[x14950];
int32_t x14952 = x14947;
float x14953 = x189[x14952];
float x14954 = x14951 + x14953;
x14923[x14949] = x14954;
x14945 += 1;
if (x520) {
x14946 += 1;
} else {
}
if (x452) {
x14947 += 1;
} else {
}

}
x14938 += -7;
if (x452) {
x14939 += -7;
} else {
}
if (x452) {
x14940 += 1;
} else {
}

}
x14931 += -7;
if (x542) {
x14932 += -7;
} else {
}
if (x542) {
x14933 += 1;
} else {
}

}
x14924 += 7;
if (x452) {
x14925 += 7;
} else {
}
if (x452) {
x14926 += -1;
} else {
}

}
float* x14992 = (float*)myMalloc(7 * sizeof(float));;
for(int x14993=0; x14993 < 7; x14993++) {
float x14994 = x14923[x14993];
bool x14995 = x14994 < 0.0f;
if (x14995) {
x14992[x14993] = 0.0f;
} else {
float x14998 = x14923[x14993];
x14992[x14993] = x14998;
}

}
float* x15004 = (float*)myMalloc(x9457 * sizeof(float));;
float* x15005 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x15006=0; x15006 < 1; x15006++) {
int32_t x15007 = x15006 * 7;
float* x15008 = x14992+x15007;
int32_t x15009 = x15006 * x9455;
float* x15010 = x15004+x15009;
int32_t x15011 = x15006 * x1520;
float* x15012 = x15005+x15011;
for(int x15013=0; x15013 < -1; x15013++) {
int32_t x15014 = x15013 / 1;
int32_t x15018 = x15014 * x1150;
int32_t x15019 = x15018 * x1152;
int32_t x15015 = x15013 % 1;
int32_t x15016 = x15015 / 1;
int32_t x15020 = x15016 * x1150;
int32_t x15021 = x15020 * x1152;
int32_t x15022 = x15019 + x15021;
int32_t x15017 = x15015 % 1;
int32_t x15023 = x15017 * x1152;
int32_t x15024 = x15023 * x1152;
int32_t x15025 = x15022 + x15024;
float* x15026 = x15012+x15025;
int32_t x15027 = x15014 * -7;
float* x15028 = x15008+x15027;
for(int x15029=0; x15029 < x1150; x15029++) {
int32_t x15031 = x15029 * x1152;
float* x15032 = x15026+x15031;
int32_t x15030 = x15029 + x15016;
int32_t x15033 = x15030 * -7;
int32_t x15034 = x15033 + x15017;
float* x15035 = x15028+x15034;
memcpy(x15032, x15035, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1024,x1153,-1,1,x97,-1,x15012,x1153,1,x15010,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(1024) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x15048 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x15049 = 0;
int32_t x15050 = 0;
int32_t x15051 = 0;
for(int x15052=0; x15052 < -7; x15052++) {
int32_t x15053 = x15049;
int32_t x15054 = x15050;
float x15055 = x15004[x15054];
int32_t x15056 = x15051;
float x15057 = x122[x15056];
float x15058 = x15055 - x15057;
x15048[x15053] = x15058;
x15049 += 1;
if (x452) {
x15050 += x9455;
} else {
}
if (x452) {
x15051 += -1;
} else {
}

}
float* x15069 = (float*)myMalloc(1024 * sizeof(float));;
for(int x15070=0; x15070 < 1024; x15070++) {
float x15071 = x183[x15070];
float x15072 = x15071 + 1.0E-5f;
x15069[x15070] = x15072;

}
float* x15076 = (float*)myMalloc(1024 * sizeof(float));;
for(int x15077=0; x15077 < 1024; x15077++) {
float x15078 = x15069[x15077];
double x15079 = (double)x15078;
double x15080 = sqrt(x15079);
float x15081 = (float)x15080;
x15076[x15077] = x15081;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x15089 = (float*)myMalloc(7 * sizeof(float));;
int32_t x15090 = 0;
int32_t x15091 = 0;
int32_t x15092 = 0;
for(int x15093=0; x15093 < 1; x15093++) {
int32_t x15094 = x15091;
int32_t x15095 = x15092;
int32_t x15096 = x15090;
int32_t x15097 = x15096;
int32_t x15098 = x15094;
int32_t x15099 = x15095;
for(int x15100=0; x15100 < -1; x15100++) {
int32_t x15101 = x15098;
int32_t x15102 = x15099;
int32_t x15103 = x15097;
int32_t x15104 = x15103;
int32_t x15105 = x15101;
int32_t x15106 = x15102;
for(int x15107=0; x15107 < 1; x15107++) {
int32_t x15108 = x15105;
int32_t x15109 = x15106;
int32_t x15110 = x15104;
int32_t x15111 = x15110;
int32_t x15112 = x15108;
int32_t x15113 = x15109;
for(int x15114=0; x15114 < -7; x15114++) {
int32_t x15115 = x15111;
int32_t x15116 = x15112;
float x15117 = x15048[x15116];
int32_t x15118 = x15113;
float x15119 = x15076[x15118];
float x15120 = x15117 / x15119;
x15089[x15115] = x15120;
x15111 += 1;
if (x520) {
x15112 += 1;
} else {
}
if (x452) {
x15113 += 1;
} else {
}

}
x15104 += -7;
if (x452) {
x15105 += -7;
} else {
}
if (x452) {
x15106 += 1;
} else {
}

}
x15097 += -7;
if (x452) {
x15098 += -7;
} else {
}
if (x542) {
x15099 += 1;
} else {
}

}
x15090 += 7;
if (x452) {
x15091 += -7;
} else {
}
if (x452) {
x15092 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x15162 = (float*)myMalloc(7 * sizeof(float));;
int32_t x15163 = 0;
int32_t x15164 = 0;
int32_t x15165 = 0;
for(int x15166=0; x15166 < 1; x15166++) {
int32_t x15167 = x15164;
int32_t x15168 = x15165;
int32_t x15169 = x15163;
int32_t x15170 = x15169;
int32_t x15171 = x15167;
int32_t x15172 = x15168;
for(int x15173=0; x15173 < -1; x15173++) {
int32_t x15174 = x15171;
int32_t x15175 = x15172;
int32_t x15176 = x15170;
int32_t x15177 = x15176;
int32_t x15178 = x15174;
int32_t x15179 = x15175;
for(int x15180=0; x15180 < 1; x15180++) {
int32_t x15181 = x15178;
int32_t x15182 = x15179;
int32_t x15183 = x15177;
int32_t x15184 = x15183;
int32_t x15185 = x15181;
int32_t x15186 = x15182;
for(int x15187=0; x15187 < -7; x15187++) {
int32_t x15188 = x15184;
int32_t x15189 = x15185;
float x15190 = x15089[x15189];
int32_t x15191 = x15186;
float x15192 = x248[x15191];
float x15193 = x15190 * x15192;
x15162[x15188] = x15193;
x15184 += 1;
if (x520) {
x15185 += 1;
} else {
}
if (x452) {
x15186 += 1;
} else {
}

}
x15177 += -7;
if (x452) {
x15178 += -7;
} else {
}
if (x452) {
x15179 += 1;
} else {
}

}
x15170 += -7;
if (x542) {
x15171 += -7;
} else {
}
if (x542) {
x15172 += 1;
} else {
}

}
x15163 += 7;
if (x452) {
x15164 += 7;
} else {
}
if (x452) {
x15165 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x15235 = (float*)myMalloc(7 * sizeof(float));;
int32_t x15236 = 0;
int32_t x15237 = 0;
int32_t x15238 = 0;
for(int x15239=0; x15239 < 1; x15239++) {
int32_t x15240 = x15237;
int32_t x15241 = x15238;
int32_t x15242 = x15236;
int32_t x15243 = x15242;
int32_t x15244 = x15240;
int32_t x15245 = x15241;
for(int x15246=0; x15246 < -1; x15246++) {
int32_t x15247 = x15244;
int32_t x15248 = x15245;
int32_t x15249 = x15243;
int32_t x15250 = x15249;
int32_t x15251 = x15247;
int32_t x15252 = x15248;
for(int x15253=0; x15253 < 1; x15253++) {
int32_t x15254 = x15251;
int32_t x15255 = x15252;
int32_t x15256 = x15250;
int32_t x15257 = x15256;
int32_t x15258 = x15254;
int32_t x15259 = x15255;
for(int x15260=0; x15260 < -7; x15260++) {
int32_t x15261 = x15257;
int32_t x15262 = x15258;
float x15263 = x15162[x15262];
int32_t x15264 = x15259;
float x15265 = x93[x15264];
float x15266 = x15263 + x15265;
x15235[x15261] = x15266;
x15257 += 1;
if (x520) {
x15258 += 1;
} else {
}
if (x452) {
x15259 += 1;
} else {
}

}
x15250 += -7;
if (x452) {
x15251 += -7;
} else {
}
if (x452) {
x15252 += 1;
} else {
}

}
x15243 += -7;
if (x542) {
x15244 += -7;
} else {
}
if (x542) {
x15245 += 1;
} else {
}

}
x15236 += 7;
if (x452) {
x15237 += 7;
} else {
}
if (x452) {
x15238 += -1;
} else {
}

}
int32_t x15304 = 0;
int32_t x15305 = 0;
int32_t x15306 = 0;
for(int x15307=0; x15307 < 1; x15307++) {
int32_t x15308 = x15305;
int32_t x15309 = x15306;
int32_t x15310 = x15304;
int32_t x15311 = x15310;
int32_t x15312 = x15308;
int32_t x15313 = x15309;
for(int x15314=0; x15314 < -1; x15314++) {
int32_t x15315 = x15312;
int32_t x15316 = x15313;
int32_t x15317 = x15311;
int32_t x15318 = x15317;
int32_t x15319 = x15315;
int32_t x15320 = x15316;
for(int x15321=0; x15321 < 1; x15321++) {
int32_t x15322 = x15319;
int32_t x15323 = x15320;
int32_t x15324 = x15318;
int32_t x15325 = x15324;
int32_t x15326 = x15322;
int32_t x15327 = x15323;
for(int x15328=0; x15328 < -7; x15328++) {
int32_t x15329 = x15326;
float x15330 = x15235[x15329];
int32_t x15331 = x15327;
float x15332 = x14324[x15331];
float x15333 = x15330 + x15332;
x15235[x15329] = x15333;
x15325 += 1;
if (x520) {
x15326 += 1;
} else {
}
if (x520) {
x15327 += 1;
} else {
}

}
x15318 += -7;
if (x452) {
x15319 += -7;
} else {
}
if (x452) {
x15320 += -7;
} else {
}

}
x15311 += -7;
if (x542) {
x15312 += -7;
} else {
}
if (x542) {
x15313 += -7;
} else {
}

}
x15304 += 7;
if (x452) {
x15305 += 7;
} else {
}
if (x452) {
x15306 += 7;
} else {
}

}
float* x15371 = (float*)myMalloc(7 * sizeof(float));;
for(int x15372=0; x15372 < 7; x15372++) {
float x15373 = x15235[x15372];
bool x15374 = x15373 < 0.0f;
if (x15374) {
x15371[x15372] = 0.0f;
} else {
float x15377 = x15235[x15372];
x15371[x15372] = x15377;
}

}
float* x15383 = (float*)myMalloc(x4967 * sizeof(float));;
float* x15384 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x15385=0; x15385 < 1; x15385++) {
int32_t x15386 = x15385 * 7;
float* x15387 = x15371+x15386;
int32_t x15388 = x15385 * x4965;
float* x15389 = x15383+x15388;
int32_t x15390 = x15385 * x1520;
float* x15391 = x15384+x15390;
for(int x15392=0; x15392 < -1; x15392++) {
int32_t x15393 = x15392 / 1;
int32_t x15397 = x15393 * x1150;
int32_t x15398 = x15397 * x1152;
int32_t x15394 = x15392 % 1;
int32_t x15395 = x15394 / 1;
int32_t x15399 = x15395 * x1150;
int32_t x15400 = x15399 * x1152;
int32_t x15401 = x15398 + x15400;
int32_t x15396 = x15394 % 1;
int32_t x15402 = x15396 * x1152;
int32_t x15403 = x15402 * x1152;
int32_t x15404 = x15401 + x15403;
float* x15405 = x15391+x15404;
int32_t x15406 = x15393 * -7;
float* x15407 = x15387+x15406;
for(int x15408=0; x15408 < x1150; x15408++) {
int32_t x15410 = x15408 * x1152;
float* x15411 = x15405+x15410;
int32_t x15409 = x15408 + x15395;
int32_t x15412 = x15409 * -7;
int32_t x15413 = x15412 + x15396;
float* x15414 = x15407+x15413;
memcpy(x15411, x15414, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x1153,-1,1,x139,-1,x15391,x1153,1,x15389,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(512) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x15427 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x15428 = 0;
int32_t x15429 = 0;
int32_t x15430 = 0;
for(int x15431=0; x15431 < -7; x15431++) {
int32_t x15432 = x15428;
int32_t x15433 = x15429;
float x15434 = x15383[x15433];
int32_t x15435 = x15430;
float x15436 = x67[x15435];
float x15437 = x15434 - x15436;
x15427[x15432] = x15437;
x15428 += 1;
if (x452) {
x15429 += x4965;
} else {
}
if (x452) {
x15430 += -1;
} else {
}

}
float* x15448 = (float*)myMalloc(512 * sizeof(float));;
for(int x15449=0; x15449 < 512; x15449++) {
float x15450 = x121[x15449];
float x15451 = x15450 + 1.0E-5f;
x15448[x15449] = x15451;

}
float* x15455 = (float*)myMalloc(512 * sizeof(float));;
for(int x15456=0; x15456 < 512; x15456++) {
float x15457 = x15448[x15456];
double x15458 = (double)x15457;
double x15459 = sqrt(x15458);
float x15460 = (float)x15459;
x15455[x15456] = x15460;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x15468 = (float*)myMalloc(7 * sizeof(float));;
int32_t x15469 = 0;
int32_t x15470 = 0;
int32_t x15471 = 0;
for(int x15472=0; x15472 < 1; x15472++) {
int32_t x15473 = x15470;
int32_t x15474 = x15471;
int32_t x15475 = x15469;
int32_t x15476 = x15475;
int32_t x15477 = x15473;
int32_t x15478 = x15474;
for(int x15479=0; x15479 < -1; x15479++) {
int32_t x15480 = x15477;
int32_t x15481 = x15478;
int32_t x15482 = x15476;
int32_t x15483 = x15482;
int32_t x15484 = x15480;
int32_t x15485 = x15481;
for(int x15486=0; x15486 < 1; x15486++) {
int32_t x15487 = x15484;
int32_t x15488 = x15485;
int32_t x15489 = x15483;
int32_t x15490 = x15489;
int32_t x15491 = x15487;
int32_t x15492 = x15488;
for(int x15493=0; x15493 < -7; x15493++) {
int32_t x15494 = x15490;
int32_t x15495 = x15491;
float x15496 = x15427[x15495];
int32_t x15497 = x15492;
float x15498 = x15455[x15497];
float x15499 = x15496 / x15498;
x15468[x15494] = x15499;
x15490 += 1;
if (x520) {
x15491 += 1;
} else {
}
if (x452) {
x15492 += 1;
} else {
}

}
x15483 += -7;
if (x452) {
x15484 += -7;
} else {
}
if (x452) {
x15485 += 1;
} else {
}

}
x15476 += -7;
if (x452) {
x15477 += -7;
} else {
}
if (x542) {
x15478 += 1;
} else {
}

}
x15469 += 7;
if (x452) {
x15470 += -7;
} else {
}
if (x452) {
x15471 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x15541 = (float*)myMalloc(7 * sizeof(float));;
int32_t x15542 = 0;
int32_t x15543 = 0;
int32_t x15544 = 0;
for(int x15545=0; x15545 < 1; x15545++) {
int32_t x15546 = x15543;
int32_t x15547 = x15544;
int32_t x15548 = x15542;
int32_t x15549 = x15548;
int32_t x15550 = x15546;
int32_t x15551 = x15547;
for(int x15552=0; x15552 < -1; x15552++) {
int32_t x15553 = x15550;
int32_t x15554 = x15551;
int32_t x15555 = x15549;
int32_t x15556 = x15555;
int32_t x15557 = x15553;
int32_t x15558 = x15554;
for(int x15559=0; x15559 < 1; x15559++) {
int32_t x15560 = x15557;
int32_t x15561 = x15558;
int32_t x15562 = x15556;
int32_t x15563 = x15562;
int32_t x15564 = x15560;
int32_t x15565 = x15561;
for(int x15566=0; x15566 < -7; x15566++) {
int32_t x15567 = x15563;
int32_t x15568 = x15564;
float x15569 = x15468[x15568];
int32_t x15570 = x15565;
float x15571 = x201[x15570];
float x15572 = x15569 * x15571;
x15541[x15567] = x15572;
x15563 += 1;
if (x520) {
x15564 += 1;
} else {
}
if (x452) {
x15565 += 1;
} else {
}

}
x15556 += -7;
if (x452) {
x15557 += -7;
} else {
}
if (x452) {
x15558 += 1;
} else {
}

}
x15549 += -7;
if (x542) {
x15550 += -7;
} else {
}
if (x542) {
x15551 += 1;
} else {
}

}
x15542 += 7;
if (x452) {
x15543 += 7;
} else {
}
if (x452) {
x15544 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x15614 = (float*)myMalloc(7 * sizeof(float));;
int32_t x15615 = 0;
int32_t x15616 = 0;
int32_t x15617 = 0;
for(int x15618=0; x15618 < 1; x15618++) {
int32_t x15619 = x15616;
int32_t x15620 = x15617;
int32_t x15621 = x15615;
int32_t x15622 = x15621;
int32_t x15623 = x15619;
int32_t x15624 = x15620;
for(int x15625=0; x15625 < -1; x15625++) {
int32_t x15626 = x15623;
int32_t x15627 = x15624;
int32_t x15628 = x15622;
int32_t x15629 = x15628;
int32_t x15630 = x15626;
int32_t x15631 = x15627;
for(int x15632=0; x15632 < 1; x15632++) {
int32_t x15633 = x15630;
int32_t x15634 = x15631;
int32_t x15635 = x15629;
int32_t x15636 = x15635;
int32_t x15637 = x15633;
int32_t x15638 = x15634;
for(int x15639=0; x15639 < -7; x15639++) {
int32_t x15640 = x15636;
int32_t x15641 = x15637;
float x15642 = x15541[x15641];
int32_t x15643 = x15638;
float x15644 = x224[x15643];
float x15645 = x15642 + x15644;
x15614[x15640] = x15645;
x15636 += 1;
if (x520) {
x15637 += 1;
} else {
}
if (x452) {
x15638 += 1;
} else {
}

}
x15629 += -7;
if (x452) {
x15630 += -7;
} else {
}
if (x452) {
x15631 += 1;
} else {
}

}
x15622 += -7;
if (x542) {
x15623 += -7;
} else {
}
if (x542) {
x15624 += 1;
} else {
}

}
x15615 += 7;
if (x452) {
x15616 += 7;
} else {
}
if (x452) {
x15617 += -1;
} else {
}

}
float* x15683 = (float*)myMalloc(7 * sizeof(float));;
for(int x15684=0; x15684 < 7; x15684++) {
float x15685 = x15614[x15684];
bool x15686 = x15685 < 0.0f;
if (x15686) {
x15683[x15684] = 0.0f;
} else {
float x15689 = x15614[x15684];
x15683[x15684] = x15689;
}

}
float* x15695 = (float*)myMalloc(x5271 * sizeof(float));;
float* x15696 = (float*)myMalloc(x4623 * sizeof(float));;
for(int x15697=0; x15697 < 1; x15697++) {
int32_t x15698 = x15697 * 7;
float* x15699 = x15683+x15698;
int32_t x15700 = x15697 * x5269;
float* x15701 = x15695+x15700;
int32_t x15702 = x15697 * x4623;
float* x15703 = x15696+x15702;
for(int x15704=0; x15704 < -9; x15704++) {
int32_t x15705 = x15704 / 9;
int32_t x15709 = x15705 * 3;
int32_t x15710 = x15709 * 3;
int32_t x15711 = x15710 * x4615;
int32_t x15712 = x15711 * x4617;
int32_t x15706 = x15704 % 9;
int32_t x15707 = x15706 / 3;
int32_t x15713 = x15707 * 3;
int32_t x15714 = x15713 * x4615;
int32_t x15715 = x15714 * x4617;
int32_t x15716 = x15712 + x15715;
int32_t x15708 = x15706 % 3;
int32_t x15717 = x15708 * x4617;
int32_t x15718 = x15717 * x4617;
int32_t x15719 = x15716 + x15718;
float* x15720 = x15703+x15719;
int32_t x15721 = x15705 * -7;
float* x15722 = x15699+x15721;
for(int x15723=0; x15723 < x4615; x15723++) {
int32_t x15724 = x15723 * 2;
int32_t x15725 = x15724 - 1;
int32_t x15726 = x15725 + x15707;
bool x15727 = x15726 < 0;
bool x15728 = x15726 >= 1;
bool x15729 = x15727 || x15728;
if (x15729) {
int32_t x15730 = x15723 * x4617;
float* x15731 = x15720+x15730;
memset(x15731, 0, 4 * x4617);;
} else {
int32_t x15730 = x15723 * x4617;
int32_t x15746 = x15726 * -7;
for(int x15734=0; x15734 < x4617; x15734++) {
int32_t x15735 = x15734 * 2;
int32_t x15736 = x15735 - 1;
int32_t x15737 = x15736 + x15708;
bool x15738 = x15737 < 0;
bool x15739 = x15737 >= -7;
bool x15740 = x15738 || x15739;
if (x15740) {
int32_t x15741 = x15730 + x15734;
float* x15742 = x15720+x15741;
memset(x15742, 0, 4 * 1);;
} else {
int32_t x15741 = x15730 + x15734;
float* x15745 = x15720+x15741;
int32_t x15747 = x15746 + x15737;
float* x15748 = x15722+x15747;
memcpy(x15745, x15748, 4 * 1);;
}

}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x4618,-9,1,x34,-9,x15703,x4618,1,x15701,x4618);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(512) x Sym(4615) x Sym(4617)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x15767 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x15768 = 0;
int32_t x15769 = 0;
int32_t x15770 = 0;
for(int x15771=0; x15771 < -7; x15771++) {
int32_t x15772 = x15768;
int32_t x15773 = x15769;
float x15774 = x15695[x15773];
int32_t x15775 = x15770;
float x15776 = x113[x15775];
float x15777 = x15774 - x15776;
x15767[x15772] = x15777;
x15768 += 1;
if (x452) {
x15769 += x5269;
} else {
}
if (x452) {
x15770 += -1;
} else {
}

}
float* x15788 = (float*)myMalloc(512 * sizeof(float));;
for(int x15789=0; x15789 < 512; x15789++) {
float x15790 = x50[x15789];
float x15791 = x15790 + 1.0E-5f;
x15788[x15789] = x15791;

}
float* x15795 = (float*)myMalloc(512 * sizeof(float));;
for(int x15796=0; x15796 < 512; x15796++) {
float x15797 = x15788[x15796];
double x15798 = (double)x15797;
double x15799 = sqrt(x15798);
float x15800 = (float)x15799;
x15795[x15796] = x15800;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x15808 = (float*)myMalloc(7 * sizeof(float));;
int32_t x15809 = 0;
int32_t x15810 = 0;
int32_t x15811 = 0;
for(int x15812=0; x15812 < 1; x15812++) {
int32_t x15813 = x15810;
int32_t x15814 = x15811;
int32_t x15815 = x15809;
int32_t x15816 = x15815;
int32_t x15817 = x15813;
int32_t x15818 = x15814;
for(int x15819=0; x15819 < -1; x15819++) {
int32_t x15820 = x15817;
int32_t x15821 = x15818;
int32_t x15822 = x15816;
int32_t x15823 = x15822;
int32_t x15824 = x15820;
int32_t x15825 = x15821;
for(int x15826=0; x15826 < 1; x15826++) {
int32_t x15827 = x15824;
int32_t x15828 = x15825;
int32_t x15829 = x15823;
int32_t x15830 = x15829;
int32_t x15831 = x15827;
int32_t x15832 = x15828;
for(int x15833=0; x15833 < -7; x15833++) {
int32_t x15834 = x15830;
int32_t x15835 = x15831;
float x15836 = x15767[x15835];
int32_t x15837 = x15832;
float x15838 = x15795[x15837];
float x15839 = x15836 / x15838;
x15808[x15834] = x15839;
x15830 += 1;
if (x520) {
x15831 += 1;
} else {
}
if (x452) {
x15832 += 1;
} else {
}

}
x15823 += -7;
if (x452) {
x15824 += -7;
} else {
}
if (x452) {
x15825 += 1;
} else {
}

}
x15816 += -7;
if (x452) {
x15817 += -7;
} else {
}
if (x542) {
x15818 += 1;
} else {
}

}
x15809 += 7;
if (x452) {
x15810 += -7;
} else {
}
if (x452) {
x15811 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x15881 = (float*)myMalloc(7 * sizeof(float));;
int32_t x15882 = 0;
int32_t x15883 = 0;
int32_t x15884 = 0;
for(int x15885=0; x15885 < 1; x15885++) {
int32_t x15886 = x15883;
int32_t x15887 = x15884;
int32_t x15888 = x15882;
int32_t x15889 = x15888;
int32_t x15890 = x15886;
int32_t x15891 = x15887;
for(int x15892=0; x15892 < -1; x15892++) {
int32_t x15893 = x15890;
int32_t x15894 = x15891;
int32_t x15895 = x15889;
int32_t x15896 = x15895;
int32_t x15897 = x15893;
int32_t x15898 = x15894;
for(int x15899=0; x15899 < 1; x15899++) {
int32_t x15900 = x15897;
int32_t x15901 = x15898;
int32_t x15902 = x15896;
int32_t x15903 = x15902;
int32_t x15904 = x15900;
int32_t x15905 = x15901;
for(int x15906=0; x15906 < -7; x15906++) {
int32_t x15907 = x15903;
int32_t x15908 = x15904;
float x15909 = x15808[x15908];
int32_t x15910 = x15905;
float x15911 = x205[x15910];
float x15912 = x15909 * x15911;
x15881[x15907] = x15912;
x15903 += 1;
if (x520) {
x15904 += 1;
} else {
}
if (x452) {
x15905 += 1;
} else {
}

}
x15896 += -7;
if (x452) {
x15897 += -7;
} else {
}
if (x452) {
x15898 += 1;
} else {
}

}
x15889 += -7;
if (x542) {
x15890 += -7;
} else {
}
if (x542) {
x15891 += 1;
} else {
}

}
x15882 += 7;
if (x452) {
x15883 += 7;
} else {
}
if (x452) {
x15884 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x15954 = (float*)myMalloc(7 * sizeof(float));;
int32_t x15955 = 0;
int32_t x15956 = 0;
int32_t x15957 = 0;
for(int x15958=0; x15958 < 1; x15958++) {
int32_t x15959 = x15956;
int32_t x15960 = x15957;
int32_t x15961 = x15955;
int32_t x15962 = x15961;
int32_t x15963 = x15959;
int32_t x15964 = x15960;
for(int x15965=0; x15965 < -1; x15965++) {
int32_t x15966 = x15963;
int32_t x15967 = x15964;
int32_t x15968 = x15962;
int32_t x15969 = x15968;
int32_t x15970 = x15966;
int32_t x15971 = x15967;
for(int x15972=0; x15972 < 1; x15972++) {
int32_t x15973 = x15970;
int32_t x15974 = x15971;
int32_t x15975 = x15969;
int32_t x15976 = x15975;
int32_t x15977 = x15973;
int32_t x15978 = x15974;
for(int x15979=0; x15979 < -7; x15979++) {
int32_t x15980 = x15976;
int32_t x15981 = x15977;
float x15982 = x15881[x15981];
int32_t x15983 = x15978;
float x15984 = x159[x15983];
float x15985 = x15982 + x15984;
x15954[x15980] = x15985;
x15976 += 1;
if (x520) {
x15977 += 1;
} else {
}
if (x452) {
x15978 += 1;
} else {
}

}
x15969 += -7;
if (x452) {
x15970 += -7;
} else {
}
if (x452) {
x15971 += 1;
} else {
}

}
x15962 += -7;
if (x542) {
x15963 += -7;
} else {
}
if (x542) {
x15964 += 1;
} else {
}

}
x15955 += 7;
if (x452) {
x15956 += 7;
} else {
}
if (x452) {
x15957 += -1;
} else {
}

}
float* x16023 = (float*)myMalloc(7 * sizeof(float));;
for(int x16024=0; x16024 < 7; x16024++) {
float x16025 = x15954[x16024];
bool x16026 = x16025 < 0.0f;
if (x16026) {
x16023[x16024] = 0.0f;
} else {
float x16029 = x15954[x16024];
x16023[x16024] = x16029;
}

}
float* x16038 = (float*)myMalloc(x16037 * sizeof(float));;
float* x16039 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x16040=0; x16040 < 1; x16040++) {
int32_t x16041 = x16040 * 7;
float* x16042 = x16023+x16041;
int32_t x16043 = x16040 * x16035;
float* x16044 = x16038+x16043;
int32_t x16045 = x16040 * x1520;
float* x16046 = x16039+x16045;
for(int x16047=0; x16047 < -1; x16047++) {
int32_t x16048 = x16047 / 1;
int32_t x16052 = x16048 * x1150;
int32_t x16053 = x16052 * x1152;
int32_t x16049 = x16047 % 1;
int32_t x16050 = x16049 / 1;
int32_t x16054 = x16050 * x1150;
int32_t x16055 = x16054 * x1152;
int32_t x16056 = x16053 + x16055;
int32_t x16051 = x16049 % 1;
int32_t x16057 = x16051 * x1152;
int32_t x16058 = x16057 * x1152;
int32_t x16059 = x16056 + x16058;
float* x16060 = x16046+x16059;
int32_t x16061 = x16048 * -7;
float* x16062 = x16042+x16061;
for(int x16063=0; x16063 < x1150; x16063++) {
int32_t x16065 = x16063 * x1152;
float* x16066 = x16060+x16065;
int32_t x16064 = x16063 + x16050;
int32_t x16067 = x16064 * -7;
int32_t x16068 = x16067 + x16051;
float* x16069 = x16062+x16068;
memcpy(x16066, x16069, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x1153,-1,1,x212,-1,x16046,x1153,1,x16044,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(2048) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x16082 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x16083 = 0;
int32_t x16084 = 0;
int32_t x16085 = 0;
for(int x16086=0; x16086 < -7; x16086++) {
int32_t x16087 = x16083;
int32_t x16088 = x16084;
float x16089 = x16038[x16088];
int32_t x16090 = x16085;
float x16091 = x115[x16090];
float x16092 = x16089 - x16091;
x16082[x16087] = x16092;
x16083 += 1;
if (x452) {
x16084 += x16035;
} else {
}
if (x452) {
x16085 += -1;
} else {
}

}
float* x16103 = (float*)myMalloc(2048 * sizeof(float));;
for(int x16105=0; x16105 < 2048; x16105++) {
float x16106 = x193[x16105];
float x16107 = x16106 + 1.0E-5f;
x16103[x16105] = x16107;

}
float* x16111 = (float*)myMalloc(2048 * sizeof(float));;
for(int x16112=0; x16112 < 2048; x16112++) {
float x16113 = x16103[x16112];
double x16114 = (double)x16113;
double x16115 = sqrt(x16114);
float x16116 = (float)x16115;
x16111[x16112] = x16116;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x16124 = (float*)myMalloc(7 * sizeof(float));;
int32_t x16125 = 0;
int32_t x16126 = 0;
int32_t x16127 = 0;
for(int x16128=0; x16128 < 1; x16128++) {
int32_t x16129 = x16126;
int32_t x16130 = x16127;
int32_t x16131 = x16125;
int32_t x16132 = x16131;
int32_t x16133 = x16129;
int32_t x16134 = x16130;
for(int x16135=0; x16135 < -1; x16135++) {
int32_t x16136 = x16133;
int32_t x16137 = x16134;
int32_t x16138 = x16132;
int32_t x16139 = x16138;
int32_t x16140 = x16136;
int32_t x16141 = x16137;
for(int x16142=0; x16142 < 1; x16142++) {
int32_t x16143 = x16140;
int32_t x16144 = x16141;
int32_t x16145 = x16139;
int32_t x16146 = x16145;
int32_t x16147 = x16143;
int32_t x16148 = x16144;
for(int x16149=0; x16149 < -7; x16149++) {
int32_t x16150 = x16146;
int32_t x16151 = x16147;
float x16152 = x16082[x16151];
int32_t x16153 = x16148;
float x16154 = x16111[x16153];
float x16155 = x16152 / x16154;
x16124[x16150] = x16155;
x16146 += 1;
if (x520) {
x16147 += 1;
} else {
}
if (x452) {
x16148 += 1;
} else {
}

}
x16139 += -7;
if (x452) {
x16140 += -7;
} else {
}
if (x452) {
x16141 += 1;
} else {
}

}
x16132 += -7;
if (x452) {
x16133 += -7;
} else {
}
if (x542) {
x16134 += 1;
} else {
}

}
x16125 += 7;
if (x452) {
x16126 += -7;
} else {
}
if (x452) {
x16127 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x16197 = (float*)myMalloc(7 * sizeof(float));;
int32_t x16198 = 0;
int32_t x16199 = 0;
int32_t x16200 = 0;
for(int x16201=0; x16201 < 1; x16201++) {
int32_t x16202 = x16199;
int32_t x16203 = x16200;
int32_t x16204 = x16198;
int32_t x16205 = x16204;
int32_t x16206 = x16202;
int32_t x16207 = x16203;
for(int x16208=0; x16208 < -1; x16208++) {
int32_t x16209 = x16206;
int32_t x16210 = x16207;
int32_t x16211 = x16205;
int32_t x16212 = x16211;
int32_t x16213 = x16209;
int32_t x16214 = x16210;
for(int x16215=0; x16215 < 1; x16215++) {
int32_t x16216 = x16213;
int32_t x16217 = x16214;
int32_t x16218 = x16212;
int32_t x16219 = x16218;
int32_t x16220 = x16216;
int32_t x16221 = x16217;
for(int x16222=0; x16222 < -7; x16222++) {
int32_t x16223 = x16219;
int32_t x16224 = x16220;
float x16225 = x16124[x16224];
int32_t x16226 = x16221;
float x16227 = x239[x16226];
float x16228 = x16225 * x16227;
x16197[x16223] = x16228;
x16219 += 1;
if (x520) {
x16220 += 1;
} else {
}
if (x452) {
x16221 += 1;
} else {
}

}
x16212 += -7;
if (x452) {
x16213 += -7;
} else {
}
if (x452) {
x16214 += 1;
} else {
}

}
x16205 += -7;
if (x542) {
x16206 += -7;
} else {
}
if (x542) {
x16207 += 1;
} else {
}

}
x16198 += 7;
if (x452) {
x16199 += 7;
} else {
}
if (x452) {
x16200 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x16270 = (float*)myMalloc(7 * sizeof(float));;
int32_t x16271 = 0;
int32_t x16272 = 0;
int32_t x16273 = 0;
for(int x16274=0; x16274 < 1; x16274++) {
int32_t x16275 = x16272;
int32_t x16276 = x16273;
int32_t x16277 = x16271;
int32_t x16278 = x16277;
int32_t x16279 = x16275;
int32_t x16280 = x16276;
for(int x16281=0; x16281 < -1; x16281++) {
int32_t x16282 = x16279;
int32_t x16283 = x16280;
int32_t x16284 = x16278;
int32_t x16285 = x16284;
int32_t x16286 = x16282;
int32_t x16287 = x16283;
for(int x16288=0; x16288 < 1; x16288++) {
int32_t x16289 = x16286;
int32_t x16290 = x16287;
int32_t x16291 = x16285;
int32_t x16292 = x16291;
int32_t x16293 = x16289;
int32_t x16294 = x16290;
for(int x16295=0; x16295 < -7; x16295++) {
int32_t x16296 = x16292;
int32_t x16297 = x16293;
float x16298 = x16197[x16297];
int32_t x16299 = x16294;
float x16300 = x62[x16299];
float x16301 = x16298 + x16300;
x16270[x16296] = x16301;
x16292 += 1;
if (x520) {
x16293 += 1;
} else {
}
if (x452) {
x16294 += 1;
} else {
}

}
x16285 += -7;
if (x452) {
x16286 += -7;
} else {
}
if (x452) {
x16287 += 1;
} else {
}

}
x16278 += -7;
if (x542) {
x16279 += -7;
} else {
}
if (x542) {
x16280 += 1;
} else {
}

}
x16271 += 7;
if (x452) {
x16272 += 7;
} else {
}
if (x452) {
x16273 += -1;
} else {
}

}
float* x16342 = (float*)myMalloc(x16341 * sizeof(float));;
float* x16343 = (float*)myMalloc(x5273 * sizeof(float));;
for(int x16344=0; x16344 < 1; x16344++) {
int32_t x16345 = x16344 * 7;
float* x16346 = x15371+x16345;
int32_t x16347 = x16344 * x16339;
float* x16348 = x16342+x16347;
int32_t x16349 = x16344 * x5273;
float* x16350 = x16343+x16349;
for(int x16351=0; x16351 < -1; x16351++) {
int32_t x16352 = x16351 / 1;
int32_t x16356 = x16352 * x4615;
int32_t x16357 = x16356 * x4617;
int32_t x16353 = x16351 % 1;
int32_t x16354 = x16353 / 1;
int32_t x16358 = x16354 * x4615;
int32_t x16359 = x16358 * x4617;
int32_t x16360 = x16357 + x16359;
int32_t x16355 = x16353 % 1;
int32_t x16361 = x16355 * x4617;
int32_t x16362 = x16361 * x4617;
int32_t x16363 = x16360 + x16362;
float* x16364 = x16350+x16363;
int32_t x16365 = x16352 * -7;
float* x16366 = x16346+x16365;
for(int x16367=0; x16367 < x4615; x16367++) {
int32_t x16371 = x16367 * x4617;
int32_t x16368 = x16367 * 2;
int32_t x16369 = x16368 + x16354;
int32_t x16374 = x16369 * -7;
int32_t x16375 = x16374 + x16355;
for(int x16370=0; x16370 < x4617; x16370++) {
int32_t x16372 = x16371 + x16370;
float* x16373 = x16364+x16372;
int32_t x16376 = x16370 * 2;
int32_t x16377 = x16375 + x16376;
float* x16378 = x16366+x16377;
memcpy(x16373, x16378, 4 * 1);;

}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x4618,-1,1,x214,-1,x16350,x4618,1,x16348,x4618);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(2048) x Sym(4615) x Sym(4617)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x16393 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x16394 = 0;
int32_t x16395 = 0;
int32_t x16396 = 0;
for(int x16397=0; x16397 < -7; x16397++) {
int32_t x16398 = x16394;
int32_t x16399 = x16395;
float x16400 = x16342[x16399];
int32_t x16401 = x16396;
float x16402 = x64[x16401];
float x16403 = x16400 - x16402;
x16393[x16398] = x16403;
x16394 += 1;
if (x452) {
x16395 += x16339;
} else {
}
if (x452) {
x16396 += -1;
} else {
}

}
float* x16414 = (float*)myMalloc(2048 * sizeof(float));;
for(int x16415=0; x16415 < 2048; x16415++) {
float x16416 = x125[x16415];
float x16417 = x16416 + 1.0E-5f;
x16414[x16415] = x16417;

}
float* x16421 = (float*)myMalloc(2048 * sizeof(float));;
for(int x16422=0; x16422 < 2048; x16422++) {
float x16423 = x16414[x16422];
double x16424 = (double)x16423;
double x16425 = sqrt(x16424);
float x16426 = (float)x16425;
x16421[x16422] = x16426;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x16434 = (float*)myMalloc(7 * sizeof(float));;
int32_t x16435 = 0;
int32_t x16436 = 0;
int32_t x16437 = 0;
for(int x16438=0; x16438 < 1; x16438++) {
int32_t x16439 = x16436;
int32_t x16440 = x16437;
int32_t x16441 = x16435;
int32_t x16442 = x16441;
int32_t x16443 = x16439;
int32_t x16444 = x16440;
for(int x16445=0; x16445 < -1; x16445++) {
int32_t x16446 = x16443;
int32_t x16447 = x16444;
int32_t x16448 = x16442;
int32_t x16449 = x16448;
int32_t x16450 = x16446;
int32_t x16451 = x16447;
for(int x16452=0; x16452 < 1; x16452++) {
int32_t x16453 = x16450;
int32_t x16454 = x16451;
int32_t x16455 = x16449;
int32_t x16456 = x16455;
int32_t x16457 = x16453;
int32_t x16458 = x16454;
for(int x16459=0; x16459 < -7; x16459++) {
int32_t x16460 = x16456;
int32_t x16461 = x16457;
float x16462 = x16393[x16461];
int32_t x16463 = x16458;
float x16464 = x16421[x16463];
float x16465 = x16462 / x16464;
x16434[x16460] = x16465;
x16456 += 1;
if (x520) {
x16457 += 1;
} else {
}
if (x452) {
x16458 += 1;
} else {
}

}
x16449 += -7;
if (x452) {
x16450 += -7;
} else {
}
if (x452) {
x16451 += 1;
} else {
}

}
x16442 += -7;
if (x452) {
x16443 += -7;
} else {
}
if (x542) {
x16444 += 1;
} else {
}

}
x16435 += 7;
if (x452) {
x16436 += -7;
} else {
}
if (x452) {
x16437 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x16507 = (float*)myMalloc(7 * sizeof(float));;
int32_t x16508 = 0;
int32_t x16509 = 0;
int32_t x16510 = 0;
for(int x16511=0; x16511 < 1; x16511++) {
int32_t x16512 = x16509;
int32_t x16513 = x16510;
int32_t x16514 = x16508;
int32_t x16515 = x16514;
int32_t x16516 = x16512;
int32_t x16517 = x16513;
for(int x16518=0; x16518 < -1; x16518++) {
int32_t x16519 = x16516;
int32_t x16520 = x16517;
int32_t x16521 = x16515;
int32_t x16522 = x16521;
int32_t x16523 = x16519;
int32_t x16524 = x16520;
for(int x16525=0; x16525 < 1; x16525++) {
int32_t x16526 = x16523;
int32_t x16527 = x16524;
int32_t x16528 = x16522;
int32_t x16529 = x16528;
int32_t x16530 = x16526;
int32_t x16531 = x16527;
for(int x16532=0; x16532 < -7; x16532++) {
int32_t x16533 = x16529;
int32_t x16534 = x16530;
float x16535 = x16434[x16534];
int32_t x16536 = x16531;
float x16537 = x173[x16536];
float x16538 = x16535 * x16537;
x16507[x16533] = x16538;
x16529 += 1;
if (x520) {
x16530 += 1;
} else {
}
if (x452) {
x16531 += 1;
} else {
}

}
x16522 += -7;
if (x452) {
x16523 += -7;
} else {
}
if (x452) {
x16524 += 1;
} else {
}

}
x16515 += -7;
if (x542) {
x16516 += -7;
} else {
}
if (x542) {
x16517 += 1;
} else {
}

}
x16508 += 7;
if (x452) {
x16509 += 7;
} else {
}
if (x452) {
x16510 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x16580 = (float*)myMalloc(7 * sizeof(float));;
int32_t x16581 = 0;
int32_t x16582 = 0;
int32_t x16583 = 0;
for(int x16584=0; x16584 < 1; x16584++) {
int32_t x16585 = x16582;
int32_t x16586 = x16583;
int32_t x16587 = x16581;
int32_t x16588 = x16587;
int32_t x16589 = x16585;
int32_t x16590 = x16586;
for(int x16591=0; x16591 < -1; x16591++) {
int32_t x16592 = x16589;
int32_t x16593 = x16590;
int32_t x16594 = x16588;
int32_t x16595 = x16594;
int32_t x16596 = x16592;
int32_t x16597 = x16593;
for(int x16598=0; x16598 < 1; x16598++) {
int32_t x16599 = x16596;
int32_t x16600 = x16597;
int32_t x16601 = x16595;
int32_t x16602 = x16601;
int32_t x16603 = x16599;
int32_t x16604 = x16600;
for(int x16605=0; x16605 < -7; x16605++) {
int32_t x16606 = x16602;
int32_t x16607 = x16603;
float x16608 = x16507[x16607];
int32_t x16609 = x16604;
float x16610 = x107[x16609];
float x16611 = x16608 + x16610;
x16580[x16606] = x16611;
x16602 += 1;
if (x520) {
x16603 += 1;
} else {
}
if (x452) {
x16604 += 1;
} else {
}

}
x16595 += -7;
if (x452) {
x16596 += -7;
} else {
}
if (x452) {
x16597 += 1;
} else {
}

}
x16588 += -7;
if (x542) {
x16589 += -7;
} else {
}
if (x542) {
x16590 += 1;
} else {
}

}
x16581 += 7;
if (x452) {
x16582 += 7;
} else {
}
if (x452) {
x16583 += -1;
} else {
}

}
int32_t x16649 = 0;
int32_t x16650 = 0;
int32_t x16651 = 0;
for(int x16652=0; x16652 < 1; x16652++) {
int32_t x16653 = x16650;
int32_t x16654 = x16651;
int32_t x16655 = x16649;
int32_t x16656 = x16655;
int32_t x16657 = x16653;
int32_t x16658 = x16654;
for(int x16659=0; x16659 < -1; x16659++) {
int32_t x16660 = x16657;
int32_t x16661 = x16658;
int32_t x16662 = x16656;
int32_t x16663 = x16662;
int32_t x16664 = x16660;
int32_t x16665 = x16661;
for(int x16666=0; x16666 < 1; x16666++) {
int32_t x16667 = x16664;
int32_t x16668 = x16665;
int32_t x16669 = x16663;
int32_t x16670 = x16669;
int32_t x16671 = x16667;
int32_t x16672 = x16668;
for(int x16673=0; x16673 < -7; x16673++) {
int32_t x16674 = x16671;
float x16675 = x16270[x16674];
int32_t x16676 = x16672;
float x16677 = x16580[x16676];
float x16678 = x16675 + x16677;
x16270[x16674] = x16678;
x16670 += 1;
if (x520) {
x16671 += 1;
} else {
}
if (x520) {
x16672 += 1;
} else {
}

}
x16663 += -7;
if (x452) {
x16664 += -7;
} else {
}
if (x452) {
x16665 += -7;
} else {
}

}
x16656 += -7;
if (x542) {
x16657 += -7;
} else {
}
if (x542) {
x16658 += -7;
} else {
}

}
x16649 += 7;
if (x452) {
x16650 += 7;
} else {
}
if (x452) {
x16651 += 7;
} else {
}

}
float* x16716 = (float*)myMalloc(7 * sizeof(float));;
for(int x16717=0; x16717 < 7; x16717++) {
float x16718 = x16270[x16717];
bool x16719 = x16718 < 0.0f;
if (x16719) {
x16716[x16717] = 0.0f;
} else {
float x16722 = x16270[x16717];
x16716[x16717] = x16722;
}

}
float* x16728 = (float*)myMalloc(x4967 * sizeof(float));;
float* x16729 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x16730=0; x16730 < 1; x16730++) {
int32_t x16731 = x16730 * 7;
float* x16732 = x16716+x16731;
int32_t x16733 = x16730 * x4965;
float* x16734 = x16728+x16733;
int32_t x16735 = x16730 * x1520;
float* x16736 = x16729+x16735;
for(int x16737=0; x16737 < -1; x16737++) {
int32_t x16738 = x16737 / 1;
int32_t x16742 = x16738 * x1150;
int32_t x16743 = x16742 * x1152;
int32_t x16739 = x16737 % 1;
int32_t x16740 = x16739 / 1;
int32_t x16744 = x16740 * x1150;
int32_t x16745 = x16744 * x1152;
int32_t x16746 = x16743 + x16745;
int32_t x16741 = x16739 % 1;
int32_t x16747 = x16741 * x1152;
int32_t x16748 = x16747 * x1152;
int32_t x16749 = x16746 + x16748;
float* x16750 = x16736+x16749;
int32_t x16751 = x16738 * -7;
float* x16752 = x16732+x16751;
for(int x16753=0; x16753 < x1150; x16753++) {
int32_t x16755 = x16753 * x1152;
float* x16756 = x16750+x16755;
int32_t x16754 = x16753 + x16740;
int32_t x16757 = x16754 * -7;
int32_t x16758 = x16757 + x16741;
float* x16759 = x16752+x16758;
memcpy(x16756, x16759, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x1153,-1,1,x215,-1,x16736,x1153,1,x16734,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(512) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x16772 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x16773 = 0;
int32_t x16774 = 0;
int32_t x16775 = 0;
for(int x16776=0; x16776 < -7; x16776++) {
int32_t x16777 = x16773;
int32_t x16778 = x16774;
float x16779 = x16728[x16778];
int32_t x16780 = x16775;
float x16781 = x154[x16780];
float x16782 = x16779 - x16781;
x16772[x16777] = x16782;
x16773 += 1;
if (x452) {
x16774 += x4965;
} else {
}
if (x452) {
x16775 += -1;
} else {
}

}
float* x16793 = (float*)myMalloc(512 * sizeof(float));;
for(int x16794=0; x16794 < 512; x16794++) {
float x16795 = x65[x16794];
float x16796 = x16795 + 1.0E-5f;
x16793[x16794] = x16796;

}
float* x16800 = (float*)myMalloc(512 * sizeof(float));;
for(int x16801=0; x16801 < 512; x16801++) {
float x16802 = x16793[x16801];
double x16803 = (double)x16802;
double x16804 = sqrt(x16803);
float x16805 = (float)x16804;
x16800[x16801] = x16805;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x16813 = (float*)myMalloc(7 * sizeof(float));;
int32_t x16814 = 0;
int32_t x16815 = 0;
int32_t x16816 = 0;
for(int x16817=0; x16817 < 1; x16817++) {
int32_t x16818 = x16815;
int32_t x16819 = x16816;
int32_t x16820 = x16814;
int32_t x16821 = x16820;
int32_t x16822 = x16818;
int32_t x16823 = x16819;
for(int x16824=0; x16824 < -1; x16824++) {
int32_t x16825 = x16822;
int32_t x16826 = x16823;
int32_t x16827 = x16821;
int32_t x16828 = x16827;
int32_t x16829 = x16825;
int32_t x16830 = x16826;
for(int x16831=0; x16831 < 1; x16831++) {
int32_t x16832 = x16829;
int32_t x16833 = x16830;
int32_t x16834 = x16828;
int32_t x16835 = x16834;
int32_t x16836 = x16832;
int32_t x16837 = x16833;
for(int x16838=0; x16838 < -7; x16838++) {
int32_t x16839 = x16835;
int32_t x16840 = x16836;
float x16841 = x16772[x16840];
int32_t x16842 = x16837;
float x16843 = x16800[x16842];
float x16844 = x16841 / x16843;
x16813[x16839] = x16844;
x16835 += 1;
if (x520) {
x16836 += 1;
} else {
}
if (x452) {
x16837 += 1;
} else {
}

}
x16828 += -7;
if (x452) {
x16829 += -7;
} else {
}
if (x452) {
x16830 += 1;
} else {
}

}
x16821 += -7;
if (x452) {
x16822 += -7;
} else {
}
if (x542) {
x16823 += 1;
} else {
}

}
x16814 += 7;
if (x452) {
x16815 += -7;
} else {
}
if (x452) {
x16816 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x16886 = (float*)myMalloc(7 * sizeof(float));;
int32_t x16887 = 0;
int32_t x16888 = 0;
int32_t x16889 = 0;
for(int x16890=0; x16890 < 1; x16890++) {
int32_t x16891 = x16888;
int32_t x16892 = x16889;
int32_t x16893 = x16887;
int32_t x16894 = x16893;
int32_t x16895 = x16891;
int32_t x16896 = x16892;
for(int x16897=0; x16897 < -1; x16897++) {
int32_t x16898 = x16895;
int32_t x16899 = x16896;
int32_t x16900 = x16894;
int32_t x16901 = x16900;
int32_t x16902 = x16898;
int32_t x16903 = x16899;
for(int x16904=0; x16904 < 1; x16904++) {
int32_t x16905 = x16902;
int32_t x16906 = x16903;
int32_t x16907 = x16901;
int32_t x16908 = x16907;
int32_t x16909 = x16905;
int32_t x16910 = x16906;
for(int x16911=0; x16911 < -7; x16911++) {
int32_t x16912 = x16908;
int32_t x16913 = x16909;
float x16914 = x16813[x16913];
int32_t x16915 = x16910;
float x16916 = x46[x16915];
float x16917 = x16914 * x16916;
x16886[x16912] = x16917;
x16908 += 1;
if (x520) {
x16909 += 1;
} else {
}
if (x452) {
x16910 += 1;
} else {
}

}
x16901 += -7;
if (x452) {
x16902 += -7;
} else {
}
if (x452) {
x16903 += 1;
} else {
}

}
x16894 += -7;
if (x542) {
x16895 += -7;
} else {
}
if (x542) {
x16896 += 1;
} else {
}

}
x16887 += 7;
if (x452) {
x16888 += 7;
} else {
}
if (x452) {
x16889 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x16959 = (float*)myMalloc(7 * sizeof(float));;
int32_t x16960 = 0;
int32_t x16961 = 0;
int32_t x16962 = 0;
for(int x16963=0; x16963 < 1; x16963++) {
int32_t x16964 = x16961;
int32_t x16965 = x16962;
int32_t x16966 = x16960;
int32_t x16967 = x16966;
int32_t x16968 = x16964;
int32_t x16969 = x16965;
for(int x16970=0; x16970 < -1; x16970++) {
int32_t x16971 = x16968;
int32_t x16972 = x16969;
int32_t x16973 = x16967;
int32_t x16974 = x16973;
int32_t x16975 = x16971;
int32_t x16976 = x16972;
for(int x16977=0; x16977 < 1; x16977++) {
int32_t x16978 = x16975;
int32_t x16979 = x16976;
int32_t x16980 = x16974;
int32_t x16981 = x16980;
int32_t x16982 = x16978;
int32_t x16983 = x16979;
for(int x16984=0; x16984 < -7; x16984++) {
int32_t x16985 = x16981;
int32_t x16986 = x16982;
float x16987 = x16886[x16986];
int32_t x16988 = x16983;
float x16989 = x137[x16988];
float x16990 = x16987 + x16989;
x16959[x16985] = x16990;
x16981 += 1;
if (x520) {
x16982 += 1;
} else {
}
if (x452) {
x16983 += 1;
} else {
}

}
x16974 += -7;
if (x452) {
x16975 += -7;
} else {
}
if (x452) {
x16976 += 1;
} else {
}

}
x16967 += -7;
if (x542) {
x16968 += -7;
} else {
}
if (x542) {
x16969 += 1;
} else {
}

}
x16960 += 7;
if (x452) {
x16961 += 7;
} else {
}
if (x452) {
x16962 += -1;
} else {
}

}
float* x17028 = (float*)myMalloc(7 * sizeof(float));;
for(int x17029=0; x17029 < 7; x17029++) {
float x17030 = x16959[x17029];
bool x17031 = x17030 < 0.0f;
if (x17031) {
x17028[x17029] = 0.0f;
} else {
float x17034 = x16959[x17029];
x17028[x17029] = x17034;
}

}
float* x17040 = (float*)myMalloc(x4967 * sizeof(float));;
float* x17041 = (float*)myMalloc(x1158 * sizeof(float));;
for(int x17042=0; x17042 < 1; x17042++) {
int32_t x17043 = x17042 * 7;
float* x17044 = x17028+x17043;
int32_t x17045 = x17042 * x4965;
float* x17046 = x17040+x17045;
int32_t x17047 = x17042 * x1158;
float* x17048 = x17041+x17047;
for(int x17049=0; x17049 < -9; x17049++) {
int32_t x17050 = x17049 / 9;
int32_t x17054 = x17050 * 3;
int32_t x17055 = x17054 * 3;
int32_t x17056 = x17055 * x1150;
int32_t x17057 = x17056 * x1152;
int32_t x17051 = x17049 % 9;
int32_t x17052 = x17051 / 3;
int32_t x17058 = x17052 * 3;
int32_t x17059 = x17058 * x1150;
int32_t x17060 = x17059 * x1152;
int32_t x17061 = x17057 + x17060;
int32_t x17053 = x17051 % 3;
int32_t x17062 = x17053 * x1152;
int32_t x17063 = x17062 * x1152;
int32_t x17064 = x17061 + x17063;
float* x17065 = x17048+x17064;
int32_t x17066 = x17050 * -7;
float* x17067 = x17044+x17066;
int32_t x17079 = 1 - x17053;
bool x17080 = x17079 > 0;
int32_t x17081;
if (x17080) {
x17081 = x17079;
} else {
x17081 = 0;
}
int32_t x17082 = 3 - x17053;
int32_t x17083 = x17082 - 1;
int32_t x17084 = 1 - x17083;
bool x17085 = x17084 > 0;
int32_t x17086;
if (x17085) {
x17086 = x17084;
} else {
x17086 = 0;
}
int32_t x17087 = x1152 - x17086;
int32_t x17088 = x17087 - x17081;
bool x17089 = x17088 <= 0;
bool x17093 = x17081 > 0;
int32_t x17078 = -1 + x17053;
bool x17106 = x17086 > 0;
for(int x17068=0; x17068 < x1150; x17068++) {
int32_t x17069 = x17068 - 1;
int32_t x17070 = x17069 + x17052;
bool x17071 = x17070 < 0;
bool x17072 = x17070 >= 1;
bool x17073 = x17071 || x17072;
if (x17073) {
int32_t x17074 = x17068 * x1152;
float* x17075 = x17065+x17074;
memset(x17075, 0, 4 * x1152);;
} else {
if (x17089) {
int32_t x17074 = x17068 * x1152;
float* x17090 = x17065+x17074;
memset(x17090, 0, 4 * x1152);;
} else {
int32_t x17074 = x17068 * x1152;
if (x17093) {
float* x17094 = x17065+x17074;
memset(x17094, 0, 4 * x17081);;
} else {
}
// may have segfault here
int32_t x17099 = x17074 + x17081;
float* x17100 = x17065+x17099;
int32_t x17101 = x17070 * -7;
int32_t x17102 = x17101 + x17078;
int32_t x17103 = x17102 + x17081;
float* x17104 = x17067+x17103;
memcpy(x17100, x17104, 4 * x17088);;
if (x17106) {
int32_t x17107 = x17074 + x1152;
int32_t x17108 = x17107 - x17086;
float* x17109 = x17065+x17108;
memset(x17109, 0, 4 * x17086);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x1153,-9,1,x155,-9,x17048,x1153,1,x17046,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(512) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x17128 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x17129 = 0;
int32_t x17130 = 0;
int32_t x17131 = 0;
for(int x17132=0; x17132 < -7; x17132++) {
int32_t x17133 = x17129;
int32_t x17134 = x17130;
float x17135 = x17040[x17134];
int32_t x17136 = x17131;
float x17137 = x138[x17136];
float x17138 = x17135 - x17137;
x17128[x17133] = x17138;
x17129 += 1;
if (x452) {
x17130 += x4965;
} else {
}
if (x452) {
x17131 += -1;
} else {
}

}
float* x17149 = (float*)myMalloc(512 * sizeof(float));;
for(int x17150=0; x17150 < 512; x17150++) {
float x17151 = x195[x17150];
float x17152 = x17151 + 1.0E-5f;
x17149[x17150] = x17152;

}
float* x17156 = (float*)myMalloc(512 * sizeof(float));;
for(int x17157=0; x17157 < 512; x17157++) {
float x17158 = x17149[x17157];
double x17159 = (double)x17158;
double x17160 = sqrt(x17159);
float x17161 = (float)x17160;
x17156[x17157] = x17161;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x17169 = (float*)myMalloc(7 * sizeof(float));;
int32_t x17170 = 0;
int32_t x17171 = 0;
int32_t x17172 = 0;
for(int x17173=0; x17173 < 1; x17173++) {
int32_t x17174 = x17171;
int32_t x17175 = x17172;
int32_t x17176 = x17170;
int32_t x17177 = x17176;
int32_t x17178 = x17174;
int32_t x17179 = x17175;
for(int x17180=0; x17180 < -1; x17180++) {
int32_t x17181 = x17178;
int32_t x17182 = x17179;
int32_t x17183 = x17177;
int32_t x17184 = x17183;
int32_t x17185 = x17181;
int32_t x17186 = x17182;
for(int x17187=0; x17187 < 1; x17187++) {
int32_t x17188 = x17185;
int32_t x17189 = x17186;
int32_t x17190 = x17184;
int32_t x17191 = x17190;
int32_t x17192 = x17188;
int32_t x17193 = x17189;
for(int x17194=0; x17194 < -7; x17194++) {
int32_t x17195 = x17191;
int32_t x17196 = x17192;
float x17197 = x17128[x17196];
int32_t x17198 = x17193;
float x17199 = x17156[x17198];
float x17200 = x17197 / x17199;
x17169[x17195] = x17200;
x17191 += 1;
if (x520) {
x17192 += 1;
} else {
}
if (x452) {
x17193 += 1;
} else {
}

}
x17184 += -7;
if (x452) {
x17185 += -7;
} else {
}
if (x452) {
x17186 += 1;
} else {
}

}
x17177 += -7;
if (x452) {
x17178 += -7;
} else {
}
if (x542) {
x17179 += 1;
} else {
}

}
x17170 += 7;
if (x452) {
x17171 += -7;
} else {
}
if (x452) {
x17172 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x17242 = (float*)myMalloc(7 * sizeof(float));;
int32_t x17243 = 0;
int32_t x17244 = 0;
int32_t x17245 = 0;
for(int x17246=0; x17246 < 1; x17246++) {
int32_t x17247 = x17244;
int32_t x17248 = x17245;
int32_t x17249 = x17243;
int32_t x17250 = x17249;
int32_t x17251 = x17247;
int32_t x17252 = x17248;
for(int x17253=0; x17253 < -1; x17253++) {
int32_t x17254 = x17251;
int32_t x17255 = x17252;
int32_t x17256 = x17250;
int32_t x17257 = x17256;
int32_t x17258 = x17254;
int32_t x17259 = x17255;
for(int x17260=0; x17260 < 1; x17260++) {
int32_t x17261 = x17258;
int32_t x17262 = x17259;
int32_t x17263 = x17257;
int32_t x17264 = x17263;
int32_t x17265 = x17261;
int32_t x17266 = x17262;
for(int x17267=0; x17267 < -7; x17267++) {
int32_t x17268 = x17264;
int32_t x17269 = x17265;
float x17270 = x17169[x17269];
int32_t x17271 = x17266;
float x17272 = x160[x17271];
float x17273 = x17270 * x17272;
x17242[x17268] = x17273;
x17264 += 1;
if (x520) {
x17265 += 1;
} else {
}
if (x452) {
x17266 += 1;
} else {
}

}
x17257 += -7;
if (x452) {
x17258 += -7;
} else {
}
if (x452) {
x17259 += 1;
} else {
}

}
x17250 += -7;
if (x542) {
x17251 += -7;
} else {
}
if (x542) {
x17252 += 1;
} else {
}

}
x17243 += 7;
if (x452) {
x17244 += 7;
} else {
}
if (x452) {
x17245 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x17315 = (float*)myMalloc(7 * sizeof(float));;
int32_t x17316 = 0;
int32_t x17317 = 0;
int32_t x17318 = 0;
for(int x17319=0; x17319 < 1; x17319++) {
int32_t x17320 = x17317;
int32_t x17321 = x17318;
int32_t x17322 = x17316;
int32_t x17323 = x17322;
int32_t x17324 = x17320;
int32_t x17325 = x17321;
for(int x17326=0; x17326 < -1; x17326++) {
int32_t x17327 = x17324;
int32_t x17328 = x17325;
int32_t x17329 = x17323;
int32_t x17330 = x17329;
int32_t x17331 = x17327;
int32_t x17332 = x17328;
for(int x17333=0; x17333 < 1; x17333++) {
int32_t x17334 = x17331;
int32_t x17335 = x17332;
int32_t x17336 = x17330;
int32_t x17337 = x17336;
int32_t x17338 = x17334;
int32_t x17339 = x17335;
for(int x17340=0; x17340 < -7; x17340++) {
int32_t x17341 = x17337;
int32_t x17342 = x17338;
float x17343 = x17242[x17342];
int32_t x17344 = x17339;
float x17345 = x66[x17344];
float x17346 = x17343 + x17345;
x17315[x17341] = x17346;
x17337 += 1;
if (x520) {
x17338 += 1;
} else {
}
if (x452) {
x17339 += 1;
} else {
}

}
x17330 += -7;
if (x452) {
x17331 += -7;
} else {
}
if (x452) {
x17332 += 1;
} else {
}

}
x17323 += -7;
if (x542) {
x17324 += -7;
} else {
}
if (x542) {
x17325 += 1;
} else {
}

}
x17316 += 7;
if (x452) {
x17317 += 7;
} else {
}
if (x452) {
x17318 += -1;
} else {
}

}
float* x17384 = (float*)myMalloc(7 * sizeof(float));;
for(int x17385=0; x17385 < 7; x17385++) {
float x17386 = x17315[x17385];
bool x17387 = x17386 < 0.0f;
if (x17387) {
x17384[x17385] = 0.0f;
} else {
float x17390 = x17315[x17385];
x17384[x17385] = x17390;
}

}
float* x17396 = (float*)myMalloc(x16037 * sizeof(float));;
float* x17397 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x17398=0; x17398 < 1; x17398++) {
int32_t x17399 = x17398 * 7;
float* x17400 = x17384+x17399;
int32_t x17401 = x17398 * x16035;
float* x17402 = x17396+x17401;
int32_t x17403 = x17398 * x1520;
float* x17404 = x17397+x17403;
for(int x17405=0; x17405 < -1; x17405++) {
int32_t x17406 = x17405 / 1;
int32_t x17410 = x17406 * x1150;
int32_t x17411 = x17410 * x1152;
int32_t x17407 = x17405 % 1;
int32_t x17408 = x17407 / 1;
int32_t x17412 = x17408 * x1150;
int32_t x17413 = x17412 * x1152;
int32_t x17414 = x17411 + x17413;
int32_t x17409 = x17407 % 1;
int32_t x17415 = x17409 * x1152;
int32_t x17416 = x17415 * x1152;
int32_t x17417 = x17414 + x17416;
float* x17418 = x17404+x17417;
int32_t x17419 = x17406 * -7;
float* x17420 = x17400+x17419;
for(int x17421=0; x17421 < x1150; x17421++) {
int32_t x17423 = x17421 * x1152;
float* x17424 = x17418+x17423;
int32_t x17422 = x17421 + x17408;
int32_t x17425 = x17422 * -7;
int32_t x17426 = x17425 + x17409;
float* x17427 = x17420+x17426;
memcpy(x17424, x17427, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x1153,-1,1,x47,-1,x17404,x1153,1,x17402,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(2048) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x17440 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x17441 = 0;
int32_t x17442 = 0;
int32_t x17443 = 0;
for(int x17444=0; x17444 < -7; x17444++) {
int32_t x17445 = x17441;
int32_t x17446 = x17442;
float x17447 = x17396[x17446];
int32_t x17448 = x17443;
float x17449 = x68[x17448];
float x17450 = x17447 - x17449;
x17440[x17445] = x17450;
x17441 += 1;
if (x452) {
x17442 += x16035;
} else {
}
if (x452) {
x17443 += -1;
} else {
}

}
float* x17461 = (float*)myMalloc(2048 * sizeof(float));;
for(int x17462=0; x17462 < 2048; x17462++) {
float x17463 = x245[x17462];
float x17464 = x17463 + 1.0E-5f;
x17461[x17462] = x17464;

}
float* x17468 = (float*)myMalloc(2048 * sizeof(float));;
for(int x17469=0; x17469 < 2048; x17469++) {
float x17470 = x17461[x17469];
double x17471 = (double)x17470;
double x17472 = sqrt(x17471);
float x17473 = (float)x17472;
x17468[x17469] = x17473;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x17481 = (float*)myMalloc(7 * sizeof(float));;
int32_t x17482 = 0;
int32_t x17483 = 0;
int32_t x17484 = 0;
for(int x17485=0; x17485 < 1; x17485++) {
int32_t x17486 = x17483;
int32_t x17487 = x17484;
int32_t x17488 = x17482;
int32_t x17489 = x17488;
int32_t x17490 = x17486;
int32_t x17491 = x17487;
for(int x17492=0; x17492 < -1; x17492++) {
int32_t x17493 = x17490;
int32_t x17494 = x17491;
int32_t x17495 = x17489;
int32_t x17496 = x17495;
int32_t x17497 = x17493;
int32_t x17498 = x17494;
for(int x17499=0; x17499 < 1; x17499++) {
int32_t x17500 = x17497;
int32_t x17501 = x17498;
int32_t x17502 = x17496;
int32_t x17503 = x17502;
int32_t x17504 = x17500;
int32_t x17505 = x17501;
for(int x17506=0; x17506 < -7; x17506++) {
int32_t x17507 = x17503;
int32_t x17508 = x17504;
float x17509 = x17440[x17508];
int32_t x17510 = x17505;
float x17511 = x17468[x17510];
float x17512 = x17509 / x17511;
x17481[x17507] = x17512;
x17503 += 1;
if (x520) {
x17504 += 1;
} else {
}
if (x452) {
x17505 += 1;
} else {
}

}
x17496 += -7;
if (x452) {
x17497 += -7;
} else {
}
if (x452) {
x17498 += 1;
} else {
}

}
x17489 += -7;
if (x452) {
x17490 += -7;
} else {
}
if (x542) {
x17491 += 1;
} else {
}

}
x17482 += 7;
if (x452) {
x17483 += -7;
} else {
}
if (x452) {
x17484 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x17554 = (float*)myMalloc(7 * sizeof(float));;
int32_t x17555 = 0;
int32_t x17556 = 0;
int32_t x17557 = 0;
for(int x17558=0; x17558 < 1; x17558++) {
int32_t x17559 = x17556;
int32_t x17560 = x17557;
int32_t x17561 = x17555;
int32_t x17562 = x17561;
int32_t x17563 = x17559;
int32_t x17564 = x17560;
for(int x17565=0; x17565 < -1; x17565++) {
int32_t x17566 = x17563;
int32_t x17567 = x17564;
int32_t x17568 = x17562;
int32_t x17569 = x17568;
int32_t x17570 = x17566;
int32_t x17571 = x17567;
for(int x17572=0; x17572 < 1; x17572++) {
int32_t x17573 = x17570;
int32_t x17574 = x17571;
int32_t x17575 = x17569;
int32_t x17576 = x17575;
int32_t x17577 = x17573;
int32_t x17578 = x17574;
for(int x17579=0; x17579 < -7; x17579++) {
int32_t x17580 = x17576;
int32_t x17581 = x17577;
float x17582 = x17481[x17581];
int32_t x17583 = x17578;
float x17584 = x94[x17583];
float x17585 = x17582 * x17584;
x17554[x17580] = x17585;
x17576 += 1;
if (x520) {
x17577 += 1;
} else {
}
if (x452) {
x17578 += 1;
} else {
}

}
x17569 += -7;
if (x452) {
x17570 += -7;
} else {
}
if (x452) {
x17571 += 1;
} else {
}

}
x17562 += -7;
if (x542) {
x17563 += -7;
} else {
}
if (x542) {
x17564 += 1;
} else {
}

}
x17555 += 7;
if (x452) {
x17556 += 7;
} else {
}
if (x452) {
x17557 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x17627 = (float*)myMalloc(7 * sizeof(float));;
int32_t x17628 = 0;
int32_t x17629 = 0;
int32_t x17630 = 0;
for(int x17631=0; x17631 < 1; x17631++) {
int32_t x17632 = x17629;
int32_t x17633 = x17630;
int32_t x17634 = x17628;
int32_t x17635 = x17634;
int32_t x17636 = x17632;
int32_t x17637 = x17633;
for(int x17638=0; x17638 < -1; x17638++) {
int32_t x17639 = x17636;
int32_t x17640 = x17637;
int32_t x17641 = x17635;
int32_t x17642 = x17641;
int32_t x17643 = x17639;
int32_t x17644 = x17640;
for(int x17645=0; x17645 < 1; x17645++) {
int32_t x17646 = x17643;
int32_t x17647 = x17644;
int32_t x17648 = x17642;
int32_t x17649 = x17648;
int32_t x17650 = x17646;
int32_t x17651 = x17647;
for(int x17652=0; x17652 < -7; x17652++) {
int32_t x17653 = x17649;
int32_t x17654 = x17650;
float x17655 = x17554[x17654];
int32_t x17656 = x17651;
float x17657 = x144[x17656];
float x17658 = x17655 + x17657;
x17627[x17653] = x17658;
x17649 += 1;
if (x520) {
x17650 += 1;
} else {
}
if (x452) {
x17651 += 1;
} else {
}

}
x17642 += -7;
if (x452) {
x17643 += -7;
} else {
}
if (x452) {
x17644 += 1;
} else {
}

}
x17635 += -7;
if (x542) {
x17636 += -7;
} else {
}
if (x542) {
x17637 += 1;
} else {
}

}
x17628 += 7;
if (x452) {
x17629 += 7;
} else {
}
if (x452) {
x17630 += -1;
} else {
}

}
int32_t x17696 = 0;
int32_t x17697 = 0;
int32_t x17698 = 0;
for(int x17699=0; x17699 < 1; x17699++) {
int32_t x17700 = x17697;
int32_t x17701 = x17698;
int32_t x17702 = x17696;
int32_t x17703 = x17702;
int32_t x17704 = x17700;
int32_t x17705 = x17701;
for(int x17706=0; x17706 < -1; x17706++) {
int32_t x17707 = x17704;
int32_t x17708 = x17705;
int32_t x17709 = x17703;
int32_t x17710 = x17709;
int32_t x17711 = x17707;
int32_t x17712 = x17708;
for(int x17713=0; x17713 < 1; x17713++) {
int32_t x17714 = x17711;
int32_t x17715 = x17712;
int32_t x17716 = x17710;
int32_t x17717 = x17716;
int32_t x17718 = x17714;
int32_t x17719 = x17715;
for(int x17720=0; x17720 < -7; x17720++) {
int32_t x17721 = x17718;
float x17722 = x17627[x17721];
int32_t x17723 = x17719;
float x17724 = x16716[x17723];
float x17725 = x17722 + x17724;
x17627[x17721] = x17725;
x17717 += 1;
if (x520) {
x17718 += 1;
} else {
}
if (x520) {
x17719 += 1;
} else {
}

}
x17710 += -7;
if (x452) {
x17711 += -7;
} else {
}
if (x452) {
x17712 += -7;
} else {
}

}
x17703 += -7;
if (x542) {
x17704 += -7;
} else {
}
if (x542) {
x17705 += -7;
} else {
}

}
x17696 += 7;
if (x452) {
x17697 += 7;
} else {
}
if (x452) {
x17698 += 7;
} else {
}

}
float* x17763 = (float*)myMalloc(7 * sizeof(float));;
for(int x17764=0; x17764 < 7; x17764++) {
float x17765 = x17627[x17764];
bool x17766 = x17765 < 0.0f;
if (x17766) {
x17763[x17764] = 0.0f;
} else {
float x17769 = x17627[x17764];
x17763[x17764] = x17769;
}

}
float* x17775 = (float*)myMalloc(x4967 * sizeof(float));;
float* x17776 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x17777=0; x17777 < 1; x17777++) {
int32_t x17778 = x17777 * 7;
float* x17779 = x17763+x17778;
int32_t x17780 = x17777 * x4965;
float* x17781 = x17775+x17780;
int32_t x17782 = x17777 * x1520;
float* x17783 = x17776+x17782;
for(int x17784=0; x17784 < -1; x17784++) {
int32_t x17785 = x17784 / 1;
int32_t x17789 = x17785 * x1150;
int32_t x17790 = x17789 * x1152;
int32_t x17786 = x17784 % 1;
int32_t x17787 = x17786 / 1;
int32_t x17791 = x17787 * x1150;
int32_t x17792 = x17791 * x1152;
int32_t x17793 = x17790 + x17792;
int32_t x17788 = x17786 % 1;
int32_t x17794 = x17788 * x1152;
int32_t x17795 = x17794 * x1152;
int32_t x17796 = x17793 + x17795;
float* x17797 = x17783+x17796;
int32_t x17798 = x17785 * -7;
float* x17799 = x17779+x17798;
for(int x17800=0; x17800 < x1150; x17800++) {
int32_t x17802 = x17800 * x1152;
float* x17803 = x17797+x17802;
int32_t x17801 = x17800 + x17787;
int32_t x17804 = x17801 * -7;
int32_t x17805 = x17804 + x17788;
float* x17806 = x17799+x17805;
memcpy(x17803, x17806, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x1153,-1,1,x265,-1,x17783,x1153,1,x17781,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(512) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x17819 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x17820 = 0;
int32_t x17821 = 0;
int32_t x17822 = 0;
for(int x17823=0; x17823 < -7; x17823++) {
int32_t x17824 = x17820;
int32_t x17825 = x17821;
float x17826 = x17775[x17825];
int32_t x17827 = x17822;
float x17828 = x213[x17827];
float x17829 = x17826 - x17828;
x17819[x17824] = x17829;
x17820 += 1;
if (x452) {
x17821 += x4965;
} else {
}
if (x452) {
x17822 += -1;
} else {
}

}
float* x17840 = (float*)myMalloc(512 * sizeof(float));;
for(int x17841=0; x17841 < 512; x17841++) {
float x17842 = x255[x17841];
float x17843 = x17842 + 1.0E-5f;
x17840[x17841] = x17843;

}
float* x17847 = (float*)myMalloc(512 * sizeof(float));;
for(int x17848=0; x17848 < 512; x17848++) {
float x17849 = x17840[x17848];
double x17850 = (double)x17849;
double x17851 = sqrt(x17850);
float x17852 = (float)x17851;
x17847[x17848] = x17852;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x17860 = (float*)myMalloc(7 * sizeof(float));;
int32_t x17861 = 0;
int32_t x17862 = 0;
int32_t x17863 = 0;
for(int x17864=0; x17864 < 1; x17864++) {
int32_t x17865 = x17862;
int32_t x17866 = x17863;
int32_t x17867 = x17861;
int32_t x17868 = x17867;
int32_t x17869 = x17865;
int32_t x17870 = x17866;
for(int x17871=0; x17871 < -1; x17871++) {
int32_t x17872 = x17869;
int32_t x17873 = x17870;
int32_t x17874 = x17868;
int32_t x17875 = x17874;
int32_t x17876 = x17872;
int32_t x17877 = x17873;
for(int x17878=0; x17878 < 1; x17878++) {
int32_t x17879 = x17876;
int32_t x17880 = x17877;
int32_t x17881 = x17875;
int32_t x17882 = x17881;
int32_t x17883 = x17879;
int32_t x17884 = x17880;
for(int x17885=0; x17885 < -7; x17885++) {
int32_t x17886 = x17882;
int32_t x17887 = x17883;
float x17888 = x17819[x17887];
int32_t x17889 = x17884;
float x17890 = x17847[x17889];
float x17891 = x17888 / x17890;
x17860[x17886] = x17891;
x17882 += 1;
if (x520) {
x17883 += 1;
} else {
}
if (x452) {
x17884 += 1;
} else {
}

}
x17875 += -7;
if (x452) {
x17876 += -7;
} else {
}
if (x452) {
x17877 += 1;
} else {
}

}
x17868 += -7;
if (x452) {
x17869 += -7;
} else {
}
if (x542) {
x17870 += 1;
} else {
}

}
x17861 += 7;
if (x452) {
x17862 += -7;
} else {
}
if (x452) {
x17863 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x17933 = (float*)myMalloc(7 * sizeof(float));;
int32_t x17934 = 0;
int32_t x17935 = 0;
int32_t x17936 = 0;
for(int x17937=0; x17937 < 1; x17937++) {
int32_t x17938 = x17935;
int32_t x17939 = x17936;
int32_t x17940 = x17934;
int32_t x17941 = x17940;
int32_t x17942 = x17938;
int32_t x17943 = x17939;
for(int x17944=0; x17944 < -1; x17944++) {
int32_t x17945 = x17942;
int32_t x17946 = x17943;
int32_t x17947 = x17941;
int32_t x17948 = x17947;
int32_t x17949 = x17945;
int32_t x17950 = x17946;
for(int x17951=0; x17951 < 1; x17951++) {
int32_t x17952 = x17949;
int32_t x17953 = x17950;
int32_t x17954 = x17948;
int32_t x17955 = x17954;
int32_t x17956 = x17952;
int32_t x17957 = x17953;
for(int x17958=0; x17958 < -7; x17958++) {
int32_t x17959 = x17955;
int32_t x17960 = x17956;
float x17961 = x17860[x17960];
int32_t x17962 = x17957;
float x17963 = x15[x17962];
float x17964 = x17961 * x17963;
x17933[x17959] = x17964;
x17955 += 1;
if (x520) {
x17956 += 1;
} else {
}
if (x452) {
x17957 += 1;
} else {
}

}
x17948 += -7;
if (x452) {
x17949 += -7;
} else {
}
if (x452) {
x17950 += 1;
} else {
}

}
x17941 += -7;
if (x542) {
x17942 += -7;
} else {
}
if (x542) {
x17943 += 1;
} else {
}

}
x17934 += 7;
if (x452) {
x17935 += 7;
} else {
}
if (x452) {
x17936 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x18006 = (float*)myMalloc(7 * sizeof(float));;
int32_t x18007 = 0;
int32_t x18008 = 0;
int32_t x18009 = 0;
for(int x18010=0; x18010 < 1; x18010++) {
int32_t x18011 = x18008;
int32_t x18012 = x18009;
int32_t x18013 = x18007;
int32_t x18014 = x18013;
int32_t x18015 = x18011;
int32_t x18016 = x18012;
for(int x18017=0; x18017 < -1; x18017++) {
int32_t x18018 = x18015;
int32_t x18019 = x18016;
int32_t x18020 = x18014;
int32_t x18021 = x18020;
int32_t x18022 = x18018;
int32_t x18023 = x18019;
for(int x18024=0; x18024 < 1; x18024++) {
int32_t x18025 = x18022;
int32_t x18026 = x18023;
int32_t x18027 = x18021;
int32_t x18028 = x18027;
int32_t x18029 = x18025;
int32_t x18030 = x18026;
for(int x18031=0; x18031 < -7; x18031++) {
int32_t x18032 = x18028;
int32_t x18033 = x18029;
float x18034 = x17933[x18033];
int32_t x18035 = x18030;
float x18036 = x78[x18035];
float x18037 = x18034 + x18036;
x18006[x18032] = x18037;
x18028 += 1;
if (x520) {
x18029 += 1;
} else {
}
if (x452) {
x18030 += 1;
} else {
}

}
x18021 += -7;
if (x452) {
x18022 += -7;
} else {
}
if (x452) {
x18023 += 1;
} else {
}

}
x18014 += -7;
if (x542) {
x18015 += -7;
} else {
}
if (x542) {
x18016 += 1;
} else {
}

}
x18007 += 7;
if (x452) {
x18008 += 7;
} else {
}
if (x452) {
x18009 += -1;
} else {
}

}
float* x18075 = (float*)myMalloc(7 * sizeof(float));;
for(int x18076=0; x18076 < 7; x18076++) {
float x18077 = x18006[x18076];
bool x18078 = x18077 < 0.0f;
if (x18078) {
x18075[x18076] = 0.0f;
} else {
float x18081 = x18006[x18076];
x18075[x18076] = x18081;
}

}
float* x18087 = (float*)myMalloc(x4967 * sizeof(float));;
float* x18088 = (float*)myMalloc(x1158 * sizeof(float));;
for(int x18089=0; x18089 < 1; x18089++) {
int32_t x18090 = x18089 * 7;
float* x18091 = x18075+x18090;
int32_t x18092 = x18089 * x4965;
float* x18093 = x18087+x18092;
int32_t x18094 = x18089 * x1158;
float* x18095 = x18088+x18094;
for(int x18096=0; x18096 < -9; x18096++) {
int32_t x18097 = x18096 / 9;
int32_t x18101 = x18097 * 3;
int32_t x18102 = x18101 * 3;
int32_t x18103 = x18102 * x1150;
int32_t x18104 = x18103 * x1152;
int32_t x18098 = x18096 % 9;
int32_t x18099 = x18098 / 3;
int32_t x18105 = x18099 * 3;
int32_t x18106 = x18105 * x1150;
int32_t x18107 = x18106 * x1152;
int32_t x18108 = x18104 + x18107;
int32_t x18100 = x18098 % 3;
int32_t x18109 = x18100 * x1152;
int32_t x18110 = x18109 * x1152;
int32_t x18111 = x18108 + x18110;
float* x18112 = x18095+x18111;
int32_t x18113 = x18097 * -7;
float* x18114 = x18091+x18113;
int32_t x18126 = 1 - x18100;
bool x18127 = x18126 > 0;
int32_t x18128;
if (x18127) {
x18128 = x18126;
} else {
x18128 = 0;
}
int32_t x18129 = 3 - x18100;
int32_t x18130 = x18129 - 1;
int32_t x18131 = 1 - x18130;
bool x18132 = x18131 > 0;
int32_t x18133;
if (x18132) {
x18133 = x18131;
} else {
x18133 = 0;
}
int32_t x18134 = x1152 - x18133;
int32_t x18135 = x18134 - x18128;
bool x18136 = x18135 <= 0;
bool x18140 = x18128 > 0;
int32_t x18125 = -1 + x18100;
bool x18153 = x18133 > 0;
for(int x18115=0; x18115 < x1150; x18115++) {
int32_t x18116 = x18115 - 1;
int32_t x18117 = x18116 + x18099;
bool x18118 = x18117 < 0;
bool x18119 = x18117 >= 1;
bool x18120 = x18118 || x18119;
if (x18120) {
int32_t x18121 = x18115 * x1152;
float* x18122 = x18112+x18121;
memset(x18122, 0, 4 * x1152);;
} else {
if (x18136) {
int32_t x18121 = x18115 * x1152;
float* x18137 = x18112+x18121;
memset(x18137, 0, 4 * x1152);;
} else {
int32_t x18121 = x18115 * x1152;
if (x18140) {
float* x18141 = x18112+x18121;
memset(x18141, 0, 4 * x18128);;
} else {
}
// may have segfault here
int32_t x18146 = x18121 + x18128;
float* x18147 = x18112+x18146;
int32_t x18148 = x18117 * -7;
int32_t x18149 = x18148 + x18125;
int32_t x18150 = x18149 + x18128;
float* x18151 = x18114+x18150;
memcpy(x18147, x18151, 4 * x18135);;
if (x18153) {
int32_t x18154 = x18121 + x1152;
int32_t x18155 = x18154 - x18133;
float* x18156 = x18112+x18155;
memset(x18156, 0, 4 * x18133);;
} else {
}
}
}

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 512,x1153,-9,1,x28,-9,x18095,x1153,1,x18093,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(512) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x18175 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x18176 = 0;
int32_t x18177 = 0;
int32_t x18178 = 0;
for(int x18179=0; x18179 < -7; x18179++) {
int32_t x18180 = x18176;
int32_t x18181 = x18177;
float x18182 = x18087[x18181];
int32_t x18183 = x18178;
float x18184 = x12[x18183];
float x18185 = x18182 - x18184;
x18175[x18180] = x18185;
x18176 += 1;
if (x452) {
x18177 += x4965;
} else {
}
if (x452) {
x18178 += -1;
} else {
}

}
float* x18196 = (float*)myMalloc(512 * sizeof(float));;
for(int x18197=0; x18197 < 512; x18197++) {
float x18198 = x202[x18197];
float x18199 = x18198 + 1.0E-5f;
x18196[x18197] = x18199;

}
float* x18203 = (float*)myMalloc(512 * sizeof(float));;
for(int x18204=0; x18204 < 512; x18204++) {
float x18205 = x18196[x18204];
double x18206 = (double)x18205;
double x18207 = sqrt(x18206);
float x18208 = (float)x18207;
x18203[x18204] = x18208;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x18216 = (float*)myMalloc(7 * sizeof(float));;
int32_t x18217 = 0;
int32_t x18218 = 0;
int32_t x18219 = 0;
for(int x18220=0; x18220 < 1; x18220++) {
int32_t x18221 = x18218;
int32_t x18222 = x18219;
int32_t x18223 = x18217;
int32_t x18224 = x18223;
int32_t x18225 = x18221;
int32_t x18226 = x18222;
for(int x18227=0; x18227 < -1; x18227++) {
int32_t x18228 = x18225;
int32_t x18229 = x18226;
int32_t x18230 = x18224;
int32_t x18231 = x18230;
int32_t x18232 = x18228;
int32_t x18233 = x18229;
for(int x18234=0; x18234 < 1; x18234++) {
int32_t x18235 = x18232;
int32_t x18236 = x18233;
int32_t x18237 = x18231;
int32_t x18238 = x18237;
int32_t x18239 = x18235;
int32_t x18240 = x18236;
for(int x18241=0; x18241 < -7; x18241++) {
int32_t x18242 = x18238;
int32_t x18243 = x18239;
float x18244 = x18175[x18243];
int32_t x18245 = x18240;
float x18246 = x18203[x18245];
float x18247 = x18244 / x18246;
x18216[x18242] = x18247;
x18238 += 1;
if (x520) {
x18239 += 1;
} else {
}
if (x452) {
x18240 += 1;
} else {
}

}
x18231 += -7;
if (x452) {
x18232 += -7;
} else {
}
if (x452) {
x18233 += 1;
} else {
}

}
x18224 += -7;
if (x452) {
x18225 += -7;
} else {
}
if (x542) {
x18226 += 1;
} else {
}

}
x18217 += 7;
if (x452) {
x18218 += -7;
} else {
}
if (x452) {
x18219 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x18289 = (float*)myMalloc(7 * sizeof(float));;
int32_t x18290 = 0;
int32_t x18291 = 0;
int32_t x18292 = 0;
for(int x18293=0; x18293 < 1; x18293++) {
int32_t x18294 = x18291;
int32_t x18295 = x18292;
int32_t x18296 = x18290;
int32_t x18297 = x18296;
int32_t x18298 = x18294;
int32_t x18299 = x18295;
for(int x18300=0; x18300 < -1; x18300++) {
int32_t x18301 = x18298;
int32_t x18302 = x18299;
int32_t x18303 = x18297;
int32_t x18304 = x18303;
int32_t x18305 = x18301;
int32_t x18306 = x18302;
for(int x18307=0; x18307 < 1; x18307++) {
int32_t x18308 = x18305;
int32_t x18309 = x18306;
int32_t x18310 = x18304;
int32_t x18311 = x18310;
int32_t x18312 = x18308;
int32_t x18313 = x18309;
for(int x18314=0; x18314 < -7; x18314++) {
int32_t x18315 = x18311;
int32_t x18316 = x18312;
float x18317 = x18216[x18316];
int32_t x18318 = x18313;
float x18319 = x194[x18318];
float x18320 = x18317 * x18319;
x18289[x18315] = x18320;
x18311 += 1;
if (x520) {
x18312 += 1;
} else {
}
if (x452) {
x18313 += 1;
} else {
}

}
x18304 += -7;
if (x452) {
x18305 += -7;
} else {
}
if (x452) {
x18306 += 1;
} else {
}

}
x18297 += -7;
if (x542) {
x18298 += -7;
} else {
}
if (x542) {
x18299 += 1;
} else {
}

}
x18290 += 7;
if (x452) {
x18291 += 7;
} else {
}
if (x452) {
x18292 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x18362 = (float*)myMalloc(7 * sizeof(float));;
int32_t x18363 = 0;
int32_t x18364 = 0;
int32_t x18365 = 0;
for(int x18366=0; x18366 < 1; x18366++) {
int32_t x18367 = x18364;
int32_t x18368 = x18365;
int32_t x18369 = x18363;
int32_t x18370 = x18369;
int32_t x18371 = x18367;
int32_t x18372 = x18368;
for(int x18373=0; x18373 < -1; x18373++) {
int32_t x18374 = x18371;
int32_t x18375 = x18372;
int32_t x18376 = x18370;
int32_t x18377 = x18376;
int32_t x18378 = x18374;
int32_t x18379 = x18375;
for(int x18380=0; x18380 < 1; x18380++) {
int32_t x18381 = x18378;
int32_t x18382 = x18379;
int32_t x18383 = x18377;
int32_t x18384 = x18383;
int32_t x18385 = x18381;
int32_t x18386 = x18382;
for(int x18387=0; x18387 < -7; x18387++) {
int32_t x18388 = x18384;
int32_t x18389 = x18385;
float x18390 = x18289[x18389];
int32_t x18391 = x18386;
float x18392 = x169[x18391];
float x18393 = x18390 + x18392;
x18362[x18388] = x18393;
x18384 += 1;
if (x520) {
x18385 += 1;
} else {
}
if (x452) {
x18386 += 1;
} else {
}

}
x18377 += -7;
if (x452) {
x18378 += -7;
} else {
}
if (x452) {
x18379 += 1;
} else {
}

}
x18370 += -7;
if (x542) {
x18371 += -7;
} else {
}
if (x542) {
x18372 += 1;
} else {
}

}
x18363 += 7;
if (x452) {
x18364 += 7;
} else {
}
if (x452) {
x18365 += -1;
} else {
}

}
float* x18431 = (float*)myMalloc(7 * sizeof(float));;
for(int x18432=0; x18432 < 7; x18432++) {
float x18433 = x18362[x18432];
bool x18434 = x18433 < 0.0f;
if (x18434) {
x18431[x18432] = 0.0f;
} else {
float x18437 = x18362[x18432];
x18431[x18432] = x18437;
}

}
float* x18443 = (float*)myMalloc(x16037 * sizeof(float));;
float* x18444 = (float*)myMalloc(x1520 * sizeof(float));;
for(int x18445=0; x18445 < 1; x18445++) {
int32_t x18446 = x18445 * 7;
float* x18447 = x18431+x18446;
int32_t x18448 = x18445 * x16035;
float* x18449 = x18443+x18448;
int32_t x18450 = x18445 * x1520;
float* x18451 = x18444+x18450;
for(int x18452=0; x18452 < -1; x18452++) {
int32_t x18453 = x18452 / 1;
int32_t x18457 = x18453 * x1150;
int32_t x18458 = x18457 * x1152;
int32_t x18454 = x18452 % 1;
int32_t x18455 = x18454 / 1;
int32_t x18459 = x18455 * x1150;
int32_t x18460 = x18459 * x1152;
int32_t x18461 = x18458 + x18460;
int32_t x18456 = x18454 % 1;
int32_t x18462 = x18456 * x1152;
int32_t x18463 = x18462 * x1152;
int32_t x18464 = x18461 + x18463;
float* x18465 = x18451+x18464;
int32_t x18466 = x18453 * -7;
float* x18467 = x18447+x18466;
for(int x18468=0; x18468 < x1150; x18468++) {
int32_t x18470 = x18468 * x1152;
float* x18471 = x18465+x18470;
int32_t x18469 = x18468 + x18455;
int32_t x18472 = x18469 * -7;
int32_t x18473 = x18472 + x18456;
float* x18474 = x18467+x18473;
memcpy(x18471, x18474, 4 * x1152);;

}

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2048,x1153,-1,1,x33,-1,x18451,x1153,1,x18449,x1153);

}
if (x428) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(2048) x Sym(1150) x Sym(1152)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x18487 = (float*)myMalloc(-7 * sizeof(float));;
int32_t x18488 = 0;
int32_t x18489 = 0;
int32_t x18490 = 0;
for(int x18491=0; x18491 < -7; x18491++) {
int32_t x18492 = x18488;
int32_t x18493 = x18489;
float x18494 = x18443[x18493];
int32_t x18495 = x18490;
float x18496 = x260[x18495];
float x18497 = x18494 - x18496;
x18487[x18492] = x18497;
x18488 += 1;
if (x452) {
x18489 += x16035;
} else {
}
if (x452) {
x18490 += -1;
} else {
}

}
float* x18508 = (float*)myMalloc(2048 * sizeof(float));;
for(int x18509=0; x18509 < 2048; x18509++) {
float x18510 = x123[x18509];
float x18511 = x18510 + 1.0E-5f;
x18508[x18509] = x18511;

}
float* x18515 = (float*)myMalloc(2048 * sizeof(float));;
for(int x18516=0; x18516 < 2048; x18516++) {
float x18517 = x18508[x18516];
double x18518 = (double)x18517;
double x18519 = sqrt(x18518);
float x18520 = (float)x18519;
x18515[x18516] = x18520;

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x18528 = (float*)myMalloc(7 * sizeof(float));;
int32_t x18529 = 0;
int32_t x18530 = 0;
int32_t x18531 = 0;
for(int x18532=0; x18532 < 1; x18532++) {
int32_t x18533 = x18530;
int32_t x18534 = x18531;
int32_t x18535 = x18529;
int32_t x18536 = x18535;
int32_t x18537 = x18533;
int32_t x18538 = x18534;
for(int x18539=0; x18539 < -1; x18539++) {
int32_t x18540 = x18537;
int32_t x18541 = x18538;
int32_t x18542 = x18536;
int32_t x18543 = x18542;
int32_t x18544 = x18540;
int32_t x18545 = x18541;
for(int x18546=0; x18546 < 1; x18546++) {
int32_t x18547 = x18544;
int32_t x18548 = x18545;
int32_t x18549 = x18543;
int32_t x18550 = x18549;
int32_t x18551 = x18547;
int32_t x18552 = x18548;
for(int x18553=0; x18553 < -7; x18553++) {
int32_t x18554 = x18550;
int32_t x18555 = x18551;
float x18556 = x18487[x18555];
int32_t x18557 = x18552;
float x18558 = x18515[x18557];
float x18559 = x18556 / x18558;
x18528[x18554] = x18559;
x18550 += 1;
if (x520) {
x18551 += 1;
} else {
}
if (x452) {
x18552 += 1;
} else {
}

}
x18543 += -7;
if (x452) {
x18544 += -7;
} else {
}
if (x452) {
x18545 += 1;
} else {
}

}
x18536 += -7;
if (x452) {
x18537 += -7;
} else {
}
if (x542) {
x18538 += 1;
} else {
}

}
x18529 += 7;
if (x452) {
x18530 += -7;
} else {
}
if (x452) {
x18531 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x18601 = (float*)myMalloc(7 * sizeof(float));;
int32_t x18602 = 0;
int32_t x18603 = 0;
int32_t x18604 = 0;
for(int x18605=0; x18605 < 1; x18605++) {
int32_t x18606 = x18603;
int32_t x18607 = x18604;
int32_t x18608 = x18602;
int32_t x18609 = x18608;
int32_t x18610 = x18606;
int32_t x18611 = x18607;
for(int x18612=0; x18612 < -1; x18612++) {
int32_t x18613 = x18610;
int32_t x18614 = x18611;
int32_t x18615 = x18609;
int32_t x18616 = x18615;
int32_t x18617 = x18613;
int32_t x18618 = x18614;
for(int x18619=0; x18619 < 1; x18619++) {
int32_t x18620 = x18617;
int32_t x18621 = x18618;
int32_t x18622 = x18616;
int32_t x18623 = x18622;
int32_t x18624 = x18620;
int32_t x18625 = x18621;
for(int x18626=0; x18626 < -7; x18626++) {
int32_t x18627 = x18623;
int32_t x18628 = x18624;
float x18629 = x18528[x18628];
int32_t x18630 = x18625;
float x18631 = x103[x18630];
float x18632 = x18629 * x18631;
x18601[x18627] = x18632;
x18623 += 1;
if (x520) {
x18624 += 1;
} else {
}
if (x452) {
x18625 += 1;
} else {
}

}
x18616 += -7;
if (x452) {
x18617 += -7;
} else {
}
if (x452) {
x18618 += 1;
} else {
}

}
x18609 += -7;
if (x542) {
x18610 += -7;
} else {
}
if (x542) {
x18611 += 1;
} else {
}

}
x18602 += 7;
if (x452) {
x18603 += 7;
} else {
}
if (x452) {
x18604 += -1;
} else {
}

}
if (x478) {
} else {
printf("broadcasting dim not match %s %s\n"," x Const(1) x Const(-1) x Const(1) x Const(-7)"," x Const(1) x Const(-1) x Const(1) x Const(1)");
assert(false && "");
}
float* x18674 = (float*)myMalloc(7 * sizeof(float));;
int32_t x18675 = 0;
int32_t x18676 = 0;
int32_t x18677 = 0;
for(int x18678=0; x18678 < 1; x18678++) {
int32_t x18679 = x18676;
int32_t x18680 = x18677;
int32_t x18681 = x18675;
int32_t x18682 = x18681;
int32_t x18683 = x18679;
int32_t x18684 = x18680;
for(int x18685=0; x18685 < -1; x18685++) {
int32_t x18686 = x18683;
int32_t x18687 = x18684;
int32_t x18688 = x18682;
int32_t x18689 = x18688;
int32_t x18690 = x18686;
int32_t x18691 = x18687;
for(int x18692=0; x18692 < 1; x18692++) {
int32_t x18693 = x18690;
int32_t x18694 = x18691;
int32_t x18695 = x18689;
int32_t x18696 = x18695;
int32_t x18697 = x18693;
int32_t x18698 = x18694;
for(int x18699=0; x18699 < -7; x18699++) {
int32_t x18700 = x18696;
int32_t x18701 = x18697;
float x18702 = x18601[x18701];
int32_t x18703 = x18698;
float x18704 = x181[x18703];
float x18705 = x18702 + x18704;
x18674[x18700] = x18705;
x18696 += 1;
if (x520) {
x18697 += 1;
} else {
}
if (x452) {
x18698 += 1;
} else {
}

}
x18689 += -7;
if (x452) {
x18690 += -7;
} else {
}
if (x452) {
x18691 += 1;
} else {
}

}
x18682 += -7;
if (x542) {
x18683 += -7;
} else {
}
if (x542) {
x18684 += 1;
} else {
}

}
x18675 += 7;
if (x452) {
x18676 += 7;
} else {
}
if (x452) {
x18677 += -1;
} else {
}

}
int32_t x18743 = 0;
int32_t x18744 = 0;
int32_t x18745 = 0;
for(int x18746=0; x18746 < 1; x18746++) {
int32_t x18747 = x18744;
int32_t x18748 = x18745;
int32_t x18749 = x18743;
int32_t x18750 = x18749;
int32_t x18751 = x18747;
int32_t x18752 = x18748;
for(int x18753=0; x18753 < -1; x18753++) {
int32_t x18754 = x18751;
int32_t x18755 = x18752;
int32_t x18756 = x18750;
int32_t x18757 = x18756;
int32_t x18758 = x18754;
int32_t x18759 = x18755;
for(int x18760=0; x18760 < 1; x18760++) {
int32_t x18761 = x18758;
int32_t x18762 = x18759;
int32_t x18763 = x18757;
int32_t x18764 = x18763;
int32_t x18765 = x18761;
int32_t x18766 = x18762;
for(int x18767=0; x18767 < -7; x18767++) {
int32_t x18768 = x18765;
float x18769 = x18674[x18768];
int32_t x18770 = x18766;
float x18771 = x17763[x18770];
float x18772 = x18769 + x18771;
x18674[x18768] = x18772;
x18764 += 1;
if (x520) {
x18765 += 1;
} else {
}
if (x520) {
x18766 += 1;
} else {
}

}
x18757 += -7;
if (x452) {
x18758 += -7;
} else {
}
if (x452) {
x18759 += -7;
} else {
}

}
x18750 += -7;
if (x542) {
x18751 += -7;
} else {
}
if (x542) {
x18752 += -7;
} else {
}

}
x18743 += 7;
if (x452) {
x18744 += 7;
} else {
}
if (x452) {
x18745 += 7;
} else {
}

}
float* x18810 = (float*)myMalloc(7 * sizeof(float));;
for(int x18811=0; x18811 < 7; x18811++) {
float x18812 = x18674[x18811];
bool x18813 = x18812 < 0.0f;
if (x18813) {
x18810[x18811] = 0.0f;
} else {
float x18816 = x18674[x18811];
x18810[x18811] = x18816;
}

}
if (x718) {
} else {
assert(false && "Image too small for averagePool_batch:  x Const(1) x Const(-1) x Const(1) x Const(-7)|(2,2)");
}
float* x18833 = (float*)myMalloc(x18832 * sizeof(float));;
for(int x18834=0; x18834 < 1; x18834++) {
int32_t x18835 = x18834 * 7;
float* x18836 = x18810+x18835;
int32_t x18837 = x18834 * x18830;
float* x18838 = x18833+x18837;
for(int x18839=0; x18839 < -1; x18839++) {
int32_t x18848 = x18839 * -7;
int32_t x18844 = x18839 * x18829;
for(int x18841=0; x18841 < x18826; x18841++) {
int32_t x18849 = x18841 * -7;
int32_t x18850 = x18848 + x18849;
int32_t x18845 = x18841 * x18828;
int32_t x18846 = x18844 + x18845;
for(int x18843=0; x18843 < x18828; x18843++) {
float x18852 = 0.0f;
int32_t x18851 = x18850 + x18843;
float x18853 = x18836[x18851];
x18852 += x18853;
int32_t x18855 = x18851 + 1;
float x18856 = x18836[x18855];
x18852 += x18856;
int32_t x18858 = x18851 + -7;
float x18859 = x18836[x18858];
x18852 += x18859;
int32_t x18861 = x18858 + 1;
float x18862 = x18836[x18861];
x18852 += x18862;
float x18864 = x18852;
int32_t x18847 = x18846 + x18843;
float x18865 = x18864 / 4.0f;
x18838[x18847] = x18865;

}

}

}

}
// gemm: List(Const(1), Const(2048)), Vector(Const(10), Const(2048))
float* x18876 = (float*)myMalloc(10 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1,10,2048,1.0,x18833,2048,x227,2048,0,x18876,10);
int32_t x18878 = 0;
int32_t x18879 = 0;
int32_t x18880 = 0;
for(int x18881=0; x18881 < 1; x18881++) {
int32_t x18882 = x18879;
int32_t x18883 = x18880;
int32_t x18884 = x18878;
int32_t x18885 = x18884;
int32_t x18886 = x18882;
int32_t x18887 = x18883;
for(int x18888=0; x18888 < 10; x18888++) {
int32_t x18889 = x18886;
float x18890 = x18876[x18889];
int32_t x18891 = x18887;
float x18892 = x48[x18891];
float x18893 = x18890 + x18892;
x18876[x18889] = x18893;
x18885 += 1;
if (x18896) {
x18886 += 1;
} else {
}
if (x18896) {
x18887 += 1;
} else {
}

}
x18878 += 10;
if (x452) {
x18879 += 10;
} else {
}
if (x452) {
x18880 += 10;
} else {
}

}
printf("output (size Const(1) x Const(10))\n");
float x18915 = 0.0f;
for(int x18916=0; x18916 < 10; x18916++) {
float x18917 = x18915;
float x18918 = x18876[x18916];
float x18919 = fabs(x18918);
float x18920 = fabs(x18917);
bool x18921 = x18919 > x18920;
float x18922;
if (x18921) {
x18922 = x18918;
} else {
x18922 = x18917;
}
x18915 = x18922;

}
float x18926 = x18915;
printf("Max Abs: %.5f || ",x18926);
for(int x18928=0; x18928 < 10; x18928++) {
float x18929 = x18876[x18928];
printf("%.5f ",x18929);

}
printf("\n");
assert(false && "stop");

}
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

