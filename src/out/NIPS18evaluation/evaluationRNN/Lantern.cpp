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
int32_t x3 = open("graham.txt",0);
int64_t x4 = fsize(x3);
int32_t x5 = (int32_t)x4;
int* x8 = (int32_t*)myMalloc(x5 * sizeof(int32_t));;
int64_t x6 = (int64_t)x5;
char* x7 = (char*)mmap(0, x6, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x3, 0);
for(int x10=0; x10 < x5; x10++) {
char x11 = x7[x10];
int32_t x12 = (int32_t ) x11;
int32_t x13 = x12 - 96;
x8[x10] = x13;

}
float* x17 = (float*)myMalloc(1300 * sizeof(float));;
for(int x19=0; x19 < 1300; x19++) {
float x20 = (float)rand()/RAND_MAX;
float x21 = x20 - 0.5f;
float x22 = x21 * 0.19611613f;
x17[x19] = x22;

}
float* x27 = (float*)myMalloc(1300 * sizeof(float));;
float* x28 = (float*)myMalloc(2500 * sizeof(float));;
for(int x30=0; x30 < 2500; x30++) {
float x31 = (float)rand()/RAND_MAX;
float x32 = x31 - 0.5f;
float x33 = x32 * 0.14142136f;
x28[x30] = x33;

}
float* x37 = (float*)myMalloc(2500 * sizeof(float));;
float* x38 = (float*)myMalloc(50 * sizeof(float));;
float* x39 = (float*)myMalloc(50 * sizeof(float));;
float* x40 = (float*)myMalloc(1300 * sizeof(float));;
for(int x41=0; x41 < 1300; x41++) {
float x42 = (float)rand()/RAND_MAX;
float x43 = x42 - 0.5f;
float x44 = x43 * 0.14142136f;
x40[x41] = x44;

}
float* x48 = (float*)myMalloc(1300 * sizeof(float));;
float* x49 = (float*)myMalloc(26 * sizeof(float));;
float* x50 = (float*)myMalloc(26 * sizeof(float));;
float* x51 = (float*)myMalloc(26 * sizeof(float));;
float* x52 = (float*)myMalloc(1300 * sizeof(float));;
float* x53 = (float*)myMalloc(2500 * sizeof(float));;
float* x54 = (float*)myMalloc(50 * sizeof(float));;
float* x55 = (float*)myMalloc(1300 * sizeof(float));;
double* x56 = (double*)myMalloc(51 * sizeof(double));;
double x57 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x58 = (long)mallocAddr;
int32_t x59 = 0;
x59 -= 400;
bool x304 = true || true;
bool x305 = x304 || true;
bool x552 = true || false;
for(int x62=0; x62 < 5001; x62++) {
float* x87 = (float*)myMalloc(1 * sizeof(float));;
float* x88 = (float*)myMalloc(10400 * sizeof(float));;
float* x105 = (float*)myMalloc(10400 * sizeof(float));;
int* x72 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
function<void(int32_t,float**)> x321 = [&](int32_t x322,float** x323) {
float** x325 = x323;
float* x326 = x325[0];
float* x327 = x325[1];
float* x328 = x325[2];
float* x329 = x325[3];
int32_t x324 = x322;
bool x330 = x324 < 20;
if (x330) {
int32_t x331 = x324 * 520;
float* x332 = x88+x331;
float* x333 = x105+x331;
float* x334 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x332,26,x17,50,0,x334,50);
float* x336 = (float*)myMalloc(1000 * sizeof(float));;
float* x337 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x328,50,x28,50,0,x337,50);
float* x339 = (float*)myMalloc(1000 * sizeof(float));;
float* x340 = (float*)myMalloc(1000 * sizeof(float));;
for(int x341=0; x341 < 20; x341++) {
int32_t x343 = 50 * x341;
for(int x342=0; x342 < 50; x342++) {
int32_t x345 = x343 + x342;
float x346 = x334[x345];
float x347 = x337[x345];
int32_t x344 = x342 + x343;
float x348 = x346 + x347;
x340[x344] = x348;

}

}
float* x354 = (float*)myMalloc(1000 * sizeof(float));;
float* x355 = (float*)myMalloc(1000 * sizeof(float));;
for(int x356=0; x356 < 20; x356++) {
int32_t x358 = 50 * x356;
for(int x357=0; x357 < 50; x357++) {
int32_t x360 = x358 + x357;
float x361 = x340[x360];
float x362 = x38[x357];
int32_t x359 = x357 + x358;
float x363 = x361 + x362;
x355[x359] = x363;

}

}
float* x369 = (float*)myMalloc(1000 * sizeof(float));;
float* x370 = (float*)myMalloc(1000 * sizeof(float));;
for(int x371=0; x371 < 1000; x371++) {
float x372 = x355[x371];
double x373 = (double)x372;
double x374 = tanh(x373);
float x375 = (float)x374;
x370[x371] = x375;

}
float* x379 = (float*)myMalloc(1000 * sizeof(float));;
float* x380 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x370,50,x40,26,0,x380,26);
float* x382 = (float*)myMalloc(520 * sizeof(float));;
for(int x383=0; x383 < 20; x383++) {
int32_t x385 = 26 * x383;
for(int x384=0; x384 < 26; x384++) {
int32_t x386 = x385 + x384;
float x387 = x380[x386];
float x388 = x49[x384];
float x389 = x387 + x388;
x380[x386] = x389;

}

}
int* x395 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x396=0; x396 < 20; x396++) {
int32_t x397 = x396 * 20;
int32_t x398 = x324 + x397;
int32_t x399 = x72[x398];
x395[x396] = x399;

}
float* x403 = (float*)myMalloc(20 * sizeof(float));;
int32_t x404 = 0;
for(int x405=0; x405 < 20; x405++) {
float x406 = -3.4028235E38f;
for(int x407=0; x407 < 26; x407++) {
int32_t x408 = x404;
float x409 = x380[x408];
float x410 = x406;
bool x411 = x409 > x410;
if (x411) {
float x412 = x380[x408];
x406 = x412;
} else {
}
x404 += 1;

}
float x419 = x406;
x403[x405] = x419;

}
float* x423 = (float*)myMalloc(520 * sizeof(float));;
int32_t x424 = 0;
for(int x425=0; x425 < 20; x425++) {
for(int x426=0; x426 < 26; x426++) {
int32_t x427 = x424;
float x428 = x380[x427];
float x429 = x403[x425];
float x430 = x428 - x429;
double x431 = (double)x430;
double x432 = exp(x431);
float x433 = (float)x432;
x423[x427] = x433;
x424 += 1;

}

}
float* x440 = (float*)myMalloc(20 * sizeof(float));;
for(int x441=0; x441 < 20; x441++) {
int32_t x442 = x441;
int32_t x443 = x441 * 26;
int32_t x444 = x443;
for(int x445=0; x445 < 26; x445++) {
for(int x446=0; x446 < 1; x446++) {
int32_t x447 = x442;
int32_t x448 = x447 + x446;
float x449 = x440[x448];
int32_t x450 = x444;
int32_t x451 = x450 + x446;
float x452 = x423[x451];
float x453 = x449 + x452;
x440[x448] = x453;

}
x444 += 1;

}

}
x424 = 0;
for(int x463=0; x463 < 20; x463++) {
float x464 = x403[x463];
float x465 = x440[x463];
double x466 = (double)x465;
double x467 = log(x466);
float x468 = (float)x467;
float x469 = x464 + x468;
for(int x470=0; x470 < 26; x470++) {
int32_t x471 = x424;
float x472 = x380[x471];
float x473 = x472 - x469;
x423[x471] = x473;
x424 += 1;

}

}
float* x480 = (float*)myMalloc(520 * sizeof(float));;
// nllLoss forward in CPU
float* x482 = (float*)myMalloc(20 * sizeof(float));;
int32_t x483 = 0;
for(int x484=0; x484 < 20; x484++) {
int32_t x485 = x483;
int32_t x486 = x395[x484];
int32_t x487 = x485 + x486;
float x488 = x423[x487];
float x489 = -1.0f * x488;
x482[x484] = x489;
x483 += 26;

}
float* x494 = (float*)myMalloc(20 * sizeof(float));;
float x495 = 0.0f;
for(int x496=0; x496 < 20; x496++) {
float x497 = x495;
float x498 = x482[x496];
float x499 = x497 + x498;
x495 = x499;

}
float x503 = x495;
float* x504 = (float*)myMalloc(1 * sizeof(float));;
for(int x505=0; x505 < 1; x505++) {
x504[x505] = x503;

}
float* x509 = (float*)myMalloc(1 * sizeof(float));;
if (x305) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x514 = (float*)myMalloc(1 * sizeof(float));;
for(int x515=0; x515 < 1; x515++) {
float x516 = x326[0];
float x517 = x504[0];
float x518 = x516 + x517;
x514[x515] = x518;

}
float* x522 = (float*)myMalloc(1 * sizeof(float));;
float** x524 = (float**)myMalloc(4 * sizeof(float*));;
x524[0] = x514;
x524[1] = x522;
x524[2] = x370;
x524[3] = x379;
int32_t x567 = 0;
float* x580 = (float*)myMalloc(20 * sizeof(float));;
int32_t x602 = 0;
int32_t x523 = x324 + 1;
x321(x523,x524);
// back prop for + op
if (x305) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x536=0; x536 < 1; x536++) {
float x537 = x327[0];
float x538 = x326[0];
float x539 = x504[0];
float x540 = x522[x536];
float x541 = x537 + x540;
x327[0] = x541;
float x543 = x509[0];
float x544 = x326[0];
float x545 = x504[0];
float x546 = x522[x536];
float x547 = x543 + x546;
x509[0] = x547;

}
// 'sum' gradient.
if (x552) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",20,1);
assert(false && "");
}
for(int x558=0; x558 < 20; x558++) {
float x559 = x494[x558];
float x560 = x509[0];
float x561 = x559 + x560;
x494[x558] = x561;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
for(int x568=0; x568 < 20; x568++) {
int32_t x569 = x567;
int32_t x570 = x395[x568];
int32_t x571 = x569 + x570;
float x572 = x480[x571];
float x573 = x494[x568];
float x574 = -1.0f * x573;
float x575 = x572 + x574;
x480[x571] = x575;
x567 += 26;

}
for(int x581=0; x581 < 20; x581++) {
int32_t x582 = x581;
int32_t x583 = x581 * 26;
int32_t x584 = x583;
for(int x585=0; x585 < 26; x585++) {
for(int x586=0; x586 < 1; x586++) {
int32_t x587 = x582;
int32_t x588 = x587 + x586;
float x589 = x580[x588];
int32_t x590 = x584;
int32_t x591 = x590 + x586;
float x592 = x480[x591];
float x593 = x589 + x592;
x580[x588] = x593;

}
x584 += 1;

}

}
for(int x603=0; x603 < 20; x603++) {
for(int x604=0; x604 < 26; x604++) {
int32_t x605 = x602;
float x606 = x382[x605];
float x607 = x480[x605];
float x608 = x423[x605];
float x612 = x580[x603];
double x609 = (double)x608;
double x610 = exp(x609);
float x611 = (float)x610;
float x613 = x611 * x612;
float x614 = x607 - x613;
float x615 = x606 + x614;
x382[x605] = x615;
x602 += 1;

}

}
for(int x622=0; x622 < 20; x622++) {
int32_t x624 = 26 * x622;
for(int x623=0; x623 < 26; x623++) {
float x626 = x50[x623];
int32_t x625 = x624 + x623;
float x627 = x382[x625];
float x628 = x626 + x627;
x50[x623] = x628;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x382,26,x40,26,1,x379,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x370,50,x382,26,1,x48,26);
for(int x637=0; x637 < 1000; x637++) {
float x638 = x369[x637];
float x639 = x370[x637];
float x642 = x379[x637];
float x640 = x639 * x639;
float x641 = 1.0f - x640;
float x643 = x641 * x642;
float x644 = x638 + x643;
x369[x637] = x644;

}
// back prop for + op
for(int x649=0; x649 < 20; x649++) {
int32_t x651 = 50 * x649;
for(int x650=0; x650 < 50; x650++) {
int32_t x652 = x651 + x650;
float x653 = x354[x652];
float x654 = x340[x652];
float x655 = x38[x650];
float x656 = x369[x652];
float x657 = x653 + x656;
x354[x652] = x657;
float x659 = x39[x650];
float x660 = x340[x652];
float x661 = x38[x650];
float x662 = x369[x652];
float x663 = x659 + x662;
x39[x650] = x663;

}

}
// back prop for + op
for(int x670=0; x670 < 20; x670++) {
int32_t x672 = 50 * x670;
for(int x671=0; x671 < 50; x671++) {
int32_t x673 = x672 + x671;
float x674 = x336[x673];
float x675 = x334[x673];
float x676 = x337[x673];
float x677 = x354[x673];
float x678 = x674 + x677;
x336[x673] = x678;
float x680 = x339[x673];
float x681 = x334[x673];
float x682 = x337[x673];
float x683 = x354[x673];
float x684 = x680 + x683;
x339[x673] = x684;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x339,50,x28,50,1,x329,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x328,50,x339,50,1,x37,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x336,50,x17,50,1,x333,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x332,26,x336,50,1,x27,50);
} else {
float x697 = 0.0f;
for(int x698=0; x698 < 1; x698++) {
float x699 = x697;
float x700 = x326[x698];
float x701 = x699 + x700;
x697 = x701;

}
float x705 = x697;
float* x706 = (float*)myMalloc(1 * sizeof(float));;
for(int x707=0; x707 < 1; x707++) {
x706[x707] = x705;

}
float* x711 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x713=0; x713 < 1; x713++) {
x711[x713] = 1.0f;

}
// backend is lantern.TensorDslCPU$BackendCPU@233cbdfe
for(int x718=0; x718 < 1; x718++) {
float x719 = x706[x718];
x87[x718] = x719;

}
// 'sum' gradient.
if (x305) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x728=0; x728 < 1; x728++) {
float x729 = x327[0];
float x730 = x711[0];
float x731 = x729 + x730;
x327[0] = x731;

}
}
};
x59 += 400;
int32_t x64 = x59;
int32_t x65 = x64 + 400;
int32_t x66 = x65 + 1;
bool x67 = x66 >= x5;
if (x67) {
x59 = 0;
} else {
}
int* x71 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
for(int x74=0; x74 < 400; x74++) {
int32_t x75 = x59;
int32_t x76 = x75 + x74;
int32_t x77 = x8[x76];
x71[x74] = x77;
int32_t x79 = x76 + 1;
int32_t x80 = x8[x79];
x72[x74] = x80;

}
float* x84 = (float*)myMalloc(1 * sizeof(float));;
float* x85 = (float*)myMalloc(1 * sizeof(float));;
// allocate memory to save the final loss in CPU Tensor
for(int x90=0; x90 < 20; x90++) {
int32_t x92 = x90 * 26;
int32_t x93 = x92 * 20;
for(int x91=0; x91 < 20; x91++) {
int32_t x96 = x91 * 20;
int32_t x97 = x96 + x90;
int32_t x98 = x71[x97];
int32_t x94 = x91 * 26;
int32_t x95 = x93 + x94;
int32_t x99 = x95 + x98;
x88[x99] = 1.0f;

}

}
float* x106 = (float*)myMalloc(1 * sizeof(float));;
float* x107 = (float*)myMalloc(1 * sizeof(float));;
float* x108 = (float*)myMalloc(1000 * sizeof(float));;
float* x109 = (float*)myMalloc(1000 * sizeof(float));;
float** x950 = (float**)myMalloc(4 * sizeof(float*));;
x950[0] = x106;
x950[1] = x107;
x950[2] = x108;
x950[3] = x109;
function<void(int32_t,float**)> x110 = [&](int32_t x111,float** x112) {
float** x114 = x112;
float* x115 = x114[0];
float* x116 = x114[1];
float* x117 = x114[2];
float* x118 = x114[3];
int32_t x113 = x111;
bool x119 = x113 < 20;
if (x119) {
int32_t x120 = x113 * 520;
float* x121 = x88+x120;
float* x122 = x105+x120;
float* x123 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x121,26,x17,50,0,x123,50);
float* x125 = (float*)myMalloc(1000 * sizeof(float));;
float* x126 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x117,50,x28,50,0,x126,50);
float* x128 = (float*)myMalloc(1000 * sizeof(float));;
float* x129 = (float*)myMalloc(1000 * sizeof(float));;
for(int x130=0; x130 < 20; x130++) {
int32_t x133 = 50 * x130;
for(int x132=0; x132 < 50; x132++) {
int32_t x135 = x133 + x132;
float x136 = x123[x135];
float x137 = x126[x135];
int32_t x134 = x132 + x133;
float x138 = x136 + x137;
x129[x134] = x138;

}

}
float* x144 = (float*)myMalloc(1000 * sizeof(float));;
float* x146 = (float*)myMalloc(1000 * sizeof(float));;
for(int x147=0; x147 < 20; x147++) {
int32_t x149 = 50 * x147;
for(int x148=0; x148 < 50; x148++) {
int32_t x151 = x149 + x148;
float x152 = x129[x151];
float x153 = x38[x148];
int32_t x150 = x148 + x149;
float x154 = x152 + x153;
x146[x150] = x154;

}

}
float* x160 = (float*)myMalloc(1000 * sizeof(float));;
float* x161 = (float*)myMalloc(1000 * sizeof(float));;
for(int x163=0; x163 < 1000; x163++) {
float x164 = x146[x163];
double x165 = (double)x164;
double x166 = tanh(x165);
float x167 = (float)x166;
x161[x163] = x167;

}
float* x171 = (float*)myMalloc(1000 * sizeof(float));;
float* x172 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x161,50,x40,26,0,x172,26);
float* x174 = (float*)myMalloc(520 * sizeof(float));;
for(int x175=0; x175 < 20; x175++) {
int32_t x178 = 26 * x175;
for(int x177=0; x177 < 26; x177++) {
int32_t x179 = x178 + x177;
float x180 = x172[x179];
float x181 = x49[x177];
float x182 = x180 + x181;
x172[x179] = x182;

}

}
int* x188 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x189=0; x189 < 20; x189++) {
int32_t x190 = x189 * 20;
int32_t x191 = x113 + x190;
int32_t x192 = x72[x191];
x188[x189] = x192;

}
float* x196 = (float*)myMalloc(20 * sizeof(float));;
int32_t x197 = 0;
for(int x198=0; x198 < 20; x198++) {
float x199 = -3.4028235E38f;
for(int x200=0; x200 < 26; x200++) {
int32_t x201 = x197;
float x202 = x172[x201];
float x203 = x199;
bool x204 = x202 > x203;
if (x204) {
float x205 = x172[x201];
x199 = x205;
} else {
}
x197 += 1;

}
float x212 = x199;
x196[x198] = x212;

}
float* x216 = (float*)myMalloc(520 * sizeof(float));;
int32_t x217 = 0;
for(int x218=0; x218 < 20; x218++) {
for(int x219=0; x219 < 26; x219++) {
int32_t x220 = x217;
float x221 = x172[x220];
float x222 = x196[x218];
float x223 = x221 - x222;
double x224 = (double)x223;
double x225 = exp(x224);
float x226 = (float)x225;
x216[x220] = x226;
x217 += 1;

}

}
float* x233 = (float*)myMalloc(20 * sizeof(float));;
for(int x234=0; x234 < 20; x234++) {
int32_t x235 = x234;
int32_t x236 = x234 * 26;
int32_t x237 = x236;
for(int x238=0; x238 < 26; x238++) {
for(int x240=0; x240 < 1; x240++) {
int32_t x241 = x235;
int32_t x242 = x241 + x240;
float x243 = x233[x242];
int32_t x244 = x237;
int32_t x245 = x244 + x240;
float x246 = x216[x245];
float x247 = x243 + x246;
x233[x242] = x247;

}
x237 += 1;

}

}
x217 = 0;
for(int x257=0; x257 < 20; x257++) {
float x258 = x196[x257];
float x259 = x233[x257];
double x260 = (double)x259;
double x261 = log(x260);
float x262 = (float)x261;
float x263 = x258 + x262;
for(int x264=0; x264 < 26; x264++) {
int32_t x265 = x217;
float x266 = x172[x265];
float x267 = x266 - x263;
x216[x265] = x267;
x217 += 1;

}

}
float* x274 = (float*)myMalloc(520 * sizeof(float));;
// nllLoss forward in CPU
float* x276 = (float*)myMalloc(20 * sizeof(float));;
int32_t x277 = 0;
for(int x278=0; x278 < 20; x278++) {
int32_t x279 = x277;
int32_t x280 = x188[x278];
int32_t x281 = x279 + x280;
float x282 = x216[x281];
float x283 = -1.0f * x282;
x276[x278] = x283;
x277 += 26;

}
float* x288 = (float*)myMalloc(20 * sizeof(float));;
float x289 = 0.0f;
for(int x290=0; x290 < 20; x290++) {
float x291 = x289;
float x292 = x276[x290];
float x293 = x291 + x292;
x289 = x293;

}
float x297 = x289;
float* x298 = (float*)myMalloc(1 * sizeof(float));;
for(int x299=0; x299 < 1; x299++) {
x298[x299] = x297;

}
float* x303 = (float*)myMalloc(1 * sizeof(float));;
if (x305) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x311 = (float*)myMalloc(1 * sizeof(float));;
for(int x312=0; x312 < 1; x312++) {
float x313 = x115[0];
float x314 = x298[0];
float x315 = x313 + x314;
x311[x312] = x315;

}
float* x319 = (float*)myMalloc(1 * sizeof(float));;
float** x738 = (float**)myMalloc(4 * sizeof(float*));;
x738[0] = x311;
x738[1] = x319;
x738[2] = x161;
x738[3] = x171;
int32_t x320 = x113 + 1;
x321(x320,x738);
// back prop for + op
if (x305) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x750=0; x750 < 1; x750++) {
float x751 = x116[0];
float x752 = x115[0];
float x753 = x298[0];
float x754 = x319[x750];
float x755 = x751 + x754;
x116[0] = x755;
float x757 = x303[0];
float x758 = x115[0];
float x759 = x298[0];
float x760 = x319[x750];
float x761 = x757 + x760;
x303[0] = x761;

}
// 'sum' gradient.
if (x552) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",20,1);
assert(false && "");
}
for(int x770=0; x770 < 20; x770++) {
float x771 = x288[x770];
float x772 = x303[0];
float x773 = x771 + x772;
x288[x770] = x773;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x779 = 0;
for(int x780=0; x780 < 20; x780++) {
int32_t x781 = x779;
int32_t x782 = x188[x780];
int32_t x783 = x781 + x782;
float x784 = x274[x783];
float x785 = x288[x780];
float x786 = -1.0f * x785;
float x787 = x784 + x786;
x274[x783] = x787;
x779 += 26;

}
float* x792 = (float*)myMalloc(20 * sizeof(float));;
for(int x793=0; x793 < 20; x793++) {
int32_t x794 = x793;
int32_t x795 = x793 * 26;
int32_t x796 = x795;
for(int x797=0; x797 < 26; x797++) {
for(int x798=0; x798 < 1; x798++) {
int32_t x799 = x794;
int32_t x800 = x799 + x798;
float x801 = x792[x800];
int32_t x802 = x796;
int32_t x803 = x802 + x798;
float x804 = x274[x803];
float x805 = x801 + x804;
x792[x800] = x805;

}
x796 += 1;

}

}
int32_t x814 = 0;
for(int x815=0; x815 < 20; x815++) {
for(int x816=0; x816 < 26; x816++) {
int32_t x817 = x814;
float x818 = x174[x817];
float x819 = x274[x817];
float x820 = x216[x817];
float x824 = x792[x815];
double x821 = (double)x820;
double x822 = exp(x821);
float x823 = (float)x822;
float x825 = x823 * x824;
float x826 = x819 - x825;
float x827 = x818 + x826;
x174[x817] = x827;
x814 += 1;

}

}
for(int x834=0; x834 < 20; x834++) {
int32_t x836 = 26 * x834;
for(int x835=0; x835 < 26; x835++) {
float x838 = x50[x835];
int32_t x837 = x836 + x835;
float x839 = x174[x837];
float x840 = x838 + x839;
x50[x835] = x840;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x174,26,x40,26,1,x171,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x161,50,x174,26,1,x48,26);
for(int x849=0; x849 < 1000; x849++) {
float x850 = x160[x849];
float x851 = x161[x849];
float x854 = x171[x849];
float x852 = x851 * x851;
float x853 = 1.0f - x852;
float x855 = x853 * x854;
float x856 = x850 + x855;
x160[x849] = x856;

}
// back prop for + op
for(int x861=0; x861 < 20; x861++) {
int32_t x863 = 50 * x861;
for(int x862=0; x862 < 50; x862++) {
int32_t x864 = x863 + x862;
float x865 = x144[x864];
float x866 = x129[x864];
float x867 = x38[x862];
float x868 = x160[x864];
float x869 = x865 + x868;
x144[x864] = x869;
float x871 = x39[x862];
float x872 = x129[x864];
float x873 = x38[x862];
float x874 = x160[x864];
float x875 = x871 + x874;
x39[x862] = x875;

}

}
// back prop for + op
for(int x882=0; x882 < 20; x882++) {
int32_t x884 = 50 * x882;
for(int x883=0; x883 < 50; x883++) {
int32_t x885 = x884 + x883;
float x886 = x125[x885];
float x887 = x123[x885];
float x888 = x126[x885];
float x889 = x144[x885];
float x890 = x886 + x889;
x125[x885] = x890;
float x892 = x128[x885];
float x893 = x123[x885];
float x894 = x126[x885];
float x895 = x144[x885];
float x896 = x892 + x895;
x128[x885] = x896;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x128,50,x28,50,1,x118,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x117,50,x128,50,1,x37,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x125,50,x17,50,1,x122,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x121,26,x125,50,1,x27,50);
} else {
float x909 = 0.0f;
for(int x910=0; x910 < 1; x910++) {
float x911 = x909;
float x912 = x115[x910];
float x913 = x911 + x912;
x909 = x913;

}
float x917 = x909;
float* x918 = (float*)myMalloc(1 * sizeof(float));;
for(int x919=0; x919 < 1; x919++) {
x918[x919] = x917;

}
float* x923 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x925=0; x925 < 1; x925++) {
x923[x925] = 1.0f;

}
// backend is lantern.TensorDslCPU$BackendCPU@233cbdfe
for(int x930=0; x930 < 1; x930++) {
float x931 = x918[x930];
x87[x930] = x931;

}
// 'sum' gradient.
if (x305) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x940=0; x940 < 1; x940++) {
float x941 = x116[0];
float x942 = x923[0];
float x943 = x941 + x942;
x116[0] = x943;

}
}
};
x110(0,x950);
float x957 = x87[0];
int32_t x958 = x62 % 100;
bool x959 = x958 == 0;
if (x959) {
printf("iter %d, loss %f\n",x62,x957);
int32_t x961 = x62 / 100;
double x962 = (double)x957;
x56[x961] = x962;
} else {
}
for(int x966=0; x966 < 26; x966++) {
float x967 = x50[x966];
float x968 = x967;
float x969 = x968;
bool x970 = x969 > 5.0f;
if (x970) {
x968 = 5.0f;
} else {
}
float x974 = x968;
bool x975 = x974 < -5.0f;
if (x975) {
x968 = -5.0f;
} else {
}
float x979 = x51[x966];
float x980 = x968;
float x981 = x980 * x980;
float x982 = x979 + x981;
x51[x966] = x982;
float x984 = x49[x966];
float x986 = x51[x966];
float x985 = 0.1f * x980;
double x987 = (double)x986;
double x988 = x987 + 9.99999993922529E-9;
double x989 = sqrt(x988);
float x990 = (float)x989;
float x991 = x985 / x990;
float x992 = x984 - x991;
x49[x966] = x992;
x50[x966] = 0.0f;

}
for(int x997=0; x997 < 1300; x997++) {
float x998 = x48[x997];
float x999 = x998;
float x1000 = x999;
bool x1001 = x1000 > 5.0f;
if (x1001) {
x999 = 5.0f;
} else {
}
float x1005 = x999;
bool x1006 = x1005 < -5.0f;
if (x1006) {
x999 = -5.0f;
} else {
}
float x1010 = x52[x997];
float x1011 = x999;
float x1012 = x1011 * x1011;
float x1013 = x1010 + x1012;
x52[x997] = x1013;
float x1015 = x40[x997];
float x1017 = x52[x997];
float x1016 = 0.1f * x1011;
double x1018 = (double)x1017;
double x1019 = x1018 + 9.99999993922529E-9;
double x1020 = sqrt(x1019);
float x1021 = (float)x1020;
float x1022 = x1016 / x1021;
float x1023 = x1015 - x1022;
x40[x997] = x1023;
x48[x997] = 0.0f;

}
for(int x1028=0; x1028 < 2500; x1028++) {
float x1029 = x37[x1028];
float x1030 = x1029;
float x1031 = x1030;
bool x1032 = x1031 > 5.0f;
if (x1032) {
x1030 = 5.0f;
} else {
}
float x1036 = x1030;
bool x1037 = x1036 < -5.0f;
if (x1037) {
x1030 = -5.0f;
} else {
}
float x1041 = x53[x1028];
float x1042 = x1030;
float x1043 = x1042 * x1042;
float x1044 = x1041 + x1043;
x53[x1028] = x1044;
float x1046 = x28[x1028];
float x1048 = x53[x1028];
float x1047 = 0.1f * x1042;
double x1049 = (double)x1048;
double x1050 = x1049 + 9.99999993922529E-9;
double x1051 = sqrt(x1050);
float x1052 = (float)x1051;
float x1053 = x1047 / x1052;
float x1054 = x1046 - x1053;
x28[x1028] = x1054;
x37[x1028] = 0.0f;

}
for(int x1059=0; x1059 < 50; x1059++) {
float x1060 = x39[x1059];
float x1061 = x1060;
float x1062 = x1061;
bool x1063 = x1062 > 5.0f;
if (x1063) {
x1061 = 5.0f;
} else {
}
float x1067 = x1061;
bool x1068 = x1067 < -5.0f;
if (x1068) {
x1061 = -5.0f;
} else {
}
float x1072 = x54[x1059];
float x1073 = x1061;
float x1074 = x1073 * x1073;
float x1075 = x1072 + x1074;
x54[x1059] = x1075;
float x1077 = x38[x1059];
float x1079 = x54[x1059];
float x1078 = 0.1f * x1073;
double x1080 = (double)x1079;
double x1081 = x1080 + 9.99999993922529E-9;
double x1082 = sqrt(x1081);
float x1083 = (float)x1082;
float x1084 = x1078 / x1083;
float x1085 = x1077 - x1084;
x38[x1059] = x1085;
x39[x1059] = 0.0f;

}
for(int x1090=0; x1090 < 1300; x1090++) {
float x1091 = x27[x1090];
float x1092 = x1091;
float x1093 = x1092;
bool x1094 = x1093 > 5.0f;
if (x1094) {
x1092 = 5.0f;
} else {
}
float x1098 = x1092;
bool x1099 = x1098 < -5.0f;
if (x1099) {
x1092 = -5.0f;
} else {
}
float x1103 = x55[x1090];
float x1104 = x1092;
float x1105 = x1104 * x1104;
float x1106 = x1103 + x1105;
x55[x1090] = x1106;
float x1108 = x17[x1090];
float x1110 = x55[x1090];
float x1109 = 0.1f * x1104;
double x1111 = (double)x1110;
double x1112 = x1111 + 9.99999993922529E-9;
double x1113 = sqrt(x1112);
float x1114 = (float)x1113;
float x1115 = x1109 / x1114;
float x1116 = x1108 - x1115;
x17[x1090] = x1116;
x27[x1090] = 0.0f;

}
int64_t x1121 = (long)mallocAddr;
int64_t x1122 = x1121 - x58;
memset((void*)x58, 0, x1122);
mallocAddr = (void*)x58;

}
double x1127 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x1130 = (long)fopen(x0, "w");
fprintf((FILE *)x1130, "unit: %s\n", "100 iteration");
for(int x1133=0; x1133 < 51; x1133++) {
double x1134 = x56[x1133];
fprintf((FILE *)x1130, "%lf\n", x1134);

}
double x1128 = x57 - x2;
double x1129 = x1127 - x57;
fprintf((FILE *)x1130, "run time: %lf %lf\n", x1128, x1129);
fclose((FILE*)x1130);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

