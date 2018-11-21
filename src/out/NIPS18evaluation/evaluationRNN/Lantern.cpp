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
double x1 = ((double)clock() / CLOCKS_PER_SEC);
int32_t x2 = open("graham.txt",0);
int64_t x3 = fsize(x2);
int32_t x4 = (int32_t)x3;
int* x7 = (int32_t*)myMalloc(x4 * sizeof(int32_t));;
int64_t x5 = (int64_t)x4;
char* x6 = (char*)mmap(0, x5, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x2, 0);
for(int x9=0; x9 < x4; x9++) {
char x10 = x6[x9];
int32_t x11 = (int32_t ) x10;
int32_t x12 = x11 - 96;
x7[x9] = x12;

}
float* x16 = (float*)myMalloc(1300 * sizeof(float));;
for(int x18=0; x18 < 1300; x18++) {
float x19 = (float)rand()/RAND_MAX;
float x20 = x19 - 0.5f;
float x21 = x20 * 0.19611613f;
x16[x18] = x21;

}
float* x26 = (float*)myMalloc(1300 * sizeof(float));;
float* x27 = (float*)myMalloc(2500 * sizeof(float));;
for(int x29=0; x29 < 2500; x29++) {
float x30 = (float)rand()/RAND_MAX;
float x31 = x30 - 0.5f;
float x32 = x31 * 0.14142136f;
x27[x29] = x32;

}
float* x36 = (float*)myMalloc(2500 * sizeof(float));;
float* x37 = (float*)myMalloc(50 * sizeof(float));;
float* x38 = (float*)myMalloc(50 * sizeof(float));;
float* x39 = (float*)myMalloc(1300 * sizeof(float));;
for(int x40=0; x40 < 1300; x40++) {
float x41 = (float)rand()/RAND_MAX;
float x42 = x41 - 0.5f;
float x43 = x42 * 0.14142136f;
x39[x40] = x43;

}
float* x47 = (float*)myMalloc(1300 * sizeof(float));;
float* x48 = (float*)myMalloc(26 * sizeof(float));;
float* x49 = (float*)myMalloc(26 * sizeof(float));;
float* x50 = (float*)myMalloc(26 * sizeof(float));;
float* x51 = (float*)myMalloc(1300 * sizeof(float));;
float* x52 = (float*)myMalloc(2500 * sizeof(float));;
float* x53 = (float*)myMalloc(50 * sizeof(float));;
float* x54 = (float*)myMalloc(1300 * sizeof(float));;
double* x55 = (double*)myMalloc(51 * sizeof(double));;
double x56 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x57 = (long)mallocAddr;
int32_t x58 = 0;
x58 -= 400;
bool x302 = true || true;
bool x303 = x302 || true;
bool x553 = true || false;
for(int x61=0; x61 < 5001; x61++) {
float* x86 = (float*)myMalloc(1 * sizeof(float));;
float* x87 = (float*)myMalloc(10400 * sizeof(float));;
float* x104 = (float*)myMalloc(10400 * sizeof(float));;
int* x71 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
function<void(int32_t,float**)> x319 = [&](int32_t x320,float** x321) {
float** x323 = x321;
float* x324 = x323[0];
float* x325 = x323[1];
float* x326 = x323[2];
float* x327 = x323[3];
int32_t x322 = x320;
bool x328 = x322 < 20;
if (x328) {
int32_t x329 = x322 * 520;
float* x330 = x87+x329;
float* x331 = x104+x329;
float* x332 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x330,26,x16,50,0,x332,50);
float* x334 = (float*)myMalloc(1000 * sizeof(float));;
float* x335 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x326,50,x27,50,0,x335,50);
float* x337 = (float*)myMalloc(1000 * sizeof(float));;
float* x338 = (float*)myMalloc(1000 * sizeof(float));;
for(int x339=0; x339 < 20; x339++) {
int32_t x341 = 50 * x339;
for(int x340=0; x340 < 50; x340++) {
int32_t x343 = x341 + x340;
float x344 = x332[x343];
float x345 = x335[x343];
int32_t x342 = x340 + x341;
float x346 = x344 + x345;
x338[x342] = x346;

}

}
float* x352 = (float*)myMalloc(1000 * sizeof(float));;
float* x353 = (float*)myMalloc(1000 * sizeof(float));;
for(int x354=0; x354 < 20; x354++) {
int32_t x356 = 50 * x354;
for(int x355=0; x355 < 50; x355++) {
int32_t x358 = x356 + x355;
float x359 = x338[x358];
float x360 = x37[x355];
int32_t x357 = x355 + x356;
float x361 = x359 + x360;
x353[x357] = x361;

}

}
float* x367 = (float*)myMalloc(1000 * sizeof(float));;
float* x368 = (float*)myMalloc(1000 * sizeof(float));;
for(int x369=0; x369 < 1000; x369++) {
float x370 = x353[x369];
double x371 = (double)x370;
double x372 = tanh(x371);
float x373 = (float)x372;
x368[x369] = x373;

}
float* x377 = (float*)myMalloc(1000 * sizeof(float));;
float* x378 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x368,50,x39,26,0,x378,26);
float* x380 = (float*)myMalloc(520 * sizeof(float));;
for(int x381=0; x381 < 20; x381++) {
int32_t x383 = 26 * x381;
for(int x382=0; x382 < 26; x382++) {
int32_t x384 = x383 + x382;
float x385 = x378[x384];
float x386 = x48[x382];
float x387 = x385 + x386;
x378[x384] = x387;

}

}
int* x393 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x394=0; x394 < 20; x394++) {
int32_t x395 = x394 * 20;
int32_t x396 = x322 + x395;
int32_t x397 = x71[x396];
x393[x394] = x397;

}
float* x401 = (float*)myMalloc(20 * sizeof(float));;
int32_t x402 = 0;
for(int x403=0; x403 < 20; x403++) {
float x404 = -3.4028235E38f;
for(int x405=0; x405 < 26; x405++) {
int32_t x406 = x402;
float x407 = x378[x406];
float x408 = x404;
bool x409 = x407 > x408;
if (x409) {
float x410 = x378[x406];
x404 = x410;
} else {
}
x402 += 1;

}
float x417 = x404;
x401[x403] = x417;

}
float* x421 = (float*)myMalloc(520 * sizeof(float));;
int32_t x422 = 0;
for(int x423=0; x423 < 20; x423++) {
for(int x424=0; x424 < 26; x424++) {
int32_t x425 = x422;
float x426 = x378[x425];
float x427 = x401[x423];
float x428 = x426 - x427;
double x429 = (double)x428;
double x430 = exp(x429);
float x431 = (float)x430;
x421[x425] = x431;
x422 += 1;

}

}
float* x438 = (float*)myMalloc(20 * sizeof(float));;
for(int x439=0; x439 < 20; x439++) {
int32_t x440 = x439;
int32_t x441 = x439 * 26;
int32_t x442 = x441;
for(int x443=0; x443 < 26; x443++) {
for(int x444=0; x444 < 1; x444++) {
int32_t x445 = x440;
int32_t x446 = x445 + x444;
float x447 = x438[x446];
int32_t x448 = x442;
int32_t x449 = x448 + x444;
float x450 = x421[x449];
float x451 = x447 + x450;
x438[x446] = x451;

}
x442 += 1;

}

}
x422 = 0;
for(int x461=0; x461 < 20; x461++) {
float x462 = x401[x461];
float x463 = x438[x461];
double x464 = (double)x463;
double x465 = log(x464);
float x466 = (float)x465;
float x467 = x462 + x466;
for(int x468=0; x468 < 26; x468++) {
int32_t x469 = x422;
float x470 = x378[x469];
float x471 = x470 - x467;
x421[x469] = x471;
x422 += 1;

}

}
float* x478 = (float*)myMalloc(520 * sizeof(float));;
// nllLoss forward in CPU
float* x480 = (float*)myMalloc(20 * sizeof(float));;
int32_t x481 = 0;
for(int x482=0; x482 < 20; x482++) {
int32_t x483 = x481;
int32_t x484 = x393[x482];
int32_t x485 = x483 + x484;
float x486 = x421[x485];
float x487 = -1.0f * x486;
x480[x482] = x487;
x481 += 26;

}
float* x492 = (float*)myMalloc(20 * sizeof(float));;
float x493 = 0.0f;
for(int x494=0; x494 < 20; x494++) {
float x495 = x493;
float x496 = x480[x494];
float x497 = x495 + x496;
x493 = x497;

}
float x501 = x493;
float* x502 = (float*)myMalloc(1 * sizeof(float));;
for(int x503=0; x503 < 1; x503++) {
x502[x503] = x501;

}
float* x507 = (float*)myMalloc(1 * sizeof(float));;
if (x303) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x512 = (float*)myMalloc(1 * sizeof(float));;
for(int x513=0; x513 < 1; x513++) {
float x514 = x324[0];
float x515 = x502[0];
float x516 = x514 + x515;
x512[x513] = x516;

}
float* x520 = (float*)myMalloc(1 * sizeof(float));;
float** x522 = (float**)myMalloc(4 * sizeof(float*));;
x522[0] = x512;
x522[1] = x520;
x522[2] = x368;
x522[3] = x377;
int32_t x568 = 0;
float* x581 = (float*)myMalloc(20 * sizeof(float));;
int32_t x603 = 0;
int32_t x521 = x322 + 1;
x319(x521,x522);
// back prop for + op
if (x303) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x534=0; x534 < 1; x534++) {
float x535 = x325[0];
float x536 = x520[0];
float x537 = x535 + x536;
x325[0] = x537;

}
if (x303) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x545=0; x545 < 1; x545++) {
float x546 = x507[0];
float x547 = x520[0];
float x548 = x546 + x547;
x507[0] = x548;

}
// 'sum' gradient.
if (x553) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",20,1);
assert(false && "");
}
for(int x559=0; x559 < 20; x559++) {
float x560 = x492[x559];
float x561 = x507[0];
float x562 = x560 + x561;
x492[x559] = x562;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
for(int x569=0; x569 < 20; x569++) {
int32_t x570 = x568;
int32_t x571 = x393[x569];
int32_t x572 = x570 + x571;
float x573 = x478[x572];
float x574 = x492[x569];
float x575 = -1.0f * x574;
float x576 = x573 + x575;
x478[x572] = x576;
x568 += 26;

}
for(int x582=0; x582 < 20; x582++) {
int32_t x583 = x582;
int32_t x584 = x582 * 26;
int32_t x585 = x584;
for(int x586=0; x586 < 26; x586++) {
for(int x587=0; x587 < 1; x587++) {
int32_t x588 = x583;
int32_t x589 = x588 + x587;
float x590 = x581[x589];
int32_t x591 = x585;
int32_t x592 = x591 + x587;
float x593 = x478[x592];
float x594 = x590 + x593;
x581[x589] = x594;

}
x585 += 1;

}

}
for(int x604=0; x604 < 20; x604++) {
for(int x605=0; x605 < 26; x605++) {
int32_t x606 = x603;
float x607 = x380[x606];
float x608 = x478[x606];
float x609 = x421[x606];
float x613 = x581[x604];
double x610 = (double)x609;
double x611 = exp(x610);
float x612 = (float)x611;
float x614 = x612 * x613;
float x615 = x608 - x614;
float x616 = x607 + x615;
x380[x606] = x616;
x603 += 1;

}

}
for(int x623=0; x623 < 20; x623++) {
int32_t x625 = 26 * x623;
for(int x624=0; x624 < 26; x624++) {
float x627 = x49[x624];
int32_t x626 = x625 + x624;
float x628 = x380[x626];
float x629 = x627 + x628;
x49[x624] = x629;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x380,26,x39,26,1,x377,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x368,50,x380,26,1,x47,26);
for(int x638=0; x638 < 1000; x638++) {
float x639 = x367[x638];
float x640 = x368[x638];
float x643 = x377[x638];
float x641 = x640 * x640;
float x642 = 1.0f - x641;
float x644 = x642 * x643;
float x645 = x639 + x644;
x367[x638] = x645;

}
// back prop for + op
for(int x650=0; x650 < 20; x650++) {
int32_t x652 = 50 * x650;
for(int x651=0; x651 < 50; x651++) {
int32_t x653 = x652 + x651;
float x654 = x352[x653];
float x655 = x367[x653];
float x656 = x654 + x655;
x352[x653] = x656;

}

}
for(int x662=0; x662 < 20; x662++) {
int32_t x664 = 50 * x662;
for(int x663=0; x663 < 50; x663++) {
float x666 = x38[x663];
int32_t x665 = x664 + x663;
float x667 = x367[x665];
float x668 = x666 + x667;
x38[x663] = x668;

}

}
// back prop for + op
for(int x675=0; x675 < 20; x675++) {
int32_t x677 = 50 * x675;
for(int x676=0; x676 < 50; x676++) {
int32_t x678 = x677 + x676;
float x679 = x334[x678];
float x680 = x352[x678];
float x681 = x679 + x680;
x334[x678] = x681;

}

}
for(int x687=0; x687 < 20; x687++) {
int32_t x689 = 50 * x687;
for(int x688=0; x688 < 50; x688++) {
int32_t x690 = x689 + x688;
float x691 = x337[x690];
float x692 = x352[x690];
float x693 = x691 + x692;
x337[x690] = x693;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x337,50,x27,50,1,x327,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x326,50,x337,50,1,x36,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x334,50,x16,50,1,x331,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x330,26,x334,50,1,x26,50);
} else {
float x706 = 0.0f;
for(int x707=0; x707 < 1; x707++) {
float x708 = x706;
float x709 = x324[x707];
float x710 = x708 + x709;
x706 = x710;

}
float x714 = x706;
float* x715 = (float*)myMalloc(1 * sizeof(float));;
for(int x716=0; x716 < 1; x716++) {
x715[x716] = x714;

}
float* x720 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x722=0; x722 < 1; x722++) {
x720[x722] = 1.0f;

}
// backend is lantern.TensorDsl$BackendCPU@12c72b3d
for(int x727=0; x727 < 1; x727++) {
float x728 = x715[x727];
x86[x727] = x728;

}
// 'sum' gradient.
if (x303) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x737=0; x737 < 1; x737++) {
float x738 = x325[0];
float x739 = x720[0];
float x740 = x738 + x739;
x325[0] = x740;

}
}
};
x58 += 400;
int32_t x63 = x58;
int32_t x64 = x63 + 400;
int32_t x65 = x64 + 1;
bool x66 = x65 >= x4;
if (x66) {
x58 = 0;
} else {
}
int* x70 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
for(int x73=0; x73 < 400; x73++) {
int32_t x74 = x58;
int32_t x75 = x74 + x73;
int32_t x76 = x7[x75];
x70[x73] = x76;
int32_t x78 = x75 + 1;
int32_t x79 = x7[x78];
x71[x73] = x79;

}
float* x83 = (float*)myMalloc(1 * sizeof(float));;
float* x84 = (float*)myMalloc(1 * sizeof(float));;
// allocate memory to save the final loss in CPU Tensor
for(int x89=0; x89 < 20; x89++) {
int32_t x91 = x89 * 26;
int32_t x92 = x91 * 20;
for(int x90=0; x90 < 20; x90++) {
int32_t x95 = x90 * 20;
int32_t x96 = x95 + x89;
int32_t x97 = x70[x96];
int32_t x93 = x90 * 26;
int32_t x94 = x92 + x93;
int32_t x98 = x94 + x97;
x87[x98] = 1.0f;

}

}
float* x105 = (float*)myMalloc(1 * sizeof(float));;
float* x106 = (float*)myMalloc(1 * sizeof(float));;
float* x107 = (float*)myMalloc(1000 * sizeof(float));;
float* x108 = (float*)myMalloc(1000 * sizeof(float));;
float** x970 = (float**)myMalloc(4 * sizeof(float*));;
x970[0] = x105;
x970[1] = x106;
x970[2] = x107;
x970[3] = x108;
function<void(int32_t,float**)> x109 = [&](int32_t x110,float** x111) {
float** x113 = x111;
float* x114 = x113[0];
float* x115 = x113[1];
float* x116 = x113[2];
float* x117 = x113[3];
int32_t x112 = x110;
bool x118 = x112 < 20;
if (x118) {
int32_t x119 = x112 * 520;
float* x120 = x87+x119;
float* x121 = x104+x119;
float* x122 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x120,26,x16,50,0,x122,50);
float* x124 = (float*)myMalloc(1000 * sizeof(float));;
float* x125 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x116,50,x27,50,0,x125,50);
float* x127 = (float*)myMalloc(1000 * sizeof(float));;
float* x128 = (float*)myMalloc(1000 * sizeof(float));;
for(int x129=0; x129 < 20; x129++) {
int32_t x132 = 50 * x129;
for(int x131=0; x131 < 50; x131++) {
int32_t x134 = x132 + x131;
float x135 = x122[x134];
float x136 = x125[x134];
int32_t x133 = x131 + x132;
float x137 = x135 + x136;
x128[x133] = x137;

}

}
float* x143 = (float*)myMalloc(1000 * sizeof(float));;
float* x144 = (float*)myMalloc(1000 * sizeof(float));;
for(int x145=0; x145 < 20; x145++) {
int32_t x147 = 50 * x145;
for(int x146=0; x146 < 50; x146++) {
int32_t x149 = x147 + x146;
float x150 = x128[x149];
float x151 = x37[x146];
int32_t x148 = x146 + x147;
float x152 = x150 + x151;
x144[x148] = x152;

}

}
float* x158 = (float*)myMalloc(1000 * sizeof(float));;
float* x159 = (float*)myMalloc(1000 * sizeof(float));;
for(int x161=0; x161 < 1000; x161++) {
float x162 = x144[x161];
double x163 = (double)x162;
double x164 = tanh(x163);
float x165 = (float)x164;
x159[x161] = x165;

}
float* x169 = (float*)myMalloc(1000 * sizeof(float));;
float* x170 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x159,50,x39,26,0,x170,26);
float* x172 = (float*)myMalloc(520 * sizeof(float));;
for(int x173=0; x173 < 20; x173++) {
int32_t x176 = 26 * x173;
for(int x175=0; x175 < 26; x175++) {
int32_t x177 = x176 + x175;
float x178 = x170[x177];
float x179 = x48[x175];
float x180 = x178 + x179;
x170[x177] = x180;

}

}
int* x186 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x187=0; x187 < 20; x187++) {
int32_t x188 = x187 * 20;
int32_t x189 = x112 + x188;
int32_t x190 = x71[x189];
x186[x187] = x190;

}
float* x194 = (float*)myMalloc(20 * sizeof(float));;
int32_t x195 = 0;
for(int x196=0; x196 < 20; x196++) {
float x197 = -3.4028235E38f;
for(int x198=0; x198 < 26; x198++) {
int32_t x199 = x195;
float x200 = x170[x199];
float x201 = x197;
bool x202 = x200 > x201;
if (x202) {
float x203 = x170[x199];
x197 = x203;
} else {
}
x195 += 1;

}
float x210 = x197;
x194[x196] = x210;

}
float* x214 = (float*)myMalloc(520 * sizeof(float));;
int32_t x215 = 0;
for(int x216=0; x216 < 20; x216++) {
for(int x217=0; x217 < 26; x217++) {
int32_t x218 = x215;
float x219 = x170[x218];
float x220 = x194[x216];
float x221 = x219 - x220;
double x222 = (double)x221;
double x223 = exp(x222);
float x224 = (float)x223;
x214[x218] = x224;
x215 += 1;

}

}
float* x231 = (float*)myMalloc(20 * sizeof(float));;
for(int x232=0; x232 < 20; x232++) {
int32_t x233 = x232;
int32_t x234 = x232 * 26;
int32_t x235 = x234;
for(int x236=0; x236 < 26; x236++) {
for(int x238=0; x238 < 1; x238++) {
int32_t x239 = x233;
int32_t x240 = x239 + x238;
float x241 = x231[x240];
int32_t x242 = x235;
int32_t x243 = x242 + x238;
float x244 = x214[x243];
float x245 = x241 + x244;
x231[x240] = x245;

}
x235 += 1;

}

}
x215 = 0;
for(int x255=0; x255 < 20; x255++) {
float x256 = x194[x255];
float x257 = x231[x255];
double x258 = (double)x257;
double x259 = log(x258);
float x260 = (float)x259;
float x261 = x256 + x260;
for(int x262=0; x262 < 26; x262++) {
int32_t x263 = x215;
float x264 = x170[x263];
float x265 = x264 - x261;
x214[x263] = x265;
x215 += 1;

}

}
float* x272 = (float*)myMalloc(520 * sizeof(float));;
// nllLoss forward in CPU
float* x274 = (float*)myMalloc(20 * sizeof(float));;
int32_t x275 = 0;
for(int x276=0; x276 < 20; x276++) {
int32_t x277 = x275;
int32_t x278 = x186[x276];
int32_t x279 = x277 + x278;
float x280 = x214[x279];
float x281 = -1.0f * x280;
x274[x276] = x281;
x275 += 26;

}
float* x286 = (float*)myMalloc(20 * sizeof(float));;
float x287 = 0.0f;
for(int x288=0; x288 < 20; x288++) {
float x289 = x287;
float x290 = x274[x288];
float x291 = x289 + x290;
x287 = x291;

}
float x295 = x287;
float* x296 = (float*)myMalloc(1 * sizeof(float));;
for(int x297=0; x297 < 1; x297++) {
x296[x297] = x295;

}
float* x301 = (float*)myMalloc(1 * sizeof(float));;
if (x303) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x309 = (float*)myMalloc(1 * sizeof(float));;
for(int x310=0; x310 < 1; x310++) {
float x311 = x114[0];
float x312 = x296[0];
float x313 = x311 + x312;
x309[x310] = x313;

}
float* x317 = (float*)myMalloc(1 * sizeof(float));;
float** x747 = (float**)myMalloc(4 * sizeof(float*));;
x747[0] = x309;
x747[1] = x317;
x747[2] = x159;
x747[3] = x169;
int32_t x318 = x112 + 1;
x319(x318,x747);
// back prop for + op
if (x303) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x759=0; x759 < 1; x759++) {
float x760 = x115[0];
float x761 = x317[0];
float x762 = x760 + x761;
x115[0] = x762;

}
if (x303) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x770=0; x770 < 1; x770++) {
float x771 = x301[0];
float x772 = x317[0];
float x773 = x771 + x772;
x301[0] = x773;

}
// 'sum' gradient.
if (x553) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",20,1);
assert(false && "");
}
for(int x782=0; x782 < 20; x782++) {
float x783 = x286[x782];
float x784 = x301[0];
float x785 = x783 + x784;
x286[x782] = x785;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x791 = 0;
for(int x792=0; x792 < 20; x792++) {
int32_t x793 = x791;
int32_t x794 = x186[x792];
int32_t x795 = x793 + x794;
float x796 = x272[x795];
float x797 = x286[x792];
float x798 = -1.0f * x797;
float x799 = x796 + x798;
x272[x795] = x799;
x791 += 26;

}
float* x804 = (float*)myMalloc(20 * sizeof(float));;
for(int x805=0; x805 < 20; x805++) {
int32_t x806 = x805;
int32_t x807 = x805 * 26;
int32_t x808 = x807;
for(int x809=0; x809 < 26; x809++) {
for(int x810=0; x810 < 1; x810++) {
int32_t x811 = x806;
int32_t x812 = x811 + x810;
float x813 = x804[x812];
int32_t x814 = x808;
int32_t x815 = x814 + x810;
float x816 = x272[x815];
float x817 = x813 + x816;
x804[x812] = x817;

}
x808 += 1;

}

}
int32_t x826 = 0;
for(int x827=0; x827 < 20; x827++) {
for(int x828=0; x828 < 26; x828++) {
int32_t x829 = x826;
float x830 = x172[x829];
float x831 = x272[x829];
float x832 = x214[x829];
float x836 = x804[x827];
double x833 = (double)x832;
double x834 = exp(x833);
float x835 = (float)x834;
float x837 = x835 * x836;
float x838 = x831 - x837;
float x839 = x830 + x838;
x172[x829] = x839;
x826 += 1;

}

}
for(int x846=0; x846 < 20; x846++) {
int32_t x848 = 26 * x846;
for(int x847=0; x847 < 26; x847++) {
float x850 = x49[x847];
int32_t x849 = x848 + x847;
float x851 = x172[x849];
float x852 = x850 + x851;
x49[x847] = x852;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x172,26,x39,26,1,x169,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x159,50,x172,26,1,x47,26);
for(int x861=0; x861 < 1000; x861++) {
float x862 = x158[x861];
float x863 = x159[x861];
float x866 = x169[x861];
float x864 = x863 * x863;
float x865 = 1.0f - x864;
float x867 = x865 * x866;
float x868 = x862 + x867;
x158[x861] = x868;

}
// back prop for + op
for(int x873=0; x873 < 20; x873++) {
int32_t x875 = 50 * x873;
for(int x874=0; x874 < 50; x874++) {
int32_t x876 = x875 + x874;
float x877 = x143[x876];
float x878 = x158[x876];
float x879 = x877 + x878;
x143[x876] = x879;

}

}
for(int x885=0; x885 < 20; x885++) {
int32_t x887 = 50 * x885;
for(int x886=0; x886 < 50; x886++) {
float x889 = x38[x886];
int32_t x888 = x887 + x886;
float x890 = x158[x888];
float x891 = x889 + x890;
x38[x886] = x891;

}

}
// back prop for + op
for(int x898=0; x898 < 20; x898++) {
int32_t x900 = 50 * x898;
for(int x899=0; x899 < 50; x899++) {
int32_t x901 = x900 + x899;
float x902 = x124[x901];
float x903 = x143[x901];
float x904 = x902 + x903;
x124[x901] = x904;

}

}
for(int x910=0; x910 < 20; x910++) {
int32_t x912 = 50 * x910;
for(int x911=0; x911 < 50; x911++) {
int32_t x913 = x912 + x911;
float x914 = x127[x913];
float x915 = x143[x913];
float x916 = x914 + x915;
x127[x913] = x916;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x127,50,x27,50,1,x117,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x116,50,x127,50,1,x36,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x124,50,x16,50,1,x121,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x120,26,x124,50,1,x26,50);
} else {
float x929 = 0.0f;
for(int x930=0; x930 < 1; x930++) {
float x931 = x929;
float x932 = x114[x930];
float x933 = x931 + x932;
x929 = x933;

}
float x937 = x929;
float* x938 = (float*)myMalloc(1 * sizeof(float));;
for(int x939=0; x939 < 1; x939++) {
x938[x939] = x937;

}
float* x943 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x945=0; x945 < 1; x945++) {
x943[x945] = 1.0f;

}
// backend is lantern.TensorDsl$BackendCPU@12c72b3d
for(int x950=0; x950 < 1; x950++) {
float x951 = x938[x950];
x86[x950] = x951;

}
// 'sum' gradient.
if (x303) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x960=0; x960 < 1; x960++) {
float x961 = x115[0];
float x962 = x943[0];
float x963 = x961 + x962;
x115[0] = x963;

}
}
};
x109(0,x970);
float x977 = x86[0];
int32_t x978 = x61 % 100;
bool x979 = x978 == 0;
if (x979) {
printf("iter %d, loss %f\n",x61,x977);
int32_t x981 = x61 / 100;
double x982 = (double)x977;
x55[x981] = x982;
} else {
}
for(int x986=0; x986 < 26; x986++) {
float x987 = x49[x986];
float x988 = x987;
float x989 = x988;
bool x990 = x989 > 5.0f;
if (x990) {
x988 = 5.0f;
} else {
}
float x994 = x988;
bool x995 = x994 < -5.0f;
if (x995) {
x988 = -5.0f;
} else {
}
float x999 = x50[x986];
float x1000 = x988;
float x1001 = x1000 * x1000;
float x1002 = x999 + x1001;
x50[x986] = x1002;
float x1004 = x48[x986];
float x1006 = x50[x986];
float x1005 = 0.1f * x1000;
double x1007 = (double)x1006;
double x1008 = x1007 + 9.99999993922529E-9;
double x1009 = sqrt(x1008);
float x1010 = (float)x1009;
float x1011 = x1005 / x1010;
float x1012 = x1004 - x1011;
x48[x986] = x1012;
x49[x986] = 0.0f;

}
for(int x1017=0; x1017 < 1300; x1017++) {
float x1018 = x47[x1017];
float x1019 = x1018;
float x1020 = x1019;
bool x1021 = x1020 > 5.0f;
if (x1021) {
x1019 = 5.0f;
} else {
}
float x1025 = x1019;
bool x1026 = x1025 < -5.0f;
if (x1026) {
x1019 = -5.0f;
} else {
}
float x1030 = x51[x1017];
float x1031 = x1019;
float x1032 = x1031 * x1031;
float x1033 = x1030 + x1032;
x51[x1017] = x1033;
float x1035 = x39[x1017];
float x1037 = x51[x1017];
float x1036 = 0.1f * x1031;
double x1038 = (double)x1037;
double x1039 = x1038 + 9.99999993922529E-9;
double x1040 = sqrt(x1039);
float x1041 = (float)x1040;
float x1042 = x1036 / x1041;
float x1043 = x1035 - x1042;
x39[x1017] = x1043;
x47[x1017] = 0.0f;

}
for(int x1048=0; x1048 < 2500; x1048++) {
float x1049 = x36[x1048];
float x1050 = x1049;
float x1051 = x1050;
bool x1052 = x1051 > 5.0f;
if (x1052) {
x1050 = 5.0f;
} else {
}
float x1056 = x1050;
bool x1057 = x1056 < -5.0f;
if (x1057) {
x1050 = -5.0f;
} else {
}
float x1061 = x52[x1048];
float x1062 = x1050;
float x1063 = x1062 * x1062;
float x1064 = x1061 + x1063;
x52[x1048] = x1064;
float x1066 = x27[x1048];
float x1068 = x52[x1048];
float x1067 = 0.1f * x1062;
double x1069 = (double)x1068;
double x1070 = x1069 + 9.99999993922529E-9;
double x1071 = sqrt(x1070);
float x1072 = (float)x1071;
float x1073 = x1067 / x1072;
float x1074 = x1066 - x1073;
x27[x1048] = x1074;
x36[x1048] = 0.0f;

}
for(int x1079=0; x1079 < 50; x1079++) {
float x1080 = x38[x1079];
float x1081 = x1080;
float x1082 = x1081;
bool x1083 = x1082 > 5.0f;
if (x1083) {
x1081 = 5.0f;
} else {
}
float x1087 = x1081;
bool x1088 = x1087 < -5.0f;
if (x1088) {
x1081 = -5.0f;
} else {
}
float x1092 = x53[x1079];
float x1093 = x1081;
float x1094 = x1093 * x1093;
float x1095 = x1092 + x1094;
x53[x1079] = x1095;
float x1097 = x37[x1079];
float x1099 = x53[x1079];
float x1098 = 0.1f * x1093;
double x1100 = (double)x1099;
double x1101 = x1100 + 9.99999993922529E-9;
double x1102 = sqrt(x1101);
float x1103 = (float)x1102;
float x1104 = x1098 / x1103;
float x1105 = x1097 - x1104;
x37[x1079] = x1105;
x38[x1079] = 0.0f;

}
for(int x1110=0; x1110 < 1300; x1110++) {
float x1111 = x26[x1110];
float x1112 = x1111;
float x1113 = x1112;
bool x1114 = x1113 > 5.0f;
if (x1114) {
x1112 = 5.0f;
} else {
}
float x1118 = x1112;
bool x1119 = x1118 < -5.0f;
if (x1119) {
x1112 = -5.0f;
} else {
}
float x1123 = x54[x1110];
float x1124 = x1112;
float x1125 = x1124 * x1124;
float x1126 = x1123 + x1125;
x54[x1110] = x1126;
float x1128 = x16[x1110];
float x1130 = x54[x1110];
float x1129 = 0.1f * x1124;
double x1131 = (double)x1130;
double x1132 = x1131 + 9.99999993922529E-9;
double x1133 = sqrt(x1132);
float x1134 = (float)x1133;
float x1135 = x1129 / x1134;
float x1136 = x1128 - x1135;
x16[x1110] = x1136;
x26[x1110] = 0.0f;

}
int64_t x1141 = (long)mallocAddr;
int64_t x1142 = x1141 - x57;
memset((void*)x57, 0, x1142);
mallocAddr = (void*)x57;

}
double x1147 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x1150 = (long)fopen(x0, "w");
fprintf((FILE *)x1150, "unit: %s\n", "100 iteration");
for(int x1153=0; x1153 < 51; x1153++) {
double x1154 = x55[x1153];
fprintf((FILE *)x1150, "%lf\n", x1154);

}
double x1148 = x56 - x1;
double x1149 = x1147 - x56;
fprintf((FILE *)x1150, "run time: %lf %lf\n", x1148, x1149);
fclose((FILE*)x1150);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

