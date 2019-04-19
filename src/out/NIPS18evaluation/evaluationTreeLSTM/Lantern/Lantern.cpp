
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
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
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
  int res = fstat(fd,&stat);
  return stat.st_size;
}
int printll(char* s) {
  while (*s != '\n' && *s != ',' && *s != '\t') {
    putchar(*s++);
  }
  return 0;
}
long hash(char *str0, int len) {
  unsigned char* str = (unsigned char*)str0;
  unsigned long hash = 5381;
  int c;

  while ((c = *str++) && len--)
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

  return hash;
}

long HEAP_SIZE_CPU = 1073741826;
void *mallocBase = calloc(HEAP_SIZE_CPU, 1);
void *mallocAddr = mallocBase;
void *waterMark = mallocBase;
void *myMalloc(size_t bytes) {
  void *res = mallocAddr;
  mallocAddr = (void *)((char *)mallocAddr + bytes);
  if ((long)mallocAddr >= (long)mallocBase + HEAP_SIZE_CPU) {
    fprintf(stderr, "CPU memory breached limit of HEAP_SIZE_CPU\n"); abort();
  }
  return res;
}
long HEAP_SIZE = 8589934608; //  4294967304; // this is for GPU
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
  long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
  result->tv_sec = diff / 1000000;
  result->tv_usec = diff % 1000000;
  return (diff < 0);
}



void Snippet(char*);
//std::random_device rd{};
//std::mt19937 gen{rd()};
//std::normal_distribution<> d{0, 0.01};

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
   
void Snippet(char* x0) {
// Backend setup.;
double x1 = ((double)clock() / CLOCKS_PER_SEC);
int* x2 = (int*)myMalloc(1 * sizeof(int));
long x3 = (long)fopen("small_glove.txt", "r");
if (fscanf((FILE *)x3,"%d", &x2[0])!=1) perror("Error reading file");
int x4 = x2[0];
float** x5 = (float**)myMalloc(x4 * sizeof(float*));
int x6 = 0;
while (x6 != x4) {
int x7 = x6;
x5[x7] = (float*)myMalloc(300 * sizeof(float));
int x8 = 0;
while (x8 != 300) {
if (fscanf((FILE *)x3,"%f", &x5[x7][x8])!=1) perror("Error reading file");
x8 = x8 + 1;
}
x6 = x6 + 1;
}
fclose((FILE*)x3);
int* x9 = (int*)myMalloc(1 * sizeof(int));
long x10 = (long)fopen("array_tree.txt", "r");
if (fscanf((FILE *)x10,"%d", &x9[0])!=1) perror("Error reading file");
int x11 = x9[0];
int** x12 = (int**)myMalloc(x11 * 4 * sizeof(int*));
int* x13 = (int*)myMalloc(1 * sizeof(int));
int x14 = 0;
while (x14 != x11) {
if (fscanf((FILE *)x10,"%d", &x13[0])!=1) perror("Error reading file");
int x15 = 0;
int x16 = x14 * 4;
while (x15 != 4) {
int x17 = x16 + x15;
x12[x17] = (int*)myMalloc(x13[0] * sizeof(int));
int x18 = x13[0];
int x19 = 0;
while (x19 != x18) {
if (fscanf((FILE *)x10,"%d", &x12[x17][x19])!=1) perror("Error reading file");
x19 = x19 + 1;
}
x15 = x15 + 1;
}
x14 = x14 + 1;
}
fclose((FILE*)x10);
float* x20 = (float*)myMalloc(45000 * sizeof(float));
int x21 = 0;
while (x21 != 45000) {
x20[x21] = (((float)rand()/RAND_MAX) - 0.5) * 0.01;
x21 = x21 + 1;
}
float* x22 = (float*)myMalloc(150 * sizeof(float));
float* x23 = (float*)myMalloc(45000 * sizeof(float));
int x24 = 0;
while (x24 != 45000) {
x23[x24] = (((float)rand()/RAND_MAX) - 0.5) * 0.01;
x24 = x24 + 1;
}
float* x25 = (float*)myMalloc(150 * sizeof(float));
float* x26 = (float*)myMalloc(45000 * sizeof(float));
int x27 = 0;
while (x27 != 45000) {
x26[x27] = (((float)rand()/RAND_MAX) - 0.5) * 0.01;
x27 = x27 + 1;
}
float* x28 = (float*)myMalloc(150 * sizeof(float));
float* x29 = (float*)myMalloc(22500 * sizeof(float));
int x30 = 0;
while (x30 != 22500) {
x29[x30] = (((float)rand()/RAND_MAX) - 0.5) * 0.01;
x30 = x30 + 1;
}
float* x31 = (float*)myMalloc(22500 * sizeof(float));
int x32 = 0;
while (x32 != 22500) {
x31[x32] = (((float)rand()/RAND_MAX) - 0.5) * 0.01;
x32 = x32 + 1;
}
float* x33 = (float*)myMalloc(150 * sizeof(float));
float* x34 = (float*)myMalloc(22500 * sizeof(float));
int x35 = 0;
while (x35 != 22500) {
x34[x35] = (((float)rand()/RAND_MAX) - 0.5) * 0.01;
x35 = x35 + 1;
}
float* x36 = (float*)myMalloc(22500 * sizeof(float));
int x37 = 0;
while (x37 != 22500) {
x36[x37] = (((float)rand()/RAND_MAX) - 0.5) * 0.01;
x37 = x37 + 1;
}
float* x38 = (float*)myMalloc(22500 * sizeof(float));
int x39 = 0;
while (x39 != 22500) {
x38[x39] = (((float)rand()/RAND_MAX) - 0.5) * 0.01;
x39 = x39 + 1;
}
float* x40 = (float*)myMalloc(22500 * sizeof(float));
int x41 = 0;
while (x41 != 22500) {
x40[x41] = (((float)rand()/RAND_MAX) - 0.5) * 0.01;
x41 = x41 + 1;
}
float* x42 = (float*)myMalloc(150 * sizeof(float));
float* x43 = (float*)myMalloc(22500 * sizeof(float));
int x44 = 0;
while (x44 != 22500) {
x43[x44] = (((float)rand()/RAND_MAX) - 0.5) * 0.01;
x44 = x44 + 1;
}
float* x45 = (float*)myMalloc(22500 * sizeof(float));
int x46 = 0;
while (x46 != 22500) {
x45[x46] = (((float)rand()/RAND_MAX) - 0.5) * 0.01;
x46 = x46 + 1;
}
float* x47 = (float*)myMalloc(150 * sizeof(float));
float* x48 = (float*)myMalloc(22500 * sizeof(float));
int x49 = 0;
while (x49 != 22500) {
x48[x49] = (((float)rand()/RAND_MAX) - 0.5) * 0.01;
x49 = x49 + 1;
}
float* x50 = (float*)myMalloc(22500 * sizeof(float));
int x51 = 0;
while (x51 != 22500) {
x50[x51] = (((float)rand()/RAND_MAX) - 0.5) * 0.01;
x51 = x51 + 1;
}
float* x52 = (float*)myMalloc(150 * sizeof(float));
float* x53 = (float*)myMalloc(750 * sizeof(float));
int x54 = 0;
while (x54 != 750) {
x53[x54] = (((float)rand()/RAND_MAX) - 0.5) * 0.01;
x54 = x54 + 1;
}
float* x55 = (float*)myMalloc(5 * sizeof(float));
float* x56 = (float*)myMalloc(45000 * sizeof(float));
float* x57 = (float*)myMalloc(150 * sizeof(float));
float* x58 = (float*)myMalloc(45000 * sizeof(float));
float* x59 = (float*)myMalloc(150 * sizeof(float));
float* x60 = (float*)myMalloc(45000 * sizeof(float));
float* x61 = (float*)myMalloc(150 * sizeof(float));
float* x62 = (float*)myMalloc(22500 * sizeof(float));
float* x63 = (float*)myMalloc(22500 * sizeof(float));
float* x64 = (float*)myMalloc(150 * sizeof(float));
float* x65 = (float*)myMalloc(22500 * sizeof(float));
float* x66 = (float*)myMalloc(22500 * sizeof(float));
float* x67 = (float*)myMalloc(22500 * sizeof(float));
float* x68 = (float*)myMalloc(22500 * sizeof(float));
float* x69 = (float*)myMalloc(150 * sizeof(float));
float* x70 = (float*)myMalloc(22500 * sizeof(float));
float* x71 = (float*)myMalloc(22500 * sizeof(float));
float* x72 = (float*)myMalloc(150 * sizeof(float));
float* x73 = (float*)myMalloc(22500 * sizeof(float));
float* x74 = (float*)myMalloc(22500 * sizeof(float));
float* x75 = (float*)myMalloc(150 * sizeof(float));
float* x76 = (float*)myMalloc(750 * sizeof(float));
float* x77 = (float*)myMalloc(5 * sizeof(float));
float* x78 = (float*)myMalloc(45000 * sizeof(float));
float* x79 = (float*)myMalloc(150 * sizeof(float));
float* x80 = (float*)myMalloc(45000 * sizeof(float));
float* x81 = (float*)myMalloc(150 * sizeof(float));
float* x82 = (float*)myMalloc(45000 * sizeof(float));
float* x83 = (float*)myMalloc(150 * sizeof(float));
float* x84 = (float*)myMalloc(22500 * sizeof(float));
float* x85 = (float*)myMalloc(22500 * sizeof(float));
float* x86 = (float*)myMalloc(150 * sizeof(float));
float* x87 = (float*)myMalloc(22500 * sizeof(float));
float* x88 = (float*)myMalloc(22500 * sizeof(float));
float* x89 = (float*)myMalloc(22500 * sizeof(float));
float* x90 = (float*)myMalloc(22500 * sizeof(float));
float* x91 = (float*)myMalloc(150 * sizeof(float));
float* x92 = (float*)myMalloc(22500 * sizeof(float));
float* x93 = (float*)myMalloc(22500 * sizeof(float));
float* x94 = (float*)myMalloc(150 * sizeof(float));
float* x95 = (float*)myMalloc(22500 * sizeof(float));
float* x96 = (float*)myMalloc(22500 * sizeof(float));
float* x97 = (float*)myMalloc(150 * sizeof(float));
float* x98 = (float*)myMalloc(750 * sizeof(float));
float* x99 = (float*)myMalloc(5 * sizeof(float));
double* x100 = (double*)myMalloc(6 * sizeof(double));
long x101 = (long)mallocAddr;
double x102 = ((double)clock() / CLOCKS_PER_SEC);
int x103 = 0;
//# lambda forward is here!
float x104 = (float)1;
while (x103 != 6) {
int x105 = x103;
float x106 = 0.0;
int x107 = 0;
while (x107 != x11) {
int x108 = x107;
int x109 = x108 % x11 * 4;
int* x110 = x12[x109];
int* x111 = x12[x109 + 1];
int* x112 = x12[x109 + 2];
int* x113 = x12[x109 + 3];
(float*)myMalloc(1 * sizeof(float));
(float*)myMalloc(1 * sizeof(float));
// allocate memory to save the final loss in CPU Tensor;
float* x114 = (float*)myMalloc(1 * sizeof(float));
float* x115 = (float*)myMalloc(1 * sizeof(float));
float* x116 = (float*)myMalloc(1 * sizeof(float));
float* x117 = (float*)myMalloc(150 * sizeof(float));
float* x118 = (float*)myMalloc(150 * sizeof(float));
float* x119 = (float*)myMalloc(150 * sizeof(float));
float* x120 = (float*)myMalloc(150 * sizeof(float));
function<void(float**)> x121 = [&](float** x123) {float* x122 = x123[0];
float* x124 = x123[1];
// make sure the size of loss is 1;
int x125 = 0;
while (x125 != 1) {
x124[x125] = x104;
x125 = x125 + 1;
}
// backend is lantern.TensorDslCPU$BackendCPU@67a4ebae;
int x126 = 0;
while (x126 != 1) {
int x127 = x126;
x114[x127] = x122[x127];
x126 = x126 + 1;
}};
float** x128 = (float**)myMalloc(6 * sizeof(float*));
x128[0] = x115;
x128[1] = x116;
x128[2] = x117;
x128[3] = x118;
x128[4] = x119;
x128[5] = x120;
function<void(int,function<void(float**)>,float**)> x129 = [&](int x130, function<void(float**)> x135, float** x592) {if (x130 >= 0) {
int* x131 = x110+x130;
function<void(float**)> x132 = [&](float** x134) {float** x133 = (float**)myMalloc(6 * sizeof(float*));
x133[0] = x134[0];
x133[1] = x134[1];
x133[2] = x134[2];
x133[3] = x134[3];
x133[4] = x134[4];
x133[5] = x134[5];
x135(x133);};
function<void(float**)> x136 = [&](float** x138) {float** x137 = (float**)myMalloc(6 * sizeof(float*));
x137[0] = x138[0];
x137[1] = x138[1];
x137[2] = x138[2];
x137[3] = x138[3];
x137[4] = x138[4];
x137[5] = x138[5];
x132(x137);};
function<void(float**)> x139 = [&](float** x141) {float** x140 = (float**)myMalloc(6 * sizeof(float*));
x140[0] = x141[0];
x140[1] = x141[1];
x140[2] = x141[2];
x140[3] = x141[3];
x140[4] = x141[4];
x140[5] = x141[5];
x135(x140);};
function<void(float**)> x142 = [&](float** x144) {float** x143 = (float**)myMalloc(6 * sizeof(float*));
x143[0] = x144[0];
x143[1] = x144[1];
x143[2] = x144[2];
x143[3] = x144[3];
x143[4] = x144[4];
x143[5] = x144[5];
x139(x143);};
function<void(float**)> x145 = [&](float** x147) {float* x146 = x147[0];
float* x148 = x147[1];
float* x149 = x147[2];
float* x150 = x147[3];
float* x151 = x147[4];
float* x152 = x147[5];
function<void(float**)> x153 = [&](float** x155) {float* x154 = x155[0];
float* x156 = x155[1];
float* x157 = x155[2];
float* x158 = x155[3];
float* x159 = x155[4];
float* x160 = x155[5];
((float*)myMalloc(5 * sizeof(float)))[x110[x130]] = x104;
(float*)myMalloc(5 * sizeof(float));
if (x112[x130] < 0) {
float* x161 = x5[x111[x130]];
float* x162 = (float*)myMalloc(300 * sizeof(float));
float* x163 = (float*)myMalloc(150 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x20,300,x161,1,0,x163,1);
float* x164 = (float*)myMalloc(150 * sizeof(float));
float* x165 = (float*)myMalloc(150 * sizeof(float));
int x166 = 0;
while (x166 != 150) {
int x167 = x166;
x165[x167] = x163[x167] + x22[x167];
x166 = x166 + 1;
}
float* x168 = (float*)myMalloc(150 * sizeof(float));
float* x169 = (float*)myMalloc(150 * sizeof(float));
int x170 = 0;
while (x170 != 150) {
int x171 = x170;
x169[x171] = 1.0 / ((float)exp((double)(-1.0 * x165[x171])) + 1.0);
x170 = x170 + 1;
}
float* x172 = (float*)myMalloc(150 * sizeof(float));
float* x173 = (float*)myMalloc(150 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x23,300,x161,1,0,x173,1);
float* x174 = (float*)myMalloc(150 * sizeof(float));
float* x175 = (float*)myMalloc(150 * sizeof(float));
int x176 = 0;
while (x176 != 150) {
int x177 = x176;
x175[x177] = x173[x177] + x25[x177];
x176 = x176 + 1;
}
float* x178 = (float*)myMalloc(150 * sizeof(float));
float* x179 = (float*)myMalloc(150 * sizeof(float));
int x180 = 0;
while (x180 != 150) {
int x181 = x180;
x179[x181] = 1.0 / ((float)exp((double)(-1.0 * x175[x181])) + 1.0);
x180 = x180 + 1;
}
float* x182 = (float*)myMalloc(150 * sizeof(float));
float* x183 = (float*)myMalloc(150 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,300,1,x26,300,x161,1,0,x183,1);
float* x184 = (float*)myMalloc(150 * sizeof(float));
float* x185 = (float*)myMalloc(150 * sizeof(float));
int x186 = 0;
while (x186 != 150) {
int x187 = x186;
x185[x187] = x183[x187] + x28[x187];
x186 = x186 + 1;
}
float* x188 = (float*)myMalloc(150 * sizeof(float));
float* x189 = (float*)myMalloc(150 * sizeof(float));
int x190 = 0;
while (x190 != 150) {
int x191 = x190;
x189[x191] = (float)tanh((double)x185[x191]);
x190 = x190 + 1;
}
float* x192 = (float*)myMalloc(150 * sizeof(float));
float* x193 = (float*)myMalloc(150 * sizeof(float));
int x194 = 0;
while (x194 != 150) {
int x195 = x194;
x193[x195] = x169[x195] * x189[x195];
x194 = x194 + 1;
}
float* x196 = (float*)myMalloc(150 * sizeof(float));
float* x197 = (float*)myMalloc(150 * sizeof(float));
int x198 = 0;
while (x198 != 150) {
int x199 = x198;
x197[x199] = (float)tanh((double)x193[x199]);
x198 = x198 + 1;
}
float* x200 = (float*)myMalloc(150 * sizeof(float));
float* x201 = (float*)myMalloc(150 * sizeof(float));
int x202 = 0;
while (x202 != 150) {
int x203 = x202;
x201[x203] = x179[x203] * x197[x203];
x202 = x202 + 1;
}
float* x204 = (float*)myMalloc(150 * sizeof(float));
float* x205 = (float*)myMalloc(5 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x53,150,x201,1,0,x205,1);
float* x206 = (float*)myMalloc(5 * sizeof(float));
float* x207 = (float*)myMalloc(5 * sizeof(float));
int x208 = 0;
while (x208 != 5) {
int x209 = x208;
x207[x209] = x205[x209] + x55[x209];
x208 = x208 + 1;
}
float* x210 = (float*)myMalloc(5 * sizeof(float));
int x211 = 0;
int x212 = 1;
x212 = x212;
x212 = x212 * 5;
if (x211 == 0) {

} else {

}
float* x213 = (float*)myMalloc(1 * sizeof(float));
int x214 = 0;
int x215 = 0;
while (x215 != 1) {
float x216 = -3.4028235E38;
int x217 = 0;
while (x217 != 5) {
if (x207[x214] > x216) x216 = x207[x214]; else {

}
x214 = x214 + 1;
x217 = x217 + 1;
}
x213[x215] = x216;
x215 = x215 + 1;
}
float* x218 = (float*)myMalloc(5 * sizeof(float));
int x219 = 0;
int x220 = 0;
while (x220 != 1) {
int x221 = x220;
int x222 = 0;
while (x222 != 5) {
x218[x219] = (float)exp((double)(x207[x219] - x213[x221]));
x219 = x219 + 1;
x222 = x222 + 1;
}
x220 = x220 + 1;
}
float* x223 = (float*)myMalloc(1 * sizeof(float));
int x224 = 0;
while (x224 != 1) {
int x225 = x224;
int x226 = x225;
int x227 = x225 * 5;
int x228 = 0;
while (x228 != 5) {
int x229 = 0;
while (x229 != 1) {
int x230 = x229;
int x231 = x226 + x230;
x223[x231] = x223[x231] + x218[x227 + x230];
x229 = x229 + 1;
}
x227 = x227 + 1;
x228 = x228 + 1;
}
x224 = x224 + 1;
}
x219 = 0;
int x232 = 0;
while (x232 != 1) {
int x233 = x232;
float x234 = x213[x233] + (float)log((double)x223[x233]);
int x235 = 0;
while (x235 != 5) {
x218[x219] = x207[x219] - x234;
x219 = x219 + 1;
x235 = x235 + 1;
}
x232 = x232 + 1;
}
float* x236 = (float*)myMalloc(5 * sizeof(float));
// nllLoss forward in CPU;
float* x237 = (float*)myMalloc(1 * sizeof(float));
int x238 = 0;
int x239 = 0;
while (x239 != 1) {
int x240 = x239;
x237[x240] = -1.0 * x218[x238 + x131[x240]];
x238 = x238 + 5;
x239 = x239 + 1;
}
float* x241 = (float*)myMalloc(1 * sizeof(float));
float** x242 = (float**)myMalloc(6 * sizeof(float*));
x242[0] = x237;
x242[1] = x241;
x242[2] = x201;
x242[3] = x204;
x242[4] = x193;
x242[5] = x196;
x136(x242);
// 'nllLossB' gradient.;
// nllLoss_grad implementation in CPU;
int x243 = 0;
int x244 = 0;
while (x244 != 1) {
int x245 = x244;
int x246 = x243 + x131[x245];
x236[x246] = x236[x246] + -1.0 * x241[x245];
x243 = x243 + 5;
x244 = x244 + 1;
}
float* x247 = (float*)myMalloc(1 * sizeof(float));
int x248 = 0;
while (x248 != 1) {
int x249 = x248;
int x250 = x249;
int x251 = x249 * 5;
int x252 = 0;
while (x252 != 5) {
int x253 = 0;
while (x253 != 1) {
int x254 = x253;
int x255 = x250 + x254;
x247[x255] = x247[x255] + x236[x251 + x254];
x253 = x253 + 1;
}
x251 = x251 + 1;
x252 = x252 + 1;
}
x248 = x248 + 1;
}
int x256 = 0;
int x257 = 0;
while (x257 != 1) {
int x258 = x257;
int x259 = 0;
while (x259 != 5) {
int x260 = x256;
x210[x260] = x210[x260] + (x236[x256] - (float)exp((double)x218[x256]) * x247[x258]);
x256 = x256 + 1;
x259 = x259 + 1;
}
x257 = x257 + 1;
}
// back prop for + op;
int x261 = 0;
while (x261 != 5) {
int x262 = x261;
x206[x262] = x206[x262] + x210[x262];
x77[x262] = x77[x262] + x210[x262];
x261 = x261 + 1;
}
// backend add_cartesian;
int x263 = 0;
int x264 = 0;
while (x264 != 5) {
int x265 = x264;
int x266 = 0;
while (x266 != 150) {
int x267 = x266;
x76[x263 + x267] = x76[x263 + x267] + x201[x267] * x206[x265];
x266 = x266 + 1;
}
x263 = x263 + 150;
x264 = x264 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x53,150,x206,1,1,x204,1);
// backprop for * op;
int x268 = 0;
while (x268 != 150) {
int x269 = x268;
x182[x269] = x182[x269] + x204[x269] * x197[x269];
x200[x269] = x200[x269] + x204[x269] * x179[x269];
x268 = x268 + 1;
}
int x270 = 0;
while (x270 != 150) {
int x271 = x270;
float x272 = x197[x271];
x196[x271] = x196[x271] + (1.0 - x272 * x272) * x200[x271];
x270 = x270 + 1;
}
// backprop for * op;
int x273 = 0;
while (x273 != 150) {
int x274 = x273;
x172[x274] = x172[x274] + x196[x274] * x189[x274];
x192[x274] = x192[x274] + x196[x274] * x169[x274];
x273 = x273 + 1;
}
int x275 = 0;
while (x275 != 150) {
int x276 = x275;
float x277 = x189[x276];
x188[x276] = x188[x276] + (1.0 - x277 * x277) * x192[x276];
x275 = x275 + 1;
}
// back prop for + op;
int x278 = 0;
while (x278 != 150) {
int x279 = x278;
x184[x279] = x184[x279] + x188[x279];
x61[x279] = x61[x279] + x188[x279];
x278 = x278 + 1;
}
// backend add_cartesian;
int x280 = 0;
int x281 = 0;
while (x281 != 150) {
int x282 = x281;
int x283 = 0;
while (x283 != 300) {
int x284 = x283;
x60[x280 + x284] = x60[x280 + x284] + x161[x284] * x184[x282];
x283 = x283 + 1;
}
x280 = x280 + 300;
x281 = x281 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x26,300,x184,1,1,x162,1);
int x285 = 0;
while (x285 != 150) {
int x286 = x285;
float x287 = x179[x286];
x178[x286] = x178[x286] + (1.0 - x287) * x287 * x182[x286];
x285 = x285 + 1;
}
// back prop for + op;
int x288 = 0;
while (x288 != 150) {
int x289 = x288;
x174[x289] = x174[x289] + x178[x289];
x59[x289] = x59[x289] + x178[x289];
x288 = x288 + 1;
}
// backend add_cartesian;
int x290 = 0;
int x291 = 0;
while (x291 != 150) {
int x292 = x291;
int x293 = 0;
while (x293 != 300) {
int x294 = x293;
x58[x290 + x294] = x58[x290 + x294] + x161[x294] * x174[x292];
x293 = x293 + 1;
}
x290 = x290 + 300;
x291 = x291 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x23,300,x174,1,1,x162,1);
int x295 = 0;
while (x295 != 150) {
int x296 = x295;
float x297 = x169[x296];
x168[x296] = x168[x296] + (1.0 - x297) * x297 * x172[x296];
x295 = x295 + 1;
}
// back prop for + op;
int x298 = 0;
while (x298 != 150) {
int x299 = x298;
x164[x299] = x164[x299] + x168[x299];
x57[x299] = x57[x299] + x168[x299];
x298 = x298 + 1;
}
// backend add_cartesian;
int x300 = 0;
int x301 = 0;
while (x301 != 150) {
int x302 = x301;
int x303 = 0;
while (x303 != 300) {
int x304 = x303;
x56[x300 + x304] = x56[x300 + x304] + x161[x304] * x164[x302];
x303 = x303 + 1;
}
x300 = x300 + 300;
x301 = x301 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 150,300,1,x20,300,x164,1,1,x162,1);
} else {
float* x305 = (float*)myMalloc(150 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x29,150,x149,1,0,x305,1);
float* x306 = (float*)myMalloc(150 * sizeof(float));
float* x307 = (float*)myMalloc(150 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x31,150,x157,1,0,x307,1);
float* x308 = (float*)myMalloc(150 * sizeof(float));
float* x309 = (float*)myMalloc(150 * sizeof(float));
int x310 = 0;
while (x310 != 150) {
int x311 = x310;
x309[x311] = x305[x311] + x307[x311];
x310 = x310 + 1;
}
float* x312 = (float*)myMalloc(150 * sizeof(float));
float* x313 = (float*)myMalloc(150 * sizeof(float));
int x314 = 0;
while (x314 != 150) {
int x315 = x314;
x313[x315] = x309[x315] + x33[x315];
x314 = x314 + 1;
}
float* x316 = (float*)myMalloc(150 * sizeof(float));
float* x317 = (float*)myMalloc(150 * sizeof(float));
int x318 = 0;
while (x318 != 150) {
int x319 = x318;
x317[x319] = 1.0 / ((float)exp((double)(-1.0 * x313[x319])) + 1.0);
x318 = x318 + 1;
}
float* x320 = (float*)myMalloc(150 * sizeof(float));
float* x321 = (float*)myMalloc(150 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x34,150,x149,1,0,x321,1);
float* x322 = (float*)myMalloc(150 * sizeof(float));
float* x323 = (float*)myMalloc(150 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x36,150,x157,1,0,x323,1);
float* x324 = (float*)myMalloc(150 * sizeof(float));
float* x325 = (float*)myMalloc(150 * sizeof(float));
int x326 = 0;
while (x326 != 150) {
int x327 = x326;
x325[x327] = x321[x327] + x323[x327];
x326 = x326 + 1;
}
float* x328 = (float*)myMalloc(150 * sizeof(float));
float* x329 = (float*)myMalloc(150 * sizeof(float));
int x330 = 0;
while (x330 != 150) {
int x331 = x330;
x329[x331] = x325[x331] + x42[x331];
x330 = x330 + 1;
}
float* x332 = (float*)myMalloc(150 * sizeof(float));
float* x333 = (float*)myMalloc(150 * sizeof(float));
int x334 = 0;
while (x334 != 150) {
int x335 = x334;
x333[x335] = 1.0 / ((float)exp((double)(-1.0 * x329[x335])) + 1.0);
x334 = x334 + 1;
}
float* x336 = (float*)myMalloc(150 * sizeof(float));
float* x337 = (float*)myMalloc(150 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x38,150,x149,1,0,x337,1);
float* x338 = (float*)myMalloc(150 * sizeof(float));
float* x339 = (float*)myMalloc(150 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x40,150,x157,1,0,x339,1);
float* x340 = (float*)myMalloc(150 * sizeof(float));
float* x341 = (float*)myMalloc(150 * sizeof(float));
int x342 = 0;
while (x342 != 150) {
int x343 = x342;
x341[x343] = x337[x343] + x339[x343];
x342 = x342 + 1;
}
float* x344 = (float*)myMalloc(150 * sizeof(float));
float* x345 = (float*)myMalloc(150 * sizeof(float));
int x346 = 0;
while (x346 != 150) {
int x347 = x346;
x345[x347] = x341[x347] + x42[x347];
x346 = x346 + 1;
}
float* x348 = (float*)myMalloc(150 * sizeof(float));
float* x349 = (float*)myMalloc(150 * sizeof(float));
int x350 = 0;
while (x350 != 150) {
int x351 = x350;
x349[x351] = 1.0 / ((float)exp((double)(-1.0 * x345[x351])) + 1.0);
x350 = x350 + 1;
}
float* x352 = (float*)myMalloc(150 * sizeof(float));
float* x353 = (float*)myMalloc(150 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x43,150,x149,1,0,x353,1);
float* x354 = (float*)myMalloc(150 * sizeof(float));
float* x355 = (float*)myMalloc(150 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x45,150,x157,1,0,x355,1);
float* x356 = (float*)myMalloc(150 * sizeof(float));
float* x357 = (float*)myMalloc(150 * sizeof(float));
int x358 = 0;
while (x358 != 150) {
int x359 = x358;
x357[x359] = x353[x359] + x355[x359];
x358 = x358 + 1;
}
float* x360 = (float*)myMalloc(150 * sizeof(float));
float* x361 = (float*)myMalloc(150 * sizeof(float));
int x362 = 0;
while (x362 != 150) {
int x363 = x362;
x361[x363] = x357[x363] + x47[x363];
x362 = x362 + 1;
}
float* x364 = (float*)myMalloc(150 * sizeof(float));
float* x365 = (float*)myMalloc(150 * sizeof(float));
int x366 = 0;
while (x366 != 150) {
int x367 = x366;
x365[x367] = 1.0 / ((float)exp((double)(-1.0 * x361[x367])) + 1.0);
x366 = x366 + 1;
}
float* x368 = (float*)myMalloc(150 * sizeof(float));
float* x369 = (float*)myMalloc(150 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x48,150,x149,1,0,x369,1);
float* x370 = (float*)myMalloc(150 * sizeof(float));
float* x371 = (float*)myMalloc(150 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 150,150,1,x50,150,x157,1,0,x371,1);
float* x372 = (float*)myMalloc(150 * sizeof(float));
float* x373 = (float*)myMalloc(150 * sizeof(float));
int x374 = 0;
while (x374 != 150) {
int x375 = x374;
x373[x375] = x369[x375] + x371[x375];
x374 = x374 + 1;
}
float* x376 = (float*)myMalloc(150 * sizeof(float));
float* x377 = (float*)myMalloc(150 * sizeof(float));
int x378 = 0;
while (x378 != 150) {
int x379 = x378;
x377[x379] = x373[x379] + x52[x379];
x378 = x378 + 1;
}
float* x380 = (float*)myMalloc(150 * sizeof(float));
float* x381 = (float*)myMalloc(150 * sizeof(float));
int x382 = 0;
while (x382 != 150) {
int x383 = x382;
x381[x383] = (float)tanh((double)x377[x383]);
x382 = x382 + 1;
}
float* x384 = (float*)myMalloc(150 * sizeof(float));
float* x385 = (float*)myMalloc(150 * sizeof(float));
int x386 = 0;
while (x386 != 150) {
int x387 = x386;
x385[x387] = x317[x387] * x381[x387];
x386 = x386 + 1;
}
float* x388 = (float*)myMalloc(150 * sizeof(float));
float* x389 = (float*)myMalloc(150 * sizeof(float));
int x390 = 0;
while (x390 != 150) {
int x391 = x390;
x389[x391] = x333[x391] * x151[x391];
x390 = x390 + 1;
}
float* x392 = (float*)myMalloc(150 * sizeof(float));
float* x393 = (float*)myMalloc(150 * sizeof(float));
int x394 = 0;
while (x394 != 150) {
int x395 = x394;
x393[x395] = x385[x395] + x389[x395];
x394 = x394 + 1;
}
float* x396 = (float*)myMalloc(150 * sizeof(float));
float* x397 = (float*)myMalloc(150 * sizeof(float));
int x398 = 0;
while (x398 != 150) {
int x399 = x398;
x397[x399] = x349[x399] * x159[x399];
x398 = x398 + 1;
}
float* x400 = (float*)myMalloc(150 * sizeof(float));
float* x401 = (float*)myMalloc(150 * sizeof(float));
int x402 = 0;
while (x402 != 150) {
int x403 = x402;
x401[x403] = x393[x403] + x397[x403];
x402 = x402 + 1;
}
float* x404 = (float*)myMalloc(150 * sizeof(float));
float* x405 = (float*)myMalloc(150 * sizeof(float));
int x406 = 0;
while (x406 != 150) {
int x407 = x406;
x405[x407] = (float)tanh((double)x401[x407]);
x406 = x406 + 1;
}
float* x408 = (float*)myMalloc(150 * sizeof(float));
float* x409 = (float*)myMalloc(150 * sizeof(float));
int x410 = 0;
while (x410 != 150) {
int x411 = x410;
x409[x411] = x365[x411] * x405[x411];
x410 = x410 + 1;
}
float* x412 = (float*)myMalloc(150 * sizeof(float));
float* x413 = (float*)myMalloc(5 * sizeof(float));
cblas_sgemv(CblasRowMajor, CblasNoTrans, 5,150,1,x53,150,x409,1,0,x413,1);
float* x414 = (float*)myMalloc(5 * sizeof(float));
float* x415 = (float*)myMalloc(5 * sizeof(float));
int x416 = 0;
while (x416 != 5) {
int x417 = x416;
x415[x417] = x413[x417] + x55[x417];
x416 = x416 + 1;
}
float* x418 = (float*)myMalloc(5 * sizeof(float));
float* x419 = (float*)myMalloc(1 * sizeof(float));
int x420 = 0;
while (x420 != 1) {
x419[x420] = x146[0] + x154[0];
x420 = x420 + 1;
}
float* x421 = (float*)myMalloc(1 * sizeof(float));
int x422 = 0;
int x423 = 1;
x423 = x423;
x423 = x423 * 5;
if (x422 == 0) {

} else {

}
float* x424 = (float*)myMalloc(1 * sizeof(float));
int x425 = 0;
int x426 = 0;
while (x426 != 1) {
float x427 = -3.4028235E38;
int x428 = 0;
while (x428 != 5) {
if (x415[x425] > x427) x427 = x415[x425]; else {

}
x425 = x425 + 1;
x428 = x428 + 1;
}
x424[x426] = x427;
x426 = x426 + 1;
}
float* x429 = (float*)myMalloc(5 * sizeof(float));
int x430 = 0;
int x431 = 0;
while (x431 != 1) {
int x432 = x431;
int x433 = 0;
while (x433 != 5) {
x429[x430] = (float)exp((double)(x415[x430] - x424[x432]));
x430 = x430 + 1;
x433 = x433 + 1;
}
x431 = x431 + 1;
}
float* x434 = (float*)myMalloc(1 * sizeof(float));
int x435 = 0;
while (x435 != 1) {
int x436 = x435;
int x437 = x436;
int x438 = x436 * 5;
int x439 = 0;
while (x439 != 5) {
int x440 = 0;
while (x440 != 1) {
int x441 = x440;
int x442 = x437 + x441;
x434[x442] = x434[x442] + x429[x438 + x441];
x440 = x440 + 1;
}
x438 = x438 + 1;
x439 = x439 + 1;
}
x435 = x435 + 1;
}
x430 = 0;
int x443 = 0;
while (x443 != 1) {
int x444 = x443;
float x445 = x424[x444] + (float)log((double)x434[x444]);
int x446 = 0;
while (x446 != 5) {
x429[x430] = x415[x430] - x445;
x430 = x430 + 1;
x446 = x446 + 1;
}
x443 = x443 + 1;
}
float* x447 = (float*)myMalloc(5 * sizeof(float));
// nllLoss forward in CPU;
float* x448 = (float*)myMalloc(1 * sizeof(float));
int x449 = 0;
int x450 = 0;
while (x450 != 1) {
int x451 = x450;
x448[x451] = -1.0 * x429[x449 + x131[x451]];
x449 = x449 + 5;
x450 = x450 + 1;
}
float* x452 = (float*)myMalloc(1 * sizeof(float));
float* x453 = (float*)myMalloc(1 * sizeof(float));
int x454 = 0;
while (x454 != 1) {
x453[x454] = x419[0] + x448[0];
x454 = x454 + 1;
}
float* x455 = (float*)myMalloc(1 * sizeof(float));
float** x456 = (float**)myMalloc(6 * sizeof(float*));
x456[0] = x453;
x456[1] = x455;
x456[2] = x409;
x456[3] = x412;
x456[4] = x401;
x456[5] = x404;
x142(x456);
// back prop for + op;
int x457 = 0;
while (x457 != 1) {
int x458 = x457;
x421[0] = x421[0] + x455[x458];
x452[0] = x452[0] + x455[x458];
x457 = x457 + 1;
}
// 'nllLossB' gradient.;
// nllLoss_grad implementation in CPU;
int x459 = 0;
int x460 = 0;
while (x460 != 1) {
int x461 = x460;
int x462 = x459 + x131[x461];
x447[x462] = x447[x462] + -1.0 * x452[x461];
x459 = x459 + 5;
x460 = x460 + 1;
}
float* x463 = (float*)myMalloc(1 * sizeof(float));
int x464 = 0;
while (x464 != 1) {
int x465 = x464;
int x466 = x465;
int x467 = x465 * 5;
int x468 = 0;
while (x468 != 5) {
int x469 = 0;
while (x469 != 1) {
int x470 = x469;
int x471 = x466 + x470;
x463[x471] = x463[x471] + x447[x467 + x470];
x469 = x469 + 1;
}
x467 = x467 + 1;
x468 = x468 + 1;
}
x464 = x464 + 1;
}
int x472 = 0;
int x473 = 0;
while (x473 != 1) {
int x474 = x473;
int x475 = 0;
while (x475 != 5) {
int x476 = x472;
x418[x476] = x418[x476] + (x447[x472] - (float)exp((double)x429[x472]) * x463[x474]);
x472 = x472 + 1;
x475 = x475 + 1;
}
x473 = x473 + 1;
}
// back prop for + op;
int x477 = 0;
while (x477 != 1) {
int x478 = x477;
x148[0] = x148[0] + x421[x478];
x156[0] = x156[0] + x421[x478];
x477 = x477 + 1;
}
// back prop for + op;
int x479 = 0;
while (x479 != 5) {
int x480 = x479;
x414[x480] = x414[x480] + x418[x480];
x77[x480] = x77[x480] + x418[x480];
x479 = x479 + 1;
}
// backend add_cartesian;
int x481 = 0;
int x482 = 0;
while (x482 != 5) {
int x483 = x482;
int x484 = 0;
while (x484 != 150) {
int x485 = x484;
x76[x481 + x485] = x76[x481 + x485] + x409[x485] * x414[x483];
x484 = x484 + 1;
}
x481 = x481 + 150;
x482 = x482 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 5,150,1,x53,150,x414,1,1,x412,1);
// backprop for * op;
int x486 = 0;
while (x486 != 150) {
int x487 = x486;
x368[x487] = x368[x487] + x412[x487] * x405[x487];
x408[x487] = x408[x487] + x412[x487] * x365[x487];
x486 = x486 + 1;
}
int x488 = 0;
while (x488 != 150) {
int x489 = x488;
float x490 = x405[x489];
x404[x489] = x404[x489] + (1.0 - x490 * x490) * x408[x489];
x488 = x488 + 1;
}
// back prop for + op;
int x491 = 0;
while (x491 != 150) {
int x492 = x491;
x396[x492] = x396[x492] + x404[x492];
x400[x492] = x400[x492] + x404[x492];
x491 = x491 + 1;
}
// backprop for * op;
int x493 = 0;
while (x493 != 150) {
int x494 = x493;
x352[x494] = x352[x494] + x400[x494] * x159[x494];
x160[x494] = x160[x494] + x400[x494] * x349[x494];
x493 = x493 + 1;
}
// back prop for + op;
int x495 = 0;
while (x495 != 150) {
int x496 = x495;
x388[x496] = x388[x496] + x396[x496];
x392[x496] = x392[x496] + x396[x496];
x495 = x495 + 1;
}
// backprop for * op;
int x497 = 0;
while (x497 != 150) {
int x498 = x497;
x336[x498] = x336[x498] + x392[x498] * x151[x498];
x152[x498] = x152[x498] + x392[x498] * x333[x498];
x497 = x497 + 1;
}
// backprop for * op;
int x499 = 0;
while (x499 != 150) {
int x500 = x499;
x320[x500] = x320[x500] + x388[x500] * x381[x500];
x384[x500] = x384[x500] + x388[x500] * x317[x500];
x499 = x499 + 1;
}
int x501 = 0;
while (x501 != 150) {
int x502 = x501;
float x503 = x381[x502];
x380[x502] = x380[x502] + (1.0 - x503 * x503) * x384[x502];
x501 = x501 + 1;
}
// back prop for + op;
int x504 = 0;
while (x504 != 150) {
int x505 = x504;
x376[x505] = x376[x505] + x380[x505];
x75[x505] = x75[x505] + x380[x505];
x504 = x504 + 1;
}
// back prop for + op;
int x506 = 0;
while (x506 != 150) {
int x507 = x506;
x370[x507] = x370[x507] + x376[x507];
x372[x507] = x372[x507] + x376[x507];
x506 = x506 + 1;
}
// backend add_cartesian;
int x508 = 0;
int x509 = 0;
while (x509 != 150) {
int x510 = x509;
int x511 = 0;
while (x511 != 150) {
int x512 = x511;
x74[x508 + x512] = x74[x508 + x512] + x157[x512] * x372[x510];
x511 = x511 + 1;
}
x508 = x508 + 150;
x509 = x509 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x50,150,x372,1,1,x158,1);
// backend add_cartesian;
int x513 = 0;
int x514 = 0;
while (x514 != 150) {
int x515 = x514;
int x516 = 0;
while (x516 != 150) {
int x517 = x516;
x73[x513 + x517] = x73[x513 + x517] + x149[x517] * x370[x515];
x516 = x516 + 1;
}
x513 = x513 + 150;
x514 = x514 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x48,150,x370,1,1,x150,1);
int x518 = 0;
while (x518 != 150) {
int x519 = x518;
float x520 = x365[x519];
x364[x519] = x364[x519] + (1.0 - x520) * x520 * x368[x519];
x518 = x518 + 1;
}
// back prop for + op;
int x521 = 0;
while (x521 != 150) {
int x522 = x521;
x360[x522] = x360[x522] + x364[x522];
x72[x522] = x72[x522] + x364[x522];
x521 = x521 + 1;
}
// back prop for + op;
int x523 = 0;
while (x523 != 150) {
int x524 = x523;
x354[x524] = x354[x524] + x360[x524];
x356[x524] = x356[x524] + x360[x524];
x523 = x523 + 1;
}
// backend add_cartesian;
int x525 = 0;
int x526 = 0;
while (x526 != 150) {
int x527 = x526;
int x528 = 0;
while (x528 != 150) {
int x529 = x528;
x71[x525 + x529] = x71[x525 + x529] + x157[x529] * x356[x527];
x528 = x528 + 1;
}
x525 = x525 + 150;
x526 = x526 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x45,150,x356,1,1,x158,1);
// backend add_cartesian;
int x530 = 0;
int x531 = 0;
while (x531 != 150) {
int x532 = x531;
int x533 = 0;
while (x533 != 150) {
int x534 = x533;
x70[x530 + x534] = x70[x530 + x534] + x149[x534] * x354[x532];
x533 = x533 + 1;
}
x530 = x530 + 150;
x531 = x531 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x43,150,x354,1,1,x150,1);
int x535 = 0;
while (x535 != 150) {
int x536 = x535;
float x537 = x349[x536];
x348[x536] = x348[x536] + (1.0 - x537) * x537 * x352[x536];
x535 = x535 + 1;
}
// back prop for + op;
int x538 = 0;
while (x538 != 150) {
int x539 = x538;
x344[x539] = x344[x539] + x348[x539];
x69[x539] = x69[x539] + x348[x539];
x538 = x538 + 1;
}
// back prop for + op;
int x540 = 0;
while (x540 != 150) {
int x541 = x540;
x338[x541] = x338[x541] + x344[x541];
x340[x541] = x340[x541] + x344[x541];
x540 = x540 + 1;
}
// backend add_cartesian;
int x542 = 0;
int x543 = 0;
while (x543 != 150) {
int x544 = x543;
int x545 = 0;
while (x545 != 150) {
int x546 = x545;
x68[x542 + x546] = x68[x542 + x546] + x157[x546] * x340[x544];
x545 = x545 + 1;
}
x542 = x542 + 150;
x543 = x543 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x40,150,x340,1,1,x158,1);
// backend add_cartesian;
int x547 = 0;
int x548 = 0;
while (x548 != 150) {
int x549 = x548;
int x550 = 0;
while (x550 != 150) {
int x551 = x550;
x67[x547 + x551] = x67[x547 + x551] + x149[x551] * x338[x549];
x550 = x550 + 1;
}
x547 = x547 + 150;
x548 = x548 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x38,150,x338,1,1,x150,1);
int x552 = 0;
while (x552 != 150) {
int x553 = x552;
float x554 = x333[x553];
x332[x553] = x332[x553] + (1.0 - x554) * x554 * x336[x553];
x552 = x552 + 1;
}
// back prop for + op;
int x555 = 0;
while (x555 != 150) {
int x556 = x555;
x328[x556] = x328[x556] + x332[x556];
x69[x556] = x69[x556] + x332[x556];
x555 = x555 + 1;
}
// back prop for + op;
int x557 = 0;
while (x557 != 150) {
int x558 = x557;
x322[x558] = x322[x558] + x328[x558];
x324[x558] = x324[x558] + x328[x558];
x557 = x557 + 1;
}
// backend add_cartesian;
int x559 = 0;
int x560 = 0;
while (x560 != 150) {
int x561 = x560;
int x562 = 0;
while (x562 != 150) {
int x563 = x562;
x66[x559 + x563] = x66[x559 + x563] + x157[x563] * x324[x561];
x562 = x562 + 1;
}
x559 = x559 + 150;
x560 = x560 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x36,150,x324,1,1,x158,1);
// backend add_cartesian;
int x564 = 0;
int x565 = 0;
while (x565 != 150) {
int x566 = x565;
int x567 = 0;
while (x567 != 150) {
int x568 = x567;
x65[x564 + x568] = x65[x564 + x568] + x149[x568] * x322[x566];
x567 = x567 + 1;
}
x564 = x564 + 150;
x565 = x565 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x34,150,x322,1,1,x150,1);
int x569 = 0;
while (x569 != 150) {
int x570 = x569;
float x571 = x317[x570];
x316[x570] = x316[x570] + (1.0 - x571) * x571 * x320[x570];
x569 = x569 + 1;
}
// back prop for + op;
int x572 = 0;
while (x572 != 150) {
int x573 = x572;
x312[x573] = x312[x573] + x316[x573];
x64[x573] = x64[x573] + x316[x573];
x572 = x572 + 1;
}
// back prop for + op;
int x574 = 0;
while (x574 != 150) {
int x575 = x574;
x306[x575] = x306[x575] + x312[x575];
x308[x575] = x308[x575] + x312[x575];
x574 = x574 + 1;
}
// backend add_cartesian;
int x576 = 0;
int x577 = 0;
while (x577 != 150) {
int x578 = x577;
int x579 = 0;
while (x579 != 150) {
int x580 = x579;
x63[x576 + x580] = x63[x576 + x580] + x157[x580] * x308[x578];
x579 = x579 + 1;
}
x576 = x576 + 150;
x577 = x577 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x31,150,x308,1,1,x158,1);
// backend add_cartesian;
int x581 = 0;
int x582 = 0;
while (x582 != 150) {
int x583 = x582;
int x584 = 0;
while (x584 != 150) {
int x585 = x584;
x62[x581 + x585] = x62[x581 + x585] + x149[x585] * x306[x583];
x584 = x584 + 1;
}
x581 = x581 + 150;
x582 = x582 + 1;
}
// bankend add_composition;
cblas_sgemv(CblasRowMajor, CblasTrans, 150,150,1,x29,150,x306,1,1,x150,1);
}};
float** x586 = (float**)myMalloc(6 * sizeof(float*));
x586[0] = x115;
x586[1] = x116;
x586[2] = x117;
x586[3] = x118;
x586[4] = x119;
x586[5] = x120;
x129(x113[x130], x153, x586);};
float** x587 = (float**)myMalloc(6 * sizeof(float*));
x587[0] = x115;
x587[1] = x116;
x587[2] = x117;
x587[3] = x118;
x587[4] = x119;
x587[5] = x120;
x129(x112[x130], x145, x587);
} else {
function<void(float**)> x588 = [&](float** x590) {float** x589 = (float**)myMalloc(6 * sizeof(float*));
x589[0] = x590[0];
x589[1] = x590[1];
x589[2] = x590[2];
x589[3] = x590[3];
x589[4] = x590[4];
x589[5] = x590[5];
x135(x589);};
float** x591 = (float**)myMalloc(6 * sizeof(float*));
x591[0] = x115;
x591[1] = x116;
x591[2] = x117;
x591[3] = x118;
x591[4] = x119;
x591[5] = x120;
x588(x591);
}};
x129(0, x121, x128);
float x593 = (float)(x108 + 1);
x106 = x106 * (float)x108 / x593 + x114[0] / x593;
int x594 = 0;
while (x594 != 45000) {
int x595 = x594;
float x596 = x56[x595];
x78[x595] = x78[x595] + x596 * x596;
x20[x595] = x20[x595] - 0.05 * x596 / (float)sqrt((double)x78[x595] + 9.99999993922529E-9);
x56[x595] = 0.0;
x594 = x594 + 1;
}
int x597 = 0;
while (x597 != 150) {
int x598 = x597;
float x599 = x57[x598];
x79[x598] = x79[x598] + x599 * x599;
x22[x598] = x22[x598] - 0.05 * x599 / (float)sqrt((double)x79[x598] + 9.99999993922529E-9);
x57[x598] = 0.0;
x597 = x597 + 1;
}
int x600 = 0;
while (x600 != 45000) {
int x601 = x600;
float x602 = x58[x601];
x80[x601] = x80[x601] + x602 * x602;
x23[x601] = x23[x601] - 0.05 * x602 / (float)sqrt((double)x80[x601] + 9.99999993922529E-9);
x58[x601] = 0.0;
x600 = x600 + 1;
}
int x603 = 0;
while (x603 != 150) {
int x604 = x603;
float x605 = x59[x604];
x81[x604] = x81[x604] + x605 * x605;
x25[x604] = x25[x604] - 0.05 * x605 / (float)sqrt((double)x81[x604] + 9.99999993922529E-9);
x59[x604] = 0.0;
x603 = x603 + 1;
}
int x606 = 0;
while (x606 != 45000) {
int x607 = x606;
float x608 = x60[x607];
x82[x607] = x82[x607] + x608 * x608;
x26[x607] = x26[x607] - 0.05 * x608 / (float)sqrt((double)x82[x607] + 9.99999993922529E-9);
x60[x607] = 0.0;
x606 = x606 + 1;
}
int x609 = 0;
while (x609 != 150) {
int x610 = x609;
float x611 = x61[x610];
x83[x610] = x83[x610] + x611 * x611;
x28[x610] = x28[x610] - 0.05 * x611 / (float)sqrt((double)x83[x610] + 9.99999993922529E-9);
x61[x610] = 0.0;
x609 = x609 + 1;
}
int x612 = 0;
while (x612 != 22500) {
int x613 = x612;
float x614 = x62[x613];
x84[x613] = x84[x613] + x614 * x614;
x29[x613] = x29[x613] - 0.05 * x614 / (float)sqrt((double)x84[x613] + 9.99999993922529E-9);
x62[x613] = 0.0;
x612 = x612 + 1;
}
int x615 = 0;
while (x615 != 22500) {
int x616 = x615;
float x617 = x63[x616];
x85[x616] = x85[x616] + x617 * x617;
x31[x616] = x31[x616] - 0.05 * x617 / (float)sqrt((double)x85[x616] + 9.99999993922529E-9);
x63[x616] = 0.0;
x615 = x615 + 1;
}
int x618 = 0;
while (x618 != 150) {
int x619 = x618;
float x620 = x64[x619];
x86[x619] = x86[x619] + x620 * x620;
x33[x619] = x33[x619] - 0.05 * x620 / (float)sqrt((double)x86[x619] + 9.99999993922529E-9);
x64[x619] = 0.0;
x618 = x618 + 1;
}
int x621 = 0;
while (x621 != 22500) {
int x622 = x621;
float x623 = x65[x622];
x87[x622] = x87[x622] + x623 * x623;
x34[x622] = x34[x622] - 0.05 * x623 / (float)sqrt((double)x87[x622] + 9.99999993922529E-9);
x65[x622] = 0.0;
x621 = x621 + 1;
}
int x624 = 0;
while (x624 != 22500) {
int x625 = x624;
float x626 = x66[x625];
x88[x625] = x88[x625] + x626 * x626;
x36[x625] = x36[x625] - 0.05 * x626 / (float)sqrt((double)x88[x625] + 9.99999993922529E-9);
x66[x625] = 0.0;
x624 = x624 + 1;
}
int x627 = 0;
while (x627 != 22500) {
int x628 = x627;
float x629 = x67[x628];
x89[x628] = x89[x628] + x629 * x629;
x38[x628] = x38[x628] - 0.05 * x629 / (float)sqrt((double)x89[x628] + 9.99999993922529E-9);
x67[x628] = 0.0;
x627 = x627 + 1;
}
int x630 = 0;
while (x630 != 22500) {
int x631 = x630;
float x632 = x68[x631];
x90[x631] = x90[x631] + x632 * x632;
x40[x631] = x40[x631] - 0.05 * x632 / (float)sqrt((double)x90[x631] + 9.99999993922529E-9);
x68[x631] = 0.0;
x630 = x630 + 1;
}
int x633 = 0;
while (x633 != 150) {
int x634 = x633;
float x635 = x69[x634];
x91[x634] = x91[x634] + x635 * x635;
x42[x634] = x42[x634] - 0.05 * x635 / (float)sqrt((double)x91[x634] + 9.99999993922529E-9);
x69[x634] = 0.0;
x633 = x633 + 1;
}
int x636 = 0;
while (x636 != 22500) {
int x637 = x636;
float x638 = x70[x637];
x92[x637] = x92[x637] + x638 * x638;
x43[x637] = x43[x637] - 0.05 * x638 / (float)sqrt((double)x92[x637] + 9.99999993922529E-9);
x70[x637] = 0.0;
x636 = x636 + 1;
}
int x639 = 0;
while (x639 != 22500) {
int x640 = x639;
float x641 = x71[x640];
x93[x640] = x93[x640] + x641 * x641;
x45[x640] = x45[x640] - 0.05 * x641 / (float)sqrt((double)x93[x640] + 9.99999993922529E-9);
x71[x640] = 0.0;
x639 = x639 + 1;
}
int x642 = 0;
while (x642 != 150) {
int x643 = x642;
float x644 = x72[x643];
x94[x643] = x94[x643] + x644 * x644;
x47[x643] = x47[x643] - 0.05 * x644 / (float)sqrt((double)x94[x643] + 9.99999993922529E-9);
x72[x643] = 0.0;
x642 = x642 + 1;
}
int x645 = 0;
while (x645 != 22500) {
int x646 = x645;
float x647 = x73[x646];
x95[x646] = x95[x646] + x647 * x647;
x48[x646] = x48[x646] - 0.05 * x647 / (float)sqrt((double)x95[x646] + 9.99999993922529E-9);
x73[x646] = 0.0;
x645 = x645 + 1;
}
int x648 = 0;
while (x648 != 22500) {
int x649 = x648;
float x650 = x74[x649];
x96[x649] = x96[x649] + x650 * x650;
x50[x649] = x50[x649] - 0.05 * x650 / (float)sqrt((double)x96[x649] + 9.99999993922529E-9);
x74[x649] = 0.0;
x648 = x648 + 1;
}
int x651 = 0;
while (x651 != 150) {
int x652 = x651;
float x653 = x75[x652];
x97[x652] = x97[x652] + x653 * x653;
x52[x652] = x52[x652] - 0.05 * x653 / (float)sqrt((double)x97[x652] + 9.99999993922529E-9);
x75[x652] = 0.0;
x651 = x651 + 1;
}
int x654 = 0;
while (x654 != 750) {
int x655 = x654;
float x656 = x76[x655];
x98[x655] = x98[x655] + x656 * x656;
x53[x655] = x53[x655] - 0.05 * x656 / (float)sqrt((double)x98[x655] + 9.99999993922529E-9);
x76[x655] = 0.0;
x654 = x654 + 1;
}
int x657 = 0;
while (x657 != 5) {
int x658 = x657;
float x659 = x77[x658];
x99[x658] = x99[x658] + x659 * x659;
x55[x658] = x55[x658] - 0.05 * x659 / (float)sqrt((double)x99[x658] + 9.99999993922529E-9);
x77[x658] = 0.0;
x657 = x657 + 1;
}
memset((void*)x101, 0, ((long)mallocAddr) - x101);
mallocAddr = (void*)x101;
x107 = x107 + 1;
}
x100[x105] = (double)x106;
printf("epoc %d, average_loss %f, time %lf\n", x105, x106, (((double)clock() / CLOCKS_PER_SEC)) - x102);
x103 = x103 + 1;
}
double x660 = ((double)clock() / CLOCKS_PER_SEC);
long x661 = (long)fopen(x0, "w");
fprintf((FILE *)x661, "unit: %s\n", "1 epoch");
int x662 = 0;
while (x662 != 6) {
fprintf((FILE *)x661, "%lf\n", x100[x662]);
x662 = x662 + 1;
}
fprintf((FILE *)x661, "run time: %lf %lf\n", x102 - x1, (x660 - x102) / 6.0);
fclose((FILE*)x661);
// Backend cleanup.;
}

    /*****************************************
    End of C Generated Code
    *******************************************/
    


