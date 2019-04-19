
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
int x2 = open("graham.txt",0);
int x3 = (int)fsize(x2);
char* x4 = (char*)mmap(0, (long)x3, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x2, 0);
printf("LSTM Test: >> data has %d chars\n", x3);
int* x5 = (int*)myMalloc(x3 * sizeof(int));
int x6 = 0;
while (x6 != x3) {
int x7 = x6;
x5[x7] = (int)x4[x7] - 96;
x6 = x6 + 1;
}
float* x8 = (float*)myMalloc(1300 * sizeof(float));
int x9 = 0;
while (x9 != 1300) {
x8[x9] = ((float)rand()/RAND_MAX - 0.5) * 0.19611613;
x9 = x9 + 1;
}
float* x10 = (float*)myMalloc(1300 * sizeof(float));
float* x11 = (float*)myMalloc(2500 * sizeof(float));
int x12 = 0;
while (x12 != 2500) {
x11[x12] = ((float)rand()/RAND_MAX - 0.5) * 0.14142136;
x12 = x12 + 1;
}
float* x13 = (float*)myMalloc(2500 * sizeof(float));
float* x14 = (float*)myMalloc(50 * sizeof(float));
float* x15 = (float*)myMalloc(50 * sizeof(float));
float* x16 = (float*)myMalloc(1300 * sizeof(float));
int x17 = 0;
while (x17 != 1300) {
x16[x17] = ((float)rand()/RAND_MAX - 0.5) * 0.19611613;
x17 = x17 + 1;
}
float* x18 = (float*)myMalloc(1300 * sizeof(float));
float* x19 = (float*)myMalloc(2500 * sizeof(float));
int x20 = 0;
while (x20 != 2500) {
x19[x20] = ((float)rand()/RAND_MAX - 0.5) * 0.14142136;
x20 = x20 + 1;
}
float* x21 = (float*)myMalloc(2500 * sizeof(float));
float* x22 = (float*)myMalloc(50 * sizeof(float));
float* x23 = (float*)myMalloc(50 * sizeof(float));
float* x24 = (float*)myMalloc(1300 * sizeof(float));
int x25 = 0;
while (x25 != 1300) {
x24[x25] = ((float)rand()/RAND_MAX - 0.5) * 0.19611613;
x25 = x25 + 1;
}
float* x26 = (float*)myMalloc(1300 * sizeof(float));
float* x27 = (float*)myMalloc(2500 * sizeof(float));
int x28 = 0;
while (x28 != 2500) {
x27[x28] = ((float)rand()/RAND_MAX - 0.5) * 0.14142136;
x28 = x28 + 1;
}
float* x29 = (float*)myMalloc(2500 * sizeof(float));
float* x30 = (float*)myMalloc(50 * sizeof(float));
float* x31 = (float*)myMalloc(50 * sizeof(float));
float* x32 = (float*)myMalloc(1300 * sizeof(float));
int x33 = 0;
while (x33 != 1300) {
x32[x33] = ((float)rand()/RAND_MAX - 0.5) * 0.19611613;
x33 = x33 + 1;
}
float* x34 = (float*)myMalloc(1300 * sizeof(float));
float* x35 = (float*)myMalloc(2500 * sizeof(float));
int x36 = 0;
while (x36 != 2500) {
x35[x36] = ((float)rand()/RAND_MAX - 0.5) * 0.14142136;
x36 = x36 + 1;
}
float* x37 = (float*)myMalloc(2500 * sizeof(float));
float* x38 = (float*)myMalloc(50 * sizeof(float));
float* x39 = (float*)myMalloc(50 * sizeof(float));
float* x40 = (float*)myMalloc(1300 * sizeof(float));
int x41 = 0;
while (x41 != 1300) {
x40[x41] = ((float)rand()/RAND_MAX - 0.5) * 0.14142136;
x41 = x41 + 1;
}
float* x42 = (float*)myMalloc(1300 * sizeof(float));
float* x43 = (float*)myMalloc(26 * sizeof(float));
float* x44 = (float*)myMalloc(26 * sizeof(float));
float* x45 = (float*)myMalloc(1300 * sizeof(float));
float* x46 = (float*)myMalloc(50 * sizeof(float));
float* x47 = (float*)myMalloc(2500 * sizeof(float));
float* x48 = (float*)myMalloc(50 * sizeof(float));
float* x49 = (float*)myMalloc(2500 * sizeof(float));
float* x50 = (float*)myMalloc(1300 * sizeof(float));
float* x51 = (float*)myMalloc(1300 * sizeof(float));
float* x52 = (float*)myMalloc(50 * sizeof(float));
float* x53 = (float*)myMalloc(2500 * sizeof(float));
float* x54 = (float*)myMalloc(26 * sizeof(float));
float* x55 = (float*)myMalloc(1300 * sizeof(float));
float* x56 = (float*)myMalloc(2500 * sizeof(float));
float* x57 = (float*)myMalloc(1300 * sizeof(float));
float* x58 = (float*)myMalloc(50 * sizeof(float));
double x59 = ((double)clock() / CLOCKS_PER_SEC);
double* x60 = (double*)myMalloc(51 * sizeof(double));
long x61 = (long)mallocAddr;
int x62 = 0;
x62 = x62 - 400;
double x63 = 70.0;
int x64 = 0;
float x65 = (float)1;
while (x64 != 5001) {
int x66 = x64;
x62 = x62 + 400;
if (x62 + 400 + 1 >= x3) x62 = 0; else {

}
int* x67 = (int*)myMalloc(400 * sizeof(int));
int* x68 = (int*)myMalloc(400 * sizeof(int));
int x69 = 0;
while (x69 != 400) {
int x70 = x69;
x67[x70] = x5[x62 + x70];
x68[x70] = x5[x62 + x70 + 1];
x69 = x69 + 1;
}
(float*)myMalloc(1 * sizeof(float));
(float*)myMalloc(1 * sizeof(float));
// allocate memory to save the final loss in CPU Tensor;
float* x71 = (float*)myMalloc(1 * sizeof(float));
float* x72 = (float*)myMalloc(10400 * sizeof(float));
int x73 = 0;
while (x73 != 20) {
int x74 = x73;
int x75 = 0;
int x76 = x74 * 26 * 20;
while (x75 != 20) {
int x77 = x75;
x72[x76 + x77 * 26 + x67[x77 * 20 + x74]] = 1.0;
x75 = x75 + 1;
}
x73 = x73 + 1;
}
float* x78 = (float*)myMalloc(10400 * sizeof(float));
float* x79 = (float*)myMalloc(1 * sizeof(float));
float* x80 = (float*)myMalloc(1 * sizeof(float));
float* x81 = (float*)myMalloc(1000 * sizeof(float));
float* x82 = (float*)myMalloc(1000 * sizeof(float));
float* x83 = (float*)myMalloc(1000 * sizeof(float));
float* x84 = (float*)myMalloc(1000 * sizeof(float));
function<void(int,float**)> x85 = [&](int x93, float** x87) {float* x86 = x87[0];
float* x88 = x87[1];
float* x89 = x87[2];
float* x90 = x87[3];
float* x91 = x87[4];
float* x92 = x87[5];
if (x93 < 20) {
int x94 = x93 * 520;
float* x95 = x72+x94;
float* x96 = x78+x94;
float* x97 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x95,26,x8,50,0,x97,50);
float* x98 = (float*)myMalloc(1000 * sizeof(float));
float* x99 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x89,50,x11,50,0,x99,50);
float* x100 = (float*)myMalloc(1000 * sizeof(float));
float* x101 = (float*)myMalloc(1000 * sizeof(float));
int x102 = 0;
while (x102 != 20) {
int x103 = 0;
int x104 = 50 * x102;
while (x103 != 50) {
int x105 = x103;
int x106 = x104 + x105;
x101[x105 + x104] = x97[x106] + x99[x106];
x103 = x103 + 1;
}
x102 = x102 + 1;
}
float* x107 = (float*)myMalloc(1000 * sizeof(float));
float* x108 = (float*)myMalloc(1000 * sizeof(float));
int x109 = 0;
while (x109 != 20) {
int x110 = 0;
int x111 = 50 * x109;
while (x110 != 50) {
int x112 = x110;
x108[x112 + x111] = x101[x111 + x112] + x14[x112];
x110 = x110 + 1;
}
x109 = x109 + 1;
}
float* x113 = (float*)myMalloc(1000 * sizeof(float));
float* x114 = (float*)myMalloc(1000 * sizeof(float));
int x115 = 0;
while (x115 != 1000) {
int x116 = x115;
x114[x116] = 1.0 / ((float)exp((double)(-1.0 * x108[x116])) + 1.0);
x115 = x115 + 1;
}
float* x117 = (float*)myMalloc(1000 * sizeof(float));
float* x118 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x95,26,x16,50,0,x118,50);
float* x119 = (float*)myMalloc(1000 * sizeof(float));
float* x120 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x89,50,x19,50,0,x120,50);
float* x121 = (float*)myMalloc(1000 * sizeof(float));
float* x122 = (float*)myMalloc(1000 * sizeof(float));
int x123 = 0;
while (x123 != 20) {
int x124 = 0;
int x125 = 50 * x123;
while (x124 != 50) {
int x126 = x124;
int x127 = x125 + x126;
x122[x126 + x125] = x118[x127] + x120[x127];
x124 = x124 + 1;
}
x123 = x123 + 1;
}
float* x128 = (float*)myMalloc(1000 * sizeof(float));
float* x129 = (float*)myMalloc(1000 * sizeof(float));
int x130 = 0;
while (x130 != 20) {
int x131 = 0;
int x132 = 50 * x130;
while (x131 != 50) {
int x133 = x131;
x129[x133 + x132] = x122[x132 + x133] + x22[x133];
x131 = x131 + 1;
}
x130 = x130 + 1;
}
float* x134 = (float*)myMalloc(1000 * sizeof(float));
float* x135 = (float*)myMalloc(1000 * sizeof(float));
int x136 = 0;
while (x136 != 1000) {
int x137 = x136;
x135[x137] = 1.0 / ((float)exp((double)(-1.0 * x129[x137])) + 1.0);
x136 = x136 + 1;
}
float* x138 = (float*)myMalloc(1000 * sizeof(float));
float* x139 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x95,26,x32,50,0,x139,50);
float* x140 = (float*)myMalloc(1000 * sizeof(float));
float* x141 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x89,50,x35,50,0,x141,50);
float* x142 = (float*)myMalloc(1000 * sizeof(float));
float* x143 = (float*)myMalloc(1000 * sizeof(float));
int x144 = 0;
while (x144 != 20) {
int x145 = 0;
int x146 = 50 * x144;
while (x145 != 50) {
int x147 = x145;
int x148 = x146 + x147;
x143[x147 + x146] = x139[x148] + x141[x148];
x145 = x145 + 1;
}
x144 = x144 + 1;
}
float* x149 = (float*)myMalloc(1000 * sizeof(float));
float* x150 = (float*)myMalloc(1000 * sizeof(float));
int x151 = 0;
while (x151 != 20) {
int x152 = 0;
int x153 = 50 * x151;
while (x152 != 50) {
int x154 = x152;
x150[x154 + x153] = x143[x153 + x154] + x38[x154];
x152 = x152 + 1;
}
x151 = x151 + 1;
}
float* x155 = (float*)myMalloc(1000 * sizeof(float));
float* x156 = (float*)myMalloc(1000 * sizeof(float));
int x157 = 0;
while (x157 != 1000) {
int x158 = x157;
x156[x158] = 1.0 / ((float)exp((double)(-1.0 * x150[x158])) + 1.0);
x157 = x157 + 1;
}
float* x159 = (float*)myMalloc(1000 * sizeof(float));
float* x160 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x95,26,x24,50,0,x160,50);
float* x161 = (float*)myMalloc(1000 * sizeof(float));
float* x162 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x89,50,x27,50,0,x162,50);
float* x163 = (float*)myMalloc(1000 * sizeof(float));
float* x164 = (float*)myMalloc(1000 * sizeof(float));
int x165 = 0;
while (x165 != 20) {
int x166 = 0;
int x167 = 50 * x165;
while (x166 != 50) {
int x168 = x166;
int x169 = x167 + x168;
x164[x168 + x167] = x160[x169] + x162[x169];
x166 = x166 + 1;
}
x165 = x165 + 1;
}
float* x170 = (float*)myMalloc(1000 * sizeof(float));
float* x171 = (float*)myMalloc(1000 * sizeof(float));
int x172 = 0;
while (x172 != 20) {
int x173 = 0;
int x174 = 50 * x172;
while (x173 != 50) {
int x175 = x173;
x171[x175 + x174] = x164[x174 + x175] + x30[x175];
x173 = x173 + 1;
}
x172 = x172 + 1;
}
float* x176 = (float*)myMalloc(1000 * sizeof(float));
float* x177 = (float*)myMalloc(1000 * sizeof(float));
int x178 = 0;
while (x178 != 1000) {
int x179 = x178;
x177[x179] = (float)tanh((double)x171[x179]);
x178 = x178 + 1;
}
float* x180 = (float*)myMalloc(1000 * sizeof(float));
float* x181 = (float*)myMalloc(1000 * sizeof(float));
int x182 = 0;
while (x182 != 20) {
int x183 = 0;
int x184 = 50 * x182;
while (x183 != 50) {
int x185 = x183;
int x186 = x184 + x185;
x181[x185 + x184] = x114[x186] * x91[x186];
x183 = x183 + 1;
}
x182 = x182 + 1;
}
float* x187 = (float*)myMalloc(1000 * sizeof(float));
float* x188 = (float*)myMalloc(1000 * sizeof(float));
int x189 = 0;
while (x189 != 20) {
int x190 = 0;
int x191 = 50 * x189;
while (x190 != 50) {
int x192 = x190;
int x193 = x191 + x192;
x188[x192 + x191] = x135[x193] * x177[x193];
x190 = x190 + 1;
}
x189 = x189 + 1;
}
float* x194 = (float*)myMalloc(1000 * sizeof(float));
float* x195 = (float*)myMalloc(1000 * sizeof(float));
int x196 = 0;
while (x196 != 20) {
int x197 = 0;
int x198 = 50 * x196;
while (x197 != 50) {
int x199 = x197;
int x200 = x198 + x199;
x195[x199 + x198] = x181[x200] + x188[x200];
x197 = x197 + 1;
}
x196 = x196 + 1;
}
float* x201 = (float*)myMalloc(1000 * sizeof(float));
float* x202 = (float*)myMalloc(1000 * sizeof(float));
int x203 = 0;
while (x203 != 1000) {
int x204 = x203;
x202[x204] = (float)tanh((double)x195[x204]);
x203 = x203 + 1;
}
float* x205 = (float*)myMalloc(1000 * sizeof(float));
float* x206 = (float*)myMalloc(1000 * sizeof(float));
int x207 = 0;
while (x207 != 20) {
int x208 = 0;
int x209 = 50 * x207;
while (x208 != 50) {
int x210 = x208;
int x211 = x209 + x210;
x206[x210 + x209] = x156[x211] * x202[x211];
x208 = x208 + 1;
}
x207 = x207 + 1;
}
float* x212 = (float*)myMalloc(1000 * sizeof(float));
float* x213 = (float*)myMalloc(520 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x206,50,x40,26,0,x213,26);
float* x214 = (float*)myMalloc(520 * sizeof(float));
int x215 = 0;
while (x215 != 20) {
int x216 = 0;
int x217 = 26 * x215;
while (x216 != 26) {
int x218 = x216;
int x219 = x217 + x218;
x213[x219] = x213[x219] + x43[x218];
x216 = x216 + 1;
}
x215 = x215 + 1;
}
int* x220 = (int*)myMalloc(20 * sizeof(int));
int x221 = 0;
while (x221 != 20) {
int x222 = x221;
x220[x222] = x68[x93 + x222 * 20];
x221 = x221 + 1;
}
float* x223 = (float*)myMalloc(20 * sizeof(float));
int x224 = 0;
int x225 = 0;
while (x225 != 20) {
float x226 = -3.4028235E38;
int x227 = 0;
while (x227 != 26) {
if (x213[x224] > x226) x226 = x213[x224]; else {

}
x224 = x224 + 1;
x227 = x227 + 1;
}
x223[x225] = x226;
x225 = x225 + 1;
}
float* x228 = (float*)myMalloc(520 * sizeof(float));
int x229 = 0;
int x230 = 0;
while (x230 != 20) {
int x231 = x230;
int x232 = 0;
while (x232 != 26) {
x228[x229] = (float)exp((double)(x213[x229] - x223[x231]));
x229 = x229 + 1;
x232 = x232 + 1;
}
x230 = x230 + 1;
}
float* x233 = (float*)myMalloc(20 * sizeof(float));
int x234 = 0;
while (x234 != 20) {
int x235 = x234;
int x236 = x235;
int x237 = x235 * 26;
int x238 = 0;
while (x238 != 26) {
int x239 = 0;
while (x239 != 1) {
int x240 = x239;
int x241 = x236 + x240;
x233[x241] = x233[x241] + x228[x237 + x240];
x239 = x239 + 1;
}
x237 = x237 + 1;
x238 = x238 + 1;
}
x234 = x234 + 1;
}
x229 = 0;
int x242 = 0;
while (x242 != 20) {
int x243 = x242;
float x244 = x223[x243] + (float)log((double)x233[x243]);
int x245 = 0;
while (x245 != 26) {
x228[x229] = x213[x229] - x244;
x229 = x229 + 1;
x245 = x245 + 1;
}
x242 = x242 + 1;
}
float* x246 = (float*)myMalloc(520 * sizeof(float));
// nllLoss forward in CPU;
float* x247 = (float*)myMalloc(20 * sizeof(float));
int x248 = 0;
int x249 = 0;
while (x249 != 20) {
int x250 = x249;
x247[x250] = -1.0 * x228[x248 + x220[x250]];
x248 = x248 + 26;
x249 = x249 + 1;
}
float* x251 = (float*)myMalloc(20 * sizeof(float));
float x252 = 0.0;
int x253 = 0;
while (x253 != 20) {
x252 = x252 + x247[x253];
x253 = x253 + 1;
}
float x254 = x252;
float* x255 = (float*)myMalloc(1 * sizeof(float));
int x256 = 0;
while (x256 != 1) {
x255[x256] = x254;
x256 = x256 + 1;
}
float* x257 = (float*)myMalloc(1 * sizeof(float));
float* x258 = (float*)myMalloc(1 * sizeof(float));
int x259 = 0;
while (x259 != 1) {
x258[x259] = x86[0] + x255[0];
x259 = x259 + 1;
}
float* x260 = (float*)myMalloc(1 * sizeof(float));
float** x261 = (float**)myMalloc(6 * sizeof(float*));
x261[0] = x258;
x261[1] = x260;
x261[2] = x206;
x261[3] = x212;
x261[4] = x195;
x261[5] = x201;
x85(x93 + 1, x261);
// back prop for + op;
int x262 = 0;
while (x262 != 1) {
int x263 = x262;
x88[0] = x88[0] + x260[x263];
x257[0] = x257[0] + x260[x263];
x262 = x262 + 1;
}
// 'sum' gradient.;
int x264 = 0;
while (x264 != 20) {
int x265 = x264;
x251[x265] = x251[x265] + x257[0];
x264 = x264 + 1;
}
// 'nllLossB' gradient.;
// nllLoss_grad implementation in CPU;
int x266 = 0;
int x267 = 0;
while (x267 != 20) {
int x268 = x267;
int x269 = x266 + x220[x268];
x246[x269] = x246[x269] + -1.0 * x251[x268];
x266 = x266 + 26;
x267 = x267 + 1;
}
float* x270 = (float*)myMalloc(20 * sizeof(float));
int x271 = 0;
while (x271 != 20) {
int x272 = x271;
int x273 = x272;
int x274 = x272 * 26;
int x275 = 0;
while (x275 != 26) {
int x276 = 0;
while (x276 != 1) {
int x277 = x276;
int x278 = x273 + x277;
x270[x278] = x270[x278] + x246[x274 + x277];
x276 = x276 + 1;
}
x274 = x274 + 1;
x275 = x275 + 1;
}
x271 = x271 + 1;
}
int x279 = 0;
int x280 = 0;
while (x280 != 20) {
int x281 = x280;
int x282 = 0;
while (x282 != 26) {
int x283 = x279;
x214[x283] = x214[x283] + (x246[x279] - (float)exp((double)x228[x279]) * x270[x281]);
x279 = x279 + 1;
x282 = x282 + 1;
}
x280 = x280 + 1;
}
int x284 = 0;
while (x284 != 20) {
int x285 = 0;
int x286 = 26 * x284;
while (x285 != 26) {
int x287 = x285;
x44[x287] = x44[x287] + x214[x286 + x287];
x285 = x285 + 1;
}
x284 = x284 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x214,26,x40,26,1,x212,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x206,50,x214,26,1,x42,26);
// backprop for * op;
int x288 = 0;
while (x288 != 20) {
int x289 = 0;
int x290 = 50 * x288;
while (x289 != 50) {
int x291 = x290 + x289;
x159[x291] = x159[x291] + x212[x291] * x202[x291];
x205[x291] = x205[x291] + x212[x291] * x156[x291];
x289 = x289 + 1;
}
x288 = x288 + 1;
}
int x292 = 0;
while (x292 != 1000) {
int x293 = x292;
float x294 = x202[x293];
x201[x293] = x201[x293] + (1.0 - x294 * x294) * x205[x293];
x292 = x292 + 1;
}
// back prop for + op;
int x295 = 0;
while (x295 != 20) {
int x296 = 0;
int x297 = 50 * x295;
while (x296 != 50) {
int x298 = x297 + x296;
x187[x298] = x187[x298] + x201[x298];
x194[x298] = x194[x298] + x201[x298];
x296 = x296 + 1;
}
x295 = x295 + 1;
}
// backprop for * op;
int x299 = 0;
while (x299 != 20) {
int x300 = 0;
int x301 = 50 * x299;
while (x300 != 50) {
int x302 = x301 + x300;
x138[x302] = x138[x302] + x194[x302] * x177[x302];
x180[x302] = x180[x302] + x194[x302] * x135[x302];
x300 = x300 + 1;
}
x299 = x299 + 1;
}
// backprop for * op;
int x303 = 0;
while (x303 != 20) {
int x304 = 0;
int x305 = 50 * x303;
while (x304 != 50) {
int x306 = x305 + x304;
x117[x306] = x117[x306] + x187[x306] * x91[x306];
x92[x306] = x92[x306] + x187[x306] * x114[x306];
x304 = x304 + 1;
}
x303 = x303 + 1;
}
int x307 = 0;
while (x307 != 1000) {
int x308 = x307;
float x309 = x177[x308];
x176[x308] = x176[x308] + (1.0 - x309 * x309) * x180[x308];
x307 = x307 + 1;
}
// back prop for + op;
int x310 = 0;
while (x310 != 20) {
int x311 = 0;
int x312 = 50 * x310;
while (x311 != 50) {
int x313 = x311;
int x314 = x312 + x313;
x170[x314] = x170[x314] + x176[x314];
x31[x313] = x31[x313] + x176[x314];
x311 = x311 + 1;
}
x310 = x310 + 1;
}
// back prop for + op;
int x315 = 0;
while (x315 != 20) {
int x316 = 0;
int x317 = 50 * x315;
while (x316 != 50) {
int x318 = x317 + x316;
x161[x318] = x161[x318] + x170[x318];
x163[x318] = x163[x318] + x170[x318];
x316 = x316 + 1;
}
x315 = x315 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x163,50,x27,50,1,x90,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x89,50,x163,50,1,x29,50);
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x161,50,x24,50,1,x96,26);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x95,26,x161,50,1,x26,50);
int x319 = 0;
while (x319 != 1000) {
int x320 = x319;
float x321 = x156[x320];
x155[x320] = x155[x320] + (1.0 - x321) * x321 * x159[x320];
x319 = x319 + 1;
}
// back prop for + op;
int x322 = 0;
while (x322 != 20) {
int x323 = 0;
int x324 = 50 * x322;
while (x323 != 50) {
int x325 = x323;
int x326 = x324 + x325;
x149[x326] = x149[x326] + x155[x326];
x39[x325] = x39[x325] + x155[x326];
x323 = x323 + 1;
}
x322 = x322 + 1;
}
// back prop for + op;
int x327 = 0;
while (x327 != 20) {
int x328 = 0;
int x329 = 50 * x327;
while (x328 != 50) {
int x330 = x329 + x328;
x140[x330] = x140[x330] + x149[x330];
x142[x330] = x142[x330] + x149[x330];
x328 = x328 + 1;
}
x327 = x327 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x142,50,x35,50,1,x90,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x89,50,x142,50,1,x37,50);
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x140,50,x32,50,1,x96,26);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x95,26,x140,50,1,x34,50);
int x331 = 0;
while (x331 != 1000) {
int x332 = x331;
float x333 = x135[x332];
x134[x332] = x134[x332] + (1.0 - x333) * x333 * x138[x332];
x331 = x331 + 1;
}
// back prop for + op;
int x334 = 0;
while (x334 != 20) {
int x335 = 0;
int x336 = 50 * x334;
while (x335 != 50) {
int x337 = x335;
int x338 = x336 + x337;
x128[x338] = x128[x338] + x134[x338];
x23[x337] = x23[x337] + x134[x338];
x335 = x335 + 1;
}
x334 = x334 + 1;
}
// back prop for + op;
int x339 = 0;
while (x339 != 20) {
int x340 = 0;
int x341 = 50 * x339;
while (x340 != 50) {
int x342 = x341 + x340;
x119[x342] = x119[x342] + x128[x342];
x121[x342] = x121[x342] + x128[x342];
x340 = x340 + 1;
}
x339 = x339 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x121,50,x19,50,1,x90,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x89,50,x121,50,1,x21,50);
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x119,50,x16,50,1,x96,26);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x95,26,x119,50,1,x18,50);
int x343 = 0;
while (x343 != 1000) {
int x344 = x343;
float x345 = x114[x344];
x113[x344] = x113[x344] + (1.0 - x345) * x345 * x117[x344];
x343 = x343 + 1;
}
// back prop for + op;
int x346 = 0;
while (x346 != 20) {
int x347 = 0;
int x348 = 50 * x346;
while (x347 != 50) {
int x349 = x347;
int x350 = x348 + x349;
x107[x350] = x107[x350] + x113[x350];
x15[x349] = x15[x349] + x113[x350];
x347 = x347 + 1;
}
x346 = x346 + 1;
}
// back prop for + op;
int x351 = 0;
while (x351 != 20) {
int x352 = 0;
int x353 = 50 * x351;
while (x352 != 50) {
int x354 = x353 + x352;
x98[x354] = x98[x354] + x107[x354];
x100[x354] = x100[x354] + x107[x354];
x352 = x352 + 1;
}
x351 = x351 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x100,50,x11,50,1,x90,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x89,50,x100,50,1,x13,50);
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x98,50,x8,50,1,x96,26);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x95,26,x98,50,1,x10,50);
} else {
float x355 = 0.0;
int x356 = 0;
while (x356 != 1) {
x355 = x355 + x86[x356];
x356 = x356 + 1;
}
float x357 = x355;
float* x358 = (float*)myMalloc(1 * sizeof(float));
int x359 = 0;
while (x359 != 1) {
x358[x359] = x357;
x359 = x359 + 1;
}
float* x360 = (float*)myMalloc(1 * sizeof(float));
// make sure the size of loss is 1;
int x361 = 0;
while (x361 != 1) {
x360[x361] = x65;
x361 = x361 + 1;
}
// backend is lantern.TensorDslCPU$BackendCPU@5fb0a9cd;
int x362 = 0;
while (x362 != 1) {
int x363 = x362;
x71[x363] = x358[x363];
x362 = x362 + 1;
}
// 'sum' gradient.;
int x364 = 0;
while (x364 != 1) {
x88[0] = x88[0] + x360[0];
x364 = x364 + 1;
}
}};
function<void(int,float**)> x365 = [&](int x373, float** x367) {float* x366 = x367[0];
float* x368 = x367[1];
float* x369 = x367[2];
float* x370 = x367[3];
float* x371 = x367[4];
float* x372 = x367[5];
if (x373 < 20) {
int x374 = x373 * 520;
float* x375 = x72+x374;
float* x376 = x78+x374;
float* x377 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x375,26,x8,50,0,x377,50);
float* x378 = (float*)myMalloc(1000 * sizeof(float));
float* x379 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x369,50,x11,50,0,x379,50);
float* x380 = (float*)myMalloc(1000 * sizeof(float));
float* x381 = (float*)myMalloc(1000 * sizeof(float));
int x382 = 0;
while (x382 != 20) {
int x383 = 0;
int x384 = 50 * x382;
while (x383 != 50) {
int x385 = x383;
int x386 = x384 + x385;
x381[x385 + x384] = x377[x386] + x379[x386];
x383 = x383 + 1;
}
x382 = x382 + 1;
}
float* x387 = (float*)myMalloc(1000 * sizeof(float));
float* x388 = (float*)myMalloc(1000 * sizeof(float));
int x389 = 0;
while (x389 != 20) {
int x390 = 0;
int x391 = 50 * x389;
while (x390 != 50) {
int x392 = x390;
x388[x392 + x391] = x381[x391 + x392] + x14[x392];
x390 = x390 + 1;
}
x389 = x389 + 1;
}
float* x393 = (float*)myMalloc(1000 * sizeof(float));
float* x394 = (float*)myMalloc(1000 * sizeof(float));
int x395 = 0;
while (x395 != 1000) {
int x396 = x395;
x394[x396] = 1.0 / ((float)exp((double)(-1.0 * x388[x396])) + 1.0);
x395 = x395 + 1;
}
float* x397 = (float*)myMalloc(1000 * sizeof(float));
float* x398 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x375,26,x16,50,0,x398,50);
float* x399 = (float*)myMalloc(1000 * sizeof(float));
float* x400 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x369,50,x19,50,0,x400,50);
float* x401 = (float*)myMalloc(1000 * sizeof(float));
float* x402 = (float*)myMalloc(1000 * sizeof(float));
int x403 = 0;
while (x403 != 20) {
int x404 = 0;
int x405 = 50 * x403;
while (x404 != 50) {
int x406 = x404;
int x407 = x405 + x406;
x402[x406 + x405] = x398[x407] + x400[x407];
x404 = x404 + 1;
}
x403 = x403 + 1;
}
float* x408 = (float*)myMalloc(1000 * sizeof(float));
float* x409 = (float*)myMalloc(1000 * sizeof(float));
int x410 = 0;
while (x410 != 20) {
int x411 = 0;
int x412 = 50 * x410;
while (x411 != 50) {
int x413 = x411;
x409[x413 + x412] = x402[x412 + x413] + x22[x413];
x411 = x411 + 1;
}
x410 = x410 + 1;
}
float* x414 = (float*)myMalloc(1000 * sizeof(float));
float* x415 = (float*)myMalloc(1000 * sizeof(float));
int x416 = 0;
while (x416 != 1000) {
int x417 = x416;
x415[x417] = 1.0 / ((float)exp((double)(-1.0 * x409[x417])) + 1.0);
x416 = x416 + 1;
}
float* x418 = (float*)myMalloc(1000 * sizeof(float));
float* x419 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x375,26,x32,50,0,x419,50);
float* x420 = (float*)myMalloc(1000 * sizeof(float));
float* x421 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x369,50,x35,50,0,x421,50);
float* x422 = (float*)myMalloc(1000 * sizeof(float));
float* x423 = (float*)myMalloc(1000 * sizeof(float));
int x424 = 0;
while (x424 != 20) {
int x425 = 0;
int x426 = 50 * x424;
while (x425 != 50) {
int x427 = x425;
int x428 = x426 + x427;
x423[x427 + x426] = x419[x428] + x421[x428];
x425 = x425 + 1;
}
x424 = x424 + 1;
}
float* x429 = (float*)myMalloc(1000 * sizeof(float));
float* x430 = (float*)myMalloc(1000 * sizeof(float));
int x431 = 0;
while (x431 != 20) {
int x432 = 0;
int x433 = 50 * x431;
while (x432 != 50) {
int x434 = x432;
x430[x434 + x433] = x423[x433 + x434] + x38[x434];
x432 = x432 + 1;
}
x431 = x431 + 1;
}
float* x435 = (float*)myMalloc(1000 * sizeof(float));
float* x436 = (float*)myMalloc(1000 * sizeof(float));
int x437 = 0;
while (x437 != 1000) {
int x438 = x437;
x436[x438] = 1.0 / ((float)exp((double)(-1.0 * x430[x438])) + 1.0);
x437 = x437 + 1;
}
float* x439 = (float*)myMalloc(1000 * sizeof(float));
float* x440 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x375,26,x24,50,0,x440,50);
float* x441 = (float*)myMalloc(1000 * sizeof(float));
float* x442 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x369,50,x27,50,0,x442,50);
float* x443 = (float*)myMalloc(1000 * sizeof(float));
float* x444 = (float*)myMalloc(1000 * sizeof(float));
int x445 = 0;
while (x445 != 20) {
int x446 = 0;
int x447 = 50 * x445;
while (x446 != 50) {
int x448 = x446;
int x449 = x447 + x448;
x444[x448 + x447] = x440[x449] + x442[x449];
x446 = x446 + 1;
}
x445 = x445 + 1;
}
float* x450 = (float*)myMalloc(1000 * sizeof(float));
float* x451 = (float*)myMalloc(1000 * sizeof(float));
int x452 = 0;
while (x452 != 20) {
int x453 = 0;
int x454 = 50 * x452;
while (x453 != 50) {
int x455 = x453;
x451[x455 + x454] = x444[x454 + x455] + x30[x455];
x453 = x453 + 1;
}
x452 = x452 + 1;
}
float* x456 = (float*)myMalloc(1000 * sizeof(float));
float* x457 = (float*)myMalloc(1000 * sizeof(float));
int x458 = 0;
while (x458 != 1000) {
int x459 = x458;
x457[x459] = (float)tanh((double)x451[x459]);
x458 = x458 + 1;
}
float* x460 = (float*)myMalloc(1000 * sizeof(float));
float* x461 = (float*)myMalloc(1000 * sizeof(float));
int x462 = 0;
while (x462 != 20) {
int x463 = 0;
int x464 = 50 * x462;
while (x463 != 50) {
int x465 = x463;
int x466 = x464 + x465;
x461[x465 + x464] = x394[x466] * x371[x466];
x463 = x463 + 1;
}
x462 = x462 + 1;
}
float* x467 = (float*)myMalloc(1000 * sizeof(float));
float* x468 = (float*)myMalloc(1000 * sizeof(float));
int x469 = 0;
while (x469 != 20) {
int x470 = 0;
int x471 = 50 * x469;
while (x470 != 50) {
int x472 = x470;
int x473 = x471 + x472;
x468[x472 + x471] = x415[x473] * x457[x473];
x470 = x470 + 1;
}
x469 = x469 + 1;
}
float* x474 = (float*)myMalloc(1000 * sizeof(float));
float* x475 = (float*)myMalloc(1000 * sizeof(float));
int x476 = 0;
while (x476 != 20) {
int x477 = 0;
int x478 = 50 * x476;
while (x477 != 50) {
int x479 = x477;
int x480 = x478 + x479;
x475[x479 + x478] = x461[x480] + x468[x480];
x477 = x477 + 1;
}
x476 = x476 + 1;
}
float* x481 = (float*)myMalloc(1000 * sizeof(float));
float* x482 = (float*)myMalloc(1000 * sizeof(float));
int x483 = 0;
while (x483 != 1000) {
int x484 = x483;
x482[x484] = (float)tanh((double)x475[x484]);
x483 = x483 + 1;
}
float* x485 = (float*)myMalloc(1000 * sizeof(float));
float* x486 = (float*)myMalloc(1000 * sizeof(float));
int x487 = 0;
while (x487 != 20) {
int x488 = 0;
int x489 = 50 * x487;
while (x488 != 50) {
int x490 = x488;
int x491 = x489 + x490;
x486[x490 + x489] = x436[x491] * x482[x491];
x488 = x488 + 1;
}
x487 = x487 + 1;
}
float* x492 = (float*)myMalloc(1000 * sizeof(float));
float* x493 = (float*)myMalloc(520 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x486,50,x40,26,0,x493,26);
float* x494 = (float*)myMalloc(520 * sizeof(float));
int x495 = 0;
while (x495 != 20) {
int x496 = 0;
int x497 = 26 * x495;
while (x496 != 26) {
int x498 = x496;
int x499 = x497 + x498;
x493[x499] = x493[x499] + x43[x498];
x496 = x496 + 1;
}
x495 = x495 + 1;
}
int* x500 = (int*)myMalloc(20 * sizeof(int));
int x501 = 0;
while (x501 != 20) {
int x502 = x501;
x500[x502] = x68[x373 + x502 * 20];
x501 = x501 + 1;
}
float* x503 = (float*)myMalloc(20 * sizeof(float));
int x504 = 0;
int x505 = 0;
while (x505 != 20) {
float x506 = -3.4028235E38;
int x507 = 0;
while (x507 != 26) {
if (x493[x504] > x506) x506 = x493[x504]; else {

}
x504 = x504 + 1;
x507 = x507 + 1;
}
x503[x505] = x506;
x505 = x505 + 1;
}
float* x508 = (float*)myMalloc(520 * sizeof(float));
int x509 = 0;
int x510 = 0;
while (x510 != 20) {
int x511 = x510;
int x512 = 0;
while (x512 != 26) {
x508[x509] = (float)exp((double)(x493[x509] - x503[x511]));
x509 = x509 + 1;
x512 = x512 + 1;
}
x510 = x510 + 1;
}
float* x513 = (float*)myMalloc(20 * sizeof(float));
int x514 = 0;
while (x514 != 20) {
int x515 = x514;
int x516 = x515;
int x517 = x515 * 26;
int x518 = 0;
while (x518 != 26) {
int x519 = 0;
while (x519 != 1) {
int x520 = x519;
int x521 = x516 + x520;
x513[x521] = x513[x521] + x508[x517 + x520];
x519 = x519 + 1;
}
x517 = x517 + 1;
x518 = x518 + 1;
}
x514 = x514 + 1;
}
x509 = 0;
int x522 = 0;
while (x522 != 20) {
int x523 = x522;
float x524 = x503[x523] + (float)log((double)x513[x523]);
int x525 = 0;
while (x525 != 26) {
x508[x509] = x493[x509] - x524;
x509 = x509 + 1;
x525 = x525 + 1;
}
x522 = x522 + 1;
}
float* x526 = (float*)myMalloc(520 * sizeof(float));
// nllLoss forward in CPU;
float* x527 = (float*)myMalloc(20 * sizeof(float));
int x528 = 0;
int x529 = 0;
while (x529 != 20) {
int x530 = x529;
x527[x530] = -1.0 * x508[x528 + x500[x530]];
x528 = x528 + 26;
x529 = x529 + 1;
}
float* x531 = (float*)myMalloc(20 * sizeof(float));
float x532 = 0.0;
int x533 = 0;
while (x533 != 20) {
x532 = x532 + x527[x533];
x533 = x533 + 1;
}
float x534 = x532;
float* x535 = (float*)myMalloc(1 * sizeof(float));
int x536 = 0;
while (x536 != 1) {
x535[x536] = x534;
x536 = x536 + 1;
}
float* x537 = (float*)myMalloc(1 * sizeof(float));
float* x538 = (float*)myMalloc(1 * sizeof(float));
int x539 = 0;
while (x539 != 1) {
x538[x539] = x366[0] + x535[0];
x539 = x539 + 1;
}
float* x540 = (float*)myMalloc(1 * sizeof(float));
float** x541 = (float**)myMalloc(6 * sizeof(float*));
x541[0] = x538;
x541[1] = x540;
x541[2] = x486;
x541[3] = x492;
x541[4] = x475;
x541[5] = x481;
x85(x373 + 1, x541);
// back prop for + op;
int x542 = 0;
while (x542 != 1) {
int x543 = x542;
x368[0] = x368[0] + x540[x543];
x537[0] = x537[0] + x540[x543];
x542 = x542 + 1;
}
// 'sum' gradient.;
int x544 = 0;
while (x544 != 20) {
int x545 = x544;
x531[x545] = x531[x545] + x537[0];
x544 = x544 + 1;
}
// 'nllLossB' gradient.;
// nllLoss_grad implementation in CPU;
int x546 = 0;
int x547 = 0;
while (x547 != 20) {
int x548 = x547;
int x549 = x546 + x500[x548];
x526[x549] = x526[x549] + -1.0 * x531[x548];
x546 = x546 + 26;
x547 = x547 + 1;
}
float* x550 = (float*)myMalloc(20 * sizeof(float));
int x551 = 0;
while (x551 != 20) {
int x552 = x551;
int x553 = x552;
int x554 = x552 * 26;
int x555 = 0;
while (x555 != 26) {
int x556 = 0;
while (x556 != 1) {
int x557 = x556;
int x558 = x553 + x557;
x550[x558] = x550[x558] + x526[x554 + x557];
x556 = x556 + 1;
}
x554 = x554 + 1;
x555 = x555 + 1;
}
x551 = x551 + 1;
}
int x559 = 0;
int x560 = 0;
while (x560 != 20) {
int x561 = x560;
int x562 = 0;
while (x562 != 26) {
int x563 = x559;
x494[x563] = x494[x563] + (x526[x559] - (float)exp((double)x508[x559]) * x550[x561]);
x559 = x559 + 1;
x562 = x562 + 1;
}
x560 = x560 + 1;
}
int x564 = 0;
while (x564 != 20) {
int x565 = 0;
int x566 = 26 * x564;
while (x565 != 26) {
int x567 = x565;
x44[x567] = x44[x567] + x494[x566 + x567];
x565 = x565 + 1;
}
x564 = x564 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x494,26,x40,26,1,x492,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x486,50,x494,26,1,x42,26);
// backprop for * op;
int x568 = 0;
while (x568 != 20) {
int x569 = 0;
int x570 = 50 * x568;
while (x569 != 50) {
int x571 = x570 + x569;
x439[x571] = x439[x571] + x492[x571] * x482[x571];
x485[x571] = x485[x571] + x492[x571] * x436[x571];
x569 = x569 + 1;
}
x568 = x568 + 1;
}
int x572 = 0;
while (x572 != 1000) {
int x573 = x572;
float x574 = x482[x573];
x481[x573] = x481[x573] + (1.0 - x574 * x574) * x485[x573];
x572 = x572 + 1;
}
// back prop for + op;
int x575 = 0;
while (x575 != 20) {
int x576 = 0;
int x577 = 50 * x575;
while (x576 != 50) {
int x578 = x577 + x576;
x467[x578] = x467[x578] + x481[x578];
x474[x578] = x474[x578] + x481[x578];
x576 = x576 + 1;
}
x575 = x575 + 1;
}
// backprop for * op;
int x579 = 0;
while (x579 != 20) {
int x580 = 0;
int x581 = 50 * x579;
while (x580 != 50) {
int x582 = x581 + x580;
x418[x582] = x418[x582] + x474[x582] * x457[x582];
x460[x582] = x460[x582] + x474[x582] * x415[x582];
x580 = x580 + 1;
}
x579 = x579 + 1;
}
// backprop for * op;
int x583 = 0;
while (x583 != 20) {
int x584 = 0;
int x585 = 50 * x583;
while (x584 != 50) {
int x586 = x585 + x584;
x397[x586] = x397[x586] + x467[x586] * x371[x586];
x372[x586] = x372[x586] + x467[x586] * x394[x586];
x584 = x584 + 1;
}
x583 = x583 + 1;
}
int x587 = 0;
while (x587 != 1000) {
int x588 = x587;
float x589 = x457[x588];
x456[x588] = x456[x588] + (1.0 - x589 * x589) * x460[x588];
x587 = x587 + 1;
}
// back prop for + op;
int x590 = 0;
while (x590 != 20) {
int x591 = 0;
int x592 = 50 * x590;
while (x591 != 50) {
int x593 = x591;
int x594 = x592 + x593;
x450[x594] = x450[x594] + x456[x594];
x31[x593] = x31[x593] + x456[x594];
x591 = x591 + 1;
}
x590 = x590 + 1;
}
// back prop for + op;
int x595 = 0;
while (x595 != 20) {
int x596 = 0;
int x597 = 50 * x595;
while (x596 != 50) {
int x598 = x597 + x596;
x441[x598] = x441[x598] + x450[x598];
x443[x598] = x443[x598] + x450[x598];
x596 = x596 + 1;
}
x595 = x595 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x443,50,x27,50,1,x370,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x369,50,x443,50,1,x29,50);
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x441,50,x24,50,1,x376,26);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x375,26,x441,50,1,x26,50);
int x599 = 0;
while (x599 != 1000) {
int x600 = x599;
float x601 = x436[x600];
x435[x600] = x435[x600] + (1.0 - x601) * x601 * x439[x600];
x599 = x599 + 1;
}
// back prop for + op;
int x602 = 0;
while (x602 != 20) {
int x603 = 0;
int x604 = 50 * x602;
while (x603 != 50) {
int x605 = x603;
int x606 = x604 + x605;
x429[x606] = x429[x606] + x435[x606];
x39[x605] = x39[x605] + x435[x606];
x603 = x603 + 1;
}
x602 = x602 + 1;
}
// back prop for + op;
int x607 = 0;
while (x607 != 20) {
int x608 = 0;
int x609 = 50 * x607;
while (x608 != 50) {
int x610 = x609 + x608;
x420[x610] = x420[x610] + x429[x610];
x422[x610] = x422[x610] + x429[x610];
x608 = x608 + 1;
}
x607 = x607 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x422,50,x35,50,1,x370,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x369,50,x422,50,1,x37,50);
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x420,50,x32,50,1,x376,26);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x375,26,x420,50,1,x34,50);
int x611 = 0;
while (x611 != 1000) {
int x612 = x611;
float x613 = x415[x612];
x414[x612] = x414[x612] + (1.0 - x613) * x613 * x418[x612];
x611 = x611 + 1;
}
// back prop for + op;
int x614 = 0;
while (x614 != 20) {
int x615 = 0;
int x616 = 50 * x614;
while (x615 != 50) {
int x617 = x615;
int x618 = x616 + x617;
x408[x618] = x408[x618] + x414[x618];
x23[x617] = x23[x617] + x414[x618];
x615 = x615 + 1;
}
x614 = x614 + 1;
}
// back prop for + op;
int x619 = 0;
while (x619 != 20) {
int x620 = 0;
int x621 = 50 * x619;
while (x620 != 50) {
int x622 = x621 + x620;
x399[x622] = x399[x622] + x408[x622];
x401[x622] = x401[x622] + x408[x622];
x620 = x620 + 1;
}
x619 = x619 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x401,50,x19,50,1,x370,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x369,50,x401,50,1,x21,50);
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x399,50,x16,50,1,x376,26);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x375,26,x399,50,1,x18,50);
int x623 = 0;
while (x623 != 1000) {
int x624 = x623;
float x625 = x394[x624];
x393[x624] = x393[x624] + (1.0 - x625) * x625 * x397[x624];
x623 = x623 + 1;
}
// back prop for + op;
int x626 = 0;
while (x626 != 20) {
int x627 = 0;
int x628 = 50 * x626;
while (x627 != 50) {
int x629 = x627;
int x630 = x628 + x629;
x387[x630] = x387[x630] + x393[x630];
x15[x629] = x15[x629] + x393[x630];
x627 = x627 + 1;
}
x626 = x626 + 1;
}
// back prop for + op;
int x631 = 0;
while (x631 != 20) {
int x632 = 0;
int x633 = 50 * x631;
while (x632 != 50) {
int x634 = x633 + x632;
x378[x634] = x378[x634] + x387[x634];
x380[x634] = x380[x634] + x387[x634];
x632 = x632 + 1;
}
x631 = x631 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x380,50,x11,50,1,x370,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x369,50,x380,50,1,x13,50);
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x378,50,x8,50,1,x376,26);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x375,26,x378,50,1,x10,50);
} else {
float x635 = 0.0;
int x636 = 0;
while (x636 != 1) {
x635 = x635 + x366[x636];
x636 = x636 + 1;
}
float x637 = x635;
float* x638 = (float*)myMalloc(1 * sizeof(float));
int x639 = 0;
while (x639 != 1) {
x638[x639] = x637;
x639 = x639 + 1;
}
float* x640 = (float*)myMalloc(1 * sizeof(float));
// make sure the size of loss is 1;
int x641 = 0;
while (x641 != 1) {
x640[x641] = x65;
x641 = x641 + 1;
}
// backend is lantern.TensorDslCPU$BackendCPU@5fb0a9cd;
int x642 = 0;
while (x642 != 1) {
int x643 = x642;
x71[x643] = x638[x643];
x642 = x642 + 1;
}
// 'sum' gradient.;
int x644 = 0;
while (x644 != 1) {
x368[0] = x368[0] + x640[0];
x644 = x644 + 1;
}
}};
float** x645 = (float**)myMalloc(6 * sizeof(float*));
x645[0] = x79;
x645[1] = x80;
x645[2] = x81;
x645[3] = x82;
x645[4] = x83;
x645[5] = x84;
x365(0, x645);
float x646 = x71[0];
if (x66 % 100 == 0) {
printf("iter %d, loss %f\n", x66, x646);
x60[x66 / 100] = (double)x646;
} else {

}
int x647 = 0;
while (x647 != 1300) {
int x648 = x647;
float x649 = x18[x648];
if (x649 > 5.0) x649 = 5.0; else {

}
if (x649 < -5.0) x649 = -5.0; else {

}
x45[x648] = x45[x648] + x649 * x649;
x16[x648] = x16[x648] - 0.1 * x649 / (float)sqrt((double)x45[x648] + 9.99999993922529E-9);
x18[x648] = 0.0;
x647 = x647 + 1;
}
int x650 = 0;
while (x650 != 50) {
int x651 = x650;
float x652 = x23[x651];
if (x652 > 5.0) x652 = 5.0; else {

}
if (x652 < -5.0) x652 = -5.0; else {

}
x46[x651] = x46[x651] + x652 * x652;
x22[x651] = x22[x651] - 0.1 * x652 / (float)sqrt((double)x46[x651] + 9.99999993922529E-9);
x23[x651] = 0.0;
x650 = x650 + 1;
}
int x653 = 0;
while (x653 != 2500) {
int x654 = x653;
float x655 = x21[x654];
if (x655 > 5.0) x655 = 5.0; else {

}
if (x655 < -5.0) x655 = -5.0; else {

}
x47[x654] = x47[x654] + x655 * x655;
x19[x654] = x19[x654] - 0.1 * x655 / (float)sqrt((double)x47[x654] + 9.99999993922529E-9);
x21[x654] = 0.0;
x653 = x653 + 1;
}
int x656 = 0;
while (x656 != 50) {
int x657 = x656;
float x658 = x15[x657];
if (x658 > 5.0) x658 = 5.0; else {

}
if (x658 < -5.0) x658 = -5.0; else {

}
x48[x657] = x48[x657] + x658 * x658;
x14[x657] = x14[x657] - 0.1 * x658 / (float)sqrt((double)x48[x657] + 9.99999993922529E-9);
x15[x657] = 0.0;
x656 = x656 + 1;
}
int x659 = 0;
while (x659 != 2500) {
int x660 = x659;
float x661 = x13[x660];
if (x661 > 5.0) x661 = 5.0; else {

}
if (x661 < -5.0) x661 = -5.0; else {

}
x49[x660] = x49[x660] + x661 * x661;
x11[x660] = x11[x660] - 0.1 * x661 / (float)sqrt((double)x49[x660] + 9.99999993922529E-9);
x13[x660] = 0.0;
x659 = x659 + 1;
}
int x662 = 0;
while (x662 != 1300) {
int x663 = x662;
float x664 = x10[x663];
if (x664 > 5.0) x664 = 5.0; else {

}
if (x664 < -5.0) x664 = -5.0; else {

}
x50[x663] = x50[x663] + x664 * x664;
x8[x663] = x8[x663] - 0.1 * x664 / (float)sqrt((double)x50[x663] + 9.99999993922529E-9);
x10[x663] = 0.0;
x662 = x662 + 1;
}
int x665 = 0;
while (x665 != 1300) {
int x666 = x665;
float x667 = x26[x666];
if (x667 > 5.0) x667 = 5.0; else {

}
if (x667 < -5.0) x667 = -5.0; else {

}
x51[x666] = x51[x666] + x667 * x667;
x24[x666] = x24[x666] - 0.1 * x667 / (float)sqrt((double)x51[x666] + 9.99999993922529E-9);
x26[x666] = 0.0;
x665 = x665 + 1;
}
int x668 = 0;
while (x668 != 50) {
int x669 = x668;
float x670 = x31[x669];
if (x670 > 5.0) x670 = 5.0; else {

}
if (x670 < -5.0) x670 = -5.0; else {

}
x52[x669] = x52[x669] + x670 * x670;
x30[x669] = x30[x669] - 0.1 * x670 / (float)sqrt((double)x52[x669] + 9.99999993922529E-9);
x31[x669] = 0.0;
x668 = x668 + 1;
}
int x671 = 0;
while (x671 != 2500) {
int x672 = x671;
float x673 = x29[x672];
if (x673 > 5.0) x673 = 5.0; else {

}
if (x673 < -5.0) x673 = -5.0; else {

}
x53[x672] = x53[x672] + x673 * x673;
x27[x672] = x27[x672] - 0.1 * x673 / (float)sqrt((double)x53[x672] + 9.99999993922529E-9);
x29[x672] = 0.0;
x671 = x671 + 1;
}
int x674 = 0;
while (x674 != 26) {
int x675 = x674;
float x676 = x44[x675];
if (x676 > 5.0) x676 = 5.0; else {

}
if (x676 < -5.0) x676 = -5.0; else {

}
x54[x675] = x54[x675] + x676 * x676;
x43[x675] = x43[x675] - 0.1 * x676 / (float)sqrt((double)x54[x675] + 9.99999993922529E-9);
x44[x675] = 0.0;
x674 = x674 + 1;
}
int x677 = 0;
while (x677 != 1300) {
int x678 = x677;
float x679 = x42[x678];
if (x679 > 5.0) x679 = 5.0; else {

}
if (x679 < -5.0) x679 = -5.0; else {

}
x55[x678] = x55[x678] + x679 * x679;
x40[x678] = x40[x678] - 0.1 * x679 / (float)sqrt((double)x55[x678] + 9.99999993922529E-9);
x42[x678] = 0.0;
x677 = x677 + 1;
}
int x680 = 0;
while (x680 != 2500) {
int x681 = x680;
float x682 = x37[x681];
if (x682 > 5.0) x682 = 5.0; else {

}
if (x682 < -5.0) x682 = -5.0; else {

}
x56[x681] = x56[x681] + x682 * x682;
x35[x681] = x35[x681] - 0.1 * x682 / (float)sqrt((double)x56[x681] + 9.99999993922529E-9);
x37[x681] = 0.0;
x680 = x680 + 1;
}
int x683 = 0;
while (x683 != 1300) {
int x684 = x683;
float x685 = x34[x684];
if (x685 > 5.0) x685 = 5.0; else {

}
if (x685 < -5.0) x685 = -5.0; else {

}
x57[x684] = x57[x684] + x685 * x685;
x32[x684] = x32[x684] - 0.1 * x685 / (float)sqrt((double)x57[x684] + 9.99999993922529E-9);
x34[x684] = 0.0;
x683 = x683 + 1;
}
int x686 = 0;
while (x686 != 50) {
int x687 = x686;
float x688 = x39[x687];
if (x688 > 5.0) x688 = 5.0; else {

}
if (x688 < -5.0) x688 = -5.0; else {

}
x58[x687] = x58[x687] + x688 * x688;
x38[x687] = x38[x687] - 0.1 * x688 / (float)sqrt((double)x58[x687] + 9.99999993922529E-9);
x39[x687] = 0.0;
x686 = x686 + 1;
}
memset((void*)x61, 0, (long)mallocAddr - x61);
mallocAddr = (void*)x61;
x64 = x64 + 1;
}
double x689 = ((double)clock() / CLOCKS_PER_SEC);
long x690 = (long)fopen(x0, "w");
fprintf((FILE *)x690, "unit: %s\n", "100 iteration");
int x691 = 0;
while (x691 != 51) {
fprintf((FILE *)x690, "%lf\n", x60[x691]);
x691 = x691 + 1;
}
fprintf((FILE *)x690, "run time: %lf %lf\n", x59 - x1, x689 - x59);
fclose((FILE*)x690);
// Backend cleanup.;
}

    /*****************************************
    End of C Generated Code
    *******************************************/
    


