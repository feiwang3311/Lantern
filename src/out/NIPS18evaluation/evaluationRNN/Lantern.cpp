
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
x16[x17] = ((float)rand()/RAND_MAX - 0.5) * 0.14142136;
x17 = x17 + 1;
}
float* x18 = (float*)myMalloc(1300 * sizeof(float));
float* x19 = (float*)myMalloc(26 * sizeof(float));
float* x20 = (float*)myMalloc(26 * sizeof(float));
float* x21 = (float*)myMalloc(26 * sizeof(float));
float* x22 = (float*)myMalloc(1300 * sizeof(float));
float* x23 = (float*)myMalloc(2500 * sizeof(float));
float* x24 = (float*)myMalloc(50 * sizeof(float));
float* x25 = (float*)myMalloc(1300 * sizeof(float));
double* x26 = (double*)myMalloc(51 * sizeof(double));
double x27 = ((double)clock() / CLOCKS_PER_SEC);
long x28 = (long)mallocAddr;
int x29 = 0;
x29 = x29 - 400;
int x30 = 0;
float x31 = (float)1;
while (x30 != 5001) {
int x32 = x30;
x29 = x29 + 400;
if (x29 + 400 + 1 >= x3) x29 = 0; else {

}
int* x33 = (int*)myMalloc(400 * sizeof(int));
int* x34 = (int*)myMalloc(400 * sizeof(int));
int x35 = 0;
while (x35 != 400) {
int x36 = x35;
x33[x36] = x5[x29 + x36];
x34[x36] = x5[x29 + x36 + 1];
x35 = x35 + 1;
}
(float*)myMalloc(1 * sizeof(float));
(float*)myMalloc(1 * sizeof(float));
// allocate memory to save the final loss in CPU Tensor;
float* x37 = (float*)myMalloc(1 * sizeof(float));
float* x38 = (float*)myMalloc(10400 * sizeof(float));
int x39 = 0;
while (x39 != 20) {
int x40 = x39;
int x41 = 0;
int x42 = x40 * 26 * 20;
while (x41 != 20) {
int x43 = x41;
x38[x42 + x43 * 26 + x33[x43 * 20 + x40]] = 1.0;
x41 = x41 + 1;
}
x39 = x39 + 1;
}
float* x44 = (float*)myMalloc(10400 * sizeof(float));
float* x45 = (float*)myMalloc(1 * sizeof(float));
float* x46 = (float*)myMalloc(1 * sizeof(float));
float* x47 = (float*)myMalloc(1000 * sizeof(float));
float* x48 = (float*)myMalloc(1000 * sizeof(float));
function<void(int,float**)> x49 = [&](int x55, float** x51) {float* x50 = x51[0];
float* x52 = x51[1];
float* x53 = x51[2];
float* x54 = x51[3];
if (x55 < 20) {
int x56 = x55 * 520;
float* x57 = x38+x56;
float* x58 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x57,26,x8,50,0,x58,50);
float* x59 = (float*)myMalloc(1000 * sizeof(float));
float* x60 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x53,50,x11,50,0,x60,50);
float* x61 = (float*)myMalloc(1000 * sizeof(float));
float* x62 = (float*)myMalloc(1000 * sizeof(float));
int x63 = 0;
while (x63 != 20) {
int x64 = 0;
int x65 = 50 * x63;
while (x64 != 50) {
int x66 = x64;
int x67 = x65 + x66;
x62[x66 + x65] = x58[x67] + x60[x67];
x64 = x64 + 1;
}
x63 = x63 + 1;
}
float* x68 = (float*)myMalloc(1000 * sizeof(float));
float* x69 = (float*)myMalloc(1000 * sizeof(float));
int x70 = 0;
while (x70 != 20) {
int x71 = 0;
int x72 = 50 * x70;
while (x71 != 50) {
int x73 = x71;
x69[x73 + x72] = x62[x72 + x73] + x14[x73];
x71 = x71 + 1;
}
x70 = x70 + 1;
}
float* x74 = (float*)myMalloc(1000 * sizeof(float));
float* x75 = (float*)myMalloc(1000 * sizeof(float));
int x76 = 0;
while (x76 != 1000) {
int x77 = x76;
x75[x77] = (float)tanh((double)x69[x77]);
x76 = x76 + 1;
}
float* x78 = (float*)myMalloc(1000 * sizeof(float));
float* x79 = (float*)myMalloc(520 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x75,50,x16,26,0,x79,26);
float* x80 = (float*)myMalloc(520 * sizeof(float));
int x81 = 0;
while (x81 != 20) {
int x82 = 0;
int x83 = 26 * x81;
while (x82 != 26) {
int x84 = x82;
int x85 = x83 + x84;
x79[x85] = x79[x85] + x19[x84];
x82 = x82 + 1;
}
x81 = x81 + 1;
}
int* x86 = (int*)myMalloc(20 * sizeof(int));
int x87 = 0;
while (x87 != 20) {
int x88 = x87;
x86[x88] = x34[x55 + x88 * 20];
x87 = x87 + 1;
}
float* x89 = (float*)myMalloc(20 * sizeof(float));
int x90 = 0;
int x91 = 0;
while (x91 != 20) {
float x92 = -3.4028235E38;
int x93 = 0;
while (x93 != 26) {
if (x79[x90] > x92) x92 = x79[x90]; else {

}
x90 = x90 + 1;
x93 = x93 + 1;
}
x89[x91] = x92;
x91 = x91 + 1;
}
float* x94 = (float*)myMalloc(520 * sizeof(float));
int x95 = 0;
int x96 = 0;
while (x96 != 20) {
int x97 = x96;
int x98 = 0;
while (x98 != 26) {
x94[x95] = (float)exp((double)(x79[x95] - x89[x97]));
x95 = x95 + 1;
x98 = x98 + 1;
}
x96 = x96 + 1;
}
float* x99 = (float*)myMalloc(20 * sizeof(float));
int x100 = 0;
while (x100 != 20) {
int x101 = x100;
int x102 = x101;
int x103 = x101 * 26;
int x104 = 0;
while (x104 != 26) {
int x105 = 0;
while (x105 != 1) {
int x106 = x105;
int x107 = x102 + x106;
x99[x107] = x99[x107] + x94[x103 + x106];
x105 = x105 + 1;
}
x103 = x103 + 1;
x104 = x104 + 1;
}
x100 = x100 + 1;
}
x95 = 0;
int x108 = 0;
while (x108 != 20) {
int x109 = x108;
float x110 = x89[x109] + (float)log((double)x99[x109]);
int x111 = 0;
while (x111 != 26) {
x94[x95] = x79[x95] - x110;
x95 = x95 + 1;
x111 = x111 + 1;
}
x108 = x108 + 1;
}
float* x112 = (float*)myMalloc(520 * sizeof(float));
// nllLoss forward in CPU;
float* x113 = (float*)myMalloc(20 * sizeof(float));
int x114 = 0;
int x115 = 0;
while (x115 != 20) {
int x116 = x115;
x113[x116] = -1.0 * x94[x114 + x86[x116]];
x114 = x114 + 26;
x115 = x115 + 1;
}
float* x117 = (float*)myMalloc(20 * sizeof(float));
float x118 = 0.0;
int x119 = 0;
while (x119 != 20) {
x118 = x118 + x113[x119];
x119 = x119 + 1;
}
float x120 = x118;
float* x121 = (float*)myMalloc(1 * sizeof(float));
int x122 = 0;
while (x122 != 1) {
x121[x122] = x120;
x122 = x122 + 1;
}
float* x123 = (float*)myMalloc(1 * sizeof(float));
float* x124 = (float*)myMalloc(1 * sizeof(float));
int x125 = 0;
while (x125 != 1) {
x124[x125] = x50[0] + x121[0];
x125 = x125 + 1;
}
float* x126 = (float*)myMalloc(1 * sizeof(float));
float** x127 = (float**)myMalloc(4 * sizeof(float*));
x127[0] = x124;
x127[1] = x126;
x127[2] = x75;
x127[3] = x78;
x49(x55 + 1, x127);
// back prop for + op;
int x128 = 0;
while (x128 != 1) {
int x129 = x128;
x52[0] = x52[0] + x126[x129];
x123[0] = x123[0] + x126[x129];
x128 = x128 + 1;
}
// 'sum' gradient.;
int x130 = 0;
while (x130 != 20) {
int x131 = x130;
x117[x131] = x117[x131] + x123[0];
x130 = x130 + 1;
}
// 'nllLossB' gradient.;
// nllLoss_grad implementation in CPU;
int x132 = 0;
int x133 = 0;
while (x133 != 20) {
int x134 = x133;
int x135 = x132 + x86[x134];
x112[x135] = x112[x135] + -1.0 * x117[x134];
x132 = x132 + 26;
x133 = x133 + 1;
}
float* x136 = (float*)myMalloc(20 * sizeof(float));
int x137 = 0;
while (x137 != 20) {
int x138 = x137;
int x139 = x138;
int x140 = x138 * 26;
int x141 = 0;
while (x141 != 26) {
int x142 = 0;
while (x142 != 1) {
int x143 = x142;
int x144 = x139 + x143;
x136[x144] = x136[x144] + x112[x140 + x143];
x142 = x142 + 1;
}
x140 = x140 + 1;
x141 = x141 + 1;
}
x137 = x137 + 1;
}
int x145 = 0;
int x146 = 0;
while (x146 != 20) {
int x147 = x146;
int x148 = 0;
while (x148 != 26) {
int x149 = x145;
x80[x149] = x80[x149] + (x112[x145] - (float)exp((double)x94[x145]) * x136[x147]);
x145 = x145 + 1;
x148 = x148 + 1;
}
x146 = x146 + 1;
}
int x150 = 0;
while (x150 != 20) {
int x151 = 0;
int x152 = 26 * x150;
while (x151 != 26) {
int x153 = x151;
x20[x153] = x20[x153] + x80[x152 + x153];
x151 = x151 + 1;
}
x150 = x150 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x80,26,x16,26,1,x78,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x75,50,x80,26,1,x18,26);
int x154 = 0;
while (x154 != 1000) {
int x155 = x154;
float x156 = x75[x155];
x74[x155] = x74[x155] + (1.0 - x156 * x156) * x78[x155];
x154 = x154 + 1;
}
// back prop for + op;
int x157 = 0;
while (x157 != 20) {
int x158 = 0;
int x159 = 50 * x157;
while (x158 != 50) {
int x160 = x158;
int x161 = x159 + x160;
x68[x161] = x68[x161] + x74[x161];
x15[x160] = x15[x160] + x74[x161];
x158 = x158 + 1;
}
x157 = x157 + 1;
}
// back prop for + op;
int x162 = 0;
while (x162 != 20) {
int x163 = 0;
int x164 = 50 * x162;
while (x163 != 50) {
int x165 = x164 + x163;
x59[x165] = x59[x165] + x68[x165];
x61[x165] = x61[x165] + x68[x165];
x163 = x163 + 1;
}
x162 = x162 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x61,50,x11,50,1,x54,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x53,50,x61,50,1,x13,50);
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x59,50,x8,50,1,x44+x56,26);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x57,26,x59,50,1,x10,50);
} else {
float x166 = 0.0;
int x167 = 0;
while (x167 != 1) {
x166 = x166 + x50[x167];
x167 = x167 + 1;
}
float x168 = x166;
float* x169 = (float*)myMalloc(1 * sizeof(float));
int x170 = 0;
while (x170 != 1) {
x169[x170] = x168;
x170 = x170 + 1;
}
float* x171 = (float*)myMalloc(1 * sizeof(float));
// make sure the size of loss is 1;
int x172 = 0;
while (x172 != 1) {
x171[x172] = x31;
x172 = x172 + 1;
}
// backend is lantern.TensorDslCPU$BackendCPU@6d554e8e;
int x173 = 0;
while (x173 != 1) {
int x174 = x173;
x37[x174] = x169[x174];
x173 = x173 + 1;
}
// 'sum' gradient.;
int x175 = 0;
while (x175 != 1) {
x52[0] = x52[0] + x171[0];
x175 = x175 + 1;
}
}};
function<void(int,float**)> x176 = [&](int x182, float** x178) {float* x177 = x178[0];
float* x179 = x178[1];
float* x180 = x178[2];
float* x181 = x178[3];
if (x182 < 20) {
int x183 = x182 * 520;
float* x184 = x38+x183;
float* x185 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x184,26,x8,50,0,x185,50);
float* x186 = (float*)myMalloc(1000 * sizeof(float));
float* x187 = (float*)myMalloc(1000 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x180,50,x11,50,0,x187,50);
float* x188 = (float*)myMalloc(1000 * sizeof(float));
float* x189 = (float*)myMalloc(1000 * sizeof(float));
int x190 = 0;
while (x190 != 20) {
int x191 = 0;
int x192 = 50 * x190;
while (x191 != 50) {
int x193 = x191;
int x194 = x192 + x193;
x189[x193 + x192] = x185[x194] + x187[x194];
x191 = x191 + 1;
}
x190 = x190 + 1;
}
float* x195 = (float*)myMalloc(1000 * sizeof(float));
float* x196 = (float*)myMalloc(1000 * sizeof(float));
int x197 = 0;
while (x197 != 20) {
int x198 = 0;
int x199 = 50 * x197;
while (x198 != 50) {
int x200 = x198;
x196[x200 + x199] = x189[x199 + x200] + x14[x200];
x198 = x198 + 1;
}
x197 = x197 + 1;
}
float* x201 = (float*)myMalloc(1000 * sizeof(float));
float* x202 = (float*)myMalloc(1000 * sizeof(float));
int x203 = 0;
while (x203 != 1000) {
int x204 = x203;
x202[x204] = (float)tanh((double)x196[x204]);
x203 = x203 + 1;
}
float* x205 = (float*)myMalloc(1000 * sizeof(float));
float* x206 = (float*)myMalloc(520 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x202,50,x16,26,0,x206,26);
float* x207 = (float*)myMalloc(520 * sizeof(float));
int x208 = 0;
while (x208 != 20) {
int x209 = 0;
int x210 = 26 * x208;
while (x209 != 26) {
int x211 = x209;
int x212 = x210 + x211;
x206[x212] = x206[x212] + x19[x211];
x209 = x209 + 1;
}
x208 = x208 + 1;
}
int* x213 = (int*)myMalloc(20 * sizeof(int));
int x214 = 0;
while (x214 != 20) {
int x215 = x214;
x213[x215] = x34[x182 + x215 * 20];
x214 = x214 + 1;
}
float* x216 = (float*)myMalloc(20 * sizeof(float));
int x217 = 0;
int x218 = 0;
while (x218 != 20) {
float x219 = -3.4028235E38;
int x220 = 0;
while (x220 != 26) {
if (x206[x217] > x219) x219 = x206[x217]; else {

}
x217 = x217 + 1;
x220 = x220 + 1;
}
x216[x218] = x219;
x218 = x218 + 1;
}
float* x221 = (float*)myMalloc(520 * sizeof(float));
int x222 = 0;
int x223 = 0;
while (x223 != 20) {
int x224 = x223;
int x225 = 0;
while (x225 != 26) {
x221[x222] = (float)exp((double)(x206[x222] - x216[x224]));
x222 = x222 + 1;
x225 = x225 + 1;
}
x223 = x223 + 1;
}
float* x226 = (float*)myMalloc(20 * sizeof(float));
int x227 = 0;
while (x227 != 20) {
int x228 = x227;
int x229 = x228;
int x230 = x228 * 26;
int x231 = 0;
while (x231 != 26) {
int x232 = 0;
while (x232 != 1) {
int x233 = x232;
int x234 = x229 + x233;
x226[x234] = x226[x234] + x221[x230 + x233];
x232 = x232 + 1;
}
x230 = x230 + 1;
x231 = x231 + 1;
}
x227 = x227 + 1;
}
x222 = 0;
int x235 = 0;
while (x235 != 20) {
int x236 = x235;
float x237 = x216[x236] + (float)log((double)x226[x236]);
int x238 = 0;
while (x238 != 26) {
x221[x222] = x206[x222] - x237;
x222 = x222 + 1;
x238 = x238 + 1;
}
x235 = x235 + 1;
}
float* x239 = (float*)myMalloc(520 * sizeof(float));
// nllLoss forward in CPU;
float* x240 = (float*)myMalloc(20 * sizeof(float));
int x241 = 0;
int x242 = 0;
while (x242 != 20) {
int x243 = x242;
x240[x243] = -1.0 * x221[x241 + x213[x243]];
x241 = x241 + 26;
x242 = x242 + 1;
}
float* x244 = (float*)myMalloc(20 * sizeof(float));
float x245 = 0.0;
int x246 = 0;
while (x246 != 20) {
x245 = x245 + x240[x246];
x246 = x246 + 1;
}
float x247 = x245;
float* x248 = (float*)myMalloc(1 * sizeof(float));
int x249 = 0;
while (x249 != 1) {
x248[x249] = x247;
x249 = x249 + 1;
}
float* x250 = (float*)myMalloc(1 * sizeof(float));
float* x251 = (float*)myMalloc(1 * sizeof(float));
int x252 = 0;
while (x252 != 1) {
x251[x252] = x177[0] + x248[0];
x252 = x252 + 1;
}
float* x253 = (float*)myMalloc(1 * sizeof(float));
float** x254 = (float**)myMalloc(4 * sizeof(float*));
x254[0] = x251;
x254[1] = x253;
x254[2] = x202;
x254[3] = x205;
x49(x182 + 1, x254);
// back prop for + op;
int x255 = 0;
while (x255 != 1) {
int x256 = x255;
x179[0] = x179[0] + x253[x256];
x250[0] = x250[0] + x253[x256];
x255 = x255 + 1;
}
// 'sum' gradient.;
int x257 = 0;
while (x257 != 20) {
int x258 = x257;
x244[x258] = x244[x258] + x250[0];
x257 = x257 + 1;
}
// 'nllLossB' gradient.;
// nllLoss_grad implementation in CPU;
int x259 = 0;
int x260 = 0;
while (x260 != 20) {
int x261 = x260;
int x262 = x259 + x213[x261];
x239[x262] = x239[x262] + -1.0 * x244[x261];
x259 = x259 + 26;
x260 = x260 + 1;
}
float* x263 = (float*)myMalloc(20 * sizeof(float));
int x264 = 0;
while (x264 != 20) {
int x265 = x264;
int x266 = x265;
int x267 = x265 * 26;
int x268 = 0;
while (x268 != 26) {
int x269 = 0;
while (x269 != 1) {
int x270 = x269;
int x271 = x266 + x270;
x263[x271] = x263[x271] + x239[x267 + x270];
x269 = x269 + 1;
}
x267 = x267 + 1;
x268 = x268 + 1;
}
x264 = x264 + 1;
}
int x272 = 0;
int x273 = 0;
while (x273 != 20) {
int x274 = x273;
int x275 = 0;
while (x275 != 26) {
int x276 = x272;
x207[x276] = x207[x276] + (x239[x272] - (float)exp((double)x221[x272]) * x263[x274]);
x272 = x272 + 1;
x275 = x275 + 1;
}
x273 = x273 + 1;
}
int x277 = 0;
while (x277 != 20) {
int x278 = 0;
int x279 = 26 * x277;
while (x278 != 26) {
int x280 = x278;
x20[x280] = x20[x280] + x207[x279 + x280];
x278 = x278 + 1;
}
x277 = x277 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x207,26,x16,26,1,x205,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x202,50,x207,26,1,x18,26);
int x281 = 0;
while (x281 != 1000) {
int x282 = x281;
float x283 = x202[x282];
x201[x282] = x201[x282] + (1.0 - x283 * x283) * x205[x282];
x281 = x281 + 1;
}
// back prop for + op;
int x284 = 0;
while (x284 != 20) {
int x285 = 0;
int x286 = 50 * x284;
while (x285 != 50) {
int x287 = x285;
int x288 = x286 + x287;
x195[x288] = x195[x288] + x201[x288];
x15[x287] = x15[x287] + x201[x288];
x285 = x285 + 1;
}
x284 = x284 + 1;
}
// back prop for + op;
int x289 = 0;
while (x289 != 20) {
int x290 = 0;
int x291 = 50 * x289;
while (x290 != 50) {
int x292 = x291 + x290;
x186[x292] = x186[x292] + x195[x292];
x188[x292] = x188[x292] + x195[x292];
x290 = x290 + 1;
}
x289 = x289 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x188,50,x11,50,1,x181,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x180,50,x188,50,1,x13,50);
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x186,50,x8,50,1,x44+x183,26);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x184,26,x186,50,1,x10,50);
} else {
float x293 = 0.0;
int x294 = 0;
while (x294 != 1) {
x293 = x293 + x177[x294];
x294 = x294 + 1;
}
float x295 = x293;
float* x296 = (float*)myMalloc(1 * sizeof(float));
int x297 = 0;
while (x297 != 1) {
x296[x297] = x295;
x297 = x297 + 1;
}
float* x298 = (float*)myMalloc(1 * sizeof(float));
// make sure the size of loss is 1;
int x299 = 0;
while (x299 != 1) {
x298[x299] = x31;
x299 = x299 + 1;
}
// backend is lantern.TensorDslCPU$BackendCPU@6d554e8e;
int x300 = 0;
while (x300 != 1) {
int x301 = x300;
x37[x301] = x296[x301];
x300 = x300 + 1;
}
// 'sum' gradient.;
int x302 = 0;
while (x302 != 1) {
x179[0] = x179[0] + x298[0];
x302 = x302 + 1;
}
}};
float** x303 = (float**)myMalloc(4 * sizeof(float*));
x303[0] = x45;
x303[1] = x46;
x303[2] = x47;
x303[3] = x48;
x176(0, x303);
float x304 = x37[0];
if (x32 % 100 == 0) {
printf("iter %d, loss %f\n", x32, x304);
x26[x32 / 100] = (double)x304;
} else {

}
int x305 = 0;
while (x305 != 26) {
int x306 = x305;
float x307 = x20[x306];
if (x307 > 5.0) x307 = 5.0; else {

}
if (x307 < -5.0) x307 = -5.0; else {

}
x21[x306] = x21[x306] + x307 * x307;
x19[x306] = x19[x306] - 0.1 * x307 / (float)sqrt((double)x21[x306] + 9.99999993922529E-9);
x20[x306] = 0.0;
x305 = x305 + 1;
}
int x308 = 0;
while (x308 != 1300) {
int x309 = x308;
float x310 = x18[x309];
if (x310 > 5.0) x310 = 5.0; else {

}
if (x310 < -5.0) x310 = -5.0; else {

}
x22[x309] = x22[x309] + x310 * x310;
x16[x309] = x16[x309] - 0.1 * x310 / (float)sqrt((double)x22[x309] + 9.99999993922529E-9);
x18[x309] = 0.0;
x308 = x308 + 1;
}
int x311 = 0;
while (x311 != 2500) {
int x312 = x311;
float x313 = x13[x312];
if (x313 > 5.0) x313 = 5.0; else {

}
if (x313 < -5.0) x313 = -5.0; else {

}
x23[x312] = x23[x312] + x313 * x313;
x11[x312] = x11[x312] - 0.1 * x313 / (float)sqrt((double)x23[x312] + 9.99999993922529E-9);
x13[x312] = 0.0;
x311 = x311 + 1;
}
int x314 = 0;
while (x314 != 50) {
int x315 = x314;
float x316 = x15[x315];
if (x316 > 5.0) x316 = 5.0; else {

}
if (x316 < -5.0) x316 = -5.0; else {

}
x24[x315] = x24[x315] + x316 * x316;
x14[x315] = x14[x315] - 0.1 * x316 / (float)sqrt((double)x24[x315] + 9.99999993922529E-9);
x15[x315] = 0.0;
x314 = x314 + 1;
}
int x317 = 0;
while (x317 != 1300) {
int x318 = x317;
float x319 = x10[x318];
if (x319 > 5.0) x319 = 5.0; else {

}
if (x319 < -5.0) x319 = -5.0; else {

}
x25[x318] = x25[x318] + x319 * x319;
x8[x318] = x8[x318] - 0.1 * x319 / (float)sqrt((double)x25[x318] + 9.99999993922529E-9);
x10[x318] = 0.0;
x317 = x317 + 1;
}
memset((void*)x28, 0, (long)mallocAddr - x28);
mallocAddr = (void*)x28;
x30 = x30 + 1;
}
double x320 = ((double)clock() / CLOCKS_PER_SEC);
long x321 = (long)fopen(x0, "w");
fprintf((FILE *)x321, "unit: %s\n", "100 iteration");
int x322 = 0;
while (x322 != 51) {
fprintf((FILE *)x321, "%lf\n", x26[x322]);
x322 = x322 + 1;
}
fprintf((FILE *)x321, "run time: %lf %lf\n", x27 - x1, x320 - x27);
fclose((FILE*)x321);
// Backend cleanup.;
}

    /*****************************************
    End of C Generated Code
    *******************************************/
    


