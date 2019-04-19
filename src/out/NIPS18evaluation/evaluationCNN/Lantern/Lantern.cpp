
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
srand(42);
struct timeval begin_0, end_0, diff_0;
gettimeofday(&begin_0, NULL);
float* x1 = (float*)myMalloc(250 * sizeof(float));
int x2 = 0;
while (x2 != 250) {
x1[x2] = ((float)rand()/RAND_MAX - 0.5) * 0.9797959;
x2 = x2 + 1;
}
float* x3 = (float*)myMalloc(250 * sizeof(float));
float* x4 = (float*)myMalloc(10 * sizeof(float));
float* x5 = (float*)myMalloc(10 * sizeof(float));
float* x6 = (float*)myMalloc(5000 * sizeof(float));
int x7 = 0;
while (x7 != 5000) {
x6[x7] = ((float)rand()/RAND_MAX - 0.5) * 0.30983868;
x7 = x7 + 1;
}
float* x8 = (float*)myMalloc(5000 * sizeof(float));
float* x9 = (float*)myMalloc(20 * sizeof(float));
float* x10 = (float*)myMalloc(20 * sizeof(float));
float* x11 = (float*)myMalloc(16000 * sizeof(float));
int x12 = 0;
while (x12 != 16000) {
x11[x12] = ((float)rand()/RAND_MAX - 0.5) * 0.0559017;
x12 = x12 + 1;
}
float* x13 = (float*)myMalloc(16000 * sizeof(float));
float* x14 = (float*)myMalloc(50 * sizeof(float));
float* x15 = (float*)myMalloc(50 * sizeof(float));
float* x16 = (float*)myMalloc(500 * sizeof(float));
int x17 = 0;
while (x17 != 500) {
x16[x17] = ((float)rand()/RAND_MAX - 0.5) * 0.14142136;
x17 = x17 + 1;
}
float* x18 = (float*)myMalloc(500 * sizeof(float));
float* x19 = (float*)myMalloc(10 * sizeof(float));
float* x20 = (float*)myMalloc(10 * sizeof(float));
(long*)myMalloc(2 * sizeof(long));
(long*)myMalloc(2 * sizeof(long));
int x21 = open("../data/bin/mnist_train.bin",0);
long x22 = fsize(x21);
float* x23 = (float*)mmap(0, x22, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x21, 0);
int x24 = open("../data/bin/mnist_train_target.bin",0);
long x25 = fsize(x24);
int* x26 = (int*)mmap(0, x25, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x24, 0);
int x27 = (int)x25 / 4;
int x28 = 0;
int x29 = x28;
int x30 = 0;
while (x30 != x27) {
float* x31 = x23+x29;
int x32 = 0;
while (x32 != 784) {
int x33 = x32;
x31[x33] = (x31[x33] - 0.1307) / 0.3081;
x32 = x32 + 1;
}
x29 = x29 + 784;
x30 = x30 + 1;
}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
float x34 = (float)((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec)) / 1000000.0;
printf("Data normalized (all prepare time) in %lf sec\n", x34);
double* x35 = (double*)myMalloc(4 * sizeof(double));
long x36 = (long)mallocAddr;
// training loop starts here;
int x37 = 0;
int x38 = x27 / 100;
float x39 = (float)1;
float x40 = (float)0;
int x41 = x27 / 10;
double x42 = (double)x27;
long x43 = (long)x27;
float x44 = (float)x27;
while (x37 != 4) {
int x45 = x37;
struct timeval begin_1, end_1, diff_1;
int x46 = 0;
int x47 = x46;
float x48 = 0.0;
float x49 = x48;
printf("Start training epoch %d\n", x45 + 1);
gettimeofday(&begin_1, NULL);
int x50 = 0;
int x51 = x50;
int x52 = 0;
while (x52 != x38) {
float* x53 = x23+x51;
int* x54 = x26+x52 * 100;
x47 = x47 + 100;
(float*)myMalloc(2 * sizeof(float));
(float*)myMalloc(4 * sizeof(float));
(float*)myMalloc(4 * sizeof(float));
// allocate memory to save the final loss in CPU Tensor;
float* x55 = (float*)myMalloc(1 * sizeof(float));
float* x56 = (float*)myMalloc(576000 * sizeof(float));
int x57 = 0;
int x58 = 0;
while (x58 != 100) {
int x59 = 0;
while (x59 != 10) {
int x60 = x59;
int x61 = 0;
while (x61 != 576) {
x56[x57] = x4[x60];
x57 = x57 + 1;
x61 = x61 + 1;
}
x59 = x59 + 1;
}
x58 = x58 + 1;
}
float* x62 = (float*)myMalloc(1440000 * sizeof(float));
int x63 = 0;
while (x63 != 100) {
int x64 = x63;
float* x65 = x53+x64 * 784;
float* x66 = x62+x64 * 14400;
int x67 = 0;
while (x67 != 25) {
int x68 = x67;
int x69 = x68 / 25;
int x70 = x68 % 25;
int x71 = x70 / 5;
int x72 = x70 % 5;
float* x73 = x66+x69 * 5 * 5 * 24 * 24 + x71 * 5 * 24 * 24 + x72 * 24 * 24;
float* x74 = x65+x69 * 28 * 28;
int x75 = 0;
while (x75 != 24) {
int x76 = x75;
memcpy(x73+x76 * 24, x74+(x76 + x71) * 28 + x72, 4 * 24);
x75 = x75 + 1;
}
x67 = x67 + 1;
}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 10,576,25,1,x1,25,x66,576,1,x56+x64 * 5760,576);
x63 = x63 + 1;
}
float* x77 = (float*)myMalloc(576000 * sizeof(float));
float* x78 = (float*)myMalloc(576000 * sizeof(float));
int x79 = 0;
while (x79 != 576000) {
int x80 = x79;
if (x56[x80] < 0.0) x78[x80] = 0.0; else x78[x80] = x56[x80];
x79 = x79 + 1;
}
float* x81 = (float*)myMalloc(576000 * sizeof(float));
float* x82 = (float*)myMalloc(144000 * sizeof(float));
int x83 = 0;
while (x83 != 144000) {
x82[x83] = -3.4028235E38;
x83 = x83 + 1;
}
int* x84 = (int*)myMalloc(144000 * sizeof(int));
int x85 = 0;
while (x85 != 100) {
int x86 = x85;
int x87 = x86 * 5760;
float* x88 = x78+x87;
int x89 = x86 * 1440;
float* x90 = x82+x89;
int* x91 = x84+x89;
int x92 = 0;
int x93 = 0;
int x94 = 0;
while (x94 != 10) {
int x95 = x92;
int x96 = x93;
int x97 = 0;
while (x97 != 12) {
int x98 = x95;
int x99 = x96;
int x100 = 0;
while (x100 != 12) {
int x101 = x99;
int x102 = x101;
if (x88[x102] > x90[x98]) {
x90[x98] = x88[x102];
x91[x98] = x102 + x87;
} else {

}
x102 = x102 + 1;
if (x88[x102] > x90[x98]) {
x90[x98] = x88[x102];
x91[x98] = x102 + x87;
} else {

}
x102 = x102 + 1;
x101 = x101 + 24;
int x103 = x101;
if (x88[x103] > x90[x98]) {
x90[x98] = x88[x103];
x91[x98] = x103 + x87;
} else {

}
x103 = x103 + 1;
if (x88[x103] > x90[x98]) {
x90[x98] = x88[x103];
x91[x98] = x103 + x87;
} else {

}
x103 = x103 + 1;
x101 = x101 + 24;
x98 = x98 + 1;
x99 = x99 + 2;
x100 = x100 + 1;
}
x95 = x95 + 12;
x96 = x96 + 48;
x97 = x97 + 1;
}
x92 = x92 + 144;
x93 = x93 + 576;
x94 = x94 + 1;
}
x85 = x85 + 1;
}
float* x104 = (float*)myMalloc(144000 * sizeof(float));
float* x105 = (float*)myMalloc(128000 * sizeof(float));
int x106 = 0;
int x107 = 0;
while (x107 != 100) {
int x108 = 0;
while (x108 != 20) {
int x109 = x108;
int x110 = 0;
while (x110 != 64) {
x105[x106] = x9[x109];
x106 = x106 + 1;
x110 = x110 + 1;
}
x108 = x108 + 1;
}
x107 = x107 + 1;
}
float* x111 = (float*)myMalloc(1600000 * sizeof(float));
int x112 = 0;
while (x112 != 100) {
int x113 = x112;
float* x114 = x82+x113 * 1440;
float* x115 = x111+x113 * 16000;
int x116 = 0;
while (x116 != 250) {
int x117 = x116;
int x118 = x117 / 25;
int x119 = x117 % 25;
int x120 = x119 / 5;
int x121 = x119 % 5;
float* x122 = x115+x118 * 5 * 5 * 8 * 8 + x120 * 5 * 8 * 8 + x121 * 8 * 8;
float* x123 = x114+x118 * 12 * 12;
int x124 = 0;
while (x124 != 8) {
int x125 = x124;
memcpy(x122+x125 * 8, x123+(x125 + x120) * 12 + x121, 4 * 8);
x124 = x124 + 1;
}
x116 = x116 + 1;
}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,64,250,1,x6,250,x115,64,1,x105+x113 * 1280,64);
x112 = x112 + 1;
}
float* x126 = (float*)myMalloc(128000 * sizeof(float));
float* x127 = (float*)myMalloc(128000 * sizeof(float));
int x128 = 0;
while (x128 != 128000) {
int x129 = x128;
if (x105[x129] < 0.0) x127[x129] = 0.0; else x127[x129] = x105[x129];
x128 = x128 + 1;
}
float* x130 = (float*)myMalloc(128000 * sizeof(float));
float* x131 = (float*)myMalloc(32000 * sizeof(float));
int x132 = 0;
while (x132 != 32000) {
x131[x132] = -3.4028235E38;
x132 = x132 + 1;
}
int* x133 = (int*)myMalloc(32000 * sizeof(int));
int x134 = 0;
while (x134 != 100) {
int x135 = x134;
int x136 = x135 * 1280;
float* x137 = x127+x136;
int x138 = x135 * 320;
float* x139 = x131+x138;
int* x140 = x133+x138;
int x141 = 0;
int x142 = 0;
int x143 = 0;
while (x143 != 20) {
int x144 = x141;
int x145 = x142;
int x146 = 0;
while (x146 != 4) {
int x147 = x144;
int x148 = x145;
int x149 = 0;
while (x149 != 4) {
int x150 = x148;
int x151 = x150;
if (x137[x151] > x139[x147]) {
x139[x147] = x137[x151];
x140[x147] = x151 + x136;
} else {

}
x151 = x151 + 1;
if (x137[x151] > x139[x147]) {
x139[x147] = x137[x151];
x140[x147] = x151 + x136;
} else {

}
x151 = x151 + 1;
x150 = x150 + 8;
int x152 = x150;
if (x137[x152] > x139[x147]) {
x139[x147] = x137[x152];
x140[x147] = x152 + x136;
} else {

}
x152 = x152 + 1;
if (x137[x152] > x139[x147]) {
x139[x147] = x137[x152];
x140[x147] = x152 + x136;
} else {

}
x152 = x152 + 1;
x150 = x150 + 8;
x147 = x147 + 1;
x148 = x148 + 2;
x149 = x149 + 1;
}
x144 = x144 + 4;
x145 = x145 + 16;
x146 = x146 + 1;
}
x141 = x141 + 16;
x142 = x142 + 64;
x143 = x143 + 1;
}
x134 = x134 + 1;
}
float* x153 = (float*)myMalloc(32000 * sizeof(float));
int x154 = 0;
int x155 = 1;
x154 = x154 + 1;
x155 = x155 * 320;
if (x154 == 0) {

} else {

}
int x156 = 32000 / x155;
int x157 = x156 * 50;
float* x158 = (float*)myMalloc(x157 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x156,50,320,1,x131,320,x11,50,0,x158,50);
float* x159 = (float*)myMalloc(x157 * sizeof(float));
bool x160 = x156 == 1;
int x161 = x160 ? 0 : 50;
int x162 = 0;
while (x162 != x156) {
int x163 = 0;
int x164 = x161 * x162;
while (x163 != 50) {
int x165 = x163;
int x166 = x164 + x165;
x158[x166] = x158[x166] + x14[x165];
x163 = x163 + 1;
}
x162 = x162 + 1;
}
float* x167 = (float*)myMalloc(x157 * sizeof(float));
float* x168 = (float*)myMalloc(x157 * sizeof(float));
int x169 = 0;
while (x169 != x157) {
int x170 = x169;
if ((float)rand()/RAND_MAX > 0.5) {
x167[x170] = x158[x170] * 2.0;
x168[x170] = 2.0;
} else {
x167[x170] = 0.0;
x168[x170] = 0.0;
}
x169 = x169 + 1;
}
float* x171 = (float*)myMalloc(x157 * sizeof(float));
int x172 = x156 * 10;
float* x173 = (float*)myMalloc(x172 * sizeof(float));
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, x156,10,50,1,x167,50,x16,10,0,x173,10);
float* x174 = (float*)myMalloc(x172 * sizeof(float));
int x175 = x160 ? 0 : 10;
int x176 = 0;
while (x176 != x156) {
int x177 = 0;
int x178 = x175 * x176;
while (x177 != 10) {
int x179 = x177;
int x180 = x178 + x179;
x173[x180] = x173[x180] + x19[x179];
x177 = x177 + 1;
}
x176 = x176 + 1;
}
float* x181 = (float*)myMalloc(x156 * sizeof(float));
int x182 = 0;
int x183 = 0;
while (x183 != x156) {
float x184 = -3.4028235E38;
int x185 = 0;
while (x185 != 10) {
if (x173[x182] > x184) x184 = x173[x182]; else {

}
x182 = x182 + 1;
x185 = x185 + 1;
}
x181[x183] = x184;
x183 = x183 + 1;
}
float* x186 = (float*)myMalloc(x172 * sizeof(float));
int x187 = 0;
int x188 = 0;
while (x188 != x156) {
int x189 = x188;
int x190 = 0;
while (x190 != 10) {
x186[x187] = (float)exp((double)(x173[x187] - x181[x189]));
x187 = x187 + 1;
x190 = x190 + 1;
}
x188 = x188 + 1;
}
float* x191 = (float*)myMalloc(x156 * sizeof(float));
int x192 = 0;
while (x192 != x156) {
int x193 = x192;
int x194 = x193;
int x195 = x193 * 10;
int x196 = 0;
while (x196 != 10) {
int x197 = 0;
while (x197 != 1) {
int x198 = x197;
int x199 = x194 + x198;
x191[x199] = x191[x199] + x186[x195 + x198];
x197 = x197 + 1;
}
x195 = x195 + 1;
x196 = x196 + 1;
}
x192 = x192 + 1;
}
x187 = 0;
int x200 = 0;
while (x200 != x156) {
int x201 = x200;
float x202 = x181[x201] + (float)log((double)x191[x201]);
int x203 = 0;
while (x203 != 10) {
x186[x187] = x173[x187] - x202;
x187 = x187 + 1;
x203 = x203 + 1;
}
x200 = x200 + 1;
}
float* x204 = (float*)myMalloc(x172 * sizeof(float));
// nllLoss forward in CPU;
float* x205 = (float*)myMalloc(x156 * sizeof(float));
int x206 = 0;
int x207 = 0;
while (x207 != x156) {
int x208 = x207;
x205[x208] = -1.0 * x186[x206 + x54[x208]];
x206 = x206 + 10;
x207 = x207 + 1;
}
float* x209 = (float*)myMalloc(x156 * sizeof(float));
float x210 = 0.0;
int x211 = 0;
while (x211 != x156) {
x210 = x210 + x205[x211];
x211 = x211 + 1;
}
float x212 = x210;
float* x213 = (float*)myMalloc(1 * sizeof(float));
int x214 = 0;
while (x214 != 1) {
x213[x214] = x212;
x214 = x214 + 1;
}
float* x215 = (float*)myMalloc(1 * sizeof(float));
// make sure the size of loss is 1;
int x216 = 0;
while (x216 != 1) {
x215[x216] = x39;
x216 = x216 + 1;
}
// backend is lantern.TensorDslCPU$BackendCPU@2030464;
int x217 = 0;
while (x217 != 1) {
int x218 = x217;
x55[x218] = x213[x218];
x217 = x217 + 1;
}
// 'sum' gradient.;
int x219 = (x156 <= 1) ? 1 : x156;
int x220 = x160 ? 0 : 1;
int x221 = 0;
while (x221 != x219) {
int x222 = x220 * x221;
x209[x222] = x209[x222] + x215[0];
x221 = x221 + 1;
}
// 'nllLossB' gradient.;
// nllLoss_grad implementation in CPU;
int x223 = 0;
int x224 = 0;
while (x224 != x156) {
int x225 = x224;
int x226 = x223 + x54[x225];
x204[x226] = x204[x226] + -1.0 * x209[x225];
x223 = x223 + 10;
x224 = x224 + 1;
}
float* x227 = (float*)myMalloc(x156 * sizeof(float));
int x228 = 0;
while (x228 != x156) {
int x229 = x228;
int x230 = x229;
int x231 = x229 * 10;
int x232 = 0;
while (x232 != 10) {
int x233 = 0;
while (x233 != 1) {
int x234 = x233;
int x235 = x230 + x234;
x227[x235] = x227[x235] + x204[x231 + x234];
x233 = x233 + 1;
}
x231 = x231 + 1;
x232 = x232 + 1;
}
x228 = x228 + 1;
}
int x236 = 0;
int x237 = 0;
while (x237 != x156) {
int x238 = x237;
int x239 = 0;
while (x239 != 10) {
int x240 = x236;
x174[x240] = x174[x240] + (x204[x236] - (float)exp((double)x186[x236]) * x227[x238]);
x236 = x236 + 1;
x239 = x239 + 1;
}
x237 = x237 + 1;
}
int x241 = x160 ? 0 : 10;
int x242 = 0;
while (x242 != x156) {
int x243 = 0;
int x244 = x241 * x242;
while (x243 != 10) {
int x245 = x243;
x20[x245] = x20[x245] + x174[x244 + x245];
x243 = x243 + 1;
}
x242 = x242 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, x156,50,10,1,x174,10,x16,10,1,x171,50);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,10,x156,1,x167,50,x174,10,1,x18,10);
int x246 = (x156 <= x156) ? x156 : x156;
float* x247 = (float*)myMalloc(x246 * 50 * sizeof(float));
int x248 = x160 ? 0 : 50;
int x249 = x160 ? 0 : 50;
int x250 = 0;
while (x250 != x246) {
int x251 = x250;
int x252 = 0;
int x253 = 50 * x251;
int x254 = x248 * x251;
int x255 = x249 * x251;
while (x252 != 50) {
int x256 = x252;
x247[x256 + x253] = x168[x254 + x256] * x171[x255 + x256];
x252 = x252 + 1;
}
x250 = x250 + 1;
}
bool x257 = x246 == 1;
int x258 = (x156 <= x246) ? x246 : x156;
int x259 = x160 ? 0 : 50;
int x260 = x257 ? 0 : 50;
int x261 = 0;
while (x261 != x258) {
int x262 = x261;
int x263 = 0;
int x264 = x259 * x262;
int x265 = x260 * x262;
while (x263 != 50) {
int x266 = x263;
int x267 = x264 + x266;
x159[x267] = x159[x267] + x247[x265 + x266];
x263 = x263 + 1;
}
x261 = x261 + 1;
}
int x268 = x160 ? 0 : 50;
int x269 = 0;
while (x269 != x156) {
int x270 = 0;
int x271 = x268 * x269;
while (x270 != 50) {
int x272 = x270;
x15[x272] = x15[x272] + x159[x271 + x272];
x270 = x270 + 1;
}
x269 = x269 + 1;
}
// backprop of matrix-matrix-dot;
// backend add_dotTrans2;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, x156,320,50,1,x159,50,x11,50,1,x153,320);
// backend add_dotTrans1;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 320,50,x156,1,x131,320,x159,50,1,x13,50);
int x273 = 0;
while (x273 != 32000) {
int x274 = x273;
int x275 = x133[x274];
x130[x275] = x130[x275] + x153[x274];
x273 = x273 + 1;
}
int x276 = 0;
while (x276 != 128000) {
int x277 = x276;
x126[x277] = x126[x277] + ((x105[x277] < 0.0) ? 0.0 : x130[x277]);
x276 = x276 + 1;
}
// conv2D back-propagate;
float* x278 = (float*)myMalloc(1600000 * sizeof(float));
int x279 = 0;
while (x279 != 100) {
int x280 = x279;
float* x281 = x104+x280 * 1440;
float* x282 = x278+x280 * 16000;
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 250,64,20,1,x6,250,x126+x280 * 1280,64,0,x282,64);
int x283 = 0;
while (x283 != 10) {
int x284 = x283;
int x285 = 0;
int x286 = x284 * 5 * 5 * 8 * 8;
float* x287 = x281+x284 * 12 * 12;
while (x285 != 5) {
int x288 = x285;
int x289 = 0;
int x290 = x286 + x288 * 5 * 8 * 8;
while (x289 != 5) {
int x291 = x289;
float* x292 = x282+x290 + x291 * 8 * 8;
int x293 = 0;
while (x293 != 8) {
int x294 = x293;
float* x295 = x287+(x294 + x288) * 12 + x291;
float* x296 = x292+x294 * 8;
int x297 = 0;
while (x297 != 8) {
int x298 = x297;
x295[x298] = x295[x298] + x296[x298];
x297 = x297 + 1;
}
x293 = x293 + 1;
}
x289 = x289 + 1;
}
x285 = x285 + 1;
}
x283 = x283 + 1;
}
x279 = x279 + 1;
}
int x299 = 0;
while (x299 != 100) {
int x300 = x299;
float* x301 = x126+x300 * 1280;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,250,64,1.0,x301,64,x111+x300 * 16000,64,1,x8,250);
int x302 = 0;
while (x302 != 20) {
int x303 = x302;
float x304 = 0.0;
float* x305 = x301+x303 * 8 * 8;
int x306 = 0;
while (x306 != 64) {
x304 = x304 + x305[x306];
x306 = x306 + 1;
}
x10[x303] = x10[x303] + x304;
x302 = x302 + 1;
}
x299 = x299 + 1;
}
int x307 = 0;
while (x307 != 144000) {
int x308 = x307;
int x309 = x84[x308];
x81[x309] = x81[x309] + x104[x308];
x307 = x307 + 1;
}
int x310 = 0;
while (x310 != 576000) {
int x311 = x310;
x77[x311] = x77[x311] + ((x56[x311] < 0.0) ? 0.0 : x81[x311]);
x310 = x310 + 1;
}
// conv2D back-propagate;
(float*)myMalloc(1440000 * sizeof(float));
int x312 = 0;
while (x312 != 100) {
int x313 = x312;
float* x314 = x77+x313 * 5760;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 10,25,576,1.0,x314,576,x62+x313 * 14400,576,1,x3,25);
int x315 = 0;
while (x315 != 10) {
int x316 = x315;
float x317 = 0.0;
float* x318 = x314+x316 * 24 * 24;
int x319 = 0;
while (x319 != 576) {
x317 = x317 + x318[x319];
x319 = x319 + 1;
}
x5[x316] = x5[x316] + x317;
x315 = x315 + 1;
}
x312 = x312 + 1;
}
x49 = x49 + x55[0];
int x320 = 0;
while (x320 != 20) {
int x321 = x320;
x9[x321] = x9[x321] + x10[x321] * -5.0E-4;
x320 = x320 + 1;
}
int x322 = 0;
while (x322 != 20) {
x10[x322] = x40;
x322 = x322 + 1;
}
int x323 = 0;
while (x323 != 5000) {
int x324 = x323;
x6[x324] = x6[x324] + x8[x324] * -5.0E-4;
x323 = x323 + 1;
}
int x325 = 0;
while (x325 != 5000) {
x8[x325] = x40;
x325 = x325 + 1;
}
int x326 = 0;
while (x326 != 16000) {
int x327 = x326;
x11[x327] = x11[x327] + x13[x327] * -5.0E-4;
x326 = x326 + 1;
}
int x328 = 0;
while (x328 != 16000) {
x13[x328] = x40;
x328 = x328 + 1;
}
int x329 = 0;
while (x329 != 50) {
int x330 = x329;
x14[x330] = x14[x330] + x15[x330] * -5.0E-4;
x329 = x329 + 1;
}
int x331 = 0;
while (x331 != 50) {
x15[x331] = x40;
x331 = x331 + 1;
}
int x332 = 0;
while (x332 != 10) {
int x333 = x332;
x4[x333] = x4[x333] + x5[x333] * -5.0E-4;
x332 = x332 + 1;
}
int x334 = 0;
while (x334 != 10) {
x5[x334] = x40;
x334 = x334 + 1;
}
int x335 = 0;
while (x335 != 250) {
int x336 = x335;
x1[x336] = x1[x336] + x3[x336] * -5.0E-4;
x335 = x335 + 1;
}
int x337 = 0;
while (x337 != 250) {
x3[x337] = x40;
x337 = x337 + 1;
}
int x338 = 0;
while (x338 != 10) {
int x339 = x338;
x19[x339] = x19[x339] + x20[x339] * -5.0E-4;
x338 = x338 + 1;
}
int x340 = 0;
while (x340 != 10) {
x20[x340] = x40;
x340 = x340 + 1;
}
int x341 = 0;
while (x341 != 500) {
int x342 = x341;
x16[x342] = x16[x342] + x18[x342] * -5.0E-4;
x341 = x341 + 1;
}
int x343 = 0;
while (x343 != 500) {
x18[x343] = x40;
x343 = x343 + 1;
}
if (x47 % x41 == 0) {
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n", x45, x47, x27, 100.0 * (double)x47 / x42, x49 / (float)x47);
fflush(stdout);
} else {

}
memset((void*)x36, 0, (long)mallocAddr - x36);
mallocAddr = (void*)x36;
x51 = x51 + 78400;
x52 = x52 + 1;
}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
long x344 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
printf("Training completed in %ldms (%ld us/images)\n", x344 / 1000L, x344 / x43);
x35[x45] = (double)(x49 / x44);
x37 = x37 + 1;
}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
long x345 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
long x346 = (long)fopen(x0, "w");
fprintf((FILE *)x346, "unit: %s\n", "1 epoch");
int x347 = 0;
while (x347 != 4) {
fprintf((FILE *)x346, "%lf\n", x35[x347]);
x347 = x347 + 1;
}
fprintf((FILE *)x346, "run time: %lf %lf\n", x34, ((float)x345 / 1000000.0 - x34) / 4.0);
fclose((FILE*)x346);
// Backend cleanup.;
}

    /*****************************************
    End of C Generated Code
    *******************************************/
    


