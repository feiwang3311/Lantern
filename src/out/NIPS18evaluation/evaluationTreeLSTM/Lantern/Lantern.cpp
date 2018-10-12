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
// Backend setup.
double x2 = ((double)clock() / CLOCKS_PER_SEC);
int* x3 = (int32_t*)myMalloc(1 * sizeof(int32_t));;
int64_t x4 = (long)fopen("small_glove.txt", "r");
if (fscanf((FILE *)x4,"%d", &x3[0])!=1) perror("Error reading file");
int32_t x6 = x3[0];
float** x7 = (float**)myMalloc(x6 * sizeof(float*));;
for(int x9=0; x9 < x6; x9++) {
float* x10 = (float*)myMalloc(300 * sizeof(float));;
x7[x9] = x10;
for(int x13=0; x13 < 300; x13++) {
float* x14 = x7[x9];
if (fscanf((FILE *)x4,"%f", &x14[x13])!=1) perror("Error reading file");

}

}
fclose((FILE*)x4);
int* x21 = (int32_t*)myMalloc(1 * sizeof(int32_t));;
int64_t x22 = (long)fopen("array_tree.txt", "r");
if (fscanf((FILE *)x22,"%d", &x21[0])!=1) perror("Error reading file");
int32_t x24 = x21[0];
int32_t x25 = x24 * 4;
int** x26 = (int**)myMalloc(x25 * sizeof(int*));;
int* x27 = (int32_t*)myMalloc(1 * sizeof(int32_t));;
for(int x29=0; x29 < x24; x29++) {
if (fscanf((FILE *)x22,"%d", &x27[0])!=1) perror("Error reading file");
int32_t x33 = x29 * 4;
for(int x32=0; x32 < 4; x32++) {
int32_t x35 = x27[0];
int* x36 = (int32_t*)myMalloc(x35 * sizeof(int32_t));;
int32_t x34 = x33 + x32;
x26[x34] = x36;
int32_t x38 = x27[0];
for(int x40=0; x40 < x38; x40++) {
int* x41 = x26[x34];
if (fscanf((FILE *)x22,"%d", &x41[x40])!=1) perror("Error reading file");

}

}

}
fclose((FILE*)x22);
float* x50 = (float*)myMalloc(45000 * sizeof(float));;
for(int x52=0; x52 < 45000; x52++) {
float x53 = (float)rand()/RAND_MAX;
float x54 = x53 - 0.5f;
float x55 = x54 * 0.01f;
x50[x52] = x55;

}
float* x59 = (float*)myMalloc(150 * sizeof(float));;
for(int x61=0; x61 < 150; x61++) {
x59[x61] = 0.0f;

}
float* x65 = (float*)myMalloc(45000 * sizeof(float));;
for(int x66=0; x66 < 45000; x66++) {
float x67 = (float)rand()/RAND_MAX;
float x68 = x67 - 0.5f;
float x69 = x68 * 0.01f;
x65[x66] = x69;

}
float* x73 = (float*)myMalloc(150 * sizeof(float));;
for(int x74=0; x74 < 150; x74++) {
x73[x74] = 0.0f;

}
float* x78 = (float*)myMalloc(45000 * sizeof(float));;
for(int x79=0; x79 < 45000; x79++) {
float x80 = (float)rand()/RAND_MAX;
float x81 = x80 - 0.5f;
float x82 = x81 * 0.01f;
x78[x79] = x82;

}
float* x86 = (float*)myMalloc(150 * sizeof(float));;
for(int x87=0; x87 < 150; x87++) {
x86[x87] = 0.0f;

}
float* x91 = (float*)myMalloc(22500 * sizeof(float));;
for(int x93=0; x93 < 22500; x93++) {
float x94 = (float)rand()/RAND_MAX;
float x95 = x94 - 0.5f;
float x96 = x95 * 0.01f;
x91[x93] = x96;

}
float* x100 = (float*)myMalloc(22500 * sizeof(float));;
for(int x101=0; x101 < 22500; x101++) {
float x102 = (float)rand()/RAND_MAX;
float x103 = x102 - 0.5f;
float x104 = x103 * 0.01f;
x100[x101] = x104;

}
float* x108 = (float*)myMalloc(150 * sizeof(float));;
for(int x109=0; x109 < 150; x109++) {
x108[x109] = 0.0f;

}
float* x113 = (float*)myMalloc(22500 * sizeof(float));;
for(int x114=0; x114 < 22500; x114++) {
float x115 = (float)rand()/RAND_MAX;
float x116 = x115 - 0.5f;
float x117 = x116 * 0.01f;
x113[x114] = x117;

}
float* x121 = (float*)myMalloc(22500 * sizeof(float));;
for(int x122=0; x122 < 22500; x122++) {
float x123 = (float)rand()/RAND_MAX;
float x124 = x123 - 0.5f;
float x125 = x124 * 0.01f;
x121[x122] = x125;

}
float* x129 = (float*)myMalloc(22500 * sizeof(float));;
for(int x130=0; x130 < 22500; x130++) {
float x131 = (float)rand()/RAND_MAX;
float x132 = x131 - 0.5f;
float x133 = x132 * 0.01f;
x129[x130] = x133;

}
float* x137 = (float*)myMalloc(22500 * sizeof(float));;
for(int x138=0; x138 < 22500; x138++) {
float x139 = (float)rand()/RAND_MAX;
float x140 = x139 - 0.5f;
float x141 = x140 * 0.01f;
x137[x138] = x141;

}
float* x145 = (float*)myMalloc(150 * sizeof(float));;
for(int x146=0; x146 < 150; x146++) {
x145[x146] = 0.0f;

}
float* x150 = (float*)myMalloc(22500 * sizeof(float));;
for(int x151=0; x151 < 22500; x151++) {
float x152 = (float)rand()/RAND_MAX;
float x153 = x152 - 0.5f;
float x154 = x153 * 0.01f;
x150[x151] = x154;

}
float* x158 = (float*)myMalloc(22500 * sizeof(float));;
for(int x159=0; x159 < 22500; x159++) {
float x160 = (float)rand()/RAND_MAX;
float x161 = x160 - 0.5f;
float x162 = x161 * 0.01f;
x158[x159] = x162;

}
float* x166 = (float*)myMalloc(150 * sizeof(float));;
for(int x167=0; x167 < 150; x167++) {
x166[x167] = 0.0f;

}
float* x171 = (float*)myMalloc(22500 * sizeof(float));;
for(int x172=0; x172 < 22500; x172++) {
float x173 = (float)rand()/RAND_MAX;
float x174 = x173 - 0.5f;
float x175 = x174 * 0.01f;
x171[x172] = x175;

}
float* x179 = (float*)myMalloc(22500 * sizeof(float));;
for(int x180=0; x180 < 22500; x180++) {
float x181 = (float)rand()/RAND_MAX;
float x182 = x181 - 0.5f;
float x183 = x182 * 0.01f;
x179[x180] = x183;

}
float* x187 = (float*)myMalloc(150 * sizeof(float));;
for(int x188=0; x188 < 150; x188++) {
x187[x188] = 0.0f;

}
float* x192 = (float*)myMalloc(750 * sizeof(float));;
for(int x194=0; x194 < 750; x194++) {
float x195 = (float)rand()/RAND_MAX;
float x196 = x195 - 0.5f;
float x197 = x196 * 0.01f;
x192[x194] = x197;

}
float* x201 = (float*)myMalloc(5 * sizeof(float));;
for(int x203=0; x203 < 5; x203++) {
x201[x203] = 0.0f;

}
float* x207 = (float*)myMalloc(45000 * sizeof(float));;
for(int x208=0; x208 < 45000; x208++) {
x207[x208] = 0.0f;

}
float* x212 = (float*)myMalloc(150 * sizeof(float));;
for(int x213=0; x213 < 150; x213++) {
x212[x213] = 0.0f;

}
float* x217 = (float*)myMalloc(45000 * sizeof(float));;
for(int x218=0; x218 < 45000; x218++) {
x217[x218] = 0.0f;

}
float* x222 = (float*)myMalloc(150 * sizeof(float));;
for(int x223=0; x223 < 150; x223++) {
x222[x223] = 0.0f;

}
float* x227 = (float*)myMalloc(45000 * sizeof(float));;
for(int x228=0; x228 < 45000; x228++) {
x227[x228] = 0.0f;

}
float* x232 = (float*)myMalloc(150 * sizeof(float));;
for(int x233=0; x233 < 150; x233++) {
x232[x233] = 0.0f;

}
float* x237 = (float*)myMalloc(22500 * sizeof(float));;
for(int x238=0; x238 < 22500; x238++) {
x237[x238] = 0.0f;

}
float* x242 = (float*)myMalloc(22500 * sizeof(float));;
for(int x243=0; x243 < 22500; x243++) {
x242[x243] = 0.0f;

}
float* x247 = (float*)myMalloc(150 * sizeof(float));;
for(int x248=0; x248 < 150; x248++) {
x247[x248] = 0.0f;

}
float* x252 = (float*)myMalloc(22500 * sizeof(float));;
for(int x253=0; x253 < 22500; x253++) {
x252[x253] = 0.0f;

}
float* x257 = (float*)myMalloc(22500 * sizeof(float));;
for(int x258=0; x258 < 22500; x258++) {
x257[x258] = 0.0f;

}
float* x262 = (float*)myMalloc(22500 * sizeof(float));;
for(int x263=0; x263 < 22500; x263++) {
x262[x263] = 0.0f;

}
float* x267 = (float*)myMalloc(22500 * sizeof(float));;
for(int x268=0; x268 < 22500; x268++) {
x267[x268] = 0.0f;

}
float* x272 = (float*)myMalloc(150 * sizeof(float));;
for(int x273=0; x273 < 150; x273++) {
x272[x273] = 0.0f;

}
float* x277 = (float*)myMalloc(22500 * sizeof(float));;
for(int x278=0; x278 < 22500; x278++) {
x277[x278] = 0.0f;

}
float* x282 = (float*)myMalloc(22500 * sizeof(float));;
for(int x283=0; x283 < 22500; x283++) {
x282[x283] = 0.0f;

}
float* x287 = (float*)myMalloc(150 * sizeof(float));;
for(int x288=0; x288 < 150; x288++) {
x287[x288] = 0.0f;

}
float* x292 = (float*)myMalloc(22500 * sizeof(float));;
for(int x293=0; x293 < 22500; x293++) {
x292[x293] = 0.0f;

}
float* x297 = (float*)myMalloc(22500 * sizeof(float));;
for(int x298=0; x298 < 22500; x298++) {
x297[x298] = 0.0f;

}
float* x302 = (float*)myMalloc(150 * sizeof(float));;
for(int x303=0; x303 < 150; x303++) {
x302[x303] = 0.0f;

}
float* x307 = (float*)myMalloc(750 * sizeof(float));;
for(int x308=0; x308 < 750; x308++) {
x307[x308] = 0.0f;

}
float* x312 = (float*)myMalloc(5 * sizeof(float));;
for(int x313=0; x313 < 5; x313++) {
x312[x313] = 0.0f;

}
float* x317 = (float*)myMalloc(300 * sizeof(float));;
for(int x318=0; x318 < 300; x318++) {
x317[x318] = 0.0f;

}
float* x322 = (float*)myMalloc(300 * sizeof(float));;
for(int x323=0; x323 < 300; x323++) {
x322[x323] = 0.0f;

}
float* x327 = (float*)myMalloc(150 * sizeof(float));;
for(int x328=0; x328 < 150; x328++) {
x327[x328] = 0.0f;

}
float* x332 = (float*)myMalloc(150 * sizeof(float));;
for(int x333=0; x333 < 150; x333++) {
x332[x333] = 0.0f;

}
float* x337 = (float*)myMalloc(45000 * sizeof(float));;
for(int x338=0; x338 < 45000; x338++) {
x337[x338] = 0.0f;

}
float* x342 = (float*)myMalloc(150 * sizeof(float));;
for(int x343=0; x343 < 150; x343++) {
x342[x343] = 0.0f;

}
float* x347 = (float*)myMalloc(45000 * sizeof(float));;
for(int x348=0; x348 < 45000; x348++) {
x347[x348] = 0.0f;

}
float* x352 = (float*)myMalloc(150 * sizeof(float));;
for(int x353=0; x353 < 150; x353++) {
x352[x353] = 0.0f;

}
float* x357 = (float*)myMalloc(45000 * sizeof(float));;
for(int x358=0; x358 < 45000; x358++) {
x357[x358] = 0.0f;

}
float* x362 = (float*)myMalloc(150 * sizeof(float));;
for(int x363=0; x363 < 150; x363++) {
x362[x363] = 0.0f;

}
float* x367 = (float*)myMalloc(22500 * sizeof(float));;
for(int x368=0; x368 < 22500; x368++) {
x367[x368] = 0.0f;

}
float* x372 = (float*)myMalloc(22500 * sizeof(float));;
for(int x373=0; x373 < 22500; x373++) {
x372[x373] = 0.0f;

}
float* x377 = (float*)myMalloc(150 * sizeof(float));;
for(int x378=0; x378 < 150; x378++) {
x377[x378] = 0.0f;

}
float* x382 = (float*)myMalloc(22500 * sizeof(float));;
for(int x383=0; x383 < 22500; x383++) {
x382[x383] = 0.0f;

}
float* x387 = (float*)myMalloc(22500 * sizeof(float));;
for(int x388=0; x388 < 22500; x388++) {
x387[x388] = 0.0f;

}
float* x392 = (float*)myMalloc(22500 * sizeof(float));;
for(int x393=0; x393 < 22500; x393++) {
x392[x393] = 0.0f;

}
float* x397 = (float*)myMalloc(22500 * sizeof(float));;
for(int x398=0; x398 < 22500; x398++) {
x397[x398] = 0.0f;

}
float* x402 = (float*)myMalloc(150 * sizeof(float));;
for(int x403=0; x403 < 150; x403++) {
x402[x403] = 0.0f;

}
float* x407 = (float*)myMalloc(22500 * sizeof(float));;
for(int x408=0; x408 < 22500; x408++) {
x407[x408] = 0.0f;

}
float* x412 = (float*)myMalloc(22500 * sizeof(float));;
for(int x413=0; x413 < 22500; x413++) {
x412[x413] = 0.0f;

}
float* x417 = (float*)myMalloc(150 * sizeof(float));;
for(int x418=0; x418 < 150; x418++) {
x417[x418] = 0.0f;

}
float* x422 = (float*)myMalloc(22500 * sizeof(float));;
for(int x423=0; x423 < 22500; x423++) {
x422[x423] = 0.0f;

}
float* x427 = (float*)myMalloc(22500 * sizeof(float));;
for(int x428=0; x428 < 22500; x428++) {
x427[x428] = 0.0f;

}
float* x432 = (float*)myMalloc(150 * sizeof(float));;
for(int x433=0; x433 < 150; x433++) {
x432[x433] = 0.0f;

}
float* x437 = (float*)myMalloc(750 * sizeof(float));;
for(int x438=0; x438 < 750; x438++) {
x437[x438] = 0.0f;

}
float* x442 = (float*)myMalloc(5 * sizeof(float));;
for(int x443=0; x443 < 5; x443++) {
x442[x443] = 0.0f;

}
double* x447 = (double*)myMalloc(30 * sizeof(double));;
int64_t x448 = (long)mallocAddr;
double x449 = ((double)clock() / CLOCKS_PER_SEC);
for(int x451=0; x451 < 30; x451++) {
float x452 = 0.0f;
for(int x453=0; x453 < x24; x453++) {
float* x479 = (float*)myMalloc(1 * sizeof(float));;
float* x484 = (float*)myMalloc(1 * sizeof(float));;
float* x489 = (float*)myMalloc(150 * sizeof(float));;
float* x494 = (float*)myMalloc(150 * sizeof(float));;
float* x499 = (float*)myMalloc(150 * sizeof(float));;
float* x504 = (float*)myMalloc(150 * sizeof(float));;
int32_t x454 = x453 % x24;
int32_t x455 = x454 * 4;
int* x456 = x26[x455];
int32_t x457 = x455 + 1;
int* x458 = x26[x457];
int32_t x459 = x455 + 2;
int* x460 = x26[x459];
int32_t x461 = x455 + 3;
int* x462 = x26[x461];
function<void(int32_t,function<void(float**)>,float**)> x509 = [&](int32_t x510,function<void(float**)> x511,float** x512) {
float** x515 = x512;
float* x516 = x515[0];
float* x517 = x515[1];
float* x518 = x515[2];
float* x519 = x515[3];
float* x520 = x515[4];
float* x521 = x515[5];
int32_t x513 = x510;
bool x522 = x513 >= 0;
if (x522) {
int32_t x523 = x460[x513];
float** x3235 = (float**)myMalloc(6 * sizeof(float*));;
x3235[0] = x479;
x3235[1] = x484;
x3235[2] = x489;
x3235[3] = x494;
x3235[4] = x499;
x3235[5] = x504;
function<void(float**)> x514 = x511;
function<void(float**)> x965 = [&](float** x966) {
float* x967 = x966[0];
float* x968 = x966[1];
float* x969 = x966[2];
float* x970 = x966[3];
float* x971 = x966[4];
float* x972 = x966[5];
float** x973 = (float**)myMalloc(6 * sizeof(float*));;
x973[0] = x967;
x973[1] = x968;
x973[2] = x969;
x973[3] = x970;
x973[4] = x971;
x973[5] = x972;
x514(x973);
};
function<void(float**)> x957 = [&](float** x958) {
float* x959 = x958[0];
float* x960 = x958[1];
float* x961 = x958[2];
float* x962 = x958[3];
float* x963 = x958[4];
float* x964 = x958[5];
float** x982 = (float**)myMalloc(6 * sizeof(float*));;
x982[0] = x959;
x982[1] = x960;
x982[2] = x961;
x982[3] = x962;
x982[4] = x963;
x982[5] = x964;
x965(x982);
};
function<void(float**)> x524 = [&](float** x525) {
float* x526 = x525[0];
float* x527 = x525[1];
float* x528 = x525[2];
float* x529 = x525[3];
float* x530 = x525[4];
float* x531 = x525[5];
int32_t x532 = x462[x513];
float** x3225 = (float**)myMalloc(6 * sizeof(float*));;
x3225[0] = x479;
x3225[1] = x484;
x3225[2] = x489;
x3225[3] = x494;
x3225[4] = x499;
x3225[5] = x504;
function<void(float**)> x533 = [&](float** x534) {
float* x535 = x534[0];
float* x536 = x534[1];
float* x537 = x534[2];
float* x538 = x534[3];
float* x539 = x534[4];
float* x540 = x534[5];
float* x541 = (float*)myMalloc(5 * sizeof(float));;
for(int x542=0; x542 < 5; x542++) {
x541[x542] = 0.0f;

}
int32_t x546 = x456[x513];
x541[x546] = 1.0f;
float* x548 = (float*)myMalloc(5 * sizeof(float));;
for(int x549=0; x549 < 5; x549++) {
x548[x549] = 0.0f;

}
int32_t x553 = x460[x513];
bool x554 = x553 < 0;
if (x554) {
int32_t x555 = x458[x513];
float* x556 = x7[x555];
float* x557 = (float*)myMalloc(300 * sizeof(float));;
for(int x558=0; x558 < 300; x558++) {
x557[x558] = 0.0f;

}
// dot: List(150, 300), WrappedArray(300)
float* x563 = (float*)myMalloc(150 * sizeof(float));;
for(int x564=0; x564 < 150; x564++) {
float x565 = 0.0f;
int32_t x567 = x564 * 300;
for(int x566=0; x566 < 300; x566++) {
int32_t x568 = x567 + x566;
float x569 = x50[x568];
float x570 = x556[x566];
float x571 = x569 * x570;
x565 += x571;

}
float x575 = x565;
x563[x564] = x575;

}
float* x579 = (float*)myMalloc(150 * sizeof(float));;
for(int x580=0; x580 < 150; x580++) {
x579[x580] = 0.0f;

}
float* x584 = (float*)myMalloc(150 * sizeof(float));;
int32_t x585 = 0;
int32_t x586 = 0;
int32_t x587 = 0;
for(int x588=0; x588 < 150; x588++) {
int32_t x589 = x585;
int32_t x590 = x586;
float x591 = x563[x590];
int32_t x592 = x587;
float x593 = x59[x592];
float x594 = x591 + x593;
x584[x589] = x594;
x585 += 1;
x586 += 1;
x587 += 1;

}
float* x601 = (float*)myMalloc(150 * sizeof(float));;
for(int x602=0; x602 < 150; x602++) {
x601[x602] = 0.0f;

}
float* x606 = (float*)myMalloc(150 * sizeof(float));;
for(int x607=0; x607 < 150; x607++) {
float x608 = x584[x607];
float x609 = -1.0f * x608;
double x610 = (double)x609;
double x611 = exp(x610);
float x612 = (float)x611;
float x613 = x612 + 1.0f;
float x614 = 1.0f / x613;
x606[x607] = x614;

}
float* x618 = (float*)myMalloc(150 * sizeof(float));;
for(int x619=0; x619 < 150; x619++) {
x618[x619] = 0.0f;

}
// dot: List(150, 300), WrappedArray(300)
float* x624 = (float*)myMalloc(150 * sizeof(float));;
for(int x625=0; x625 < 150; x625++) {
float x626 = 0.0f;
int32_t x628 = x625 * 300;
for(int x627=0; x627 < 300; x627++) {
int32_t x629 = x628 + x627;
float x630 = x65[x629];
float x631 = x556[x627];
float x632 = x630 * x631;
x626 += x632;

}
float x636 = x626;
x624[x625] = x636;

}
float* x640 = (float*)myMalloc(150 * sizeof(float));;
for(int x641=0; x641 < 150; x641++) {
x640[x641] = 0.0f;

}
float* x645 = (float*)myMalloc(150 * sizeof(float));;
int32_t x646 = 0;
int32_t x647 = 0;
int32_t x648 = 0;
for(int x649=0; x649 < 150; x649++) {
int32_t x650 = x646;
int32_t x651 = x647;
float x652 = x624[x651];
int32_t x653 = x648;
float x654 = x73[x653];
float x655 = x652 + x654;
x645[x650] = x655;
x646 += 1;
x647 += 1;
x648 += 1;

}
float* x662 = (float*)myMalloc(150 * sizeof(float));;
for(int x663=0; x663 < 150; x663++) {
x662[x663] = 0.0f;

}
float* x667 = (float*)myMalloc(150 * sizeof(float));;
for(int x668=0; x668 < 150; x668++) {
float x669 = x645[x668];
float x670 = -1.0f * x669;
double x671 = (double)x670;
double x672 = exp(x671);
float x673 = (float)x672;
float x674 = x673 + 1.0f;
float x675 = 1.0f / x674;
x667[x668] = x675;

}
float* x679 = (float*)myMalloc(150 * sizeof(float));;
for(int x680=0; x680 < 150; x680++) {
x679[x680] = 0.0f;

}
// dot: List(150, 300), WrappedArray(300)
float* x685 = (float*)myMalloc(150 * sizeof(float));;
for(int x686=0; x686 < 150; x686++) {
float x687 = 0.0f;
int32_t x689 = x686 * 300;
for(int x688=0; x688 < 300; x688++) {
int32_t x690 = x689 + x688;
float x691 = x78[x690];
float x692 = x556[x688];
float x693 = x691 * x692;
x687 += x693;

}
float x697 = x687;
x685[x686] = x697;

}
float* x701 = (float*)myMalloc(150 * sizeof(float));;
for(int x702=0; x702 < 150; x702++) {
x701[x702] = 0.0f;

}
float* x706 = (float*)myMalloc(150 * sizeof(float));;
int32_t x707 = 0;
int32_t x708 = 0;
int32_t x709 = 0;
for(int x710=0; x710 < 150; x710++) {
int32_t x711 = x707;
int32_t x712 = x708;
float x713 = x685[x712];
int32_t x714 = x709;
float x715 = x86[x714];
float x716 = x713 + x715;
x706[x711] = x716;
x707 += 1;
x708 += 1;
x709 += 1;

}
float* x723 = (float*)myMalloc(150 * sizeof(float));;
for(int x724=0; x724 < 150; x724++) {
x723[x724] = 0.0f;

}
float* x728 = (float*)myMalloc(150 * sizeof(float));;
for(int x729=0; x729 < 150; x729++) {
float x730 = x706[x729];
double x731 = (double)x730;
double x732 = tanh(x731);
float x733 = (float)x732;
x728[x729] = x733;

}
float* x737 = (float*)myMalloc(150 * sizeof(float));;
for(int x738=0; x738 < 150; x738++) {
x737[x738] = 0.0f;

}
float* x742 = (float*)myMalloc(150 * sizeof(float));;
int32_t x743 = 0;
int32_t x744 = 0;
int32_t x745 = 0;
for(int x746=0; x746 < 150; x746++) {
int32_t x747 = x743;
int32_t x748 = x744;
float x749 = x606[x748];
int32_t x750 = x745;
float x751 = x728[x750];
float x752 = x749 * x751;
x742[x747] = x752;
x743 += 1;
x744 += 1;
x745 += 1;

}
float* x759 = (float*)myMalloc(150 * sizeof(float));;
for(int x760=0; x760 < 150; x760++) {
x759[x760] = 0.0f;

}
float* x764 = (float*)myMalloc(150 * sizeof(float));;
for(int x765=0; x765 < 150; x765++) {
float x766 = x742[x765];
double x767 = (double)x766;
double x768 = tanh(x767);
float x769 = (float)x768;
x764[x765] = x769;

}
float* x773 = (float*)myMalloc(150 * sizeof(float));;
for(int x774=0; x774 < 150; x774++) {
x773[x774] = 0.0f;

}
float* x778 = (float*)myMalloc(150 * sizeof(float));;
int32_t x779 = 0;
int32_t x780 = 0;
int32_t x781 = 0;
for(int x782=0; x782 < 150; x782++) {
int32_t x783 = x779;
int32_t x784 = x780;
float x785 = x667[x784];
int32_t x786 = x781;
float x787 = x764[x786];
float x788 = x785 * x787;
x778[x783] = x788;
x779 += 1;
x780 += 1;
x781 += 1;

}
float* x795 = (float*)myMalloc(150 * sizeof(float));;
for(int x796=0; x796 < 150; x796++) {
x795[x796] = 0.0f;

}
// dot: List(5, 150), List(150)
float* x801 = (float*)myMalloc(5 * sizeof(float));;
for(int x802=0; x802 < 5; x802++) {
float x803 = 0.0f;
int32_t x805 = x802 * 150;
for(int x804=0; x804 < 150; x804++) {
int32_t x806 = x805 + x804;
float x807 = x192[x806];
float x808 = x778[x804];
float x809 = x807 * x808;
x803 += x809;

}
float x813 = x803;
x801[x802] = x813;

}
float* x817 = (float*)myMalloc(5 * sizeof(float));;
for(int x818=0; x818 < 5; x818++) {
x817[x818] = 0.0f;

}
float* x822 = (float*)myMalloc(5 * sizeof(float));;
int32_t x823 = 0;
int32_t x824 = 0;
int32_t x825 = 0;
for(int x826=0; x826 < 5; x826++) {
int32_t x827 = x823;
int32_t x828 = x824;
float x829 = x801[x828];
int32_t x830 = x825;
float x831 = x201[x830];
float x832 = x829 + x831;
x822[x827] = x832;
x823 += 1;
x824 += 1;
x825 += 1;

}
float* x839 = (float*)myMalloc(5 * sizeof(float));;
for(int x840=0; x840 < 5; x840++) {
x839[x840] = 0.0f;

}
float* x844 = (float*)myMalloc(5 * sizeof(float));;
for(int x845=0; x845 < 5; x845++) {
float x846 = x822[x845];
double x847 = (double)x846;
double x848 = exp(x847);
float x849 = (float)x848;
x844[x845] = x849;

}
float* x853 = (float*)myMalloc(5 * sizeof(float));;
for(int x854=0; x854 < 5; x854++) {
x853[x854] = 0.0f;

}
float x858 = 0.0f;
for(int x859=0; x859 < 5; x859++) {
float x860 = x858;
float x861 = x844[x859];
float x862 = x860 + x861;
x858 = x862;

}
float x866 = x858;
float* x867 = (float*)myMalloc(1 * sizeof(float));;
x867[0] = x866;
float* x869 = (float*)myMalloc(1 * sizeof(float));;
for(int x870=0; x870 < 1; x870++) {
x869[x870] = 0.0f;

}
float* x874 = (float*)myMalloc(5 * sizeof(float));;
int32_t x875 = 0;
int32_t x876 = 0;
int32_t x877 = 0;
for(int x878=0; x878 < 5; x878++) {
int32_t x879 = x875;
int32_t x880 = x876;
float x881 = x844[x880];
int32_t x882 = x877;
float x883 = x867[x882];
float x884 = x881 / x883;
x874[x879] = x884;
x875 += 1;
x876 += 1;

}
float* x890 = (float*)myMalloc(5 * sizeof(float));;
for(int x891=0; x891 < 5; x891++) {
x890[x891] = 0.0f;

}
float* x895 = (float*)myMalloc(1 * sizeof(float));;
int32_t x896 = 0;
int32_t x897 = 0;
int32_t x898 = 0;
int32_t x899 = x896;
int32_t x900 = x897;
float x901 = x526[x900];
int32_t x902 = x898;
float x903 = x535[x902];
float x904 = x901 + x903;
x895[x899] = x904;
x896 += 1;
float* x907 = (float*)myMalloc(1 * sizeof(float));;
for(int x908=0; x908 < 1; x908++) {
x907[x908] = 0.0f;

}
// dot: List(5), WrappedArray(5)
float x913 = 0.0f;
for(int x914=0; x914 < 5; x914++) {
float x915 = x874[x914];
float x916 = x541[x914];
float x917 = x915 * x916;
x913 += x917;

}
float* x921 = (float*)myMalloc(1 * sizeof(float));;
float x922 = x913;
x921[0] = x922;
float* x924 = (float*)myMalloc(1 * sizeof(float));;
for(int x925=0; x925 < 1; x925++) {
x924[x925] = 0.0f;

}
float* x929 = (float*)myMalloc(1 * sizeof(float));;
float x930 = x921[0];
double x931 = (double)x930;
double x932 = log(x931);
float x933 = (float)x932;
x929[0] = x933;
float* x935 = (float*)myMalloc(1 * sizeof(float));;
for(int x936=0; x936 < 1; x936++) {
x935[x936] = 0.0f;

}
float* x940 = (float*)myMalloc(1 * sizeof(float));;
int32_t x941 = 0;
int32_t x942 = 0;
int32_t x943 = 0;
int32_t x944 = x941;
int32_t x945 = x942;
float x946 = x895[x945];
int32_t x947 = x943;
float x948 = x929[x947];
float x949 = x946 - x948;
x940[x944] = x949;
x941 += 1;
float* x952 = (float*)myMalloc(1 * sizeof(float));;
for(int x953=0; x953 < 1; x953++) {
x952[x953] = 0.0f;

}
float** x991 = (float**)myMalloc(6 * sizeof(float*));;
x991[0] = x940;
x991[1] = x952;
x991[2] = x778;
x991[3] = x795;
x991[4] = x742;
x991[5] = x759;
x957(x991);
int32_t x999 = 0;
int32_t x1000 = 0;
int32_t x1001 = 0;
int32_t x1002 = x999;
float x1003 = x907[x1002];
float x1004 = x895[x1002];
int32_t x1005 = x1000;
float x1006 = x929[x1005];
int32_t x1007 = x1001;
float x1008 = x952[x1007];
float x1009 = x1003 + x1008;
x907[x1002] = x1009;
float x1011 = x935[x1005];
float x1012 = x895[x1002];
float x1013 = x929[x1005];
float x1014 = x952[x1007];
float x1015 = -1.0f * x1014;
float x1016 = x1011 + x1015;
x935[x1005] = x1016;
x1001 += 1;
float x1019 = x924[0];
float x1020 = x935[0];
float x1021 = x921[0];
float x1022 = x1020 / x1021;
float x1023 = x1019 + x1022;
x924[0] = x1023;
float x1025 = x924[0];
// Generate code for addMul
for(int x1027=0; x1027 < 5; x1027++) {
float x1028 = x890[x1027];
float x1029 = x541[x1027];
float x1030 = x1025 * x1029;
float x1031 = x1028 + x1030;
x890[x1027] = x1031;

}
float x1035 = x924[0];
// Generate code for addMul
for(int x1037=0; x1037 < 5; x1037++) {
float x1038 = x548[x1037];
float x1039 = x874[x1037];
float x1040 = x1035 * x1039;
float x1041 = x1038 + x1040;
x548[x1037] = x1041;

}
int32_t x1045 = 0;
int32_t x1046 = 0;
int32_t x1047 = 0;
int32_t x1048 = x1045;
float x1049 = x527[x1048];
float x1050 = x526[x1048];
int32_t x1051 = x1046;
float x1052 = x535[x1051];
int32_t x1053 = x1047;
float x1054 = x907[x1053];
float x1055 = x1049 + x1054;
x527[x1048] = x1055;
float x1057 = x536[x1051];
float x1058 = x526[x1048];
float x1059 = x535[x1051];
float x1060 = x907[x1053];
float x1061 = x1057 + x1060;
x536[x1051] = x1061;
x1047 += 1;
int32_t x1064 = 0;
int32_t x1065 = 0;
int32_t x1066 = 0;
for(int x1067=0; x1067 < 5; x1067++) {
int32_t x1068 = x1064;
float x1069 = x853[x1068];
float x1070 = x844[x1068];
int32_t x1071 = x1065;
float x1072 = x867[x1071];
int32_t x1073 = x1066;
float x1074 = x890[x1073];
float x1075 = x1074 / x1072;
float x1076 = x1069 + x1075;
x853[x1068] = x1076;
float x1078 = x869[x1071];
float x1079 = x844[x1068];
float x1080 = x867[x1071];
float x1081 = x890[x1073];
float x1082 = -1.0f * x1079;
float x1083 = x1082 * x1081;
float x1084 = x1080 * x1080;
float x1085 = x1083 / x1084;
float x1086 = x1078 + x1085;
x869[x1071] = x1086;
x1066 += 1;
x1064 += 1;

}
// += tensor of dim 0
float x1093 = x869[0];
for(int x1094=0; x1094 < 5; x1094++) {
float x1095 = x853[x1094];
float x1096 = x1095 + x1093;
x853[x1094] = x1096;

}
for(int x1100=0; x1100 < 5; x1100++) {
float x1101 = x839[x1100];
float x1102 = x844[x1100];
float x1103 = x853[x1100];
float x1104 = x1102 * x1103;
float x1105 = x1101 + x1104;
x839[x1100] = x1105;

}
int32_t x1109 = 0;
int32_t x1110 = 0;
int32_t x1111 = 0;
for(int x1112=0; x1112 < 5; x1112++) {
int32_t x1113 = x1109;
float x1114 = x817[x1113];
float x1115 = x801[x1113];
int32_t x1116 = x1110;
float x1117 = x201[x1116];
int32_t x1118 = x1111;
float x1119 = x839[x1118];
float x1120 = x1114 + x1119;
x817[x1113] = x1120;
float x1122 = x312[x1116];
float x1123 = x801[x1113];
float x1124 = x201[x1116];
float x1125 = x839[x1118];
float x1126 = x1122 + x1125;
x312[x1116] = x1126;
x1111 += 1;
x1109 += 1;
x1110 += 1;

}
// add_cartesian
int32_t x1134 = 0;
for(int x1135=0; x1135 < 5; x1135++) {
for(int x1136=0; x1136 < 150; x1136++) {
int32_t x1137 = x1134;
int32_t x1138 = x1137 + x1136;
float x1139 = x307[x1138];
float x1140 = x778[x1136];
float x1141 = x817[x1135];
float x1142 = x1140 * x1141;
float x1143 = x1139 + x1142;
x307[x1138] = x1143;

}
x1134 += 150;

}
int32_t x1150 = 0;
for(int x1151=0; x1151 < 5; x1151++) {
for(int x1152=0; x1152 < 150; x1152++) {
float x1153 = x795[x1152];
int32_t x1154 = x1150;
int32_t x1155 = x1154 + x1152;
float x1156 = x192[x1155];
float x1157 = x817[x1151];
float x1158 = x1156 * x1157;
float x1159 = x1153 + x1158;
x795[x1152] = x1159;

}
x1150 += 150;

}
int32_t x1166 = 0;
int32_t x1167 = 0;
int32_t x1168 = 0;
for(int x1169=0; x1169 < 150; x1169++) {
int32_t x1170 = x1166;
float x1171 = x679[x1170];
float x1172 = x667[x1170];
int32_t x1173 = x1167;
float x1174 = x764[x1173];
int32_t x1175 = x1168;
float x1176 = x795[x1175];
float x1177 = x1176 * x1174;
float x1178 = x1171 + x1177;
x679[x1170] = x1178;
float x1180 = x773[x1173];
float x1181 = x667[x1170];
float x1182 = x764[x1173];
float x1183 = x795[x1175];
float x1184 = x1183 * x1181;
float x1185 = x1180 + x1184;
x773[x1173] = x1185;
x1168 += 1;
x1166 += 1;
x1167 += 1;

}
for(int x1192=0; x1192 < 150; x1192++) {
float x1193 = x759[x1192];
float x1194 = x764[x1192];
float x1197 = x773[x1192];
float x1195 = x1194 * x1194;
float x1196 = 1.0f - x1195;
float x1198 = x1196 * x1197;
float x1199 = x1193 + x1198;
x759[x1192] = x1199;

}
int32_t x1203 = 0;
int32_t x1204 = 0;
int32_t x1205 = 0;
for(int x1206=0; x1206 < 150; x1206++) {
int32_t x1207 = x1203;
float x1208 = x618[x1207];
float x1209 = x606[x1207];
int32_t x1210 = x1204;
float x1211 = x728[x1210];
int32_t x1212 = x1205;
float x1213 = x759[x1212];
float x1214 = x1213 * x1211;
float x1215 = x1208 + x1214;
x618[x1207] = x1215;
float x1217 = x737[x1210];
float x1218 = x606[x1207];
float x1219 = x728[x1210];
float x1220 = x759[x1212];
float x1221 = x1220 * x1218;
float x1222 = x1217 + x1221;
x737[x1210] = x1222;
x1205 += 1;
x1203 += 1;
x1204 += 1;

}
for(int x1229=0; x1229 < 150; x1229++) {
float x1230 = x723[x1229];
float x1231 = x728[x1229];
float x1234 = x737[x1229];
float x1232 = x1231 * x1231;
float x1233 = 1.0f - x1232;
float x1235 = x1233 * x1234;
float x1236 = x1230 + x1235;
x723[x1229] = x1236;

}
int32_t x1240 = 0;
int32_t x1241 = 0;
int32_t x1242 = 0;
for(int x1243=0; x1243 < 150; x1243++) {
int32_t x1244 = x1240;
float x1245 = x701[x1244];
float x1246 = x685[x1244];
int32_t x1247 = x1241;
float x1248 = x86[x1247];
int32_t x1249 = x1242;
float x1250 = x723[x1249];
float x1251 = x1245 + x1250;
x701[x1244] = x1251;
float x1253 = x232[x1247];
float x1254 = x685[x1244];
float x1255 = x86[x1247];
float x1256 = x723[x1249];
float x1257 = x1253 + x1256;
x232[x1247] = x1257;
x1242 += 1;
x1240 += 1;
x1241 += 1;

}
// add_cartesian
int32_t x1265 = 0;
for(int x1266=0; x1266 < 150; x1266++) {
for(int x1267=0; x1267 < 300; x1267++) {
int32_t x1268 = x1265;
int32_t x1269 = x1268 + x1267;
float x1270 = x227[x1269];
float x1271 = x556[x1267];
float x1272 = x701[x1266];
float x1273 = x1271 * x1272;
float x1274 = x1270 + x1273;
x227[x1269] = x1274;

}
x1265 += 300;

}
int32_t x1281 = 0;
for(int x1282=0; x1282 < 150; x1282++) {
for(int x1283=0; x1283 < 300; x1283++) {
float x1284 = x557[x1283];
int32_t x1285 = x1281;
int32_t x1286 = x1285 + x1283;
float x1287 = x78[x1286];
float x1288 = x701[x1282];
float x1289 = x1287 * x1288;
float x1290 = x1284 + x1289;
x557[x1283] = x1290;

}
x1281 += 300;

}
for(int x1297=0; x1297 < 150; x1297++) {
float x1298 = x662[x1297];
float x1299 = x667[x1297];
float x1302 = x679[x1297];
float x1300 = 1.0f - x1299;
float x1301 = x1300 * x1299;
float x1303 = x1301 * x1302;
float x1304 = x1298 + x1303;
x662[x1297] = x1304;

}
int32_t x1308 = 0;
int32_t x1309 = 0;
int32_t x1310 = 0;
for(int x1311=0; x1311 < 150; x1311++) {
int32_t x1312 = x1308;
float x1313 = x640[x1312];
float x1314 = x624[x1312];
int32_t x1315 = x1309;
float x1316 = x73[x1315];
int32_t x1317 = x1310;
float x1318 = x662[x1317];
float x1319 = x1313 + x1318;
x640[x1312] = x1319;
float x1321 = x222[x1315];
float x1322 = x624[x1312];
float x1323 = x73[x1315];
float x1324 = x662[x1317];
float x1325 = x1321 + x1324;
x222[x1315] = x1325;
x1310 += 1;
x1308 += 1;
x1309 += 1;

}
// add_cartesian
int32_t x1333 = 0;
for(int x1334=0; x1334 < 150; x1334++) {
for(int x1335=0; x1335 < 300; x1335++) {
int32_t x1336 = x1333;
int32_t x1337 = x1336 + x1335;
float x1338 = x217[x1337];
float x1339 = x556[x1335];
float x1340 = x640[x1334];
float x1341 = x1339 * x1340;
float x1342 = x1338 + x1341;
x217[x1337] = x1342;

}
x1333 += 300;

}
int32_t x1349 = 0;
for(int x1350=0; x1350 < 150; x1350++) {
for(int x1351=0; x1351 < 300; x1351++) {
float x1352 = x557[x1351];
int32_t x1353 = x1349;
int32_t x1354 = x1353 + x1351;
float x1355 = x65[x1354];
float x1356 = x640[x1350];
float x1357 = x1355 * x1356;
float x1358 = x1352 + x1357;
x557[x1351] = x1358;

}
x1349 += 300;

}
for(int x1365=0; x1365 < 150; x1365++) {
float x1366 = x601[x1365];
float x1367 = x606[x1365];
float x1370 = x618[x1365];
float x1368 = 1.0f - x1367;
float x1369 = x1368 * x1367;
float x1371 = x1369 * x1370;
float x1372 = x1366 + x1371;
x601[x1365] = x1372;

}
int32_t x1376 = 0;
int32_t x1377 = 0;
int32_t x1378 = 0;
for(int x1379=0; x1379 < 150; x1379++) {
int32_t x1380 = x1376;
float x1381 = x579[x1380];
float x1382 = x563[x1380];
int32_t x1383 = x1377;
float x1384 = x59[x1383];
int32_t x1385 = x1378;
float x1386 = x601[x1385];
float x1387 = x1381 + x1386;
x579[x1380] = x1387;
float x1389 = x212[x1383];
float x1390 = x563[x1380];
float x1391 = x59[x1383];
float x1392 = x601[x1385];
float x1393 = x1389 + x1392;
x212[x1383] = x1393;
x1378 += 1;
x1376 += 1;
x1377 += 1;

}
// add_cartesian
int32_t x1401 = 0;
for(int x1402=0; x1402 < 150; x1402++) {
for(int x1403=0; x1403 < 300; x1403++) {
int32_t x1404 = x1401;
int32_t x1405 = x1404 + x1403;
float x1406 = x207[x1405];
float x1407 = x556[x1403];
float x1408 = x579[x1402];
float x1409 = x1407 * x1408;
float x1410 = x1406 + x1409;
x207[x1405] = x1410;

}
x1401 += 300;

}
int32_t x1417 = 0;
for(int x1418=0; x1418 < 150; x1418++) {
for(int x1419=0; x1419 < 300; x1419++) {
float x1420 = x557[x1419];
int32_t x1421 = x1417;
int32_t x1422 = x1421 + x1419;
float x1423 = x50[x1422];
float x1424 = x579[x1418];
float x1425 = x1423 * x1424;
float x1426 = x1420 + x1425;
x557[x1419] = x1426;

}
x1417 += 300;

}
} else {
// dot: List(150, 150), WrappedArray(150)
float* x1435 = (float*)myMalloc(150 * sizeof(float));;
for(int x1436=0; x1436 < 150; x1436++) {
float x1437 = 0.0f;
int32_t x1439 = x1436 * 150;
for(int x1438=0; x1438 < 150; x1438++) {
int32_t x1440 = x1439 + x1438;
float x1441 = x91[x1440];
float x1442 = x528[x1438];
float x1443 = x1441 * x1442;
x1437 += x1443;

}
float x1447 = x1437;
x1435[x1436] = x1447;

}
float* x1451 = (float*)myMalloc(150 * sizeof(float));;
for(int x1452=0; x1452 < 150; x1452++) {
x1451[x1452] = 0.0f;

}
// dot: List(150, 150), WrappedArray(150)
float* x1457 = (float*)myMalloc(150 * sizeof(float));;
for(int x1458=0; x1458 < 150; x1458++) {
float x1459 = 0.0f;
int32_t x1461 = x1458 * 150;
for(int x1460=0; x1460 < 150; x1460++) {
int32_t x1462 = x1461 + x1460;
float x1463 = x100[x1462];
float x1464 = x537[x1460];
float x1465 = x1463 * x1464;
x1459 += x1465;

}
float x1469 = x1459;
x1457[x1458] = x1469;

}
float* x1473 = (float*)myMalloc(150 * sizeof(float));;
for(int x1474=0; x1474 < 150; x1474++) {
x1473[x1474] = 0.0f;

}
float* x1478 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1479 = 0;
int32_t x1480 = 0;
int32_t x1481 = 0;
for(int x1482=0; x1482 < 150; x1482++) {
int32_t x1483 = x1479;
int32_t x1484 = x1480;
float x1485 = x1435[x1484];
int32_t x1486 = x1481;
float x1487 = x1457[x1486];
float x1488 = x1485 + x1487;
x1478[x1483] = x1488;
x1479 += 1;
x1480 += 1;
x1481 += 1;

}
float* x1495 = (float*)myMalloc(150 * sizeof(float));;
for(int x1496=0; x1496 < 150; x1496++) {
x1495[x1496] = 0.0f;

}
float* x1500 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1501 = 0;
int32_t x1502 = 0;
int32_t x1503 = 0;
for(int x1504=0; x1504 < 150; x1504++) {
int32_t x1505 = x1501;
int32_t x1506 = x1502;
float x1507 = x1478[x1506];
int32_t x1508 = x1503;
float x1509 = x108[x1508];
float x1510 = x1507 + x1509;
x1500[x1505] = x1510;
x1501 += 1;
x1502 += 1;
x1503 += 1;

}
float* x1517 = (float*)myMalloc(150 * sizeof(float));;
for(int x1518=0; x1518 < 150; x1518++) {
x1517[x1518] = 0.0f;

}
float* x1522 = (float*)myMalloc(150 * sizeof(float));;
for(int x1523=0; x1523 < 150; x1523++) {
float x1524 = x1500[x1523];
float x1525 = -1.0f * x1524;
double x1526 = (double)x1525;
double x1527 = exp(x1526);
float x1528 = (float)x1527;
float x1529 = x1528 + 1.0f;
float x1530 = 1.0f / x1529;
x1522[x1523] = x1530;

}
float* x1534 = (float*)myMalloc(150 * sizeof(float));;
for(int x1535=0; x1535 < 150; x1535++) {
x1534[x1535] = 0.0f;

}
// dot: List(150, 150), WrappedArray(150)
float* x1540 = (float*)myMalloc(150 * sizeof(float));;
for(int x1541=0; x1541 < 150; x1541++) {
float x1542 = 0.0f;
int32_t x1544 = x1541 * 150;
for(int x1543=0; x1543 < 150; x1543++) {
int32_t x1545 = x1544 + x1543;
float x1546 = x113[x1545];
float x1547 = x528[x1543];
float x1548 = x1546 * x1547;
x1542 += x1548;

}
float x1552 = x1542;
x1540[x1541] = x1552;

}
float* x1556 = (float*)myMalloc(150 * sizeof(float));;
for(int x1557=0; x1557 < 150; x1557++) {
x1556[x1557] = 0.0f;

}
// dot: List(150, 150), WrappedArray(150)
float* x1562 = (float*)myMalloc(150 * sizeof(float));;
for(int x1563=0; x1563 < 150; x1563++) {
float x1564 = 0.0f;
int32_t x1566 = x1563 * 150;
for(int x1565=0; x1565 < 150; x1565++) {
int32_t x1567 = x1566 + x1565;
float x1568 = x121[x1567];
float x1569 = x537[x1565];
float x1570 = x1568 * x1569;
x1564 += x1570;

}
float x1574 = x1564;
x1562[x1563] = x1574;

}
float* x1578 = (float*)myMalloc(150 * sizeof(float));;
for(int x1579=0; x1579 < 150; x1579++) {
x1578[x1579] = 0.0f;

}
float* x1583 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1584 = 0;
int32_t x1585 = 0;
int32_t x1586 = 0;
for(int x1587=0; x1587 < 150; x1587++) {
int32_t x1588 = x1584;
int32_t x1589 = x1585;
float x1590 = x1540[x1589];
int32_t x1591 = x1586;
float x1592 = x1562[x1591];
float x1593 = x1590 + x1592;
x1583[x1588] = x1593;
x1584 += 1;
x1585 += 1;
x1586 += 1;

}
float* x1600 = (float*)myMalloc(150 * sizeof(float));;
for(int x1601=0; x1601 < 150; x1601++) {
x1600[x1601] = 0.0f;

}
float* x1605 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1606 = 0;
int32_t x1607 = 0;
int32_t x1608 = 0;
for(int x1609=0; x1609 < 150; x1609++) {
int32_t x1610 = x1606;
int32_t x1611 = x1607;
float x1612 = x1583[x1611];
int32_t x1613 = x1608;
float x1614 = x145[x1613];
float x1615 = x1612 + x1614;
x1605[x1610] = x1615;
x1606 += 1;
x1607 += 1;
x1608 += 1;

}
float* x1622 = (float*)myMalloc(150 * sizeof(float));;
for(int x1623=0; x1623 < 150; x1623++) {
x1622[x1623] = 0.0f;

}
float* x1627 = (float*)myMalloc(150 * sizeof(float));;
for(int x1628=0; x1628 < 150; x1628++) {
float x1629 = x1605[x1628];
float x1630 = -1.0f * x1629;
double x1631 = (double)x1630;
double x1632 = exp(x1631);
float x1633 = (float)x1632;
float x1634 = x1633 + 1.0f;
float x1635 = 1.0f / x1634;
x1627[x1628] = x1635;

}
float* x1639 = (float*)myMalloc(150 * sizeof(float));;
for(int x1640=0; x1640 < 150; x1640++) {
x1639[x1640] = 0.0f;

}
// dot: List(150, 150), WrappedArray(150)
float* x1645 = (float*)myMalloc(150 * sizeof(float));;
for(int x1646=0; x1646 < 150; x1646++) {
float x1647 = 0.0f;
int32_t x1649 = x1646 * 150;
for(int x1648=0; x1648 < 150; x1648++) {
int32_t x1650 = x1649 + x1648;
float x1651 = x129[x1650];
float x1652 = x528[x1648];
float x1653 = x1651 * x1652;
x1647 += x1653;

}
float x1657 = x1647;
x1645[x1646] = x1657;

}
float* x1661 = (float*)myMalloc(150 * sizeof(float));;
for(int x1662=0; x1662 < 150; x1662++) {
x1661[x1662] = 0.0f;

}
// dot: List(150, 150), WrappedArray(150)
float* x1667 = (float*)myMalloc(150 * sizeof(float));;
for(int x1668=0; x1668 < 150; x1668++) {
float x1669 = 0.0f;
int32_t x1671 = x1668 * 150;
for(int x1670=0; x1670 < 150; x1670++) {
int32_t x1672 = x1671 + x1670;
float x1673 = x137[x1672];
float x1674 = x537[x1670];
float x1675 = x1673 * x1674;
x1669 += x1675;

}
float x1679 = x1669;
x1667[x1668] = x1679;

}
float* x1683 = (float*)myMalloc(150 * sizeof(float));;
for(int x1684=0; x1684 < 150; x1684++) {
x1683[x1684] = 0.0f;

}
float* x1688 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1689 = 0;
int32_t x1690 = 0;
int32_t x1691 = 0;
for(int x1692=0; x1692 < 150; x1692++) {
int32_t x1693 = x1689;
int32_t x1694 = x1690;
float x1695 = x1645[x1694];
int32_t x1696 = x1691;
float x1697 = x1667[x1696];
float x1698 = x1695 + x1697;
x1688[x1693] = x1698;
x1689 += 1;
x1690 += 1;
x1691 += 1;

}
float* x1705 = (float*)myMalloc(150 * sizeof(float));;
for(int x1706=0; x1706 < 150; x1706++) {
x1705[x1706] = 0.0f;

}
float* x1710 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1711 = 0;
int32_t x1712 = 0;
int32_t x1713 = 0;
for(int x1714=0; x1714 < 150; x1714++) {
int32_t x1715 = x1711;
int32_t x1716 = x1712;
float x1717 = x1688[x1716];
int32_t x1718 = x1713;
float x1719 = x145[x1718];
float x1720 = x1717 + x1719;
x1710[x1715] = x1720;
x1711 += 1;
x1712 += 1;
x1713 += 1;

}
float* x1727 = (float*)myMalloc(150 * sizeof(float));;
for(int x1728=0; x1728 < 150; x1728++) {
x1727[x1728] = 0.0f;

}
float* x1732 = (float*)myMalloc(150 * sizeof(float));;
for(int x1733=0; x1733 < 150; x1733++) {
float x1734 = x1710[x1733];
float x1735 = -1.0f * x1734;
double x1736 = (double)x1735;
double x1737 = exp(x1736);
float x1738 = (float)x1737;
float x1739 = x1738 + 1.0f;
float x1740 = 1.0f / x1739;
x1732[x1733] = x1740;

}
float* x1744 = (float*)myMalloc(150 * sizeof(float));;
for(int x1745=0; x1745 < 150; x1745++) {
x1744[x1745] = 0.0f;

}
// dot: List(150, 150), WrappedArray(150)
float* x1750 = (float*)myMalloc(150 * sizeof(float));;
for(int x1751=0; x1751 < 150; x1751++) {
float x1752 = 0.0f;
int32_t x1754 = x1751 * 150;
for(int x1753=0; x1753 < 150; x1753++) {
int32_t x1755 = x1754 + x1753;
float x1756 = x150[x1755];
float x1757 = x528[x1753];
float x1758 = x1756 * x1757;
x1752 += x1758;

}
float x1762 = x1752;
x1750[x1751] = x1762;

}
float* x1766 = (float*)myMalloc(150 * sizeof(float));;
for(int x1767=0; x1767 < 150; x1767++) {
x1766[x1767] = 0.0f;

}
// dot: List(150, 150), WrappedArray(150)
float* x1772 = (float*)myMalloc(150 * sizeof(float));;
for(int x1773=0; x1773 < 150; x1773++) {
float x1774 = 0.0f;
int32_t x1776 = x1773 * 150;
for(int x1775=0; x1775 < 150; x1775++) {
int32_t x1777 = x1776 + x1775;
float x1778 = x158[x1777];
float x1779 = x537[x1775];
float x1780 = x1778 * x1779;
x1774 += x1780;

}
float x1784 = x1774;
x1772[x1773] = x1784;

}
float* x1788 = (float*)myMalloc(150 * sizeof(float));;
for(int x1789=0; x1789 < 150; x1789++) {
x1788[x1789] = 0.0f;

}
float* x1793 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1794 = 0;
int32_t x1795 = 0;
int32_t x1796 = 0;
for(int x1797=0; x1797 < 150; x1797++) {
int32_t x1798 = x1794;
int32_t x1799 = x1795;
float x1800 = x1750[x1799];
int32_t x1801 = x1796;
float x1802 = x1772[x1801];
float x1803 = x1800 + x1802;
x1793[x1798] = x1803;
x1794 += 1;
x1795 += 1;
x1796 += 1;

}
float* x1810 = (float*)myMalloc(150 * sizeof(float));;
for(int x1811=0; x1811 < 150; x1811++) {
x1810[x1811] = 0.0f;

}
float* x1815 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1816 = 0;
int32_t x1817 = 0;
int32_t x1818 = 0;
for(int x1819=0; x1819 < 150; x1819++) {
int32_t x1820 = x1816;
int32_t x1821 = x1817;
float x1822 = x1793[x1821];
int32_t x1823 = x1818;
float x1824 = x166[x1823];
float x1825 = x1822 + x1824;
x1815[x1820] = x1825;
x1816 += 1;
x1817 += 1;
x1818 += 1;

}
float* x1832 = (float*)myMalloc(150 * sizeof(float));;
for(int x1833=0; x1833 < 150; x1833++) {
x1832[x1833] = 0.0f;

}
float* x1837 = (float*)myMalloc(150 * sizeof(float));;
for(int x1838=0; x1838 < 150; x1838++) {
float x1839 = x1815[x1838];
float x1840 = -1.0f * x1839;
double x1841 = (double)x1840;
double x1842 = exp(x1841);
float x1843 = (float)x1842;
float x1844 = x1843 + 1.0f;
float x1845 = 1.0f / x1844;
x1837[x1838] = x1845;

}
float* x1849 = (float*)myMalloc(150 * sizeof(float));;
for(int x1850=0; x1850 < 150; x1850++) {
x1849[x1850] = 0.0f;

}
// dot: List(150, 150), WrappedArray(150)
float* x1855 = (float*)myMalloc(150 * sizeof(float));;
for(int x1856=0; x1856 < 150; x1856++) {
float x1857 = 0.0f;
int32_t x1859 = x1856 * 150;
for(int x1858=0; x1858 < 150; x1858++) {
int32_t x1860 = x1859 + x1858;
float x1861 = x171[x1860];
float x1862 = x528[x1858];
float x1863 = x1861 * x1862;
x1857 += x1863;

}
float x1867 = x1857;
x1855[x1856] = x1867;

}
float* x1871 = (float*)myMalloc(150 * sizeof(float));;
for(int x1872=0; x1872 < 150; x1872++) {
x1871[x1872] = 0.0f;

}
// dot: List(150, 150), WrappedArray(150)
float* x1877 = (float*)myMalloc(150 * sizeof(float));;
for(int x1878=0; x1878 < 150; x1878++) {
float x1879 = 0.0f;
int32_t x1881 = x1878 * 150;
for(int x1880=0; x1880 < 150; x1880++) {
int32_t x1882 = x1881 + x1880;
float x1883 = x179[x1882];
float x1884 = x537[x1880];
float x1885 = x1883 * x1884;
x1879 += x1885;

}
float x1889 = x1879;
x1877[x1878] = x1889;

}
float* x1893 = (float*)myMalloc(150 * sizeof(float));;
for(int x1894=0; x1894 < 150; x1894++) {
x1893[x1894] = 0.0f;

}
float* x1898 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1899 = 0;
int32_t x1900 = 0;
int32_t x1901 = 0;
for(int x1902=0; x1902 < 150; x1902++) {
int32_t x1903 = x1899;
int32_t x1904 = x1900;
float x1905 = x1855[x1904];
int32_t x1906 = x1901;
float x1907 = x1877[x1906];
float x1908 = x1905 + x1907;
x1898[x1903] = x1908;
x1899 += 1;
x1900 += 1;
x1901 += 1;

}
float* x1915 = (float*)myMalloc(150 * sizeof(float));;
for(int x1916=0; x1916 < 150; x1916++) {
x1915[x1916] = 0.0f;

}
float* x1920 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1921 = 0;
int32_t x1922 = 0;
int32_t x1923 = 0;
for(int x1924=0; x1924 < 150; x1924++) {
int32_t x1925 = x1921;
int32_t x1926 = x1922;
float x1927 = x1898[x1926];
int32_t x1928 = x1923;
float x1929 = x187[x1928];
float x1930 = x1927 + x1929;
x1920[x1925] = x1930;
x1921 += 1;
x1922 += 1;
x1923 += 1;

}
float* x1937 = (float*)myMalloc(150 * sizeof(float));;
for(int x1938=0; x1938 < 150; x1938++) {
x1937[x1938] = 0.0f;

}
float* x1942 = (float*)myMalloc(150 * sizeof(float));;
for(int x1943=0; x1943 < 150; x1943++) {
float x1944 = x1920[x1943];
double x1945 = (double)x1944;
double x1946 = tanh(x1945);
float x1947 = (float)x1946;
x1942[x1943] = x1947;

}
float* x1951 = (float*)myMalloc(150 * sizeof(float));;
for(int x1952=0; x1952 < 150; x1952++) {
x1951[x1952] = 0.0f;

}
float* x1956 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1957 = 0;
int32_t x1958 = 0;
int32_t x1959 = 0;
for(int x1960=0; x1960 < 150; x1960++) {
int32_t x1961 = x1957;
int32_t x1962 = x1958;
float x1963 = x1522[x1962];
int32_t x1964 = x1959;
float x1965 = x1942[x1964];
float x1966 = x1963 * x1965;
x1956[x1961] = x1966;
x1957 += 1;
x1958 += 1;
x1959 += 1;

}
float* x1973 = (float*)myMalloc(150 * sizeof(float));;
for(int x1974=0; x1974 < 150; x1974++) {
x1973[x1974] = 0.0f;

}
float* x1978 = (float*)myMalloc(150 * sizeof(float));;
int32_t x1979 = 0;
int32_t x1980 = 0;
int32_t x1981 = 0;
for(int x1982=0; x1982 < 150; x1982++) {
int32_t x1983 = x1979;
int32_t x1984 = x1980;
float x1985 = x1627[x1984];
int32_t x1986 = x1981;
float x1987 = x530[x1986];
float x1988 = x1985 * x1987;
x1978[x1983] = x1988;
x1979 += 1;
x1980 += 1;
x1981 += 1;

}
float* x1995 = (float*)myMalloc(150 * sizeof(float));;
for(int x1996=0; x1996 < 150; x1996++) {
x1995[x1996] = 0.0f;

}
float* x2000 = (float*)myMalloc(150 * sizeof(float));;
int32_t x2001 = 0;
int32_t x2002 = 0;
int32_t x2003 = 0;
for(int x2004=0; x2004 < 150; x2004++) {
int32_t x2005 = x2001;
int32_t x2006 = x2002;
float x2007 = x1956[x2006];
int32_t x2008 = x2003;
float x2009 = x1978[x2008];
float x2010 = x2007 + x2009;
x2000[x2005] = x2010;
x2001 += 1;
x2002 += 1;
x2003 += 1;

}
float* x2017 = (float*)myMalloc(150 * sizeof(float));;
for(int x2018=0; x2018 < 150; x2018++) {
x2017[x2018] = 0.0f;

}
float* x2022 = (float*)myMalloc(150 * sizeof(float));;
int32_t x2023 = 0;
int32_t x2024 = 0;
int32_t x2025 = 0;
for(int x2026=0; x2026 < 150; x2026++) {
int32_t x2027 = x2023;
int32_t x2028 = x2024;
float x2029 = x1732[x2028];
int32_t x2030 = x2025;
float x2031 = x539[x2030];
float x2032 = x2029 * x2031;
x2022[x2027] = x2032;
x2023 += 1;
x2024 += 1;
x2025 += 1;

}
float* x2039 = (float*)myMalloc(150 * sizeof(float));;
for(int x2040=0; x2040 < 150; x2040++) {
x2039[x2040] = 0.0f;

}
float* x2044 = (float*)myMalloc(150 * sizeof(float));;
int32_t x2045 = 0;
int32_t x2046 = 0;
int32_t x2047 = 0;
for(int x2048=0; x2048 < 150; x2048++) {
int32_t x2049 = x2045;
int32_t x2050 = x2046;
float x2051 = x2000[x2050];
int32_t x2052 = x2047;
float x2053 = x2022[x2052];
float x2054 = x2051 + x2053;
x2044[x2049] = x2054;
x2045 += 1;
x2046 += 1;
x2047 += 1;

}
float* x2061 = (float*)myMalloc(150 * sizeof(float));;
for(int x2062=0; x2062 < 150; x2062++) {
x2061[x2062] = 0.0f;

}
float* x2066 = (float*)myMalloc(150 * sizeof(float));;
for(int x2067=0; x2067 < 150; x2067++) {
float x2068 = x2044[x2067];
double x2069 = (double)x2068;
double x2070 = tanh(x2069);
float x2071 = (float)x2070;
x2066[x2067] = x2071;

}
float* x2075 = (float*)myMalloc(150 * sizeof(float));;
for(int x2076=0; x2076 < 150; x2076++) {
x2075[x2076] = 0.0f;

}
float* x2080 = (float*)myMalloc(150 * sizeof(float));;
int32_t x2081 = 0;
int32_t x2082 = 0;
int32_t x2083 = 0;
for(int x2084=0; x2084 < 150; x2084++) {
int32_t x2085 = x2081;
int32_t x2086 = x2082;
float x2087 = x1837[x2086];
int32_t x2088 = x2083;
float x2089 = x2066[x2088];
float x2090 = x2087 * x2089;
x2080[x2085] = x2090;
x2081 += 1;
x2082 += 1;
x2083 += 1;

}
float* x2097 = (float*)myMalloc(150 * sizeof(float));;
for(int x2098=0; x2098 < 150; x2098++) {
x2097[x2098] = 0.0f;

}
// dot: List(5, 150), List(150)
float* x2103 = (float*)myMalloc(5 * sizeof(float));;
for(int x2104=0; x2104 < 5; x2104++) {
float x2105 = 0.0f;
int32_t x2107 = x2104 * 150;
for(int x2106=0; x2106 < 150; x2106++) {
int32_t x2108 = x2107 + x2106;
float x2109 = x192[x2108];
float x2110 = x2080[x2106];
float x2111 = x2109 * x2110;
x2105 += x2111;

}
float x2115 = x2105;
x2103[x2104] = x2115;

}
float* x2119 = (float*)myMalloc(5 * sizeof(float));;
for(int x2120=0; x2120 < 5; x2120++) {
x2119[x2120] = 0.0f;

}
float* x2124 = (float*)myMalloc(5 * sizeof(float));;
int32_t x2125 = 0;
int32_t x2126 = 0;
int32_t x2127 = 0;
for(int x2128=0; x2128 < 5; x2128++) {
int32_t x2129 = x2125;
int32_t x2130 = x2126;
float x2131 = x2103[x2130];
int32_t x2132 = x2127;
float x2133 = x201[x2132];
float x2134 = x2131 + x2133;
x2124[x2129] = x2134;
x2125 += 1;
x2126 += 1;
x2127 += 1;

}
float* x2141 = (float*)myMalloc(5 * sizeof(float));;
for(int x2142=0; x2142 < 5; x2142++) {
x2141[x2142] = 0.0f;

}
float* x2146 = (float*)myMalloc(5 * sizeof(float));;
for(int x2147=0; x2147 < 5; x2147++) {
float x2148 = x2124[x2147];
double x2149 = (double)x2148;
double x2150 = exp(x2149);
float x2151 = (float)x2150;
x2146[x2147] = x2151;

}
float* x2155 = (float*)myMalloc(5 * sizeof(float));;
for(int x2156=0; x2156 < 5; x2156++) {
x2155[x2156] = 0.0f;

}
float x2160 = 0.0f;
for(int x2161=0; x2161 < 5; x2161++) {
float x2162 = x2160;
float x2163 = x2146[x2161];
float x2164 = x2162 + x2163;
x2160 = x2164;

}
float x2168 = x2160;
float* x2169 = (float*)myMalloc(1 * sizeof(float));;
x2169[0] = x2168;
float* x2171 = (float*)myMalloc(1 * sizeof(float));;
for(int x2172=0; x2172 < 1; x2172++) {
x2171[x2172] = 0.0f;

}
float* x2176 = (float*)myMalloc(5 * sizeof(float));;
int32_t x2177 = 0;
int32_t x2178 = 0;
int32_t x2179 = 0;
for(int x2180=0; x2180 < 5; x2180++) {
int32_t x2181 = x2177;
int32_t x2182 = x2178;
float x2183 = x2146[x2182];
int32_t x2184 = x2179;
float x2185 = x2169[x2184];
float x2186 = x2183 / x2185;
x2176[x2181] = x2186;
x2177 += 1;
x2178 += 1;

}
float* x2192 = (float*)myMalloc(5 * sizeof(float));;
for(int x2193=0; x2193 < 5; x2193++) {
x2192[x2193] = 0.0f;

}
float* x2197 = (float*)myMalloc(1 * sizeof(float));;
int32_t x2198 = 0;
int32_t x2199 = 0;
int32_t x2200 = 0;
int32_t x2201 = x2198;
int32_t x2202 = x2199;
float x2203 = x526[x2202];
int32_t x2204 = x2200;
float x2205 = x535[x2204];
float x2206 = x2203 + x2205;
x2197[x2201] = x2206;
x2198 += 1;
float* x2209 = (float*)myMalloc(1 * sizeof(float));;
for(int x2210=0; x2210 < 1; x2210++) {
x2209[x2210] = 0.0f;

}
// dot: List(5), WrappedArray(5)
float x2215 = 0.0f;
for(int x2216=0; x2216 < 5; x2216++) {
float x2217 = x2176[x2216];
float x2218 = x541[x2216];
float x2219 = x2217 * x2218;
x2215 += x2219;

}
float* x2223 = (float*)myMalloc(1 * sizeof(float));;
float x2224 = x2215;
x2223[0] = x2224;
float* x2226 = (float*)myMalloc(1 * sizeof(float));;
for(int x2227=0; x2227 < 1; x2227++) {
x2226[x2227] = 0.0f;

}
float* x2231 = (float*)myMalloc(1 * sizeof(float));;
float x2232 = x2223[0];
double x2233 = (double)x2232;
double x2234 = log(x2233);
float x2235 = (float)x2234;
x2231[0] = x2235;
float* x2237 = (float*)myMalloc(1 * sizeof(float));;
for(int x2238=0; x2238 < 1; x2238++) {
x2237[x2238] = 0.0f;

}
float* x2242 = (float*)myMalloc(1 * sizeof(float));;
int32_t x2243 = 0;
int32_t x2244 = 0;
int32_t x2245 = 0;
int32_t x2246 = x2243;
int32_t x2247 = x2244;
float x2248 = x2197[x2247];
int32_t x2249 = x2245;
float x2250 = x2231[x2249];
float x2251 = x2248 - x2250;
x2242[x2246] = x2251;
x2243 += 1;
float* x2254 = (float*)myMalloc(1 * sizeof(float));;
for(int x2255=0; x2255 < 1; x2255++) {
x2254[x2255] = 0.0f;

}
float** x2259 = (float**)myMalloc(6 * sizeof(float*));;
x2259[0] = x2242;
x2259[1] = x2254;
x2259[2] = x2080;
x2259[3] = x2097;
x2259[4] = x2044;
x2259[5] = x2061;
x957(x2259);
int32_t x2267 = 0;
int32_t x2268 = 0;
int32_t x2269 = 0;
int32_t x2270 = x2267;
float x2271 = x2209[x2270];
float x2272 = x2197[x2270];
int32_t x2273 = x2268;
float x2274 = x2231[x2273];
int32_t x2275 = x2269;
float x2276 = x2254[x2275];
float x2277 = x2271 + x2276;
x2209[x2270] = x2277;
float x2279 = x2237[x2273];
float x2280 = x2197[x2270];
float x2281 = x2231[x2273];
float x2282 = x2254[x2275];
float x2283 = -1.0f * x2282;
float x2284 = x2279 + x2283;
x2237[x2273] = x2284;
x2269 += 1;
float x2287 = x2226[0];
float x2288 = x2237[0];
float x2289 = x2223[0];
float x2290 = x2288 / x2289;
float x2291 = x2287 + x2290;
x2226[0] = x2291;
float x2293 = x2226[0];
// Generate code for addMul
for(int x2295=0; x2295 < 5; x2295++) {
float x2296 = x2192[x2295];
float x2297 = x541[x2295];
float x2298 = x2293 * x2297;
float x2299 = x2296 + x2298;
x2192[x2295] = x2299;

}
float x2303 = x2226[0];
// Generate code for addMul
for(int x2305=0; x2305 < 5; x2305++) {
float x2306 = x548[x2305];
float x2307 = x2176[x2305];
float x2308 = x2303 * x2307;
float x2309 = x2306 + x2308;
x548[x2305] = x2309;

}
int32_t x2313 = 0;
int32_t x2314 = 0;
int32_t x2315 = 0;
int32_t x2316 = x2313;
float x2317 = x527[x2316];
float x2318 = x526[x2316];
int32_t x2319 = x2314;
float x2320 = x535[x2319];
int32_t x2321 = x2315;
float x2322 = x2209[x2321];
float x2323 = x2317 + x2322;
x527[x2316] = x2323;
float x2325 = x536[x2319];
float x2326 = x526[x2316];
float x2327 = x535[x2319];
float x2328 = x2209[x2321];
float x2329 = x2325 + x2328;
x536[x2319] = x2329;
x2315 += 1;
int32_t x2332 = 0;
int32_t x2333 = 0;
int32_t x2334 = 0;
for(int x2335=0; x2335 < 5; x2335++) {
int32_t x2336 = x2332;
float x2337 = x2155[x2336];
float x2338 = x2146[x2336];
int32_t x2339 = x2333;
float x2340 = x2169[x2339];
int32_t x2341 = x2334;
float x2342 = x2192[x2341];
float x2343 = x2342 / x2340;
float x2344 = x2337 + x2343;
x2155[x2336] = x2344;
float x2346 = x2171[x2339];
float x2347 = x2146[x2336];
float x2348 = x2169[x2339];
float x2349 = x2192[x2341];
float x2350 = -1.0f * x2347;
float x2351 = x2350 * x2349;
float x2352 = x2348 * x2348;
float x2353 = x2351 / x2352;
float x2354 = x2346 + x2353;
x2171[x2339] = x2354;
x2334 += 1;
x2332 += 1;

}
// += tensor of dim 0
float x2361 = x2171[0];
for(int x2362=0; x2362 < 5; x2362++) {
float x2363 = x2155[x2362];
float x2364 = x2363 + x2361;
x2155[x2362] = x2364;

}
for(int x2368=0; x2368 < 5; x2368++) {
float x2369 = x2141[x2368];
float x2370 = x2146[x2368];
float x2371 = x2155[x2368];
float x2372 = x2370 * x2371;
float x2373 = x2369 + x2372;
x2141[x2368] = x2373;

}
int32_t x2377 = 0;
int32_t x2378 = 0;
int32_t x2379 = 0;
for(int x2380=0; x2380 < 5; x2380++) {
int32_t x2381 = x2377;
float x2382 = x2119[x2381];
float x2383 = x2103[x2381];
int32_t x2384 = x2378;
float x2385 = x201[x2384];
int32_t x2386 = x2379;
float x2387 = x2141[x2386];
float x2388 = x2382 + x2387;
x2119[x2381] = x2388;
float x2390 = x312[x2384];
float x2391 = x2103[x2381];
float x2392 = x201[x2384];
float x2393 = x2141[x2386];
float x2394 = x2390 + x2393;
x312[x2384] = x2394;
x2379 += 1;
x2377 += 1;
x2378 += 1;

}
// add_cartesian
int32_t x2402 = 0;
for(int x2403=0; x2403 < 5; x2403++) {
for(int x2404=0; x2404 < 150; x2404++) {
int32_t x2405 = x2402;
int32_t x2406 = x2405 + x2404;
float x2407 = x307[x2406];
float x2408 = x2080[x2404];
float x2409 = x2119[x2403];
float x2410 = x2408 * x2409;
float x2411 = x2407 + x2410;
x307[x2406] = x2411;

}
x2402 += 150;

}
int32_t x2418 = 0;
for(int x2419=0; x2419 < 5; x2419++) {
for(int x2420=0; x2420 < 150; x2420++) {
float x2421 = x2097[x2420];
int32_t x2422 = x2418;
int32_t x2423 = x2422 + x2420;
float x2424 = x192[x2423];
float x2425 = x2119[x2419];
float x2426 = x2424 * x2425;
float x2427 = x2421 + x2426;
x2097[x2420] = x2427;

}
x2418 += 150;

}
int32_t x2434 = 0;
int32_t x2435 = 0;
int32_t x2436 = 0;
for(int x2437=0; x2437 < 150; x2437++) {
int32_t x2438 = x2434;
float x2439 = x1849[x2438];
float x2440 = x1837[x2438];
int32_t x2441 = x2435;
float x2442 = x2066[x2441];
int32_t x2443 = x2436;
float x2444 = x2097[x2443];
float x2445 = x2444 * x2442;
float x2446 = x2439 + x2445;
x1849[x2438] = x2446;
float x2448 = x2075[x2441];
float x2449 = x1837[x2438];
float x2450 = x2066[x2441];
float x2451 = x2097[x2443];
float x2452 = x2451 * x2449;
float x2453 = x2448 + x2452;
x2075[x2441] = x2453;
x2436 += 1;
x2434 += 1;
x2435 += 1;

}
for(int x2460=0; x2460 < 150; x2460++) {
float x2461 = x2061[x2460];
float x2462 = x2066[x2460];
float x2465 = x2075[x2460];
float x2463 = x2462 * x2462;
float x2464 = 1.0f - x2463;
float x2466 = x2464 * x2465;
float x2467 = x2461 + x2466;
x2061[x2460] = x2467;

}
int32_t x2471 = 0;
int32_t x2472 = 0;
int32_t x2473 = 0;
for(int x2474=0; x2474 < 150; x2474++) {
int32_t x2475 = x2471;
float x2476 = x2017[x2475];
float x2477 = x2000[x2475];
int32_t x2478 = x2472;
float x2479 = x2022[x2478];
int32_t x2480 = x2473;
float x2481 = x2061[x2480];
float x2482 = x2476 + x2481;
x2017[x2475] = x2482;
float x2484 = x2039[x2478];
float x2485 = x2000[x2475];
float x2486 = x2022[x2478];
float x2487 = x2061[x2480];
float x2488 = x2484 + x2487;
x2039[x2478] = x2488;
x2473 += 1;
x2471 += 1;
x2472 += 1;

}
int32_t x2495 = 0;
int32_t x2496 = 0;
int32_t x2497 = 0;
for(int x2498=0; x2498 < 150; x2498++) {
int32_t x2499 = x2495;
float x2500 = x1744[x2499];
float x2501 = x1732[x2499];
int32_t x2502 = x2496;
float x2503 = x539[x2502];
int32_t x2504 = x2497;
float x2505 = x2039[x2504];
float x2506 = x2505 * x2503;
float x2507 = x2500 + x2506;
x1744[x2499] = x2507;
float x2509 = x540[x2502];
float x2510 = x1732[x2499];
float x2511 = x539[x2502];
float x2512 = x2039[x2504];
float x2513 = x2512 * x2510;
float x2514 = x2509 + x2513;
x540[x2502] = x2514;
x2497 += 1;
x2495 += 1;
x2496 += 1;

}
int32_t x2521 = 0;
int32_t x2522 = 0;
int32_t x2523 = 0;
for(int x2524=0; x2524 < 150; x2524++) {
int32_t x2525 = x2521;
float x2526 = x1973[x2525];
float x2527 = x1956[x2525];
int32_t x2528 = x2522;
float x2529 = x1978[x2528];
int32_t x2530 = x2523;
float x2531 = x2017[x2530];
float x2532 = x2526 + x2531;
x1973[x2525] = x2532;
float x2534 = x1995[x2528];
float x2535 = x1956[x2525];
float x2536 = x1978[x2528];
float x2537 = x2017[x2530];
float x2538 = x2534 + x2537;
x1995[x2528] = x2538;
x2523 += 1;
x2521 += 1;
x2522 += 1;

}
int32_t x2545 = 0;
int32_t x2546 = 0;
int32_t x2547 = 0;
for(int x2548=0; x2548 < 150; x2548++) {
int32_t x2549 = x2545;
float x2550 = x1639[x2549];
float x2551 = x1627[x2549];
int32_t x2552 = x2546;
float x2553 = x530[x2552];
int32_t x2554 = x2547;
float x2555 = x1995[x2554];
float x2556 = x2555 * x2553;
float x2557 = x2550 + x2556;
x1639[x2549] = x2557;
float x2559 = x531[x2552];
float x2560 = x1627[x2549];
float x2561 = x530[x2552];
float x2562 = x1995[x2554];
float x2563 = x2562 * x2560;
float x2564 = x2559 + x2563;
x531[x2552] = x2564;
x2547 += 1;
x2545 += 1;
x2546 += 1;

}
int32_t x2571 = 0;
int32_t x2572 = 0;
int32_t x2573 = 0;
for(int x2574=0; x2574 < 150; x2574++) {
int32_t x2575 = x2571;
float x2576 = x1534[x2575];
float x2577 = x1522[x2575];
int32_t x2578 = x2572;
float x2579 = x1942[x2578];
int32_t x2580 = x2573;
float x2581 = x1973[x2580];
float x2582 = x2581 * x2579;
float x2583 = x2576 + x2582;
x1534[x2575] = x2583;
float x2585 = x1951[x2578];
float x2586 = x1522[x2575];
float x2587 = x1942[x2578];
float x2588 = x1973[x2580];
float x2589 = x2588 * x2586;
float x2590 = x2585 + x2589;
x1951[x2578] = x2590;
x2573 += 1;
x2571 += 1;
x2572 += 1;

}
for(int x2597=0; x2597 < 150; x2597++) {
float x2598 = x1937[x2597];
float x2599 = x1942[x2597];
float x2602 = x1951[x2597];
float x2600 = x2599 * x2599;
float x2601 = 1.0f - x2600;
float x2603 = x2601 * x2602;
float x2604 = x2598 + x2603;
x1937[x2597] = x2604;

}
int32_t x2608 = 0;
int32_t x2609 = 0;
int32_t x2610 = 0;
for(int x2611=0; x2611 < 150; x2611++) {
int32_t x2612 = x2608;
float x2613 = x1915[x2612];
float x2614 = x1898[x2612];
int32_t x2615 = x2609;
float x2616 = x187[x2615];
int32_t x2617 = x2610;
float x2618 = x1937[x2617];
float x2619 = x2613 + x2618;
x1915[x2612] = x2619;
float x2621 = x302[x2615];
float x2622 = x1898[x2612];
float x2623 = x187[x2615];
float x2624 = x1937[x2617];
float x2625 = x2621 + x2624;
x302[x2615] = x2625;
x2610 += 1;
x2608 += 1;
x2609 += 1;

}
int32_t x2632 = 0;
int32_t x2633 = 0;
int32_t x2634 = 0;
for(int x2635=0; x2635 < 150; x2635++) {
int32_t x2636 = x2632;
float x2637 = x1871[x2636];
float x2638 = x1855[x2636];
int32_t x2639 = x2633;
float x2640 = x1877[x2639];
int32_t x2641 = x2634;
float x2642 = x1915[x2641];
float x2643 = x2637 + x2642;
x1871[x2636] = x2643;
float x2645 = x1893[x2639];
float x2646 = x1855[x2636];
float x2647 = x1877[x2639];
float x2648 = x1915[x2641];
float x2649 = x2645 + x2648;
x1893[x2639] = x2649;
x2634 += 1;
x2632 += 1;
x2633 += 1;

}
// add_cartesian
int32_t x2657 = 0;
for(int x2658=0; x2658 < 150; x2658++) {
for(int x2659=0; x2659 < 150; x2659++) {
int32_t x2660 = x2657;
int32_t x2661 = x2660 + x2659;
float x2662 = x297[x2661];
float x2663 = x537[x2659];
float x2664 = x1893[x2658];
float x2665 = x2663 * x2664;
float x2666 = x2662 + x2665;
x297[x2661] = x2666;

}
x2657 += 150;

}
int32_t x2673 = 0;
for(int x2674=0; x2674 < 150; x2674++) {
for(int x2675=0; x2675 < 150; x2675++) {
float x2676 = x538[x2675];
int32_t x2677 = x2673;
int32_t x2678 = x2677 + x2675;
float x2679 = x179[x2678];
float x2680 = x1893[x2674];
float x2681 = x2679 * x2680;
float x2682 = x2676 + x2681;
x538[x2675] = x2682;

}
x2673 += 150;

}
// add_cartesian
int32_t x2690 = 0;
for(int x2691=0; x2691 < 150; x2691++) {
for(int x2692=0; x2692 < 150; x2692++) {
int32_t x2693 = x2690;
int32_t x2694 = x2693 + x2692;
float x2695 = x292[x2694];
float x2696 = x528[x2692];
float x2697 = x1871[x2691];
float x2698 = x2696 * x2697;
float x2699 = x2695 + x2698;
x292[x2694] = x2699;

}
x2690 += 150;

}
int32_t x2706 = 0;
for(int x2707=0; x2707 < 150; x2707++) {
for(int x2708=0; x2708 < 150; x2708++) {
float x2709 = x529[x2708];
int32_t x2710 = x2706;
int32_t x2711 = x2710 + x2708;
float x2712 = x171[x2711];
float x2713 = x1871[x2707];
float x2714 = x2712 * x2713;
float x2715 = x2709 + x2714;
x529[x2708] = x2715;

}
x2706 += 150;

}
for(int x2722=0; x2722 < 150; x2722++) {
float x2723 = x1832[x2722];
float x2724 = x1837[x2722];
float x2727 = x1849[x2722];
float x2725 = 1.0f - x2724;
float x2726 = x2725 * x2724;
float x2728 = x2726 * x2727;
float x2729 = x2723 + x2728;
x1832[x2722] = x2729;

}
int32_t x2733 = 0;
int32_t x2734 = 0;
int32_t x2735 = 0;
for(int x2736=0; x2736 < 150; x2736++) {
int32_t x2737 = x2733;
float x2738 = x1810[x2737];
float x2739 = x1793[x2737];
int32_t x2740 = x2734;
float x2741 = x166[x2740];
int32_t x2742 = x2735;
float x2743 = x1832[x2742];
float x2744 = x2738 + x2743;
x1810[x2737] = x2744;
float x2746 = x287[x2740];
float x2747 = x1793[x2737];
float x2748 = x166[x2740];
float x2749 = x1832[x2742];
float x2750 = x2746 + x2749;
x287[x2740] = x2750;
x2735 += 1;
x2733 += 1;
x2734 += 1;

}
int32_t x2757 = 0;
int32_t x2758 = 0;
int32_t x2759 = 0;
for(int x2760=0; x2760 < 150; x2760++) {
int32_t x2761 = x2757;
float x2762 = x1766[x2761];
float x2763 = x1750[x2761];
int32_t x2764 = x2758;
float x2765 = x1772[x2764];
int32_t x2766 = x2759;
float x2767 = x1810[x2766];
float x2768 = x2762 + x2767;
x1766[x2761] = x2768;
float x2770 = x1788[x2764];
float x2771 = x1750[x2761];
float x2772 = x1772[x2764];
float x2773 = x1810[x2766];
float x2774 = x2770 + x2773;
x1788[x2764] = x2774;
x2759 += 1;
x2757 += 1;
x2758 += 1;

}
// add_cartesian
int32_t x2782 = 0;
for(int x2783=0; x2783 < 150; x2783++) {
for(int x2784=0; x2784 < 150; x2784++) {
int32_t x2785 = x2782;
int32_t x2786 = x2785 + x2784;
float x2787 = x282[x2786];
float x2788 = x537[x2784];
float x2789 = x1788[x2783];
float x2790 = x2788 * x2789;
float x2791 = x2787 + x2790;
x282[x2786] = x2791;

}
x2782 += 150;

}
int32_t x2798 = 0;
for(int x2799=0; x2799 < 150; x2799++) {
for(int x2800=0; x2800 < 150; x2800++) {
float x2801 = x538[x2800];
int32_t x2802 = x2798;
int32_t x2803 = x2802 + x2800;
float x2804 = x158[x2803];
float x2805 = x1788[x2799];
float x2806 = x2804 * x2805;
float x2807 = x2801 + x2806;
x538[x2800] = x2807;

}
x2798 += 150;

}
// add_cartesian
int32_t x2815 = 0;
for(int x2816=0; x2816 < 150; x2816++) {
for(int x2817=0; x2817 < 150; x2817++) {
int32_t x2818 = x2815;
int32_t x2819 = x2818 + x2817;
float x2820 = x277[x2819];
float x2821 = x528[x2817];
float x2822 = x1766[x2816];
float x2823 = x2821 * x2822;
float x2824 = x2820 + x2823;
x277[x2819] = x2824;

}
x2815 += 150;

}
int32_t x2831 = 0;
for(int x2832=0; x2832 < 150; x2832++) {
for(int x2833=0; x2833 < 150; x2833++) {
float x2834 = x529[x2833];
int32_t x2835 = x2831;
int32_t x2836 = x2835 + x2833;
float x2837 = x150[x2836];
float x2838 = x1766[x2832];
float x2839 = x2837 * x2838;
float x2840 = x2834 + x2839;
x529[x2833] = x2840;

}
x2831 += 150;

}
for(int x2847=0; x2847 < 150; x2847++) {
float x2848 = x1727[x2847];
float x2849 = x1732[x2847];
float x2852 = x1744[x2847];
float x2850 = 1.0f - x2849;
float x2851 = x2850 * x2849;
float x2853 = x2851 * x2852;
float x2854 = x2848 + x2853;
x1727[x2847] = x2854;

}
int32_t x2858 = 0;
int32_t x2859 = 0;
int32_t x2860 = 0;
for(int x2861=0; x2861 < 150; x2861++) {
int32_t x2862 = x2858;
float x2863 = x1705[x2862];
float x2864 = x1688[x2862];
int32_t x2865 = x2859;
float x2866 = x145[x2865];
int32_t x2867 = x2860;
float x2868 = x1727[x2867];
float x2869 = x2863 + x2868;
x1705[x2862] = x2869;
float x2871 = x272[x2865];
float x2872 = x1688[x2862];
float x2873 = x145[x2865];
float x2874 = x1727[x2867];
float x2875 = x2871 + x2874;
x272[x2865] = x2875;
x2860 += 1;
x2858 += 1;
x2859 += 1;

}
int32_t x2882 = 0;
int32_t x2883 = 0;
int32_t x2884 = 0;
for(int x2885=0; x2885 < 150; x2885++) {
int32_t x2886 = x2882;
float x2887 = x1661[x2886];
float x2888 = x1645[x2886];
int32_t x2889 = x2883;
float x2890 = x1667[x2889];
int32_t x2891 = x2884;
float x2892 = x1705[x2891];
float x2893 = x2887 + x2892;
x1661[x2886] = x2893;
float x2895 = x1683[x2889];
float x2896 = x1645[x2886];
float x2897 = x1667[x2889];
float x2898 = x1705[x2891];
float x2899 = x2895 + x2898;
x1683[x2889] = x2899;
x2884 += 1;
x2882 += 1;
x2883 += 1;

}
// add_cartesian
int32_t x2907 = 0;
for(int x2908=0; x2908 < 150; x2908++) {
for(int x2909=0; x2909 < 150; x2909++) {
int32_t x2910 = x2907;
int32_t x2911 = x2910 + x2909;
float x2912 = x267[x2911];
float x2913 = x537[x2909];
float x2914 = x1683[x2908];
float x2915 = x2913 * x2914;
float x2916 = x2912 + x2915;
x267[x2911] = x2916;

}
x2907 += 150;

}
int32_t x2923 = 0;
for(int x2924=0; x2924 < 150; x2924++) {
for(int x2925=0; x2925 < 150; x2925++) {
float x2926 = x538[x2925];
int32_t x2927 = x2923;
int32_t x2928 = x2927 + x2925;
float x2929 = x137[x2928];
float x2930 = x1683[x2924];
float x2931 = x2929 * x2930;
float x2932 = x2926 + x2931;
x538[x2925] = x2932;

}
x2923 += 150;

}
// add_cartesian
int32_t x2940 = 0;
for(int x2941=0; x2941 < 150; x2941++) {
for(int x2942=0; x2942 < 150; x2942++) {
int32_t x2943 = x2940;
int32_t x2944 = x2943 + x2942;
float x2945 = x262[x2944];
float x2946 = x528[x2942];
float x2947 = x1661[x2941];
float x2948 = x2946 * x2947;
float x2949 = x2945 + x2948;
x262[x2944] = x2949;

}
x2940 += 150;

}
int32_t x2956 = 0;
for(int x2957=0; x2957 < 150; x2957++) {
for(int x2958=0; x2958 < 150; x2958++) {
float x2959 = x529[x2958];
int32_t x2960 = x2956;
int32_t x2961 = x2960 + x2958;
float x2962 = x129[x2961];
float x2963 = x1661[x2957];
float x2964 = x2962 * x2963;
float x2965 = x2959 + x2964;
x529[x2958] = x2965;

}
x2956 += 150;

}
for(int x2972=0; x2972 < 150; x2972++) {
float x2973 = x1622[x2972];
float x2974 = x1627[x2972];
float x2977 = x1639[x2972];
float x2975 = 1.0f - x2974;
float x2976 = x2975 * x2974;
float x2978 = x2976 * x2977;
float x2979 = x2973 + x2978;
x1622[x2972] = x2979;

}
int32_t x2983 = 0;
int32_t x2984 = 0;
int32_t x2985 = 0;
for(int x2986=0; x2986 < 150; x2986++) {
int32_t x2987 = x2983;
float x2988 = x1600[x2987];
float x2989 = x1583[x2987];
int32_t x2990 = x2984;
float x2991 = x145[x2990];
int32_t x2992 = x2985;
float x2993 = x1622[x2992];
float x2994 = x2988 + x2993;
x1600[x2987] = x2994;
float x2996 = x272[x2990];
float x2997 = x1583[x2987];
float x2998 = x145[x2990];
float x2999 = x1622[x2992];
float x3000 = x2996 + x2999;
x272[x2990] = x3000;
x2985 += 1;
x2983 += 1;
x2984 += 1;

}
int32_t x3007 = 0;
int32_t x3008 = 0;
int32_t x3009 = 0;
for(int x3010=0; x3010 < 150; x3010++) {
int32_t x3011 = x3007;
float x3012 = x1556[x3011];
float x3013 = x1540[x3011];
int32_t x3014 = x3008;
float x3015 = x1562[x3014];
int32_t x3016 = x3009;
float x3017 = x1600[x3016];
float x3018 = x3012 + x3017;
x1556[x3011] = x3018;
float x3020 = x1578[x3014];
float x3021 = x1540[x3011];
float x3022 = x1562[x3014];
float x3023 = x1600[x3016];
float x3024 = x3020 + x3023;
x1578[x3014] = x3024;
x3009 += 1;
x3007 += 1;
x3008 += 1;

}
// add_cartesian
int32_t x3032 = 0;
for(int x3033=0; x3033 < 150; x3033++) {
for(int x3034=0; x3034 < 150; x3034++) {
int32_t x3035 = x3032;
int32_t x3036 = x3035 + x3034;
float x3037 = x257[x3036];
float x3038 = x537[x3034];
float x3039 = x1578[x3033];
float x3040 = x3038 * x3039;
float x3041 = x3037 + x3040;
x257[x3036] = x3041;

}
x3032 += 150;

}
int32_t x3048 = 0;
for(int x3049=0; x3049 < 150; x3049++) {
for(int x3050=0; x3050 < 150; x3050++) {
float x3051 = x538[x3050];
int32_t x3052 = x3048;
int32_t x3053 = x3052 + x3050;
float x3054 = x121[x3053];
float x3055 = x1578[x3049];
float x3056 = x3054 * x3055;
float x3057 = x3051 + x3056;
x538[x3050] = x3057;

}
x3048 += 150;

}
// add_cartesian
int32_t x3065 = 0;
for(int x3066=0; x3066 < 150; x3066++) {
for(int x3067=0; x3067 < 150; x3067++) {
int32_t x3068 = x3065;
int32_t x3069 = x3068 + x3067;
float x3070 = x252[x3069];
float x3071 = x528[x3067];
float x3072 = x1556[x3066];
float x3073 = x3071 * x3072;
float x3074 = x3070 + x3073;
x252[x3069] = x3074;

}
x3065 += 150;

}
int32_t x3081 = 0;
for(int x3082=0; x3082 < 150; x3082++) {
for(int x3083=0; x3083 < 150; x3083++) {
float x3084 = x529[x3083];
int32_t x3085 = x3081;
int32_t x3086 = x3085 + x3083;
float x3087 = x113[x3086];
float x3088 = x1556[x3082];
float x3089 = x3087 * x3088;
float x3090 = x3084 + x3089;
x529[x3083] = x3090;

}
x3081 += 150;

}
for(int x3097=0; x3097 < 150; x3097++) {
float x3098 = x1517[x3097];
float x3099 = x1522[x3097];
float x3102 = x1534[x3097];
float x3100 = 1.0f - x3099;
float x3101 = x3100 * x3099;
float x3103 = x3101 * x3102;
float x3104 = x3098 + x3103;
x1517[x3097] = x3104;

}
int32_t x3108 = 0;
int32_t x3109 = 0;
int32_t x3110 = 0;
for(int x3111=0; x3111 < 150; x3111++) {
int32_t x3112 = x3108;
float x3113 = x1495[x3112];
float x3114 = x1478[x3112];
int32_t x3115 = x3109;
float x3116 = x108[x3115];
int32_t x3117 = x3110;
float x3118 = x1517[x3117];
float x3119 = x3113 + x3118;
x1495[x3112] = x3119;
float x3121 = x247[x3115];
float x3122 = x1478[x3112];
float x3123 = x108[x3115];
float x3124 = x1517[x3117];
float x3125 = x3121 + x3124;
x247[x3115] = x3125;
x3110 += 1;
x3108 += 1;
x3109 += 1;

}
int32_t x3132 = 0;
int32_t x3133 = 0;
int32_t x3134 = 0;
for(int x3135=0; x3135 < 150; x3135++) {
int32_t x3136 = x3132;
float x3137 = x1451[x3136];
float x3138 = x1435[x3136];
int32_t x3139 = x3133;
float x3140 = x1457[x3139];
int32_t x3141 = x3134;
float x3142 = x1495[x3141];
float x3143 = x3137 + x3142;
x1451[x3136] = x3143;
float x3145 = x1473[x3139];
float x3146 = x1435[x3136];
float x3147 = x1457[x3139];
float x3148 = x1495[x3141];
float x3149 = x3145 + x3148;
x1473[x3139] = x3149;
x3134 += 1;
x3132 += 1;
x3133 += 1;

}
// add_cartesian
int32_t x3157 = 0;
for(int x3158=0; x3158 < 150; x3158++) {
for(int x3159=0; x3159 < 150; x3159++) {
int32_t x3160 = x3157;
int32_t x3161 = x3160 + x3159;
float x3162 = x242[x3161];
float x3163 = x537[x3159];
float x3164 = x1473[x3158];
float x3165 = x3163 * x3164;
float x3166 = x3162 + x3165;
x242[x3161] = x3166;

}
x3157 += 150;

}
int32_t x3173 = 0;
for(int x3174=0; x3174 < 150; x3174++) {
for(int x3175=0; x3175 < 150; x3175++) {
float x3176 = x538[x3175];
int32_t x3177 = x3173;
int32_t x3178 = x3177 + x3175;
float x3179 = x100[x3178];
float x3180 = x1473[x3174];
float x3181 = x3179 * x3180;
float x3182 = x3176 + x3181;
x538[x3175] = x3182;

}
x3173 += 150;

}
// add_cartesian
int32_t x3190 = 0;
for(int x3191=0; x3191 < 150; x3191++) {
for(int x3192=0; x3192 < 150; x3192++) {
int32_t x3193 = x3190;
int32_t x3194 = x3193 + x3192;
float x3195 = x237[x3194];
float x3196 = x528[x3192];
float x3197 = x1451[x3191];
float x3198 = x3196 * x3197;
float x3199 = x3195 + x3198;
x237[x3194] = x3199;

}
x3190 += 150;

}
int32_t x3206 = 0;
for(int x3207=0; x3207 < 150; x3207++) {
for(int x3208=0; x3208 < 150; x3208++) {
float x3209 = x529[x3208];
int32_t x3210 = x3206;
int32_t x3211 = x3210 + x3208;
float x3212 = x91[x3211];
float x3213 = x1451[x3207];
float x3214 = x3212 * x3213;
float x3215 = x3209 + x3214;
x529[x3208] = x3215;

}
x3206 += 150;

}
}
};
x509(x532,x533,x3225);
};
x509(x523,x524,x3235);
} else {
float** x3262 = (float**)myMalloc(6 * sizeof(float*));;
x3262[0] = x479;
x3262[1] = x484;
x3262[2] = x489;
x3262[3] = x494;
x3262[4] = x499;
x3262[5] = x504;
function<void(float**)> x514 = x511;
function<void(float**)> x3245 = [&](float** x3246) {
float* x3247 = x3246[0];
float* x3248 = x3246[1];
float* x3249 = x3246[2];
float* x3250 = x3246[3];
float* x3251 = x3246[4];
float* x3252 = x3246[5];
float** x3253 = (float**)myMalloc(6 * sizeof(float*));;
x3253[0] = x3247;
x3253[1] = x3248;
x3253[2] = x3249;
x3253[3] = x3250;
x3253[4] = x3251;
x3253[5] = x3252;
x514(x3253);
};
x3245(x3262);
}
};
float* x463 = (float*)myMalloc(1 * sizeof(float));;
for(int x465=0; x465 < 1; x465++) {
x463[x465] = 0.0f;

}
float* x469 = (float*)myMalloc(1 * sizeof(float));;
for(int x470=0; x470 < 1; x470++) {
x469[x470] = 0.0f;

}
float* x474 = (float*)myMalloc(1 * sizeof(float));;
for(int x475=0; x475 < 1; x475++) {
x474[x475] = 0.0f;

}
for(int x480=0; x480 < 1; x480++) {
x479[x480] = 0.0f;

}
for(int x485=0; x485 < 1; x485++) {
x484[x485] = 0.0f;

}
for(int x490=0; x490 < 150; x490++) {
x489[x490] = 0.0f;

}
for(int x495=0; x495 < 150; x495++) {
x494[x495] = 0.0f;

}
for(int x500=0; x500 < 150; x500++) {
x499[x500] = 0.0f;

}
for(int x505=0; x505 < 150; x505++) {
x504[x505] = 0.0f;

}
float** x3286 = (float**)myMalloc(6 * sizeof(float*));;
x3286[0] = x479;
x3286[1] = x484;
x3286[2] = x489;
x3286[3] = x494;
x3286[4] = x499;
x3286[5] = x504;
function<void(float**)> x3273 = [&](float** x3274) {
float* x3275 = x3274[0];
float* x3276 = x3274[1];
float* x3277 = x3274[2];
float* x3278 = x3274[3];
float* x3279 = x3274[4];
float* x3280 = x3274[5];
float x3281 = x3276[0];
x3276[0] = 1.0f;
float x3283 = x3275[0];
x474[0] = x3283;
};
x509(0,x3273,x3286);
float x3295 = x474[0];
float x3296 = x452;
float x3297 = (float)x453;
float x3298 = x3296 * x3297;
int32_t x3299 = x453 + 1;
float x3300 = (float)x3299;
float x3301 = x3298 / x3300;
float x3302 = x3295 / x3300;
float x3303 = x3301 + x3302;
x452 = x3303;
for(int x3305=0; x3305 < 45000; x3305++) {
float x3306 = x207[x3305];
bool x3307 = x3306 > 5.0f;
if (x3307) {
x207[x3305] = 5.0f;
} else {
}
float x3311 = x207[x3305];
bool x3312 = x3311 < -5.0f;
if (x3312) {
x207[x3305] = -5.0f;
} else {
}

}
float* x3318 = (float*)myMalloc(45000 * sizeof(float));;
int32_t x3319 = 0;
int32_t x3320 = 0;
int32_t x3321 = 0;
for(int x3322=0; x3322 < 150; x3322++) {
int32_t x3323 = x3320;
int32_t x3324 = x3321;
int32_t x3325 = x3319;
int32_t x3326 = x3325;
int32_t x3327 = x3323;
int32_t x3328 = x3324;
for(int x3329=0; x3329 < 300; x3329++) {
int32_t x3330 = x3326;
int32_t x3331 = x3327;
float x3332 = x207[x3331];
int32_t x3333 = x3328;
float x3334 = x207[x3333];
float x3335 = x3332 * x3334;
x3318[x3330] = x3335;
x3326 += 1;
x3327 += 1;
x3328 += 1;

}
x3319 += 300;
x3320 += 300;
x3321 += 300;

}
for(int x3347=0; x3347 < 45000; x3347++) {
float x3348 = x337[x3347];
float x3349 = x3318[x3347];
float x3350 = x3348 + x3349;
x337[x3347] = x3350;

}
float* x3354 = (float*)myMalloc(45000 * sizeof(float));;
for(int x3355=0; x3355 < 45000; x3355++) {
float x3356 = x207[x3355];
float x3357 = x3356 * 0.05f;
x3354[x3355] = x3357;

}
float* x3361 = (float*)myMalloc(45000 * sizeof(float));;
for(int x3362=0; x3362 < 45000; x3362++) {
float x3363 = x337[x3362];
float x3364 = x3363 + 1.0E-8f;
x3361[x3362] = x3364;

}
float* x3368 = (float*)myMalloc(45000 * sizeof(float));;
for(int x3369=0; x3369 < 45000; x3369++) {
float x3370 = x3361[x3369];
double x3371 = (double)x3370;
double x3372 = sqrt(x3371);
float x3373 = (float)x3372;
x3368[x3369] = x3373;

}
float* x3377 = (float*)myMalloc(45000 * sizeof(float));;
int32_t x3378 = 0;
int32_t x3379 = 0;
int32_t x3380 = 0;
for(int x3381=0; x3381 < 150; x3381++) {
int32_t x3382 = x3379;
int32_t x3383 = x3380;
int32_t x3384 = x3378;
int32_t x3385 = x3384;
int32_t x3386 = x3382;
int32_t x3387 = x3383;
for(int x3388=0; x3388 < 300; x3388++) {
int32_t x3389 = x3385;
int32_t x3390 = x3386;
float x3391 = x3354[x3390];
int32_t x3392 = x3387;
float x3393 = x3368[x3392];
float x3394 = x3391 / x3393;
x3377[x3389] = x3394;
x3385 += 1;
x3386 += 1;
x3387 += 1;

}
x3378 += 300;
x3379 += 300;
x3380 += 300;

}
for(int x3406=0; x3406 < 45000; x3406++) {
float x3407 = x50[x3406];
float x3408 = x3377[x3406];
float x3409 = x3407 - x3408;
x50[x3406] = x3409;

}
for(int x3413=0; x3413 < 45000; x3413++) {
float x3414 = x207[x3413];
x207[x3413] = 0.0f;

}
for(int x3418=0; x3418 < 150; x3418++) {
float x3419 = x212[x3418];
bool x3420 = x3419 > 5.0f;
if (x3420) {
x212[x3418] = 5.0f;
} else {
}
float x3424 = x212[x3418];
bool x3425 = x3424 < -5.0f;
if (x3425) {
x212[x3418] = -5.0f;
} else {
}

}
float* x3431 = (float*)myMalloc(150 * sizeof(float));;
int32_t x3432 = 0;
int32_t x3433 = 0;
int32_t x3434 = 0;
for(int x3435=0; x3435 < 150; x3435++) {
int32_t x3436 = x3432;
int32_t x3437 = x3433;
float x3438 = x212[x3437];
int32_t x3439 = x3434;
float x3440 = x212[x3439];
float x3441 = x3438 * x3440;
x3431[x3436] = x3441;
x3432 += 1;
x3433 += 1;
x3434 += 1;

}
for(int x3448=0; x3448 < 150; x3448++) {
float x3449 = x342[x3448];
float x3450 = x3431[x3448];
float x3451 = x3449 + x3450;
x342[x3448] = x3451;

}
float* x3455 = (float*)myMalloc(150 * sizeof(float));;
for(int x3456=0; x3456 < 150; x3456++) {
float x3457 = x212[x3456];
float x3458 = x3457 * 0.05f;
x3455[x3456] = x3458;

}
float* x3462 = (float*)myMalloc(150 * sizeof(float));;
for(int x3463=0; x3463 < 150; x3463++) {
float x3464 = x342[x3463];
float x3465 = x3464 + 1.0E-8f;
x3462[x3463] = x3465;

}
float* x3469 = (float*)myMalloc(150 * sizeof(float));;
for(int x3470=0; x3470 < 150; x3470++) {
float x3471 = x3462[x3470];
double x3472 = (double)x3471;
double x3473 = sqrt(x3472);
float x3474 = (float)x3473;
x3469[x3470] = x3474;

}
float* x3478 = (float*)myMalloc(150 * sizeof(float));;
int32_t x3479 = 0;
int32_t x3480 = 0;
int32_t x3481 = 0;
for(int x3482=0; x3482 < 150; x3482++) {
int32_t x3483 = x3479;
int32_t x3484 = x3480;
float x3485 = x3455[x3484];
int32_t x3486 = x3481;
float x3487 = x3469[x3486];
float x3488 = x3485 / x3487;
x3478[x3483] = x3488;
x3479 += 1;
x3480 += 1;
x3481 += 1;

}
for(int x3495=0; x3495 < 150; x3495++) {
float x3496 = x59[x3495];
float x3497 = x3478[x3495];
float x3498 = x3496 - x3497;
x59[x3495] = x3498;

}
for(int x3502=0; x3502 < 150; x3502++) {
float x3503 = x212[x3502];
x212[x3502] = 0.0f;

}
for(int x3507=0; x3507 < 45000; x3507++) {
float x3508 = x217[x3507];
bool x3509 = x3508 > 5.0f;
if (x3509) {
x217[x3507] = 5.0f;
} else {
}
float x3513 = x217[x3507];
bool x3514 = x3513 < -5.0f;
if (x3514) {
x217[x3507] = -5.0f;
} else {
}

}
float* x3520 = (float*)myMalloc(45000 * sizeof(float));;
int32_t x3521 = 0;
int32_t x3522 = 0;
int32_t x3523 = 0;
for(int x3524=0; x3524 < 150; x3524++) {
int32_t x3525 = x3522;
int32_t x3526 = x3523;
int32_t x3527 = x3521;
int32_t x3528 = x3527;
int32_t x3529 = x3525;
int32_t x3530 = x3526;
for(int x3531=0; x3531 < 300; x3531++) {
int32_t x3532 = x3528;
int32_t x3533 = x3529;
float x3534 = x217[x3533];
int32_t x3535 = x3530;
float x3536 = x217[x3535];
float x3537 = x3534 * x3536;
x3520[x3532] = x3537;
x3528 += 1;
x3529 += 1;
x3530 += 1;

}
x3521 += 300;
x3522 += 300;
x3523 += 300;

}
for(int x3549=0; x3549 < 45000; x3549++) {
float x3550 = x347[x3549];
float x3551 = x3520[x3549];
float x3552 = x3550 + x3551;
x347[x3549] = x3552;

}
float* x3556 = (float*)myMalloc(45000 * sizeof(float));;
for(int x3557=0; x3557 < 45000; x3557++) {
float x3558 = x217[x3557];
float x3559 = x3558 * 0.05f;
x3556[x3557] = x3559;

}
float* x3563 = (float*)myMalloc(45000 * sizeof(float));;
for(int x3564=0; x3564 < 45000; x3564++) {
float x3565 = x347[x3564];
float x3566 = x3565 + 1.0E-8f;
x3563[x3564] = x3566;

}
float* x3570 = (float*)myMalloc(45000 * sizeof(float));;
for(int x3571=0; x3571 < 45000; x3571++) {
float x3572 = x3563[x3571];
double x3573 = (double)x3572;
double x3574 = sqrt(x3573);
float x3575 = (float)x3574;
x3570[x3571] = x3575;

}
float* x3579 = (float*)myMalloc(45000 * sizeof(float));;
int32_t x3580 = 0;
int32_t x3581 = 0;
int32_t x3582 = 0;
for(int x3583=0; x3583 < 150; x3583++) {
int32_t x3584 = x3581;
int32_t x3585 = x3582;
int32_t x3586 = x3580;
int32_t x3587 = x3586;
int32_t x3588 = x3584;
int32_t x3589 = x3585;
for(int x3590=0; x3590 < 300; x3590++) {
int32_t x3591 = x3587;
int32_t x3592 = x3588;
float x3593 = x3556[x3592];
int32_t x3594 = x3589;
float x3595 = x3570[x3594];
float x3596 = x3593 / x3595;
x3579[x3591] = x3596;
x3587 += 1;
x3588 += 1;
x3589 += 1;

}
x3580 += 300;
x3581 += 300;
x3582 += 300;

}
for(int x3608=0; x3608 < 45000; x3608++) {
float x3609 = x65[x3608];
float x3610 = x3579[x3608];
float x3611 = x3609 - x3610;
x65[x3608] = x3611;

}
for(int x3615=0; x3615 < 45000; x3615++) {
float x3616 = x217[x3615];
x217[x3615] = 0.0f;

}
for(int x3620=0; x3620 < 150; x3620++) {
float x3621 = x222[x3620];
bool x3622 = x3621 > 5.0f;
if (x3622) {
x222[x3620] = 5.0f;
} else {
}
float x3626 = x222[x3620];
bool x3627 = x3626 < -5.0f;
if (x3627) {
x222[x3620] = -5.0f;
} else {
}

}
float* x3633 = (float*)myMalloc(150 * sizeof(float));;
int32_t x3634 = 0;
int32_t x3635 = 0;
int32_t x3636 = 0;
for(int x3637=0; x3637 < 150; x3637++) {
int32_t x3638 = x3634;
int32_t x3639 = x3635;
float x3640 = x222[x3639];
int32_t x3641 = x3636;
float x3642 = x222[x3641];
float x3643 = x3640 * x3642;
x3633[x3638] = x3643;
x3634 += 1;
x3635 += 1;
x3636 += 1;

}
for(int x3650=0; x3650 < 150; x3650++) {
float x3651 = x352[x3650];
float x3652 = x3633[x3650];
float x3653 = x3651 + x3652;
x352[x3650] = x3653;

}
float* x3657 = (float*)myMalloc(150 * sizeof(float));;
for(int x3658=0; x3658 < 150; x3658++) {
float x3659 = x222[x3658];
float x3660 = x3659 * 0.05f;
x3657[x3658] = x3660;

}
float* x3664 = (float*)myMalloc(150 * sizeof(float));;
for(int x3665=0; x3665 < 150; x3665++) {
float x3666 = x352[x3665];
float x3667 = x3666 + 1.0E-8f;
x3664[x3665] = x3667;

}
float* x3671 = (float*)myMalloc(150 * sizeof(float));;
for(int x3672=0; x3672 < 150; x3672++) {
float x3673 = x3664[x3672];
double x3674 = (double)x3673;
double x3675 = sqrt(x3674);
float x3676 = (float)x3675;
x3671[x3672] = x3676;

}
float* x3680 = (float*)myMalloc(150 * sizeof(float));;
int32_t x3681 = 0;
int32_t x3682 = 0;
int32_t x3683 = 0;
for(int x3684=0; x3684 < 150; x3684++) {
int32_t x3685 = x3681;
int32_t x3686 = x3682;
float x3687 = x3657[x3686];
int32_t x3688 = x3683;
float x3689 = x3671[x3688];
float x3690 = x3687 / x3689;
x3680[x3685] = x3690;
x3681 += 1;
x3682 += 1;
x3683 += 1;

}
for(int x3697=0; x3697 < 150; x3697++) {
float x3698 = x73[x3697];
float x3699 = x3680[x3697];
float x3700 = x3698 - x3699;
x73[x3697] = x3700;

}
for(int x3704=0; x3704 < 150; x3704++) {
float x3705 = x222[x3704];
x222[x3704] = 0.0f;

}
for(int x3709=0; x3709 < 45000; x3709++) {
float x3710 = x227[x3709];
bool x3711 = x3710 > 5.0f;
if (x3711) {
x227[x3709] = 5.0f;
} else {
}
float x3715 = x227[x3709];
bool x3716 = x3715 < -5.0f;
if (x3716) {
x227[x3709] = -5.0f;
} else {
}

}
float* x3722 = (float*)myMalloc(45000 * sizeof(float));;
int32_t x3723 = 0;
int32_t x3724 = 0;
int32_t x3725 = 0;
for(int x3726=0; x3726 < 150; x3726++) {
int32_t x3727 = x3724;
int32_t x3728 = x3725;
int32_t x3729 = x3723;
int32_t x3730 = x3729;
int32_t x3731 = x3727;
int32_t x3732 = x3728;
for(int x3733=0; x3733 < 300; x3733++) {
int32_t x3734 = x3730;
int32_t x3735 = x3731;
float x3736 = x227[x3735];
int32_t x3737 = x3732;
float x3738 = x227[x3737];
float x3739 = x3736 * x3738;
x3722[x3734] = x3739;
x3730 += 1;
x3731 += 1;
x3732 += 1;

}
x3723 += 300;
x3724 += 300;
x3725 += 300;

}
for(int x3751=0; x3751 < 45000; x3751++) {
float x3752 = x357[x3751];
float x3753 = x3722[x3751];
float x3754 = x3752 + x3753;
x357[x3751] = x3754;

}
float* x3758 = (float*)myMalloc(45000 * sizeof(float));;
for(int x3759=0; x3759 < 45000; x3759++) {
float x3760 = x227[x3759];
float x3761 = x3760 * 0.05f;
x3758[x3759] = x3761;

}
float* x3765 = (float*)myMalloc(45000 * sizeof(float));;
for(int x3766=0; x3766 < 45000; x3766++) {
float x3767 = x357[x3766];
float x3768 = x3767 + 1.0E-8f;
x3765[x3766] = x3768;

}
float* x3772 = (float*)myMalloc(45000 * sizeof(float));;
for(int x3773=0; x3773 < 45000; x3773++) {
float x3774 = x3765[x3773];
double x3775 = (double)x3774;
double x3776 = sqrt(x3775);
float x3777 = (float)x3776;
x3772[x3773] = x3777;

}
float* x3781 = (float*)myMalloc(45000 * sizeof(float));;
int32_t x3782 = 0;
int32_t x3783 = 0;
int32_t x3784 = 0;
for(int x3785=0; x3785 < 150; x3785++) {
int32_t x3786 = x3783;
int32_t x3787 = x3784;
int32_t x3788 = x3782;
int32_t x3789 = x3788;
int32_t x3790 = x3786;
int32_t x3791 = x3787;
for(int x3792=0; x3792 < 300; x3792++) {
int32_t x3793 = x3789;
int32_t x3794 = x3790;
float x3795 = x3758[x3794];
int32_t x3796 = x3791;
float x3797 = x3772[x3796];
float x3798 = x3795 / x3797;
x3781[x3793] = x3798;
x3789 += 1;
x3790 += 1;
x3791 += 1;

}
x3782 += 300;
x3783 += 300;
x3784 += 300;

}
for(int x3810=0; x3810 < 45000; x3810++) {
float x3811 = x78[x3810];
float x3812 = x3781[x3810];
float x3813 = x3811 - x3812;
x78[x3810] = x3813;

}
for(int x3817=0; x3817 < 45000; x3817++) {
float x3818 = x227[x3817];
x227[x3817] = 0.0f;

}
for(int x3822=0; x3822 < 150; x3822++) {
float x3823 = x232[x3822];
bool x3824 = x3823 > 5.0f;
if (x3824) {
x232[x3822] = 5.0f;
} else {
}
float x3828 = x232[x3822];
bool x3829 = x3828 < -5.0f;
if (x3829) {
x232[x3822] = -5.0f;
} else {
}

}
float* x3835 = (float*)myMalloc(150 * sizeof(float));;
int32_t x3836 = 0;
int32_t x3837 = 0;
int32_t x3838 = 0;
for(int x3839=0; x3839 < 150; x3839++) {
int32_t x3840 = x3836;
int32_t x3841 = x3837;
float x3842 = x232[x3841];
int32_t x3843 = x3838;
float x3844 = x232[x3843];
float x3845 = x3842 * x3844;
x3835[x3840] = x3845;
x3836 += 1;
x3837 += 1;
x3838 += 1;

}
for(int x3852=0; x3852 < 150; x3852++) {
float x3853 = x362[x3852];
float x3854 = x3835[x3852];
float x3855 = x3853 + x3854;
x362[x3852] = x3855;

}
float* x3859 = (float*)myMalloc(150 * sizeof(float));;
for(int x3860=0; x3860 < 150; x3860++) {
float x3861 = x232[x3860];
float x3862 = x3861 * 0.05f;
x3859[x3860] = x3862;

}
float* x3866 = (float*)myMalloc(150 * sizeof(float));;
for(int x3867=0; x3867 < 150; x3867++) {
float x3868 = x362[x3867];
float x3869 = x3868 + 1.0E-8f;
x3866[x3867] = x3869;

}
float* x3873 = (float*)myMalloc(150 * sizeof(float));;
for(int x3874=0; x3874 < 150; x3874++) {
float x3875 = x3866[x3874];
double x3876 = (double)x3875;
double x3877 = sqrt(x3876);
float x3878 = (float)x3877;
x3873[x3874] = x3878;

}
float* x3882 = (float*)myMalloc(150 * sizeof(float));;
int32_t x3883 = 0;
int32_t x3884 = 0;
int32_t x3885 = 0;
for(int x3886=0; x3886 < 150; x3886++) {
int32_t x3887 = x3883;
int32_t x3888 = x3884;
float x3889 = x3859[x3888];
int32_t x3890 = x3885;
float x3891 = x3873[x3890];
float x3892 = x3889 / x3891;
x3882[x3887] = x3892;
x3883 += 1;
x3884 += 1;
x3885 += 1;

}
for(int x3899=0; x3899 < 150; x3899++) {
float x3900 = x86[x3899];
float x3901 = x3882[x3899];
float x3902 = x3900 - x3901;
x86[x3899] = x3902;

}
for(int x3906=0; x3906 < 150; x3906++) {
float x3907 = x232[x3906];
x232[x3906] = 0.0f;

}
for(int x3911=0; x3911 < 22500; x3911++) {
float x3912 = x237[x3911];
bool x3913 = x3912 > 5.0f;
if (x3913) {
x237[x3911] = 5.0f;
} else {
}
float x3917 = x237[x3911];
bool x3918 = x3917 < -5.0f;
if (x3918) {
x237[x3911] = -5.0f;
} else {
}

}
float* x3924 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x3925 = 0;
int32_t x3926 = 0;
int32_t x3927 = 0;
for(int x3928=0; x3928 < 150; x3928++) {
int32_t x3929 = x3926;
int32_t x3930 = x3927;
int32_t x3931 = x3925;
int32_t x3932 = x3931;
int32_t x3933 = x3929;
int32_t x3934 = x3930;
for(int x3935=0; x3935 < 150; x3935++) {
int32_t x3936 = x3932;
int32_t x3937 = x3933;
float x3938 = x237[x3937];
int32_t x3939 = x3934;
float x3940 = x237[x3939];
float x3941 = x3938 * x3940;
x3924[x3936] = x3941;
x3932 += 1;
x3933 += 1;
x3934 += 1;

}
x3925 += 150;
x3926 += 150;
x3927 += 150;

}
for(int x3953=0; x3953 < 22500; x3953++) {
float x3954 = x367[x3953];
float x3955 = x3924[x3953];
float x3956 = x3954 + x3955;
x367[x3953] = x3956;

}
float* x3960 = (float*)myMalloc(22500 * sizeof(float));;
for(int x3961=0; x3961 < 22500; x3961++) {
float x3962 = x237[x3961];
float x3963 = x3962 * 0.05f;
x3960[x3961] = x3963;

}
float* x3967 = (float*)myMalloc(22500 * sizeof(float));;
for(int x3968=0; x3968 < 22500; x3968++) {
float x3969 = x367[x3968];
float x3970 = x3969 + 1.0E-8f;
x3967[x3968] = x3970;

}
float* x3974 = (float*)myMalloc(22500 * sizeof(float));;
for(int x3975=0; x3975 < 22500; x3975++) {
float x3976 = x3967[x3975];
double x3977 = (double)x3976;
double x3978 = sqrt(x3977);
float x3979 = (float)x3978;
x3974[x3975] = x3979;

}
float* x3983 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x3984 = 0;
int32_t x3985 = 0;
int32_t x3986 = 0;
for(int x3987=0; x3987 < 150; x3987++) {
int32_t x3988 = x3985;
int32_t x3989 = x3986;
int32_t x3990 = x3984;
int32_t x3991 = x3990;
int32_t x3992 = x3988;
int32_t x3993 = x3989;
for(int x3994=0; x3994 < 150; x3994++) {
int32_t x3995 = x3991;
int32_t x3996 = x3992;
float x3997 = x3960[x3996];
int32_t x3998 = x3993;
float x3999 = x3974[x3998];
float x4000 = x3997 / x3999;
x3983[x3995] = x4000;
x3991 += 1;
x3992 += 1;
x3993 += 1;

}
x3984 += 150;
x3985 += 150;
x3986 += 150;

}
for(int x4012=0; x4012 < 22500; x4012++) {
float x4013 = x91[x4012];
float x4014 = x3983[x4012];
float x4015 = x4013 - x4014;
x91[x4012] = x4015;

}
for(int x4019=0; x4019 < 22500; x4019++) {
float x4020 = x237[x4019];
x237[x4019] = 0.0f;

}
for(int x4024=0; x4024 < 22500; x4024++) {
float x4025 = x242[x4024];
bool x4026 = x4025 > 5.0f;
if (x4026) {
x242[x4024] = 5.0f;
} else {
}
float x4030 = x242[x4024];
bool x4031 = x4030 < -5.0f;
if (x4031) {
x242[x4024] = -5.0f;
} else {
}

}
float* x4037 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x4038 = 0;
int32_t x4039 = 0;
int32_t x4040 = 0;
for(int x4041=0; x4041 < 150; x4041++) {
int32_t x4042 = x4039;
int32_t x4043 = x4040;
int32_t x4044 = x4038;
int32_t x4045 = x4044;
int32_t x4046 = x4042;
int32_t x4047 = x4043;
for(int x4048=0; x4048 < 150; x4048++) {
int32_t x4049 = x4045;
int32_t x4050 = x4046;
float x4051 = x242[x4050];
int32_t x4052 = x4047;
float x4053 = x242[x4052];
float x4054 = x4051 * x4053;
x4037[x4049] = x4054;
x4045 += 1;
x4046 += 1;
x4047 += 1;

}
x4038 += 150;
x4039 += 150;
x4040 += 150;

}
for(int x4066=0; x4066 < 22500; x4066++) {
float x4067 = x372[x4066];
float x4068 = x4037[x4066];
float x4069 = x4067 + x4068;
x372[x4066] = x4069;

}
float* x4073 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4074=0; x4074 < 22500; x4074++) {
float x4075 = x242[x4074];
float x4076 = x4075 * 0.05f;
x4073[x4074] = x4076;

}
float* x4080 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4081=0; x4081 < 22500; x4081++) {
float x4082 = x372[x4081];
float x4083 = x4082 + 1.0E-8f;
x4080[x4081] = x4083;

}
float* x4087 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4088=0; x4088 < 22500; x4088++) {
float x4089 = x4080[x4088];
double x4090 = (double)x4089;
double x4091 = sqrt(x4090);
float x4092 = (float)x4091;
x4087[x4088] = x4092;

}
float* x4096 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x4097 = 0;
int32_t x4098 = 0;
int32_t x4099 = 0;
for(int x4100=0; x4100 < 150; x4100++) {
int32_t x4101 = x4098;
int32_t x4102 = x4099;
int32_t x4103 = x4097;
int32_t x4104 = x4103;
int32_t x4105 = x4101;
int32_t x4106 = x4102;
for(int x4107=0; x4107 < 150; x4107++) {
int32_t x4108 = x4104;
int32_t x4109 = x4105;
float x4110 = x4073[x4109];
int32_t x4111 = x4106;
float x4112 = x4087[x4111];
float x4113 = x4110 / x4112;
x4096[x4108] = x4113;
x4104 += 1;
x4105 += 1;
x4106 += 1;

}
x4097 += 150;
x4098 += 150;
x4099 += 150;

}
for(int x4125=0; x4125 < 22500; x4125++) {
float x4126 = x100[x4125];
float x4127 = x4096[x4125];
float x4128 = x4126 - x4127;
x100[x4125] = x4128;

}
for(int x4132=0; x4132 < 22500; x4132++) {
float x4133 = x242[x4132];
x242[x4132] = 0.0f;

}
for(int x4137=0; x4137 < 150; x4137++) {
float x4138 = x247[x4137];
bool x4139 = x4138 > 5.0f;
if (x4139) {
x247[x4137] = 5.0f;
} else {
}
float x4143 = x247[x4137];
bool x4144 = x4143 < -5.0f;
if (x4144) {
x247[x4137] = -5.0f;
} else {
}

}
float* x4150 = (float*)myMalloc(150 * sizeof(float));;
int32_t x4151 = 0;
int32_t x4152 = 0;
int32_t x4153 = 0;
for(int x4154=0; x4154 < 150; x4154++) {
int32_t x4155 = x4151;
int32_t x4156 = x4152;
float x4157 = x247[x4156];
int32_t x4158 = x4153;
float x4159 = x247[x4158];
float x4160 = x4157 * x4159;
x4150[x4155] = x4160;
x4151 += 1;
x4152 += 1;
x4153 += 1;

}
for(int x4167=0; x4167 < 150; x4167++) {
float x4168 = x377[x4167];
float x4169 = x4150[x4167];
float x4170 = x4168 + x4169;
x377[x4167] = x4170;

}
float* x4174 = (float*)myMalloc(150 * sizeof(float));;
for(int x4175=0; x4175 < 150; x4175++) {
float x4176 = x247[x4175];
float x4177 = x4176 * 0.05f;
x4174[x4175] = x4177;

}
float* x4181 = (float*)myMalloc(150 * sizeof(float));;
for(int x4182=0; x4182 < 150; x4182++) {
float x4183 = x377[x4182];
float x4184 = x4183 + 1.0E-8f;
x4181[x4182] = x4184;

}
float* x4188 = (float*)myMalloc(150 * sizeof(float));;
for(int x4189=0; x4189 < 150; x4189++) {
float x4190 = x4181[x4189];
double x4191 = (double)x4190;
double x4192 = sqrt(x4191);
float x4193 = (float)x4192;
x4188[x4189] = x4193;

}
float* x4197 = (float*)myMalloc(150 * sizeof(float));;
int32_t x4198 = 0;
int32_t x4199 = 0;
int32_t x4200 = 0;
for(int x4201=0; x4201 < 150; x4201++) {
int32_t x4202 = x4198;
int32_t x4203 = x4199;
float x4204 = x4174[x4203];
int32_t x4205 = x4200;
float x4206 = x4188[x4205];
float x4207 = x4204 / x4206;
x4197[x4202] = x4207;
x4198 += 1;
x4199 += 1;
x4200 += 1;

}
for(int x4214=0; x4214 < 150; x4214++) {
float x4215 = x108[x4214];
float x4216 = x4197[x4214];
float x4217 = x4215 - x4216;
x108[x4214] = x4217;

}
for(int x4221=0; x4221 < 150; x4221++) {
float x4222 = x247[x4221];
x247[x4221] = 0.0f;

}
for(int x4226=0; x4226 < 22500; x4226++) {
float x4227 = x252[x4226];
bool x4228 = x4227 > 5.0f;
if (x4228) {
x252[x4226] = 5.0f;
} else {
}
float x4232 = x252[x4226];
bool x4233 = x4232 < -5.0f;
if (x4233) {
x252[x4226] = -5.0f;
} else {
}

}
float* x4239 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x4240 = 0;
int32_t x4241 = 0;
int32_t x4242 = 0;
for(int x4243=0; x4243 < 150; x4243++) {
int32_t x4244 = x4241;
int32_t x4245 = x4242;
int32_t x4246 = x4240;
int32_t x4247 = x4246;
int32_t x4248 = x4244;
int32_t x4249 = x4245;
for(int x4250=0; x4250 < 150; x4250++) {
int32_t x4251 = x4247;
int32_t x4252 = x4248;
float x4253 = x252[x4252];
int32_t x4254 = x4249;
float x4255 = x252[x4254];
float x4256 = x4253 * x4255;
x4239[x4251] = x4256;
x4247 += 1;
x4248 += 1;
x4249 += 1;

}
x4240 += 150;
x4241 += 150;
x4242 += 150;

}
for(int x4268=0; x4268 < 22500; x4268++) {
float x4269 = x382[x4268];
float x4270 = x4239[x4268];
float x4271 = x4269 + x4270;
x382[x4268] = x4271;

}
float* x4275 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4276=0; x4276 < 22500; x4276++) {
float x4277 = x252[x4276];
float x4278 = x4277 * 0.05f;
x4275[x4276] = x4278;

}
float* x4282 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4283=0; x4283 < 22500; x4283++) {
float x4284 = x382[x4283];
float x4285 = x4284 + 1.0E-8f;
x4282[x4283] = x4285;

}
float* x4289 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4290=0; x4290 < 22500; x4290++) {
float x4291 = x4282[x4290];
double x4292 = (double)x4291;
double x4293 = sqrt(x4292);
float x4294 = (float)x4293;
x4289[x4290] = x4294;

}
float* x4298 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x4299 = 0;
int32_t x4300 = 0;
int32_t x4301 = 0;
for(int x4302=0; x4302 < 150; x4302++) {
int32_t x4303 = x4300;
int32_t x4304 = x4301;
int32_t x4305 = x4299;
int32_t x4306 = x4305;
int32_t x4307 = x4303;
int32_t x4308 = x4304;
for(int x4309=0; x4309 < 150; x4309++) {
int32_t x4310 = x4306;
int32_t x4311 = x4307;
float x4312 = x4275[x4311];
int32_t x4313 = x4308;
float x4314 = x4289[x4313];
float x4315 = x4312 / x4314;
x4298[x4310] = x4315;
x4306 += 1;
x4307 += 1;
x4308 += 1;

}
x4299 += 150;
x4300 += 150;
x4301 += 150;

}
for(int x4327=0; x4327 < 22500; x4327++) {
float x4328 = x113[x4327];
float x4329 = x4298[x4327];
float x4330 = x4328 - x4329;
x113[x4327] = x4330;

}
for(int x4334=0; x4334 < 22500; x4334++) {
float x4335 = x252[x4334];
x252[x4334] = 0.0f;

}
for(int x4339=0; x4339 < 22500; x4339++) {
float x4340 = x257[x4339];
bool x4341 = x4340 > 5.0f;
if (x4341) {
x257[x4339] = 5.0f;
} else {
}
float x4345 = x257[x4339];
bool x4346 = x4345 < -5.0f;
if (x4346) {
x257[x4339] = -5.0f;
} else {
}

}
float* x4352 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x4353 = 0;
int32_t x4354 = 0;
int32_t x4355 = 0;
for(int x4356=0; x4356 < 150; x4356++) {
int32_t x4357 = x4354;
int32_t x4358 = x4355;
int32_t x4359 = x4353;
int32_t x4360 = x4359;
int32_t x4361 = x4357;
int32_t x4362 = x4358;
for(int x4363=0; x4363 < 150; x4363++) {
int32_t x4364 = x4360;
int32_t x4365 = x4361;
float x4366 = x257[x4365];
int32_t x4367 = x4362;
float x4368 = x257[x4367];
float x4369 = x4366 * x4368;
x4352[x4364] = x4369;
x4360 += 1;
x4361 += 1;
x4362 += 1;

}
x4353 += 150;
x4354 += 150;
x4355 += 150;

}
for(int x4381=0; x4381 < 22500; x4381++) {
float x4382 = x387[x4381];
float x4383 = x4352[x4381];
float x4384 = x4382 + x4383;
x387[x4381] = x4384;

}
float* x4388 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4389=0; x4389 < 22500; x4389++) {
float x4390 = x257[x4389];
float x4391 = x4390 * 0.05f;
x4388[x4389] = x4391;

}
float* x4395 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4396=0; x4396 < 22500; x4396++) {
float x4397 = x387[x4396];
float x4398 = x4397 + 1.0E-8f;
x4395[x4396] = x4398;

}
float* x4402 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4403=0; x4403 < 22500; x4403++) {
float x4404 = x4395[x4403];
double x4405 = (double)x4404;
double x4406 = sqrt(x4405);
float x4407 = (float)x4406;
x4402[x4403] = x4407;

}
float* x4411 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x4412 = 0;
int32_t x4413 = 0;
int32_t x4414 = 0;
for(int x4415=0; x4415 < 150; x4415++) {
int32_t x4416 = x4413;
int32_t x4417 = x4414;
int32_t x4418 = x4412;
int32_t x4419 = x4418;
int32_t x4420 = x4416;
int32_t x4421 = x4417;
for(int x4422=0; x4422 < 150; x4422++) {
int32_t x4423 = x4419;
int32_t x4424 = x4420;
float x4425 = x4388[x4424];
int32_t x4426 = x4421;
float x4427 = x4402[x4426];
float x4428 = x4425 / x4427;
x4411[x4423] = x4428;
x4419 += 1;
x4420 += 1;
x4421 += 1;

}
x4412 += 150;
x4413 += 150;
x4414 += 150;

}
for(int x4440=0; x4440 < 22500; x4440++) {
float x4441 = x121[x4440];
float x4442 = x4411[x4440];
float x4443 = x4441 - x4442;
x121[x4440] = x4443;

}
for(int x4447=0; x4447 < 22500; x4447++) {
float x4448 = x257[x4447];
x257[x4447] = 0.0f;

}
for(int x4452=0; x4452 < 22500; x4452++) {
float x4453 = x262[x4452];
bool x4454 = x4453 > 5.0f;
if (x4454) {
x262[x4452] = 5.0f;
} else {
}
float x4458 = x262[x4452];
bool x4459 = x4458 < -5.0f;
if (x4459) {
x262[x4452] = -5.0f;
} else {
}

}
float* x4465 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x4466 = 0;
int32_t x4467 = 0;
int32_t x4468 = 0;
for(int x4469=0; x4469 < 150; x4469++) {
int32_t x4470 = x4467;
int32_t x4471 = x4468;
int32_t x4472 = x4466;
int32_t x4473 = x4472;
int32_t x4474 = x4470;
int32_t x4475 = x4471;
for(int x4476=0; x4476 < 150; x4476++) {
int32_t x4477 = x4473;
int32_t x4478 = x4474;
float x4479 = x262[x4478];
int32_t x4480 = x4475;
float x4481 = x262[x4480];
float x4482 = x4479 * x4481;
x4465[x4477] = x4482;
x4473 += 1;
x4474 += 1;
x4475 += 1;

}
x4466 += 150;
x4467 += 150;
x4468 += 150;

}
for(int x4494=0; x4494 < 22500; x4494++) {
float x4495 = x392[x4494];
float x4496 = x4465[x4494];
float x4497 = x4495 + x4496;
x392[x4494] = x4497;

}
float* x4501 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4502=0; x4502 < 22500; x4502++) {
float x4503 = x262[x4502];
float x4504 = x4503 * 0.05f;
x4501[x4502] = x4504;

}
float* x4508 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4509=0; x4509 < 22500; x4509++) {
float x4510 = x392[x4509];
float x4511 = x4510 + 1.0E-8f;
x4508[x4509] = x4511;

}
float* x4515 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4516=0; x4516 < 22500; x4516++) {
float x4517 = x4508[x4516];
double x4518 = (double)x4517;
double x4519 = sqrt(x4518);
float x4520 = (float)x4519;
x4515[x4516] = x4520;

}
float* x4524 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x4525 = 0;
int32_t x4526 = 0;
int32_t x4527 = 0;
for(int x4528=0; x4528 < 150; x4528++) {
int32_t x4529 = x4526;
int32_t x4530 = x4527;
int32_t x4531 = x4525;
int32_t x4532 = x4531;
int32_t x4533 = x4529;
int32_t x4534 = x4530;
for(int x4535=0; x4535 < 150; x4535++) {
int32_t x4536 = x4532;
int32_t x4537 = x4533;
float x4538 = x4501[x4537];
int32_t x4539 = x4534;
float x4540 = x4515[x4539];
float x4541 = x4538 / x4540;
x4524[x4536] = x4541;
x4532 += 1;
x4533 += 1;
x4534 += 1;

}
x4525 += 150;
x4526 += 150;
x4527 += 150;

}
for(int x4553=0; x4553 < 22500; x4553++) {
float x4554 = x129[x4553];
float x4555 = x4524[x4553];
float x4556 = x4554 - x4555;
x129[x4553] = x4556;

}
for(int x4560=0; x4560 < 22500; x4560++) {
float x4561 = x262[x4560];
x262[x4560] = 0.0f;

}
for(int x4565=0; x4565 < 22500; x4565++) {
float x4566 = x267[x4565];
bool x4567 = x4566 > 5.0f;
if (x4567) {
x267[x4565] = 5.0f;
} else {
}
float x4571 = x267[x4565];
bool x4572 = x4571 < -5.0f;
if (x4572) {
x267[x4565] = -5.0f;
} else {
}

}
float* x4578 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x4579 = 0;
int32_t x4580 = 0;
int32_t x4581 = 0;
for(int x4582=0; x4582 < 150; x4582++) {
int32_t x4583 = x4580;
int32_t x4584 = x4581;
int32_t x4585 = x4579;
int32_t x4586 = x4585;
int32_t x4587 = x4583;
int32_t x4588 = x4584;
for(int x4589=0; x4589 < 150; x4589++) {
int32_t x4590 = x4586;
int32_t x4591 = x4587;
float x4592 = x267[x4591];
int32_t x4593 = x4588;
float x4594 = x267[x4593];
float x4595 = x4592 * x4594;
x4578[x4590] = x4595;
x4586 += 1;
x4587 += 1;
x4588 += 1;

}
x4579 += 150;
x4580 += 150;
x4581 += 150;

}
for(int x4607=0; x4607 < 22500; x4607++) {
float x4608 = x397[x4607];
float x4609 = x4578[x4607];
float x4610 = x4608 + x4609;
x397[x4607] = x4610;

}
float* x4614 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4615=0; x4615 < 22500; x4615++) {
float x4616 = x267[x4615];
float x4617 = x4616 * 0.05f;
x4614[x4615] = x4617;

}
float* x4621 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4622=0; x4622 < 22500; x4622++) {
float x4623 = x397[x4622];
float x4624 = x4623 + 1.0E-8f;
x4621[x4622] = x4624;

}
float* x4628 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4629=0; x4629 < 22500; x4629++) {
float x4630 = x4621[x4629];
double x4631 = (double)x4630;
double x4632 = sqrt(x4631);
float x4633 = (float)x4632;
x4628[x4629] = x4633;

}
float* x4637 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x4638 = 0;
int32_t x4639 = 0;
int32_t x4640 = 0;
for(int x4641=0; x4641 < 150; x4641++) {
int32_t x4642 = x4639;
int32_t x4643 = x4640;
int32_t x4644 = x4638;
int32_t x4645 = x4644;
int32_t x4646 = x4642;
int32_t x4647 = x4643;
for(int x4648=0; x4648 < 150; x4648++) {
int32_t x4649 = x4645;
int32_t x4650 = x4646;
float x4651 = x4614[x4650];
int32_t x4652 = x4647;
float x4653 = x4628[x4652];
float x4654 = x4651 / x4653;
x4637[x4649] = x4654;
x4645 += 1;
x4646 += 1;
x4647 += 1;

}
x4638 += 150;
x4639 += 150;
x4640 += 150;

}
for(int x4666=0; x4666 < 22500; x4666++) {
float x4667 = x137[x4666];
float x4668 = x4637[x4666];
float x4669 = x4667 - x4668;
x137[x4666] = x4669;

}
for(int x4673=0; x4673 < 22500; x4673++) {
float x4674 = x267[x4673];
x267[x4673] = 0.0f;

}
for(int x4678=0; x4678 < 150; x4678++) {
float x4679 = x272[x4678];
bool x4680 = x4679 > 5.0f;
if (x4680) {
x272[x4678] = 5.0f;
} else {
}
float x4684 = x272[x4678];
bool x4685 = x4684 < -5.0f;
if (x4685) {
x272[x4678] = -5.0f;
} else {
}

}
float* x4691 = (float*)myMalloc(150 * sizeof(float));;
int32_t x4692 = 0;
int32_t x4693 = 0;
int32_t x4694 = 0;
for(int x4695=0; x4695 < 150; x4695++) {
int32_t x4696 = x4692;
int32_t x4697 = x4693;
float x4698 = x272[x4697];
int32_t x4699 = x4694;
float x4700 = x272[x4699];
float x4701 = x4698 * x4700;
x4691[x4696] = x4701;
x4692 += 1;
x4693 += 1;
x4694 += 1;

}
for(int x4708=0; x4708 < 150; x4708++) {
float x4709 = x402[x4708];
float x4710 = x4691[x4708];
float x4711 = x4709 + x4710;
x402[x4708] = x4711;

}
float* x4715 = (float*)myMalloc(150 * sizeof(float));;
for(int x4716=0; x4716 < 150; x4716++) {
float x4717 = x272[x4716];
float x4718 = x4717 * 0.05f;
x4715[x4716] = x4718;

}
float* x4722 = (float*)myMalloc(150 * sizeof(float));;
for(int x4723=0; x4723 < 150; x4723++) {
float x4724 = x402[x4723];
float x4725 = x4724 + 1.0E-8f;
x4722[x4723] = x4725;

}
float* x4729 = (float*)myMalloc(150 * sizeof(float));;
for(int x4730=0; x4730 < 150; x4730++) {
float x4731 = x4722[x4730];
double x4732 = (double)x4731;
double x4733 = sqrt(x4732);
float x4734 = (float)x4733;
x4729[x4730] = x4734;

}
float* x4738 = (float*)myMalloc(150 * sizeof(float));;
int32_t x4739 = 0;
int32_t x4740 = 0;
int32_t x4741 = 0;
for(int x4742=0; x4742 < 150; x4742++) {
int32_t x4743 = x4739;
int32_t x4744 = x4740;
float x4745 = x4715[x4744];
int32_t x4746 = x4741;
float x4747 = x4729[x4746];
float x4748 = x4745 / x4747;
x4738[x4743] = x4748;
x4739 += 1;
x4740 += 1;
x4741 += 1;

}
for(int x4755=0; x4755 < 150; x4755++) {
float x4756 = x145[x4755];
float x4757 = x4738[x4755];
float x4758 = x4756 - x4757;
x145[x4755] = x4758;

}
for(int x4762=0; x4762 < 150; x4762++) {
float x4763 = x272[x4762];
x272[x4762] = 0.0f;

}
for(int x4767=0; x4767 < 22500; x4767++) {
float x4768 = x277[x4767];
bool x4769 = x4768 > 5.0f;
if (x4769) {
x277[x4767] = 5.0f;
} else {
}
float x4773 = x277[x4767];
bool x4774 = x4773 < -5.0f;
if (x4774) {
x277[x4767] = -5.0f;
} else {
}

}
float* x4780 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x4781 = 0;
int32_t x4782 = 0;
int32_t x4783 = 0;
for(int x4784=0; x4784 < 150; x4784++) {
int32_t x4785 = x4782;
int32_t x4786 = x4783;
int32_t x4787 = x4781;
int32_t x4788 = x4787;
int32_t x4789 = x4785;
int32_t x4790 = x4786;
for(int x4791=0; x4791 < 150; x4791++) {
int32_t x4792 = x4788;
int32_t x4793 = x4789;
float x4794 = x277[x4793];
int32_t x4795 = x4790;
float x4796 = x277[x4795];
float x4797 = x4794 * x4796;
x4780[x4792] = x4797;
x4788 += 1;
x4789 += 1;
x4790 += 1;

}
x4781 += 150;
x4782 += 150;
x4783 += 150;

}
for(int x4809=0; x4809 < 22500; x4809++) {
float x4810 = x407[x4809];
float x4811 = x4780[x4809];
float x4812 = x4810 + x4811;
x407[x4809] = x4812;

}
float* x4816 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4817=0; x4817 < 22500; x4817++) {
float x4818 = x277[x4817];
float x4819 = x4818 * 0.05f;
x4816[x4817] = x4819;

}
float* x4823 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4824=0; x4824 < 22500; x4824++) {
float x4825 = x407[x4824];
float x4826 = x4825 + 1.0E-8f;
x4823[x4824] = x4826;

}
float* x4830 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4831=0; x4831 < 22500; x4831++) {
float x4832 = x4823[x4831];
double x4833 = (double)x4832;
double x4834 = sqrt(x4833);
float x4835 = (float)x4834;
x4830[x4831] = x4835;

}
float* x4839 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x4840 = 0;
int32_t x4841 = 0;
int32_t x4842 = 0;
for(int x4843=0; x4843 < 150; x4843++) {
int32_t x4844 = x4841;
int32_t x4845 = x4842;
int32_t x4846 = x4840;
int32_t x4847 = x4846;
int32_t x4848 = x4844;
int32_t x4849 = x4845;
for(int x4850=0; x4850 < 150; x4850++) {
int32_t x4851 = x4847;
int32_t x4852 = x4848;
float x4853 = x4816[x4852];
int32_t x4854 = x4849;
float x4855 = x4830[x4854];
float x4856 = x4853 / x4855;
x4839[x4851] = x4856;
x4847 += 1;
x4848 += 1;
x4849 += 1;

}
x4840 += 150;
x4841 += 150;
x4842 += 150;

}
for(int x4868=0; x4868 < 22500; x4868++) {
float x4869 = x150[x4868];
float x4870 = x4839[x4868];
float x4871 = x4869 - x4870;
x150[x4868] = x4871;

}
for(int x4875=0; x4875 < 22500; x4875++) {
float x4876 = x277[x4875];
x277[x4875] = 0.0f;

}
for(int x4880=0; x4880 < 22500; x4880++) {
float x4881 = x282[x4880];
bool x4882 = x4881 > 5.0f;
if (x4882) {
x282[x4880] = 5.0f;
} else {
}
float x4886 = x282[x4880];
bool x4887 = x4886 < -5.0f;
if (x4887) {
x282[x4880] = -5.0f;
} else {
}

}
float* x4893 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x4894 = 0;
int32_t x4895 = 0;
int32_t x4896 = 0;
for(int x4897=0; x4897 < 150; x4897++) {
int32_t x4898 = x4895;
int32_t x4899 = x4896;
int32_t x4900 = x4894;
int32_t x4901 = x4900;
int32_t x4902 = x4898;
int32_t x4903 = x4899;
for(int x4904=0; x4904 < 150; x4904++) {
int32_t x4905 = x4901;
int32_t x4906 = x4902;
float x4907 = x282[x4906];
int32_t x4908 = x4903;
float x4909 = x282[x4908];
float x4910 = x4907 * x4909;
x4893[x4905] = x4910;
x4901 += 1;
x4902 += 1;
x4903 += 1;

}
x4894 += 150;
x4895 += 150;
x4896 += 150;

}
for(int x4922=0; x4922 < 22500; x4922++) {
float x4923 = x412[x4922];
float x4924 = x4893[x4922];
float x4925 = x4923 + x4924;
x412[x4922] = x4925;

}
float* x4929 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4930=0; x4930 < 22500; x4930++) {
float x4931 = x282[x4930];
float x4932 = x4931 * 0.05f;
x4929[x4930] = x4932;

}
float* x4936 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4937=0; x4937 < 22500; x4937++) {
float x4938 = x412[x4937];
float x4939 = x4938 + 1.0E-8f;
x4936[x4937] = x4939;

}
float* x4943 = (float*)myMalloc(22500 * sizeof(float));;
for(int x4944=0; x4944 < 22500; x4944++) {
float x4945 = x4936[x4944];
double x4946 = (double)x4945;
double x4947 = sqrt(x4946);
float x4948 = (float)x4947;
x4943[x4944] = x4948;

}
float* x4952 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x4953 = 0;
int32_t x4954 = 0;
int32_t x4955 = 0;
for(int x4956=0; x4956 < 150; x4956++) {
int32_t x4957 = x4954;
int32_t x4958 = x4955;
int32_t x4959 = x4953;
int32_t x4960 = x4959;
int32_t x4961 = x4957;
int32_t x4962 = x4958;
for(int x4963=0; x4963 < 150; x4963++) {
int32_t x4964 = x4960;
int32_t x4965 = x4961;
float x4966 = x4929[x4965];
int32_t x4967 = x4962;
float x4968 = x4943[x4967];
float x4969 = x4966 / x4968;
x4952[x4964] = x4969;
x4960 += 1;
x4961 += 1;
x4962 += 1;

}
x4953 += 150;
x4954 += 150;
x4955 += 150;

}
for(int x4981=0; x4981 < 22500; x4981++) {
float x4982 = x158[x4981];
float x4983 = x4952[x4981];
float x4984 = x4982 - x4983;
x158[x4981] = x4984;

}
for(int x4988=0; x4988 < 22500; x4988++) {
float x4989 = x282[x4988];
x282[x4988] = 0.0f;

}
for(int x4993=0; x4993 < 150; x4993++) {
float x4994 = x287[x4993];
bool x4995 = x4994 > 5.0f;
if (x4995) {
x287[x4993] = 5.0f;
} else {
}
float x4999 = x287[x4993];
bool x5000 = x4999 < -5.0f;
if (x5000) {
x287[x4993] = -5.0f;
} else {
}

}
float* x5006 = (float*)myMalloc(150 * sizeof(float));;
int32_t x5007 = 0;
int32_t x5008 = 0;
int32_t x5009 = 0;
for(int x5010=0; x5010 < 150; x5010++) {
int32_t x5011 = x5007;
int32_t x5012 = x5008;
float x5013 = x287[x5012];
int32_t x5014 = x5009;
float x5015 = x287[x5014];
float x5016 = x5013 * x5015;
x5006[x5011] = x5016;
x5007 += 1;
x5008 += 1;
x5009 += 1;

}
for(int x5023=0; x5023 < 150; x5023++) {
float x5024 = x417[x5023];
float x5025 = x5006[x5023];
float x5026 = x5024 + x5025;
x417[x5023] = x5026;

}
float* x5030 = (float*)myMalloc(150 * sizeof(float));;
for(int x5031=0; x5031 < 150; x5031++) {
float x5032 = x287[x5031];
float x5033 = x5032 * 0.05f;
x5030[x5031] = x5033;

}
float* x5037 = (float*)myMalloc(150 * sizeof(float));;
for(int x5038=0; x5038 < 150; x5038++) {
float x5039 = x417[x5038];
float x5040 = x5039 + 1.0E-8f;
x5037[x5038] = x5040;

}
float* x5044 = (float*)myMalloc(150 * sizeof(float));;
for(int x5045=0; x5045 < 150; x5045++) {
float x5046 = x5037[x5045];
double x5047 = (double)x5046;
double x5048 = sqrt(x5047);
float x5049 = (float)x5048;
x5044[x5045] = x5049;

}
float* x5053 = (float*)myMalloc(150 * sizeof(float));;
int32_t x5054 = 0;
int32_t x5055 = 0;
int32_t x5056 = 0;
for(int x5057=0; x5057 < 150; x5057++) {
int32_t x5058 = x5054;
int32_t x5059 = x5055;
float x5060 = x5030[x5059];
int32_t x5061 = x5056;
float x5062 = x5044[x5061];
float x5063 = x5060 / x5062;
x5053[x5058] = x5063;
x5054 += 1;
x5055 += 1;
x5056 += 1;

}
for(int x5070=0; x5070 < 150; x5070++) {
float x5071 = x166[x5070];
float x5072 = x5053[x5070];
float x5073 = x5071 - x5072;
x166[x5070] = x5073;

}
for(int x5077=0; x5077 < 150; x5077++) {
float x5078 = x287[x5077];
x287[x5077] = 0.0f;

}
for(int x5082=0; x5082 < 22500; x5082++) {
float x5083 = x292[x5082];
bool x5084 = x5083 > 5.0f;
if (x5084) {
x292[x5082] = 5.0f;
} else {
}
float x5088 = x292[x5082];
bool x5089 = x5088 < -5.0f;
if (x5089) {
x292[x5082] = -5.0f;
} else {
}

}
float* x5095 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x5096 = 0;
int32_t x5097 = 0;
int32_t x5098 = 0;
for(int x5099=0; x5099 < 150; x5099++) {
int32_t x5100 = x5097;
int32_t x5101 = x5098;
int32_t x5102 = x5096;
int32_t x5103 = x5102;
int32_t x5104 = x5100;
int32_t x5105 = x5101;
for(int x5106=0; x5106 < 150; x5106++) {
int32_t x5107 = x5103;
int32_t x5108 = x5104;
float x5109 = x292[x5108];
int32_t x5110 = x5105;
float x5111 = x292[x5110];
float x5112 = x5109 * x5111;
x5095[x5107] = x5112;
x5103 += 1;
x5104 += 1;
x5105 += 1;

}
x5096 += 150;
x5097 += 150;
x5098 += 150;

}
for(int x5124=0; x5124 < 22500; x5124++) {
float x5125 = x422[x5124];
float x5126 = x5095[x5124];
float x5127 = x5125 + x5126;
x422[x5124] = x5127;

}
float* x5131 = (float*)myMalloc(22500 * sizeof(float));;
for(int x5132=0; x5132 < 22500; x5132++) {
float x5133 = x292[x5132];
float x5134 = x5133 * 0.05f;
x5131[x5132] = x5134;

}
float* x5138 = (float*)myMalloc(22500 * sizeof(float));;
for(int x5139=0; x5139 < 22500; x5139++) {
float x5140 = x422[x5139];
float x5141 = x5140 + 1.0E-8f;
x5138[x5139] = x5141;

}
float* x5145 = (float*)myMalloc(22500 * sizeof(float));;
for(int x5146=0; x5146 < 22500; x5146++) {
float x5147 = x5138[x5146];
double x5148 = (double)x5147;
double x5149 = sqrt(x5148);
float x5150 = (float)x5149;
x5145[x5146] = x5150;

}
float* x5154 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x5155 = 0;
int32_t x5156 = 0;
int32_t x5157 = 0;
for(int x5158=0; x5158 < 150; x5158++) {
int32_t x5159 = x5156;
int32_t x5160 = x5157;
int32_t x5161 = x5155;
int32_t x5162 = x5161;
int32_t x5163 = x5159;
int32_t x5164 = x5160;
for(int x5165=0; x5165 < 150; x5165++) {
int32_t x5166 = x5162;
int32_t x5167 = x5163;
float x5168 = x5131[x5167];
int32_t x5169 = x5164;
float x5170 = x5145[x5169];
float x5171 = x5168 / x5170;
x5154[x5166] = x5171;
x5162 += 1;
x5163 += 1;
x5164 += 1;

}
x5155 += 150;
x5156 += 150;
x5157 += 150;

}
for(int x5183=0; x5183 < 22500; x5183++) {
float x5184 = x171[x5183];
float x5185 = x5154[x5183];
float x5186 = x5184 - x5185;
x171[x5183] = x5186;

}
for(int x5190=0; x5190 < 22500; x5190++) {
float x5191 = x292[x5190];
x292[x5190] = 0.0f;

}
for(int x5195=0; x5195 < 22500; x5195++) {
float x5196 = x297[x5195];
bool x5197 = x5196 > 5.0f;
if (x5197) {
x297[x5195] = 5.0f;
} else {
}
float x5201 = x297[x5195];
bool x5202 = x5201 < -5.0f;
if (x5202) {
x297[x5195] = -5.0f;
} else {
}

}
float* x5208 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x5209 = 0;
int32_t x5210 = 0;
int32_t x5211 = 0;
for(int x5212=0; x5212 < 150; x5212++) {
int32_t x5213 = x5210;
int32_t x5214 = x5211;
int32_t x5215 = x5209;
int32_t x5216 = x5215;
int32_t x5217 = x5213;
int32_t x5218 = x5214;
for(int x5219=0; x5219 < 150; x5219++) {
int32_t x5220 = x5216;
int32_t x5221 = x5217;
float x5222 = x297[x5221];
int32_t x5223 = x5218;
float x5224 = x297[x5223];
float x5225 = x5222 * x5224;
x5208[x5220] = x5225;
x5216 += 1;
x5217 += 1;
x5218 += 1;

}
x5209 += 150;
x5210 += 150;
x5211 += 150;

}
for(int x5237=0; x5237 < 22500; x5237++) {
float x5238 = x427[x5237];
float x5239 = x5208[x5237];
float x5240 = x5238 + x5239;
x427[x5237] = x5240;

}
float* x5244 = (float*)myMalloc(22500 * sizeof(float));;
for(int x5245=0; x5245 < 22500; x5245++) {
float x5246 = x297[x5245];
float x5247 = x5246 * 0.05f;
x5244[x5245] = x5247;

}
float* x5251 = (float*)myMalloc(22500 * sizeof(float));;
for(int x5252=0; x5252 < 22500; x5252++) {
float x5253 = x427[x5252];
float x5254 = x5253 + 1.0E-8f;
x5251[x5252] = x5254;

}
float* x5258 = (float*)myMalloc(22500 * sizeof(float));;
for(int x5259=0; x5259 < 22500; x5259++) {
float x5260 = x5251[x5259];
double x5261 = (double)x5260;
double x5262 = sqrt(x5261);
float x5263 = (float)x5262;
x5258[x5259] = x5263;

}
float* x5267 = (float*)myMalloc(22500 * sizeof(float));;
int32_t x5268 = 0;
int32_t x5269 = 0;
int32_t x5270 = 0;
for(int x5271=0; x5271 < 150; x5271++) {
int32_t x5272 = x5269;
int32_t x5273 = x5270;
int32_t x5274 = x5268;
int32_t x5275 = x5274;
int32_t x5276 = x5272;
int32_t x5277 = x5273;
for(int x5278=0; x5278 < 150; x5278++) {
int32_t x5279 = x5275;
int32_t x5280 = x5276;
float x5281 = x5244[x5280];
int32_t x5282 = x5277;
float x5283 = x5258[x5282];
float x5284 = x5281 / x5283;
x5267[x5279] = x5284;
x5275 += 1;
x5276 += 1;
x5277 += 1;

}
x5268 += 150;
x5269 += 150;
x5270 += 150;

}
for(int x5296=0; x5296 < 22500; x5296++) {
float x5297 = x179[x5296];
float x5298 = x5267[x5296];
float x5299 = x5297 - x5298;
x179[x5296] = x5299;

}
for(int x5303=0; x5303 < 22500; x5303++) {
float x5304 = x297[x5303];
x297[x5303] = 0.0f;

}
for(int x5308=0; x5308 < 150; x5308++) {
float x5309 = x302[x5308];
bool x5310 = x5309 > 5.0f;
if (x5310) {
x302[x5308] = 5.0f;
} else {
}
float x5314 = x302[x5308];
bool x5315 = x5314 < -5.0f;
if (x5315) {
x302[x5308] = -5.0f;
} else {
}

}
float* x5321 = (float*)myMalloc(150 * sizeof(float));;
int32_t x5322 = 0;
int32_t x5323 = 0;
int32_t x5324 = 0;
for(int x5325=0; x5325 < 150; x5325++) {
int32_t x5326 = x5322;
int32_t x5327 = x5323;
float x5328 = x302[x5327];
int32_t x5329 = x5324;
float x5330 = x302[x5329];
float x5331 = x5328 * x5330;
x5321[x5326] = x5331;
x5322 += 1;
x5323 += 1;
x5324 += 1;

}
for(int x5338=0; x5338 < 150; x5338++) {
float x5339 = x432[x5338];
float x5340 = x5321[x5338];
float x5341 = x5339 + x5340;
x432[x5338] = x5341;

}
float* x5345 = (float*)myMalloc(150 * sizeof(float));;
for(int x5346=0; x5346 < 150; x5346++) {
float x5347 = x302[x5346];
float x5348 = x5347 * 0.05f;
x5345[x5346] = x5348;

}
float* x5352 = (float*)myMalloc(150 * sizeof(float));;
for(int x5353=0; x5353 < 150; x5353++) {
float x5354 = x432[x5353];
float x5355 = x5354 + 1.0E-8f;
x5352[x5353] = x5355;

}
float* x5359 = (float*)myMalloc(150 * sizeof(float));;
for(int x5360=0; x5360 < 150; x5360++) {
float x5361 = x5352[x5360];
double x5362 = (double)x5361;
double x5363 = sqrt(x5362);
float x5364 = (float)x5363;
x5359[x5360] = x5364;

}
float* x5368 = (float*)myMalloc(150 * sizeof(float));;
int32_t x5369 = 0;
int32_t x5370 = 0;
int32_t x5371 = 0;
for(int x5372=0; x5372 < 150; x5372++) {
int32_t x5373 = x5369;
int32_t x5374 = x5370;
float x5375 = x5345[x5374];
int32_t x5376 = x5371;
float x5377 = x5359[x5376];
float x5378 = x5375 / x5377;
x5368[x5373] = x5378;
x5369 += 1;
x5370 += 1;
x5371 += 1;

}
for(int x5385=0; x5385 < 150; x5385++) {
float x5386 = x187[x5385];
float x5387 = x5368[x5385];
float x5388 = x5386 - x5387;
x187[x5385] = x5388;

}
for(int x5392=0; x5392 < 150; x5392++) {
float x5393 = x302[x5392];
x302[x5392] = 0.0f;

}
for(int x5397=0; x5397 < 750; x5397++) {
float x5398 = x307[x5397];
bool x5399 = x5398 > 5.0f;
if (x5399) {
x307[x5397] = 5.0f;
} else {
}
float x5403 = x307[x5397];
bool x5404 = x5403 < -5.0f;
if (x5404) {
x307[x5397] = -5.0f;
} else {
}

}
float* x5410 = (float*)myMalloc(750 * sizeof(float));;
int32_t x5411 = 0;
int32_t x5412 = 0;
int32_t x5413 = 0;
for(int x5414=0; x5414 < 5; x5414++) {
int32_t x5415 = x5412;
int32_t x5416 = x5413;
int32_t x5417 = x5411;
int32_t x5418 = x5417;
int32_t x5419 = x5415;
int32_t x5420 = x5416;
for(int x5421=0; x5421 < 150; x5421++) {
int32_t x5422 = x5418;
int32_t x5423 = x5419;
float x5424 = x307[x5423];
int32_t x5425 = x5420;
float x5426 = x307[x5425];
float x5427 = x5424 * x5426;
x5410[x5422] = x5427;
x5418 += 1;
x5419 += 1;
x5420 += 1;

}
x5411 += 150;
x5412 += 150;
x5413 += 150;

}
for(int x5439=0; x5439 < 750; x5439++) {
float x5440 = x437[x5439];
float x5441 = x5410[x5439];
float x5442 = x5440 + x5441;
x437[x5439] = x5442;

}
float* x5446 = (float*)myMalloc(750 * sizeof(float));;
for(int x5447=0; x5447 < 750; x5447++) {
float x5448 = x307[x5447];
float x5449 = x5448 * 0.05f;
x5446[x5447] = x5449;

}
float* x5453 = (float*)myMalloc(750 * sizeof(float));;
for(int x5454=0; x5454 < 750; x5454++) {
float x5455 = x437[x5454];
float x5456 = x5455 + 1.0E-8f;
x5453[x5454] = x5456;

}
float* x5460 = (float*)myMalloc(750 * sizeof(float));;
for(int x5461=0; x5461 < 750; x5461++) {
float x5462 = x5453[x5461];
double x5463 = (double)x5462;
double x5464 = sqrt(x5463);
float x5465 = (float)x5464;
x5460[x5461] = x5465;

}
float* x5469 = (float*)myMalloc(750 * sizeof(float));;
int32_t x5470 = 0;
int32_t x5471 = 0;
int32_t x5472 = 0;
for(int x5473=0; x5473 < 5; x5473++) {
int32_t x5474 = x5471;
int32_t x5475 = x5472;
int32_t x5476 = x5470;
int32_t x5477 = x5476;
int32_t x5478 = x5474;
int32_t x5479 = x5475;
for(int x5480=0; x5480 < 150; x5480++) {
int32_t x5481 = x5477;
int32_t x5482 = x5478;
float x5483 = x5446[x5482];
int32_t x5484 = x5479;
float x5485 = x5460[x5484];
float x5486 = x5483 / x5485;
x5469[x5481] = x5486;
x5477 += 1;
x5478 += 1;
x5479 += 1;

}
x5470 += 150;
x5471 += 150;
x5472 += 150;

}
for(int x5498=0; x5498 < 750; x5498++) {
float x5499 = x192[x5498];
float x5500 = x5469[x5498];
float x5501 = x5499 - x5500;
x192[x5498] = x5501;

}
for(int x5505=0; x5505 < 750; x5505++) {
float x5506 = x307[x5505];
x307[x5505] = 0.0f;

}
for(int x5510=0; x5510 < 5; x5510++) {
float x5511 = x312[x5510];
bool x5512 = x5511 > 5.0f;
if (x5512) {
x312[x5510] = 5.0f;
} else {
}
float x5516 = x312[x5510];
bool x5517 = x5516 < -5.0f;
if (x5517) {
x312[x5510] = -5.0f;
} else {
}

}
float* x5523 = (float*)myMalloc(5 * sizeof(float));;
int32_t x5524 = 0;
int32_t x5525 = 0;
int32_t x5526 = 0;
for(int x5527=0; x5527 < 5; x5527++) {
int32_t x5528 = x5524;
int32_t x5529 = x5525;
float x5530 = x312[x5529];
int32_t x5531 = x5526;
float x5532 = x312[x5531];
float x5533 = x5530 * x5532;
x5523[x5528] = x5533;
x5524 += 1;
x5525 += 1;
x5526 += 1;

}
for(int x5540=0; x5540 < 5; x5540++) {
float x5541 = x442[x5540];
float x5542 = x5523[x5540];
float x5543 = x5541 + x5542;
x442[x5540] = x5543;

}
float* x5547 = (float*)myMalloc(5 * sizeof(float));;
for(int x5548=0; x5548 < 5; x5548++) {
float x5549 = x312[x5548];
float x5550 = x5549 * 0.05f;
x5547[x5548] = x5550;

}
float* x5554 = (float*)myMalloc(5 * sizeof(float));;
for(int x5555=0; x5555 < 5; x5555++) {
float x5556 = x442[x5555];
float x5557 = x5556 + 1.0E-8f;
x5554[x5555] = x5557;

}
float* x5561 = (float*)myMalloc(5 * sizeof(float));;
for(int x5562=0; x5562 < 5; x5562++) {
float x5563 = x5554[x5562];
double x5564 = (double)x5563;
double x5565 = sqrt(x5564);
float x5566 = (float)x5565;
x5561[x5562] = x5566;

}
float* x5570 = (float*)myMalloc(5 * sizeof(float));;
int32_t x5571 = 0;
int32_t x5572 = 0;
int32_t x5573 = 0;
for(int x5574=0; x5574 < 5; x5574++) {
int32_t x5575 = x5571;
int32_t x5576 = x5572;
float x5577 = x5547[x5576];
int32_t x5578 = x5573;
float x5579 = x5561[x5578];
float x5580 = x5577 / x5579;
x5570[x5575] = x5580;
x5571 += 1;
x5572 += 1;
x5573 += 1;

}
for(int x5587=0; x5587 < 5; x5587++) {
float x5588 = x201[x5587];
float x5589 = x5570[x5587];
float x5590 = x5588 - x5589;
x201[x5587] = x5590;

}
for(int x5594=0; x5594 < 5; x5594++) {
float x5595 = x312[x5594];
x312[x5594] = 0.0f;

}
mallocAddr = (void*)x448;

}
float x5602 = x452;
double x5603 = (double)x5602;
x447[x451] = x5603;
double x5605 = ((double)clock() / CLOCKS_PER_SEC);
double x5606 = x5605 - x449;
printf("epoc %d, average_loss %f, time %lf\n",x451,x5602,x5606);

}
double x5610 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x5614 = (long)fopen(x0, "w");
fprintf((FILE *)x5614, "unit: %s\n", "1 epoch");
for(int x5616=0; x5616 < 30; x5616++) {
double x5617 = x447[x5616];
fprintf((FILE *)x5614, "%lf\n", x5617);

}
double x5611 = x449 - x2;
double x5612 = x5610 - x449;
double x5613 = x5612 / 30.0;
fprintf((FILE *)x5614, "run time: %lf %lf\n", x5611, x5613);
fclose((FILE*)x5614);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

