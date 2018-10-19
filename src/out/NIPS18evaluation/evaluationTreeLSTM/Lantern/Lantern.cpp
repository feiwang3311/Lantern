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
  float* x317 = (float*)myMalloc(45000 * sizeof(float));;
  for(int x318=0; x318 < 45000; x318++) {
    x317[x318] = 0.0f;

  }
  float* x322 = (float*)myMalloc(150 * sizeof(float));;
  for(int x323=0; x323 < 150; x323++) {
    x322[x323] = 0.0f;

  }
  float* x327 = (float*)myMalloc(45000 * sizeof(float));;
  for(int x328=0; x328 < 45000; x328++) {
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
  float* x347 = (float*)myMalloc(22500 * sizeof(float));;
  for(int x348=0; x348 < 22500; x348++) {
    x347[x348] = 0.0f;

  }
  float* x352 = (float*)myMalloc(22500 * sizeof(float));;
  for(int x353=0; x353 < 22500; x353++) {
    x352[x353] = 0.0f;

  }
  float* x357 = (float*)myMalloc(150 * sizeof(float));;
  for(int x358=0; x358 < 150; x358++) {
    x357[x358] = 0.0f;

  }
  float* x362 = (float*)myMalloc(22500 * sizeof(float));;
  for(int x363=0; x363 < 22500; x363++) {
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
  float* x377 = (float*)myMalloc(22500 * sizeof(float));;
  for(int x378=0; x378 < 22500; x378++) {
    x377[x378] = 0.0f;

  }
  float* x382 = (float*)myMalloc(150 * sizeof(float));;
  for(int x383=0; x383 < 150; x383++) {
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
  float* x397 = (float*)myMalloc(150 * sizeof(float));;
  for(int x398=0; x398 < 150; x398++) {
    x397[x398] = 0.0f;

  }
  float* x402 = (float*)myMalloc(22500 * sizeof(float));;
  for(int x403=0; x403 < 22500; x403++) {
    x402[x403] = 0.0f;

  }
  float* x407 = (float*)myMalloc(22500 * sizeof(float));;
  for(int x408=0; x408 < 22500; x408++) {
    x407[x408] = 0.0f;

  }
  float* x412 = (float*)myMalloc(150 * sizeof(float));;
  for(int x413=0; x413 < 150; x413++) {
    x412[x413] = 0.0f;

  }
  float* x417 = (float*)myMalloc(750 * sizeof(float));;
  for(int x418=0; x418 < 750; x418++) {
    x417[x418] = 0.0f;

  }
  float* x422 = (float*)myMalloc(5 * sizeof(float));;
  for(int x423=0; x423 < 5; x423++) {
    x422[x423] = 0.0f;

  }
  double* x427 = (double*)myMalloc(6 * sizeof(double));;
  int64_t x428 = (long)mallocAddr;
  double x429 = ((double)clock() / CLOCKS_PER_SEC);
  for(int x431=0; x431 < 6; x431++) {
    float x432 = 0.0f;
    for(int x433=0; x433 < x24; x433++) {
      float* x459 = (float*)myMalloc(1 * sizeof(float));;
      float* x464 = (float*)myMalloc(1 * sizeof(float));;
      float* x469 = (float*)myMalloc(150 * sizeof(float));;
      float* x474 = (float*)myMalloc(150 * sizeof(float));;
      float* x479 = (float*)myMalloc(150 * sizeof(float));;
      float* x484 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x434 = x433 % x24;
      int32_t x435 = x434 * 4;
      int* x436 = x26[x435];
      int32_t x437 = x435 + 1;
      int* x438 = x26[x437];
      int32_t x439 = x435 + 2;
      int* x440 = x26[x439];
      int32_t x441 = x435 + 3;
      int* x442 = x26[x441];
      function<void(int32_t,function<void(float**)>,float**)> x489 = [&](int32_t x490,function<void(float**)> x491,float** x492) {
        float** x495 = x492;
        float* x496 = x495[0];
        float* x497 = x495[1];
        float* x498 = x495[2];
        float* x499 = x495[3];
        float* x500 = x495[4];
        float* x501 = x495[5];
        int32_t x493 = x490;
        bool x502 = x493 >= 0;
        if (x502) {
          int32_t x503 = x440[x493];
          float** x3215 = (float**)myMalloc(6 * sizeof(float*));;
          x3215[0] = x459;
          x3215[1] = x464;
          x3215[2] = x469;
          x3215[3] = x474;
          x3215[4] = x479;
          x3215[5] = x484;
          function<void(float**)> x494 = x491;
          function<void(float**)> x945 = [&](float** x946) {
            float* x947 = x946[0];
            float* x948 = x946[1];
            float* x949 = x946[2];
            float* x950 = x946[3];
            float* x951 = x946[4];
            float* x952 = x946[5];
            float** x953 = (float**)myMalloc(6 * sizeof(float*));;
            x953[0] = x947;
            x953[1] = x948;
            x953[2] = x949;
            x953[3] = x950;
            x953[4] = x951;
            x953[5] = x952;
            x494(x953);
          };
          function<void(float**)> x937 = [&](float** x938) {
            float* x939 = x938[0];
            float* x940 = x938[1];
            float* x941 = x938[2];
            float* x942 = x938[3];
            float* x943 = x938[4];
            float* x944 = x938[5];
            float** x962 = (float**)myMalloc(6 * sizeof(float*));;
            x962[0] = x939;
            x962[1] = x940;
            x962[2] = x941;
            x962[3] = x942;
            x962[4] = x943;
            x962[5] = x944;
            x945(x962);
          };
          function<void(float**)> x504 = [&](float** x505) {
            float* x506 = x505[0];
            float* x507 = x505[1];
            float* x508 = x505[2];
            float* x509 = x505[3];
            float* x510 = x505[4];
            float* x511 = x505[5];
            int32_t x512 = x442[x493];
            float** x3205 = (float**)myMalloc(6 * sizeof(float*));;
            x3205[0] = x459;
            x3205[1] = x464;
            x3205[2] = x469;
            x3205[3] = x474;
            x3205[4] = x479;
            x3205[5] = x484;
            function<void(float**)> x513 = [&](float** x514) {
              float* x515 = x514[0];
              float* x516 = x514[1];
              float* x517 = x514[2];
              float* x518 = x514[3];
              float* x519 = x514[4];
              float* x520 = x514[5];
              float* x521 = (float*)myMalloc(5 * sizeof(float));;
              for(int x522=0; x522 < 5; x522++) {
                x521[x522] = 0.0f;

              }
              int32_t x526 = x436[x493];
              x521[x526] = 1.0f;
              float* x528 = (float*)myMalloc(5 * sizeof(float));;
              for(int x529=0; x529 < 5; x529++) {
                x528[x529] = 0.0f;

              }
              int32_t x533 = x440[x493];
              bool x534 = x533 < 0;
              if (x534) {
                int32_t x535 = x438[x493];
                float* x536 = x7[x535];
                float* x537 = (float*)myMalloc(300 * sizeof(float));;
                for(int x538=0; x538 < 300; x538++) {
                  x537[x538] = 0.0f;

                }
                // dot: List(150, 300), WrappedArray(300)
                float* x543 = (float*)myMalloc(150 * sizeof(float));;
                for(int x544=0; x544 < 150; x544++) {
                  float x545 = 0.0f;
                  int32_t x547 = x544 * 300;
                  for(int x546=0; x546 < 300; x546++) {
                    int32_t x548 = x547 + x546;
                    float x549 = x50[x548];
                    float x550 = x536[x546];
                    float x551 = x549 * x550;
                    x545 += x551;

                  }
                  float x555 = x545;
                  x543[x544] = x555;

                }
                float* x559 = (float*)myMalloc(150 * sizeof(float));;
                for(int x560=0; x560 < 150; x560++) {
                  x559[x560] = 0.0f;

                }
                float* x564 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x565 = 0;
                int32_t x566 = 0;
                int32_t x567 = 0;
                for(int x568=0; x568 < 150; x568++) {
                  int32_t x569 = x565;
                  int32_t x570 = x566;
                  float x571 = x543[x570];
                  int32_t x572 = x567;
                  float x573 = x59[x572];
                  float x574 = x571 + x573;
                  x564[x569] = x574;
                  x565 += 1;
                  x566 += 1;
                  x567 += 1;

                }
                float* x581 = (float*)myMalloc(150 * sizeof(float));;
                for(int x582=0; x582 < 150; x582++) {
                  x581[x582] = 0.0f;

                }
                float* x586 = (float*)myMalloc(150 * sizeof(float));;
                for(int x587=0; x587 < 150; x587++) {
                  float x588 = x564[x587];
                  float x589 = -1.0f * x588;
                  double x590 = (double)x589;
                  double x591 = exp(x590);
                  float x592 = (float)x591;
                  float x593 = x592 + 1.0f;
                  float x594 = 1.0f / x593;
                  x586[x587] = x594;

                }
                float* x598 = (float*)myMalloc(150 * sizeof(float));;
                for(int x599=0; x599 < 150; x599++) {
                  x598[x599] = 0.0f;

                }
                // dot: List(150, 300), WrappedArray(300)
                float* x604 = (float*)myMalloc(150 * sizeof(float));;
                for(int x605=0; x605 < 150; x605++) {
                  float x606 = 0.0f;
                  int32_t x608 = x605 * 300;
                  for(int x607=0; x607 < 300; x607++) {
                    int32_t x609 = x608 + x607;
                    float x610 = x65[x609];
                    float x611 = x536[x607];
                    float x612 = x610 * x611;
                    x606 += x612;

                  }
                  float x616 = x606;
                  x604[x605] = x616;

                }
                float* x620 = (float*)myMalloc(150 * sizeof(float));;
                for(int x621=0; x621 < 150; x621++) {
                  x620[x621] = 0.0f;

                }
                float* x625 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x626 = 0;
                int32_t x627 = 0;
                int32_t x628 = 0;
                for(int x629=0; x629 < 150; x629++) {
                  int32_t x630 = x626;
                  int32_t x631 = x627;
                  float x632 = x604[x631];
                  int32_t x633 = x628;
                  float x634 = x73[x633];
                  float x635 = x632 + x634;
                  x625[x630] = x635;
                  x626 += 1;
                  x627 += 1;
                  x628 += 1;

                }
                float* x642 = (float*)myMalloc(150 * sizeof(float));;
                for(int x643=0; x643 < 150; x643++) {
                  x642[x643] = 0.0f;

                }
                float* x647 = (float*)myMalloc(150 * sizeof(float));;
                for(int x648=0; x648 < 150; x648++) {
                  float x649 = x625[x648];
                  float x650 = -1.0f * x649;
                  double x651 = (double)x650;
                  double x652 = exp(x651);
                  float x653 = (float)x652;
                  float x654 = x653 + 1.0f;
                  float x655 = 1.0f / x654;
                  x647[x648] = x655;

                }
                float* x659 = (float*)myMalloc(150 * sizeof(float));;
                for(int x660=0; x660 < 150; x660++) {
                  x659[x660] = 0.0f;

                }
                // dot: List(150, 300), WrappedArray(300)
                float* x665 = (float*)myMalloc(150 * sizeof(float));;
                for(int x666=0; x666 < 150; x666++) {
                  float x667 = 0.0f;
                  int32_t x669 = x666 * 300;
                  for(int x668=0; x668 < 300; x668++) {
                    int32_t x670 = x669 + x668;
                    float x671 = x78[x670];
                    float x672 = x536[x668];
                    float x673 = x671 * x672;
                    x667 += x673;

                  }
                  float x677 = x667;
                  x665[x666] = x677;

                }
                float* x681 = (float*)myMalloc(150 * sizeof(float));;
                for(int x682=0; x682 < 150; x682++) {
                  x681[x682] = 0.0f;

                }
                float* x686 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x687 = 0;
                int32_t x688 = 0;
                int32_t x689 = 0;
                for(int x690=0; x690 < 150; x690++) {
                  int32_t x691 = x687;
                  int32_t x692 = x688;
                  float x693 = x665[x692];
                  int32_t x694 = x689;
                  float x695 = x86[x694];
                  float x696 = x693 + x695;
                  x686[x691] = x696;
                  x687 += 1;
                  x688 += 1;
                  x689 += 1;

                }
                float* x703 = (float*)myMalloc(150 * sizeof(float));;
                for(int x704=0; x704 < 150; x704++) {
                  x703[x704] = 0.0f;

                }
                float* x708 = (float*)myMalloc(150 * sizeof(float));;
                for(int x709=0; x709 < 150; x709++) {
                  float x710 = x686[x709];
                  double x711 = (double)x710;
                  double x712 = tanh(x711);
                  float x713 = (float)x712;
                  x708[x709] = x713;

                }
                float* x717 = (float*)myMalloc(150 * sizeof(float));;
                for(int x718=0; x718 < 150; x718++) {
                  x717[x718] = 0.0f;

                }
                float* x722 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x723 = 0;
                int32_t x724 = 0;
                int32_t x725 = 0;
                for(int x726=0; x726 < 150; x726++) {
                  int32_t x727 = x723;
                  int32_t x728 = x724;
                  float x729 = x586[x728];
                  int32_t x730 = x725;
                  float x731 = x708[x730];
                  float x732 = x729 * x731;
                  x722[x727] = x732;
                  x723 += 1;
                  x724 += 1;
                  x725 += 1;

                }
                float* x739 = (float*)myMalloc(150 * sizeof(float));;
                for(int x740=0; x740 < 150; x740++) {
                  x739[x740] = 0.0f;

                }
                float* x744 = (float*)myMalloc(150 * sizeof(float));;
                for(int x745=0; x745 < 150; x745++) {
                  float x746 = x722[x745];
                  double x747 = (double)x746;
                  double x748 = tanh(x747);
                  float x749 = (float)x748;
                  x744[x745] = x749;

                }
                float* x753 = (float*)myMalloc(150 * sizeof(float));;
                for(int x754=0; x754 < 150; x754++) {
                  x753[x754] = 0.0f;

                }
                float* x758 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x759 = 0;
                int32_t x760 = 0;
                int32_t x761 = 0;
                for(int x762=0; x762 < 150; x762++) {
                  int32_t x763 = x759;
                  int32_t x764 = x760;
                  float x765 = x647[x764];
                  int32_t x766 = x761;
                  float x767 = x744[x766];
                  float x768 = x765 * x767;
                  x758[x763] = x768;
                  x759 += 1;
                  x760 += 1;
                  x761 += 1;

                }
                float* x775 = (float*)myMalloc(150 * sizeof(float));;
                for(int x776=0; x776 < 150; x776++) {
                  x775[x776] = 0.0f;

                }
                // dot: List(5, 150), List(150)
                float* x781 = (float*)myMalloc(5 * sizeof(float));;
                for(int x782=0; x782 < 5; x782++) {
                  float x783 = 0.0f;
                  int32_t x785 = x782 * 150;
                  for(int x784=0; x784 < 150; x784++) {
                    int32_t x786 = x785 + x784;
                    float x787 = x192[x786];
                    float x788 = x758[x784];
                    float x789 = x787 * x788;
                    x783 += x789;

                  }
                  float x793 = x783;
                  x781[x782] = x793;

                }
                float* x797 = (float*)myMalloc(5 * sizeof(float));;
                for(int x798=0; x798 < 5; x798++) {
                  x797[x798] = 0.0f;

                }
                float* x802 = (float*)myMalloc(5 * sizeof(float));;
                int32_t x803 = 0;
                int32_t x804 = 0;
                int32_t x805 = 0;
                for(int x806=0; x806 < 5; x806++) {
                  int32_t x807 = x803;
                  int32_t x808 = x804;
                  float x809 = x781[x808];
                  int32_t x810 = x805;
                  float x811 = x201[x810];
                  float x812 = x809 + x811;
                  x802[x807] = x812;
                  x803 += 1;
                  x804 += 1;
                  x805 += 1;

                }
                float* x819 = (float*)myMalloc(5 * sizeof(float));;
                for(int x820=0; x820 < 5; x820++) {
                  x819[x820] = 0.0f;

                }
                float* x824 = (float*)myMalloc(5 * sizeof(float));;
                for(int x825=0; x825 < 5; x825++) {
                  float x826 = x802[x825];
                  double x827 = (double)x826;
                  double x828 = exp(x827);
                  float x829 = (float)x828;
                  x824[x825] = x829;

                }
                float* x833 = (float*)myMalloc(5 * sizeof(float));;
                for(int x834=0; x834 < 5; x834++) {
                  x833[x834] = 0.0f;

                }
                float x838 = 0.0f;
                for(int x839=0; x839 < 5; x839++) {
                  float x840 = x838;
                  float x841 = x824[x839];
                  float x842 = x840 + x841;
                  x838 = x842;

                }
                float x846 = x838;
                float* x847 = (float*)myMalloc(1 * sizeof(float));;
                x847[0] = x846;
                float* x849 = (float*)myMalloc(1 * sizeof(float));;
                for(int x850=0; x850 < 1; x850++) {
                  x849[x850] = 0.0f;

                }
                float* x854 = (float*)myMalloc(5 * sizeof(float));;
                int32_t x855 = 0;
                int32_t x856 = 0;
                int32_t x857 = 0;
                for(int x858=0; x858 < 5; x858++) {
                  int32_t x859 = x855;
                  int32_t x860 = x856;
                  float x861 = x824[x860];
                  int32_t x862 = x857;
                  float x863 = x847[x862];
                  float x864 = x861 / x863;
                  x854[x859] = x864;
                  x855 += 1;
                  x856 += 1;

                }
                float* x870 = (float*)myMalloc(5 * sizeof(float));;
                for(int x871=0; x871 < 5; x871++) {
                  x870[x871] = 0.0f;

                }
                float* x875 = (float*)myMalloc(1 * sizeof(float));;
                int32_t x876 = 0;
                int32_t x877 = 0;
                int32_t x878 = 0;
                int32_t x879 = x876;
                int32_t x880 = x877;
                float x881 = x506[x880];
                int32_t x882 = x878;
                float x883 = x515[x882];
                float x884 = x881 + x883;
                x875[x879] = x884;
                x876 += 1;
                float* x887 = (float*)myMalloc(1 * sizeof(float));;
                for(int x888=0; x888 < 1; x888++) {
                  x887[x888] = 0.0f;

                }
                // dot: List(5), WrappedArray(5)
                float x893 = 0.0f;
                for(int x894=0; x894 < 5; x894++) {
                  float x895 = x854[x894];
                  float x896 = x521[x894];
                  float x897 = x895 * x896;
                  x893 += x897;

                }
                float* x901 = (float*)myMalloc(1 * sizeof(float));;
                float x902 = x893;
                x901[0] = x902;
                float* x904 = (float*)myMalloc(1 * sizeof(float));;
                for(int x905=0; x905 < 1; x905++) {
                  x904[x905] = 0.0f;

                }
                float* x909 = (float*)myMalloc(1 * sizeof(float));;
                float x910 = x901[0];
                double x911 = (double)x910;
                double x912 = log(x911);
                float x913 = (float)x912;
                x909[0] = x913;
                float* x915 = (float*)myMalloc(1 * sizeof(float));;
                for(int x916=0; x916 < 1; x916++) {
                  x915[x916] = 0.0f;

                }
                float* x920 = (float*)myMalloc(1 * sizeof(float));;
                int32_t x921 = 0;
                int32_t x922 = 0;
                int32_t x923 = 0;
                int32_t x924 = x921;
                int32_t x925 = x922;
                float x926 = x875[x925];
                int32_t x927 = x923;
                float x928 = x909[x927];
                float x929 = x926 - x928;
                x920[x924] = x929;
                x921 += 1;
                float* x932 = (float*)myMalloc(1 * sizeof(float));;
                for(int x933=0; x933 < 1; x933++) {
                  x932[x933] = 0.0f;

                }
                float** x971 = (float**)myMalloc(6 * sizeof(float*));;
                x971[0] = x920;
                x971[1] = x932;
                x971[2] = x758;
                x971[3] = x775;
                x971[4] = x722;
                x971[5] = x739;
                x937(x971);
                int32_t x979 = 0;
                int32_t x980 = 0;
                int32_t x981 = 0;
                int32_t x982 = x979;
                float x983 = x887[x982];
                float x984 = x875[x982];
                int32_t x985 = x980;
                float x986 = x909[x985];
                int32_t x987 = x981;
                float x988 = x932[x987];
                float x989 = x983 + x988;
                x887[x982] = x989;
                float x991 = x915[x985];
                float x992 = x875[x982];
                float x993 = x909[x985];
                float x994 = x932[x987];
                float x995 = -1.0f * x994;
                float x996 = x991 + x995;
                x915[x985] = x996;
                x981 += 1;
                float x999 = x904[0];
                float x1000 = x915[0];
                float x1001 = x901[0];
                float x1002 = x1000 / x1001;
                float x1003 = x999 + x1002;
                x904[0] = x1003;
                float x1005 = x904[0];
                // Generate code for addMul
                for(int x1007=0; x1007 < 5; x1007++) {
                  float x1008 = x870[x1007];
                  float x1009 = x521[x1007];
                  float x1010 = x1005 * x1009;
                  float x1011 = x1008 + x1010;
                  x870[x1007] = x1011;

                }
                float x1015 = x904[0];
                // Generate code for addMul
                for(int x1017=0; x1017 < 5; x1017++) {
                  float x1018 = x528[x1017];
                  float x1019 = x854[x1017];
                  float x1020 = x1015 * x1019;
                  float x1021 = x1018 + x1020;
                  x528[x1017] = x1021;

                }
                int32_t x1025 = 0;
                int32_t x1026 = 0;
                int32_t x1027 = 0;
                int32_t x1028 = x1025;
                float x1029 = x507[x1028];
                float x1030 = x506[x1028];
                int32_t x1031 = x1026;
                float x1032 = x515[x1031];
                int32_t x1033 = x1027;
                float x1034 = x887[x1033];
                float x1035 = x1029 + x1034;
                x507[x1028] = x1035;
                float x1037 = x516[x1031];
                float x1038 = x506[x1028];
                float x1039 = x515[x1031];
                float x1040 = x887[x1033];
                float x1041 = x1037 + x1040;
                x516[x1031] = x1041;
                x1027 += 1;
                int32_t x1044 = 0;
                int32_t x1045 = 0;
                int32_t x1046 = 0;
                for(int x1047=0; x1047 < 5; x1047++) {
                  int32_t x1048 = x1044;
                  float x1049 = x833[x1048];
                  float x1050 = x824[x1048];
                  int32_t x1051 = x1045;
                  float x1052 = x847[x1051];
                  int32_t x1053 = x1046;
                  float x1054 = x870[x1053];
                  float x1055 = x1054 / x1052;
                  float x1056 = x1049 + x1055;
                  x833[x1048] = x1056;
                  float x1058 = x849[x1051];
                  float x1059 = x824[x1048];
                  float x1060 = x847[x1051];
                  float x1061 = x870[x1053];
                  float x1062 = -1.0f * x1059;
                  float x1063 = x1062 * x1061;
                  float x1064 = x1060 * x1060;
                  float x1065 = x1063 / x1064;
                  float x1066 = x1058 + x1065;
                  x849[x1051] = x1066;
                  x1046 += 1;
                  x1044 += 1;

                }
                // += tensor of dim 0
                float x1073 = x849[0];
                for(int x1074=0; x1074 < 5; x1074++) {
                  float x1075 = x833[x1074];
                  float x1076 = x1075 + x1073;
                  x833[x1074] = x1076;

                }
                for(int x1080=0; x1080 < 5; x1080++) {
                  float x1081 = x819[x1080];
                  float x1082 = x824[x1080];
                  float x1083 = x833[x1080];
                  float x1084 = x1082 * x1083;
                  float x1085 = x1081 + x1084;
                  x819[x1080] = x1085;

                }
                int32_t x1089 = 0;
                int32_t x1090 = 0;
                int32_t x1091 = 0;
                for(int x1092=0; x1092 < 5; x1092++) {
                  int32_t x1093 = x1089;
                  float x1094 = x797[x1093];
                  float x1095 = x781[x1093];
                  int32_t x1096 = x1090;
                  float x1097 = x201[x1096];
                  int32_t x1098 = x1091;
                  float x1099 = x819[x1098];
                  float x1100 = x1094 + x1099;
                  x797[x1093] = x1100;
                  float x1102 = x312[x1096];
                  float x1103 = x781[x1093];
                  float x1104 = x201[x1096];
                  float x1105 = x819[x1098];
                  float x1106 = x1102 + x1105;
                  x312[x1096] = x1106;
                  x1091 += 1;
                  x1089 += 1;
                  x1090 += 1;

                }
                // add_cartesian
                int32_t x1114 = 0;
                for(int x1115=0; x1115 < 5; x1115++) {
                  for(int x1116=0; x1116 < 150; x1116++) {
                    int32_t x1117 = x1114;
                    int32_t x1118 = x1117 + x1116;
                    float x1119 = x307[x1118];
                    float x1120 = x758[x1116];
                    float x1121 = x797[x1115];
                    float x1122 = x1120 * x1121;
                    float x1123 = x1119 + x1122;
                    x307[x1118] = x1123;

                  }
                  x1114 += 150;

                }
                int32_t x1130 = 0;
                for(int x1131=0; x1131 < 5; x1131++) {
                  for(int x1132=0; x1132 < 150; x1132++) {
                    float x1133 = x775[x1132];
                    int32_t x1134 = x1130;
                    int32_t x1135 = x1134 + x1132;
                    float x1136 = x192[x1135];
                    float x1137 = x797[x1131];
                    float x1138 = x1136 * x1137;
                    float x1139 = x1133 + x1138;
                    x775[x1132] = x1139;

                  }
                  x1130 += 150;

                }
                int32_t x1146 = 0;
                int32_t x1147 = 0;
                int32_t x1148 = 0;
                for(int x1149=0; x1149 < 150; x1149++) {
                  int32_t x1150 = x1146;
                  float x1151 = x659[x1150];
                  float x1152 = x647[x1150];
                  int32_t x1153 = x1147;
                  float x1154 = x744[x1153];
                  int32_t x1155 = x1148;
                  float x1156 = x775[x1155];
                  float x1157 = x1156 * x1154;
                  float x1158 = x1151 + x1157;
                  x659[x1150] = x1158;
                  float x1160 = x753[x1153];
                  float x1161 = x647[x1150];
                  float x1162 = x744[x1153];
                  float x1163 = x775[x1155];
                  float x1164 = x1163 * x1161;
                  float x1165 = x1160 + x1164;
                  x753[x1153] = x1165;
                  x1148 += 1;
                  x1146 += 1;
                  x1147 += 1;

                }
                for(int x1172=0; x1172 < 150; x1172++) {
                  float x1173 = x739[x1172];
                  float x1174 = x744[x1172];
                  float x1177 = x753[x1172];
                  float x1175 = x1174 * x1174;
                  float x1176 = 1.0f - x1175;
                  float x1178 = x1176 * x1177;
                  float x1179 = x1173 + x1178;
                  x739[x1172] = x1179;

                }
                int32_t x1183 = 0;
                int32_t x1184 = 0;
                int32_t x1185 = 0;
                for(int x1186=0; x1186 < 150; x1186++) {
                  int32_t x1187 = x1183;
                  float x1188 = x598[x1187];
                  float x1189 = x586[x1187];
                  int32_t x1190 = x1184;
                  float x1191 = x708[x1190];
                  int32_t x1192 = x1185;
                  float x1193 = x739[x1192];
                  float x1194 = x1193 * x1191;
                  float x1195 = x1188 + x1194;
                  x598[x1187] = x1195;
                  float x1197 = x717[x1190];
                  float x1198 = x586[x1187];
                  float x1199 = x708[x1190];
                  float x1200 = x739[x1192];
                  float x1201 = x1200 * x1198;
                  float x1202 = x1197 + x1201;
                  x717[x1190] = x1202;
                  x1185 += 1;
                  x1183 += 1;
                  x1184 += 1;

                }
                for(int x1209=0; x1209 < 150; x1209++) {
                  float x1210 = x703[x1209];
                  float x1211 = x708[x1209];
                  float x1214 = x717[x1209];
                  float x1212 = x1211 * x1211;
                  float x1213 = 1.0f - x1212;
                  float x1215 = x1213 * x1214;
                  float x1216 = x1210 + x1215;
                  x703[x1209] = x1216;

                }
                int32_t x1220 = 0;
                int32_t x1221 = 0;
                int32_t x1222 = 0;
                for(int x1223=0; x1223 < 150; x1223++) {
                  int32_t x1224 = x1220;
                  float x1225 = x681[x1224];
                  float x1226 = x665[x1224];
                  int32_t x1227 = x1221;
                  float x1228 = x86[x1227];
                  int32_t x1229 = x1222;
                  float x1230 = x703[x1229];
                  float x1231 = x1225 + x1230;
                  x681[x1224] = x1231;
                  float x1233 = x232[x1227];
                  float x1234 = x665[x1224];
                  float x1235 = x86[x1227];
                  float x1236 = x703[x1229];
                  float x1237 = x1233 + x1236;
                  x232[x1227] = x1237;
                  x1222 += 1;
                  x1220 += 1;
                  x1221 += 1;

                }
                // add_cartesian
                int32_t x1245 = 0;
                for(int x1246=0; x1246 < 150; x1246++) {
                  for(int x1247=0; x1247 < 300; x1247++) {
                    int32_t x1248 = x1245;
                    int32_t x1249 = x1248 + x1247;
                    float x1250 = x227[x1249];
                    float x1251 = x536[x1247];
                    float x1252 = x681[x1246];
                    float x1253 = x1251 * x1252;
                    float x1254 = x1250 + x1253;
                    x227[x1249] = x1254;

                  }
                  x1245 += 300;

                }
                int32_t x1261 = 0;
                for(int x1262=0; x1262 < 150; x1262++) {
                  for(int x1263=0; x1263 < 300; x1263++) {
                    float x1264 = x537[x1263];
                    int32_t x1265 = x1261;
                    int32_t x1266 = x1265 + x1263;
                    float x1267 = x78[x1266];
                    float x1268 = x681[x1262];
                    float x1269 = x1267 * x1268;
                    float x1270 = x1264 + x1269;
                    x537[x1263] = x1270;

                  }
                  x1261 += 300;

                }
                for(int x1277=0; x1277 < 150; x1277++) {
                  float x1278 = x642[x1277];
                  float x1279 = x647[x1277];
                  float x1282 = x659[x1277];
                  float x1280 = 1.0f - x1279;
                  float x1281 = x1280 * x1279;
                  float x1283 = x1281 * x1282;
                  float x1284 = x1278 + x1283;
                  x642[x1277] = x1284;

                }
                int32_t x1288 = 0;
                int32_t x1289 = 0;
                int32_t x1290 = 0;
                for(int x1291=0; x1291 < 150; x1291++) {
                  int32_t x1292 = x1288;
                  float x1293 = x620[x1292];
                  float x1294 = x604[x1292];
                  int32_t x1295 = x1289;
                  float x1296 = x73[x1295];
                  int32_t x1297 = x1290;
                  float x1298 = x642[x1297];
                  float x1299 = x1293 + x1298;
                  x620[x1292] = x1299;
                  float x1301 = x222[x1295];
                  float x1302 = x604[x1292];
                  float x1303 = x73[x1295];
                  float x1304 = x642[x1297];
                  float x1305 = x1301 + x1304;
                  x222[x1295] = x1305;
                  x1290 += 1;
                  x1288 += 1;
                  x1289 += 1;

                }
                // add_cartesian
                int32_t x1313 = 0;
                for(int x1314=0; x1314 < 150; x1314++) {
                  for(int x1315=0; x1315 < 300; x1315++) {
                    int32_t x1316 = x1313;
                    int32_t x1317 = x1316 + x1315;
                    float x1318 = x217[x1317];
                    float x1319 = x536[x1315];
                    float x1320 = x620[x1314];
                    float x1321 = x1319 * x1320;
                    float x1322 = x1318 + x1321;
                    x217[x1317] = x1322;

                  }
                  x1313 += 300;

                }
                int32_t x1329 = 0;
                for(int x1330=0; x1330 < 150; x1330++) {
                  for(int x1331=0; x1331 < 300; x1331++) {
                    float x1332 = x537[x1331];
                    int32_t x1333 = x1329;
                    int32_t x1334 = x1333 + x1331;
                    float x1335 = x65[x1334];
                    float x1336 = x620[x1330];
                    float x1337 = x1335 * x1336;
                    float x1338 = x1332 + x1337;
                    x537[x1331] = x1338;

                  }
                  x1329 += 300;

                }
                for(int x1345=0; x1345 < 150; x1345++) {
                  float x1346 = x581[x1345];
                  float x1347 = x586[x1345];
                  float x1350 = x598[x1345];
                  float x1348 = 1.0f - x1347;
                  float x1349 = x1348 * x1347;
                  float x1351 = x1349 * x1350;
                  float x1352 = x1346 + x1351;
                  x581[x1345] = x1352;

                }
                int32_t x1356 = 0;
                int32_t x1357 = 0;
                int32_t x1358 = 0;
                for(int x1359=0; x1359 < 150; x1359++) {
                  int32_t x1360 = x1356;
                  float x1361 = x559[x1360];
                  float x1362 = x543[x1360];
                  int32_t x1363 = x1357;
                  float x1364 = x59[x1363];
                  int32_t x1365 = x1358;
                  float x1366 = x581[x1365];
                  float x1367 = x1361 + x1366;
                  x559[x1360] = x1367;
                  float x1369 = x212[x1363];
                  float x1370 = x543[x1360];
                  float x1371 = x59[x1363];
                  float x1372 = x581[x1365];
                  float x1373 = x1369 + x1372;
                  x212[x1363] = x1373;
                  x1358 += 1;
                  x1356 += 1;
                  x1357 += 1;

                }
                // add_cartesian
                int32_t x1381 = 0;
                for(int x1382=0; x1382 < 150; x1382++) {
                  for(int x1383=0; x1383 < 300; x1383++) {
                    int32_t x1384 = x1381;
                    int32_t x1385 = x1384 + x1383;
                    float x1386 = x207[x1385];
                    float x1387 = x536[x1383];
                    float x1388 = x559[x1382];
                    float x1389 = x1387 * x1388;
                    float x1390 = x1386 + x1389;
                    x207[x1385] = x1390;

                  }
                  x1381 += 300;

                }
                int32_t x1397 = 0;
                for(int x1398=0; x1398 < 150; x1398++) {
                  for(int x1399=0; x1399 < 300; x1399++) {
                    float x1400 = x537[x1399];
                    int32_t x1401 = x1397;
                    int32_t x1402 = x1401 + x1399;
                    float x1403 = x50[x1402];
                    float x1404 = x559[x1398];
                    float x1405 = x1403 * x1404;
                    float x1406 = x1400 + x1405;
                    x537[x1399] = x1406;

                  }
                  x1397 += 300;

                }
              } else {
                // dot: List(150, 150), WrappedArray(150)
                float* x1415 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1416=0; x1416 < 150; x1416++) {
                  float x1417 = 0.0f;
                  int32_t x1419 = x1416 * 150;
                  for(int x1418=0; x1418 < 150; x1418++) {
                    int32_t x1420 = x1419 + x1418;
                    float x1421 = x91[x1420];
                    float x1422 = x508[x1418];
                    float x1423 = x1421 * x1422;
                    x1417 += x1423;

                  }
                  float x1427 = x1417;
                  x1415[x1416] = x1427;

                }
                float* x1431 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1432=0; x1432 < 150; x1432++) {
                  x1431[x1432] = 0.0f;

                }
                // dot: List(150, 150), WrappedArray(150)
                float* x1437 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1438=0; x1438 < 150; x1438++) {
                  float x1439 = 0.0f;
                  int32_t x1441 = x1438 * 150;
                  for(int x1440=0; x1440 < 150; x1440++) {
                    int32_t x1442 = x1441 + x1440;
                    float x1443 = x100[x1442];
                    float x1444 = x517[x1440];
                    float x1445 = x1443 * x1444;
                    x1439 += x1445;

                  }
                  float x1449 = x1439;
                  x1437[x1438] = x1449;

                }
                float* x1453 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1454=0; x1454 < 150; x1454++) {
                  x1453[x1454] = 0.0f;

                }
                float* x1458 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x1459 = 0;
                int32_t x1460 = 0;
                int32_t x1461 = 0;
                for(int x1462=0; x1462 < 150; x1462++) {
                  int32_t x1463 = x1459;
                  int32_t x1464 = x1460;
                  float x1465 = x1415[x1464];
                  int32_t x1466 = x1461;
                  float x1467 = x1437[x1466];
                  float x1468 = x1465 + x1467;
                  x1458[x1463] = x1468;
                  x1459 += 1;
                  x1460 += 1;
                  x1461 += 1;

                }
                float* x1475 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1476=0; x1476 < 150; x1476++) {
                  x1475[x1476] = 0.0f;

                }
                float* x1480 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x1481 = 0;
                int32_t x1482 = 0;
                int32_t x1483 = 0;
                for(int x1484=0; x1484 < 150; x1484++) {
                  int32_t x1485 = x1481;
                  int32_t x1486 = x1482;
                  float x1487 = x1458[x1486];
                  int32_t x1488 = x1483;
                  float x1489 = x108[x1488];
                  float x1490 = x1487 + x1489;
                  x1480[x1485] = x1490;
                  x1481 += 1;
                  x1482 += 1;
                  x1483 += 1;

                }
                float* x1497 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1498=0; x1498 < 150; x1498++) {
                  x1497[x1498] = 0.0f;

                }
                float* x1502 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1503=0; x1503 < 150; x1503++) {
                  float x1504 = x1480[x1503];
                  float x1505 = -1.0f * x1504;
                  double x1506 = (double)x1505;
                  double x1507 = exp(x1506);
                  float x1508 = (float)x1507;
                  float x1509 = x1508 + 1.0f;
                  float x1510 = 1.0f / x1509;
                  x1502[x1503] = x1510;

                }
                float* x1514 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1515=0; x1515 < 150; x1515++) {
                  x1514[x1515] = 0.0f;

                }
                // dot: List(150, 150), WrappedArray(150)
                float* x1520 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1521=0; x1521 < 150; x1521++) {
                  float x1522 = 0.0f;
                  int32_t x1524 = x1521 * 150;
                  for(int x1523=0; x1523 < 150; x1523++) {
                    int32_t x1525 = x1524 + x1523;
                    float x1526 = x113[x1525];
                    float x1527 = x508[x1523];
                    float x1528 = x1526 * x1527;
                    x1522 += x1528;

                  }
                  float x1532 = x1522;
                  x1520[x1521] = x1532;

                }
                float* x1536 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1537=0; x1537 < 150; x1537++) {
                  x1536[x1537] = 0.0f;

                }
                // dot: List(150, 150), WrappedArray(150)
                float* x1542 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1543=0; x1543 < 150; x1543++) {
                  float x1544 = 0.0f;
                  int32_t x1546 = x1543 * 150;
                  for(int x1545=0; x1545 < 150; x1545++) {
                    int32_t x1547 = x1546 + x1545;
                    float x1548 = x121[x1547];
                    float x1549 = x517[x1545];
                    float x1550 = x1548 * x1549;
                    x1544 += x1550;

                  }
                  float x1554 = x1544;
                  x1542[x1543] = x1554;

                }
                float* x1558 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1559=0; x1559 < 150; x1559++) {
                  x1558[x1559] = 0.0f;

                }
                float* x1563 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x1564 = 0;
                int32_t x1565 = 0;
                int32_t x1566 = 0;
                for(int x1567=0; x1567 < 150; x1567++) {
                  int32_t x1568 = x1564;
                  int32_t x1569 = x1565;
                  float x1570 = x1520[x1569];
                  int32_t x1571 = x1566;
                  float x1572 = x1542[x1571];
                  float x1573 = x1570 + x1572;
                  x1563[x1568] = x1573;
                  x1564 += 1;
                  x1565 += 1;
                  x1566 += 1;

                }
                float* x1580 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1581=0; x1581 < 150; x1581++) {
                  x1580[x1581] = 0.0f;

                }
                float* x1585 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x1586 = 0;
                int32_t x1587 = 0;
                int32_t x1588 = 0;
                for(int x1589=0; x1589 < 150; x1589++) {
                  int32_t x1590 = x1586;
                  int32_t x1591 = x1587;
                  float x1592 = x1563[x1591];
                  int32_t x1593 = x1588;
                  float x1594 = x145[x1593];
                  float x1595 = x1592 + x1594;
                  x1585[x1590] = x1595;
                  x1586 += 1;
                  x1587 += 1;
                  x1588 += 1;

                }
                float* x1602 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1603=0; x1603 < 150; x1603++) {
                  x1602[x1603] = 0.0f;

                }
                float* x1607 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1608=0; x1608 < 150; x1608++) {
                  float x1609 = x1585[x1608];
                  float x1610 = -1.0f * x1609;
                  double x1611 = (double)x1610;
                  double x1612 = exp(x1611);
                  float x1613 = (float)x1612;
                  float x1614 = x1613 + 1.0f;
                  float x1615 = 1.0f / x1614;
                  x1607[x1608] = x1615;

                }
                float* x1619 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1620=0; x1620 < 150; x1620++) {
                  x1619[x1620] = 0.0f;

                }
                // dot: List(150, 150), WrappedArray(150)
                float* x1625 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1626=0; x1626 < 150; x1626++) {
                  float x1627 = 0.0f;
                  int32_t x1629 = x1626 * 150;
                  for(int x1628=0; x1628 < 150; x1628++) {
                    int32_t x1630 = x1629 + x1628;
                    float x1631 = x129[x1630];
                    float x1632 = x508[x1628];
                    float x1633 = x1631 * x1632;
                    x1627 += x1633;

                  }
                  float x1637 = x1627;
                  x1625[x1626] = x1637;

                }
                float* x1641 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1642=0; x1642 < 150; x1642++) {
                  x1641[x1642] = 0.0f;

                }
                // dot: List(150, 150), WrappedArray(150)
                float* x1647 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1648=0; x1648 < 150; x1648++) {
                  float x1649 = 0.0f;
                  int32_t x1651 = x1648 * 150;
                  for(int x1650=0; x1650 < 150; x1650++) {
                    int32_t x1652 = x1651 + x1650;
                    float x1653 = x137[x1652];
                    float x1654 = x517[x1650];
                    float x1655 = x1653 * x1654;
                    x1649 += x1655;

                  }
                  float x1659 = x1649;
                  x1647[x1648] = x1659;

                }
                float* x1663 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1664=0; x1664 < 150; x1664++) {
                  x1663[x1664] = 0.0f;

                }
                float* x1668 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x1669 = 0;
                int32_t x1670 = 0;
                int32_t x1671 = 0;
                for(int x1672=0; x1672 < 150; x1672++) {
                  int32_t x1673 = x1669;
                  int32_t x1674 = x1670;
                  float x1675 = x1625[x1674];
                  int32_t x1676 = x1671;
                  float x1677 = x1647[x1676];
                  float x1678 = x1675 + x1677;
                  x1668[x1673] = x1678;
                  x1669 += 1;
                  x1670 += 1;
                  x1671 += 1;

                }
                float* x1685 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1686=0; x1686 < 150; x1686++) {
                  x1685[x1686] = 0.0f;

                }
                float* x1690 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x1691 = 0;
                int32_t x1692 = 0;
                int32_t x1693 = 0;
                for(int x1694=0; x1694 < 150; x1694++) {
                  int32_t x1695 = x1691;
                  int32_t x1696 = x1692;
                  float x1697 = x1668[x1696];
                  int32_t x1698 = x1693;
                  float x1699 = x145[x1698];
                  float x1700 = x1697 + x1699;
                  x1690[x1695] = x1700;
                  x1691 += 1;
                  x1692 += 1;
                  x1693 += 1;

                }
                float* x1707 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1708=0; x1708 < 150; x1708++) {
                  x1707[x1708] = 0.0f;

                }
                float* x1712 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1713=0; x1713 < 150; x1713++) {
                  float x1714 = x1690[x1713];
                  float x1715 = -1.0f * x1714;
                  double x1716 = (double)x1715;
                  double x1717 = exp(x1716);
                  float x1718 = (float)x1717;
                  float x1719 = x1718 + 1.0f;
                  float x1720 = 1.0f / x1719;
                  x1712[x1713] = x1720;

                }
                float* x1724 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1725=0; x1725 < 150; x1725++) {
                  x1724[x1725] = 0.0f;

                }
                // dot: List(150, 150), WrappedArray(150)
                float* x1730 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1731=0; x1731 < 150; x1731++) {
                  float x1732 = 0.0f;
                  int32_t x1734 = x1731 * 150;
                  for(int x1733=0; x1733 < 150; x1733++) {
                    int32_t x1735 = x1734 + x1733;
                    float x1736 = x150[x1735];
                    float x1737 = x508[x1733];
                    float x1738 = x1736 * x1737;
                    x1732 += x1738;

                  }
                  float x1742 = x1732;
                  x1730[x1731] = x1742;

                }
                float* x1746 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1747=0; x1747 < 150; x1747++) {
                  x1746[x1747] = 0.0f;

                }
                // dot: List(150, 150), WrappedArray(150)
                float* x1752 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1753=0; x1753 < 150; x1753++) {
                  float x1754 = 0.0f;
                  int32_t x1756 = x1753 * 150;
                  for(int x1755=0; x1755 < 150; x1755++) {
                    int32_t x1757 = x1756 + x1755;
                    float x1758 = x158[x1757];
                    float x1759 = x517[x1755];
                    float x1760 = x1758 * x1759;
                    x1754 += x1760;

                  }
                  float x1764 = x1754;
                  x1752[x1753] = x1764;

                }
                float* x1768 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1769=0; x1769 < 150; x1769++) {
                  x1768[x1769] = 0.0f;

                }
                float* x1773 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x1774 = 0;
                int32_t x1775 = 0;
                int32_t x1776 = 0;
                for(int x1777=0; x1777 < 150; x1777++) {
                  int32_t x1778 = x1774;
                  int32_t x1779 = x1775;
                  float x1780 = x1730[x1779];
                  int32_t x1781 = x1776;
                  float x1782 = x1752[x1781];
                  float x1783 = x1780 + x1782;
                  x1773[x1778] = x1783;
                  x1774 += 1;
                  x1775 += 1;
                  x1776 += 1;

                }
                float* x1790 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1791=0; x1791 < 150; x1791++) {
                  x1790[x1791] = 0.0f;

                }
                float* x1795 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x1796 = 0;
                int32_t x1797 = 0;
                int32_t x1798 = 0;
                for(int x1799=0; x1799 < 150; x1799++) {
                  int32_t x1800 = x1796;
                  int32_t x1801 = x1797;
                  float x1802 = x1773[x1801];
                  int32_t x1803 = x1798;
                  float x1804 = x166[x1803];
                  float x1805 = x1802 + x1804;
                  x1795[x1800] = x1805;
                  x1796 += 1;
                  x1797 += 1;
                  x1798 += 1;

                }
                float* x1812 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1813=0; x1813 < 150; x1813++) {
                  x1812[x1813] = 0.0f;

                }
                float* x1817 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1818=0; x1818 < 150; x1818++) {
                  float x1819 = x1795[x1818];
                  float x1820 = -1.0f * x1819;
                  double x1821 = (double)x1820;
                  double x1822 = exp(x1821);
                  float x1823 = (float)x1822;
                  float x1824 = x1823 + 1.0f;
                  float x1825 = 1.0f / x1824;
                  x1817[x1818] = x1825;

                }
                float* x1829 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1830=0; x1830 < 150; x1830++) {
                  x1829[x1830] = 0.0f;

                }
                // dot: List(150, 150), WrappedArray(150)
                float* x1835 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1836=0; x1836 < 150; x1836++) {
                  float x1837 = 0.0f;
                  int32_t x1839 = x1836 * 150;
                  for(int x1838=0; x1838 < 150; x1838++) {
                    int32_t x1840 = x1839 + x1838;
                    float x1841 = x171[x1840];
                    float x1842 = x508[x1838];
                    float x1843 = x1841 * x1842;
                    x1837 += x1843;

                  }
                  float x1847 = x1837;
                  x1835[x1836] = x1847;

                }
                float* x1851 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1852=0; x1852 < 150; x1852++) {
                  x1851[x1852] = 0.0f;

                }
                // dot: List(150, 150), WrappedArray(150)
                float* x1857 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1858=0; x1858 < 150; x1858++) {
                  float x1859 = 0.0f;
                  int32_t x1861 = x1858 * 150;
                  for(int x1860=0; x1860 < 150; x1860++) {
                    int32_t x1862 = x1861 + x1860;
                    float x1863 = x179[x1862];
                    float x1864 = x517[x1860];
                    float x1865 = x1863 * x1864;
                    x1859 += x1865;

                  }
                  float x1869 = x1859;
                  x1857[x1858] = x1869;

                }
                float* x1873 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1874=0; x1874 < 150; x1874++) {
                  x1873[x1874] = 0.0f;

                }
                float* x1878 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x1879 = 0;
                int32_t x1880 = 0;
                int32_t x1881 = 0;
                for(int x1882=0; x1882 < 150; x1882++) {
                  int32_t x1883 = x1879;
                  int32_t x1884 = x1880;
                  float x1885 = x1835[x1884];
                  int32_t x1886 = x1881;
                  float x1887 = x1857[x1886];
                  float x1888 = x1885 + x1887;
                  x1878[x1883] = x1888;
                  x1879 += 1;
                  x1880 += 1;
                  x1881 += 1;

                }
                float* x1895 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1896=0; x1896 < 150; x1896++) {
                  x1895[x1896] = 0.0f;

                }
                float* x1900 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x1901 = 0;
                int32_t x1902 = 0;
                int32_t x1903 = 0;
                for(int x1904=0; x1904 < 150; x1904++) {
                  int32_t x1905 = x1901;
                  int32_t x1906 = x1902;
                  float x1907 = x1878[x1906];
                  int32_t x1908 = x1903;
                  float x1909 = x187[x1908];
                  float x1910 = x1907 + x1909;
                  x1900[x1905] = x1910;
                  x1901 += 1;
                  x1902 += 1;
                  x1903 += 1;

                }
                float* x1917 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1918=0; x1918 < 150; x1918++) {
                  x1917[x1918] = 0.0f;

                }
                float* x1922 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1923=0; x1923 < 150; x1923++) {
                  float x1924 = x1900[x1923];
                  double x1925 = (double)x1924;
                  double x1926 = tanh(x1925);
                  float x1927 = (float)x1926;
                  x1922[x1923] = x1927;

                }
                float* x1931 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1932=0; x1932 < 150; x1932++) {
                  x1931[x1932] = 0.0f;

                }
                float* x1936 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x1937 = 0;
                int32_t x1938 = 0;
                int32_t x1939 = 0;
                for(int x1940=0; x1940 < 150; x1940++) {
                  int32_t x1941 = x1937;
                  int32_t x1942 = x1938;
                  float x1943 = x1502[x1942];
                  int32_t x1944 = x1939;
                  float x1945 = x1922[x1944];
                  float x1946 = x1943 * x1945;
                  x1936[x1941] = x1946;
                  x1937 += 1;
                  x1938 += 1;
                  x1939 += 1;

                }
                float* x1953 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1954=0; x1954 < 150; x1954++) {
                  x1953[x1954] = 0.0f;

                }
                float* x1958 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x1959 = 0;
                int32_t x1960 = 0;
                int32_t x1961 = 0;
                for(int x1962=0; x1962 < 150; x1962++) {
                  int32_t x1963 = x1959;
                  int32_t x1964 = x1960;
                  float x1965 = x1607[x1964];
                  int32_t x1966 = x1961;
                  float x1967 = x510[x1966];
                  float x1968 = x1965 * x1967;
                  x1958[x1963] = x1968;
                  x1959 += 1;
                  x1960 += 1;
                  x1961 += 1;

                }
                float* x1975 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1976=0; x1976 < 150; x1976++) {
                  x1975[x1976] = 0.0f;

                }
                float* x1980 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x1981 = 0;
                int32_t x1982 = 0;
                int32_t x1983 = 0;
                for(int x1984=0; x1984 < 150; x1984++) {
                  int32_t x1985 = x1981;
                  int32_t x1986 = x1982;
                  float x1987 = x1936[x1986];
                  int32_t x1988 = x1983;
                  float x1989 = x1958[x1988];
                  float x1990 = x1987 + x1989;
                  x1980[x1985] = x1990;
                  x1981 += 1;
                  x1982 += 1;
                  x1983 += 1;

                }
                float* x1997 = (float*)myMalloc(150 * sizeof(float));;
                for(int x1998=0; x1998 < 150; x1998++) {
                  x1997[x1998] = 0.0f;

                }
                float* x2002 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x2003 = 0;
                int32_t x2004 = 0;
                int32_t x2005 = 0;
                for(int x2006=0; x2006 < 150; x2006++) {
                  int32_t x2007 = x2003;
                  int32_t x2008 = x2004;
                  float x2009 = x1712[x2008];
                  int32_t x2010 = x2005;
                  float x2011 = x519[x2010];
                  float x2012 = x2009 * x2011;
                  x2002[x2007] = x2012;
                  x2003 += 1;
                  x2004 += 1;
                  x2005 += 1;

                }
                float* x2019 = (float*)myMalloc(150 * sizeof(float));;
                for(int x2020=0; x2020 < 150; x2020++) {
                  x2019[x2020] = 0.0f;

                }
                float* x2024 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x2025 = 0;
                int32_t x2026 = 0;
                int32_t x2027 = 0;
                for(int x2028=0; x2028 < 150; x2028++) {
                  int32_t x2029 = x2025;
                  int32_t x2030 = x2026;
                  float x2031 = x1980[x2030];
                  int32_t x2032 = x2027;
                  float x2033 = x2002[x2032];
                  float x2034 = x2031 + x2033;
                  x2024[x2029] = x2034;
                  x2025 += 1;
                  x2026 += 1;
                  x2027 += 1;

                }
                float* x2041 = (float*)myMalloc(150 * sizeof(float));;
                for(int x2042=0; x2042 < 150; x2042++) {
                  x2041[x2042] = 0.0f;

                }
                float* x2046 = (float*)myMalloc(150 * sizeof(float));;
                for(int x2047=0; x2047 < 150; x2047++) {
                  float x2048 = x2024[x2047];
                  double x2049 = (double)x2048;
                  double x2050 = tanh(x2049);
                  float x2051 = (float)x2050;
                  x2046[x2047] = x2051;

                }
                float* x2055 = (float*)myMalloc(150 * sizeof(float));;
                for(int x2056=0; x2056 < 150; x2056++) {
                  x2055[x2056] = 0.0f;

                }
                float* x2060 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x2061 = 0;
                int32_t x2062 = 0;
                int32_t x2063 = 0;
                for(int x2064=0; x2064 < 150; x2064++) {
                  int32_t x2065 = x2061;
                  int32_t x2066 = x2062;
                  float x2067 = x1817[x2066];
                  int32_t x2068 = x2063;
                  float x2069 = x2046[x2068];
                  float x2070 = x2067 * x2069;
                  x2060[x2065] = x2070;
                  x2061 += 1;
                  x2062 += 1;
                  x2063 += 1;

                }
                float* x2077 = (float*)myMalloc(150 * sizeof(float));;
                for(int x2078=0; x2078 < 150; x2078++) {
                  x2077[x2078] = 0.0f;

                }
                // dot: List(5, 150), List(150)
                float* x2083 = (float*)myMalloc(5 * sizeof(float));;
                for(int x2084=0; x2084 < 5; x2084++) {
                  float x2085 = 0.0f;
                  int32_t x2087 = x2084 * 150;
                  for(int x2086=0; x2086 < 150; x2086++) {
                    int32_t x2088 = x2087 + x2086;
                    float x2089 = x192[x2088];
                    float x2090 = x2060[x2086];
                    float x2091 = x2089 * x2090;
                    x2085 += x2091;

                  }
                  float x2095 = x2085;
                  x2083[x2084] = x2095;

                }
                float* x2099 = (float*)myMalloc(5 * sizeof(float));;
                for(int x2100=0; x2100 < 5; x2100++) {
                  x2099[x2100] = 0.0f;

                }
                float* x2104 = (float*)myMalloc(5 * sizeof(float));;
                int32_t x2105 = 0;
                int32_t x2106 = 0;
                int32_t x2107 = 0;
                for(int x2108=0; x2108 < 5; x2108++) {
                  int32_t x2109 = x2105;
                  int32_t x2110 = x2106;
                  float x2111 = x2083[x2110];
                  int32_t x2112 = x2107;
                  float x2113 = x201[x2112];
                  float x2114 = x2111 + x2113;
                  x2104[x2109] = x2114;
                  x2105 += 1;
                  x2106 += 1;
                  x2107 += 1;

                }
                float* x2121 = (float*)myMalloc(5 * sizeof(float));;
                for(int x2122=0; x2122 < 5; x2122++) {
                  x2121[x2122] = 0.0f;

                }
                float* x2126 = (float*)myMalloc(5 * sizeof(float));;
                for(int x2127=0; x2127 < 5; x2127++) {
                  float x2128 = x2104[x2127];
                  double x2129 = (double)x2128;
                  double x2130 = exp(x2129);
                  float x2131 = (float)x2130;
                  x2126[x2127] = x2131;

                }
                float* x2135 = (float*)myMalloc(5 * sizeof(float));;
                for(int x2136=0; x2136 < 5; x2136++) {
                  x2135[x2136] = 0.0f;

                }
                float x2140 = 0.0f;
                for(int x2141=0; x2141 < 5; x2141++) {
                  float x2142 = x2140;
                  float x2143 = x2126[x2141];
                  float x2144 = x2142 + x2143;
                  x2140 = x2144;

                }
                float x2148 = x2140;
                float* x2149 = (float*)myMalloc(1 * sizeof(float));;
                x2149[0] = x2148;
                float* x2151 = (float*)myMalloc(1 * sizeof(float));;
                for(int x2152=0; x2152 < 1; x2152++) {
                  x2151[x2152] = 0.0f;

                }
                float* x2156 = (float*)myMalloc(5 * sizeof(float));;
                int32_t x2157 = 0;
                int32_t x2158 = 0;
                int32_t x2159 = 0;
                for(int x2160=0; x2160 < 5; x2160++) {
                  int32_t x2161 = x2157;
                  int32_t x2162 = x2158;
                  float x2163 = x2126[x2162];
                  int32_t x2164 = x2159;
                  float x2165 = x2149[x2164];
                  float x2166 = x2163 / x2165;
                  x2156[x2161] = x2166;
                  x2157 += 1;
                  x2158 += 1;

                }
                float* x2172 = (float*)myMalloc(5 * sizeof(float));;
                for(int x2173=0; x2173 < 5; x2173++) {
                  x2172[x2173] = 0.0f;

                }
                float* x2177 = (float*)myMalloc(1 * sizeof(float));;
                int32_t x2178 = 0;
                int32_t x2179 = 0;
                int32_t x2180 = 0;
                int32_t x2181 = x2178;
                int32_t x2182 = x2179;
                float x2183 = x506[x2182];
                int32_t x2184 = x2180;
                float x2185 = x515[x2184];
                float x2186 = x2183 + x2185;
                x2177[x2181] = x2186;
                x2178 += 1;
                float* x2189 = (float*)myMalloc(1 * sizeof(float));;
                for(int x2190=0; x2190 < 1; x2190++) {
                  x2189[x2190] = 0.0f;

                }
                // dot: List(5), WrappedArray(5)
                float x2195 = 0.0f;
                for(int x2196=0; x2196 < 5; x2196++) {
                  float x2197 = x2156[x2196];
                  float x2198 = x521[x2196];
                  float x2199 = x2197 * x2198;
                  x2195 += x2199;

                }
                float* x2203 = (float*)myMalloc(1 * sizeof(float));;
                float x2204 = x2195;
                x2203[0] = x2204;
                float* x2206 = (float*)myMalloc(1 * sizeof(float));;
                for(int x2207=0; x2207 < 1; x2207++) {
                  x2206[x2207] = 0.0f;

                }
                float* x2211 = (float*)myMalloc(1 * sizeof(float));;
                float x2212 = x2203[0];
                double x2213 = (double)x2212;
                double x2214 = log(x2213);
                float x2215 = (float)x2214;
                x2211[0] = x2215;
                float* x2217 = (float*)myMalloc(1 * sizeof(float));;
                for(int x2218=0; x2218 < 1; x2218++) {
                  x2217[x2218] = 0.0f;

                }
                float* x2222 = (float*)myMalloc(1 * sizeof(float));;
                int32_t x2223 = 0;
                int32_t x2224 = 0;
                int32_t x2225 = 0;
                int32_t x2226 = x2223;
                int32_t x2227 = x2224;
                float x2228 = x2177[x2227];
                int32_t x2229 = x2225;
                float x2230 = x2211[x2229];
                float x2231 = x2228 - x2230;
                x2222[x2226] = x2231;
                x2223 += 1;
                float* x2234 = (float*)myMalloc(1 * sizeof(float));;
                for(int x2235=0; x2235 < 1; x2235++) {
                  x2234[x2235] = 0.0f;

                }
                float** x2239 = (float**)myMalloc(6 * sizeof(float*));;
                x2239[0] = x2222;
                x2239[1] = x2234;
                x2239[2] = x2060;
                x2239[3] = x2077;
                x2239[4] = x2024;
                x2239[5] = x2041;
                x937(x2239);
                int32_t x2247 = 0;
                int32_t x2248 = 0;
                int32_t x2249 = 0;
                int32_t x2250 = x2247;
                float x2251 = x2189[x2250];
                float x2252 = x2177[x2250];
                int32_t x2253 = x2248;
                float x2254 = x2211[x2253];
                int32_t x2255 = x2249;
                float x2256 = x2234[x2255];
                float x2257 = x2251 + x2256;
                x2189[x2250] = x2257;
                float x2259 = x2217[x2253];
                float x2260 = x2177[x2250];
                float x2261 = x2211[x2253];
                float x2262 = x2234[x2255];
                float x2263 = -1.0f * x2262;
                float x2264 = x2259 + x2263;
                x2217[x2253] = x2264;
                x2249 += 1;
                float x2267 = x2206[0];
                float x2268 = x2217[0];
                float x2269 = x2203[0];
                float x2270 = x2268 / x2269;
                float x2271 = x2267 + x2270;
                x2206[0] = x2271;
                float x2273 = x2206[0];
                // Generate code for addMul
                for(int x2275=0; x2275 < 5; x2275++) {
                  float x2276 = x2172[x2275];
                  float x2277 = x521[x2275];
                  float x2278 = x2273 * x2277;
                  float x2279 = x2276 + x2278;
                  x2172[x2275] = x2279;

                }
                float x2283 = x2206[0];
                // Generate code for addMul
                for(int x2285=0; x2285 < 5; x2285++) {
                  float x2286 = x528[x2285];
                  float x2287 = x2156[x2285];
                  float x2288 = x2283 * x2287;
                  float x2289 = x2286 + x2288;
                  x528[x2285] = x2289;

                }
                int32_t x2293 = 0;
                int32_t x2294 = 0;
                int32_t x2295 = 0;
                int32_t x2296 = x2293;
                float x2297 = x507[x2296];
                float x2298 = x506[x2296];
                int32_t x2299 = x2294;
                float x2300 = x515[x2299];
                int32_t x2301 = x2295;
                float x2302 = x2189[x2301];
                float x2303 = x2297 + x2302;
                x507[x2296] = x2303;
                float x2305 = x516[x2299];
                float x2306 = x506[x2296];
                float x2307 = x515[x2299];
                float x2308 = x2189[x2301];
                float x2309 = x2305 + x2308;
                x516[x2299] = x2309;
                x2295 += 1;
                int32_t x2312 = 0;
                int32_t x2313 = 0;
                int32_t x2314 = 0;
                for(int x2315=0; x2315 < 5; x2315++) {
                  int32_t x2316 = x2312;
                  float x2317 = x2135[x2316];
                  float x2318 = x2126[x2316];
                  int32_t x2319 = x2313;
                  float x2320 = x2149[x2319];
                  int32_t x2321 = x2314;
                  float x2322 = x2172[x2321];
                  float x2323 = x2322 / x2320;
                  float x2324 = x2317 + x2323;
                  x2135[x2316] = x2324;
                  float x2326 = x2151[x2319];
                  float x2327 = x2126[x2316];
                  float x2328 = x2149[x2319];
                  float x2329 = x2172[x2321];
                  float x2330 = -1.0f * x2327;
                  float x2331 = x2330 * x2329;
                  float x2332 = x2328 * x2328;
                  float x2333 = x2331 / x2332;
                  float x2334 = x2326 + x2333;
                  x2151[x2319] = x2334;
                  x2314 += 1;
                  x2312 += 1;

                }
                // += tensor of dim 0
                float x2341 = x2151[0];
                for(int x2342=0; x2342 < 5; x2342++) {
                  float x2343 = x2135[x2342];
                  float x2344 = x2343 + x2341;
                  x2135[x2342] = x2344;

                }
                for(int x2348=0; x2348 < 5; x2348++) {
                  float x2349 = x2121[x2348];
                  float x2350 = x2126[x2348];
                  float x2351 = x2135[x2348];
                  float x2352 = x2350 * x2351;
                  float x2353 = x2349 + x2352;
                  x2121[x2348] = x2353;

                }
                int32_t x2357 = 0;
                int32_t x2358 = 0;
                int32_t x2359 = 0;
                for(int x2360=0; x2360 < 5; x2360++) {
                  int32_t x2361 = x2357;
                  float x2362 = x2099[x2361];
                  float x2363 = x2083[x2361];
                  int32_t x2364 = x2358;
                  float x2365 = x201[x2364];
                  int32_t x2366 = x2359;
                  float x2367 = x2121[x2366];
                  float x2368 = x2362 + x2367;
                  x2099[x2361] = x2368;
                  float x2370 = x312[x2364];
                  float x2371 = x2083[x2361];
                  float x2372 = x201[x2364];
                  float x2373 = x2121[x2366];
                  float x2374 = x2370 + x2373;
                  x312[x2364] = x2374;
                  x2359 += 1;
                  x2357 += 1;
                  x2358 += 1;

                }
                // add_cartesian
                int32_t x2382 = 0;
                for(int x2383=0; x2383 < 5; x2383++) {
                  for(int x2384=0; x2384 < 150; x2384++) {
                    int32_t x2385 = x2382;
                    int32_t x2386 = x2385 + x2384;
                    float x2387 = x307[x2386];
                    float x2388 = x2060[x2384];
                    float x2389 = x2099[x2383];
                    float x2390 = x2388 * x2389;
                    float x2391 = x2387 + x2390;
                    x307[x2386] = x2391;

                  }
                  x2382 += 150;

                }
                int32_t x2398 = 0;
                for(int x2399=0; x2399 < 5; x2399++) {
                  for(int x2400=0; x2400 < 150; x2400++) {
                    float x2401 = x2077[x2400];
                    int32_t x2402 = x2398;
                    int32_t x2403 = x2402 + x2400;
                    float x2404 = x192[x2403];
                    float x2405 = x2099[x2399];
                    float x2406 = x2404 * x2405;
                    float x2407 = x2401 + x2406;
                    x2077[x2400] = x2407;

                  }
                  x2398 += 150;

                }
                int32_t x2414 = 0;
                int32_t x2415 = 0;
                int32_t x2416 = 0;
                for(int x2417=0; x2417 < 150; x2417++) {
                  int32_t x2418 = x2414;
                  float x2419 = x1829[x2418];
                  float x2420 = x1817[x2418];
                  int32_t x2421 = x2415;
                  float x2422 = x2046[x2421];
                  int32_t x2423 = x2416;
                  float x2424 = x2077[x2423];
                  float x2425 = x2424 * x2422;
                  float x2426 = x2419 + x2425;
                  x1829[x2418] = x2426;
                  float x2428 = x2055[x2421];
                  float x2429 = x1817[x2418];
                  float x2430 = x2046[x2421];
                  float x2431 = x2077[x2423];
                  float x2432 = x2431 * x2429;
                  float x2433 = x2428 + x2432;
                  x2055[x2421] = x2433;
                  x2416 += 1;
                  x2414 += 1;
                  x2415 += 1;

                }
                for(int x2440=0; x2440 < 150; x2440++) {
                  float x2441 = x2041[x2440];
                  float x2442 = x2046[x2440];
                  float x2445 = x2055[x2440];
                  float x2443 = x2442 * x2442;
                  float x2444 = 1.0f - x2443;
                  float x2446 = x2444 * x2445;
                  float x2447 = x2441 + x2446;
                  x2041[x2440] = x2447;

                }
                int32_t x2451 = 0;
                int32_t x2452 = 0;
                int32_t x2453 = 0;
                for(int x2454=0; x2454 < 150; x2454++) {
                  int32_t x2455 = x2451;
                  float x2456 = x1997[x2455];
                  float x2457 = x1980[x2455];
                  int32_t x2458 = x2452;
                  float x2459 = x2002[x2458];
                  int32_t x2460 = x2453;
                  float x2461 = x2041[x2460];
                  float x2462 = x2456 + x2461;
                  x1997[x2455] = x2462;
                  float x2464 = x2019[x2458];
                  float x2465 = x1980[x2455];
                  float x2466 = x2002[x2458];
                  float x2467 = x2041[x2460];
                  float x2468 = x2464 + x2467;
                  x2019[x2458] = x2468;
                  x2453 += 1;
                  x2451 += 1;
                  x2452 += 1;

                }
                int32_t x2475 = 0;
                int32_t x2476 = 0;
                int32_t x2477 = 0;
                for(int x2478=0; x2478 < 150; x2478++) {
                  int32_t x2479 = x2475;
                  float x2480 = x1724[x2479];
                  float x2481 = x1712[x2479];
                  int32_t x2482 = x2476;
                  float x2483 = x519[x2482];
                  int32_t x2484 = x2477;
                  float x2485 = x2019[x2484];
                  float x2486 = x2485 * x2483;
                  float x2487 = x2480 + x2486;
                  x1724[x2479] = x2487;
                  float x2489 = x520[x2482];
                  float x2490 = x1712[x2479];
                  float x2491 = x519[x2482];
                  float x2492 = x2019[x2484];
                  float x2493 = x2492 * x2490;
                  float x2494 = x2489 + x2493;
                  x520[x2482] = x2494;
                  x2477 += 1;
                  x2475 += 1;
                  x2476 += 1;

                }
                int32_t x2501 = 0;
                int32_t x2502 = 0;
                int32_t x2503 = 0;
                for(int x2504=0; x2504 < 150; x2504++) {
                  int32_t x2505 = x2501;
                  float x2506 = x1953[x2505];
                  float x2507 = x1936[x2505];
                  int32_t x2508 = x2502;
                  float x2509 = x1958[x2508];
                  int32_t x2510 = x2503;
                  float x2511 = x1997[x2510];
                  float x2512 = x2506 + x2511;
                  x1953[x2505] = x2512;
                  float x2514 = x1975[x2508];
                  float x2515 = x1936[x2505];
                  float x2516 = x1958[x2508];
                  float x2517 = x1997[x2510];
                  float x2518 = x2514 + x2517;
                  x1975[x2508] = x2518;
                  x2503 += 1;
                  x2501 += 1;
                  x2502 += 1;

                }
                int32_t x2525 = 0;
                int32_t x2526 = 0;
                int32_t x2527 = 0;
                for(int x2528=0; x2528 < 150; x2528++) {
                  int32_t x2529 = x2525;
                  float x2530 = x1619[x2529];
                  float x2531 = x1607[x2529];
                  int32_t x2532 = x2526;
                  float x2533 = x510[x2532];
                  int32_t x2534 = x2527;
                  float x2535 = x1975[x2534];
                  float x2536 = x2535 * x2533;
                  float x2537 = x2530 + x2536;
                  x1619[x2529] = x2537;
                  float x2539 = x511[x2532];
                  float x2540 = x1607[x2529];
                  float x2541 = x510[x2532];
                  float x2542 = x1975[x2534];
                  float x2543 = x2542 * x2540;
                  float x2544 = x2539 + x2543;
                  x511[x2532] = x2544;
                  x2527 += 1;
                  x2525 += 1;
                  x2526 += 1;

                }
                int32_t x2551 = 0;
                int32_t x2552 = 0;
                int32_t x2553 = 0;
                for(int x2554=0; x2554 < 150; x2554++) {
                  int32_t x2555 = x2551;
                  float x2556 = x1514[x2555];
                  float x2557 = x1502[x2555];
                  int32_t x2558 = x2552;
                  float x2559 = x1922[x2558];
                  int32_t x2560 = x2553;
                  float x2561 = x1953[x2560];
                  float x2562 = x2561 * x2559;
                  float x2563 = x2556 + x2562;
                  x1514[x2555] = x2563;
                  float x2565 = x1931[x2558];
                  float x2566 = x1502[x2555];
                  float x2567 = x1922[x2558];
                  float x2568 = x1953[x2560];
                  float x2569 = x2568 * x2566;
                  float x2570 = x2565 + x2569;
                  x1931[x2558] = x2570;
                  x2553 += 1;
                  x2551 += 1;
                  x2552 += 1;

                }
                for(int x2577=0; x2577 < 150; x2577++) {
                  float x2578 = x1917[x2577];
                  float x2579 = x1922[x2577];
                  float x2582 = x1931[x2577];
                  float x2580 = x2579 * x2579;
                  float x2581 = 1.0f - x2580;
                  float x2583 = x2581 * x2582;
                  float x2584 = x2578 + x2583;
                  x1917[x2577] = x2584;

                }
                int32_t x2588 = 0;
                int32_t x2589 = 0;
                int32_t x2590 = 0;
                for(int x2591=0; x2591 < 150; x2591++) {
                  int32_t x2592 = x2588;
                  float x2593 = x1895[x2592];
                  float x2594 = x1878[x2592];
                  int32_t x2595 = x2589;
                  float x2596 = x187[x2595];
                  int32_t x2597 = x2590;
                  float x2598 = x1917[x2597];
                  float x2599 = x2593 + x2598;
                  x1895[x2592] = x2599;
                  float x2601 = x302[x2595];
                  float x2602 = x1878[x2592];
                  float x2603 = x187[x2595];
                  float x2604 = x1917[x2597];
                  float x2605 = x2601 + x2604;
                  x302[x2595] = x2605;
                  x2590 += 1;
                  x2588 += 1;
                  x2589 += 1;

                }
                int32_t x2612 = 0;
                int32_t x2613 = 0;
                int32_t x2614 = 0;
                for(int x2615=0; x2615 < 150; x2615++) {
                  int32_t x2616 = x2612;
                  float x2617 = x1851[x2616];
                  float x2618 = x1835[x2616];
                  int32_t x2619 = x2613;
                  float x2620 = x1857[x2619];
                  int32_t x2621 = x2614;
                  float x2622 = x1895[x2621];
                  float x2623 = x2617 + x2622;
                  x1851[x2616] = x2623;
                  float x2625 = x1873[x2619];
                  float x2626 = x1835[x2616];
                  float x2627 = x1857[x2619];
                  float x2628 = x1895[x2621];
                  float x2629 = x2625 + x2628;
                  x1873[x2619] = x2629;
                  x2614 += 1;
                  x2612 += 1;
                  x2613 += 1;

                }
                // add_cartesian
                int32_t x2637 = 0;
                for(int x2638=0; x2638 < 150; x2638++) {
                  for(int x2639=0; x2639 < 150; x2639++) {
                    int32_t x2640 = x2637;
                    int32_t x2641 = x2640 + x2639;
                    float x2642 = x297[x2641];
                    float x2643 = x517[x2639];
                    float x2644 = x1873[x2638];
                    float x2645 = x2643 * x2644;
                    float x2646 = x2642 + x2645;
                    x297[x2641] = x2646;

                  }
                  x2637 += 150;

                }
                int32_t x2653 = 0;
                for(int x2654=0; x2654 < 150; x2654++) {
                  for(int x2655=0; x2655 < 150; x2655++) {
                    float x2656 = x518[x2655];
                    int32_t x2657 = x2653;
                    int32_t x2658 = x2657 + x2655;
                    float x2659 = x179[x2658];
                    float x2660 = x1873[x2654];
                    float x2661 = x2659 * x2660;
                    float x2662 = x2656 + x2661;
                    x518[x2655] = x2662;

                  }
                  x2653 += 150;

                }
                // add_cartesian
                int32_t x2670 = 0;
                for(int x2671=0; x2671 < 150; x2671++) {
                  for(int x2672=0; x2672 < 150; x2672++) {
                    int32_t x2673 = x2670;
                    int32_t x2674 = x2673 + x2672;
                    float x2675 = x292[x2674];
                    float x2676 = x508[x2672];
                    float x2677 = x1851[x2671];
                    float x2678 = x2676 * x2677;
                    float x2679 = x2675 + x2678;
                    x292[x2674] = x2679;

                  }
                  x2670 += 150;

                }
                int32_t x2686 = 0;
                for(int x2687=0; x2687 < 150; x2687++) {
                  for(int x2688=0; x2688 < 150; x2688++) {
                    float x2689 = x509[x2688];
                    int32_t x2690 = x2686;
                    int32_t x2691 = x2690 + x2688;
                    float x2692 = x171[x2691];
                    float x2693 = x1851[x2687];
                    float x2694 = x2692 * x2693;
                    float x2695 = x2689 + x2694;
                    x509[x2688] = x2695;

                  }
                  x2686 += 150;

                }
                for(int x2702=0; x2702 < 150; x2702++) {
                  float x2703 = x1812[x2702];
                  float x2704 = x1817[x2702];
                  float x2707 = x1829[x2702];
                  float x2705 = 1.0f - x2704;
                  float x2706 = x2705 * x2704;
                  float x2708 = x2706 * x2707;
                  float x2709 = x2703 + x2708;
                  x1812[x2702] = x2709;

                }
                int32_t x2713 = 0;
                int32_t x2714 = 0;
                int32_t x2715 = 0;
                for(int x2716=0; x2716 < 150; x2716++) {
                  int32_t x2717 = x2713;
                  float x2718 = x1790[x2717];
                  float x2719 = x1773[x2717];
                  int32_t x2720 = x2714;
                  float x2721 = x166[x2720];
                  int32_t x2722 = x2715;
                  float x2723 = x1812[x2722];
                  float x2724 = x2718 + x2723;
                  x1790[x2717] = x2724;
                  float x2726 = x287[x2720];
                  float x2727 = x1773[x2717];
                  float x2728 = x166[x2720];
                  float x2729 = x1812[x2722];
                  float x2730 = x2726 + x2729;
                  x287[x2720] = x2730;
                  x2715 += 1;
                  x2713 += 1;
                  x2714 += 1;

                }
                int32_t x2737 = 0;
                int32_t x2738 = 0;
                int32_t x2739 = 0;
                for(int x2740=0; x2740 < 150; x2740++) {
                  int32_t x2741 = x2737;
                  float x2742 = x1746[x2741];
                  float x2743 = x1730[x2741];
                  int32_t x2744 = x2738;
                  float x2745 = x1752[x2744];
                  int32_t x2746 = x2739;
                  float x2747 = x1790[x2746];
                  float x2748 = x2742 + x2747;
                  x1746[x2741] = x2748;
                  float x2750 = x1768[x2744];
                  float x2751 = x1730[x2741];
                  float x2752 = x1752[x2744];
                  float x2753 = x1790[x2746];
                  float x2754 = x2750 + x2753;
                  x1768[x2744] = x2754;
                  x2739 += 1;
                  x2737 += 1;
                  x2738 += 1;

                }
                // add_cartesian
                int32_t x2762 = 0;
                for(int x2763=0; x2763 < 150; x2763++) {
                  for(int x2764=0; x2764 < 150; x2764++) {
                    int32_t x2765 = x2762;
                    int32_t x2766 = x2765 + x2764;
                    float x2767 = x282[x2766];
                    float x2768 = x517[x2764];
                    float x2769 = x1768[x2763];
                    float x2770 = x2768 * x2769;
                    float x2771 = x2767 + x2770;
                    x282[x2766] = x2771;

                  }
                  x2762 += 150;

                }
                int32_t x2778 = 0;
                for(int x2779=0; x2779 < 150; x2779++) {
                  for(int x2780=0; x2780 < 150; x2780++) {
                    float x2781 = x518[x2780];
                    int32_t x2782 = x2778;
                    int32_t x2783 = x2782 + x2780;
                    float x2784 = x158[x2783];
                    float x2785 = x1768[x2779];
                    float x2786 = x2784 * x2785;
                    float x2787 = x2781 + x2786;
                    x518[x2780] = x2787;

                  }
                  x2778 += 150;

                }
                // add_cartesian
                int32_t x2795 = 0;
                for(int x2796=0; x2796 < 150; x2796++) {
                  for(int x2797=0; x2797 < 150; x2797++) {
                    int32_t x2798 = x2795;
                    int32_t x2799 = x2798 + x2797;
                    float x2800 = x277[x2799];
                    float x2801 = x508[x2797];
                    float x2802 = x1746[x2796];
                    float x2803 = x2801 * x2802;
                    float x2804 = x2800 + x2803;
                    x277[x2799] = x2804;

                  }
                  x2795 += 150;

                }
                int32_t x2811 = 0;
                for(int x2812=0; x2812 < 150; x2812++) {
                  for(int x2813=0; x2813 < 150; x2813++) {
                    float x2814 = x509[x2813];
                    int32_t x2815 = x2811;
                    int32_t x2816 = x2815 + x2813;
                    float x2817 = x150[x2816];
                    float x2818 = x1746[x2812];
                    float x2819 = x2817 * x2818;
                    float x2820 = x2814 + x2819;
                    x509[x2813] = x2820;

                  }
                  x2811 += 150;

                }
                for(int x2827=0; x2827 < 150; x2827++) {
                  float x2828 = x1707[x2827];
                  float x2829 = x1712[x2827];
                  float x2832 = x1724[x2827];
                  float x2830 = 1.0f - x2829;
                  float x2831 = x2830 * x2829;
                  float x2833 = x2831 * x2832;
                  float x2834 = x2828 + x2833;
                  x1707[x2827] = x2834;

                }
                int32_t x2838 = 0;
                int32_t x2839 = 0;
                int32_t x2840 = 0;
                for(int x2841=0; x2841 < 150; x2841++) {
                  int32_t x2842 = x2838;
                  float x2843 = x1685[x2842];
                  float x2844 = x1668[x2842];
                  int32_t x2845 = x2839;
                  float x2846 = x145[x2845];
                  int32_t x2847 = x2840;
                  float x2848 = x1707[x2847];
                  float x2849 = x2843 + x2848;
                  x1685[x2842] = x2849;
                  float x2851 = x272[x2845];
                  float x2852 = x1668[x2842];
                  float x2853 = x145[x2845];
                  float x2854 = x1707[x2847];
                  float x2855 = x2851 + x2854;
                  x272[x2845] = x2855;
                  x2840 += 1;
                  x2838 += 1;
                  x2839 += 1;

                }
                int32_t x2862 = 0;
                int32_t x2863 = 0;
                int32_t x2864 = 0;
                for(int x2865=0; x2865 < 150; x2865++) {
                  int32_t x2866 = x2862;
                  float x2867 = x1641[x2866];
                  float x2868 = x1625[x2866];
                  int32_t x2869 = x2863;
                  float x2870 = x1647[x2869];
                  int32_t x2871 = x2864;
                  float x2872 = x1685[x2871];
                  float x2873 = x2867 + x2872;
                  x1641[x2866] = x2873;
                  float x2875 = x1663[x2869];
                  float x2876 = x1625[x2866];
                  float x2877 = x1647[x2869];
                  float x2878 = x1685[x2871];
                  float x2879 = x2875 + x2878;
                  x1663[x2869] = x2879;
                  x2864 += 1;
                  x2862 += 1;
                  x2863 += 1;

                }
                // add_cartesian
                int32_t x2887 = 0;
                for(int x2888=0; x2888 < 150; x2888++) {
                  for(int x2889=0; x2889 < 150; x2889++) {
                    int32_t x2890 = x2887;
                    int32_t x2891 = x2890 + x2889;
                    float x2892 = x267[x2891];
                    float x2893 = x517[x2889];
                    float x2894 = x1663[x2888];
                    float x2895 = x2893 * x2894;
                    float x2896 = x2892 + x2895;
                    x267[x2891] = x2896;

                  }
                  x2887 += 150;

                }
                int32_t x2903 = 0;
                for(int x2904=0; x2904 < 150; x2904++) {
                  for(int x2905=0; x2905 < 150; x2905++) {
                    float x2906 = x518[x2905];
                    int32_t x2907 = x2903;
                    int32_t x2908 = x2907 + x2905;
                    float x2909 = x137[x2908];
                    float x2910 = x1663[x2904];
                    float x2911 = x2909 * x2910;
                    float x2912 = x2906 + x2911;
                    x518[x2905] = x2912;

                  }
                  x2903 += 150;

                }
                // add_cartesian
                int32_t x2920 = 0;
                for(int x2921=0; x2921 < 150; x2921++) {
                  for(int x2922=0; x2922 < 150; x2922++) {
                    int32_t x2923 = x2920;
                    int32_t x2924 = x2923 + x2922;
                    float x2925 = x262[x2924];
                    float x2926 = x508[x2922];
                    float x2927 = x1641[x2921];
                    float x2928 = x2926 * x2927;
                    float x2929 = x2925 + x2928;
                    x262[x2924] = x2929;

                  }
                  x2920 += 150;

                }
                int32_t x2936 = 0;
                for(int x2937=0; x2937 < 150; x2937++) {
                  for(int x2938=0; x2938 < 150; x2938++) {
                    float x2939 = x509[x2938];
                    int32_t x2940 = x2936;
                    int32_t x2941 = x2940 + x2938;
                    float x2942 = x129[x2941];
                    float x2943 = x1641[x2937];
                    float x2944 = x2942 * x2943;
                    float x2945 = x2939 + x2944;
                    x509[x2938] = x2945;

                  }
                  x2936 += 150;

                }
                for(int x2952=0; x2952 < 150; x2952++) {
                  float x2953 = x1602[x2952];
                  float x2954 = x1607[x2952];
                  float x2957 = x1619[x2952];
                  float x2955 = 1.0f - x2954;
                  float x2956 = x2955 * x2954;
                  float x2958 = x2956 * x2957;
                  float x2959 = x2953 + x2958;
                  x1602[x2952] = x2959;

                }
                int32_t x2963 = 0;
                int32_t x2964 = 0;
                int32_t x2965 = 0;
                for(int x2966=0; x2966 < 150; x2966++) {
                  int32_t x2967 = x2963;
                  float x2968 = x1580[x2967];
                  float x2969 = x1563[x2967];
                  int32_t x2970 = x2964;
                  float x2971 = x145[x2970];
                  int32_t x2972 = x2965;
                  float x2973 = x1602[x2972];
                  float x2974 = x2968 + x2973;
                  x1580[x2967] = x2974;
                  float x2976 = x272[x2970];
                  float x2977 = x1563[x2967];
                  float x2978 = x145[x2970];
                  float x2979 = x1602[x2972];
                  float x2980 = x2976 + x2979;
                  x272[x2970] = x2980;
                  x2965 += 1;
                  x2963 += 1;
                  x2964 += 1;

                }
                int32_t x2987 = 0;
                int32_t x2988 = 0;
                int32_t x2989 = 0;
                for(int x2990=0; x2990 < 150; x2990++) {
                  int32_t x2991 = x2987;
                  float x2992 = x1536[x2991];
                  float x2993 = x1520[x2991];
                  int32_t x2994 = x2988;
                  float x2995 = x1542[x2994];
                  int32_t x2996 = x2989;
                  float x2997 = x1580[x2996];
                  float x2998 = x2992 + x2997;
                  x1536[x2991] = x2998;
                  float x3000 = x1558[x2994];
                  float x3001 = x1520[x2991];
                  float x3002 = x1542[x2994];
                  float x3003 = x1580[x2996];
                  float x3004 = x3000 + x3003;
                  x1558[x2994] = x3004;
                  x2989 += 1;
                  x2987 += 1;
                  x2988 += 1;

                }
                // add_cartesian
                int32_t x3012 = 0;
                for(int x3013=0; x3013 < 150; x3013++) {
                  for(int x3014=0; x3014 < 150; x3014++) {
                    int32_t x3015 = x3012;
                    int32_t x3016 = x3015 + x3014;
                    float x3017 = x257[x3016];
                    float x3018 = x517[x3014];
                    float x3019 = x1558[x3013];
                    float x3020 = x3018 * x3019;
                    float x3021 = x3017 + x3020;
                    x257[x3016] = x3021;

                  }
                  x3012 += 150;

                }
                int32_t x3028 = 0;
                for(int x3029=0; x3029 < 150; x3029++) {
                  for(int x3030=0; x3030 < 150; x3030++) {
                    float x3031 = x518[x3030];
                    int32_t x3032 = x3028;
                    int32_t x3033 = x3032 + x3030;
                    float x3034 = x121[x3033];
                    float x3035 = x1558[x3029];
                    float x3036 = x3034 * x3035;
                    float x3037 = x3031 + x3036;
                    x518[x3030] = x3037;

                  }
                  x3028 += 150;

                }
                // add_cartesian
                int32_t x3045 = 0;
                for(int x3046=0; x3046 < 150; x3046++) {
                  for(int x3047=0; x3047 < 150; x3047++) {
                    int32_t x3048 = x3045;
                    int32_t x3049 = x3048 + x3047;
                    float x3050 = x252[x3049];
                    float x3051 = x508[x3047];
                    float x3052 = x1536[x3046];
                    float x3053 = x3051 * x3052;
                    float x3054 = x3050 + x3053;
                    x252[x3049] = x3054;

                  }
                  x3045 += 150;

                }
                int32_t x3061 = 0;
                for(int x3062=0; x3062 < 150; x3062++) {
                  for(int x3063=0; x3063 < 150; x3063++) {
                    float x3064 = x509[x3063];
                    int32_t x3065 = x3061;
                    int32_t x3066 = x3065 + x3063;
                    float x3067 = x113[x3066];
                    float x3068 = x1536[x3062];
                    float x3069 = x3067 * x3068;
                    float x3070 = x3064 + x3069;
                    x509[x3063] = x3070;

                  }
                  x3061 += 150;

                }
                for(int x3077=0; x3077 < 150; x3077++) {
                  float x3078 = x1497[x3077];
                  float x3079 = x1502[x3077];
                  float x3082 = x1514[x3077];
                  float x3080 = 1.0f - x3079;
                  float x3081 = x3080 * x3079;
                  float x3083 = x3081 * x3082;
                  float x3084 = x3078 + x3083;
                  x1497[x3077] = x3084;

                }
                int32_t x3088 = 0;
                int32_t x3089 = 0;
                int32_t x3090 = 0;
                for(int x3091=0; x3091 < 150; x3091++) {
                  int32_t x3092 = x3088;
                  float x3093 = x1475[x3092];
                  float x3094 = x1458[x3092];
                  int32_t x3095 = x3089;
                  float x3096 = x108[x3095];
                  int32_t x3097 = x3090;
                  float x3098 = x1497[x3097];
                  float x3099 = x3093 + x3098;
                  x1475[x3092] = x3099;
                  float x3101 = x247[x3095];
                  float x3102 = x1458[x3092];
                  float x3103 = x108[x3095];
                  float x3104 = x1497[x3097];
                  float x3105 = x3101 + x3104;
                  x247[x3095] = x3105;
                  x3090 += 1;
                  x3088 += 1;
                  x3089 += 1;

                }
                int32_t x3112 = 0;
                int32_t x3113 = 0;
                int32_t x3114 = 0;
                for(int x3115=0; x3115 < 150; x3115++) {
                  int32_t x3116 = x3112;
                  float x3117 = x1431[x3116];
                  float x3118 = x1415[x3116];
                  int32_t x3119 = x3113;
                  float x3120 = x1437[x3119];
                  int32_t x3121 = x3114;
                  float x3122 = x1475[x3121];
                  float x3123 = x3117 + x3122;
                  x1431[x3116] = x3123;
                  float x3125 = x1453[x3119];
                  float x3126 = x1415[x3116];
                  float x3127 = x1437[x3119];
                  float x3128 = x1475[x3121];
                  float x3129 = x3125 + x3128;
                  x1453[x3119] = x3129;
                  x3114 += 1;
                  x3112 += 1;
                  x3113 += 1;

                }
                // add_cartesian
                int32_t x3137 = 0;
                for(int x3138=0; x3138 < 150; x3138++) {
                  for(int x3139=0; x3139 < 150; x3139++) {
                    int32_t x3140 = x3137;
                    int32_t x3141 = x3140 + x3139;
                    float x3142 = x242[x3141];
                    float x3143 = x517[x3139];
                    float x3144 = x1453[x3138];
                    float x3145 = x3143 * x3144;
                    float x3146 = x3142 + x3145;
                    x242[x3141] = x3146;

                  }
                  x3137 += 150;

                }
                int32_t x3153 = 0;
                for(int x3154=0; x3154 < 150; x3154++) {
                  for(int x3155=0; x3155 < 150; x3155++) {
                    float x3156 = x518[x3155];
                    int32_t x3157 = x3153;
                    int32_t x3158 = x3157 + x3155;
                    float x3159 = x100[x3158];
                    float x3160 = x1453[x3154];
                    float x3161 = x3159 * x3160;
                    float x3162 = x3156 + x3161;
                    x518[x3155] = x3162;

                  }
                  x3153 += 150;

                }
                // add_cartesian
                int32_t x3170 = 0;
                for(int x3171=0; x3171 < 150; x3171++) {
                  for(int x3172=0; x3172 < 150; x3172++) {
                    int32_t x3173 = x3170;
                    int32_t x3174 = x3173 + x3172;
                    float x3175 = x237[x3174];
                    float x3176 = x508[x3172];
                    float x3177 = x1431[x3171];
                    float x3178 = x3176 * x3177;
                    float x3179 = x3175 + x3178;
                    x237[x3174] = x3179;

                  }
                  x3170 += 150;

                }
                int32_t x3186 = 0;
                for(int x3187=0; x3187 < 150; x3187++) {
                  for(int x3188=0; x3188 < 150; x3188++) {
                    float x3189 = x509[x3188];
                    int32_t x3190 = x3186;
                    int32_t x3191 = x3190 + x3188;
                    float x3192 = x91[x3191];
                    float x3193 = x1431[x3187];
                    float x3194 = x3192 * x3193;
                    float x3195 = x3189 + x3194;
                    x509[x3188] = x3195;

                  }
                  x3186 += 150;

                }
              }
            };
            x489(x512,x513,x3205);
          };
          x489(x503,x504,x3215);
        } else {
          float** x3242 = (float**)myMalloc(6 * sizeof(float*));;
          x3242[0] = x459;
          x3242[1] = x464;
          x3242[2] = x469;
          x3242[3] = x474;
          x3242[4] = x479;
          x3242[5] = x484;
          function<void(float**)> x494 = x491;
          function<void(float**)> x3225 = [&](float** x3226) {
            float* x3227 = x3226[0];
            float* x3228 = x3226[1];
            float* x3229 = x3226[2];
            float* x3230 = x3226[3];
            float* x3231 = x3226[4];
            float* x3232 = x3226[5];
            float** x3233 = (float**)myMalloc(6 * sizeof(float*));;
            x3233[0] = x3227;
            x3233[1] = x3228;
            x3233[2] = x3229;
            x3233[3] = x3230;
            x3233[4] = x3231;
            x3233[5] = x3232;
            x494(x3233);
          };
          x3225(x3242);
        }
      };
      float* x443 = (float*)myMalloc(1 * sizeof(float));;
      for(int x445=0; x445 < 1; x445++) {
        x443[x445] = 0.0f;

      }
      float* x449 = (float*)myMalloc(1 * sizeof(float));;
      for(int x450=0; x450 < 1; x450++) {
        x449[x450] = 0.0f;

      }
      float* x454 = (float*)myMalloc(1 * sizeof(float));;
      for(int x455=0; x455 < 1; x455++) {
        x454[x455] = 0.0f;

      }
      for(int x460=0; x460 < 1; x460++) {
        x459[x460] = 0.0f;

      }
      for(int x465=0; x465 < 1; x465++) {
        x464[x465] = 0.0f;

      }
      for(int x470=0; x470 < 150; x470++) {
        x469[x470] = 0.0f;

      }
      for(int x475=0; x475 < 150; x475++) {
        x474[x475] = 0.0f;

      }
      for(int x480=0; x480 < 150; x480++) {
        x479[x480] = 0.0f;

      }
      for(int x485=0; x485 < 150; x485++) {
        x484[x485] = 0.0f;

      }
      float** x3266 = (float**)myMalloc(6 * sizeof(float*));;
      x3266[0] = x459;
      x3266[1] = x464;
      x3266[2] = x469;
      x3266[3] = x474;
      x3266[4] = x479;
      x3266[5] = x484;
      function<void(float**)> x3253 = [&](float** x3254) {
        float* x3255 = x3254[0];
        float* x3256 = x3254[1];
        float* x3257 = x3254[2];
        float* x3258 = x3254[3];
        float* x3259 = x3254[4];
        float* x3260 = x3254[5];
        float x3261 = x3256[0];
        x3256[0] = 1.0f;
        float x3263 = x3255[0];
        x454[0] = x3263;
      };
      x489(0,x3253,x3266);
      float x3275 = x454[0];
      float x3276 = x432;
      float x3277 = (float)x433;
      float x3278 = x3276 * x3277;
      int32_t x3279 = x433 + 1;
      float x3280 = (float)x3279;
      float x3281 = x3278 / x3280;
      float x3282 = x3275 / x3280;
      float x3283 = x3281 + x3282;
      x432 = x3283;
      for(int x3285=0; x3285 < 45000; x3285++) {
        float x3286 = x207[x3285];
        bool x3287 = x3286 > 5.0f;
        if (x3287) {
          x207[x3285] = 5.0f;
        } else {
        }
        float x3291 = x207[x3285];
        bool x3292 = x3291 < -5.0f;
        if (x3292) {
          x207[x3285] = -5.0f;
        } else {
        }

      }
      float* x3298 = (float*)myMalloc(45000 * sizeof(float));;
      int32_t x3299 = 0;
      int32_t x3300 = 0;
      int32_t x3301 = 0;
      for(int x3302=0; x3302 < 150; x3302++) {
        int32_t x3303 = x3300;
        int32_t x3304 = x3301;
        int32_t x3305 = x3299;
        int32_t x3306 = x3305;
        int32_t x3307 = x3303;
        int32_t x3308 = x3304;
        for(int x3309=0; x3309 < 300; x3309++) {
          int32_t x3310 = x3306;
          int32_t x3311 = x3307;
          float x3312 = x207[x3311];
          int32_t x3313 = x3308;
          float x3314 = x207[x3313];
          float x3315 = x3312 * x3314;
          x3298[x3310] = x3315;
          x3306 += 1;
          x3307 += 1;
          x3308 += 1;

        }
        x3299 += 300;
        x3300 += 300;
        x3301 += 300;

      }
      for(int x3327=0; x3327 < 45000; x3327++) {
        float x3328 = x317[x3327];
        float x3329 = x3298[x3327];
        float x3330 = x3328 + x3329;
        x317[x3327] = x3330;

      }
      float* x3334 = (float*)myMalloc(45000 * sizeof(float));;
      for(int x3335=0; x3335 < 45000; x3335++) {
        float x3336 = x207[x3335];
        float x3337 = x3336 * 0.05f;
        x3334[x3335] = x3337;

      }
      float* x3341 = (float*)myMalloc(45000 * sizeof(float));;
      for(int x3342=0; x3342 < 45000; x3342++) {
        float x3343 = x317[x3342];
        float x3344 = x3343 + 1.0E-8f;
        x3341[x3342] = x3344;

      }
      float* x3348 = (float*)myMalloc(45000 * sizeof(float));;
      for(int x3349=0; x3349 < 45000; x3349++) {
        float x3350 = x3341[x3349];
        double x3351 = (double)x3350;
        double x3352 = sqrt(x3351);
        float x3353 = (float)x3352;
        x3348[x3349] = x3353;

      }
      float* x3357 = (float*)myMalloc(45000 * sizeof(float));;
      int32_t x3358 = 0;
      int32_t x3359 = 0;
      int32_t x3360 = 0;
      for(int x3361=0; x3361 < 150; x3361++) {
        int32_t x3362 = x3359;
        int32_t x3363 = x3360;
        int32_t x3364 = x3358;
        int32_t x3365 = x3364;
        int32_t x3366 = x3362;
        int32_t x3367 = x3363;
        for(int x3368=0; x3368 < 300; x3368++) {
          int32_t x3369 = x3365;
          int32_t x3370 = x3366;
          float x3371 = x3334[x3370];
          int32_t x3372 = x3367;
          float x3373 = x3348[x3372];
          float x3374 = x3371 / x3373;
          x3357[x3369] = x3374;
          x3365 += 1;
          x3366 += 1;
          x3367 += 1;

        }
        x3358 += 300;
        x3359 += 300;
        x3360 += 300;

      }
      for(int x3386=0; x3386 < 45000; x3386++) {
        float x3387 = x50[x3386];
        float x3388 = x3357[x3386];
        float x3389 = x3387 - x3388;
        x50[x3386] = x3389;

      }
      for(int x3393=0; x3393 < 45000; x3393++) {
        float x3394 = x207[x3393];
        x207[x3393] = 0.0f;

      }
      for(int x3398=0; x3398 < 150; x3398++) {
        float x3399 = x212[x3398];
        bool x3400 = x3399 > 5.0f;
        if (x3400) {
          x212[x3398] = 5.0f;
        } else {
        }
        float x3404 = x212[x3398];
        bool x3405 = x3404 < -5.0f;
        if (x3405) {
          x212[x3398] = -5.0f;
        } else {
        }

      }
      float* x3411 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x3412 = 0;
      int32_t x3413 = 0;
      int32_t x3414 = 0;
      for(int x3415=0; x3415 < 150; x3415++) {
        int32_t x3416 = x3412;
        int32_t x3417 = x3413;
        float x3418 = x212[x3417];
        int32_t x3419 = x3414;
        float x3420 = x212[x3419];
        float x3421 = x3418 * x3420;
        x3411[x3416] = x3421;
        x3412 += 1;
        x3413 += 1;
        x3414 += 1;

      }
      for(int x3428=0; x3428 < 150; x3428++) {
        float x3429 = x322[x3428];
        float x3430 = x3411[x3428];
        float x3431 = x3429 + x3430;
        x322[x3428] = x3431;

      }
      float* x3435 = (float*)myMalloc(150 * sizeof(float));;
      for(int x3436=0; x3436 < 150; x3436++) {
        float x3437 = x212[x3436];
        float x3438 = x3437 * 0.05f;
        x3435[x3436] = x3438;

      }
      float* x3442 = (float*)myMalloc(150 * sizeof(float));;
      for(int x3443=0; x3443 < 150; x3443++) {
        float x3444 = x322[x3443];
        float x3445 = x3444 + 1.0E-8f;
        x3442[x3443] = x3445;

      }
      float* x3449 = (float*)myMalloc(150 * sizeof(float));;
      for(int x3450=0; x3450 < 150; x3450++) {
        float x3451 = x3442[x3450];
        double x3452 = (double)x3451;
        double x3453 = sqrt(x3452);
        float x3454 = (float)x3453;
        x3449[x3450] = x3454;

      }
      float* x3458 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x3459 = 0;
      int32_t x3460 = 0;
      int32_t x3461 = 0;
      for(int x3462=0; x3462 < 150; x3462++) {
        int32_t x3463 = x3459;
        int32_t x3464 = x3460;
        float x3465 = x3435[x3464];
        int32_t x3466 = x3461;
        float x3467 = x3449[x3466];
        float x3468 = x3465 / x3467;
        x3458[x3463] = x3468;
        x3459 += 1;
        x3460 += 1;
        x3461 += 1;

      }
      for(int x3475=0; x3475 < 150; x3475++) {
        float x3476 = x59[x3475];
        float x3477 = x3458[x3475];
        float x3478 = x3476 - x3477;
        x59[x3475] = x3478;

      }
      for(int x3482=0; x3482 < 150; x3482++) {
        float x3483 = x212[x3482];
        x212[x3482] = 0.0f;

      }
      for(int x3487=0; x3487 < 45000; x3487++) {
        float x3488 = x217[x3487];
        bool x3489 = x3488 > 5.0f;
        if (x3489) {
          x217[x3487] = 5.0f;
        } else {
        }
        float x3493 = x217[x3487];
        bool x3494 = x3493 < -5.0f;
        if (x3494) {
          x217[x3487] = -5.0f;
        } else {
        }

      }
      float* x3500 = (float*)myMalloc(45000 * sizeof(float));;
      int32_t x3501 = 0;
      int32_t x3502 = 0;
      int32_t x3503 = 0;
      for(int x3504=0; x3504 < 150; x3504++) {
        int32_t x3505 = x3502;
        int32_t x3506 = x3503;
        int32_t x3507 = x3501;
        int32_t x3508 = x3507;
        int32_t x3509 = x3505;
        int32_t x3510 = x3506;
        for(int x3511=0; x3511 < 300; x3511++) {
          int32_t x3512 = x3508;
          int32_t x3513 = x3509;
          float x3514 = x217[x3513];
          int32_t x3515 = x3510;
          float x3516 = x217[x3515];
          float x3517 = x3514 * x3516;
          x3500[x3512] = x3517;
          x3508 += 1;
          x3509 += 1;
          x3510 += 1;

        }
        x3501 += 300;
        x3502 += 300;
        x3503 += 300;

      }
      for(int x3529=0; x3529 < 45000; x3529++) {
        float x3530 = x327[x3529];
        float x3531 = x3500[x3529];
        float x3532 = x3530 + x3531;
        x327[x3529] = x3532;

      }
      float* x3536 = (float*)myMalloc(45000 * sizeof(float));;
      for(int x3537=0; x3537 < 45000; x3537++) {
        float x3538 = x217[x3537];
        float x3539 = x3538 * 0.05f;
        x3536[x3537] = x3539;

      }
      float* x3543 = (float*)myMalloc(45000 * sizeof(float));;
      for(int x3544=0; x3544 < 45000; x3544++) {
        float x3545 = x327[x3544];
        float x3546 = x3545 + 1.0E-8f;
        x3543[x3544] = x3546;

      }
      float* x3550 = (float*)myMalloc(45000 * sizeof(float));;
      for(int x3551=0; x3551 < 45000; x3551++) {
        float x3552 = x3543[x3551];
        double x3553 = (double)x3552;
        double x3554 = sqrt(x3553);
        float x3555 = (float)x3554;
        x3550[x3551] = x3555;

      }
      float* x3559 = (float*)myMalloc(45000 * sizeof(float));;
      int32_t x3560 = 0;
      int32_t x3561 = 0;
      int32_t x3562 = 0;
      for(int x3563=0; x3563 < 150; x3563++) {
        int32_t x3564 = x3561;
        int32_t x3565 = x3562;
        int32_t x3566 = x3560;
        int32_t x3567 = x3566;
        int32_t x3568 = x3564;
        int32_t x3569 = x3565;
        for(int x3570=0; x3570 < 300; x3570++) {
          int32_t x3571 = x3567;
          int32_t x3572 = x3568;
          float x3573 = x3536[x3572];
          int32_t x3574 = x3569;
          float x3575 = x3550[x3574];
          float x3576 = x3573 / x3575;
          x3559[x3571] = x3576;
          x3567 += 1;
          x3568 += 1;
          x3569 += 1;

        }
        x3560 += 300;
        x3561 += 300;
        x3562 += 300;

      }
      for(int x3588=0; x3588 < 45000; x3588++) {
        float x3589 = x65[x3588];
        float x3590 = x3559[x3588];
        float x3591 = x3589 - x3590;
        x65[x3588] = x3591;

      }
      for(int x3595=0; x3595 < 45000; x3595++) {
        float x3596 = x217[x3595];
        x217[x3595] = 0.0f;

      }
      for(int x3600=0; x3600 < 150; x3600++) {
        float x3601 = x222[x3600];
        bool x3602 = x3601 > 5.0f;
        if (x3602) {
          x222[x3600] = 5.0f;
        } else {
        }
        float x3606 = x222[x3600];
        bool x3607 = x3606 < -5.0f;
        if (x3607) {
          x222[x3600] = -5.0f;
        } else {
        }

      }
      float* x3613 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x3614 = 0;
      int32_t x3615 = 0;
      int32_t x3616 = 0;
      for(int x3617=0; x3617 < 150; x3617++) {
        int32_t x3618 = x3614;
        int32_t x3619 = x3615;
        float x3620 = x222[x3619];
        int32_t x3621 = x3616;
        float x3622 = x222[x3621];
        float x3623 = x3620 * x3622;
        x3613[x3618] = x3623;
        x3614 += 1;
        x3615 += 1;
        x3616 += 1;

      }
      for(int x3630=0; x3630 < 150; x3630++) {
        float x3631 = x332[x3630];
        float x3632 = x3613[x3630];
        float x3633 = x3631 + x3632;
        x332[x3630] = x3633;

      }
      float* x3637 = (float*)myMalloc(150 * sizeof(float));;
      for(int x3638=0; x3638 < 150; x3638++) {
        float x3639 = x222[x3638];
        float x3640 = x3639 * 0.05f;
        x3637[x3638] = x3640;

      }
      float* x3644 = (float*)myMalloc(150 * sizeof(float));;
      for(int x3645=0; x3645 < 150; x3645++) {
        float x3646 = x332[x3645];
        float x3647 = x3646 + 1.0E-8f;
        x3644[x3645] = x3647;

      }
      float* x3651 = (float*)myMalloc(150 * sizeof(float));;
      for(int x3652=0; x3652 < 150; x3652++) {
        float x3653 = x3644[x3652];
        double x3654 = (double)x3653;
        double x3655 = sqrt(x3654);
        float x3656 = (float)x3655;
        x3651[x3652] = x3656;

      }
      float* x3660 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x3661 = 0;
      int32_t x3662 = 0;
      int32_t x3663 = 0;
      for(int x3664=0; x3664 < 150; x3664++) {
        int32_t x3665 = x3661;
        int32_t x3666 = x3662;
        float x3667 = x3637[x3666];
        int32_t x3668 = x3663;
        float x3669 = x3651[x3668];
        float x3670 = x3667 / x3669;
        x3660[x3665] = x3670;
        x3661 += 1;
        x3662 += 1;
        x3663 += 1;

      }
      for(int x3677=0; x3677 < 150; x3677++) {
        float x3678 = x73[x3677];
        float x3679 = x3660[x3677];
        float x3680 = x3678 - x3679;
        x73[x3677] = x3680;

      }
      for(int x3684=0; x3684 < 150; x3684++) {
        float x3685 = x222[x3684];
        x222[x3684] = 0.0f;

      }
      for(int x3689=0; x3689 < 45000; x3689++) {
        float x3690 = x227[x3689];
        bool x3691 = x3690 > 5.0f;
        if (x3691) {
          x227[x3689] = 5.0f;
        } else {
        }
        float x3695 = x227[x3689];
        bool x3696 = x3695 < -5.0f;
        if (x3696) {
          x227[x3689] = -5.0f;
        } else {
        }

      }
      float* x3702 = (float*)myMalloc(45000 * sizeof(float));;
      int32_t x3703 = 0;
      int32_t x3704 = 0;
      int32_t x3705 = 0;
      for(int x3706=0; x3706 < 150; x3706++) {
        int32_t x3707 = x3704;
        int32_t x3708 = x3705;
        int32_t x3709 = x3703;
        int32_t x3710 = x3709;
        int32_t x3711 = x3707;
        int32_t x3712 = x3708;
        for(int x3713=0; x3713 < 300; x3713++) {
          int32_t x3714 = x3710;
          int32_t x3715 = x3711;
          float x3716 = x227[x3715];
          int32_t x3717 = x3712;
          float x3718 = x227[x3717];
          float x3719 = x3716 * x3718;
          x3702[x3714] = x3719;
          x3710 += 1;
          x3711 += 1;
          x3712 += 1;

        }
        x3703 += 300;
        x3704 += 300;
        x3705 += 300;

      }
      for(int x3731=0; x3731 < 45000; x3731++) {
        float x3732 = x337[x3731];
        float x3733 = x3702[x3731];
        float x3734 = x3732 + x3733;
        x337[x3731] = x3734;

      }
      float* x3738 = (float*)myMalloc(45000 * sizeof(float));;
      for(int x3739=0; x3739 < 45000; x3739++) {
        float x3740 = x227[x3739];
        float x3741 = x3740 * 0.05f;
        x3738[x3739] = x3741;

      }
      float* x3745 = (float*)myMalloc(45000 * sizeof(float));;
      for(int x3746=0; x3746 < 45000; x3746++) {
        float x3747 = x337[x3746];
        float x3748 = x3747 + 1.0E-8f;
        x3745[x3746] = x3748;

      }
      float* x3752 = (float*)myMalloc(45000 * sizeof(float));;
      for(int x3753=0; x3753 < 45000; x3753++) {
        float x3754 = x3745[x3753];
        double x3755 = (double)x3754;
        double x3756 = sqrt(x3755);
        float x3757 = (float)x3756;
        x3752[x3753] = x3757;

      }
      float* x3761 = (float*)myMalloc(45000 * sizeof(float));;
      int32_t x3762 = 0;
      int32_t x3763 = 0;
      int32_t x3764 = 0;
      for(int x3765=0; x3765 < 150; x3765++) {
        int32_t x3766 = x3763;
        int32_t x3767 = x3764;
        int32_t x3768 = x3762;
        int32_t x3769 = x3768;
        int32_t x3770 = x3766;
        int32_t x3771 = x3767;
        for(int x3772=0; x3772 < 300; x3772++) {
          int32_t x3773 = x3769;
          int32_t x3774 = x3770;
          float x3775 = x3738[x3774];
          int32_t x3776 = x3771;
          float x3777 = x3752[x3776];
          float x3778 = x3775 / x3777;
          x3761[x3773] = x3778;
          x3769 += 1;
          x3770 += 1;
          x3771 += 1;

        }
        x3762 += 300;
        x3763 += 300;
        x3764 += 300;

      }
      for(int x3790=0; x3790 < 45000; x3790++) {
        float x3791 = x78[x3790];
        float x3792 = x3761[x3790];
        float x3793 = x3791 - x3792;
        x78[x3790] = x3793;

      }
      for(int x3797=0; x3797 < 45000; x3797++) {
        float x3798 = x227[x3797];
        x227[x3797] = 0.0f;

      }
      for(int x3802=0; x3802 < 150; x3802++) {
        float x3803 = x232[x3802];
        bool x3804 = x3803 > 5.0f;
        if (x3804) {
          x232[x3802] = 5.0f;
        } else {
        }
        float x3808 = x232[x3802];
        bool x3809 = x3808 < -5.0f;
        if (x3809) {
          x232[x3802] = -5.0f;
        } else {
        }

      }
      float* x3815 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x3816 = 0;
      int32_t x3817 = 0;
      int32_t x3818 = 0;
      for(int x3819=0; x3819 < 150; x3819++) {
        int32_t x3820 = x3816;
        int32_t x3821 = x3817;
        float x3822 = x232[x3821];
        int32_t x3823 = x3818;
        float x3824 = x232[x3823];
        float x3825 = x3822 * x3824;
        x3815[x3820] = x3825;
        x3816 += 1;
        x3817 += 1;
        x3818 += 1;

      }
      for(int x3832=0; x3832 < 150; x3832++) {
        float x3833 = x342[x3832];
        float x3834 = x3815[x3832];
        float x3835 = x3833 + x3834;
        x342[x3832] = x3835;

      }
      float* x3839 = (float*)myMalloc(150 * sizeof(float));;
      for(int x3840=0; x3840 < 150; x3840++) {
        float x3841 = x232[x3840];
        float x3842 = x3841 * 0.05f;
        x3839[x3840] = x3842;

      }
      float* x3846 = (float*)myMalloc(150 * sizeof(float));;
      for(int x3847=0; x3847 < 150; x3847++) {
        float x3848 = x342[x3847];
        float x3849 = x3848 + 1.0E-8f;
        x3846[x3847] = x3849;

      }
      float* x3853 = (float*)myMalloc(150 * sizeof(float));;
      for(int x3854=0; x3854 < 150; x3854++) {
        float x3855 = x3846[x3854];
        double x3856 = (double)x3855;
        double x3857 = sqrt(x3856);
        float x3858 = (float)x3857;
        x3853[x3854] = x3858;

      }
      float* x3862 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x3863 = 0;
      int32_t x3864 = 0;
      int32_t x3865 = 0;
      for(int x3866=0; x3866 < 150; x3866++) {
        int32_t x3867 = x3863;
        int32_t x3868 = x3864;
        float x3869 = x3839[x3868];
        int32_t x3870 = x3865;
        float x3871 = x3853[x3870];
        float x3872 = x3869 / x3871;
        x3862[x3867] = x3872;
        x3863 += 1;
        x3864 += 1;
        x3865 += 1;

      }
      for(int x3879=0; x3879 < 150; x3879++) {
        float x3880 = x86[x3879];
        float x3881 = x3862[x3879];
        float x3882 = x3880 - x3881;
        x86[x3879] = x3882;

      }
      for(int x3886=0; x3886 < 150; x3886++) {
        float x3887 = x232[x3886];
        x232[x3886] = 0.0f;

      }
      for(int x3891=0; x3891 < 22500; x3891++) {
        float x3892 = x237[x3891];
        bool x3893 = x3892 > 5.0f;
        if (x3893) {
          x237[x3891] = 5.0f;
        } else {
        }
        float x3897 = x237[x3891];
        bool x3898 = x3897 < -5.0f;
        if (x3898) {
          x237[x3891] = -5.0f;
        } else {
        }

      }
      float* x3904 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x3905 = 0;
      int32_t x3906 = 0;
      int32_t x3907 = 0;
      for(int x3908=0; x3908 < 150; x3908++) {
        int32_t x3909 = x3906;
        int32_t x3910 = x3907;
        int32_t x3911 = x3905;
        int32_t x3912 = x3911;
        int32_t x3913 = x3909;
        int32_t x3914 = x3910;
        for(int x3915=0; x3915 < 150; x3915++) {
          int32_t x3916 = x3912;
          int32_t x3917 = x3913;
          float x3918 = x237[x3917];
          int32_t x3919 = x3914;
          float x3920 = x237[x3919];
          float x3921 = x3918 * x3920;
          x3904[x3916] = x3921;
          x3912 += 1;
          x3913 += 1;
          x3914 += 1;

        }
        x3905 += 150;
        x3906 += 150;
        x3907 += 150;

      }
      for(int x3933=0; x3933 < 22500; x3933++) {
        float x3934 = x347[x3933];
        float x3935 = x3904[x3933];
        float x3936 = x3934 + x3935;
        x347[x3933] = x3936;

      }
      float* x3940 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x3941=0; x3941 < 22500; x3941++) {
        float x3942 = x237[x3941];
        float x3943 = x3942 * 0.05f;
        x3940[x3941] = x3943;

      }
      float* x3947 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x3948=0; x3948 < 22500; x3948++) {
        float x3949 = x347[x3948];
        float x3950 = x3949 + 1.0E-8f;
        x3947[x3948] = x3950;

      }
      float* x3954 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x3955=0; x3955 < 22500; x3955++) {
        float x3956 = x3947[x3955];
        double x3957 = (double)x3956;
        double x3958 = sqrt(x3957);
        float x3959 = (float)x3958;
        x3954[x3955] = x3959;

      }
      float* x3963 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x3964 = 0;
      int32_t x3965 = 0;
      int32_t x3966 = 0;
      for(int x3967=0; x3967 < 150; x3967++) {
        int32_t x3968 = x3965;
        int32_t x3969 = x3966;
        int32_t x3970 = x3964;
        int32_t x3971 = x3970;
        int32_t x3972 = x3968;
        int32_t x3973 = x3969;
        for(int x3974=0; x3974 < 150; x3974++) {
          int32_t x3975 = x3971;
          int32_t x3976 = x3972;
          float x3977 = x3940[x3976];
          int32_t x3978 = x3973;
          float x3979 = x3954[x3978];
          float x3980 = x3977 / x3979;
          x3963[x3975] = x3980;
          x3971 += 1;
          x3972 += 1;
          x3973 += 1;

        }
        x3964 += 150;
        x3965 += 150;
        x3966 += 150;

      }
      for(int x3992=0; x3992 < 22500; x3992++) {
        float x3993 = x91[x3992];
        float x3994 = x3963[x3992];
        float x3995 = x3993 - x3994;
        x91[x3992] = x3995;

      }
      for(int x3999=0; x3999 < 22500; x3999++) {
        float x4000 = x237[x3999];
        x237[x3999] = 0.0f;

      }
      for(int x4004=0; x4004 < 22500; x4004++) {
        float x4005 = x242[x4004];
        bool x4006 = x4005 > 5.0f;
        if (x4006) {
          x242[x4004] = 5.0f;
        } else {
        }
        float x4010 = x242[x4004];
        bool x4011 = x4010 < -5.0f;
        if (x4011) {
          x242[x4004] = -5.0f;
        } else {
        }

      }
      float* x4017 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x4018 = 0;
      int32_t x4019 = 0;
      int32_t x4020 = 0;
      for(int x4021=0; x4021 < 150; x4021++) {
        int32_t x4022 = x4019;
        int32_t x4023 = x4020;
        int32_t x4024 = x4018;
        int32_t x4025 = x4024;
        int32_t x4026 = x4022;
        int32_t x4027 = x4023;
        for(int x4028=0; x4028 < 150; x4028++) {
          int32_t x4029 = x4025;
          int32_t x4030 = x4026;
          float x4031 = x242[x4030];
          int32_t x4032 = x4027;
          float x4033 = x242[x4032];
          float x4034 = x4031 * x4033;
          x4017[x4029] = x4034;
          x4025 += 1;
          x4026 += 1;
          x4027 += 1;

        }
        x4018 += 150;
        x4019 += 150;
        x4020 += 150;

      }
      for(int x4046=0; x4046 < 22500; x4046++) {
        float x4047 = x352[x4046];
        float x4048 = x4017[x4046];
        float x4049 = x4047 + x4048;
        x352[x4046] = x4049;

      }
      float* x4053 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4054=0; x4054 < 22500; x4054++) {
        float x4055 = x242[x4054];
        float x4056 = x4055 * 0.05f;
        x4053[x4054] = x4056;

      }
      float* x4060 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4061=0; x4061 < 22500; x4061++) {
        float x4062 = x352[x4061];
        float x4063 = x4062 + 1.0E-8f;
        x4060[x4061] = x4063;

      }
      float* x4067 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4068=0; x4068 < 22500; x4068++) {
        float x4069 = x4060[x4068];
        double x4070 = (double)x4069;
        double x4071 = sqrt(x4070);
        float x4072 = (float)x4071;
        x4067[x4068] = x4072;

      }
      float* x4076 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x4077 = 0;
      int32_t x4078 = 0;
      int32_t x4079 = 0;
      for(int x4080=0; x4080 < 150; x4080++) {
        int32_t x4081 = x4078;
        int32_t x4082 = x4079;
        int32_t x4083 = x4077;
        int32_t x4084 = x4083;
        int32_t x4085 = x4081;
        int32_t x4086 = x4082;
        for(int x4087=0; x4087 < 150; x4087++) {
          int32_t x4088 = x4084;
          int32_t x4089 = x4085;
          float x4090 = x4053[x4089];
          int32_t x4091 = x4086;
          float x4092 = x4067[x4091];
          float x4093 = x4090 / x4092;
          x4076[x4088] = x4093;
          x4084 += 1;
          x4085 += 1;
          x4086 += 1;

        }
        x4077 += 150;
        x4078 += 150;
        x4079 += 150;

      }
      for(int x4105=0; x4105 < 22500; x4105++) {
        float x4106 = x100[x4105];
        float x4107 = x4076[x4105];
        float x4108 = x4106 - x4107;
        x100[x4105] = x4108;

      }
      for(int x4112=0; x4112 < 22500; x4112++) {
        float x4113 = x242[x4112];
        x242[x4112] = 0.0f;

      }
      for(int x4117=0; x4117 < 150; x4117++) {
        float x4118 = x247[x4117];
        bool x4119 = x4118 > 5.0f;
        if (x4119) {
          x247[x4117] = 5.0f;
        } else {
        }
        float x4123 = x247[x4117];
        bool x4124 = x4123 < -5.0f;
        if (x4124) {
          x247[x4117] = -5.0f;
        } else {
        }

      }
      float* x4130 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x4131 = 0;
      int32_t x4132 = 0;
      int32_t x4133 = 0;
      for(int x4134=0; x4134 < 150; x4134++) {
        int32_t x4135 = x4131;
        int32_t x4136 = x4132;
        float x4137 = x247[x4136];
        int32_t x4138 = x4133;
        float x4139 = x247[x4138];
        float x4140 = x4137 * x4139;
        x4130[x4135] = x4140;
        x4131 += 1;
        x4132 += 1;
        x4133 += 1;

      }
      for(int x4147=0; x4147 < 150; x4147++) {
        float x4148 = x357[x4147];
        float x4149 = x4130[x4147];
        float x4150 = x4148 + x4149;
        x357[x4147] = x4150;

      }
      float* x4154 = (float*)myMalloc(150 * sizeof(float));;
      for(int x4155=0; x4155 < 150; x4155++) {
        float x4156 = x247[x4155];
        float x4157 = x4156 * 0.05f;
        x4154[x4155] = x4157;

      }
      float* x4161 = (float*)myMalloc(150 * sizeof(float));;
      for(int x4162=0; x4162 < 150; x4162++) {
        float x4163 = x357[x4162];
        float x4164 = x4163 + 1.0E-8f;
        x4161[x4162] = x4164;

      }
      float* x4168 = (float*)myMalloc(150 * sizeof(float));;
      for(int x4169=0; x4169 < 150; x4169++) {
        float x4170 = x4161[x4169];
        double x4171 = (double)x4170;
        double x4172 = sqrt(x4171);
        float x4173 = (float)x4172;
        x4168[x4169] = x4173;

      }
      float* x4177 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x4178 = 0;
      int32_t x4179 = 0;
      int32_t x4180 = 0;
      for(int x4181=0; x4181 < 150; x4181++) {
        int32_t x4182 = x4178;
        int32_t x4183 = x4179;
        float x4184 = x4154[x4183];
        int32_t x4185 = x4180;
        float x4186 = x4168[x4185];
        float x4187 = x4184 / x4186;
        x4177[x4182] = x4187;
        x4178 += 1;
        x4179 += 1;
        x4180 += 1;

      }
      for(int x4194=0; x4194 < 150; x4194++) {
        float x4195 = x108[x4194];
        float x4196 = x4177[x4194];
        float x4197 = x4195 - x4196;
        x108[x4194] = x4197;

      }
      for(int x4201=0; x4201 < 150; x4201++) {
        float x4202 = x247[x4201];
        x247[x4201] = 0.0f;

      }
      for(int x4206=0; x4206 < 22500; x4206++) {
        float x4207 = x252[x4206];
        bool x4208 = x4207 > 5.0f;
        if (x4208) {
          x252[x4206] = 5.0f;
        } else {
        }
        float x4212 = x252[x4206];
        bool x4213 = x4212 < -5.0f;
        if (x4213) {
          x252[x4206] = -5.0f;
        } else {
        }

      }
      float* x4219 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x4220 = 0;
      int32_t x4221 = 0;
      int32_t x4222 = 0;
      for(int x4223=0; x4223 < 150; x4223++) {
        int32_t x4224 = x4221;
        int32_t x4225 = x4222;
        int32_t x4226 = x4220;
        int32_t x4227 = x4226;
        int32_t x4228 = x4224;
        int32_t x4229 = x4225;
        for(int x4230=0; x4230 < 150; x4230++) {
          int32_t x4231 = x4227;
          int32_t x4232 = x4228;
          float x4233 = x252[x4232];
          int32_t x4234 = x4229;
          float x4235 = x252[x4234];
          float x4236 = x4233 * x4235;
          x4219[x4231] = x4236;
          x4227 += 1;
          x4228 += 1;
          x4229 += 1;

        }
        x4220 += 150;
        x4221 += 150;
        x4222 += 150;

      }
      for(int x4248=0; x4248 < 22500; x4248++) {
        float x4249 = x362[x4248];
        float x4250 = x4219[x4248];
        float x4251 = x4249 + x4250;
        x362[x4248] = x4251;

      }
      float* x4255 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4256=0; x4256 < 22500; x4256++) {
        float x4257 = x252[x4256];
        float x4258 = x4257 * 0.05f;
        x4255[x4256] = x4258;

      }
      float* x4262 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4263=0; x4263 < 22500; x4263++) {
        float x4264 = x362[x4263];
        float x4265 = x4264 + 1.0E-8f;
        x4262[x4263] = x4265;

      }
      float* x4269 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4270=0; x4270 < 22500; x4270++) {
        float x4271 = x4262[x4270];
        double x4272 = (double)x4271;
        double x4273 = sqrt(x4272);
        float x4274 = (float)x4273;
        x4269[x4270] = x4274;

      }
      float* x4278 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x4279 = 0;
      int32_t x4280 = 0;
      int32_t x4281 = 0;
      for(int x4282=0; x4282 < 150; x4282++) {
        int32_t x4283 = x4280;
        int32_t x4284 = x4281;
        int32_t x4285 = x4279;
        int32_t x4286 = x4285;
        int32_t x4287 = x4283;
        int32_t x4288 = x4284;
        for(int x4289=0; x4289 < 150; x4289++) {
          int32_t x4290 = x4286;
          int32_t x4291 = x4287;
          float x4292 = x4255[x4291];
          int32_t x4293 = x4288;
          float x4294 = x4269[x4293];
          float x4295 = x4292 / x4294;
          x4278[x4290] = x4295;
          x4286 += 1;
          x4287 += 1;
          x4288 += 1;

        }
        x4279 += 150;
        x4280 += 150;
        x4281 += 150;

      }
      for(int x4307=0; x4307 < 22500; x4307++) {
        float x4308 = x113[x4307];
        float x4309 = x4278[x4307];
        float x4310 = x4308 - x4309;
        x113[x4307] = x4310;

      }
      for(int x4314=0; x4314 < 22500; x4314++) {
        float x4315 = x252[x4314];
        x252[x4314] = 0.0f;

      }
      for(int x4319=0; x4319 < 22500; x4319++) {
        float x4320 = x257[x4319];
        bool x4321 = x4320 > 5.0f;
        if (x4321) {
          x257[x4319] = 5.0f;
        } else {
        }
        float x4325 = x257[x4319];
        bool x4326 = x4325 < -5.0f;
        if (x4326) {
          x257[x4319] = -5.0f;
        } else {
        }

      }
      float* x4332 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x4333 = 0;
      int32_t x4334 = 0;
      int32_t x4335 = 0;
      for(int x4336=0; x4336 < 150; x4336++) {
        int32_t x4337 = x4334;
        int32_t x4338 = x4335;
        int32_t x4339 = x4333;
        int32_t x4340 = x4339;
        int32_t x4341 = x4337;
        int32_t x4342 = x4338;
        for(int x4343=0; x4343 < 150; x4343++) {
          int32_t x4344 = x4340;
          int32_t x4345 = x4341;
          float x4346 = x257[x4345];
          int32_t x4347 = x4342;
          float x4348 = x257[x4347];
          float x4349 = x4346 * x4348;
          x4332[x4344] = x4349;
          x4340 += 1;
          x4341 += 1;
          x4342 += 1;

        }
        x4333 += 150;
        x4334 += 150;
        x4335 += 150;

      }
      for(int x4361=0; x4361 < 22500; x4361++) {
        float x4362 = x367[x4361];
        float x4363 = x4332[x4361];
        float x4364 = x4362 + x4363;
        x367[x4361] = x4364;

      }
      float* x4368 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4369=0; x4369 < 22500; x4369++) {
        float x4370 = x257[x4369];
        float x4371 = x4370 * 0.05f;
        x4368[x4369] = x4371;

      }
      float* x4375 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4376=0; x4376 < 22500; x4376++) {
        float x4377 = x367[x4376];
        float x4378 = x4377 + 1.0E-8f;
        x4375[x4376] = x4378;

      }
      float* x4382 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4383=0; x4383 < 22500; x4383++) {
        float x4384 = x4375[x4383];
        double x4385 = (double)x4384;
        double x4386 = sqrt(x4385);
        float x4387 = (float)x4386;
        x4382[x4383] = x4387;

      }
      float* x4391 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x4392 = 0;
      int32_t x4393 = 0;
      int32_t x4394 = 0;
      for(int x4395=0; x4395 < 150; x4395++) {
        int32_t x4396 = x4393;
        int32_t x4397 = x4394;
        int32_t x4398 = x4392;
        int32_t x4399 = x4398;
        int32_t x4400 = x4396;
        int32_t x4401 = x4397;
        for(int x4402=0; x4402 < 150; x4402++) {
          int32_t x4403 = x4399;
          int32_t x4404 = x4400;
          float x4405 = x4368[x4404];
          int32_t x4406 = x4401;
          float x4407 = x4382[x4406];
          float x4408 = x4405 / x4407;
          x4391[x4403] = x4408;
          x4399 += 1;
          x4400 += 1;
          x4401 += 1;

        }
        x4392 += 150;
        x4393 += 150;
        x4394 += 150;

      }
      for(int x4420=0; x4420 < 22500; x4420++) {
        float x4421 = x121[x4420];
        float x4422 = x4391[x4420];
        float x4423 = x4421 - x4422;
        x121[x4420] = x4423;

      }
      for(int x4427=0; x4427 < 22500; x4427++) {
        float x4428 = x257[x4427];
        x257[x4427] = 0.0f;

      }
      for(int x4432=0; x4432 < 22500; x4432++) {
        float x4433 = x262[x4432];
        bool x4434 = x4433 > 5.0f;
        if (x4434) {
          x262[x4432] = 5.0f;
        } else {
        }
        float x4438 = x262[x4432];
        bool x4439 = x4438 < -5.0f;
        if (x4439) {
          x262[x4432] = -5.0f;
        } else {
        }

      }
      float* x4445 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x4446 = 0;
      int32_t x4447 = 0;
      int32_t x4448 = 0;
      for(int x4449=0; x4449 < 150; x4449++) {
        int32_t x4450 = x4447;
        int32_t x4451 = x4448;
        int32_t x4452 = x4446;
        int32_t x4453 = x4452;
        int32_t x4454 = x4450;
        int32_t x4455 = x4451;
        for(int x4456=0; x4456 < 150; x4456++) {
          int32_t x4457 = x4453;
          int32_t x4458 = x4454;
          float x4459 = x262[x4458];
          int32_t x4460 = x4455;
          float x4461 = x262[x4460];
          float x4462 = x4459 * x4461;
          x4445[x4457] = x4462;
          x4453 += 1;
          x4454 += 1;
          x4455 += 1;

        }
        x4446 += 150;
        x4447 += 150;
        x4448 += 150;

      }
      for(int x4474=0; x4474 < 22500; x4474++) {
        float x4475 = x372[x4474];
        float x4476 = x4445[x4474];
        float x4477 = x4475 + x4476;
        x372[x4474] = x4477;

      }
      float* x4481 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4482=0; x4482 < 22500; x4482++) {
        float x4483 = x262[x4482];
        float x4484 = x4483 * 0.05f;
        x4481[x4482] = x4484;

      }
      float* x4488 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4489=0; x4489 < 22500; x4489++) {
        float x4490 = x372[x4489];
        float x4491 = x4490 + 1.0E-8f;
        x4488[x4489] = x4491;

      }
      float* x4495 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4496=0; x4496 < 22500; x4496++) {
        float x4497 = x4488[x4496];
        double x4498 = (double)x4497;
        double x4499 = sqrt(x4498);
        float x4500 = (float)x4499;
        x4495[x4496] = x4500;

      }
      float* x4504 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x4505 = 0;
      int32_t x4506 = 0;
      int32_t x4507 = 0;
      for(int x4508=0; x4508 < 150; x4508++) {
        int32_t x4509 = x4506;
        int32_t x4510 = x4507;
        int32_t x4511 = x4505;
        int32_t x4512 = x4511;
        int32_t x4513 = x4509;
        int32_t x4514 = x4510;
        for(int x4515=0; x4515 < 150; x4515++) {
          int32_t x4516 = x4512;
          int32_t x4517 = x4513;
          float x4518 = x4481[x4517];
          int32_t x4519 = x4514;
          float x4520 = x4495[x4519];
          float x4521 = x4518 / x4520;
          x4504[x4516] = x4521;
          x4512 += 1;
          x4513 += 1;
          x4514 += 1;

        }
        x4505 += 150;
        x4506 += 150;
        x4507 += 150;

      }
      for(int x4533=0; x4533 < 22500; x4533++) {
        float x4534 = x129[x4533];
        float x4535 = x4504[x4533];
        float x4536 = x4534 - x4535;
        x129[x4533] = x4536;

      }
      for(int x4540=0; x4540 < 22500; x4540++) {
        float x4541 = x262[x4540];
        x262[x4540] = 0.0f;

      }
      for(int x4545=0; x4545 < 22500; x4545++) {
        float x4546 = x267[x4545];
        bool x4547 = x4546 > 5.0f;
        if (x4547) {
          x267[x4545] = 5.0f;
        } else {
        }
        float x4551 = x267[x4545];
        bool x4552 = x4551 < -5.0f;
        if (x4552) {
          x267[x4545] = -5.0f;
        } else {
        }

      }
      float* x4558 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x4559 = 0;
      int32_t x4560 = 0;
      int32_t x4561 = 0;
      for(int x4562=0; x4562 < 150; x4562++) {
        int32_t x4563 = x4560;
        int32_t x4564 = x4561;
        int32_t x4565 = x4559;
        int32_t x4566 = x4565;
        int32_t x4567 = x4563;
        int32_t x4568 = x4564;
        for(int x4569=0; x4569 < 150; x4569++) {
          int32_t x4570 = x4566;
          int32_t x4571 = x4567;
          float x4572 = x267[x4571];
          int32_t x4573 = x4568;
          float x4574 = x267[x4573];
          float x4575 = x4572 * x4574;
          x4558[x4570] = x4575;
          x4566 += 1;
          x4567 += 1;
          x4568 += 1;

        }
        x4559 += 150;
        x4560 += 150;
        x4561 += 150;

      }
      for(int x4587=0; x4587 < 22500; x4587++) {
        float x4588 = x377[x4587];
        float x4589 = x4558[x4587];
        float x4590 = x4588 + x4589;
        x377[x4587] = x4590;

      }
      float* x4594 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4595=0; x4595 < 22500; x4595++) {
        float x4596 = x267[x4595];
        float x4597 = x4596 * 0.05f;
        x4594[x4595] = x4597;

      }
      float* x4601 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4602=0; x4602 < 22500; x4602++) {
        float x4603 = x377[x4602];
        float x4604 = x4603 + 1.0E-8f;
        x4601[x4602] = x4604;

      }
      float* x4608 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4609=0; x4609 < 22500; x4609++) {
        float x4610 = x4601[x4609];
        double x4611 = (double)x4610;
        double x4612 = sqrt(x4611);
        float x4613 = (float)x4612;
        x4608[x4609] = x4613;

      }
      float* x4617 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x4618 = 0;
      int32_t x4619 = 0;
      int32_t x4620 = 0;
      for(int x4621=0; x4621 < 150; x4621++) {
        int32_t x4622 = x4619;
        int32_t x4623 = x4620;
        int32_t x4624 = x4618;
        int32_t x4625 = x4624;
        int32_t x4626 = x4622;
        int32_t x4627 = x4623;
        for(int x4628=0; x4628 < 150; x4628++) {
          int32_t x4629 = x4625;
          int32_t x4630 = x4626;
          float x4631 = x4594[x4630];
          int32_t x4632 = x4627;
          float x4633 = x4608[x4632];
          float x4634 = x4631 / x4633;
          x4617[x4629] = x4634;
          x4625 += 1;
          x4626 += 1;
          x4627 += 1;

        }
        x4618 += 150;
        x4619 += 150;
        x4620 += 150;

      }
      for(int x4646=0; x4646 < 22500; x4646++) {
        float x4647 = x137[x4646];
        float x4648 = x4617[x4646];
        float x4649 = x4647 - x4648;
        x137[x4646] = x4649;

      }
      for(int x4653=0; x4653 < 22500; x4653++) {
        float x4654 = x267[x4653];
        x267[x4653] = 0.0f;

      }
      for(int x4658=0; x4658 < 150; x4658++) {
        float x4659 = x272[x4658];
        bool x4660 = x4659 > 5.0f;
        if (x4660) {
          x272[x4658] = 5.0f;
        } else {
        }
        float x4664 = x272[x4658];
        bool x4665 = x4664 < -5.0f;
        if (x4665) {
          x272[x4658] = -5.0f;
        } else {
        }

      }
      float* x4671 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x4672 = 0;
      int32_t x4673 = 0;
      int32_t x4674 = 0;
      for(int x4675=0; x4675 < 150; x4675++) {
        int32_t x4676 = x4672;
        int32_t x4677 = x4673;
        float x4678 = x272[x4677];
        int32_t x4679 = x4674;
        float x4680 = x272[x4679];
        float x4681 = x4678 * x4680;
        x4671[x4676] = x4681;
        x4672 += 1;
        x4673 += 1;
        x4674 += 1;

      }
      for(int x4688=0; x4688 < 150; x4688++) {
        float x4689 = x382[x4688];
        float x4690 = x4671[x4688];
        float x4691 = x4689 + x4690;
        x382[x4688] = x4691;

      }
      float* x4695 = (float*)myMalloc(150 * sizeof(float));;
      for(int x4696=0; x4696 < 150; x4696++) {
        float x4697 = x272[x4696];
        float x4698 = x4697 * 0.05f;
        x4695[x4696] = x4698;

      }
      float* x4702 = (float*)myMalloc(150 * sizeof(float));;
      for(int x4703=0; x4703 < 150; x4703++) {
        float x4704 = x382[x4703];
        float x4705 = x4704 + 1.0E-8f;
        x4702[x4703] = x4705;

      }
      float* x4709 = (float*)myMalloc(150 * sizeof(float));;
      for(int x4710=0; x4710 < 150; x4710++) {
        float x4711 = x4702[x4710];
        double x4712 = (double)x4711;
        double x4713 = sqrt(x4712);
        float x4714 = (float)x4713;
        x4709[x4710] = x4714;

      }
      float* x4718 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x4719 = 0;
      int32_t x4720 = 0;
      int32_t x4721 = 0;
      for(int x4722=0; x4722 < 150; x4722++) {
        int32_t x4723 = x4719;
        int32_t x4724 = x4720;
        float x4725 = x4695[x4724];
        int32_t x4726 = x4721;
        float x4727 = x4709[x4726];
        float x4728 = x4725 / x4727;
        x4718[x4723] = x4728;
        x4719 += 1;
        x4720 += 1;
        x4721 += 1;

      }
      for(int x4735=0; x4735 < 150; x4735++) {
        float x4736 = x145[x4735];
        float x4737 = x4718[x4735];
        float x4738 = x4736 - x4737;
        x145[x4735] = x4738;

      }
      for(int x4742=0; x4742 < 150; x4742++) {
        float x4743 = x272[x4742];
        x272[x4742] = 0.0f;

      }
      for(int x4747=0; x4747 < 22500; x4747++) {
        float x4748 = x277[x4747];
        bool x4749 = x4748 > 5.0f;
        if (x4749) {
          x277[x4747] = 5.0f;
        } else {
        }
        float x4753 = x277[x4747];
        bool x4754 = x4753 < -5.0f;
        if (x4754) {
          x277[x4747] = -5.0f;
        } else {
        }

      }
      float* x4760 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x4761 = 0;
      int32_t x4762 = 0;
      int32_t x4763 = 0;
      for(int x4764=0; x4764 < 150; x4764++) {
        int32_t x4765 = x4762;
        int32_t x4766 = x4763;
        int32_t x4767 = x4761;
        int32_t x4768 = x4767;
        int32_t x4769 = x4765;
        int32_t x4770 = x4766;
        for(int x4771=0; x4771 < 150; x4771++) {
          int32_t x4772 = x4768;
          int32_t x4773 = x4769;
          float x4774 = x277[x4773];
          int32_t x4775 = x4770;
          float x4776 = x277[x4775];
          float x4777 = x4774 * x4776;
          x4760[x4772] = x4777;
          x4768 += 1;
          x4769 += 1;
          x4770 += 1;

        }
        x4761 += 150;
        x4762 += 150;
        x4763 += 150;

      }
      for(int x4789=0; x4789 < 22500; x4789++) {
        float x4790 = x387[x4789];
        float x4791 = x4760[x4789];
        float x4792 = x4790 + x4791;
        x387[x4789] = x4792;

      }
      float* x4796 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4797=0; x4797 < 22500; x4797++) {
        float x4798 = x277[x4797];
        float x4799 = x4798 * 0.05f;
        x4796[x4797] = x4799;

      }
      float* x4803 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4804=0; x4804 < 22500; x4804++) {
        float x4805 = x387[x4804];
        float x4806 = x4805 + 1.0E-8f;
        x4803[x4804] = x4806;

      }
      float* x4810 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4811=0; x4811 < 22500; x4811++) {
        float x4812 = x4803[x4811];
        double x4813 = (double)x4812;
        double x4814 = sqrt(x4813);
        float x4815 = (float)x4814;
        x4810[x4811] = x4815;

      }
      float* x4819 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x4820 = 0;
      int32_t x4821 = 0;
      int32_t x4822 = 0;
      for(int x4823=0; x4823 < 150; x4823++) {
        int32_t x4824 = x4821;
        int32_t x4825 = x4822;
        int32_t x4826 = x4820;
        int32_t x4827 = x4826;
        int32_t x4828 = x4824;
        int32_t x4829 = x4825;
        for(int x4830=0; x4830 < 150; x4830++) {
          int32_t x4831 = x4827;
          int32_t x4832 = x4828;
          float x4833 = x4796[x4832];
          int32_t x4834 = x4829;
          float x4835 = x4810[x4834];
          float x4836 = x4833 / x4835;
          x4819[x4831] = x4836;
          x4827 += 1;
          x4828 += 1;
          x4829 += 1;

        }
        x4820 += 150;
        x4821 += 150;
        x4822 += 150;

      }
      for(int x4848=0; x4848 < 22500; x4848++) {
        float x4849 = x150[x4848];
        float x4850 = x4819[x4848];
        float x4851 = x4849 - x4850;
        x150[x4848] = x4851;

      }
      for(int x4855=0; x4855 < 22500; x4855++) {
        float x4856 = x277[x4855];
        x277[x4855] = 0.0f;

      }
      for(int x4860=0; x4860 < 22500; x4860++) {
        float x4861 = x282[x4860];
        bool x4862 = x4861 > 5.0f;
        if (x4862) {
          x282[x4860] = 5.0f;
        } else {
        }
        float x4866 = x282[x4860];
        bool x4867 = x4866 < -5.0f;
        if (x4867) {
          x282[x4860] = -5.0f;
        } else {
        }

      }
      float* x4873 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x4874 = 0;
      int32_t x4875 = 0;
      int32_t x4876 = 0;
      for(int x4877=0; x4877 < 150; x4877++) {
        int32_t x4878 = x4875;
        int32_t x4879 = x4876;
        int32_t x4880 = x4874;
        int32_t x4881 = x4880;
        int32_t x4882 = x4878;
        int32_t x4883 = x4879;
        for(int x4884=0; x4884 < 150; x4884++) {
          int32_t x4885 = x4881;
          int32_t x4886 = x4882;
          float x4887 = x282[x4886];
          int32_t x4888 = x4883;
          float x4889 = x282[x4888];
          float x4890 = x4887 * x4889;
          x4873[x4885] = x4890;
          x4881 += 1;
          x4882 += 1;
          x4883 += 1;

        }
        x4874 += 150;
        x4875 += 150;
        x4876 += 150;

      }
      for(int x4902=0; x4902 < 22500; x4902++) {
        float x4903 = x392[x4902];
        float x4904 = x4873[x4902];
        float x4905 = x4903 + x4904;
        x392[x4902] = x4905;

      }
      float* x4909 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4910=0; x4910 < 22500; x4910++) {
        float x4911 = x282[x4910];
        float x4912 = x4911 * 0.05f;
        x4909[x4910] = x4912;

      }
      float* x4916 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4917=0; x4917 < 22500; x4917++) {
        float x4918 = x392[x4917];
        float x4919 = x4918 + 1.0E-8f;
        x4916[x4917] = x4919;

      }
      float* x4923 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x4924=0; x4924 < 22500; x4924++) {
        float x4925 = x4916[x4924];
        double x4926 = (double)x4925;
        double x4927 = sqrt(x4926);
        float x4928 = (float)x4927;
        x4923[x4924] = x4928;

      }
      float* x4932 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x4933 = 0;
      int32_t x4934 = 0;
      int32_t x4935 = 0;
      for(int x4936=0; x4936 < 150; x4936++) {
        int32_t x4937 = x4934;
        int32_t x4938 = x4935;
        int32_t x4939 = x4933;
        int32_t x4940 = x4939;
        int32_t x4941 = x4937;
        int32_t x4942 = x4938;
        for(int x4943=0; x4943 < 150; x4943++) {
          int32_t x4944 = x4940;
          int32_t x4945 = x4941;
          float x4946 = x4909[x4945];
          int32_t x4947 = x4942;
          float x4948 = x4923[x4947];
          float x4949 = x4946 / x4948;
          x4932[x4944] = x4949;
          x4940 += 1;
          x4941 += 1;
          x4942 += 1;

        }
        x4933 += 150;
        x4934 += 150;
        x4935 += 150;

      }
      for(int x4961=0; x4961 < 22500; x4961++) {
        float x4962 = x158[x4961];
        float x4963 = x4932[x4961];
        float x4964 = x4962 - x4963;
        x158[x4961] = x4964;

      }
      for(int x4968=0; x4968 < 22500; x4968++) {
        float x4969 = x282[x4968];
        x282[x4968] = 0.0f;

      }
      for(int x4973=0; x4973 < 150; x4973++) {
        float x4974 = x287[x4973];
        bool x4975 = x4974 > 5.0f;
        if (x4975) {
          x287[x4973] = 5.0f;
        } else {
        }
        float x4979 = x287[x4973];
        bool x4980 = x4979 < -5.0f;
        if (x4980) {
          x287[x4973] = -5.0f;
        } else {
        }

      }
      float* x4986 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x4987 = 0;
      int32_t x4988 = 0;
      int32_t x4989 = 0;
      for(int x4990=0; x4990 < 150; x4990++) {
        int32_t x4991 = x4987;
        int32_t x4992 = x4988;
        float x4993 = x287[x4992];
        int32_t x4994 = x4989;
        float x4995 = x287[x4994];
        float x4996 = x4993 * x4995;
        x4986[x4991] = x4996;
        x4987 += 1;
        x4988 += 1;
        x4989 += 1;

      }
      for(int x5003=0; x5003 < 150; x5003++) {
        float x5004 = x397[x5003];
        float x5005 = x4986[x5003];
        float x5006 = x5004 + x5005;
        x397[x5003] = x5006;

      }
      float* x5010 = (float*)myMalloc(150 * sizeof(float));;
      for(int x5011=0; x5011 < 150; x5011++) {
        float x5012 = x287[x5011];
        float x5013 = x5012 * 0.05f;
        x5010[x5011] = x5013;

      }
      float* x5017 = (float*)myMalloc(150 * sizeof(float));;
      for(int x5018=0; x5018 < 150; x5018++) {
        float x5019 = x397[x5018];
        float x5020 = x5019 + 1.0E-8f;
        x5017[x5018] = x5020;

      }
      float* x5024 = (float*)myMalloc(150 * sizeof(float));;
      for(int x5025=0; x5025 < 150; x5025++) {
        float x5026 = x5017[x5025];
        double x5027 = (double)x5026;
        double x5028 = sqrt(x5027);
        float x5029 = (float)x5028;
        x5024[x5025] = x5029;

      }
      float* x5033 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x5034 = 0;
      int32_t x5035 = 0;
      int32_t x5036 = 0;
      for(int x5037=0; x5037 < 150; x5037++) {
        int32_t x5038 = x5034;
        int32_t x5039 = x5035;
        float x5040 = x5010[x5039];
        int32_t x5041 = x5036;
        float x5042 = x5024[x5041];
        float x5043 = x5040 / x5042;
        x5033[x5038] = x5043;
        x5034 += 1;
        x5035 += 1;
        x5036 += 1;

      }
      for(int x5050=0; x5050 < 150; x5050++) {
        float x5051 = x166[x5050];
        float x5052 = x5033[x5050];
        float x5053 = x5051 - x5052;
        x166[x5050] = x5053;

      }
      for(int x5057=0; x5057 < 150; x5057++) {
        float x5058 = x287[x5057];
        x287[x5057] = 0.0f;

      }
      for(int x5062=0; x5062 < 22500; x5062++) {
        float x5063 = x292[x5062];
        bool x5064 = x5063 > 5.0f;
        if (x5064) {
          x292[x5062] = 5.0f;
        } else {
        }
        float x5068 = x292[x5062];
        bool x5069 = x5068 < -5.0f;
        if (x5069) {
          x292[x5062] = -5.0f;
        } else {
        }

      }
      float* x5075 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x5076 = 0;
      int32_t x5077 = 0;
      int32_t x5078 = 0;
      for(int x5079=0; x5079 < 150; x5079++) {
        int32_t x5080 = x5077;
        int32_t x5081 = x5078;
        int32_t x5082 = x5076;
        int32_t x5083 = x5082;
        int32_t x5084 = x5080;
        int32_t x5085 = x5081;
        for(int x5086=0; x5086 < 150; x5086++) {
          int32_t x5087 = x5083;
          int32_t x5088 = x5084;
          float x5089 = x292[x5088];
          int32_t x5090 = x5085;
          float x5091 = x292[x5090];
          float x5092 = x5089 * x5091;
          x5075[x5087] = x5092;
          x5083 += 1;
          x5084 += 1;
          x5085 += 1;

        }
        x5076 += 150;
        x5077 += 150;
        x5078 += 150;

      }
      for(int x5104=0; x5104 < 22500; x5104++) {
        float x5105 = x402[x5104];
        float x5106 = x5075[x5104];
        float x5107 = x5105 + x5106;
        x402[x5104] = x5107;

      }
      float* x5111 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x5112=0; x5112 < 22500; x5112++) {
        float x5113 = x292[x5112];
        float x5114 = x5113 * 0.05f;
        x5111[x5112] = x5114;

      }
      float* x5118 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x5119=0; x5119 < 22500; x5119++) {
        float x5120 = x402[x5119];
        float x5121 = x5120 + 1.0E-8f;
        x5118[x5119] = x5121;

      }
      float* x5125 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x5126=0; x5126 < 22500; x5126++) {
        float x5127 = x5118[x5126];
        double x5128 = (double)x5127;
        double x5129 = sqrt(x5128);
        float x5130 = (float)x5129;
        x5125[x5126] = x5130;

      }
      float* x5134 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x5135 = 0;
      int32_t x5136 = 0;
      int32_t x5137 = 0;
      for(int x5138=0; x5138 < 150; x5138++) {
        int32_t x5139 = x5136;
        int32_t x5140 = x5137;
        int32_t x5141 = x5135;
        int32_t x5142 = x5141;
        int32_t x5143 = x5139;
        int32_t x5144 = x5140;
        for(int x5145=0; x5145 < 150; x5145++) {
          int32_t x5146 = x5142;
          int32_t x5147 = x5143;
          float x5148 = x5111[x5147];
          int32_t x5149 = x5144;
          float x5150 = x5125[x5149];
          float x5151 = x5148 / x5150;
          x5134[x5146] = x5151;
          x5142 += 1;
          x5143 += 1;
          x5144 += 1;

        }
        x5135 += 150;
        x5136 += 150;
        x5137 += 150;

      }
      for(int x5163=0; x5163 < 22500; x5163++) {
        float x5164 = x171[x5163];
        float x5165 = x5134[x5163];
        float x5166 = x5164 - x5165;
        x171[x5163] = x5166;

      }
      for(int x5170=0; x5170 < 22500; x5170++) {
        float x5171 = x292[x5170];
        x292[x5170] = 0.0f;

      }
      for(int x5175=0; x5175 < 22500; x5175++) {
        float x5176 = x297[x5175];
        bool x5177 = x5176 > 5.0f;
        if (x5177) {
          x297[x5175] = 5.0f;
        } else {
        }
        float x5181 = x297[x5175];
        bool x5182 = x5181 < -5.0f;
        if (x5182) {
          x297[x5175] = -5.0f;
        } else {
        }

      }
      float* x5188 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x5189 = 0;
      int32_t x5190 = 0;
      int32_t x5191 = 0;
      for(int x5192=0; x5192 < 150; x5192++) {
        int32_t x5193 = x5190;
        int32_t x5194 = x5191;
        int32_t x5195 = x5189;
        int32_t x5196 = x5195;
        int32_t x5197 = x5193;
        int32_t x5198 = x5194;
        for(int x5199=0; x5199 < 150; x5199++) {
          int32_t x5200 = x5196;
          int32_t x5201 = x5197;
          float x5202 = x297[x5201];
          int32_t x5203 = x5198;
          float x5204 = x297[x5203];
          float x5205 = x5202 * x5204;
          x5188[x5200] = x5205;
          x5196 += 1;
          x5197 += 1;
          x5198 += 1;

        }
        x5189 += 150;
        x5190 += 150;
        x5191 += 150;

      }
      for(int x5217=0; x5217 < 22500; x5217++) {
        float x5218 = x407[x5217];
        float x5219 = x5188[x5217];
        float x5220 = x5218 + x5219;
        x407[x5217] = x5220;

      }
      float* x5224 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x5225=0; x5225 < 22500; x5225++) {
        float x5226 = x297[x5225];
        float x5227 = x5226 * 0.05f;
        x5224[x5225] = x5227;

      }
      float* x5231 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x5232=0; x5232 < 22500; x5232++) {
        float x5233 = x407[x5232];
        float x5234 = x5233 + 1.0E-8f;
        x5231[x5232] = x5234;

      }
      float* x5238 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x5239=0; x5239 < 22500; x5239++) {
        float x5240 = x5231[x5239];
        double x5241 = (double)x5240;
        double x5242 = sqrt(x5241);
        float x5243 = (float)x5242;
        x5238[x5239] = x5243;

      }
      float* x5247 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x5248 = 0;
      int32_t x5249 = 0;
      int32_t x5250 = 0;
      for(int x5251=0; x5251 < 150; x5251++) {
        int32_t x5252 = x5249;
        int32_t x5253 = x5250;
        int32_t x5254 = x5248;
        int32_t x5255 = x5254;
        int32_t x5256 = x5252;
        int32_t x5257 = x5253;
        for(int x5258=0; x5258 < 150; x5258++) {
          int32_t x5259 = x5255;
          int32_t x5260 = x5256;
          float x5261 = x5224[x5260];
          int32_t x5262 = x5257;
          float x5263 = x5238[x5262];
          float x5264 = x5261 / x5263;
          x5247[x5259] = x5264;
          x5255 += 1;
          x5256 += 1;
          x5257 += 1;

        }
        x5248 += 150;
        x5249 += 150;
        x5250 += 150;

      }
      for(int x5276=0; x5276 < 22500; x5276++) {
        float x5277 = x179[x5276];
        float x5278 = x5247[x5276];
        float x5279 = x5277 - x5278;
        x179[x5276] = x5279;

      }
      for(int x5283=0; x5283 < 22500; x5283++) {
        float x5284 = x297[x5283];
        x297[x5283] = 0.0f;

      }
      for(int x5288=0; x5288 < 150; x5288++) {
        float x5289 = x302[x5288];
        bool x5290 = x5289 > 5.0f;
        if (x5290) {
          x302[x5288] = 5.0f;
        } else {
        }
        float x5294 = x302[x5288];
        bool x5295 = x5294 < -5.0f;
        if (x5295) {
          x302[x5288] = -5.0f;
        } else {
        }

      }
      float* x5301 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x5302 = 0;
      int32_t x5303 = 0;
      int32_t x5304 = 0;
      for(int x5305=0; x5305 < 150; x5305++) {
        int32_t x5306 = x5302;
        int32_t x5307 = x5303;
        float x5308 = x302[x5307];
        int32_t x5309 = x5304;
        float x5310 = x302[x5309];
        float x5311 = x5308 * x5310;
        x5301[x5306] = x5311;
        x5302 += 1;
        x5303 += 1;
        x5304 += 1;

      }
      for(int x5318=0; x5318 < 150; x5318++) {
        float x5319 = x412[x5318];
        float x5320 = x5301[x5318];
        float x5321 = x5319 + x5320;
        x412[x5318] = x5321;

      }
      float* x5325 = (float*)myMalloc(150 * sizeof(float));;
      for(int x5326=0; x5326 < 150; x5326++) {
        float x5327 = x302[x5326];
        float x5328 = x5327 * 0.05f;
        x5325[x5326] = x5328;

      }
      float* x5332 = (float*)myMalloc(150 * sizeof(float));;
      for(int x5333=0; x5333 < 150; x5333++) {
        float x5334 = x412[x5333];
        float x5335 = x5334 + 1.0E-8f;
        x5332[x5333] = x5335;

      }
      float* x5339 = (float*)myMalloc(150 * sizeof(float));;
      for(int x5340=0; x5340 < 150; x5340++) {
        float x5341 = x5332[x5340];
        double x5342 = (double)x5341;
        double x5343 = sqrt(x5342);
        float x5344 = (float)x5343;
        x5339[x5340] = x5344;

      }
      float* x5348 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x5349 = 0;
      int32_t x5350 = 0;
      int32_t x5351 = 0;
      for(int x5352=0; x5352 < 150; x5352++) {
        int32_t x5353 = x5349;
        int32_t x5354 = x5350;
        float x5355 = x5325[x5354];
        int32_t x5356 = x5351;
        float x5357 = x5339[x5356];
        float x5358 = x5355 / x5357;
        x5348[x5353] = x5358;
        x5349 += 1;
        x5350 += 1;
        x5351 += 1;

      }
      for(int x5365=0; x5365 < 150; x5365++) {
        float x5366 = x187[x5365];
        float x5367 = x5348[x5365];
        float x5368 = x5366 - x5367;
        x187[x5365] = x5368;

      }
      for(int x5372=0; x5372 < 150; x5372++) {
        float x5373 = x302[x5372];
        x302[x5372] = 0.0f;

      }
      for(int x5377=0; x5377 < 750; x5377++) {
        float x5378 = x307[x5377];
        bool x5379 = x5378 > 5.0f;
        if (x5379) {
          x307[x5377] = 5.0f;
        } else {
        }
        float x5383 = x307[x5377];
        bool x5384 = x5383 < -5.0f;
        if (x5384) {
          x307[x5377] = -5.0f;
        } else {
        }

      }
      float* x5390 = (float*)myMalloc(750 * sizeof(float));;
      int32_t x5391 = 0;
      int32_t x5392 = 0;
      int32_t x5393 = 0;
      for(int x5394=0; x5394 < 5; x5394++) {
        int32_t x5395 = x5392;
        int32_t x5396 = x5393;
        int32_t x5397 = x5391;
        int32_t x5398 = x5397;
        int32_t x5399 = x5395;
        int32_t x5400 = x5396;
        for(int x5401=0; x5401 < 150; x5401++) {
          int32_t x5402 = x5398;
          int32_t x5403 = x5399;
          float x5404 = x307[x5403];
          int32_t x5405 = x5400;
          float x5406 = x307[x5405];
          float x5407 = x5404 * x5406;
          x5390[x5402] = x5407;
          x5398 += 1;
          x5399 += 1;
          x5400 += 1;

        }
        x5391 += 150;
        x5392 += 150;
        x5393 += 150;

      }
      for(int x5419=0; x5419 < 750; x5419++) {
        float x5420 = x417[x5419];
        float x5421 = x5390[x5419];
        float x5422 = x5420 + x5421;
        x417[x5419] = x5422;

      }
      float* x5426 = (float*)myMalloc(750 * sizeof(float));;
      for(int x5427=0; x5427 < 750; x5427++) {
        float x5428 = x307[x5427];
        float x5429 = x5428 * 0.05f;
        x5426[x5427] = x5429;

      }
      float* x5433 = (float*)myMalloc(750 * sizeof(float));;
      for(int x5434=0; x5434 < 750; x5434++) {
        float x5435 = x417[x5434];
        float x5436 = x5435 + 1.0E-8f;
        x5433[x5434] = x5436;

      }
      float* x5440 = (float*)myMalloc(750 * sizeof(float));;
      for(int x5441=0; x5441 < 750; x5441++) {
        float x5442 = x5433[x5441];
        double x5443 = (double)x5442;
        double x5444 = sqrt(x5443);
        float x5445 = (float)x5444;
        x5440[x5441] = x5445;

      }
      float* x5449 = (float*)myMalloc(750 * sizeof(float));;
      int32_t x5450 = 0;
      int32_t x5451 = 0;
      int32_t x5452 = 0;
      for(int x5453=0; x5453 < 5; x5453++) {
        int32_t x5454 = x5451;
        int32_t x5455 = x5452;
        int32_t x5456 = x5450;
        int32_t x5457 = x5456;
        int32_t x5458 = x5454;
        int32_t x5459 = x5455;
        for(int x5460=0; x5460 < 150; x5460++) {
          int32_t x5461 = x5457;
          int32_t x5462 = x5458;
          float x5463 = x5426[x5462];
          int32_t x5464 = x5459;
          float x5465 = x5440[x5464];
          float x5466 = x5463 / x5465;
          x5449[x5461] = x5466;
          x5457 += 1;
          x5458 += 1;
          x5459 += 1;

        }
        x5450 += 150;
        x5451 += 150;
        x5452 += 150;

      }
      for(int x5478=0; x5478 < 750; x5478++) {
        float x5479 = x192[x5478];
        float x5480 = x5449[x5478];
        float x5481 = x5479 - x5480;
        x192[x5478] = x5481;

      }
      for(int x5485=0; x5485 < 750; x5485++) {
        float x5486 = x307[x5485];
        x307[x5485] = 0.0f;

      }
      for(int x5490=0; x5490 < 5; x5490++) {
        float x5491 = x312[x5490];
        bool x5492 = x5491 > 5.0f;
        if (x5492) {
          x312[x5490] = 5.0f;
        } else {
        }
        float x5496 = x312[x5490];
        bool x5497 = x5496 < -5.0f;
        if (x5497) {
          x312[x5490] = -5.0f;
        } else {
        }

      }
      float* x5503 = (float*)myMalloc(5 * sizeof(float));;
      int32_t x5504 = 0;
      int32_t x5505 = 0;
      int32_t x5506 = 0;
      for(int x5507=0; x5507 < 5; x5507++) {
        int32_t x5508 = x5504;
        int32_t x5509 = x5505;
        float x5510 = x312[x5509];
        int32_t x5511 = x5506;
        float x5512 = x312[x5511];
        float x5513 = x5510 * x5512;
        x5503[x5508] = x5513;
        x5504 += 1;
        x5505 += 1;
        x5506 += 1;

      }
      for(int x5520=0; x5520 < 5; x5520++) {
        float x5521 = x422[x5520];
        float x5522 = x5503[x5520];
        float x5523 = x5521 + x5522;
        x422[x5520] = x5523;

      }
      float* x5527 = (float*)myMalloc(5 * sizeof(float));;
      for(int x5528=0; x5528 < 5; x5528++) {
        float x5529 = x312[x5528];
        float x5530 = x5529 * 0.05f;
        x5527[x5528] = x5530;

      }
      float* x5534 = (float*)myMalloc(5 * sizeof(float));;
      for(int x5535=0; x5535 < 5; x5535++) {
        float x5536 = x422[x5535];
        float x5537 = x5536 + 1.0E-8f;
        x5534[x5535] = x5537;

      }
      float* x5541 = (float*)myMalloc(5 * sizeof(float));;
      for(int x5542=0; x5542 < 5; x5542++) {
        float x5543 = x5534[x5542];
        double x5544 = (double)x5543;
        double x5545 = sqrt(x5544);
        float x5546 = (float)x5545;
        x5541[x5542] = x5546;

      }
      float* x5550 = (float*)myMalloc(5 * sizeof(float));;
      int32_t x5551 = 0;
      int32_t x5552 = 0;
      int32_t x5553 = 0;
      for(int x5554=0; x5554 < 5; x5554++) {
        int32_t x5555 = x5551;
        int32_t x5556 = x5552;
        float x5557 = x5527[x5556];
        int32_t x5558 = x5553;
        float x5559 = x5541[x5558];
        float x5560 = x5557 / x5559;
        x5550[x5555] = x5560;
        x5551 += 1;
        x5552 += 1;
        x5553 += 1;

      }
      for(int x5567=0; x5567 < 5; x5567++) {
        float x5568 = x201[x5567];
        float x5569 = x5550[x5567];
        float x5570 = x5568 - x5569;
        x201[x5567] = x5570;

      }
      for(int x5574=0; x5574 < 5; x5574++) {
        float x5575 = x312[x5574];
        x312[x5574] = 0.0f;

      }
      mallocAddr = (void*)x428;

    }
    float x5582 = x432;
    double x5583 = (double)x5582;
    x427[x431] = x5583;
    double x5585 = ((double)clock() / CLOCKS_PER_SEC);
    double x5586 = x5585 - x429;
    printf("epoc %d, average_loss %f, time %lf\n",x431,x5582,x5586);

  }
  double x5590 = ((double)clock() / CLOCKS_PER_SEC);
  int64_t x5594 = (long)fopen(x0, "w");
  fprintf((FILE *)x5594, "unit: %s\n", "1 epoch");
  for(int x5596=0; x5596 < 6; x5596++) {
    double x5597 = x427[x5596];
    fprintf((FILE *)x5594, "%lf\n", x5597);

  }
  double x5591 = x429 - x2;
  double x5592 = x5590 - x429;
  double x5593 = x5592 / 6.0;
  fprintf((FILE *)x5594, "run time: %lf %lf\n", x5591, x5593);
  fclose((FILE*)x5594);
  // Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
 *******************************************/

