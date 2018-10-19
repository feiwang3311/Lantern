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
  float* x65 = (float*)myMalloc(22500 * sizeof(float));;
  for(int x67=0; x67 < 22500; x67++) {
    float x68 = (float)rand()/RAND_MAX;
    float x69 = x68 - 0.5f;
    float x70 = x69 * 0.01f;
    x65[x67] = x70;

  }
  float* x74 = (float*)myMalloc(22500 * sizeof(float));;
  for(int x75=0; x75 < 22500; x75++) {
    float x76 = (float)rand()/RAND_MAX;
    float x77 = x76 - 0.5f;
    float x78 = x77 * 0.01f;
    x74[x75] = x78;

  }
  float* x82 = (float*)myMalloc(150 * sizeof(float));;
  for(int x83=0; x83 < 150; x83++) {
    x82[x83] = 0.0f;

  }
  float* x87 = (float*)myMalloc(750 * sizeof(float));;
  for(int x89=0; x89 < 750; x89++) {
    float x90 = (float)rand()/RAND_MAX;
    float x91 = x90 - 0.5f;
    float x92 = x91 * 0.01f;
    x87[x89] = x92;

  }
  float* x96 = (float*)myMalloc(5 * sizeof(float));;
  for(int x98=0; x98 < 5; x98++) {
    x96[x98] = 0.0f;

  }
  float* x102 = (float*)myMalloc(45000 * sizeof(float));;
  for(int x103=0; x103 < 45000; x103++) {
    x102[x103] = 0.0f;

  }
  float* x107 = (float*)myMalloc(150 * sizeof(float));;
  for(int x108=0; x108 < 150; x108++) {
    x107[x108] = 0.0f;

  }
  float* x112 = (float*)myMalloc(22500 * sizeof(float));;
  for(int x113=0; x113 < 22500; x113++) {
    x112[x113] = 0.0f;

  }
  float* x117 = (float*)myMalloc(22500 * sizeof(float));;
  for(int x118=0; x118 < 22500; x118++) {
    x117[x118] = 0.0f;

  }
  float* x122 = (float*)myMalloc(150 * sizeof(float));;
  for(int x123=0; x123 < 150; x123++) {
    x122[x123] = 0.0f;

  }
  float* x127 = (float*)myMalloc(750 * sizeof(float));;
  for(int x128=0; x128 < 750; x128++) {
    x127[x128] = 0.0f;

  }
  float* x132 = (float*)myMalloc(5 * sizeof(float));;
  for(int x133=0; x133 < 5; x133++) {
    x132[x133] = 0.0f;

  }
  float* x137 = (float*)myMalloc(45000 * sizeof(float));;
  for(int x138=0; x138 < 45000; x138++) {
    x137[x138] = 0.0f;

  }
  float* x142 = (float*)myMalloc(150 * sizeof(float));;
  for(int x143=0; x143 < 150; x143++) {
    x142[x143] = 0.0f;

  }
  float* x147 = (float*)myMalloc(22500 * sizeof(float));;
  for(int x148=0; x148 < 22500; x148++) {
    x147[x148] = 0.0f;

  }
  float* x152 = (float*)myMalloc(22500 * sizeof(float));;
  for(int x153=0; x153 < 22500; x153++) {
    x152[x153] = 0.0f;

  }
  float* x157 = (float*)myMalloc(150 * sizeof(float));;
  for(int x158=0; x158 < 150; x158++) {
    x157[x158] = 0.0f;

  }
  float* x162 = (float*)myMalloc(750 * sizeof(float));;
  for(int x163=0; x163 < 750; x163++) {
    x162[x163] = 0.0f;

  }
  float* x167 = (float*)myMalloc(5 * sizeof(float));;
  for(int x168=0; x168 < 5; x168++) {
    x167[x168] = 0.0f;

  }
  double* x172 = (double*)myMalloc(6 * sizeof(double));;
  int64_t x173 = (long)mallocAddr;
  double x174 = ((double)clock() / CLOCKS_PER_SEC);
  for(int x176=0; x176 < 6; x176++) {
    float x177 = 0.0f;
    for(int x178=0; x178 < x24; x178++) {
      float* x203 = (float*)myMalloc(1 * sizeof(float));;
      float* x208 = (float*)myMalloc(1 * sizeof(float));;
      float* x213 = (float*)myMalloc(150 * sizeof(float));;
      float* x218 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x179 = x178 * 4;
      int* x180 = x26[x179];
      int32_t x181 = x179 + 1;
      int* x182 = x26[x181];
      int32_t x183 = x179 + 2;
      int* x184 = x26[x183];
      int32_t x185 = x179 + 3;
      int* x186 = x26[x185];
      function<void(int32_t,function<void(float**)>,float**)> x223 = [&](int32_t x224,function<void(float**)> x225,float** x226) {
        float** x229 = x226;
        float* x230 = x229[0];
        float* x231 = x229[1];
        float* x232 = x229[2];
        float* x233 = x229[3];
        int32_t x227 = x224;
        bool x234 = x227 >= 0;
        if (x234) {
          int32_t x235 = x184[x227];
          float** x1121 = (float**)myMalloc(4 * sizeof(float*));;
          x1121[0] = x203;
          x1121[1] = x208;
          x1121[2] = x213;
          x1121[3] = x218;
          function<void(float**)> x228 = x225;
          function<void(float**)> x416 = [&](float** x417) {
            float* x418 = x417[0];
            float* x419 = x417[1];
            float* x420 = x417[2];
            float* x421 = x417[3];
            float** x422 = (float**)myMalloc(4 * sizeof(float*));;
            x422[0] = x418;
            x422[1] = x419;
            x422[2] = x420;
            x422[3] = x421;
            x228(x422);
          };
          function<void(float**)> x410 = [&](float** x411) {
            float* x412 = x411[0];
            float* x413 = x411[1];
            float* x414 = x411[2];
            float* x415 = x411[3];
            float** x429 = (float**)myMalloc(4 * sizeof(float*));;
            x429[0] = x412;
            x429[1] = x413;
            x429[2] = x414;
            x429[3] = x415;
            x416(x429);
          };
          function<void(float**)> x834 = [&](float** x835) {
            float* x836 = x835[0];
            float* x837 = x835[1];
            float* x838 = x835[2];
            float* x839 = x835[3];
            float** x840 = (float**)myMalloc(4 * sizeof(float*));;
            x840[0] = x836;
            x840[1] = x837;
            x840[2] = x838;
            x840[3] = x839;
            x228(x840);
          };
          function<void(float**)> x828 = [&](float** x829) {
            float* x830 = x829[0];
            float* x831 = x829[1];
            float* x832 = x829[2];
            float* x833 = x829[3];
            float** x847 = (float**)myMalloc(4 * sizeof(float*));;
            x847[0] = x830;
            x847[1] = x831;
            x847[2] = x832;
            x847[3] = x833;
            x834(x847);
          };
          function<void(float**)> x236 = [&](float** x237) {
            float* x238 = x237[0];
            float* x239 = x237[1];
            float* x240 = x237[2];
            float* x241 = x237[3];
            int32_t x242 = x186[x227];
            float** x1113 = (float**)myMalloc(4 * sizeof(float*));;
            x1113[0] = x203;
            x1113[1] = x208;
            x1113[2] = x213;
            x1113[3] = x218;
            function<void(float**)> x243 = [&](float** x244) {
              float* x245 = x244[0];
              float* x246 = x244[1];
              float* x247 = x244[2];
              float* x248 = x244[3];
              int32_t x249 = x184[x227];
              bool x250 = x249 < 0;
              if (x250) {
                int32_t x251 = x182[x227];
                float* x252 = x7[x251];
                float* x253 = (float*)myMalloc(300 * sizeof(float));;
                for(int x254=0; x254 < 300; x254++) {
                  x253[x254] = 0.0f;

                }
                // dot: List(150, 300), WrappedArray(300)
                float* x259 = (float*)myMalloc(150 * sizeof(float));;
                for(int x260=0; x260 < 150; x260++) {
                  float x261 = 0.0f;
                  int32_t x263 = x260 * 300;
                  for(int x262=0; x262 < 300; x262++) {
                    int32_t x264 = x263 + x262;
                    float x265 = x50[x264];
                    float x266 = x252[x262];
                    float x267 = x265 * x266;
                    x261 += x267;

                  }
                  float x271 = x261;
                  x259[x260] = x271;

                }
                float* x275 = (float*)myMalloc(150 * sizeof(float));;
                for(int x276=0; x276 < 150; x276++) {
                  x275[x276] = 0.0f;

                }
                float* x280 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x281 = 0;
                int32_t x282 = 0;
                int32_t x283 = 0;
                for(int x284=0; x284 < 150; x284++) {
                  int32_t x285 = x281;
                  int32_t x286 = x282;
                  float x287 = x259[x286];
                  int32_t x288 = x283;
                  float x289 = x59[x288];
                  float x290 = x287 + x289;
                  x280[x285] = x290;
                  x281 += 1;
                  x282 += 1;
                  x283 += 1;

                }
                float* x297 = (float*)myMalloc(150 * sizeof(float));;
                for(int x298=0; x298 < 150; x298++) {
                  x297[x298] = 0.0f;

                }
                float* x302 = (float*)myMalloc(150 * sizeof(float));;
                for(int x303=0; x303 < 150; x303++) {
                  float x304 = x280[x303];
                  double x305 = (double)x304;
                  double x306 = tanh(x305);
                  float x307 = (float)x306;
                  x302[x303] = x307;

                }
                float* x311 = (float*)myMalloc(150 * sizeof(float));;
                for(int x312=0; x312 < 150; x312++) {
                  x311[x312] = 0.0f;

                }
                // dot: List(5, 150), List(150)
                float* x317 = (float*)myMalloc(5 * sizeof(float));;
                for(int x318=0; x318 < 5; x318++) {
                  float x319 = 0.0f;
                  int32_t x321 = x318 * 150;
                  for(int x320=0; x320 < 150; x320++) {
                    int32_t x322 = x321 + x320;
                    float x323 = x87[x322];
                    float x324 = x302[x320];
                    float x325 = x323 * x324;
                    x319 += x325;

                  }
                  float x329 = x319;
                  x317[x318] = x329;

                }
                float* x333 = (float*)myMalloc(5 * sizeof(float));;
                for(int x334=0; x334 < 5; x334++) {
                  x333[x334] = 0.0f;

                }
                float* x338 = (float*)myMalloc(5 * sizeof(float));;
                int32_t x339 = 0;
                int32_t x340 = 0;
                int32_t x341 = 0;
                for(int x342=0; x342 < 5; x342++) {
                  int32_t x343 = x339;
                  int32_t x344 = x340;
                  float x345 = x317[x344];
                  int32_t x346 = x341;
                  float x347 = x96[x346];
                  float x348 = x345 + x347;
                  x338[x343] = x348;
                  x339 += 1;
                  x340 += 1;
                  x341 += 1;

                }
                float* x355 = (float*)myMalloc(5 * sizeof(float));;
                for(int x356=0; x356 < 5; x356++) {
                  x355[x356] = 0.0f;

                }
                float x360 = -3.4028235E38f;
                for(int x361=0; x361 < 5; x361++) {
                  float x362 = x360;
                  float x363 = x338[x361];
                  bool x364 = x363 > x362;
                  float x365;
                  if (x364) {
                    x365 = x363;
                  } else {
                    x365 = x362;
                  }
                  x360 = x365;

                }
                float x369 = x360;
                float x370 = 0.0f;
                for(int x371=0; x371 < 5; x371++) {
                  float x372 = x370;
                  float x373 = x338[x371];
                  float x374 = x360;
                  float x375 = x373 - x374;
                  double x376 = (double)x375;
                  double x377 = exp(x376);
                  float x378 = (float)x377;
                  float x379 = x372 + x378;
                  x370 = x379;

                }
                float x383 = x370;
                float* x388 = (float*)myMalloc(5 * sizeof(float));;
                double x384 = (double)x383;
                double x385 = log(x384);
                float x386 = (float)x385;
                float x387 = x369 + x386;
                for(int x389=0; x389 < 5; x389++) {
                  float x390 = x338[x389];
                  float x391 = x390 - x387;
                  x388[x389] = x391;

                }
                float* x395 = (float*)myMalloc(5 * sizeof(float));;
                for(int x396=0; x396 < 5; x396++) {
                  x395[x396] = 0.0f;

                }
                int32_t x400 = x180[x227];
                float x401 = x388[x400];
                float* x403 = (float*)myMalloc(1 * sizeof(float));;
                float x402 = -1.0f * x401;
                x403[0] = x402;
                float* x405 = (float*)myMalloc(1 * sizeof(float));;
                for(int x406=0; x406 < 1; x406++) {
                  x405[x406] = 0.0f;

                }
                float** x436 = (float**)myMalloc(4 * sizeof(float*));;
                x436[0] = x403;
                x436[1] = x405;
                x436[2] = x302;
                x436[3] = x311;
                x410(x436);
                float x442 = x395[x400];
                float x443 = x405[0];
                float x444 = -1.0f * x443;
                float x445 = x442 + x444;
                x395[x400] = x445;
                float x447 = 0.0f;
                for(int x448=0; x448 < 5; x448++) {
                  float x449 = x447;
                  float x450 = x395[x448];
                  float x451 = x449 + x450;
                  x447 = x451;

                }
                float x455 = x447;
                float* x456 = (float*)myMalloc(1 * sizeof(float));;
                x456[0] = x455;
                float x458 = x456[0];
                for(int x459=0; x459 < 5; x459++) {
                  float x460 = x355[x459];
                  float x461 = x395[x459];
                  float x462 = x388[x459];
                  double x463 = (double)x462;
                  double x464 = exp(x463);
                  float x465 = (float)x464;
                  float x466 = x465 * x458;
                  float x467 = x461 - x466;
                  float x468 = x460 + x467;
                  x355[x459] = x468;

                }
                int32_t x472 = 0;
                int32_t x473 = 0;
                int32_t x474 = 0;
                for(int x475=0; x475 < 5; x475++) {
                  int32_t x476 = x472;
                  float x477 = x333[x476];
                  float x478 = x317[x476];
                  int32_t x479 = x473;
                  float x480 = x96[x479];
                  int32_t x481 = x474;
                  float x482 = x355[x481];
                  float x483 = x477 + x482;
                  x333[x476] = x483;
                  float x485 = x132[x479];
                  float x486 = x317[x476];
                  float x487 = x96[x479];
                  float x488 = x355[x481];
                  float x489 = x485 + x488;
                  x132[x479] = x489;
                  x474 += 1;
                  x472 += 1;
                  x473 += 1;

                }
                // add_cartesian
                int32_t x497 = 0;
                for(int x498=0; x498 < 5; x498++) {
                  for(int x499=0; x499 < 150; x499++) {
                    int32_t x500 = x497;
                    int32_t x501 = x500 + x499;
                    float x502 = x127[x501];
                    float x503 = x302[x499];
                    float x504 = x333[x498];
                    float x505 = x503 * x504;
                    float x506 = x502 + x505;
                    x127[x501] = x506;

                  }
                  x497 += 150;

                }
                int32_t x513 = 0;
                for(int x514=0; x514 < 5; x514++) {
                  for(int x515=0; x515 < 150; x515++) {
                    float x516 = x311[x515];
                    int32_t x517 = x513;
                    int32_t x518 = x517 + x515;
                    float x519 = x87[x518];
                    float x520 = x333[x514];
                    float x521 = x519 * x520;
                    float x522 = x516 + x521;
                    x311[x515] = x522;

                  }
                  x513 += 150;

                }
                for(int x529=0; x529 < 150; x529++) {
                  float x530 = x297[x529];
                  float x531 = x302[x529];
                  float x534 = x311[x529];
                  float x532 = x531 * x531;
                  float x533 = 1.0f - x532;
                  float x535 = x533 * x534;
                  float x536 = x530 + x535;
                  x297[x529] = x536;

                }
                int32_t x540 = 0;
                int32_t x541 = 0;
                int32_t x542 = 0;
                for(int x543=0; x543 < 150; x543++) {
                  int32_t x544 = x540;
                  float x545 = x275[x544];
                  float x546 = x259[x544];
                  int32_t x547 = x541;
                  float x548 = x59[x547];
                  int32_t x549 = x542;
                  float x550 = x297[x549];
                  float x551 = x545 + x550;
                  x275[x544] = x551;
                  float x553 = x107[x547];
                  float x554 = x259[x544];
                  float x555 = x59[x547];
                  float x556 = x297[x549];
                  float x557 = x553 + x556;
                  x107[x547] = x557;
                  x542 += 1;
                  x540 += 1;
                  x541 += 1;

                }
                // add_cartesian
                int32_t x565 = 0;
                for(int x566=0; x566 < 150; x566++) {
                  for(int x567=0; x567 < 300; x567++) {
                    int32_t x568 = x565;
                    int32_t x569 = x568 + x567;
                    float x570 = x102[x569];
                    float x571 = x252[x567];
                    float x572 = x275[x566];
                    float x573 = x571 * x572;
                    float x574 = x570 + x573;
                    x102[x569] = x574;

                  }
                  x565 += 300;

                }
                int32_t x581 = 0;
                for(int x582=0; x582 < 150; x582++) {
                  for(int x583=0; x583 < 300; x583++) {
                    float x584 = x253[x583];
                    int32_t x585 = x581;
                    int32_t x586 = x585 + x583;
                    float x587 = x50[x586];
                    float x588 = x275[x582];
                    float x589 = x587 * x588;
                    float x590 = x584 + x589;
                    x253[x583] = x590;

                  }
                  x581 += 300;

                }
              } else {
                // dot: List(150, 150), WrappedArray(150)
                float* x599 = (float*)myMalloc(150 * sizeof(float));;
                for(int x600=0; x600 < 150; x600++) {
                  float x601 = 0.0f;
                  int32_t x603 = x600 * 150;
                  for(int x602=0; x602 < 150; x602++) {
                    int32_t x604 = x603 + x602;
                    float x605 = x65[x604];
                    float x606 = x240[x602];
                    float x607 = x605 * x606;
                    x601 += x607;

                  }
                  float x611 = x601;
                  x599[x600] = x611;

                }
                float* x615 = (float*)myMalloc(150 * sizeof(float));;
                for(int x616=0; x616 < 150; x616++) {
                  x615[x616] = 0.0f;

                }
                // dot: List(150, 150), WrappedArray(150)
                float* x621 = (float*)myMalloc(150 * sizeof(float));;
                for(int x622=0; x622 < 150; x622++) {
                  float x623 = 0.0f;
                  int32_t x625 = x622 * 150;
                  for(int x624=0; x624 < 150; x624++) {
                    int32_t x626 = x625 + x624;
                    float x627 = x74[x626];
                    float x628 = x247[x624];
                    float x629 = x627 * x628;
                    x623 += x629;

                  }
                  float x633 = x623;
                  x621[x622] = x633;

                }
                float* x637 = (float*)myMalloc(150 * sizeof(float));;
                for(int x638=0; x638 < 150; x638++) {
                  x637[x638] = 0.0f;

                }
                float* x642 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x643 = 0;
                int32_t x644 = 0;
                int32_t x645 = 0;
                for(int x646=0; x646 < 150; x646++) {
                  int32_t x647 = x643;
                  int32_t x648 = x644;
                  float x649 = x599[x648];
                  int32_t x650 = x645;
                  float x651 = x621[x650];
                  float x652 = x649 + x651;
                  x642[x647] = x652;
                  x643 += 1;
                  x644 += 1;
                  x645 += 1;

                }
                float* x659 = (float*)myMalloc(150 * sizeof(float));;
                for(int x660=0; x660 < 150; x660++) {
                  x659[x660] = 0.0f;

                }
                float* x664 = (float*)myMalloc(150 * sizeof(float));;
                int32_t x665 = 0;
                int32_t x666 = 0;
                int32_t x667 = 0;
                for(int x668=0; x668 < 150; x668++) {
                  int32_t x669 = x665;
                  int32_t x670 = x666;
                  float x671 = x642[x670];
                  int32_t x672 = x667;
                  float x673 = x82[x672];
                  float x674 = x671 + x673;
                  x664[x669] = x674;
                  x665 += 1;
                  x666 += 1;
                  x667 += 1;

                }
                float* x681 = (float*)myMalloc(150 * sizeof(float));;
                for(int x682=0; x682 < 150; x682++) {
                  x681[x682] = 0.0f;

                }
                float* x686 = (float*)myMalloc(150 * sizeof(float));;
                for(int x687=0; x687 < 150; x687++) {
                  float x688 = x664[x687];
                  double x689 = (double)x688;
                  double x690 = tanh(x689);
                  float x691 = (float)x690;
                  x686[x687] = x691;

                }
                float* x695 = (float*)myMalloc(150 * sizeof(float));;
                for(int x696=0; x696 < 150; x696++) {
                  x695[x696] = 0.0f;

                }
                // dot: List(5, 150), List(150)
                float* x701 = (float*)myMalloc(5 * sizeof(float));;
                for(int x702=0; x702 < 5; x702++) {
                  float x703 = 0.0f;
                  int32_t x705 = x702 * 150;
                  for(int x704=0; x704 < 150; x704++) {
                    int32_t x706 = x705 + x704;
                    float x707 = x87[x706];
                    float x708 = x686[x704];
                    float x709 = x707 * x708;
                    x703 += x709;

                  }
                  float x713 = x703;
                  x701[x702] = x713;

                }
                float* x717 = (float*)myMalloc(5 * sizeof(float));;
                for(int x718=0; x718 < 5; x718++) {
                  x717[x718] = 0.0f;

                }
                float* x722 = (float*)myMalloc(5 * sizeof(float));;
                int32_t x723 = 0;
                int32_t x724 = 0;
                int32_t x725 = 0;
                for(int x726=0; x726 < 5; x726++) {
                  int32_t x727 = x723;
                  int32_t x728 = x724;
                  float x729 = x701[x728];
                  int32_t x730 = x725;
                  float x731 = x96[x730];
                  float x732 = x729 + x731;
                  x722[x727] = x732;
                  x723 += 1;
                  x724 += 1;
                  x725 += 1;

                }
                float* x739 = (float*)myMalloc(5 * sizeof(float));;
                for(int x740=0; x740 < 5; x740++) {
                  x739[x740] = 0.0f;

                }
                float* x744 = (float*)myMalloc(1 * sizeof(float));;
                int32_t x745 = 0;
                int32_t x746 = 0;
                int32_t x747 = 0;
                int32_t x748 = x745;
                int32_t x749 = x746;
                float x750 = x238[x749];
                int32_t x751 = x747;
                float x752 = x245[x751];
                float x753 = x750 + x752;
                x744[x748] = x753;
                x745 += 1;
                float* x756 = (float*)myMalloc(1 * sizeof(float));;
                for(int x757=0; x757 < 1; x757++) {
                  x756[x757] = 0.0f;

                }
                float x761 = -3.4028235E38f;
                for(int x762=0; x762 < 5; x762++) {
                  float x763 = x761;
                  float x764 = x722[x762];
                  bool x765 = x764 > x763;
                  float x766;
                  if (x765) {
                    x766 = x764;
                  } else {
                    x766 = x763;
                  }
                  x761 = x766;

                }
                float x770 = x761;
                float x771 = 0.0f;
                for(int x772=0; x772 < 5; x772++) {
                  float x773 = x771;
                  float x774 = x722[x772];
                  float x775 = x761;
                  float x776 = x774 - x775;
                  double x777 = (double)x776;
                  double x778 = exp(x777);
                  float x779 = (float)x778;
                  float x780 = x773 + x779;
                  x771 = x780;

                }
                float x784 = x771;
                float* x789 = (float*)myMalloc(5 * sizeof(float));;
                double x785 = (double)x784;
                double x786 = log(x785);
                float x787 = (float)x786;
                float x788 = x770 + x787;
                for(int x790=0; x790 < 5; x790++) {
                  float x791 = x722[x790];
                  float x792 = x791 - x788;
                  x789[x790] = x792;

                }
                float* x796 = (float*)myMalloc(5 * sizeof(float));;
                for(int x797=0; x797 < 5; x797++) {
                  x796[x797] = 0.0f;

                }
                int32_t x801 = x180[x227];
                float x802 = x789[x801];
                float* x804 = (float*)myMalloc(1 * sizeof(float));;
                float x803 = -1.0f * x802;
                x804[0] = x803;
                float* x806 = (float*)myMalloc(1 * sizeof(float));;
                for(int x807=0; x807 < 1; x807++) {
                  x806[x807] = 0.0f;

                }
                float* x811 = (float*)myMalloc(1 * sizeof(float));;
                int32_t x812 = 0;
                int32_t x813 = 0;
                int32_t x814 = 0;
                int32_t x815 = x812;
                int32_t x816 = x813;
                float x817 = x744[x816];
                int32_t x818 = x814;
                float x819 = x804[x818];
                float x820 = x817 + x819;
                x811[x815] = x820;
                x812 += 1;
                float* x823 = (float*)myMalloc(1 * sizeof(float));;
                for(int x824=0; x824 < 1; x824++) {
                  x823[x824] = 0.0f;

                }
                float** x854 = (float**)myMalloc(4 * sizeof(float*));;
                x854[0] = x811;
                x854[1] = x823;
                x854[2] = x686;
                x854[3] = x695;
                x828(x854);
                int32_t x860 = 0;
                int32_t x861 = 0;
                int32_t x862 = 0;
                int32_t x863 = x860;
                float x864 = x756[x863];
                float x865 = x744[x863];
                int32_t x866 = x861;
                float x867 = x804[x866];
                int32_t x868 = x862;
                float x869 = x823[x868];
                float x870 = x864 + x869;
                x756[x863] = x870;
                float x872 = x806[x866];
                float x873 = x744[x863];
                float x874 = x804[x866];
                float x875 = x823[x868];
                float x876 = x872 + x875;
                x806[x866] = x876;
                x862 += 1;
                float x879 = x796[x801];
                float x880 = x806[0];
                float x881 = -1.0f * x880;
                float x882 = x879 + x881;
                x796[x801] = x882;
                float x884 = 0.0f;
                for(int x885=0; x885 < 5; x885++) {
                  float x886 = x884;
                  float x887 = x796[x885];
                  float x888 = x886 + x887;
                  x884 = x888;

                }
                float x892 = x884;
                float* x893 = (float*)myMalloc(1 * sizeof(float));;
                x893[0] = x892;
                float x895 = x893[0];
                for(int x896=0; x896 < 5; x896++) {
                  float x897 = x739[x896];
                  float x898 = x796[x896];
                  float x899 = x789[x896];
                  double x900 = (double)x899;
                  double x901 = exp(x900);
                  float x902 = (float)x901;
                  float x903 = x902 * x895;
                  float x904 = x898 - x903;
                  float x905 = x897 + x904;
                  x739[x896] = x905;

                }
                int32_t x909 = 0;
                int32_t x910 = 0;
                int32_t x911 = 0;
                int32_t x912 = x909;
                float x913 = x239[x912];
                float x914 = x238[x912];
                int32_t x915 = x910;
                float x916 = x245[x915];
                int32_t x917 = x911;
                float x918 = x756[x917];
                float x919 = x913 + x918;
                x239[x912] = x919;
                float x921 = x246[x915];
                float x922 = x238[x912];
                float x923 = x245[x915];
                float x924 = x756[x917];
                float x925 = x921 + x924;
                x246[x915] = x925;
                x911 += 1;
                int32_t x928 = 0;
                int32_t x929 = 0;
                int32_t x930 = 0;
                for(int x931=0; x931 < 5; x931++) {
                  int32_t x932 = x928;
                  float x933 = x717[x932];
                  float x934 = x701[x932];
                  int32_t x935 = x929;
                  float x936 = x96[x935];
                  int32_t x937 = x930;
                  float x938 = x739[x937];
                  float x939 = x933 + x938;
                  x717[x932] = x939;
                  float x941 = x132[x935];
                  float x942 = x701[x932];
                  float x943 = x96[x935];
                  float x944 = x739[x937];
                  float x945 = x941 + x944;
                  x132[x935] = x945;
                  x930 += 1;
                  x928 += 1;
                  x929 += 1;

                }
                // add_cartesian
                int32_t x953 = 0;
                for(int x954=0; x954 < 5; x954++) {
                  for(int x955=0; x955 < 150; x955++) {
                    int32_t x956 = x953;
                    int32_t x957 = x956 + x955;
                    float x958 = x127[x957];
                    float x959 = x686[x955];
                    float x960 = x717[x954];
                    float x961 = x959 * x960;
                    float x962 = x958 + x961;
                    x127[x957] = x962;

                  }
                  x953 += 150;

                }
                int32_t x969 = 0;
                for(int x970=0; x970 < 5; x970++) {
                  for(int x971=0; x971 < 150; x971++) {
                    float x972 = x695[x971];
                    int32_t x973 = x969;
                    int32_t x974 = x973 + x971;
                    float x975 = x87[x974];
                    float x976 = x717[x970];
                    float x977 = x975 * x976;
                    float x978 = x972 + x977;
                    x695[x971] = x978;

                  }
                  x969 += 150;

                }
                for(int x985=0; x985 < 150; x985++) {
                  float x986 = x681[x985];
                  float x987 = x686[x985];
                  float x990 = x695[x985];
                  float x988 = x987 * x987;
                  float x989 = 1.0f - x988;
                  float x991 = x989 * x990;
                  float x992 = x986 + x991;
                  x681[x985] = x992;

                }
                int32_t x996 = 0;
                int32_t x997 = 0;
                int32_t x998 = 0;
                for(int x999=0; x999 < 150; x999++) {
                  int32_t x1000 = x996;
                  float x1001 = x659[x1000];
                  float x1002 = x642[x1000];
                  int32_t x1003 = x997;
                  float x1004 = x82[x1003];
                  int32_t x1005 = x998;
                  float x1006 = x681[x1005];
                  float x1007 = x1001 + x1006;
                  x659[x1000] = x1007;
                  float x1009 = x122[x1003];
                  float x1010 = x642[x1000];
                  float x1011 = x82[x1003];
                  float x1012 = x681[x1005];
                  float x1013 = x1009 + x1012;
                  x122[x1003] = x1013;
                  x998 += 1;
                  x996 += 1;
                  x997 += 1;

                }
                int32_t x1020 = 0;
                int32_t x1021 = 0;
                int32_t x1022 = 0;
                for(int x1023=0; x1023 < 150; x1023++) {
                  int32_t x1024 = x1020;
                  float x1025 = x615[x1024];
                  float x1026 = x599[x1024];
                  int32_t x1027 = x1021;
                  float x1028 = x621[x1027];
                  int32_t x1029 = x1022;
                  float x1030 = x659[x1029];
                  float x1031 = x1025 + x1030;
                  x615[x1024] = x1031;
                  float x1033 = x637[x1027];
                  float x1034 = x599[x1024];
                  float x1035 = x621[x1027];
                  float x1036 = x659[x1029];
                  float x1037 = x1033 + x1036;
                  x637[x1027] = x1037;
                  x1022 += 1;
                  x1020 += 1;
                  x1021 += 1;

                }
                // add_cartesian
                int32_t x1045 = 0;
                for(int x1046=0; x1046 < 150; x1046++) {
                  for(int x1047=0; x1047 < 150; x1047++) {
                    int32_t x1048 = x1045;
                    int32_t x1049 = x1048 + x1047;
                    float x1050 = x117[x1049];
                    float x1051 = x247[x1047];
                    float x1052 = x637[x1046];
                    float x1053 = x1051 * x1052;
                    float x1054 = x1050 + x1053;
                    x117[x1049] = x1054;

                  }
                  x1045 += 150;

                }
                int32_t x1061 = 0;
                for(int x1062=0; x1062 < 150; x1062++) {
                  for(int x1063=0; x1063 < 150; x1063++) {
                    float x1064 = x248[x1063];
                    int32_t x1065 = x1061;
                    int32_t x1066 = x1065 + x1063;
                    float x1067 = x74[x1066];
                    float x1068 = x637[x1062];
                    float x1069 = x1067 * x1068;
                    float x1070 = x1064 + x1069;
                    x248[x1063] = x1070;

                  }
                  x1061 += 150;

                }
                // add_cartesian
                int32_t x1078 = 0;
                for(int x1079=0; x1079 < 150; x1079++) {
                  for(int x1080=0; x1080 < 150; x1080++) {
                    int32_t x1081 = x1078;
                    int32_t x1082 = x1081 + x1080;
                    float x1083 = x112[x1082];
                    float x1084 = x240[x1080];
                    float x1085 = x615[x1079];
                    float x1086 = x1084 * x1085;
                    float x1087 = x1083 + x1086;
                    x112[x1082] = x1087;

                  }
                  x1078 += 150;

                }
                int32_t x1094 = 0;
                for(int x1095=0; x1095 < 150; x1095++) {
                  for(int x1096=0; x1096 < 150; x1096++) {
                    float x1097 = x241[x1096];
                    int32_t x1098 = x1094;
                    int32_t x1099 = x1098 + x1096;
                    float x1100 = x65[x1099];
                    float x1101 = x615[x1095];
                    float x1102 = x1100 * x1101;
                    float x1103 = x1097 + x1102;
                    x241[x1096] = x1103;

                  }
                  x1094 += 150;

                }
              }
            };
            x223(x242,x243,x1113);
          };
          x223(x235,x236,x1121);
        } else {
          float** x1142 = (float**)myMalloc(4 * sizeof(float*));;
          x1142[0] = x203;
          x1142[1] = x208;
          x1142[2] = x213;
          x1142[3] = x218;
          function<void(float**)> x228 = x225;
          function<void(float**)> x1129 = [&](float** x1130) {
            float* x1131 = x1130[0];
            float* x1132 = x1130[1];
            float* x1133 = x1130[2];
            float* x1134 = x1130[3];
            float** x1135 = (float**)myMalloc(4 * sizeof(float*));;
            x1135[0] = x1131;
            x1135[1] = x1132;
            x1135[2] = x1133;
            x1135[3] = x1134;
            x228(x1135);
          };
          x1129(x1142);
        }
      };
      float* x187 = (float*)myMalloc(1 * sizeof(float));;
      for(int x189=0; x189 < 1; x189++) {
        x187[x189] = 0.0f;

      }
      float* x193 = (float*)myMalloc(1 * sizeof(float));;
      for(int x194=0; x194 < 1; x194++) {
        x193[x194] = 0.0f;

      }
      float* x198 = (float*)myMalloc(1 * sizeof(float));;
      for(int x199=0; x199 < 1; x199++) {
        x198[x199] = 0.0f;

      }
      for(int x204=0; x204 < 1; x204++) {
        x203[x204] = 0.0f;

      }
      for(int x209=0; x209 < 1; x209++) {
        x208[x209] = 0.0f;

      }
      for(int x214=0; x214 < 150; x214++) {
        x213[x214] = 0.0f;

      }
      for(int x219=0; x219 < 150; x219++) {
        x218[x219] = 0.0f;

      }
      float** x1162 = (float**)myMalloc(4 * sizeof(float*));;
      x1162[0] = x203;
      x1162[1] = x208;
      x1162[2] = x213;
      x1162[3] = x218;
      function<void(float**)> x1151 = [&](float** x1152) {
        float* x1153 = x1152[0];
        float* x1154 = x1152[1];
        float* x1155 = x1152[2];
        float* x1156 = x1152[3];
        float x1157 = x1154[0];
        x1154[0] = 1.0f;
        float x1159 = x1153[0];
        x198[0] = x1159;
      };
      x223(0,x1151,x1162);
      float x1169 = x198[0];
      float x1170 = x177;
      float x1171 = (float)x178;
      float x1172 = x1170 * x1171;
      int32_t x1173 = x178 + 1;
      float x1174 = (float)x1173;
      float x1175 = x1172 / x1174;
      float x1176 = x1169 / x1174;
      float x1177 = x1175 + x1176;
      x177 = x1177;
      for(int x1179=0; x1179 < 45000; x1179++) {
        float x1180 = x102[x1179];
        bool x1181 = x1180 > 5.0f;
        if (x1181) {
          x102[x1179] = 5.0f;
        } else {
        }
        float x1185 = x102[x1179];
        bool x1186 = x1185 < -5.0f;
        if (x1186) {
          x102[x1179] = -5.0f;
        } else {
        }

      }
      float* x1192 = (float*)myMalloc(45000 * sizeof(float));;
      int32_t x1193 = 0;
      int32_t x1194 = 0;
      int32_t x1195 = 0;
      for(int x1196=0; x1196 < 150; x1196++) {
        int32_t x1197 = x1194;
        int32_t x1198 = x1195;
        int32_t x1199 = x1193;
        int32_t x1200 = x1199;
        int32_t x1201 = x1197;
        int32_t x1202 = x1198;
        for(int x1203=0; x1203 < 300; x1203++) {
          int32_t x1204 = x1200;
          int32_t x1205 = x1201;
          float x1206 = x102[x1205];
          int32_t x1207 = x1202;
          float x1208 = x102[x1207];
          float x1209 = x1206 * x1208;
          x1192[x1204] = x1209;
          x1200 += 1;
          x1201 += 1;
          x1202 += 1;

        }
        x1193 += 300;
        x1194 += 300;
        x1195 += 300;

      }
      for(int x1221=0; x1221 < 45000; x1221++) {
        float x1222 = x137[x1221];
        float x1223 = x1192[x1221];
        float x1224 = x1222 + x1223;
        x137[x1221] = x1224;

      }
      float* x1228 = (float*)myMalloc(45000 * sizeof(float));;
      for(int x1229=0; x1229 < 45000; x1229++) {
        float x1230 = x102[x1229];
        float x1231 = x1230 * 0.05f;
        x1228[x1229] = x1231;

      }
      float* x1235 = (float*)myMalloc(45000 * sizeof(float));;
      for(int x1236=0; x1236 < 45000; x1236++) {
        float x1237 = x137[x1236];
        float x1238 = x1237 + 1.0E-8f;
        x1235[x1236] = x1238;

      }
      float* x1242 = (float*)myMalloc(45000 * sizeof(float));;
      for(int x1243=0; x1243 < 45000; x1243++) {
        float x1244 = x1235[x1243];
        double x1245 = (double)x1244;
        double x1246 = sqrt(x1245);
        float x1247 = (float)x1246;
        x1242[x1243] = x1247;

      }
      float* x1251 = (float*)myMalloc(45000 * sizeof(float));;
      int32_t x1252 = 0;
      int32_t x1253 = 0;
      int32_t x1254 = 0;
      for(int x1255=0; x1255 < 150; x1255++) {
        int32_t x1256 = x1253;
        int32_t x1257 = x1254;
        int32_t x1258 = x1252;
        int32_t x1259 = x1258;
        int32_t x1260 = x1256;
        int32_t x1261 = x1257;
        for(int x1262=0; x1262 < 300; x1262++) {
          int32_t x1263 = x1259;
          int32_t x1264 = x1260;
          float x1265 = x1228[x1264];
          int32_t x1266 = x1261;
          float x1267 = x1242[x1266];
          float x1268 = x1265 / x1267;
          x1251[x1263] = x1268;
          x1259 += 1;
          x1260 += 1;
          x1261 += 1;

        }
        x1252 += 300;
        x1253 += 300;
        x1254 += 300;

      }
      for(int x1280=0; x1280 < 45000; x1280++) {
        float x1281 = x50[x1280];
        float x1282 = x1251[x1280];
        float x1283 = x1281 - x1282;
        x50[x1280] = x1283;

      }
      for(int x1287=0; x1287 < 45000; x1287++) {
        float x1288 = x102[x1287];
        x102[x1287] = 0.0f;

      }
      for(int x1292=0; x1292 < 150; x1292++) {
        float x1293 = x107[x1292];
        bool x1294 = x1293 > 5.0f;
        if (x1294) {
          x107[x1292] = 5.0f;
        } else {
        }
        float x1298 = x107[x1292];
        bool x1299 = x1298 < -5.0f;
        if (x1299) {
          x107[x1292] = -5.0f;
        } else {
        }

      }
      float* x1305 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x1306 = 0;
      int32_t x1307 = 0;
      int32_t x1308 = 0;
      for(int x1309=0; x1309 < 150; x1309++) {
        int32_t x1310 = x1306;
        int32_t x1311 = x1307;
        float x1312 = x107[x1311];
        int32_t x1313 = x1308;
        float x1314 = x107[x1313];
        float x1315 = x1312 * x1314;
        x1305[x1310] = x1315;
        x1306 += 1;
        x1307 += 1;
        x1308 += 1;

      }
      for(int x1322=0; x1322 < 150; x1322++) {
        float x1323 = x142[x1322];
        float x1324 = x1305[x1322];
        float x1325 = x1323 + x1324;
        x142[x1322] = x1325;

      }
      float* x1329 = (float*)myMalloc(150 * sizeof(float));;
      for(int x1330=0; x1330 < 150; x1330++) {
        float x1331 = x107[x1330];
        float x1332 = x1331 * 0.05f;
        x1329[x1330] = x1332;

      }
      float* x1336 = (float*)myMalloc(150 * sizeof(float));;
      for(int x1337=0; x1337 < 150; x1337++) {
        float x1338 = x142[x1337];
        float x1339 = x1338 + 1.0E-8f;
        x1336[x1337] = x1339;

      }
      float* x1343 = (float*)myMalloc(150 * sizeof(float));;
      for(int x1344=0; x1344 < 150; x1344++) {
        float x1345 = x1336[x1344];
        double x1346 = (double)x1345;
        double x1347 = sqrt(x1346);
        float x1348 = (float)x1347;
        x1343[x1344] = x1348;

      }
      float* x1352 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x1353 = 0;
      int32_t x1354 = 0;
      int32_t x1355 = 0;
      for(int x1356=0; x1356 < 150; x1356++) {
        int32_t x1357 = x1353;
        int32_t x1358 = x1354;
        float x1359 = x1329[x1358];
        int32_t x1360 = x1355;
        float x1361 = x1343[x1360];
        float x1362 = x1359 / x1361;
        x1352[x1357] = x1362;
        x1353 += 1;
        x1354 += 1;
        x1355 += 1;

      }
      for(int x1369=0; x1369 < 150; x1369++) {
        float x1370 = x59[x1369];
        float x1371 = x1352[x1369];
        float x1372 = x1370 - x1371;
        x59[x1369] = x1372;

      }
      for(int x1376=0; x1376 < 150; x1376++) {
        float x1377 = x107[x1376];
        x107[x1376] = 0.0f;

      }
      for(int x1381=0; x1381 < 22500; x1381++) {
        float x1382 = x112[x1381];
        bool x1383 = x1382 > 5.0f;
        if (x1383) {
          x112[x1381] = 5.0f;
        } else {
        }
        float x1387 = x112[x1381];
        bool x1388 = x1387 < -5.0f;
        if (x1388) {
          x112[x1381] = -5.0f;
        } else {
        }

      }
      float* x1394 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x1395 = 0;
      int32_t x1396 = 0;
      int32_t x1397 = 0;
      for(int x1398=0; x1398 < 150; x1398++) {
        int32_t x1399 = x1396;
        int32_t x1400 = x1397;
        int32_t x1401 = x1395;
        int32_t x1402 = x1401;
        int32_t x1403 = x1399;
        int32_t x1404 = x1400;
        for(int x1405=0; x1405 < 150; x1405++) {
          int32_t x1406 = x1402;
          int32_t x1407 = x1403;
          float x1408 = x112[x1407];
          int32_t x1409 = x1404;
          float x1410 = x112[x1409];
          float x1411 = x1408 * x1410;
          x1394[x1406] = x1411;
          x1402 += 1;
          x1403 += 1;
          x1404 += 1;

        }
        x1395 += 150;
        x1396 += 150;
        x1397 += 150;

      }
      for(int x1423=0; x1423 < 22500; x1423++) {
        float x1424 = x147[x1423];
        float x1425 = x1394[x1423];
        float x1426 = x1424 + x1425;
        x147[x1423] = x1426;

      }
      float* x1430 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x1431=0; x1431 < 22500; x1431++) {
        float x1432 = x112[x1431];
        float x1433 = x1432 * 0.05f;
        x1430[x1431] = x1433;

      }
      float* x1437 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x1438=0; x1438 < 22500; x1438++) {
        float x1439 = x147[x1438];
        float x1440 = x1439 + 1.0E-8f;
        x1437[x1438] = x1440;

      }
      float* x1444 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x1445=0; x1445 < 22500; x1445++) {
        float x1446 = x1437[x1445];
        double x1447 = (double)x1446;
        double x1448 = sqrt(x1447);
        float x1449 = (float)x1448;
        x1444[x1445] = x1449;

      }
      float* x1453 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x1454 = 0;
      int32_t x1455 = 0;
      int32_t x1456 = 0;
      for(int x1457=0; x1457 < 150; x1457++) {
        int32_t x1458 = x1455;
        int32_t x1459 = x1456;
        int32_t x1460 = x1454;
        int32_t x1461 = x1460;
        int32_t x1462 = x1458;
        int32_t x1463 = x1459;
        for(int x1464=0; x1464 < 150; x1464++) {
          int32_t x1465 = x1461;
          int32_t x1466 = x1462;
          float x1467 = x1430[x1466];
          int32_t x1468 = x1463;
          float x1469 = x1444[x1468];
          float x1470 = x1467 / x1469;
          x1453[x1465] = x1470;
          x1461 += 1;
          x1462 += 1;
          x1463 += 1;

        }
        x1454 += 150;
        x1455 += 150;
        x1456 += 150;

      }
      for(int x1482=0; x1482 < 22500; x1482++) {
        float x1483 = x65[x1482];
        float x1484 = x1453[x1482];
        float x1485 = x1483 - x1484;
        x65[x1482] = x1485;

      }
      for(int x1489=0; x1489 < 22500; x1489++) {
        float x1490 = x112[x1489];
        x112[x1489] = 0.0f;

      }
      for(int x1494=0; x1494 < 22500; x1494++) {
        float x1495 = x117[x1494];
        bool x1496 = x1495 > 5.0f;
        if (x1496) {
          x117[x1494] = 5.0f;
        } else {
        }
        float x1500 = x117[x1494];
        bool x1501 = x1500 < -5.0f;
        if (x1501) {
          x117[x1494] = -5.0f;
        } else {
        }

      }
      float* x1507 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x1508 = 0;
      int32_t x1509 = 0;
      int32_t x1510 = 0;
      for(int x1511=0; x1511 < 150; x1511++) {
        int32_t x1512 = x1509;
        int32_t x1513 = x1510;
        int32_t x1514 = x1508;
        int32_t x1515 = x1514;
        int32_t x1516 = x1512;
        int32_t x1517 = x1513;
        for(int x1518=0; x1518 < 150; x1518++) {
          int32_t x1519 = x1515;
          int32_t x1520 = x1516;
          float x1521 = x117[x1520];
          int32_t x1522 = x1517;
          float x1523 = x117[x1522];
          float x1524 = x1521 * x1523;
          x1507[x1519] = x1524;
          x1515 += 1;
          x1516 += 1;
          x1517 += 1;

        }
        x1508 += 150;
        x1509 += 150;
        x1510 += 150;

      }
      for(int x1536=0; x1536 < 22500; x1536++) {
        float x1537 = x152[x1536];
        float x1538 = x1507[x1536];
        float x1539 = x1537 + x1538;
        x152[x1536] = x1539;

      }
      float* x1543 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x1544=0; x1544 < 22500; x1544++) {
        float x1545 = x117[x1544];
        float x1546 = x1545 * 0.05f;
        x1543[x1544] = x1546;

      }
      float* x1550 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x1551=0; x1551 < 22500; x1551++) {
        float x1552 = x152[x1551];
        float x1553 = x1552 + 1.0E-8f;
        x1550[x1551] = x1553;

      }
      float* x1557 = (float*)myMalloc(22500 * sizeof(float));;
      for(int x1558=0; x1558 < 22500; x1558++) {
        float x1559 = x1550[x1558];
        double x1560 = (double)x1559;
        double x1561 = sqrt(x1560);
        float x1562 = (float)x1561;
        x1557[x1558] = x1562;

      }
      float* x1566 = (float*)myMalloc(22500 * sizeof(float));;
      int32_t x1567 = 0;
      int32_t x1568 = 0;
      int32_t x1569 = 0;
      for(int x1570=0; x1570 < 150; x1570++) {
        int32_t x1571 = x1568;
        int32_t x1572 = x1569;
        int32_t x1573 = x1567;
        int32_t x1574 = x1573;
        int32_t x1575 = x1571;
        int32_t x1576 = x1572;
        for(int x1577=0; x1577 < 150; x1577++) {
          int32_t x1578 = x1574;
          int32_t x1579 = x1575;
          float x1580 = x1543[x1579];
          int32_t x1581 = x1576;
          float x1582 = x1557[x1581];
          float x1583 = x1580 / x1582;
          x1566[x1578] = x1583;
          x1574 += 1;
          x1575 += 1;
          x1576 += 1;

        }
        x1567 += 150;
        x1568 += 150;
        x1569 += 150;

      }
      for(int x1595=0; x1595 < 22500; x1595++) {
        float x1596 = x74[x1595];
        float x1597 = x1566[x1595];
        float x1598 = x1596 - x1597;
        x74[x1595] = x1598;

      }
      for(int x1602=0; x1602 < 22500; x1602++) {
        float x1603 = x117[x1602];
        x117[x1602] = 0.0f;

      }
      for(int x1607=0; x1607 < 150; x1607++) {
        float x1608 = x122[x1607];
        bool x1609 = x1608 > 5.0f;
        if (x1609) {
          x122[x1607] = 5.0f;
        } else {
        }
        float x1613 = x122[x1607];
        bool x1614 = x1613 < -5.0f;
        if (x1614) {
          x122[x1607] = -5.0f;
        } else {
        }

      }
      float* x1620 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x1621 = 0;
      int32_t x1622 = 0;
      int32_t x1623 = 0;
      for(int x1624=0; x1624 < 150; x1624++) {
        int32_t x1625 = x1621;
        int32_t x1626 = x1622;
        float x1627 = x122[x1626];
        int32_t x1628 = x1623;
        float x1629 = x122[x1628];
        float x1630 = x1627 * x1629;
        x1620[x1625] = x1630;
        x1621 += 1;
        x1622 += 1;
        x1623 += 1;

      }
      for(int x1637=0; x1637 < 150; x1637++) {
        float x1638 = x157[x1637];
        float x1639 = x1620[x1637];
        float x1640 = x1638 + x1639;
        x157[x1637] = x1640;

      }
      float* x1644 = (float*)myMalloc(150 * sizeof(float));;
      for(int x1645=0; x1645 < 150; x1645++) {
        float x1646 = x122[x1645];
        float x1647 = x1646 * 0.05f;
        x1644[x1645] = x1647;

      }
      float* x1651 = (float*)myMalloc(150 * sizeof(float));;
      for(int x1652=0; x1652 < 150; x1652++) {
        float x1653 = x157[x1652];
        float x1654 = x1653 + 1.0E-8f;
        x1651[x1652] = x1654;

      }
      float* x1658 = (float*)myMalloc(150 * sizeof(float));;
      for(int x1659=0; x1659 < 150; x1659++) {
        float x1660 = x1651[x1659];
        double x1661 = (double)x1660;
        double x1662 = sqrt(x1661);
        float x1663 = (float)x1662;
        x1658[x1659] = x1663;

      }
      float* x1667 = (float*)myMalloc(150 * sizeof(float));;
      int32_t x1668 = 0;
      int32_t x1669 = 0;
      int32_t x1670 = 0;
      for(int x1671=0; x1671 < 150; x1671++) {
        int32_t x1672 = x1668;
        int32_t x1673 = x1669;
        float x1674 = x1644[x1673];
        int32_t x1675 = x1670;
        float x1676 = x1658[x1675];
        float x1677 = x1674 / x1676;
        x1667[x1672] = x1677;
        x1668 += 1;
        x1669 += 1;
        x1670 += 1;

      }
      for(int x1684=0; x1684 < 150; x1684++) {
        float x1685 = x82[x1684];
        float x1686 = x1667[x1684];
        float x1687 = x1685 - x1686;
        x82[x1684] = x1687;

      }
      for(int x1691=0; x1691 < 150; x1691++) {
        float x1692 = x122[x1691];
        x122[x1691] = 0.0f;

      }
      for(int x1696=0; x1696 < 750; x1696++) {
        float x1697 = x127[x1696];
        bool x1698 = x1697 > 5.0f;
        if (x1698) {
          x127[x1696] = 5.0f;
        } else {
        }
        float x1702 = x127[x1696];
        bool x1703 = x1702 < -5.0f;
        if (x1703) {
          x127[x1696] = -5.0f;
        } else {
        }

      }
      float* x1709 = (float*)myMalloc(750 * sizeof(float));;
      int32_t x1710 = 0;
      int32_t x1711 = 0;
      int32_t x1712 = 0;
      for(int x1713=0; x1713 < 5; x1713++) {
        int32_t x1714 = x1711;
        int32_t x1715 = x1712;
        int32_t x1716 = x1710;
        int32_t x1717 = x1716;
        int32_t x1718 = x1714;
        int32_t x1719 = x1715;
        for(int x1720=0; x1720 < 150; x1720++) {
          int32_t x1721 = x1717;
          int32_t x1722 = x1718;
          float x1723 = x127[x1722];
          int32_t x1724 = x1719;
          float x1725 = x127[x1724];
          float x1726 = x1723 * x1725;
          x1709[x1721] = x1726;
          x1717 += 1;
          x1718 += 1;
          x1719 += 1;

        }
        x1710 += 150;
        x1711 += 150;
        x1712 += 150;

      }
      for(int x1738=0; x1738 < 750; x1738++) {
        float x1739 = x162[x1738];
        float x1740 = x1709[x1738];
        float x1741 = x1739 + x1740;
        x162[x1738] = x1741;

      }
      float* x1745 = (float*)myMalloc(750 * sizeof(float));;
      for(int x1746=0; x1746 < 750; x1746++) {
        float x1747 = x127[x1746];
        float x1748 = x1747 * 0.05f;
        x1745[x1746] = x1748;

      }
      float* x1752 = (float*)myMalloc(750 * sizeof(float));;
      for(int x1753=0; x1753 < 750; x1753++) {
        float x1754 = x162[x1753];
        float x1755 = x1754 + 1.0E-8f;
        x1752[x1753] = x1755;

      }
      float* x1759 = (float*)myMalloc(750 * sizeof(float));;
      for(int x1760=0; x1760 < 750; x1760++) {
        float x1761 = x1752[x1760];
        double x1762 = (double)x1761;
        double x1763 = sqrt(x1762);
        float x1764 = (float)x1763;
        x1759[x1760] = x1764;

      }
      float* x1768 = (float*)myMalloc(750 * sizeof(float));;
      int32_t x1769 = 0;
      int32_t x1770 = 0;
      int32_t x1771 = 0;
      for(int x1772=0; x1772 < 5; x1772++) {
        int32_t x1773 = x1770;
        int32_t x1774 = x1771;
        int32_t x1775 = x1769;
        int32_t x1776 = x1775;
        int32_t x1777 = x1773;
        int32_t x1778 = x1774;
        for(int x1779=0; x1779 < 150; x1779++) {
          int32_t x1780 = x1776;
          int32_t x1781 = x1777;
          float x1782 = x1745[x1781];
          int32_t x1783 = x1778;
          float x1784 = x1759[x1783];
          float x1785 = x1782 / x1784;
          x1768[x1780] = x1785;
          x1776 += 1;
          x1777 += 1;
          x1778 += 1;

        }
        x1769 += 150;
        x1770 += 150;
        x1771 += 150;

      }
      for(int x1797=0; x1797 < 750; x1797++) {
        float x1798 = x87[x1797];
        float x1799 = x1768[x1797];
        float x1800 = x1798 - x1799;
        x87[x1797] = x1800;

      }
      for(int x1804=0; x1804 < 750; x1804++) {
        float x1805 = x127[x1804];
        x127[x1804] = 0.0f;

      }
      for(int x1809=0; x1809 < 5; x1809++) {
        float x1810 = x132[x1809];
        bool x1811 = x1810 > 5.0f;
        if (x1811) {
          x132[x1809] = 5.0f;
        } else {
        }
        float x1815 = x132[x1809];
        bool x1816 = x1815 < -5.0f;
        if (x1816) {
          x132[x1809] = -5.0f;
        } else {
        }

      }
      float* x1822 = (float*)myMalloc(5 * sizeof(float));;
      int32_t x1823 = 0;
      int32_t x1824 = 0;
      int32_t x1825 = 0;
      for(int x1826=0; x1826 < 5; x1826++) {
        int32_t x1827 = x1823;
        int32_t x1828 = x1824;
        float x1829 = x132[x1828];
        int32_t x1830 = x1825;
        float x1831 = x132[x1830];
        float x1832 = x1829 * x1831;
        x1822[x1827] = x1832;
        x1823 += 1;
        x1824 += 1;
        x1825 += 1;

      }
      for(int x1839=0; x1839 < 5; x1839++) {
        float x1840 = x167[x1839];
        float x1841 = x1822[x1839];
        float x1842 = x1840 + x1841;
        x167[x1839] = x1842;

      }
      float* x1846 = (float*)myMalloc(5 * sizeof(float));;
      for(int x1847=0; x1847 < 5; x1847++) {
        float x1848 = x132[x1847];
        float x1849 = x1848 * 0.05f;
        x1846[x1847] = x1849;

      }
      float* x1853 = (float*)myMalloc(5 * sizeof(float));;
      for(int x1854=0; x1854 < 5; x1854++) {
        float x1855 = x167[x1854];
        float x1856 = x1855 + 1.0E-8f;
        x1853[x1854] = x1856;

      }
      float* x1860 = (float*)myMalloc(5 * sizeof(float));;
      for(int x1861=0; x1861 < 5; x1861++) {
        float x1862 = x1853[x1861];
        double x1863 = (double)x1862;
        double x1864 = sqrt(x1863);
        float x1865 = (float)x1864;
        x1860[x1861] = x1865;

      }
      float* x1869 = (float*)myMalloc(5 * sizeof(float));;
      int32_t x1870 = 0;
      int32_t x1871 = 0;
      int32_t x1872 = 0;
      for(int x1873=0; x1873 < 5; x1873++) {
        int32_t x1874 = x1870;
        int32_t x1875 = x1871;
        float x1876 = x1846[x1875];
        int32_t x1877 = x1872;
        float x1878 = x1860[x1877];
        float x1879 = x1876 / x1878;
        x1869[x1874] = x1879;
        x1870 += 1;
        x1871 += 1;
        x1872 += 1;

      }
      for(int x1886=0; x1886 < 5; x1886++) {
        float x1887 = x96[x1886];
        float x1888 = x1869[x1886];
        float x1889 = x1887 - x1888;
        x96[x1886] = x1889;

      }
      for(int x1893=0; x1893 < 5; x1893++) {
        float x1894 = x132[x1893];
        x132[x1893] = 0.0f;

      }
      mallocAddr = (void*)x173;

    }
    float x1901 = x177;
    double x1902 = (double)x1901;
    x172[x176] = x1902;
    double x1904 = ((double)clock() / CLOCKS_PER_SEC);
    double x1905 = x1904 - x174;
    printf("epoc %d, average_loss %f, time %lf\n",x176,x1901,x1905);

  }
  double x1909 = ((double)clock() / CLOCKS_PER_SEC);
  int64_t x1913 = (long)fopen(x0, "w");
  fprintf((FILE *)x1913, "unit: %s\n", "1 epoch");
  for(int x1915=0; x1915 < 6; x1915++) {
    double x1916 = x172[x1915];
    fprintf((FILE *)x1913, "%lf\n", x1916);

  }
  double x1910 = x174 - x2;
  double x1911 = x1909 - x174;
  double x1912 = x1911 / 6.0;
  fprintf((FILE *)x1913, "run time: %lf %lf\n", x1910, x1912);
  fclose((FILE*)x1913);
  // Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
 *******************************************/

