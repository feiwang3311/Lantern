
      #include <fcntl.h>
      #include <errno.h>
      #include <err.h>
      #include <sys/mman.h>
      #include <sys/stat.h>
      #include <sys/time.h>
      #include <stdio.h>
      #include <stdint.h>
      #include <unistd.h>
      #include <time.h>
      #include <functional>
      #include <memory>
      #include <math.h>
      #include <random>

      using namespace std;
      #ifndef MAP_FILE
      #define MAP_FILE MAP_SHARED
      #endif
      int fsize(int fd) {
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
      long hash(char *str0, int len)
      {
        unsigned char* str = (unsigned char*)str0;
        unsigned long hash = 5381;
        int c;

        while ((c = *str++) && len--)
          hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

        return hash;
      }
      int HEAP_SIZE = 1073741826; // 1048576;  //2147483652; //536870912; // 268435456; //2097152;
      void *mallocBase = malloc(HEAP_SIZE);
      void *mallocAddr = mallocBase;
      void *waterMark  = mallocBase;
      void* myMalloc(size_t bytes) {
        void* res = mallocAddr;
        mallocAddr = (void *)((char *)mallocAddr + bytes);
        return res;
      }

      int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
        long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
        result->tv_sec = diff / 1000000;
        result->tv_usec = diff % 1000000;
        return (diff<0);
      }



      void Snippet(char*);

      std::random_device rd{};
      std::mt19937 gen{rd()};
      std::normal_distribution<> d{0,1};

      int main(int argc, char *argv[])
      {

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
int64_t x1 = (long)fopen("senti/small_glove.txt", "r");
float** x2 = (float**)myMalloc(5265 * sizeof(float*));
for(int x4=0; x4 < 5265; x4++) {
float* x5 = (float*)myMalloc(300 * sizeof(float));
x2[x4] = x5;
for(int x8=0; x8 < 300; x8++) {
float* x9 = x2[x4];
if (fscanf((FILE *)x1,"%f", &x9[x8])!=1) perror("Error reading file");

}

}
fclose((FILE*)x1);
int64_t x16 = (long)fopen("senti/array_seq.txt", "r");
int** x17 = (int**)myMalloc(1101 * sizeof(int*));
int32_t* x18 = (int32_t*)myMalloc(1101 * sizeof(int32_t));
int32_t* x19 = (int32_t*)myMalloc(1 * sizeof(int32_t));
for(int x21=0; x21 < 1101; x21++) {
if (fscanf((FILE *)x16,"%d", &x19[0])!=1) perror("Error reading file");
int32_t x23 = x19[0];
int32_t* x24 = (int32_t*)myMalloc(x23 * sizeof(int32_t));
x17[x21] = x24;
if (fscanf((FILE *)x16,"%d", &x18[x21])!=1) perror("Error reading file");
int32_t x27 = x19[0];
for(int x29=0; x29 < x27; x29++) {
int* x30 = x17[x21];
if (fscanf((FILE *)x16,"%d", &x30[x29])!=1) perror("Error reading file");

}

}
float* x36 = (float*)myMalloc(22500 * sizeof(float));
for(int x38=0; x38 < 22500; x38++) {
float x39 = d(gen);
float x40 = x39 * 0.01f;
x36[x38] = x40;

}
float* x44 = (float*)myMalloc(45000 * sizeof(float));
for(int x46=0; x46 < 45000; x46++) {
float x47 = d(gen);
float x48 = x47 * 0.01f;
x44[x46] = x48;

}
float* x52 = (float*)myMalloc(150 * sizeof(float));
for(int x54=0; x54 < 150; x54++) {
x52[x54] = 0.0f;

}
float* x58 = (float*)myMalloc(22500 * sizeof(float));
for(int x59=0; x59 < 22500; x59++) {
float x60 = d(gen);
float x61 = x60 * 0.01f;
x58[x59] = x61;

}
float* x65 = (float*)myMalloc(45000 * sizeof(float));
for(int x66=0; x66 < 45000; x66++) {
float x67 = d(gen);
float x68 = x67 * 0.01f;
x65[x66] = x68;

}
float* x72 = (float*)myMalloc(150 * sizeof(float));
for(int x73=0; x73 < 150; x73++) {
x72[x73] = 0.0f;

}
float* x77 = (float*)myMalloc(22500 * sizeof(float));
for(int x78=0; x78 < 22500; x78++) {
float x79 = d(gen);
float x80 = x79 * 0.01f;
x77[x78] = x80;

}
float* x84 = (float*)myMalloc(45000 * sizeof(float));
for(int x85=0; x85 < 45000; x85++) {
float x86 = d(gen);
float x87 = x86 * 0.01f;
x84[x85] = x87;

}
float* x91 = (float*)myMalloc(150 * sizeof(float));
for(int x92=0; x92 < 150; x92++) {
x91[x92] = 0.0f;

}
float* x96 = (float*)myMalloc(22500 * sizeof(float));
for(int x97=0; x97 < 22500; x97++) {
float x98 = d(gen);
float x99 = x98 * 0.01f;
x96[x97] = x99;

}
float* x103 = (float*)myMalloc(45000 * sizeof(float));
for(int x104=0; x104 < 45000; x104++) {
float x105 = d(gen);
float x106 = x105 * 0.01f;
x103[x104] = x106;

}
float* x110 = (float*)myMalloc(150 * sizeof(float));
for(int x111=0; x111 < 150; x111++) {
x110[x111] = 0.0f;

}
float* x115 = (float*)myMalloc(750 * sizeof(float));
for(int x117=0; x117 < 750; x117++) {
float x118 = d(gen);
float x119 = x118 * 0.01f;
x115[x117] = x119;

}
float* x123 = (float*)myMalloc(5 * sizeof(float));
for(int x125=0; x125 < 5; x125++) {
x123[x125] = 0.0f;

}
float* x129 = (float*)myMalloc(150 * sizeof(float));
for(int x130=0; x130 < 150; x130++) {
x129[x130] = 0.0f;

}
float* x134 = (float*)myMalloc(150 * sizeof(float));
for(int x135=0; x135 < 150; x135++) {
x134[x135] = 0.0f;

}
float* x139 = (float*)myMalloc(22500 * sizeof(float));
for(int x140=0; x140 < 22500; x140++) {
x139[x140] = 0.0f;

}
float* x144 = (float*)myMalloc(45000 * sizeof(float));
for(int x145=0; x145 < 45000; x145++) {
x144[x145] = 0.0f;

}
float* x149 = (float*)myMalloc(150 * sizeof(float));
for(int x150=0; x150 < 150; x150++) {
x149[x150] = 0.0f;

}
float* x154 = (float*)myMalloc(22500 * sizeof(float));
for(int x155=0; x155 < 22500; x155++) {
x154[x155] = 0.0f;

}
float* x159 = (float*)myMalloc(45000 * sizeof(float));
for(int x160=0; x160 < 45000; x160++) {
x159[x160] = 0.0f;

}
float* x164 = (float*)myMalloc(150 * sizeof(float));
for(int x165=0; x165 < 150; x165++) {
x164[x165] = 0.0f;

}
float* x169 = (float*)myMalloc(22500 * sizeof(float));
for(int x170=0; x170 < 22500; x170++) {
x169[x170] = 0.0f;

}
float* x174 = (float*)myMalloc(45000 * sizeof(float));
for(int x175=0; x175 < 45000; x175++) {
x174[x175] = 0.0f;

}
float* x179 = (float*)myMalloc(150 * sizeof(float));
for(int x180=0; x180 < 150; x180++) {
x179[x180] = 0.0f;

}
float* x184 = (float*)myMalloc(22500 * sizeof(float));
for(int x185=0; x185 < 22500; x185++) {
x184[x185] = 0.0f;

}
float* x189 = (float*)myMalloc(45000 * sizeof(float));
for(int x190=0; x190 < 45000; x190++) {
x189[x190] = 0.0f;

}
float* x194 = (float*)myMalloc(150 * sizeof(float));
for(int x195=0; x195 < 150; x195++) {
x194[x195] = 0.0f;

}
float* x199 = (float*)myMalloc(750 * sizeof(float));
for(int x200=0; x200 < 750; x200++) {
x199[x200] = 0.0f;

}
float* x204 = (float*)myMalloc(5 * sizeof(float));
for(int x205=0; x205 < 5; x205++) {
x204[x205] = 0.0f;

}
float* x209 = (float*)myMalloc(150 * sizeof(float));
for(int x210=0; x210 < 150; x210++) {
x209[x210] = 0.0f;

}
float* x214 = (float*)myMalloc(150 * sizeof(float));
for(int x215=0; x215 < 150; x215++) {
x214[x215] = 0.0f;

}
float* x219 = (float*)myMalloc(22500 * sizeof(float));
for(int x220=0; x220 < 22500; x220++) {
x219[x220] = 0.0f;

}
float* x224 = (float*)myMalloc(45000 * sizeof(float));
for(int x225=0; x225 < 45000; x225++) {
x224[x225] = 0.0f;

}
float* x229 = (float*)myMalloc(150 * sizeof(float));
for(int x230=0; x230 < 150; x230++) {
x229[x230] = 0.0f;

}
float* x234 = (float*)myMalloc(22500 * sizeof(float));
for(int x235=0; x235 < 22500; x235++) {
x234[x235] = 0.0f;

}
float* x239 = (float*)myMalloc(45000 * sizeof(float));
for(int x240=0; x240 < 45000; x240++) {
x239[x240] = 0.0f;

}
float* x244 = (float*)myMalloc(150 * sizeof(float));
for(int x245=0; x245 < 150; x245++) {
x244[x245] = 0.0f;

}
float* x249 = (float*)myMalloc(22500 * sizeof(float));
for(int x250=0; x250 < 22500; x250++) {
x249[x250] = 0.0f;

}
float* x254 = (float*)myMalloc(45000 * sizeof(float));
for(int x255=0; x255 < 45000; x255++) {
x254[x255] = 0.0f;

}
float* x259 = (float*)myMalloc(150 * sizeof(float));
for(int x260=0; x260 < 150; x260++) {
x259[x260] = 0.0f;

}
float* x264 = (float*)myMalloc(22500 * sizeof(float));
for(int x265=0; x265 < 22500; x265++) {
x264[x265] = 0.0f;

}
float* x269 = (float*)myMalloc(45000 * sizeof(float));
for(int x270=0; x270 < 45000; x270++) {
x269[x270] = 0.0f;

}
float* x274 = (float*)myMalloc(150 * sizeof(float));
for(int x275=0; x275 < 150; x275++) {
x274[x275] = 0.0f;

}
float* x279 = (float*)myMalloc(750 * sizeof(float));
for(int x280=0; x280 < 750; x280++) {
x279[x280] = 0.0f;

}
float* x284 = (float*)myMalloc(5 * sizeof(float));
for(int x285=0; x285 < 5; x285++) {
x284[x285] = 0.0f;

}
int64_t x289 = (long)mallocAddr;
for(int x291=0; x291 < 2001; x291++) {
int32_t x292 = x291 % 1101;
int* x293 = x17[x292];
int32_t x311 = x293->length;
int32_t x294 = x18[x292];
float* x306 = (float*)myMalloc(1 * sizeof(float));
function<void(int32_t,float**)> x312 = [&](int32_t x313,float** x314) {
float** x316 = x314;
float* x317 = x316[0];
float* x318 = x316[1];
float* x319 = x316[2];
float* x320 = x316[3];
int32_t x315 = x313;
bool x321 = x315 < x311;
if (x321) {
int32_t x322 = x293[x315];
float* x323 = x2[x322];
float* x324 = (float*)myMalloc(300 * sizeof(float));
for(int x325=0; x325 < 300; x325++) {
x324[x325] = 0.0f;

}
// dot WrappedArray(150, 150) - WrappedArray(150)
int32_t x330 = 0;
float* x331 = (float*)myMalloc(150 * sizeof(float));
for(int x332=0; x332 < 150; x332++) {
float x333 = 0.0f;
for(int x334=0; x334 < 150; x334++) {
int32_t x335 = x330;
float x336 = x36[x335];
float x337 = x317[x334];
float x338 = x336 * x337;
x333 += x338;
x330 += 1;

}
float x343 = x333;
x331[x332] = x343;

}
float* x347 = (float*)myMalloc(150 * sizeof(float));
for(int x348=0; x348 < 150; x348++) {
x347[x348] = 0.0f;

}
// dot WrappedArray(150, 300) - WrappedArray(300)
int32_t x353 = 0;
float* x354 = (float*)myMalloc(150 * sizeof(float));
for(int x355=0; x355 < 150; x355++) {
float x356 = 0.0f;
for(int x357=0; x357 < 300; x357++) {
int32_t x358 = x353;
float x359 = x44[x358];
float x360 = x323[x357];
float x361 = x359 * x360;
x356 += x361;
x353 += 1;

}
float x366 = x356;
x354[x355] = x366;

}
float* x370 = (float*)myMalloc(150 * sizeof(float));
for(int x371=0; x371 < 150; x371++) {
x370[x371] = 0.0f;

}
float* x375 = (float*)myMalloc(150 * sizeof(float));
for(int x376=0; x376 < 150; x376++) {
float x377 = x331[x376];
float x378 = x354[x376];
float x379 = x377 + x378;
x375[x376] = x379;

}
float* x383 = (float*)myMalloc(150 * sizeof(float));
for(int x384=0; x384 < 150; x384++) {
x383[x384] = 0.0f;

}
float* x388 = (float*)myMalloc(150 * sizeof(float));
for(int x389=0; x389 < 150; x389++) {
float x390 = x375[x389];
float x391 = x52[x389];
float x392 = x390 + x391;
x388[x389] = x392;

}
float* x396 = (float*)myMalloc(150 * sizeof(float));
for(int x397=0; x397 < 150; x397++) {
x396[x397] = 0.0f;

}
float* x401 = (float*)myMalloc(150 * sizeof(float));
for(int x402=0; x402 < 150; x402++) {
float x403 = x388[x402];
float x404 = -1.0f * x403;
double x405 = (double)x404;
double x406 = exp(x405);
float x407 = (float)x406;
float x408 = x407 + 1.0f;
float x409 = 1.0f / x408;
x401[x402] = x409;

}
float* x413 = (float*)myMalloc(150 * sizeof(float));
for(int x414=0; x414 < 150; x414++) {
x413[x414] = 0.0f;

}
// dot WrappedArray(150, 150) - WrappedArray(150)
int32_t x419 = 0;
float* x420 = (float*)myMalloc(150 * sizeof(float));
for(int x421=0; x421 < 150; x421++) {
float x422 = 0.0f;
for(int x423=0; x423 < 150; x423++) {
int32_t x424 = x419;
float x425 = x58[x424];
float x426 = x317[x423];
float x427 = x425 * x426;
x422 += x427;
x419 += 1;

}
float x432 = x422;
x420[x421] = x432;

}
float* x436 = (float*)myMalloc(150 * sizeof(float));
for(int x437=0; x437 < 150; x437++) {
x436[x437] = 0.0f;

}
// dot WrappedArray(150, 300) - WrappedArray(300)
int32_t x442 = 0;
float* x443 = (float*)myMalloc(150 * sizeof(float));
for(int x444=0; x444 < 150; x444++) {
float x445 = 0.0f;
for(int x446=0; x446 < 300; x446++) {
int32_t x447 = x442;
float x448 = x65[x447];
float x449 = x323[x446];
float x450 = x448 * x449;
x445 += x450;
x442 += 1;

}
float x455 = x445;
x443[x444] = x455;

}
float* x459 = (float*)myMalloc(150 * sizeof(float));
for(int x460=0; x460 < 150; x460++) {
x459[x460] = 0.0f;

}
float* x464 = (float*)myMalloc(150 * sizeof(float));
for(int x465=0; x465 < 150; x465++) {
float x466 = x420[x465];
float x467 = x443[x465];
float x468 = x466 + x467;
x464[x465] = x468;

}
float* x472 = (float*)myMalloc(150 * sizeof(float));
for(int x473=0; x473 < 150; x473++) {
x472[x473] = 0.0f;

}
float* x477 = (float*)myMalloc(150 * sizeof(float));
for(int x478=0; x478 < 150; x478++) {
float x479 = x464[x478];
float x480 = x72[x478];
float x481 = x479 + x480;
x477[x478] = x481;

}
float* x485 = (float*)myMalloc(150 * sizeof(float));
for(int x486=0; x486 < 150; x486++) {
x485[x486] = 0.0f;

}
float* x490 = (float*)myMalloc(150 * sizeof(float));
for(int x491=0; x491 < 150; x491++) {
float x492 = x477[x491];
float x493 = -1.0f * x492;
double x494 = (double)x493;
double x495 = exp(x494);
float x496 = (float)x495;
float x497 = x496 + 1.0f;
float x498 = 1.0f / x497;
x490[x491] = x498;

}
float* x502 = (float*)myMalloc(150 * sizeof(float));
for(int x503=0; x503 < 150; x503++) {
x502[x503] = 0.0f;

}
// dot WrappedArray(150, 150) - WrappedArray(150)
int32_t x508 = 0;
float* x509 = (float*)myMalloc(150 * sizeof(float));
for(int x510=0; x510 < 150; x510++) {
float x511 = 0.0f;
for(int x512=0; x512 < 150; x512++) {
int32_t x513 = x508;
float x514 = x96[x513];
float x515 = x317[x512];
float x516 = x514 * x515;
x511 += x516;
x508 += 1;

}
float x521 = x511;
x509[x510] = x521;

}
float* x525 = (float*)myMalloc(150 * sizeof(float));
for(int x526=0; x526 < 150; x526++) {
x525[x526] = 0.0f;

}
// dot WrappedArray(150, 300) - WrappedArray(300)
int32_t x531 = 0;
float* x532 = (float*)myMalloc(150 * sizeof(float));
for(int x533=0; x533 < 150; x533++) {
float x534 = 0.0f;
for(int x535=0; x535 < 300; x535++) {
int32_t x536 = x531;
float x537 = x103[x536];
float x538 = x323[x535];
float x539 = x537 * x538;
x534 += x539;
x531 += 1;

}
float x544 = x534;
x532[x533] = x544;

}
float* x548 = (float*)myMalloc(150 * sizeof(float));
for(int x549=0; x549 < 150; x549++) {
x548[x549] = 0.0f;

}
float* x553 = (float*)myMalloc(150 * sizeof(float));
for(int x554=0; x554 < 150; x554++) {
float x555 = x509[x554];
float x556 = x532[x554];
float x557 = x555 + x556;
x553[x554] = x557;

}
float* x561 = (float*)myMalloc(150 * sizeof(float));
for(int x562=0; x562 < 150; x562++) {
x561[x562] = 0.0f;

}
float* x566 = (float*)myMalloc(150 * sizeof(float));
for(int x567=0; x567 < 150; x567++) {
float x568 = x553[x567];
float x569 = x110[x567];
float x570 = x568 + x569;
x566[x567] = x570;

}
float* x574 = (float*)myMalloc(150 * sizeof(float));
for(int x575=0; x575 < 150; x575++) {
x574[x575] = 0.0f;

}
float* x579 = (float*)myMalloc(150 * sizeof(float));
for(int x580=0; x580 < 150; x580++) {
float x581 = x566[x580];
float x582 = -1.0f * x581;
double x583 = (double)x582;
double x584 = exp(x583);
float x585 = (float)x584;
float x586 = x585 + 1.0f;
float x587 = 1.0f / x586;
x579[x580] = x587;

}
float* x591 = (float*)myMalloc(150 * sizeof(float));
for(int x592=0; x592 < 150; x592++) {
x591[x592] = 0.0f;

}
// dot WrappedArray(150, 150) - WrappedArray(150)
int32_t x597 = 0;
float* x598 = (float*)myMalloc(150 * sizeof(float));
for(int x599=0; x599 < 150; x599++) {
float x600 = 0.0f;
for(int x601=0; x601 < 150; x601++) {
int32_t x602 = x597;
float x603 = x77[x602];
float x604 = x317[x601];
float x605 = x603 * x604;
x600 += x605;
x597 += 1;

}
float x610 = x600;
x598[x599] = x610;

}
float* x614 = (float*)myMalloc(150 * sizeof(float));
for(int x615=0; x615 < 150; x615++) {
x614[x615] = 0.0f;

}
// dot WrappedArray(150, 300) - WrappedArray(300)
int32_t x620 = 0;
float* x621 = (float*)myMalloc(150 * sizeof(float));
for(int x622=0; x622 < 150; x622++) {
float x623 = 0.0f;
for(int x624=0; x624 < 300; x624++) {
int32_t x625 = x620;
float x626 = x84[x625];
float x627 = x323[x624];
float x628 = x626 * x627;
x623 += x628;
x620 += 1;

}
float x633 = x623;
x621[x622] = x633;

}
float* x637 = (float*)myMalloc(150 * sizeof(float));
for(int x638=0; x638 < 150; x638++) {
x637[x638] = 0.0f;

}
float* x642 = (float*)myMalloc(150 * sizeof(float));
for(int x643=0; x643 < 150; x643++) {
float x644 = x598[x643];
float x645 = x621[x643];
float x646 = x644 + x645;
x642[x643] = x646;

}
float* x650 = (float*)myMalloc(150 * sizeof(float));
for(int x651=0; x651 < 150; x651++) {
x650[x651] = 0.0f;

}
float* x655 = (float*)myMalloc(150 * sizeof(float));
for(int x656=0; x656 < 150; x656++) {
float x657 = x642[x656];
float x658 = x91[x656];
float x659 = x657 + x658;
x655[x656] = x659;

}
float* x663 = (float*)myMalloc(150 * sizeof(float));
for(int x664=0; x664 < 150; x664++) {
x663[x664] = 0.0f;

}
float* x668 = (float*)myMalloc(150 * sizeof(float));
for(int x669=0; x669 < 150; x669++) {
float x670 = x655[x669];
double x671 = (double)x670;
double x672 = tanh(x671);
float x673 = (float)x672;
x668[x669] = x673;

}
float* x677 = (float*)myMalloc(150 * sizeof(float));
for(int x678=0; x678 < 150; x678++) {
x677[x678] = 0.0f;

}
float* x682 = (float*)myMalloc(150 * sizeof(float));
for(int x683=0; x683 < 150; x683++) {
float x684 = x401[x683];
float x685 = x319[x683];
float x686 = x684 * x685;
x682[x683] = x686;

}
float* x690 = (float*)myMalloc(150 * sizeof(float));
for(int x691=0; x691 < 150; x691++) {
x690[x691] = 0.0f;

}
float* x695 = (float*)myMalloc(150 * sizeof(float));
for(int x696=0; x696 < 150; x696++) {
float x697 = x490[x696];
float x698 = x668[x696];
float x699 = x697 * x698;
x695[x696] = x699;

}
float* x703 = (float*)myMalloc(150 * sizeof(float));
for(int x704=0; x704 < 150; x704++) {
x703[x704] = 0.0f;

}
float* x708 = (float*)myMalloc(150 * sizeof(float));
for(int x709=0; x709 < 150; x709++) {
float x710 = x682[x709];
float x711 = x695[x709];
float x712 = x710 + x711;
x708[x709] = x712;

}
float* x716 = (float*)myMalloc(150 * sizeof(float));
for(int x717=0; x717 < 150; x717++) {
x716[x717] = 0.0f;

}
float* x721 = (float*)myMalloc(150 * sizeof(float));
for(int x722=0; x722 < 150; x722++) {
float x723 = x708[x722];
double x724 = (double)x723;
double x725 = tanh(x724);
float x726 = (float)x725;
x721[x722] = x726;

}
float* x730 = (float*)myMalloc(150 * sizeof(float));
for(int x731=0; x731 < 150; x731++) {
x730[x731] = 0.0f;

}
float* x735 = (float*)myMalloc(150 * sizeof(float));
for(int x736=0; x736 < 150; x736++) {
float x737 = x579[x736];
float x738 = x721[x736];
float x739 = x737 * x738;
x735[x736] = x739;

}
float* x743 = (float*)myMalloc(150 * sizeof(float));
for(int x744=0; x744 < 150; x744++) {
x743[x744] = 0.0f;

}
float** x749 = (float**)myMalloc(4 * sizeof(float*));
x749[0] = x735;
x749[1] = x743;
x749[2] = x708;
x749[3] = x716;
int32_t x880 = 0;
int32_t x896 = 0;
int32_t x913 = 0;
int32_t x929 = 0;
int32_t x987 = 0;
int32_t x1003 = 0;
int32_t x1020 = 0;
int32_t x1036 = 0;
int32_t x1094 = 0;
int32_t x1110 = 0;
int32_t x1127 = 0;
int32_t x1143 = 0;
int32_t x1201 = 0;
int32_t x1217 = 0;
int32_t x1234 = 0;
int32_t x1250 = 0;
int32_t x748 = x315 + 1;
x312(x748,x749);
for(int x756=0; x756 < 150; x756++) {
float x757 = x591[x756];
float x758 = x721[x756];
float x759 = x743[x756];
float x760 = x758 * x759;
float x761 = x757 + x760;
x591[x756] = x761;

}
for(int x765=0; x765 < 150; x765++) {
float x766 = x730[x765];
float x767 = x579[x765];
float x768 = x743[x765];
float x769 = x767 * x768;
float x770 = x766 + x769;
x730[x765] = x770;

}
// backpropagate tanh
for(int x775=0; x775 < 150; x775++) {
float x776 = x716[x775];
float x777 = x721[x775];
float x780 = x730[x775];
float x778 = x777 * x777;
float x779 = 1.0f - x778;
float x781 = x779 * x780;
float x782 = x776 + x781;
x716[x775] = x782;

}
// backpropagate +
for(int x787=0; x787 < 150; x787++) {
float x788 = x690[x787];
float x789 = x716[x787];
float x790 = x788 + x789;
x690[x787] = x790;

}
for(int x794=0; x794 < 150; x794++) {
float x795 = x703[x794];
float x796 = x716[x794];
float x797 = x795 + x796;
x703[x794] = x797;

}
for(int x801=0; x801 < 150; x801++) {
float x802 = x502[x801];
float x803 = x668[x801];
float x804 = x703[x801];
float x805 = x803 * x804;
float x806 = x802 + x805;
x502[x801] = x806;

}
for(int x810=0; x810 < 150; x810++) {
float x811 = x677[x810];
float x812 = x490[x810];
float x813 = x703[x810];
float x814 = x812 * x813;
float x815 = x811 + x814;
x677[x810] = x815;

}
for(int x819=0; x819 < 150; x819++) {
float x820 = x413[x819];
float x821 = x319[x819];
float x822 = x690[x819];
float x823 = x821 * x822;
float x824 = x820 + x823;
x413[x819] = x824;

}
for(int x828=0; x828 < 150; x828++) {
float x829 = x320[x828];
float x830 = x401[x828];
float x831 = x690[x828];
float x832 = x830 * x831;
float x833 = x829 + x832;
x320[x828] = x833;

}
// backpropagate tanh
for(int x838=0; x838 < 150; x838++) {
float x839 = x663[x838];
float x840 = x668[x838];
float x843 = x677[x838];
float x841 = x840 * x840;
float x842 = 1.0f - x841;
float x844 = x842 * x843;
float x845 = x839 + x844;
x663[x838] = x845;

}
// backpropagate +
for(int x850=0; x850 < 150; x850++) {
float x851 = x650[x850];
float x852 = x663[x850];
float x853 = x851 + x852;
x650[x850] = x853;

}
for(int x857=0; x857 < 150; x857++) {
float x858 = x179[x857];
float x859 = x663[x857];
float x860 = x858 + x859;
x179[x857] = x860;

}
// backpropagate +
for(int x865=0; x865 < 150; x865++) {
float x866 = x614[x865];
float x867 = x650[x865];
float x868 = x866 + x867;
x614[x865] = x868;

}
for(int x872=0; x872 < 150; x872++) {
float x873 = x637[x872];
float x874 = x650[x872];
float x875 = x873 + x874;
x637[x872] = x875;

}
// add_cartesian
for(int x881=0; x881 < 150; x881++) {
for(int x882=0; x882 < 300; x882++) {
int32_t x883 = x880;
int32_t x884 = x883 + x882;
float x885 = x174[x884];
float x886 = x323[x882];
float x887 = x637[x881];
float x888 = x886 * x887;
float x889 = x885 + x888;
x174[x884] = x889;

}
x880 += 300;

}
for(int x897=0; x897 < 150; x897++) {
for(int x898=0; x898 < 300; x898++) {
float x899 = x324[x898];
int32_t x900 = x896;
int32_t x901 = x900 + x898;
float x902 = x84[x901];
float x903 = x637[x897];
float x904 = x902 * x903;
float x905 = x899 + x904;
x324[x898] = x905;

}
x896 += 300;

}
// add_cartesian
for(int x914=0; x914 < 150; x914++) {
for(int x915=0; x915 < 150; x915++) {
int32_t x916 = x913;
int32_t x917 = x916 + x915;
float x918 = x169[x917];
float x919 = x317[x915];
float x920 = x614[x914];
float x921 = x919 * x920;
float x922 = x918 + x921;
x169[x917] = x922;

}
x913 += 150;

}
for(int x930=0; x930 < 150; x930++) {
for(int x931=0; x931 < 150; x931++) {
float x932 = x318[x931];
int32_t x933 = x929;
int32_t x934 = x933 + x931;
float x935 = x77[x934];
float x936 = x614[x930];
float x937 = x935 * x936;
float x938 = x932 + x937;
x318[x931] = x938;

}
x929 += 150;

}
for(int x945=0; x945 < 150; x945++) {
float x946 = x574[x945];
float x947 = x579[x945];
float x950 = x591[x945];
float x948 = 1.0f - x947;
float x949 = x948 * x947;
float x951 = x949 * x950;
float x952 = x946 + x951;
x574[x945] = x952;

}
// backpropagate +
for(int x957=0; x957 < 150; x957++) {
float x958 = x561[x957];
float x959 = x574[x957];
float x960 = x958 + x959;
x561[x957] = x960;

}
for(int x964=0; x964 < 150; x964++) {
float x965 = x194[x964];
float x966 = x574[x964];
float x967 = x965 + x966;
x194[x964] = x967;

}
// backpropagate +
for(int x972=0; x972 < 150; x972++) {
float x973 = x525[x972];
float x974 = x561[x972];
float x975 = x973 + x974;
x525[x972] = x975;

}
for(int x979=0; x979 < 150; x979++) {
float x980 = x548[x979];
float x981 = x561[x979];
float x982 = x980 + x981;
x548[x979] = x982;

}
// add_cartesian
for(int x988=0; x988 < 150; x988++) {
for(int x989=0; x989 < 300; x989++) {
int32_t x990 = x987;
int32_t x991 = x990 + x989;
float x992 = x189[x991];
float x993 = x323[x989];
float x994 = x548[x988];
float x995 = x993 * x994;
float x996 = x992 + x995;
x189[x991] = x996;

}
x987 += 300;

}
for(int x1004=0; x1004 < 150; x1004++) {
for(int x1005=0; x1005 < 300; x1005++) {
float x1006 = x324[x1005];
int32_t x1007 = x1003;
int32_t x1008 = x1007 + x1005;
float x1009 = x103[x1008];
float x1010 = x548[x1004];
float x1011 = x1009 * x1010;
float x1012 = x1006 + x1011;
x324[x1005] = x1012;

}
x1003 += 300;

}
// add_cartesian
for(int x1021=0; x1021 < 150; x1021++) {
for(int x1022=0; x1022 < 150; x1022++) {
int32_t x1023 = x1020;
int32_t x1024 = x1023 + x1022;
float x1025 = x184[x1024];
float x1026 = x317[x1022];
float x1027 = x525[x1021];
float x1028 = x1026 * x1027;
float x1029 = x1025 + x1028;
x184[x1024] = x1029;

}
x1020 += 150;

}
for(int x1037=0; x1037 < 150; x1037++) {
for(int x1038=0; x1038 < 150; x1038++) {
float x1039 = x318[x1038];
int32_t x1040 = x1036;
int32_t x1041 = x1040 + x1038;
float x1042 = x96[x1041];
float x1043 = x525[x1037];
float x1044 = x1042 * x1043;
float x1045 = x1039 + x1044;
x318[x1038] = x1045;

}
x1036 += 150;

}
for(int x1052=0; x1052 < 150; x1052++) {
float x1053 = x485[x1052];
float x1054 = x490[x1052];
float x1057 = x502[x1052];
float x1055 = 1.0f - x1054;
float x1056 = x1055 * x1054;
float x1058 = x1056 * x1057;
float x1059 = x1053 + x1058;
x485[x1052] = x1059;

}
// backpropagate +
for(int x1064=0; x1064 < 150; x1064++) {
float x1065 = x472[x1064];
float x1066 = x485[x1064];
float x1067 = x1065 + x1066;
x472[x1064] = x1067;

}
for(int x1071=0; x1071 < 150; x1071++) {
float x1072 = x164[x1071];
float x1073 = x485[x1071];
float x1074 = x1072 + x1073;
x164[x1071] = x1074;

}
// backpropagate +
for(int x1079=0; x1079 < 150; x1079++) {
float x1080 = x436[x1079];
float x1081 = x472[x1079];
float x1082 = x1080 + x1081;
x436[x1079] = x1082;

}
for(int x1086=0; x1086 < 150; x1086++) {
float x1087 = x459[x1086];
float x1088 = x472[x1086];
float x1089 = x1087 + x1088;
x459[x1086] = x1089;

}
// add_cartesian
for(int x1095=0; x1095 < 150; x1095++) {
for(int x1096=0; x1096 < 300; x1096++) {
int32_t x1097 = x1094;
int32_t x1098 = x1097 + x1096;
float x1099 = x159[x1098];
float x1100 = x323[x1096];
float x1101 = x459[x1095];
float x1102 = x1100 * x1101;
float x1103 = x1099 + x1102;
x159[x1098] = x1103;

}
x1094 += 300;

}
for(int x1111=0; x1111 < 150; x1111++) {
for(int x1112=0; x1112 < 300; x1112++) {
float x1113 = x324[x1112];
int32_t x1114 = x1110;
int32_t x1115 = x1114 + x1112;
float x1116 = x65[x1115];
float x1117 = x459[x1111];
float x1118 = x1116 * x1117;
float x1119 = x1113 + x1118;
x324[x1112] = x1119;

}
x1110 += 300;

}
// add_cartesian
for(int x1128=0; x1128 < 150; x1128++) {
for(int x1129=0; x1129 < 150; x1129++) {
int32_t x1130 = x1127;
int32_t x1131 = x1130 + x1129;
float x1132 = x154[x1131];
float x1133 = x317[x1129];
float x1134 = x436[x1128];
float x1135 = x1133 * x1134;
float x1136 = x1132 + x1135;
x154[x1131] = x1136;

}
x1127 += 150;

}
for(int x1144=0; x1144 < 150; x1144++) {
for(int x1145=0; x1145 < 150; x1145++) {
float x1146 = x318[x1145];
int32_t x1147 = x1143;
int32_t x1148 = x1147 + x1145;
float x1149 = x58[x1148];
float x1150 = x436[x1144];
float x1151 = x1149 * x1150;
float x1152 = x1146 + x1151;
x318[x1145] = x1152;

}
x1143 += 150;

}
for(int x1159=0; x1159 < 150; x1159++) {
float x1160 = x396[x1159];
float x1161 = x401[x1159];
float x1164 = x413[x1159];
float x1162 = 1.0f - x1161;
float x1163 = x1162 * x1161;
float x1165 = x1163 * x1164;
float x1166 = x1160 + x1165;
x396[x1159] = x1166;

}
// backpropagate +
for(int x1171=0; x1171 < 150; x1171++) {
float x1172 = x383[x1171];
float x1173 = x396[x1171];
float x1174 = x1172 + x1173;
x383[x1171] = x1174;

}
for(int x1178=0; x1178 < 150; x1178++) {
float x1179 = x149[x1178];
float x1180 = x396[x1178];
float x1181 = x1179 + x1180;
x149[x1178] = x1181;

}
// backpropagate +
for(int x1186=0; x1186 < 150; x1186++) {
float x1187 = x347[x1186];
float x1188 = x383[x1186];
float x1189 = x1187 + x1188;
x347[x1186] = x1189;

}
for(int x1193=0; x1193 < 150; x1193++) {
float x1194 = x370[x1193];
float x1195 = x383[x1193];
float x1196 = x1194 + x1195;
x370[x1193] = x1196;

}
// add_cartesian
for(int x1202=0; x1202 < 150; x1202++) {
for(int x1203=0; x1203 < 300; x1203++) {
int32_t x1204 = x1201;
int32_t x1205 = x1204 + x1203;
float x1206 = x144[x1205];
float x1207 = x323[x1203];
float x1208 = x370[x1202];
float x1209 = x1207 * x1208;
float x1210 = x1206 + x1209;
x144[x1205] = x1210;

}
x1201 += 300;

}
for(int x1218=0; x1218 < 150; x1218++) {
for(int x1219=0; x1219 < 300; x1219++) {
float x1220 = x324[x1219];
int32_t x1221 = x1217;
int32_t x1222 = x1221 + x1219;
float x1223 = x44[x1222];
float x1224 = x370[x1218];
float x1225 = x1223 * x1224;
float x1226 = x1220 + x1225;
x324[x1219] = x1226;

}
x1217 += 300;

}
// add_cartesian
for(int x1235=0; x1235 < 150; x1235++) {
for(int x1236=0; x1236 < 150; x1236++) {
int32_t x1237 = x1234;
int32_t x1238 = x1237 + x1236;
float x1239 = x139[x1238];
float x1240 = x317[x1236];
float x1241 = x347[x1235];
float x1242 = x1240 * x1241;
float x1243 = x1239 + x1242;
x139[x1238] = x1243;

}
x1234 += 150;

}
for(int x1251=0; x1251 < 150; x1251++) {
for(int x1252=0; x1252 < 150; x1252++) {
float x1253 = x318[x1252];
int32_t x1254 = x1250;
int32_t x1255 = x1254 + x1252;
float x1256 = x36[x1255];
float x1257 = x347[x1251];
float x1258 = x1256 * x1257;
float x1259 = x1253 + x1258;
x318[x1252] = x1259;

}
x1250 += 150;

}
} else {
// dot WrappedArray(5, 150) - WrappedArray(150)
int32_t x1268 = 0;
float* x1269 = (float*)myMalloc(5 * sizeof(float));
for(int x1270=0; x1270 < 5; x1270++) {
float x1271 = 0.0f;
for(int x1272=0; x1272 < 150; x1272++) {
int32_t x1273 = x1268;
float x1274 = x115[x1273];
float x1275 = x317[x1272];
float x1276 = x1274 * x1275;
x1271 += x1276;
x1268 += 1;

}
float x1281 = x1271;
x1269[x1270] = x1281;

}
float* x1285 = (float*)myMalloc(5 * sizeof(float));
for(int x1286=0; x1286 < 5; x1286++) {
x1285[x1286] = 0.0f;

}
float* x1290 = (float*)myMalloc(5 * sizeof(float));
for(int x1291=0; x1291 < 5; x1291++) {
float x1292 = x1269[x1291];
float x1293 = x123[x1291];
float x1294 = x1292 + x1293;
x1290[x1291] = x1294;

}
float* x1298 = (float*)myMalloc(5 * sizeof(float));
for(int x1299=0; x1299 < 5; x1299++) {
x1298[x1299] = 0.0f;

}
float* x1303 = (float*)myMalloc(5 * sizeof(float));
for(int x1304=0; x1304 < 5; x1304++) {
float x1305 = x1290[x1304];
double x1306 = (double)x1305;
double x1307 = exp(x1306);
float x1308 = (float)x1307;
x1303[x1304] = x1308;

}
float* x1312 = (float*)myMalloc(5 * sizeof(float));
for(int x1313=0; x1313 < 5; x1313++) {
x1312[x1313] = 0.0f;

}
float x1317 = 0.0f;
for(int x1318=0; x1318 < 5; x1318++) {
float x1319 = x1317;
float x1320 = x1303[x1318];
float x1321 = x1319 + x1320;
x1317 = x1321;

}
float x1325 = x1317;
float* x1326 = (float*)myMalloc(1 * sizeof(float));
x1326[0] = x1325;
float* x1328 = (float*)myMalloc(1 * sizeof(float));
for(int x1329=0; x1329 < 1; x1329++) {
x1328[x1329] = 0.0f;

}
float x1333 = x1326[0];
float* x1334 = (float*)myMalloc(5 * sizeof(float));
for(int x1335=0; x1335 < 5; x1335++) {
float x1336 = x1303[x1335];
float x1337 = x1336 / x1333;
x1334[x1335] = x1337;

}
float* x1341 = (float*)myMalloc(5 * sizeof(float));
for(int x1342=0; x1342 < 5; x1342++) {
x1341[x1342] = 0.0f;

}
float* x1346 = (float*)myMalloc(5 * sizeof(float));
for(int x1347=0; x1347 < 5; x1347++) {
x1346[x1347] = 0.0f;

}
x1346[x294] = 1.0f;
float* x1352 = (float*)myMalloc(5 * sizeof(float));
for(int x1353=0; x1353 < 5; x1353++) {
x1352[x1353] = 0.0f;

}
// dot WrappedArray(5) - WrappedArray(5)
int32_t x1358 = 0;
float* x1359 = (float*)myMalloc(1 * sizeof(float));
float x1360 = 0.0f;
for(int x1361=0; x1361 < 5; x1361++) {
int32_t x1362 = x1358;
float x1363 = x1334[x1362];
float x1364 = x1346[x1361];
float x1365 = x1363 * x1364;
x1360 += x1365;
x1358 += 1;

}
float x1370 = x1360;
x1359[0] = x1370;
float* x1372 = (float*)myMalloc(1 * sizeof(float));
for(int x1373=0; x1373 < 1; x1373++) {
x1372[x1373] = 0.0f;

}
float* x1377 = (float*)myMalloc(1 * sizeof(float));
float x1378 = x1359[0];
double x1379 = (double)x1378;
double x1380 = log(x1379);
float x1381 = (float)x1380;
x1377[0] = x1381;
float* x1383 = (float*)myMalloc(1 * sizeof(float));
for(int x1384=0; x1384 < 1; x1384++) {
x1383[x1384] = 0.0f;

}
float* x1388 = (float*)myMalloc(1 * sizeof(float));
for(int x1389=0; x1389 < 1; x1389++) {
x1388[x1389] = 0.0f;

}
float* x1393 = (float*)myMalloc(1 * sizeof(float));
for(int x1394=0; x1394 < 1; x1394++) {
x1393[x1394] = 0.0f;

}
float* x1398 = (float*)myMalloc(1 * sizeof(float));
float x1399 = x1377[0];
float x1400 = x1388[0];
float x1401 = x1400 - x1399;
x1398[0] = x1401;
float* x1403 = (float*)myMalloc(1 * sizeof(float));
for(int x1404=0; x1404 < 1; x1404++) {
x1403[x1404] = 0.0f;

}
float x1408 = x1403[0];
x1403[0] = 1.0f;
float x1410 = x1398[0];
x306[0] = x1410;
// += tensor of dim 0
float x1413 = x1403[0];
float x1414 = x1393[0];
float x1415 = x1414 + x1413;
x1393[0] = x1415;
float x1417 = x1403[0];
float x1418 = x1383[0];
float x1419 = x1418 - x1417;
x1383[0] = x1419;
float x1421 = x1372[0];
float x1422 = x1383[0];
float x1423 = x1359[0];
float x1424 = x1422 / x1423;
float x1425 = x1421 + x1424;
x1372[0] = x1425;
float x1427 = x1372[0];
// Generate code for addMul
for(int x1429=0; x1429 < 5; x1429++) {
float x1430 = x1341[x1429];
float x1431 = x1346[x1429];
float x1432 = x1427 * x1431;
float x1433 = x1430 + x1432;
x1341[x1429] = x1433;

}
float x1437 = x1372[0];
// Generate code for addMul
for(int x1439=0; x1439 < 5; x1439++) {
float x1440 = x1352[x1439];
float x1441 = x1334[x1439];
float x1442 = x1437 * x1441;
float x1443 = x1440 + x1442;
x1352[x1439] = x1443;

}
for(int x1447=0; x1447 < 5; x1447++) {
float x1448 = x1312[x1447];
float x1449 = x1341[x1447];
float x1450 = x1326[0];
float x1451 = x1449 / x1450;
float x1452 = x1448 + x1451;
x1312[x1447] = x1452;

}
for(int x1456=0; x1456 < 5; x1456++) {
float x1457 = x1328[0];
float x1458 = x1303[x1456];
float x1459 = x1341[x1456];
float x1461 = x1326[0];
float x1460 = x1458 * x1459;
float x1462 = x1461 * x1461;
float x1463 = x1460 / x1462;
float x1464 = x1457 - x1463;
x1328[0] = x1464;

}
// += tensor of dim 0
float x1469 = x1328[0];
for(int x1470=0; x1470 < 5; x1470++) {
float x1471 = x1312[x1470];
float x1472 = x1471 + x1469;
x1312[x1470] = x1472;

}
// backpropage exp
for(int x1477=0; x1477 < 5; x1477++) {
float x1478 = x1298[x1477];
float x1479 = x1303[x1477];
float x1480 = x1312[x1477];
float x1481 = x1479 * x1480;
float x1482 = x1478 + x1481;
x1298[x1477] = x1482;

}
// backpropagate +
for(int x1487=0; x1487 < 5; x1487++) {
float x1488 = x1285[x1487];
float x1489 = x1298[x1487];
float x1490 = x1488 + x1489;
x1285[x1487] = x1490;

}
for(int x1494=0; x1494 < 5; x1494++) {
float x1495 = x204[x1494];
float x1496 = x1298[x1494];
float x1497 = x1495 + x1496;
x204[x1494] = x1497;

}
// add_cartesian
int32_t x1502 = 0;
for(int x1503=0; x1503 < 5; x1503++) {
for(int x1504=0; x1504 < 150; x1504++) {
int32_t x1505 = x1502;
int32_t x1506 = x1505 + x1504;
float x1507 = x199[x1506];
float x1508 = x317[x1504];
float x1509 = x1285[x1503];
float x1510 = x1508 * x1509;
float x1511 = x1507 + x1510;
x199[x1506] = x1511;

}
x1502 += 150;

}
int32_t x1518 = 0;
for(int x1519=0; x1519 < 5; x1519++) {
for(int x1520=0; x1520 < 150; x1520++) {
float x1521 = x318[x1520];
int32_t x1522 = x1518;
int32_t x1523 = x1522 + x1520;
float x1524 = x115[x1523];
float x1525 = x1285[x1519];
float x1526 = x1524 * x1525;
float x1527 = x1521 + x1526;
x318[x1520] = x1527;

}
x1518 += 150;

}
}
};
float* x295 = (float*)myMalloc(1 * sizeof(float));
for(int x297=0; x297 < 1; x297++) {
x295[x297] = 0.0f;

}
float* x301 = (float*)myMalloc(1 * sizeof(float));
for(int x302=0; x302 < 1; x302++) {
x301[x302] = 0.0f;

}
for(int x307=0; x307 < 1; x307++) {
x306[x307] = 0.0f;

}
float** x1537 = (float**)myMalloc(4 * sizeof(float*));
x1537[0] = x129;
x1537[1] = x209;
x1537[2] = x134;
x1537[3] = x214;
x312(0,x1537);
float x1544 = x306[0];
int32_t x1545 = x291 % 100;
bool x1546 = x1545 == 0;
if (x1546) {
printf("iter %d, loss %f\n",x291,x1544);
} else {
}
for(int x1550=0; x1550 < 22500; x1550++) {
float x1551 = x139[x1550];
bool x1552 = x1551 > 5.0f;
if (x1552) {
x139[x1550] = 5.0f;
} else {
}
float x1556 = x139[x1550];
bool x1557 = x1556 < -5.0f;
if (x1557) {
x139[x1550] = -5.0f;
} else {
}

}
float* x1563 = (float*)myMalloc(22500 * sizeof(float));
for(int x1564=0; x1564 < 22500; x1564++) {
float x1565 = x139[x1564];
float x1566 = x139[x1564];
float x1567 = x1565 * x1566;
x1563[x1564] = x1567;

}
for(int x1571=0; x1571 < 22500; x1571++) {
float x1572 = x219[x1571];
float x1573 = x1563[x1571];
float x1574 = x1572 + x1573;
x219[x1571] = x1574;

}
float* x1578 = (float*)myMalloc(22500 * sizeof(float));
for(int x1579=0; x1579 < 22500; x1579++) {
float x1580 = x139[x1579];
float x1581 = x1580 * 0.1f;
x1578[x1579] = x1581;

}
float* x1585 = (float*)myMalloc(22500 * sizeof(float));
for(int x1586=0; x1586 < 22500; x1586++) {
float x1587 = x219[x1586];
float x1588 = x1587 + 1.0E-8f;
x1585[x1586] = x1588;

}
float* x1592 = (float*)myMalloc(22500 * sizeof(float));
for(int x1593=0; x1593 < 22500; x1593++) {
float x1594 = x1585[x1593];
double x1595 = (double)x1594;
double x1596 = sqrt(x1595);
float x1597 = (float)x1596;
x1592[x1593] = x1597;

}
float* x1601 = (float*)myMalloc(22500 * sizeof(float));
for(int x1602=0; x1602 < 22500; x1602++) {
float x1603 = x1578[x1602];
float x1604 = x1592[x1602];
float x1605 = x1603 / x1604;
x1601[x1602] = x1605;

}
for(int x1609=0; x1609 < 22500; x1609++) {
float x1610 = x36[x1609];
float x1611 = x1601[x1609];
float x1612 = x1610 - x1611;
x36[x1609] = x1612;

}
for(int x1616=0; x1616 < 22500; x1616++) {
float x1617 = x139[x1616];
x139[x1616] = 0.0f;

}
for(int x1621=0; x1621 < 45000; x1621++) {
float x1622 = x144[x1621];
bool x1623 = x1622 > 5.0f;
if (x1623) {
x144[x1621] = 5.0f;
} else {
}
float x1627 = x144[x1621];
bool x1628 = x1627 < -5.0f;
if (x1628) {
x144[x1621] = -5.0f;
} else {
}

}
float* x1634 = (float*)myMalloc(45000 * sizeof(float));
for(int x1635=0; x1635 < 45000; x1635++) {
float x1636 = x144[x1635];
float x1637 = x144[x1635];
float x1638 = x1636 * x1637;
x1634[x1635] = x1638;

}
for(int x1642=0; x1642 < 45000; x1642++) {
float x1643 = x224[x1642];
float x1644 = x1634[x1642];
float x1645 = x1643 + x1644;
x224[x1642] = x1645;

}
float* x1649 = (float*)myMalloc(45000 * sizeof(float));
for(int x1650=0; x1650 < 45000; x1650++) {
float x1651 = x144[x1650];
float x1652 = x1651 * 0.1f;
x1649[x1650] = x1652;

}
float* x1656 = (float*)myMalloc(45000 * sizeof(float));
for(int x1657=0; x1657 < 45000; x1657++) {
float x1658 = x224[x1657];
float x1659 = x1658 + 1.0E-8f;
x1656[x1657] = x1659;

}
float* x1663 = (float*)myMalloc(45000 * sizeof(float));
for(int x1664=0; x1664 < 45000; x1664++) {
float x1665 = x1656[x1664];
double x1666 = (double)x1665;
double x1667 = sqrt(x1666);
float x1668 = (float)x1667;
x1663[x1664] = x1668;

}
float* x1672 = (float*)myMalloc(45000 * sizeof(float));
for(int x1673=0; x1673 < 45000; x1673++) {
float x1674 = x1649[x1673];
float x1675 = x1663[x1673];
float x1676 = x1674 / x1675;
x1672[x1673] = x1676;

}
for(int x1680=0; x1680 < 45000; x1680++) {
float x1681 = x44[x1680];
float x1682 = x1672[x1680];
float x1683 = x1681 - x1682;
x44[x1680] = x1683;

}
for(int x1687=0; x1687 < 45000; x1687++) {
float x1688 = x144[x1687];
x144[x1687] = 0.0f;

}
for(int x1692=0; x1692 < 150; x1692++) {
float x1693 = x149[x1692];
bool x1694 = x1693 > 5.0f;
if (x1694) {
x149[x1692] = 5.0f;
} else {
}
float x1698 = x149[x1692];
bool x1699 = x1698 < -5.0f;
if (x1699) {
x149[x1692] = -5.0f;
} else {
}

}
float* x1705 = (float*)myMalloc(150 * sizeof(float));
for(int x1706=0; x1706 < 150; x1706++) {
float x1707 = x149[x1706];
float x1708 = x149[x1706];
float x1709 = x1707 * x1708;
x1705[x1706] = x1709;

}
for(int x1713=0; x1713 < 150; x1713++) {
float x1714 = x229[x1713];
float x1715 = x1705[x1713];
float x1716 = x1714 + x1715;
x229[x1713] = x1716;

}
float* x1720 = (float*)myMalloc(150 * sizeof(float));
for(int x1721=0; x1721 < 150; x1721++) {
float x1722 = x149[x1721];
float x1723 = x1722 * 0.1f;
x1720[x1721] = x1723;

}
float* x1727 = (float*)myMalloc(150 * sizeof(float));
for(int x1728=0; x1728 < 150; x1728++) {
float x1729 = x229[x1728];
float x1730 = x1729 + 1.0E-8f;
x1727[x1728] = x1730;

}
float* x1734 = (float*)myMalloc(150 * sizeof(float));
for(int x1735=0; x1735 < 150; x1735++) {
float x1736 = x1727[x1735];
double x1737 = (double)x1736;
double x1738 = sqrt(x1737);
float x1739 = (float)x1738;
x1734[x1735] = x1739;

}
float* x1743 = (float*)myMalloc(150 * sizeof(float));
for(int x1744=0; x1744 < 150; x1744++) {
float x1745 = x1720[x1744];
float x1746 = x1734[x1744];
float x1747 = x1745 / x1746;
x1743[x1744] = x1747;

}
for(int x1751=0; x1751 < 150; x1751++) {
float x1752 = x52[x1751];
float x1753 = x1743[x1751];
float x1754 = x1752 - x1753;
x52[x1751] = x1754;

}
for(int x1758=0; x1758 < 150; x1758++) {
float x1759 = x149[x1758];
x149[x1758] = 0.0f;

}
for(int x1763=0; x1763 < 22500; x1763++) {
float x1764 = x154[x1763];
bool x1765 = x1764 > 5.0f;
if (x1765) {
x154[x1763] = 5.0f;
} else {
}
float x1769 = x154[x1763];
bool x1770 = x1769 < -5.0f;
if (x1770) {
x154[x1763] = -5.0f;
} else {
}

}
float* x1776 = (float*)myMalloc(22500 * sizeof(float));
for(int x1777=0; x1777 < 22500; x1777++) {
float x1778 = x154[x1777];
float x1779 = x154[x1777];
float x1780 = x1778 * x1779;
x1776[x1777] = x1780;

}
for(int x1784=0; x1784 < 22500; x1784++) {
float x1785 = x234[x1784];
float x1786 = x1776[x1784];
float x1787 = x1785 + x1786;
x234[x1784] = x1787;

}
float* x1791 = (float*)myMalloc(22500 * sizeof(float));
for(int x1792=0; x1792 < 22500; x1792++) {
float x1793 = x154[x1792];
float x1794 = x1793 * 0.1f;
x1791[x1792] = x1794;

}
float* x1798 = (float*)myMalloc(22500 * sizeof(float));
for(int x1799=0; x1799 < 22500; x1799++) {
float x1800 = x234[x1799];
float x1801 = x1800 + 1.0E-8f;
x1798[x1799] = x1801;

}
float* x1805 = (float*)myMalloc(22500 * sizeof(float));
for(int x1806=0; x1806 < 22500; x1806++) {
float x1807 = x1798[x1806];
double x1808 = (double)x1807;
double x1809 = sqrt(x1808);
float x1810 = (float)x1809;
x1805[x1806] = x1810;

}
float* x1814 = (float*)myMalloc(22500 * sizeof(float));
for(int x1815=0; x1815 < 22500; x1815++) {
float x1816 = x1791[x1815];
float x1817 = x1805[x1815];
float x1818 = x1816 / x1817;
x1814[x1815] = x1818;

}
for(int x1822=0; x1822 < 22500; x1822++) {
float x1823 = x58[x1822];
float x1824 = x1814[x1822];
float x1825 = x1823 - x1824;
x58[x1822] = x1825;

}
for(int x1829=0; x1829 < 22500; x1829++) {
float x1830 = x154[x1829];
x154[x1829] = 0.0f;

}
for(int x1834=0; x1834 < 45000; x1834++) {
float x1835 = x159[x1834];
bool x1836 = x1835 > 5.0f;
if (x1836) {
x159[x1834] = 5.0f;
} else {
}
float x1840 = x159[x1834];
bool x1841 = x1840 < -5.0f;
if (x1841) {
x159[x1834] = -5.0f;
} else {
}

}
float* x1847 = (float*)myMalloc(45000 * sizeof(float));
for(int x1848=0; x1848 < 45000; x1848++) {
float x1849 = x159[x1848];
float x1850 = x159[x1848];
float x1851 = x1849 * x1850;
x1847[x1848] = x1851;

}
for(int x1855=0; x1855 < 45000; x1855++) {
float x1856 = x239[x1855];
float x1857 = x1847[x1855];
float x1858 = x1856 + x1857;
x239[x1855] = x1858;

}
float* x1862 = (float*)myMalloc(45000 * sizeof(float));
for(int x1863=0; x1863 < 45000; x1863++) {
float x1864 = x159[x1863];
float x1865 = x1864 * 0.1f;
x1862[x1863] = x1865;

}
float* x1869 = (float*)myMalloc(45000 * sizeof(float));
for(int x1870=0; x1870 < 45000; x1870++) {
float x1871 = x239[x1870];
float x1872 = x1871 + 1.0E-8f;
x1869[x1870] = x1872;

}
float* x1876 = (float*)myMalloc(45000 * sizeof(float));
for(int x1877=0; x1877 < 45000; x1877++) {
float x1878 = x1869[x1877];
double x1879 = (double)x1878;
double x1880 = sqrt(x1879);
float x1881 = (float)x1880;
x1876[x1877] = x1881;

}
float* x1885 = (float*)myMalloc(45000 * sizeof(float));
for(int x1886=0; x1886 < 45000; x1886++) {
float x1887 = x1862[x1886];
float x1888 = x1876[x1886];
float x1889 = x1887 / x1888;
x1885[x1886] = x1889;

}
for(int x1893=0; x1893 < 45000; x1893++) {
float x1894 = x65[x1893];
float x1895 = x1885[x1893];
float x1896 = x1894 - x1895;
x65[x1893] = x1896;

}
for(int x1900=0; x1900 < 45000; x1900++) {
float x1901 = x159[x1900];
x159[x1900] = 0.0f;

}
for(int x1905=0; x1905 < 150; x1905++) {
float x1906 = x164[x1905];
bool x1907 = x1906 > 5.0f;
if (x1907) {
x164[x1905] = 5.0f;
} else {
}
float x1911 = x164[x1905];
bool x1912 = x1911 < -5.0f;
if (x1912) {
x164[x1905] = -5.0f;
} else {
}

}
float* x1918 = (float*)myMalloc(150 * sizeof(float));
for(int x1919=0; x1919 < 150; x1919++) {
float x1920 = x164[x1919];
float x1921 = x164[x1919];
float x1922 = x1920 * x1921;
x1918[x1919] = x1922;

}
for(int x1926=0; x1926 < 150; x1926++) {
float x1927 = x244[x1926];
float x1928 = x1918[x1926];
float x1929 = x1927 + x1928;
x244[x1926] = x1929;

}
float* x1933 = (float*)myMalloc(150 * sizeof(float));
for(int x1934=0; x1934 < 150; x1934++) {
float x1935 = x164[x1934];
float x1936 = x1935 * 0.1f;
x1933[x1934] = x1936;

}
float* x1940 = (float*)myMalloc(150 * sizeof(float));
for(int x1941=0; x1941 < 150; x1941++) {
float x1942 = x244[x1941];
float x1943 = x1942 + 1.0E-8f;
x1940[x1941] = x1943;

}
float* x1947 = (float*)myMalloc(150 * sizeof(float));
for(int x1948=0; x1948 < 150; x1948++) {
float x1949 = x1940[x1948];
double x1950 = (double)x1949;
double x1951 = sqrt(x1950);
float x1952 = (float)x1951;
x1947[x1948] = x1952;

}
float* x1956 = (float*)myMalloc(150 * sizeof(float));
for(int x1957=0; x1957 < 150; x1957++) {
float x1958 = x1933[x1957];
float x1959 = x1947[x1957];
float x1960 = x1958 / x1959;
x1956[x1957] = x1960;

}
for(int x1964=0; x1964 < 150; x1964++) {
float x1965 = x72[x1964];
float x1966 = x1956[x1964];
float x1967 = x1965 - x1966;
x72[x1964] = x1967;

}
for(int x1971=0; x1971 < 150; x1971++) {
float x1972 = x164[x1971];
x164[x1971] = 0.0f;

}
for(int x1976=0; x1976 < 22500; x1976++) {
float x1977 = x169[x1976];
bool x1978 = x1977 > 5.0f;
if (x1978) {
x169[x1976] = 5.0f;
} else {
}
float x1982 = x169[x1976];
bool x1983 = x1982 < -5.0f;
if (x1983) {
x169[x1976] = -5.0f;
} else {
}

}
float* x1989 = (float*)myMalloc(22500 * sizeof(float));
for(int x1990=0; x1990 < 22500; x1990++) {
float x1991 = x169[x1990];
float x1992 = x169[x1990];
float x1993 = x1991 * x1992;
x1989[x1990] = x1993;

}
for(int x1997=0; x1997 < 22500; x1997++) {
float x1998 = x249[x1997];
float x1999 = x1989[x1997];
float x2000 = x1998 + x1999;
x249[x1997] = x2000;

}
float* x2004 = (float*)myMalloc(22500 * sizeof(float));
for(int x2005=0; x2005 < 22500; x2005++) {
float x2006 = x169[x2005];
float x2007 = x2006 * 0.1f;
x2004[x2005] = x2007;

}
float* x2011 = (float*)myMalloc(22500 * sizeof(float));
for(int x2012=0; x2012 < 22500; x2012++) {
float x2013 = x249[x2012];
float x2014 = x2013 + 1.0E-8f;
x2011[x2012] = x2014;

}
float* x2018 = (float*)myMalloc(22500 * sizeof(float));
for(int x2019=0; x2019 < 22500; x2019++) {
float x2020 = x2011[x2019];
double x2021 = (double)x2020;
double x2022 = sqrt(x2021);
float x2023 = (float)x2022;
x2018[x2019] = x2023;

}
float* x2027 = (float*)myMalloc(22500 * sizeof(float));
for(int x2028=0; x2028 < 22500; x2028++) {
float x2029 = x2004[x2028];
float x2030 = x2018[x2028];
float x2031 = x2029 / x2030;
x2027[x2028] = x2031;

}
for(int x2035=0; x2035 < 22500; x2035++) {
float x2036 = x77[x2035];
float x2037 = x2027[x2035];
float x2038 = x2036 - x2037;
x77[x2035] = x2038;

}
for(int x2042=0; x2042 < 22500; x2042++) {
float x2043 = x169[x2042];
x169[x2042] = 0.0f;

}
for(int x2047=0; x2047 < 45000; x2047++) {
float x2048 = x174[x2047];
bool x2049 = x2048 > 5.0f;
if (x2049) {
x174[x2047] = 5.0f;
} else {
}
float x2053 = x174[x2047];
bool x2054 = x2053 < -5.0f;
if (x2054) {
x174[x2047] = -5.0f;
} else {
}

}
float* x2060 = (float*)myMalloc(45000 * sizeof(float));
for(int x2061=0; x2061 < 45000; x2061++) {
float x2062 = x174[x2061];
float x2063 = x174[x2061];
float x2064 = x2062 * x2063;
x2060[x2061] = x2064;

}
for(int x2068=0; x2068 < 45000; x2068++) {
float x2069 = x254[x2068];
float x2070 = x2060[x2068];
float x2071 = x2069 + x2070;
x254[x2068] = x2071;

}
float* x2075 = (float*)myMalloc(45000 * sizeof(float));
for(int x2076=0; x2076 < 45000; x2076++) {
float x2077 = x174[x2076];
float x2078 = x2077 * 0.1f;
x2075[x2076] = x2078;

}
float* x2082 = (float*)myMalloc(45000 * sizeof(float));
for(int x2083=0; x2083 < 45000; x2083++) {
float x2084 = x254[x2083];
float x2085 = x2084 + 1.0E-8f;
x2082[x2083] = x2085;

}
float* x2089 = (float*)myMalloc(45000 * sizeof(float));
for(int x2090=0; x2090 < 45000; x2090++) {
float x2091 = x2082[x2090];
double x2092 = (double)x2091;
double x2093 = sqrt(x2092);
float x2094 = (float)x2093;
x2089[x2090] = x2094;

}
float* x2098 = (float*)myMalloc(45000 * sizeof(float));
for(int x2099=0; x2099 < 45000; x2099++) {
float x2100 = x2075[x2099];
float x2101 = x2089[x2099];
float x2102 = x2100 / x2101;
x2098[x2099] = x2102;

}
for(int x2106=0; x2106 < 45000; x2106++) {
float x2107 = x84[x2106];
float x2108 = x2098[x2106];
float x2109 = x2107 - x2108;
x84[x2106] = x2109;

}
for(int x2113=0; x2113 < 45000; x2113++) {
float x2114 = x174[x2113];
x174[x2113] = 0.0f;

}
for(int x2118=0; x2118 < 150; x2118++) {
float x2119 = x179[x2118];
bool x2120 = x2119 > 5.0f;
if (x2120) {
x179[x2118] = 5.0f;
} else {
}
float x2124 = x179[x2118];
bool x2125 = x2124 < -5.0f;
if (x2125) {
x179[x2118] = -5.0f;
} else {
}

}
float* x2131 = (float*)myMalloc(150 * sizeof(float));
for(int x2132=0; x2132 < 150; x2132++) {
float x2133 = x179[x2132];
float x2134 = x179[x2132];
float x2135 = x2133 * x2134;
x2131[x2132] = x2135;

}
for(int x2139=0; x2139 < 150; x2139++) {
float x2140 = x259[x2139];
float x2141 = x2131[x2139];
float x2142 = x2140 + x2141;
x259[x2139] = x2142;

}
float* x2146 = (float*)myMalloc(150 * sizeof(float));
for(int x2147=0; x2147 < 150; x2147++) {
float x2148 = x179[x2147];
float x2149 = x2148 * 0.1f;
x2146[x2147] = x2149;

}
float* x2153 = (float*)myMalloc(150 * sizeof(float));
for(int x2154=0; x2154 < 150; x2154++) {
float x2155 = x259[x2154];
float x2156 = x2155 + 1.0E-8f;
x2153[x2154] = x2156;

}
float* x2160 = (float*)myMalloc(150 * sizeof(float));
for(int x2161=0; x2161 < 150; x2161++) {
float x2162 = x2153[x2161];
double x2163 = (double)x2162;
double x2164 = sqrt(x2163);
float x2165 = (float)x2164;
x2160[x2161] = x2165;

}
float* x2169 = (float*)myMalloc(150 * sizeof(float));
for(int x2170=0; x2170 < 150; x2170++) {
float x2171 = x2146[x2170];
float x2172 = x2160[x2170];
float x2173 = x2171 / x2172;
x2169[x2170] = x2173;

}
for(int x2177=0; x2177 < 150; x2177++) {
float x2178 = x91[x2177];
float x2179 = x2169[x2177];
float x2180 = x2178 - x2179;
x91[x2177] = x2180;

}
for(int x2184=0; x2184 < 150; x2184++) {
float x2185 = x179[x2184];
x179[x2184] = 0.0f;

}
for(int x2189=0; x2189 < 22500; x2189++) {
float x2190 = x184[x2189];
bool x2191 = x2190 > 5.0f;
if (x2191) {
x184[x2189] = 5.0f;
} else {
}
float x2195 = x184[x2189];
bool x2196 = x2195 < -5.0f;
if (x2196) {
x184[x2189] = -5.0f;
} else {
}

}
float* x2202 = (float*)myMalloc(22500 * sizeof(float));
for(int x2203=0; x2203 < 22500; x2203++) {
float x2204 = x184[x2203];
float x2205 = x184[x2203];
float x2206 = x2204 * x2205;
x2202[x2203] = x2206;

}
for(int x2210=0; x2210 < 22500; x2210++) {
float x2211 = x264[x2210];
float x2212 = x2202[x2210];
float x2213 = x2211 + x2212;
x264[x2210] = x2213;

}
float* x2217 = (float*)myMalloc(22500 * sizeof(float));
for(int x2218=0; x2218 < 22500; x2218++) {
float x2219 = x184[x2218];
float x2220 = x2219 * 0.1f;
x2217[x2218] = x2220;

}
float* x2224 = (float*)myMalloc(22500 * sizeof(float));
for(int x2225=0; x2225 < 22500; x2225++) {
float x2226 = x264[x2225];
float x2227 = x2226 + 1.0E-8f;
x2224[x2225] = x2227;

}
float* x2231 = (float*)myMalloc(22500 * sizeof(float));
for(int x2232=0; x2232 < 22500; x2232++) {
float x2233 = x2224[x2232];
double x2234 = (double)x2233;
double x2235 = sqrt(x2234);
float x2236 = (float)x2235;
x2231[x2232] = x2236;

}
float* x2240 = (float*)myMalloc(22500 * sizeof(float));
for(int x2241=0; x2241 < 22500; x2241++) {
float x2242 = x2217[x2241];
float x2243 = x2231[x2241];
float x2244 = x2242 / x2243;
x2240[x2241] = x2244;

}
for(int x2248=0; x2248 < 22500; x2248++) {
float x2249 = x96[x2248];
float x2250 = x2240[x2248];
float x2251 = x2249 - x2250;
x96[x2248] = x2251;

}
for(int x2255=0; x2255 < 22500; x2255++) {
float x2256 = x184[x2255];
x184[x2255] = 0.0f;

}
for(int x2260=0; x2260 < 45000; x2260++) {
float x2261 = x189[x2260];
bool x2262 = x2261 > 5.0f;
if (x2262) {
x189[x2260] = 5.0f;
} else {
}
float x2266 = x189[x2260];
bool x2267 = x2266 < -5.0f;
if (x2267) {
x189[x2260] = -5.0f;
} else {
}

}
float* x2273 = (float*)myMalloc(45000 * sizeof(float));
for(int x2274=0; x2274 < 45000; x2274++) {
float x2275 = x189[x2274];
float x2276 = x189[x2274];
float x2277 = x2275 * x2276;
x2273[x2274] = x2277;

}
for(int x2281=0; x2281 < 45000; x2281++) {
float x2282 = x269[x2281];
float x2283 = x2273[x2281];
float x2284 = x2282 + x2283;
x269[x2281] = x2284;

}
float* x2288 = (float*)myMalloc(45000 * sizeof(float));
for(int x2289=0; x2289 < 45000; x2289++) {
float x2290 = x189[x2289];
float x2291 = x2290 * 0.1f;
x2288[x2289] = x2291;

}
float* x2295 = (float*)myMalloc(45000 * sizeof(float));
for(int x2296=0; x2296 < 45000; x2296++) {
float x2297 = x269[x2296];
float x2298 = x2297 + 1.0E-8f;
x2295[x2296] = x2298;

}
float* x2302 = (float*)myMalloc(45000 * sizeof(float));
for(int x2303=0; x2303 < 45000; x2303++) {
float x2304 = x2295[x2303];
double x2305 = (double)x2304;
double x2306 = sqrt(x2305);
float x2307 = (float)x2306;
x2302[x2303] = x2307;

}
float* x2311 = (float*)myMalloc(45000 * sizeof(float));
for(int x2312=0; x2312 < 45000; x2312++) {
float x2313 = x2288[x2312];
float x2314 = x2302[x2312];
float x2315 = x2313 / x2314;
x2311[x2312] = x2315;

}
for(int x2319=0; x2319 < 45000; x2319++) {
float x2320 = x103[x2319];
float x2321 = x2311[x2319];
float x2322 = x2320 - x2321;
x103[x2319] = x2322;

}
for(int x2326=0; x2326 < 45000; x2326++) {
float x2327 = x189[x2326];
x189[x2326] = 0.0f;

}
for(int x2331=0; x2331 < 150; x2331++) {
float x2332 = x194[x2331];
bool x2333 = x2332 > 5.0f;
if (x2333) {
x194[x2331] = 5.0f;
} else {
}
float x2337 = x194[x2331];
bool x2338 = x2337 < -5.0f;
if (x2338) {
x194[x2331] = -5.0f;
} else {
}

}
float* x2344 = (float*)myMalloc(150 * sizeof(float));
for(int x2345=0; x2345 < 150; x2345++) {
float x2346 = x194[x2345];
float x2347 = x194[x2345];
float x2348 = x2346 * x2347;
x2344[x2345] = x2348;

}
for(int x2352=0; x2352 < 150; x2352++) {
float x2353 = x274[x2352];
float x2354 = x2344[x2352];
float x2355 = x2353 + x2354;
x274[x2352] = x2355;

}
float* x2359 = (float*)myMalloc(150 * sizeof(float));
for(int x2360=0; x2360 < 150; x2360++) {
float x2361 = x194[x2360];
float x2362 = x2361 * 0.1f;
x2359[x2360] = x2362;

}
float* x2366 = (float*)myMalloc(150 * sizeof(float));
for(int x2367=0; x2367 < 150; x2367++) {
float x2368 = x274[x2367];
float x2369 = x2368 + 1.0E-8f;
x2366[x2367] = x2369;

}
float* x2373 = (float*)myMalloc(150 * sizeof(float));
for(int x2374=0; x2374 < 150; x2374++) {
float x2375 = x2366[x2374];
double x2376 = (double)x2375;
double x2377 = sqrt(x2376);
float x2378 = (float)x2377;
x2373[x2374] = x2378;

}
float* x2382 = (float*)myMalloc(150 * sizeof(float));
for(int x2383=0; x2383 < 150; x2383++) {
float x2384 = x2359[x2383];
float x2385 = x2373[x2383];
float x2386 = x2384 / x2385;
x2382[x2383] = x2386;

}
for(int x2390=0; x2390 < 150; x2390++) {
float x2391 = x110[x2390];
float x2392 = x2382[x2390];
float x2393 = x2391 - x2392;
x110[x2390] = x2393;

}
for(int x2397=0; x2397 < 150; x2397++) {
float x2398 = x194[x2397];
x194[x2397] = 0.0f;

}
for(int x2402=0; x2402 < 750; x2402++) {
float x2403 = x199[x2402];
bool x2404 = x2403 > 5.0f;
if (x2404) {
x199[x2402] = 5.0f;
} else {
}
float x2408 = x199[x2402];
bool x2409 = x2408 < -5.0f;
if (x2409) {
x199[x2402] = -5.0f;
} else {
}

}
float* x2415 = (float*)myMalloc(750 * sizeof(float));
for(int x2416=0; x2416 < 750; x2416++) {
float x2417 = x199[x2416];
float x2418 = x199[x2416];
float x2419 = x2417 * x2418;
x2415[x2416] = x2419;

}
for(int x2423=0; x2423 < 750; x2423++) {
float x2424 = x279[x2423];
float x2425 = x2415[x2423];
float x2426 = x2424 + x2425;
x279[x2423] = x2426;

}
float* x2430 = (float*)myMalloc(750 * sizeof(float));
for(int x2431=0; x2431 < 750; x2431++) {
float x2432 = x199[x2431];
float x2433 = x2432 * 0.1f;
x2430[x2431] = x2433;

}
float* x2437 = (float*)myMalloc(750 * sizeof(float));
for(int x2438=0; x2438 < 750; x2438++) {
float x2439 = x279[x2438];
float x2440 = x2439 + 1.0E-8f;
x2437[x2438] = x2440;

}
float* x2444 = (float*)myMalloc(750 * sizeof(float));
for(int x2445=0; x2445 < 750; x2445++) {
float x2446 = x2437[x2445];
double x2447 = (double)x2446;
double x2448 = sqrt(x2447);
float x2449 = (float)x2448;
x2444[x2445] = x2449;

}
float* x2453 = (float*)myMalloc(750 * sizeof(float));
for(int x2454=0; x2454 < 750; x2454++) {
float x2455 = x2430[x2454];
float x2456 = x2444[x2454];
float x2457 = x2455 / x2456;
x2453[x2454] = x2457;

}
for(int x2461=0; x2461 < 750; x2461++) {
float x2462 = x115[x2461];
float x2463 = x2453[x2461];
float x2464 = x2462 - x2463;
x115[x2461] = x2464;

}
for(int x2468=0; x2468 < 750; x2468++) {
float x2469 = x199[x2468];
x199[x2468] = 0.0f;

}
for(int x2473=0; x2473 < 5; x2473++) {
float x2474 = x204[x2473];
bool x2475 = x2474 > 5.0f;
if (x2475) {
x204[x2473] = 5.0f;
} else {
}
float x2479 = x204[x2473];
bool x2480 = x2479 < -5.0f;
if (x2480) {
x204[x2473] = -5.0f;
} else {
}

}
float* x2486 = (float*)myMalloc(5 * sizeof(float));
for(int x2487=0; x2487 < 5; x2487++) {
float x2488 = x204[x2487];
float x2489 = x204[x2487];
float x2490 = x2488 * x2489;
x2486[x2487] = x2490;

}
for(int x2494=0; x2494 < 5; x2494++) {
float x2495 = x284[x2494];
float x2496 = x2486[x2494];
float x2497 = x2495 + x2496;
x284[x2494] = x2497;

}
float* x2501 = (float*)myMalloc(5 * sizeof(float));
for(int x2502=0; x2502 < 5; x2502++) {
float x2503 = x204[x2502];
float x2504 = x2503 * 0.1f;
x2501[x2502] = x2504;

}
float* x2508 = (float*)myMalloc(5 * sizeof(float));
for(int x2509=0; x2509 < 5; x2509++) {
float x2510 = x284[x2509];
float x2511 = x2510 + 1.0E-8f;
x2508[x2509] = x2511;

}
float* x2515 = (float*)myMalloc(5 * sizeof(float));
for(int x2516=0; x2516 < 5; x2516++) {
float x2517 = x2508[x2516];
double x2518 = (double)x2517;
double x2519 = sqrt(x2518);
float x2520 = (float)x2519;
x2515[x2516] = x2520;

}
float* x2524 = (float*)myMalloc(5 * sizeof(float));
for(int x2525=0; x2525 < 5; x2525++) {
float x2526 = x2501[x2525];
float x2527 = x2515[x2525];
float x2528 = x2526 / x2527;
x2524[x2525] = x2528;

}
for(int x2532=0; x2532 < 5; x2532++) {
float x2533 = x123[x2532];
float x2534 = x2524[x2532];
float x2535 = x2533 - x2534;
x123[x2532] = x2535;

}
for(int x2539=0; x2539 < 5; x2539++) {
float x2540 = x204[x2539];
x204[x2539] = 0.0f;

}
for(int x2544=0; x2544 < 150; x2544++) {
float x2545 = x209[x2544];
x209[x2544] = 0.0f;

}
for(int x2549=0; x2549 < 150; x2549++) {
float x2550 = x214[x2549];
x214[x2549] = 0.0f;

}
mallocAddr = (void*)x289;

}
}
/*****************************************
  End of C Generated Code                  
*******************************************/

