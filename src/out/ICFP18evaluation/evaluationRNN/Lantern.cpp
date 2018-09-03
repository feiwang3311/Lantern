
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
double x1 = ((double)clock() / CLOCKS_PER_SEC);
int32_t x2 = open("graham.txt",0);
int32_t x3 = fsize(x2);
printf("data has %d chars\n",x3);
int32_t* x6 = (int32_t*)myMalloc(x3 * sizeof(int32_t));
char* x4 = (char *)mmap(0, x3, PROT_READ, MAP_FILE | MAP_SHARED, x2, 0);
for(int x8=0; x8 < x3; x8++) {
char x9 = x4[x8];
int32_t x10 = (int32_t ) x9;
int32_t x11 = x10 - 96;
x6[x8] = x11;

}
float* x15 = (float*)myMalloc(1300 * sizeof(float));
for(int x17=0; x17 < 1300; x17++) {
float x18 = d(gen);
float x19 = x18 * 0.01f;
x15[x17] = x19;

}
float* x23 = (float*)myMalloc(2500 * sizeof(float));
for(int x25=0; x25 < 2500; x25++) {
float x26 = d(gen);
float x27 = x26 * 0.01f;
x23[x25] = x27;

}
float* x31 = (float*)myMalloc(1300 * sizeof(float));
for(int x32=0; x32 < 1300; x32++) {
float x33 = d(gen);
float x34 = x33 * 0.01f;
x31[x32] = x34;

}
float* x38 = (float*)myMalloc(50 * sizeof(float));
for(int x40=0; x40 < 50; x40++) {
x38[x40] = 0.0f;

}
float* x44 = (float*)myMalloc(26 * sizeof(float));
for(int x46=0; x46 < 26; x46++) {
x44[x46] = 0.0f;

}
float* x50 = (float*)myMalloc(50 * sizeof(float));
for(int x51=0; x51 < 50; x51++) {
x50[x51] = 0.0f;

}
float* x55 = (float*)myMalloc(50 * sizeof(float));
for(int x56=0; x56 < 50; x56++) {
x55[x56] = 0.0f;

}
float* x60 = (float*)myMalloc(1300 * sizeof(float));
for(int x61=0; x61 < 1300; x61++) {
x60[x61] = 0.0f;

}
float* x65 = (float*)myMalloc(2500 * sizeof(float));
for(int x66=0; x66 < 2500; x66++) {
x65[x66] = 0.0f;

}
float* x70 = (float*)myMalloc(1300 * sizeof(float));
for(int x71=0; x71 < 1300; x71++) {
x70[x71] = 0.0f;

}
float* x75 = (float*)myMalloc(50 * sizeof(float));
for(int x76=0; x76 < 50; x76++) {
x75[x76] = 0.0f;

}
float* x80 = (float*)myMalloc(26 * sizeof(float));
for(int x81=0; x81 < 26; x81++) {
x80[x81] = 0.0f;

}
float* x85 = (float*)myMalloc(50 * sizeof(float));
for(int x86=0; x86 < 50; x86++) {
x85[x86] = 0.0f;

}
float* x90 = (float*)myMalloc(1300 * sizeof(float));
for(int x91=0; x91 < 1300; x91++) {
x90[x91] = 0.0f;

}
float* x95 = (float*)myMalloc(2500 * sizeof(float));
for(int x96=0; x96 < 2500; x96++) {
x95[x96] = 0.0f;

}
float* x100 = (float*)myMalloc(1300 * sizeof(float));
for(int x101=0; x101 < 1300; x101++) {
x100[x101] = 0.0f;

}
float* x105 = (float*)myMalloc(50 * sizeof(float));
for(int x106=0; x106 < 50; x106++) {
x105[x106] = 0.0f;

}
float* x110 = (float*)myMalloc(26 * sizeof(float));
for(int x111=0; x111 < 26; x111++) {
x110[x111] = 0.0f;

}
double* x115 = (double*)myMalloc(51 * sizeof(double));
double x116 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x117 = (long)mallocAddr;
int32_t x118 = 0;
x118 -= 20;
float x120 = 60.0f;
for(int x122=0; x122 < 5001; x122++) {
float* x160 = (float*)myMalloc(1 * sizeof(float));
int32_t* x136 = (int32_t*)myMalloc(20 * sizeof(int32_t));
int32_t* x137 = (int32_t*)myMalloc(20 * sizeof(int32_t));
function<void(int32_t,float**)> x175 = [&](int32_t x176,float** x177) {
float** x179 = x177;
float* x180 = x179[0];
float* x181 = x179[1];
float* x182 = x179[2];
float* x183 = x179[3];
int32_t x178 = x176;
bool x184 = x178 < 20;
if (x184) {
float* x185 = (float*)myMalloc(26 * sizeof(float));
for(int x186=0; x186 < 26; x186++) {
x185[x186] = 0.0f;

}
int32_t x190 = x136[x178];
x185[x190] = 1.0f;
float* x192 = (float*)myMalloc(26 * sizeof(float));
for(int x193=0; x193 < 26; x193++) {
x192[x193] = 0.0f;

}
float* x197 = (float*)myMalloc(26 * sizeof(float));
for(int x198=0; x198 < 26; x198++) {
x197[x198] = 0.0f;

}
int32_t x202 = x137[x178];
x197[x202] = 1.0f;
float* x204 = (float*)myMalloc(26 * sizeof(float));
for(int x205=0; x205 < 26; x205++) {
x204[x205] = 0.0f;

}
// dot WrappedArray(50, 26) - WrappedArray(26)
int32_t x210 = 0;
float* x211 = (float*)myMalloc(50 * sizeof(float));
for(int x212=0; x212 < 50; x212++) {
float x213 = 0.0f;
for(int x214=0; x214 < 26; x214++) {
int32_t x215 = x210;
float x216 = x15[x215];
float x217 = x185[x214];
float x218 = x216 * x217;
x213 += x218;
x210 += 1;

}
float x223 = x213;
x211[x212] = x223;

}
float* x227 = (float*)myMalloc(50 * sizeof(float));
for(int x228=0; x228 < 50; x228++) {
x227[x228] = 0.0f;

}
// dot WrappedArray(50, 50) - WrappedArray(50)
int32_t x233 = 0;
float* x234 = (float*)myMalloc(50 * sizeof(float));
for(int x235=0; x235 < 50; x235++) {
float x236 = 0.0f;
for(int x237=0; x237 < 50; x237++) {
int32_t x238 = x233;
float x239 = x23[x238];
float x240 = x182[x237];
float x241 = x239 * x240;
x236 += x241;
x233 += 1;

}
float x246 = x236;
x234[x235] = x246;

}
float* x250 = (float*)myMalloc(50 * sizeof(float));
for(int x251=0; x251 < 50; x251++) {
x250[x251] = 0.0f;

}
float* x255 = (float*)myMalloc(50 * sizeof(float));
for(int x256=0; x256 < 50; x256++) {
float x257 = x211[x256];
float x258 = x234[x256];
float x259 = x257 + x258;
x255[x256] = x259;

}
float* x263 = (float*)myMalloc(50 * sizeof(float));
for(int x264=0; x264 < 50; x264++) {
x263[x264] = 0.0f;

}
float* x268 = (float*)myMalloc(50 * sizeof(float));
for(int x269=0; x269 < 50; x269++) {
float x270 = x255[x269];
float x271 = x38[x269];
float x272 = x270 + x271;
x268[x269] = x272;

}
float* x276 = (float*)myMalloc(50 * sizeof(float));
for(int x277=0; x277 < 50; x277++) {
x276[x277] = 0.0f;

}
float* x281 = (float*)myMalloc(50 * sizeof(float));
for(int x282=0; x282 < 50; x282++) {
float x283 = x268[x282];
double x284 = (double)x283;
double x285 = tanh(x284);
float x286 = (float)x285;
x281[x282] = x286;

}
float* x290 = (float*)myMalloc(50 * sizeof(float));
for(int x291=0; x291 < 50; x291++) {
x290[x291] = 0.0f;

}
// dot WrappedArray(26, 50) - WrappedArray(50)
int32_t x296 = 0;
float* x297 = (float*)myMalloc(26 * sizeof(float));
for(int x298=0; x298 < 26; x298++) {
float x299 = 0.0f;
for(int x300=0; x300 < 50; x300++) {
int32_t x301 = x296;
float x302 = x31[x301];
float x303 = x281[x300];
float x304 = x302 * x303;
x299 += x304;
x296 += 1;

}
float x309 = x299;
x297[x298] = x309;

}
float* x313 = (float*)myMalloc(26 * sizeof(float));
for(int x314=0; x314 < 26; x314++) {
x313[x314] = 0.0f;

}
float* x318 = (float*)myMalloc(26 * sizeof(float));
for(int x319=0; x319 < 26; x319++) {
float x320 = x297[x319];
float x321 = x44[x319];
float x322 = x320 + x321;
x318[x319] = x322;

}
float* x326 = (float*)myMalloc(26 * sizeof(float));
for(int x327=0; x327 < 26; x327++) {
x326[x327] = 0.0f;

}
float* x331 = (float*)myMalloc(26 * sizeof(float));
for(int x332=0; x332 < 26; x332++) {
float x333 = x318[x332];
double x334 = (double)x333;
double x335 = exp(x334);
float x336 = (float)x335;
x331[x332] = x336;

}
float* x340 = (float*)myMalloc(26 * sizeof(float));
for(int x341=0; x341 < 26; x341++) {
x340[x341] = 0.0f;

}
float x345 = 0.0f;
for(int x346=0; x346 < 26; x346++) {
float x347 = x345;
float x348 = x331[x346];
float x349 = x347 + x348;
x345 = x349;

}
float x353 = x345;
float* x354 = (float*)myMalloc(1 * sizeof(float));
x354[0] = x353;
float* x356 = (float*)myMalloc(1 * sizeof(float));
for(int x357=0; x357 < 1; x357++) {
x356[x357] = 0.0f;

}
float x361 = x354[0];
float* x362 = (float*)myMalloc(26 * sizeof(float));
for(int x363=0; x363 < 26; x363++) {
float x364 = x331[x363];
float x365 = x364 / x361;
x362[x363] = x365;

}
float* x369 = (float*)myMalloc(26 * sizeof(float));
for(int x370=0; x370 < 26; x370++) {
x369[x370] = 0.0f;

}
// dot WrappedArray(26) - WrappedArray(26)
int32_t x375 = 0;
float* x376 = (float*)myMalloc(1 * sizeof(float));
float x377 = 0.0f;
for(int x378=0; x378 < 26; x378++) {
int32_t x379 = x375;
float x380 = x362[x379];
float x381 = x197[x378];
float x382 = x380 * x381;
x377 += x382;
x375 += 1;

}
float x387 = x377;
x376[0] = x387;
float* x389 = (float*)myMalloc(1 * sizeof(float));
for(int x390=0; x390 < 1; x390++) {
x389[x390] = 0.0f;

}
float* x394 = (float*)myMalloc(1 * sizeof(float));
float x395 = x376[0];
double x396 = (double)x395;
double x397 = log(x396);
float x398 = (float)x397;
x394[0] = x398;
float* x400 = (float*)myMalloc(1 * sizeof(float));
for(int x401=0; x401 < 1; x401++) {
x400[x401] = 0.0f;

}
float* x405 = (float*)myMalloc(1 * sizeof(float));
float x406 = x394[0];
float x407 = x180[0];
float x408 = x407 - x406;
x405[0] = x408;
float* x410 = (float*)myMalloc(1 * sizeof(float));
for(int x411=0; x411 < 1; x411++) {
x410[x411] = 0.0f;

}
float** x416 = (float**)myMalloc(4 * sizeof(float*));
x416[0] = x405;
x416[1] = x410;
x416[2] = x281;
x416[3] = x290;
int32_t x513 = 0;
int32_t x529 = 0;
int32_t x588 = 0;
int32_t x604 = 0;
int32_t x621 = 0;
int32_t x637 = 0;
int32_t x415 = x178 + 1;
x175(x415,x416);
// += tensor of dim 0
float x424 = x410[0];
float x425 = x181[0];
float x426 = x425 + x424;
x181[0] = x426;
float x428 = x410[0];
float x429 = x400[0];
float x430 = x429 - x428;
x400[0] = x430;
float x432 = x389[0];
float x433 = x400[0];
float x434 = x376[0];
float x435 = x433 / x434;
float x436 = x432 + x435;
x389[0] = x436;
float x438 = x389[0];
// Generate code for addMul
for(int x440=0; x440 < 26; x440++) {
float x441 = x369[x440];
float x442 = x197[x440];
float x443 = x438 * x442;
float x444 = x441 + x443;
x369[x440] = x444;

}
float x448 = x389[0];
// Generate code for addMul
for(int x450=0; x450 < 26; x450++) {
float x451 = x204[x450];
float x452 = x362[x450];
float x453 = x448 * x452;
float x454 = x451 + x453;
x204[x450] = x454;

}
for(int x458=0; x458 < 26; x458++) {
float x459 = x340[x458];
float x460 = x369[x458];
float x461 = x354[0];
float x462 = x460 / x461;
float x463 = x459 + x462;
x340[x458] = x463;

}
for(int x467=0; x467 < 26; x467++) {
float x468 = x356[0];
float x469 = x331[x467];
float x470 = x369[x467];
float x472 = x354[0];
float x471 = x469 * x470;
float x473 = x472 * x472;
float x474 = x471 / x473;
float x475 = x468 - x474;
x356[0] = x475;

}
// += tensor of dim 0
float x480 = x356[0];
for(int x481=0; x481 < 26; x481++) {
float x482 = x340[x481];
float x483 = x482 + x480;
x340[x481] = x483;

}
// backpropage exp
for(int x488=0; x488 < 26; x488++) {
float x489 = x326[x488];
float x490 = x331[x488];
float x491 = x340[x488];
float x492 = x490 * x491;
float x493 = x489 + x492;
x326[x488] = x493;

}
// backpropagate +
for(int x498=0; x498 < 26; x498++) {
float x499 = x313[x498];
float x500 = x326[x498];
float x501 = x499 + x500;
x313[x498] = x501;

}
for(int x505=0; x505 < 26; x505++) {
float x506 = x80[x505];
float x507 = x326[x505];
float x508 = x506 + x507;
x80[x505] = x508;

}
// add_cartesian
for(int x514=0; x514 < 26; x514++) {
for(int x515=0; x515 < 50; x515++) {
int32_t x516 = x513;
int32_t x517 = x516 + x515;
float x518 = x70[x517];
float x519 = x281[x515];
float x520 = x313[x514];
float x521 = x519 * x520;
float x522 = x518 + x521;
x70[x517] = x522;

}
x513 += 50;

}
for(int x530=0; x530 < 26; x530++) {
for(int x531=0; x531 < 50; x531++) {
float x532 = x290[x531];
int32_t x533 = x529;
int32_t x534 = x533 + x531;
float x535 = x31[x534];
float x536 = x313[x530];
float x537 = x535 * x536;
float x538 = x532 + x537;
x290[x531] = x538;

}
x529 += 50;

}
// backpropagate tanh
for(int x546=0; x546 < 50; x546++) {
float x547 = x276[x546];
float x548 = x281[x546];
float x551 = x290[x546];
float x549 = x548 * x548;
float x550 = 1.0f - x549;
float x552 = x550 * x551;
float x553 = x547 + x552;
x276[x546] = x553;

}
// backpropagate +
for(int x558=0; x558 < 50; x558++) {
float x559 = x263[x558];
float x560 = x276[x558];
float x561 = x559 + x560;
x263[x558] = x561;

}
for(int x565=0; x565 < 50; x565++) {
float x566 = x75[x565];
float x567 = x276[x565];
float x568 = x566 + x567;
x75[x565] = x568;

}
// backpropagate +
for(int x573=0; x573 < 50; x573++) {
float x574 = x227[x573];
float x575 = x263[x573];
float x576 = x574 + x575;
x227[x573] = x576;

}
for(int x580=0; x580 < 50; x580++) {
float x581 = x250[x580];
float x582 = x263[x580];
float x583 = x581 + x582;
x250[x580] = x583;

}
// add_cartesian
for(int x589=0; x589 < 50; x589++) {
for(int x590=0; x590 < 50; x590++) {
int32_t x591 = x588;
int32_t x592 = x591 + x590;
float x593 = x65[x592];
float x594 = x182[x590];
float x595 = x250[x589];
float x596 = x594 * x595;
float x597 = x593 + x596;
x65[x592] = x597;

}
x588 += 50;

}
for(int x605=0; x605 < 50; x605++) {
for(int x606=0; x606 < 50; x606++) {
float x607 = x183[x606];
int32_t x608 = x604;
int32_t x609 = x608 + x606;
float x610 = x23[x609];
float x611 = x250[x605];
float x612 = x610 * x611;
float x613 = x607 + x612;
x183[x606] = x613;

}
x604 += 50;

}
// add_cartesian
for(int x622=0; x622 < 50; x622++) {
for(int x623=0; x623 < 26; x623++) {
int32_t x624 = x621;
int32_t x625 = x624 + x623;
float x626 = x60[x625];
float x627 = x185[x623];
float x628 = x227[x622];
float x629 = x627 * x628;
float x630 = x626 + x629;
x60[x625] = x630;

}
x621 += 26;

}
for(int x638=0; x638 < 50; x638++) {
for(int x639=0; x639 < 26; x639++) {
float x640 = x192[x639];
int32_t x641 = x637;
int32_t x642 = x641 + x639;
float x643 = x15[x642];
float x644 = x227[x638];
float x645 = x643 * x644;
float x646 = x640 + x645;
x192[x639] = x646;

}
x637 += 26;

}
} else {
for(int x654=0; x654 < 50; x654++) {
float x655 = x182[x654];
x55[x654] = x655;

}
float x659 = x181[0];
x181[0] = 1.0f;
float x661 = x180[0];
x160[0] = x661;
}
};
x118 += 20;
int32_t x124 = x118;
int32_t x125 = x124 + 20;
int32_t x126 = x125 + 1;
bool x127 = x126 >= x3;
if (x127) {
x118 = 0;
for(int x129=0; x129 < 50; x129++) {
float x130 = x50[x129];
x50[x129] = 0.0f;

}
} else {
}
for(int x139=0; x139 < 20; x139++) {
int32_t x140 = x118;
int32_t x141 = x140 + x139;
int32_t x142 = x6[x141];
x136[x139] = x142;
int32_t x144 = x141 + 1;
int32_t x145 = x6[x144];
x137[x139] = x145;

}
float* x149 = (float*)myMalloc(1 * sizeof(float));
for(int x151=0; x151 < 1; x151++) {
x149[x151] = 0.0f;

}
float* x155 = (float*)myMalloc(1 * sizeof(float));
for(int x156=0; x156 < 1; x156++) {
x155[x156] = 0.0f;

}
for(int x161=0; x161 < 1; x161++) {
x160[x161] = 0.0f;

}
float* x165 = (float*)myMalloc(1 * sizeof(float));
for(int x166=0; x166 < 1; x166++) {
x165[x166] = 0.0f;

}
float* x170 = (float*)myMalloc(1 * sizeof(float));
for(int x171=0; x171 < 1; x171++) {
x170[x171] = 0.0f;

}
float** x666 = (float**)myMalloc(4 * sizeof(float*));
x666[0] = x165;
x666[1] = x170;
x666[2] = x50;
x666[3] = x85;
x175(0,x666);
float x673 = x160[0];
float x674 = x120;
float x675 = x674 * 0.9f;
float x676 = x673 * 0.1f;
float x677 = x675 + x676;
x120 = x677;
int32_t x679 = x122 % 100;
bool x680 = x679 == 0;
if (x680) {
float x681 = x120;
printf("iter %d, loss %f\n",x122,x681);
int32_t x683 = x122 / 100;
double x684 = (double)x681;
x115[x683] = x684;
} else {
}
for(int x688=0; x688 < 1300; x688++) {
float x689 = x60[x688];
bool x690 = x689 > 5.0f;
if (x690) {
x60[x688] = 5.0f;
} else {
}
float x694 = x60[x688];
bool x695 = x694 < -5.0f;
if (x695) {
x60[x688] = -5.0f;
} else {
}

}
float* x701 = (float*)myMalloc(1300 * sizeof(float));
for(int x702=0; x702 < 1300; x702++) {
float x703 = x60[x702];
float x704 = x60[x702];
float x705 = x703 * x704;
x701[x702] = x705;

}
for(int x709=0; x709 < 1300; x709++) {
float x710 = x90[x709];
float x711 = x701[x709];
float x712 = x710 + x711;
x90[x709] = x712;

}
float* x716 = (float*)myMalloc(1300 * sizeof(float));
for(int x717=0; x717 < 1300; x717++) {
float x718 = x60[x717];
float x719 = x718 * 0.1f;
x716[x717] = x719;

}
float* x723 = (float*)myMalloc(1300 * sizeof(float));
for(int x724=0; x724 < 1300; x724++) {
float x725 = x90[x724];
float x726 = x725 + 1.0E-8f;
x723[x724] = x726;

}
float* x730 = (float*)myMalloc(1300 * sizeof(float));
for(int x731=0; x731 < 1300; x731++) {
float x732 = x723[x731];
double x733 = (double)x732;
double x734 = sqrt(x733);
float x735 = (float)x734;
x730[x731] = x735;

}
float* x739 = (float*)myMalloc(1300 * sizeof(float));
for(int x740=0; x740 < 1300; x740++) {
float x741 = x716[x740];
float x742 = x730[x740];
float x743 = x741 / x742;
x739[x740] = x743;

}
for(int x747=0; x747 < 1300; x747++) {
float x748 = x15[x747];
float x749 = x739[x747];
float x750 = x748 - x749;
x15[x747] = x750;

}
for(int x754=0; x754 < 1300; x754++) {
float x755 = x60[x754];
x60[x754] = 0.0f;

}
for(int x759=0; x759 < 2500; x759++) {
float x760 = x65[x759];
bool x761 = x760 > 5.0f;
if (x761) {
x65[x759] = 5.0f;
} else {
}
float x765 = x65[x759];
bool x766 = x765 < -5.0f;
if (x766) {
x65[x759] = -5.0f;
} else {
}

}
float* x772 = (float*)myMalloc(2500 * sizeof(float));
for(int x773=0; x773 < 2500; x773++) {
float x774 = x65[x773];
float x775 = x65[x773];
float x776 = x774 * x775;
x772[x773] = x776;

}
for(int x780=0; x780 < 2500; x780++) {
float x781 = x95[x780];
float x782 = x772[x780];
float x783 = x781 + x782;
x95[x780] = x783;

}
float* x787 = (float*)myMalloc(2500 * sizeof(float));
for(int x788=0; x788 < 2500; x788++) {
float x789 = x65[x788];
float x790 = x789 * 0.1f;
x787[x788] = x790;

}
float* x794 = (float*)myMalloc(2500 * sizeof(float));
for(int x795=0; x795 < 2500; x795++) {
float x796 = x95[x795];
float x797 = x796 + 1.0E-8f;
x794[x795] = x797;

}
float* x801 = (float*)myMalloc(2500 * sizeof(float));
for(int x802=0; x802 < 2500; x802++) {
float x803 = x794[x802];
double x804 = (double)x803;
double x805 = sqrt(x804);
float x806 = (float)x805;
x801[x802] = x806;

}
float* x810 = (float*)myMalloc(2500 * sizeof(float));
for(int x811=0; x811 < 2500; x811++) {
float x812 = x787[x811];
float x813 = x801[x811];
float x814 = x812 / x813;
x810[x811] = x814;

}
for(int x818=0; x818 < 2500; x818++) {
float x819 = x23[x818];
float x820 = x810[x818];
float x821 = x819 - x820;
x23[x818] = x821;

}
for(int x825=0; x825 < 2500; x825++) {
float x826 = x65[x825];
x65[x825] = 0.0f;

}
for(int x830=0; x830 < 1300; x830++) {
float x831 = x70[x830];
bool x832 = x831 > 5.0f;
if (x832) {
x70[x830] = 5.0f;
} else {
}
float x836 = x70[x830];
bool x837 = x836 < -5.0f;
if (x837) {
x70[x830] = -5.0f;
} else {
}

}
float* x843 = (float*)myMalloc(1300 * sizeof(float));
for(int x844=0; x844 < 1300; x844++) {
float x845 = x70[x844];
float x846 = x70[x844];
float x847 = x845 * x846;
x843[x844] = x847;

}
for(int x851=0; x851 < 1300; x851++) {
float x852 = x100[x851];
float x853 = x843[x851];
float x854 = x852 + x853;
x100[x851] = x854;

}
float* x858 = (float*)myMalloc(1300 * sizeof(float));
for(int x859=0; x859 < 1300; x859++) {
float x860 = x70[x859];
float x861 = x860 * 0.1f;
x858[x859] = x861;

}
float* x865 = (float*)myMalloc(1300 * sizeof(float));
for(int x866=0; x866 < 1300; x866++) {
float x867 = x100[x866];
float x868 = x867 + 1.0E-8f;
x865[x866] = x868;

}
float* x872 = (float*)myMalloc(1300 * sizeof(float));
for(int x873=0; x873 < 1300; x873++) {
float x874 = x865[x873];
double x875 = (double)x874;
double x876 = sqrt(x875);
float x877 = (float)x876;
x872[x873] = x877;

}
float* x881 = (float*)myMalloc(1300 * sizeof(float));
for(int x882=0; x882 < 1300; x882++) {
float x883 = x858[x882];
float x884 = x872[x882];
float x885 = x883 / x884;
x881[x882] = x885;

}
for(int x889=0; x889 < 1300; x889++) {
float x890 = x31[x889];
float x891 = x881[x889];
float x892 = x890 - x891;
x31[x889] = x892;

}
for(int x896=0; x896 < 1300; x896++) {
float x897 = x70[x896];
x70[x896] = 0.0f;

}
for(int x901=0; x901 < 50; x901++) {
float x902 = x75[x901];
bool x903 = x902 > 5.0f;
if (x903) {
x75[x901] = 5.0f;
} else {
}
float x907 = x75[x901];
bool x908 = x907 < -5.0f;
if (x908) {
x75[x901] = -5.0f;
} else {
}

}
float* x914 = (float*)myMalloc(50 * sizeof(float));
for(int x915=0; x915 < 50; x915++) {
float x916 = x75[x915];
float x917 = x75[x915];
float x918 = x916 * x917;
x914[x915] = x918;

}
for(int x922=0; x922 < 50; x922++) {
float x923 = x105[x922];
float x924 = x914[x922];
float x925 = x923 + x924;
x105[x922] = x925;

}
float* x929 = (float*)myMalloc(50 * sizeof(float));
for(int x930=0; x930 < 50; x930++) {
float x931 = x75[x930];
float x932 = x931 * 0.1f;
x929[x930] = x932;

}
float* x936 = (float*)myMalloc(50 * sizeof(float));
for(int x937=0; x937 < 50; x937++) {
float x938 = x105[x937];
float x939 = x938 + 1.0E-8f;
x936[x937] = x939;

}
float* x943 = (float*)myMalloc(50 * sizeof(float));
for(int x944=0; x944 < 50; x944++) {
float x945 = x936[x944];
double x946 = (double)x945;
double x947 = sqrt(x946);
float x948 = (float)x947;
x943[x944] = x948;

}
float* x952 = (float*)myMalloc(50 * sizeof(float));
for(int x953=0; x953 < 50; x953++) {
float x954 = x929[x953];
float x955 = x943[x953];
float x956 = x954 / x955;
x952[x953] = x956;

}
for(int x960=0; x960 < 50; x960++) {
float x961 = x38[x960];
float x962 = x952[x960];
float x963 = x961 - x962;
x38[x960] = x963;

}
for(int x967=0; x967 < 50; x967++) {
float x968 = x75[x967];
x75[x967] = 0.0f;

}
for(int x972=0; x972 < 26; x972++) {
float x973 = x80[x972];
bool x974 = x973 > 5.0f;
if (x974) {
x80[x972] = 5.0f;
} else {
}
float x978 = x80[x972];
bool x979 = x978 < -5.0f;
if (x979) {
x80[x972] = -5.0f;
} else {
}

}
float* x985 = (float*)myMalloc(26 * sizeof(float));
for(int x986=0; x986 < 26; x986++) {
float x987 = x80[x986];
float x988 = x80[x986];
float x989 = x987 * x988;
x985[x986] = x989;

}
for(int x993=0; x993 < 26; x993++) {
float x994 = x110[x993];
float x995 = x985[x993];
float x996 = x994 + x995;
x110[x993] = x996;

}
float* x1000 = (float*)myMalloc(26 * sizeof(float));
for(int x1001=0; x1001 < 26; x1001++) {
float x1002 = x80[x1001];
float x1003 = x1002 * 0.1f;
x1000[x1001] = x1003;

}
float* x1007 = (float*)myMalloc(26 * sizeof(float));
for(int x1008=0; x1008 < 26; x1008++) {
float x1009 = x110[x1008];
float x1010 = x1009 + 1.0E-8f;
x1007[x1008] = x1010;

}
float* x1014 = (float*)myMalloc(26 * sizeof(float));
for(int x1015=0; x1015 < 26; x1015++) {
float x1016 = x1007[x1015];
double x1017 = (double)x1016;
double x1018 = sqrt(x1017);
float x1019 = (float)x1018;
x1014[x1015] = x1019;

}
float* x1023 = (float*)myMalloc(26 * sizeof(float));
for(int x1024=0; x1024 < 26; x1024++) {
float x1025 = x1000[x1024];
float x1026 = x1014[x1024];
float x1027 = x1025 / x1026;
x1023[x1024] = x1027;

}
for(int x1031=0; x1031 < 26; x1031++) {
float x1032 = x44[x1031];
float x1033 = x1023[x1031];
float x1034 = x1032 - x1033;
x44[x1031] = x1034;

}
for(int x1038=0; x1038 < 26; x1038++) {
float x1039 = x80[x1038];
x80[x1038] = 0.0f;

}
for(int x1043=0; x1043 < 50; x1043++) {
float x1044 = x85[x1043];
x85[x1043] = 0.0f;

}
for(int x1048=0; x1048 < 50; x1048++) {
float x1049 = x55[x1048];
x50[x1048] = x1049;

}
mallocAddr = (void*)x117;

}
double x1056 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x1059 = (long)fopen(x0, "w");
fprintf((FILE *)x1059, "unit: %s\n", "100 iteration");
for(int x1062=0; x1062 < 51; x1062++) {
double x1063 = x115[x1062];
fprintf((FILE *)x1059, "%lf\n", x1063);

}
double x1057 = x116 - x1;
double x1058 = x1056 - x116;
fprintf((FILE *)x1059, "run time: %lf %lf\n", x1057, x1058);
fclose((FILE*)x1059);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

