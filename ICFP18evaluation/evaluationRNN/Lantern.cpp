
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
        mallocAddr += bytes;
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
for(int x377=0; x377 < 1; x377++) {
float x378 = 0.0f;
for(int x379=0; x379 < 26; x379++) {
int32_t x380 = x375;
float x381 = x362[x380];
float x382 = x197[x379];
float x383 = x381 * x382;
x378 += x383;
x375 += 1;

}
float x388 = x378;
x376[x377] = x388;

}
float* x392 = (float*)myMalloc(1 * sizeof(float));
for(int x393=0; x393 < 1; x393++) {
x392[x393] = 0.0f;

}
float* x397 = (float*)myMalloc(1 * sizeof(float));
for(int x398=0; x398 < 1; x398++) {
float x399 = x376[x398];
double x400 = (double)x399;
double x401 = log(x400);
float x402 = (float)x401;
x397[x398] = x402;

}
float* x406 = (float*)myMalloc(1 * sizeof(float));
for(int x407=0; x407 < 1; x407++) {
x406[x407] = 0.0f;

}
float* x411 = (float*)myMalloc(1 * sizeof(float));
for(int x412=0; x412 < 1; x412++) {
float x413 = x397[x412];
float x414 = x180[0];
float x415 = x414 - x413;
x411[x412] = x415;

}
float* x419 = (float*)myMalloc(1 * sizeof(float));
for(int x420=0; x420 < 1; x420++) {
x419[x420] = 0.0f;

}
float** x425 = (float**)myMalloc(4 * sizeof(float*));
x425[0] = x411;
x425[1] = x419;
x425[2] = x281;
x425[3] = x290;
int32_t x531 = 0;
int32_t x547 = 0;
int32_t x606 = 0;
int32_t x622 = 0;
int32_t x639 = 0;
int32_t x655 = 0;
int32_t x424 = x178 + 1;
x175(x424,x425);
// += tensor of dim 0
float x433 = x419[0];
for(int x434=0; x434 < 1; x434++) {
float x435 = x181[x434];
float x436 = x435 + x433;
x181[x434] = x436;

}
float x440 = x419[0];
for(int x441=0; x441 < 1; x441++) {
float x442 = x406[x441];
float x443 = x442 - x440;
x406[x441] = x443;

}
for(int x447=0; x447 < 1; x447++) {
float x448 = x392[0];
float x449 = x406[0];
float x450 = x376[0];
float x451 = x449 / x450;
float x452 = x448 + x451;
x392[0] = x452;

}
float x456 = x392[0];
// Generate code for addMul
for(int x458=0; x458 < 26; x458++) {
float x459 = x369[x458];
float x460 = x197[x458];
float x461 = x456 * x460;
float x462 = x459 + x461;
x369[x458] = x462;

}
float x466 = x392[0];
// Generate code for addMul
for(int x468=0; x468 < 26; x468++) {
float x469 = x204[x468];
float x470 = x362[x468];
float x471 = x466 * x470;
float x472 = x469 + x471;
x204[x468] = x472;

}
for(int x476=0; x476 < 26; x476++) {
float x477 = x340[x476];
float x478 = x369[x476];
float x479 = x354[0];
float x480 = x478 / x479;
float x481 = x477 + x480;
x340[x476] = x481;

}
for(int x485=0; x485 < 26; x485++) {
float x486 = x356[0];
float x487 = x331[x485];
float x488 = x369[x485];
float x490 = x354[0];
float x489 = x487 * x488;
float x491 = x490 * x490;
float x492 = x489 / x491;
float x493 = x486 - x492;
x356[0] = x493;

}
// += tensor of dim 0
float x498 = x356[0];
for(int x499=0; x499 < 26; x499++) {
float x500 = x340[x499];
float x501 = x500 + x498;
x340[x499] = x501;

}
// backpropage exp
for(int x506=0; x506 < 26; x506++) {
float x507 = x326[x506];
float x508 = x331[x506];
float x509 = x340[x506];
float x510 = x508 * x509;
float x511 = x507 + x510;
x326[x506] = x511;

}
// backpropagate +
for(int x516=0; x516 < 26; x516++) {
float x517 = x313[x516];
float x518 = x326[x516];
float x519 = x517 + x518;
x313[x516] = x519;

}
for(int x523=0; x523 < 26; x523++) {
float x524 = x80[x523];
float x525 = x326[x523];
float x526 = x524 + x525;
x80[x523] = x526;

}
// add_cartesian
for(int x532=0; x532 < 26; x532++) {
for(int x533=0; x533 < 50; x533++) {
int32_t x534 = x531;
int32_t x535 = x534 + x533;
float x536 = x70[x535];
float x537 = x281[x533];
float x538 = x313[x532];
float x539 = x537 * x538;
float x540 = x536 + x539;
x70[x535] = x540;

}
x531 += 50;

}
for(int x548=0; x548 < 26; x548++) {
for(int x549=0; x549 < 50; x549++) {
float x550 = x290[x549];
int32_t x551 = x547;
int32_t x552 = x551 + x549;
float x553 = x31[x552];
float x554 = x313[x548];
float x555 = x553 * x554;
float x556 = x550 + x555;
x290[x549] = x556;

}
x547 += 50;

}
// backpropagate tanh
for(int x564=0; x564 < 50; x564++) {
float x565 = x276[x564];
float x566 = x281[x564];
float x569 = x290[x564];
float x567 = x566 * x566;
float x568 = 1.0f - x567;
float x570 = x568 * x569;
float x571 = x565 + x570;
x276[x564] = x571;

}
// backpropagate +
for(int x576=0; x576 < 50; x576++) {
float x577 = x263[x576];
float x578 = x276[x576];
float x579 = x577 + x578;
x263[x576] = x579;

}
for(int x583=0; x583 < 50; x583++) {
float x584 = x75[x583];
float x585 = x276[x583];
float x586 = x584 + x585;
x75[x583] = x586;

}
// backpropagate +
for(int x591=0; x591 < 50; x591++) {
float x592 = x227[x591];
float x593 = x263[x591];
float x594 = x592 + x593;
x227[x591] = x594;

}
for(int x598=0; x598 < 50; x598++) {
float x599 = x250[x598];
float x600 = x263[x598];
float x601 = x599 + x600;
x250[x598] = x601;

}
// add_cartesian
for(int x607=0; x607 < 50; x607++) {
for(int x608=0; x608 < 50; x608++) {
int32_t x609 = x606;
int32_t x610 = x609 + x608;
float x611 = x65[x610];
float x612 = x182[x608];
float x613 = x250[x607];
float x614 = x612 * x613;
float x615 = x611 + x614;
x65[x610] = x615;

}
x606 += 50;

}
for(int x623=0; x623 < 50; x623++) {
for(int x624=0; x624 < 50; x624++) {
float x625 = x183[x624];
int32_t x626 = x622;
int32_t x627 = x626 + x624;
float x628 = x23[x627];
float x629 = x250[x623];
float x630 = x628 * x629;
float x631 = x625 + x630;
x183[x624] = x631;

}
x622 += 50;

}
// add_cartesian
for(int x640=0; x640 < 50; x640++) {
for(int x641=0; x641 < 26; x641++) {
int32_t x642 = x639;
int32_t x643 = x642 + x641;
float x644 = x60[x643];
float x645 = x185[x641];
float x646 = x227[x640];
float x647 = x645 * x646;
float x648 = x644 + x647;
x60[x643] = x648;

}
x639 += 26;

}
for(int x656=0; x656 < 50; x656++) {
for(int x657=0; x657 < 26; x657++) {
float x658 = x192[x657];
int32_t x659 = x655;
int32_t x660 = x659 + x657;
float x661 = x15[x660];
float x662 = x227[x656];
float x663 = x661 * x662;
float x664 = x658 + x663;
x192[x657] = x664;

}
x655 += 26;

}
} else {
for(int x672=0; x672 < 50; x672++) {
float x673 = x182[x672];
x55[x672] = x673;

}
for(int x677=0; x677 < 1; x677++) {
float x678 = x181[x677];
x181[x677] = 1.0f;

}
for(int x682=0; x682 < 1; x682++) {
float x683 = x180[x682];
x160[x682] = x683;

}
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
float** x690 = (float**)myMalloc(4 * sizeof(float*));
x690[0] = x165;
x690[1] = x170;
x690[2] = x50;
x690[3] = x85;
x175(0,x690);
float x697 = x160[0];
float x698 = x120;
float x699 = x698 * 0.9f;
float x700 = x697 * 0.1f;
float x701 = x699 + x700;
x120 = x701;
int32_t x703 = x122 % 100;
bool x704 = x703 == 0;
if (x704) {
float x705 = x120;
printf("iter %d, loss %f\n",x122,x705);
int32_t x707 = x122 / 100;
double x708 = (double)x705;
x115[x707] = x708;
} else {
}
for(int x712=0; x712 < 1300; x712++) {
float x713 = x60[x712];
bool x714 = x713 > 5.0f;
if (x714) {
x60[x712] = 5.0f;
} else {
}
float x718 = x60[x712];
bool x719 = x718 < -5.0f;
if (x719) {
x60[x712] = -5.0f;
} else {
}

}
float* x725 = (float*)myMalloc(1300 * sizeof(float));
for(int x726=0; x726 < 1300; x726++) {
float x727 = x60[x726];
float x728 = x60[x726];
float x729 = x727 * x728;
x725[x726] = x729;

}
for(int x733=0; x733 < 1300; x733++) {
float x734 = x90[x733];
float x735 = x725[x733];
float x736 = x734 + x735;
x90[x733] = x736;

}
float* x740 = (float*)myMalloc(1300 * sizeof(float));
for(int x741=0; x741 < 1300; x741++) {
float x742 = x60[x741];
float x743 = x742 * 0.1f;
x740[x741] = x743;

}
float* x747 = (float*)myMalloc(1300 * sizeof(float));
for(int x748=0; x748 < 1300; x748++) {
float x749 = x90[x748];
float x750 = x749 + 1.0E-8f;
x747[x748] = x750;

}
float* x754 = (float*)myMalloc(1300 * sizeof(float));
for(int x755=0; x755 < 1300; x755++) {
float x756 = x747[x755];
double x757 = (double)x756;
double x758 = sqrt(x757);
float x759 = (float)x758;
x754[x755] = x759;

}
float* x763 = (float*)myMalloc(1300 * sizeof(float));
for(int x764=0; x764 < 1300; x764++) {
float x765 = x740[x764];
float x766 = x754[x764];
float x767 = x765 / x766;
x763[x764] = x767;

}
for(int x771=0; x771 < 1300; x771++) {
float x772 = x15[x771];
float x773 = x763[x771];
float x774 = x772 - x773;
x15[x771] = x774;

}
for(int x778=0; x778 < 1300; x778++) {
float x779 = x60[x778];
x60[x778] = 0.0f;

}
for(int x783=0; x783 < 2500; x783++) {
float x784 = x65[x783];
bool x785 = x784 > 5.0f;
if (x785) {
x65[x783] = 5.0f;
} else {
}
float x789 = x65[x783];
bool x790 = x789 < -5.0f;
if (x790) {
x65[x783] = -5.0f;
} else {
}

}
float* x796 = (float*)myMalloc(2500 * sizeof(float));
for(int x797=0; x797 < 2500; x797++) {
float x798 = x65[x797];
float x799 = x65[x797];
float x800 = x798 * x799;
x796[x797] = x800;

}
for(int x804=0; x804 < 2500; x804++) {
float x805 = x95[x804];
float x806 = x796[x804];
float x807 = x805 + x806;
x95[x804] = x807;

}
float* x811 = (float*)myMalloc(2500 * sizeof(float));
for(int x812=0; x812 < 2500; x812++) {
float x813 = x65[x812];
float x814 = x813 * 0.1f;
x811[x812] = x814;

}
float* x818 = (float*)myMalloc(2500 * sizeof(float));
for(int x819=0; x819 < 2500; x819++) {
float x820 = x95[x819];
float x821 = x820 + 1.0E-8f;
x818[x819] = x821;

}
float* x825 = (float*)myMalloc(2500 * sizeof(float));
for(int x826=0; x826 < 2500; x826++) {
float x827 = x818[x826];
double x828 = (double)x827;
double x829 = sqrt(x828);
float x830 = (float)x829;
x825[x826] = x830;

}
float* x834 = (float*)myMalloc(2500 * sizeof(float));
for(int x835=0; x835 < 2500; x835++) {
float x836 = x811[x835];
float x837 = x825[x835];
float x838 = x836 / x837;
x834[x835] = x838;

}
for(int x842=0; x842 < 2500; x842++) {
float x843 = x23[x842];
float x844 = x834[x842];
float x845 = x843 - x844;
x23[x842] = x845;

}
for(int x849=0; x849 < 2500; x849++) {
float x850 = x65[x849];
x65[x849] = 0.0f;

}
for(int x854=0; x854 < 1300; x854++) {
float x855 = x70[x854];
bool x856 = x855 > 5.0f;
if (x856) {
x70[x854] = 5.0f;
} else {
}
float x860 = x70[x854];
bool x861 = x860 < -5.0f;
if (x861) {
x70[x854] = -5.0f;
} else {
}

}
float* x867 = (float*)myMalloc(1300 * sizeof(float));
for(int x868=0; x868 < 1300; x868++) {
float x869 = x70[x868];
float x870 = x70[x868];
float x871 = x869 * x870;
x867[x868] = x871;

}
for(int x875=0; x875 < 1300; x875++) {
float x876 = x100[x875];
float x877 = x867[x875];
float x878 = x876 + x877;
x100[x875] = x878;

}
float* x882 = (float*)myMalloc(1300 * sizeof(float));
for(int x883=0; x883 < 1300; x883++) {
float x884 = x70[x883];
float x885 = x884 * 0.1f;
x882[x883] = x885;

}
float* x889 = (float*)myMalloc(1300 * sizeof(float));
for(int x890=0; x890 < 1300; x890++) {
float x891 = x100[x890];
float x892 = x891 + 1.0E-8f;
x889[x890] = x892;

}
float* x896 = (float*)myMalloc(1300 * sizeof(float));
for(int x897=0; x897 < 1300; x897++) {
float x898 = x889[x897];
double x899 = (double)x898;
double x900 = sqrt(x899);
float x901 = (float)x900;
x896[x897] = x901;

}
float* x905 = (float*)myMalloc(1300 * sizeof(float));
for(int x906=0; x906 < 1300; x906++) {
float x907 = x882[x906];
float x908 = x896[x906];
float x909 = x907 / x908;
x905[x906] = x909;

}
for(int x913=0; x913 < 1300; x913++) {
float x914 = x31[x913];
float x915 = x905[x913];
float x916 = x914 - x915;
x31[x913] = x916;

}
for(int x920=0; x920 < 1300; x920++) {
float x921 = x70[x920];
x70[x920] = 0.0f;

}
for(int x925=0; x925 < 50; x925++) {
float x926 = x75[x925];
bool x927 = x926 > 5.0f;
if (x927) {
x75[x925] = 5.0f;
} else {
}
float x931 = x75[x925];
bool x932 = x931 < -5.0f;
if (x932) {
x75[x925] = -5.0f;
} else {
}

}
float* x938 = (float*)myMalloc(50 * sizeof(float));
for(int x939=0; x939 < 50; x939++) {
float x940 = x75[x939];
float x941 = x75[x939];
float x942 = x940 * x941;
x938[x939] = x942;

}
for(int x946=0; x946 < 50; x946++) {
float x947 = x105[x946];
float x948 = x938[x946];
float x949 = x947 + x948;
x105[x946] = x949;

}
float* x953 = (float*)myMalloc(50 * sizeof(float));
for(int x954=0; x954 < 50; x954++) {
float x955 = x75[x954];
float x956 = x955 * 0.1f;
x953[x954] = x956;

}
float* x960 = (float*)myMalloc(50 * sizeof(float));
for(int x961=0; x961 < 50; x961++) {
float x962 = x105[x961];
float x963 = x962 + 1.0E-8f;
x960[x961] = x963;

}
float* x967 = (float*)myMalloc(50 * sizeof(float));
for(int x968=0; x968 < 50; x968++) {
float x969 = x960[x968];
double x970 = (double)x969;
double x971 = sqrt(x970);
float x972 = (float)x971;
x967[x968] = x972;

}
float* x976 = (float*)myMalloc(50 * sizeof(float));
for(int x977=0; x977 < 50; x977++) {
float x978 = x953[x977];
float x979 = x967[x977];
float x980 = x978 / x979;
x976[x977] = x980;

}
for(int x984=0; x984 < 50; x984++) {
float x985 = x38[x984];
float x986 = x976[x984];
float x987 = x985 - x986;
x38[x984] = x987;

}
for(int x991=0; x991 < 50; x991++) {
float x992 = x75[x991];
x75[x991] = 0.0f;

}
for(int x996=0; x996 < 26; x996++) {
float x997 = x80[x996];
bool x998 = x997 > 5.0f;
if (x998) {
x80[x996] = 5.0f;
} else {
}
float x1002 = x80[x996];
bool x1003 = x1002 < -5.0f;
if (x1003) {
x80[x996] = -5.0f;
} else {
}

}
float* x1009 = (float*)myMalloc(26 * sizeof(float));
for(int x1010=0; x1010 < 26; x1010++) {
float x1011 = x80[x1010];
float x1012 = x80[x1010];
float x1013 = x1011 * x1012;
x1009[x1010] = x1013;

}
for(int x1017=0; x1017 < 26; x1017++) {
float x1018 = x110[x1017];
float x1019 = x1009[x1017];
float x1020 = x1018 + x1019;
x110[x1017] = x1020;

}
float* x1024 = (float*)myMalloc(26 * sizeof(float));
for(int x1025=0; x1025 < 26; x1025++) {
float x1026 = x80[x1025];
float x1027 = x1026 * 0.1f;
x1024[x1025] = x1027;

}
float* x1031 = (float*)myMalloc(26 * sizeof(float));
for(int x1032=0; x1032 < 26; x1032++) {
float x1033 = x110[x1032];
float x1034 = x1033 + 1.0E-8f;
x1031[x1032] = x1034;

}
float* x1038 = (float*)myMalloc(26 * sizeof(float));
for(int x1039=0; x1039 < 26; x1039++) {
float x1040 = x1031[x1039];
double x1041 = (double)x1040;
double x1042 = sqrt(x1041);
float x1043 = (float)x1042;
x1038[x1039] = x1043;

}
float* x1047 = (float*)myMalloc(26 * sizeof(float));
for(int x1048=0; x1048 < 26; x1048++) {
float x1049 = x1024[x1048];
float x1050 = x1038[x1048];
float x1051 = x1049 / x1050;
x1047[x1048] = x1051;

}
for(int x1055=0; x1055 < 26; x1055++) {
float x1056 = x44[x1055];
float x1057 = x1047[x1055];
float x1058 = x1056 - x1057;
x44[x1055] = x1058;

}
for(int x1062=0; x1062 < 26; x1062++) {
float x1063 = x80[x1062];
x80[x1062] = 0.0f;

}
for(int x1067=0; x1067 < 50; x1067++) {
float x1068 = x85[x1067];
x85[x1067] = 0.0f;

}
for(int x1072=0; x1072 < 50; x1072++) {
float x1073 = x55[x1072];
x50[x1072] = x1073;

}
mallocAddr = (void*)x117;

}
double x1080 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x1083 = (long)fopen(x0, "w");
fprintf((FILE *)x1083, "unit: %s\n", "100 iteration");
for(int x1086=0; x1086 < 51; x1086++) {
double x1087 = x115[x1086];
fprintf((FILE *)x1083, "%lf\n", x1087);

}
double x1081 = x116 - x1;
double x1082 = x1080 - x116;
fprintf((FILE *)x1083, "run time: %lf %lf\n", x1081, x1082);
fclose((FILE*)x1083);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

