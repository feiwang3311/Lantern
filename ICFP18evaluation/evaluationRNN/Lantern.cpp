
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
double* x15 = (double*)myMalloc(1300 * sizeof(double));
for(int x17=0; x17 < 1300; x17++) {
double x18 = d(gen);
double x19 = x18 * 0.01;
x15[x17] = x19;

}
double* x23 = (double*)myMalloc(2500 * sizeof(double));
for(int x25=0; x25 < 2500; x25++) {
double x26 = d(gen);
double x27 = x26 * 0.01;
x23[x25] = x27;

}
double* x31 = (double*)myMalloc(1300 * sizeof(double));
for(int x32=0; x32 < 1300; x32++) {
double x33 = d(gen);
double x34 = x33 * 0.01;
x31[x32] = x34;

}
double* x38 = (double*)myMalloc(50 * sizeof(double));
for(int x40=0; x40 < 50; x40++) {
x38[x40] = 0.0;

}
double* x44 = (double*)myMalloc(26 * sizeof(double));
for(int x46=0; x46 < 26; x46++) {
x44[x46] = 0.0;

}
double* x50 = (double*)myMalloc(50 * sizeof(double));
for(int x51=0; x51 < 50; x51++) {
x50[x51] = 0.0;

}
double* x55 = (double*)myMalloc(50 * sizeof(double));
for(int x56=0; x56 < 50; x56++) {
x55[x56] = 0.0;

}
double* x60 = (double*)myMalloc(1300 * sizeof(double));
for(int x61=0; x61 < 1300; x61++) {
x60[x61] = 0.0;

}
double* x65 = (double*)myMalloc(2500 * sizeof(double));
for(int x66=0; x66 < 2500; x66++) {
x65[x66] = 0.0;

}
double* x70 = (double*)myMalloc(1300 * sizeof(double));
for(int x71=0; x71 < 1300; x71++) {
x70[x71] = 0.0;

}
double* x75 = (double*)myMalloc(50 * sizeof(double));
for(int x76=0; x76 < 50; x76++) {
x75[x76] = 0.0;

}
double* x80 = (double*)myMalloc(26 * sizeof(double));
for(int x81=0; x81 < 26; x81++) {
x80[x81] = 0.0;

}
double* x85 = (double*)myMalloc(50 * sizeof(double));
for(int x86=0; x86 < 50; x86++) {
x85[x86] = 0.0;

}
double* x90 = (double*)myMalloc(1300 * sizeof(double));
for(int x91=0; x91 < 1300; x91++) {
x90[x91] = 0.0;

}
double* x95 = (double*)myMalloc(2500 * sizeof(double));
for(int x96=0; x96 < 2500; x96++) {
x95[x96] = 0.0;

}
double* x100 = (double*)myMalloc(1300 * sizeof(double));
for(int x101=0; x101 < 1300; x101++) {
x100[x101] = 0.0;

}
double* x105 = (double*)myMalloc(50 * sizeof(double));
for(int x106=0; x106 < 50; x106++) {
x105[x106] = 0.0;

}
double* x110 = (double*)myMalloc(26 * sizeof(double));
for(int x111=0; x111 < 26; x111++) {
x110[x111] = 0.0;

}
double* x115 = (double*)myMalloc(51 * sizeof(double));
double x116 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x117 = (long)mallocAddr;
int32_t x118 = 0;
x118 -= 20;
double x120 = 60.0;
for(int x122=0; x122 < 5001; x122++) {
double* x160 = (double*)myMalloc(1 * sizeof(double));
int32_t* x136 = (int32_t*)myMalloc(20 * sizeof(int32_t));
int32_t* x137 = (int32_t*)myMalloc(20 * sizeof(int32_t));
function<void(int32_t,double**)> x175 = [&](int32_t x176,double** x177) {
double** x179 = x177;
double* x180 = x179[0];
double* x181 = x179[1];
double* x182 = x179[2];
double* x183 = x179[3];
int32_t x178 = x176;
bool x184 = x178 < 20;
if (x184) {
double* x185 = (double*)myMalloc(26 * sizeof(double));
for(int x186=0; x186 < 26; x186++) {
x185[x186] = 0.0;

}
int32_t x190 = x136[x178];
x185[x190] = 1.0;
double* x192 = (double*)myMalloc(26 * sizeof(double));
for(int x193=0; x193 < 26; x193++) {
x192[x193] = 0.0;

}
double* x197 = (double*)myMalloc(26 * sizeof(double));
for(int x198=0; x198 < 26; x198++) {
x197[x198] = 0.0;

}
int32_t x202 = x137[x178];
x197[x202] = 1.0;
double* x204 = (double*)myMalloc(26 * sizeof(double));
for(int x205=0; x205 < 26; x205++) {
x204[x205] = 0.0;

}
// dot WrappedArray(50, 26) - WrappedArray(26)
int32_t x210 = 0;
double* x211 = (double*)myMalloc(50 * sizeof(double));
for(int x212=0; x212 < 50; x212++) {
double x213 = 0.0;
for(int x214=0; x214 < 26; x214++) {
int32_t x215 = x210;
double x216 = x15[x215];
double x217 = x185[x214];
double x218 = x216 * x217;
x213 += x218;
x210 += 1;

}
double x223 = x213;
x211[x212] = x223;

}
double* x227 = (double*)myMalloc(50 * sizeof(double));
for(int x228=0; x228 < 50; x228++) {
x227[x228] = 0.0;

}
// dot WrappedArray(50, 50) - WrappedArray(50)
int32_t x233 = 0;
double* x234 = (double*)myMalloc(50 * sizeof(double));
for(int x235=0; x235 < 50; x235++) {
double x236 = 0.0;
for(int x237=0; x237 < 50; x237++) {
int32_t x238 = x233;
double x239 = x23[x238];
double x240 = x182[x237];
double x241 = x239 * x240;
x236 += x241;
x233 += 1;

}
double x246 = x236;
x234[x235] = x246;

}
double* x250 = (double*)myMalloc(50 * sizeof(double));
for(int x251=0; x251 < 50; x251++) {
x250[x251] = 0.0;

}
double* x255 = (double*)myMalloc(50 * sizeof(double));
for(int x256=0; x256 < 50; x256++) {
double x257 = x211[x256];
double x258 = x234[x256];
double x259 = x257 + x258;
x255[x256] = x259;

}
double* x263 = (double*)myMalloc(50 * sizeof(double));
for(int x264=0; x264 < 50; x264++) {
x263[x264] = 0.0;

}
double* x268 = (double*)myMalloc(50 * sizeof(double));
for(int x269=0; x269 < 50; x269++) {
double x270 = x255[x269];
double x271 = x38[x269];
double x272 = x270 + x271;
x268[x269] = x272;

}
double* x276 = (double*)myMalloc(50 * sizeof(double));
for(int x277=0; x277 < 50; x277++) {
x276[x277] = 0.0;

}
double* x281 = (double*)myMalloc(50 * sizeof(double));
for(int x282=0; x282 < 50; x282++) {
double x283 = x268[x282];
double x284 = tanh(x283);
x281[x282] = x284;

}
double* x288 = (double*)myMalloc(50 * sizeof(double));
for(int x289=0; x289 < 50; x289++) {
x288[x289] = 0.0;

}
// dot WrappedArray(26, 50) - WrappedArray(50)
int32_t x294 = 0;
double* x295 = (double*)myMalloc(26 * sizeof(double));
for(int x296=0; x296 < 26; x296++) {
double x297 = 0.0;
for(int x298=0; x298 < 50; x298++) {
int32_t x299 = x294;
double x300 = x31[x299];
double x301 = x281[x298];
double x302 = x300 * x301;
x297 += x302;
x294 += 1;

}
double x307 = x297;
x295[x296] = x307;

}
double* x311 = (double*)myMalloc(26 * sizeof(double));
for(int x312=0; x312 < 26; x312++) {
x311[x312] = 0.0;

}
double* x316 = (double*)myMalloc(26 * sizeof(double));
for(int x317=0; x317 < 26; x317++) {
double x318 = x295[x317];
double x319 = x44[x317];
double x320 = x318 + x319;
x316[x317] = x320;

}
double* x324 = (double*)myMalloc(26 * sizeof(double));
for(int x325=0; x325 < 26; x325++) {
x324[x325] = 0.0;

}
double* x329 = (double*)myMalloc(26 * sizeof(double));
for(int x330=0; x330 < 26; x330++) {
double x331 = x316[x330];
double x332 = exp(x331);
x329[x330] = x332;

}
double* x336 = (double*)myMalloc(26 * sizeof(double));
for(int x337=0; x337 < 26; x337++) {
x336[x337] = 0.0;

}
// Here
double x342 = 0.0;
for(int x343=0; x343 < 26; x343++) {
double x344 = x342;
double x345 = x329[x343];
double x346 = x344 + x345;
x342 = x346;

}
double x350 = x342;
double* x351 = (double*)myMalloc(1 * sizeof(double));
x351[0] = x350;
double* x353 = (double*)myMalloc(1 * sizeof(double));
for(int x354=0; x354 < 1; x354++) {
x353[x354] = 0.0;

}
double x358 = x351[0];
double* x359 = (double*)myMalloc(26 * sizeof(double));
for(int x360=0; x360 < 26; x360++) {
double x361 = x329[x360];
double x362 = x361 / x358;
x359[x360] = x362;

}
double* x366 = (double*)myMalloc(26 * sizeof(double));
for(int x367=0; x367 < 26; x367++) {
x366[x367] = 0.0;

}
// dot WrappedArray(26) - WrappedArray(26)
int32_t x372 = 0;
double* x373 = (double*)myMalloc(1 * sizeof(double));
for(int x374=0; x374 < 1; x374++) {
double x375 = 0.0;
for(int x376=0; x376 < 26; x376++) {
int32_t x377 = x372;
double x378 = x359[x377];
double x379 = x197[x376];
double x380 = x378 * x379;
x375 += x380;
x372 += 1;

}
double x385 = x375;
x373[x374] = x385;

}
double* x389 = (double*)myMalloc(1 * sizeof(double));
for(int x390=0; x390 < 1; x390++) {
x389[x390] = 0.0;

}
double* x394 = (double*)myMalloc(1 * sizeof(double));
for(int x395=0; x395 < 1; x395++) {
double x396 = x373[x395];
double x397 = log(x396);
x394[x395] = x397;

}
double* x401 = (double*)myMalloc(1 * sizeof(double));
for(int x402=0; x402 < 1; x402++) {
x401[x402] = 0.0;

}
double* x406 = (double*)myMalloc(1 * sizeof(double));
for(int x407=0; x407 < 1; x407++) {
double x408 = x394[x407];
double x409 = x180[0];
double x410 = x409 - x408;
x406[x407] = x410;

}
double* x414 = (double*)myMalloc(1 * sizeof(double));
for(int x415=0; x415 < 1; x415++) {
x414[x415] = 0.0;

}
double** x420 = (double**)myMalloc(4 * sizeof(double*));
x420[0] = x406;
x420[1] = x414;
x420[2] = x281;
x420[3] = x288;
int32_t x525 = 0;
int32_t x526 = x525;
int32_t x527 = x526;
int32_t x528 = 0;
int32_t x529 = x528;
int32_t x530 = x529;
int32_t x531 = 0;
int32_t x532 = x531;
int32_t x533 = x532;
int32_t x564 = 0;
int32_t x565 = x564;
int32_t x566 = x565;
int32_t x567 = 0;
int32_t x568 = x567;
int32_t x569 = x568;
int32_t x570 = 0;
int32_t x571 = x570;
int32_t x572 = x571;
int32_t x645 = 0;
int32_t x646 = x645;
int32_t x647 = x646;
int32_t x648 = 0;
int32_t x649 = x648;
int32_t x650 = x649;
int32_t x651 = 0;
int32_t x652 = x651;
int32_t x653 = x652;
int32_t x684 = 0;
int32_t x685 = x684;
int32_t x686 = x685;
int32_t x687 = 0;
int32_t x688 = x687;
int32_t x689 = x688;
int32_t x690 = 0;
int32_t x691 = x690;
int32_t x692 = x691;
int32_t x723 = 0;
int32_t x724 = x723;
int32_t x725 = x724;
int32_t x726 = 0;
int32_t x727 = x726;
int32_t x728 = x727;
int32_t x729 = 0;
int32_t x730 = x729;
int32_t x731 = x730;
int32_t x762 = 0;
int32_t x763 = x762;
int32_t x764 = x763;
int32_t x765 = 0;
int32_t x766 = x765;
int32_t x767 = x766;
int32_t x768 = 0;
int32_t x769 = x768;
int32_t x770 = x769;
int32_t x419 = x178 + 1;
x175(x419,x420);
// += tensor of dim 0
double x428 = x414[0];
for(int x429=0; x429 < 1; x429++) {
double x430 = x181[x429];
double x431 = x430 + x428;
x181[x429] = x431;

}
double x435 = x414[0];
for(int x436=0; x436 < 1; x436++) {
double x437 = x401[x436];
double x438 = x437 - x435;
x401[x436] = x438;

}
for(int x442=0; x442 < 1; x442++) {
double x443 = x389[0];
double x444 = x401[0];
double x445 = x373[0];
double x446 = x444 / x445;
double x447 = x443 + x446;
x389[0] = x447;

}
double x451 = x389[0];
// Generate code for addMul
for(int x453=0; x453 < 26; x453++) {
double x454 = x366[x453];
double x455 = x197[x453];
double x456 = x451 * x455;
double x457 = x454 + x456;
x366[x453] = x457;

}
double x461 = x389[0];
// Generate code for addMul
for(int x463=0; x463 < 26; x463++) {
double x464 = x204[x463];
double x465 = x359[x463];
double x466 = x461 * x465;
double x467 = x464 + x466;
x204[x463] = x467;

}
for(int x471=0; x471 < 26; x471++) {
double x472 = x336[x471];
double x473 = x366[x471];
double x474 = x351[0];
double x475 = x473 / x474;
double x476 = x472 + x475;
x336[x471] = x476;

}
for(int x480=0; x480 < 26; x480++) {
double x481 = x353[0];
double x482 = x329[x480];
double x483 = x366[x480];
double x485 = x351[0];
double x484 = x482 * x483;
double x486 = x485 * x485;
double x487 = x484 / x486;
double x488 = x481 - x487;
x353[0] = x488;

}
// += tensor of dim 0
double x493 = x353[0];
for(int x494=0; x494 < 26; x494++) {
double x495 = x336[x494];
double x496 = x495 + x493;
x336[x494] = x496;

}
// backpropage exp
for(int x501=0; x501 < 26; x501++) {
double x502 = x324[x501];
double x503 = x329[x501];
double x504 = x336[x501];
double x505 = x503 * x504;
double x506 = x502 + x505;
x324[x501] = x506;

}
// backpropagate +
for(int x511=0; x511 < 26; x511++) {
double x512 = x311[x511];
double x513 = x324[x511];
double x514 = x512 + x513;
x311[x511] = x514;

}
for(int x518=0; x518 < 26; x518++) {
double x519 = x80[x518];
double x520 = x324[x518];
double x521 = x519 + x520;
x80[x518] = x521;

}
for(int x534=0; x534 < 26; x534++) {
int32_t x535 = x533;
int32_t x536 = x535;
for(int x537=0; x537 < 50; x537++) {
int32_t x538 = x536;
int32_t x539 = x538;
int32_t x540 = x530;
int32_t x541 = x540;
for(int x542=0; x542 < 1; x542++) {
int32_t x543 = x527;
double x544 = x70[x543];
int32_t x545 = x541;
double x546 = x311[x545];
int32_t x547 = x539;
double x548 = x281[x547];
double x549 = x546 * x548;
double x550 = x544 + x549;
x70[x543] = x550;
x541 += 1;
x539 += 50;

}
x527 += 1;
x536 += 1;

}
x530 += 1;
x533 *= 0;

}
for(int x573=0; x573 < 1; x573++) {
int32_t x574 = x572;
int32_t x575 = x574;
for(int x576=0; x576 < 50; x576++) {
int32_t x577 = x575;
int32_t x578 = x577;
int32_t x579 = x569;
int32_t x580 = x579;
for(int x581=0; x581 < 26; x581++) {
int32_t x582 = x566;
double x583 = x288[x582];
int32_t x584 = x580;
double x585 = x311[x584];
int32_t x586 = x578;
double x587 = x31[x586];
double x588 = x585 * x587;
double x589 = x583 + x588;
x288[x582] = x589;
x580 += 1;
x578 += 50;

}
x566 += 1;
x575 += 1;

}
x569 += 26;
x572 *= 0;

}
// backpropagate tanh
for(int x604=0; x604 < 50; x604++) {
double x605 = x276[x604];
double x606 = x281[x604];
double x609 = x288[x604];
double x607 = x606 * x606;
double x608 = 1.0 - x607;
double x610 = x608 * x609;
double x611 = x605 + x610;
x276[x604] = x611;

}
// backpropagate +
for(int x616=0; x616 < 50; x616++) {
double x617 = x263[x616];
double x618 = x276[x616];
double x619 = x617 + x618;
x263[x616] = x619;

}
for(int x623=0; x623 < 50; x623++) {
double x624 = x75[x623];
double x625 = x276[x623];
double x626 = x624 + x625;
x75[x623] = x626;

}
// backpropagate +
for(int x631=0; x631 < 50; x631++) {
double x632 = x227[x631];
double x633 = x263[x631];
double x634 = x632 + x633;
x227[x631] = x634;

}
for(int x638=0; x638 < 50; x638++) {
double x639 = x250[x638];
double x640 = x263[x638];
double x641 = x639 + x640;
x250[x638] = x641;

}
for(int x654=0; x654 < 50; x654++) {
int32_t x655 = x653;
int32_t x656 = x655;
for(int x657=0; x657 < 50; x657++) {
int32_t x658 = x656;
int32_t x659 = x658;
int32_t x660 = x650;
int32_t x661 = x660;
for(int x662=0; x662 < 1; x662++) {
int32_t x663 = x647;
double x664 = x65[x663];
int32_t x665 = x661;
double x666 = x250[x665];
int32_t x667 = x659;
double x668 = x182[x667];
double x669 = x666 * x668;
double x670 = x664 + x669;
x65[x663] = x670;
x661 += 1;
x659 += 50;

}
x647 += 1;
x656 += 1;

}
x650 += 1;
x653 *= 0;

}
for(int x693=0; x693 < 1; x693++) {
int32_t x694 = x692;
int32_t x695 = x694;
for(int x696=0; x696 < 50; x696++) {
int32_t x697 = x695;
int32_t x698 = x697;
int32_t x699 = x689;
int32_t x700 = x699;
for(int x701=0; x701 < 50; x701++) {
int32_t x702 = x686;
double x703 = x183[x702];
int32_t x704 = x700;
double x705 = x250[x704];
int32_t x706 = x698;
double x707 = x23[x706];
double x708 = x705 * x707;
double x709 = x703 + x708;
x183[x702] = x709;
x700 += 1;
x698 += 50;

}
x686 += 1;
x695 += 1;

}
x689 += 50;
x692 *= 0;

}
for(int x732=0; x732 < 50; x732++) {
int32_t x733 = x731;
int32_t x734 = x733;
for(int x735=0; x735 < 26; x735++) {
int32_t x736 = x734;
int32_t x737 = x736;
int32_t x738 = x728;
int32_t x739 = x738;
for(int x740=0; x740 < 1; x740++) {
int32_t x741 = x725;
double x742 = x60[x741];
int32_t x743 = x739;
double x744 = x227[x743];
int32_t x745 = x737;
double x746 = x185[x745];
double x747 = x744 * x746;
double x748 = x742 + x747;
x60[x741] = x748;
x739 += 1;
x737 += 26;

}
x725 += 1;
x734 += 1;

}
x728 += 1;
x731 *= 0;

}
for(int x771=0; x771 < 1; x771++) {
int32_t x772 = x770;
int32_t x773 = x772;
for(int x774=0; x774 < 26; x774++) {
int32_t x775 = x773;
int32_t x776 = x775;
int32_t x777 = x767;
int32_t x778 = x777;
for(int x779=0; x779 < 50; x779++) {
int32_t x780 = x764;
double x781 = x192[x780];
int32_t x782 = x778;
double x783 = x227[x782];
int32_t x784 = x776;
double x785 = x15[x784];
double x786 = x783 * x785;
double x787 = x781 + x786;
x192[x780] = x787;
x778 += 1;
x776 += 26;

}
x764 += 1;
x773 += 1;

}
x767 += 50;
x770 *= 0;

}
} else {
for(int x802=0; x802 < 50; x802++) {
double x803 = x182[x802];
x55[x802] = x803;

}
for(int x807=0; x807 < 1; x807++) {
double x808 = x181[x807];
x181[x807] = 1.0;

}
for(int x812=0; x812 < 1; x812++) {
double x813 = x180[x812];
x160[x812] = x813;

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
double x130 = x50[x129];
x50[x129] = 0.0;

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
double* x149 = (double*)myMalloc(1 * sizeof(double));
for(int x151=0; x151 < 1; x151++) {
x149[x151] = 0.0;

}
double* x155 = (double*)myMalloc(1 * sizeof(double));
for(int x156=0; x156 < 1; x156++) {
x155[x156] = 0.0;

}
for(int x161=0; x161 < 1; x161++) {
x160[x161] = 0.0;

}
double* x165 = (double*)myMalloc(1 * sizeof(double));
for(int x166=0; x166 < 1; x166++) {
x165[x166] = 0.0;

}
double* x170 = (double*)myMalloc(1 * sizeof(double));
for(int x171=0; x171 < 1; x171++) {
x170[x171] = 0.0;

}
double** x820 = (double**)myMalloc(4 * sizeof(double*));
x820[0] = x165;
x820[1] = x170;
x820[2] = x50;
x820[3] = x85;
x175(0,x820);
double x827 = x160[0];
double x828 = x120;
double x829 = x828 * 0.9;
double x830 = x827 * 0.1;
double x831 = x829 + x830;
x120 = x831;
int32_t x833 = x122 % 100;
bool x834 = x833 == 0;
if (x834) {
double x835 = x120;
printf("iter %d, loss %f\n",x122,x835);
int32_t x837 = x122 / 100;
x115[x837] = x835;
} else {
}
for(int x841=0; x841 < 1300; x841++) {
double x842 = x60[x841];
bool x843 = x842 > 5.0;
if (x843) {
x60[x841] = 5.0;
} else {
}
double x847 = x60[x841];
bool x848 = x847 < -5.0;
if (x848) {
x60[x841] = -5.0;
} else {
}

}
double* x854 = (double*)myMalloc(1300 * sizeof(double));
for(int x855=0; x855 < 1300; x855++) {
double x856 = x60[x855];
double x857 = x60[x855];
double x858 = x856 * x857;
x854[x855] = x858;

}
for(int x862=0; x862 < 1300; x862++) {
double x863 = x90[x862];
double x864 = x854[x862];
double x865 = x863 + x864;
x90[x862] = x865;

}
double* x869 = (double*)myMalloc(1300 * sizeof(double));
for(int x870=0; x870 < 1300; x870++) {
double x871 = x60[x870];
double x872 = x871 * 0.1;
x869[x870] = x872;

}
double* x876 = (double*)myMalloc(1300 * sizeof(double));
for(int x877=0; x877 < 1300; x877++) {
double x878 = x90[x877];
double x879 = x878 + 1.0E-8;
x876[x877] = x879;

}
double* x883 = (double*)myMalloc(1300 * sizeof(double));
for(int x884=0; x884 < 1300; x884++) {
double x885 = x876[x884];
double x886 = sqrt(x885);
x883[x884] = x886;

}
double* x890 = (double*)myMalloc(1300 * sizeof(double));
for(int x891=0; x891 < 1300; x891++) {
double x892 = x869[x891];
double x893 = x883[x891];
double x894 = x892 / x893;
x890[x891] = x894;

}
for(int x898=0; x898 < 1300; x898++) {
double x899 = x15[x898];
double x900 = x890[x898];
double x901 = x899 - x900;
x15[x898] = x901;

}
for(int x905=0; x905 < 1300; x905++) {
double x906 = x60[x905];
x60[x905] = 0.0;

}
for(int x910=0; x910 < 2500; x910++) {
double x911 = x65[x910];
bool x912 = x911 > 5.0;
if (x912) {
x65[x910] = 5.0;
} else {
}
double x916 = x65[x910];
bool x917 = x916 < -5.0;
if (x917) {
x65[x910] = -5.0;
} else {
}

}
double* x923 = (double*)myMalloc(2500 * sizeof(double));
for(int x924=0; x924 < 2500; x924++) {
double x925 = x65[x924];
double x926 = x65[x924];
double x927 = x925 * x926;
x923[x924] = x927;

}
for(int x931=0; x931 < 2500; x931++) {
double x932 = x95[x931];
double x933 = x923[x931];
double x934 = x932 + x933;
x95[x931] = x934;

}
double* x938 = (double*)myMalloc(2500 * sizeof(double));
for(int x939=0; x939 < 2500; x939++) {
double x940 = x65[x939];
double x941 = x940 * 0.1;
x938[x939] = x941;

}
double* x945 = (double*)myMalloc(2500 * sizeof(double));
for(int x946=0; x946 < 2500; x946++) {
double x947 = x95[x946];
double x948 = x947 + 1.0E-8;
x945[x946] = x948;

}
double* x952 = (double*)myMalloc(2500 * sizeof(double));
for(int x953=0; x953 < 2500; x953++) {
double x954 = x945[x953];
double x955 = sqrt(x954);
x952[x953] = x955;

}
double* x959 = (double*)myMalloc(2500 * sizeof(double));
for(int x960=0; x960 < 2500; x960++) {
double x961 = x938[x960];
double x962 = x952[x960];
double x963 = x961 / x962;
x959[x960] = x963;

}
for(int x967=0; x967 < 2500; x967++) {
double x968 = x23[x967];
double x969 = x959[x967];
double x970 = x968 - x969;
x23[x967] = x970;

}
for(int x974=0; x974 < 2500; x974++) {
double x975 = x65[x974];
x65[x974] = 0.0;

}
for(int x979=0; x979 < 1300; x979++) {
double x980 = x70[x979];
bool x981 = x980 > 5.0;
if (x981) {
x70[x979] = 5.0;
} else {
}
double x985 = x70[x979];
bool x986 = x985 < -5.0;
if (x986) {
x70[x979] = -5.0;
} else {
}

}
double* x992 = (double*)myMalloc(1300 * sizeof(double));
for(int x993=0; x993 < 1300; x993++) {
double x994 = x70[x993];
double x995 = x70[x993];
double x996 = x994 * x995;
x992[x993] = x996;

}
for(int x1000=0; x1000 < 1300; x1000++) {
double x1001 = x100[x1000];
double x1002 = x992[x1000];
double x1003 = x1001 + x1002;
x100[x1000] = x1003;

}
double* x1007 = (double*)myMalloc(1300 * sizeof(double));
for(int x1008=0; x1008 < 1300; x1008++) {
double x1009 = x70[x1008];
double x1010 = x1009 * 0.1;
x1007[x1008] = x1010;

}
double* x1014 = (double*)myMalloc(1300 * sizeof(double));
for(int x1015=0; x1015 < 1300; x1015++) {
double x1016 = x100[x1015];
double x1017 = x1016 + 1.0E-8;
x1014[x1015] = x1017;

}
double* x1021 = (double*)myMalloc(1300 * sizeof(double));
for(int x1022=0; x1022 < 1300; x1022++) {
double x1023 = x1014[x1022];
double x1024 = sqrt(x1023);
x1021[x1022] = x1024;

}
double* x1028 = (double*)myMalloc(1300 * sizeof(double));
for(int x1029=0; x1029 < 1300; x1029++) {
double x1030 = x1007[x1029];
double x1031 = x1021[x1029];
double x1032 = x1030 / x1031;
x1028[x1029] = x1032;

}
for(int x1036=0; x1036 < 1300; x1036++) {
double x1037 = x31[x1036];
double x1038 = x1028[x1036];
double x1039 = x1037 - x1038;
x31[x1036] = x1039;

}
for(int x1043=0; x1043 < 1300; x1043++) {
double x1044 = x70[x1043];
x70[x1043] = 0.0;

}
for(int x1048=0; x1048 < 50; x1048++) {
double x1049 = x75[x1048];
bool x1050 = x1049 > 5.0;
if (x1050) {
x75[x1048] = 5.0;
} else {
}
double x1054 = x75[x1048];
bool x1055 = x1054 < -5.0;
if (x1055) {
x75[x1048] = -5.0;
} else {
}

}
double* x1061 = (double*)myMalloc(50 * sizeof(double));
for(int x1062=0; x1062 < 50; x1062++) {
double x1063 = x75[x1062];
double x1064 = x75[x1062];
double x1065 = x1063 * x1064;
x1061[x1062] = x1065;

}
for(int x1069=0; x1069 < 50; x1069++) {
double x1070 = x105[x1069];
double x1071 = x1061[x1069];
double x1072 = x1070 + x1071;
x105[x1069] = x1072;

}
double* x1076 = (double*)myMalloc(50 * sizeof(double));
for(int x1077=0; x1077 < 50; x1077++) {
double x1078 = x75[x1077];
double x1079 = x1078 * 0.1;
x1076[x1077] = x1079;

}
double* x1083 = (double*)myMalloc(50 * sizeof(double));
for(int x1084=0; x1084 < 50; x1084++) {
double x1085 = x105[x1084];
double x1086 = x1085 + 1.0E-8;
x1083[x1084] = x1086;

}
double* x1090 = (double*)myMalloc(50 * sizeof(double));
for(int x1091=0; x1091 < 50; x1091++) {
double x1092 = x1083[x1091];
double x1093 = sqrt(x1092);
x1090[x1091] = x1093;

}
double* x1097 = (double*)myMalloc(50 * sizeof(double));
for(int x1098=0; x1098 < 50; x1098++) {
double x1099 = x1076[x1098];
double x1100 = x1090[x1098];
double x1101 = x1099 / x1100;
x1097[x1098] = x1101;

}
for(int x1105=0; x1105 < 50; x1105++) {
double x1106 = x38[x1105];
double x1107 = x1097[x1105];
double x1108 = x1106 - x1107;
x38[x1105] = x1108;

}
for(int x1112=0; x1112 < 50; x1112++) {
double x1113 = x75[x1112];
x75[x1112] = 0.0;

}
for(int x1117=0; x1117 < 26; x1117++) {
double x1118 = x80[x1117];
bool x1119 = x1118 > 5.0;
if (x1119) {
x80[x1117] = 5.0;
} else {
}
double x1123 = x80[x1117];
bool x1124 = x1123 < -5.0;
if (x1124) {
x80[x1117] = -5.0;
} else {
}

}
double* x1130 = (double*)myMalloc(26 * sizeof(double));
for(int x1131=0; x1131 < 26; x1131++) {
double x1132 = x80[x1131];
double x1133 = x80[x1131];
double x1134 = x1132 * x1133;
x1130[x1131] = x1134;

}
for(int x1138=0; x1138 < 26; x1138++) {
double x1139 = x110[x1138];
double x1140 = x1130[x1138];
double x1141 = x1139 + x1140;
x110[x1138] = x1141;

}
double* x1145 = (double*)myMalloc(26 * sizeof(double));
for(int x1146=0; x1146 < 26; x1146++) {
double x1147 = x80[x1146];
double x1148 = x1147 * 0.1;
x1145[x1146] = x1148;

}
double* x1152 = (double*)myMalloc(26 * sizeof(double));
for(int x1153=0; x1153 < 26; x1153++) {
double x1154 = x110[x1153];
double x1155 = x1154 + 1.0E-8;
x1152[x1153] = x1155;

}
double* x1159 = (double*)myMalloc(26 * sizeof(double));
for(int x1160=0; x1160 < 26; x1160++) {
double x1161 = x1152[x1160];
double x1162 = sqrt(x1161);
x1159[x1160] = x1162;

}
double* x1166 = (double*)myMalloc(26 * sizeof(double));
for(int x1167=0; x1167 < 26; x1167++) {
double x1168 = x1145[x1167];
double x1169 = x1159[x1167];
double x1170 = x1168 / x1169;
x1166[x1167] = x1170;

}
for(int x1174=0; x1174 < 26; x1174++) {
double x1175 = x44[x1174];
double x1176 = x1166[x1174];
double x1177 = x1175 - x1176;
x44[x1174] = x1177;

}
for(int x1181=0; x1181 < 26; x1181++) {
double x1182 = x80[x1181];
x80[x1181] = 0.0;

}
for(int x1186=0; x1186 < 50; x1186++) {
double x1187 = x85[x1186];
x85[x1186] = 0.0;

}
for(int x1191=0; x1191 < 50; x1191++) {
double x1192 = x55[x1191];
x50[x1191] = x1192;

}
mallocAddr = (void*)x117;

}
double x1199 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x1202 = (long)fopen(x0, "w");
fprintf((FILE *)x1202, "unit: %s\n", "100 iteration");
for(int x1205=0; x1205 < 51; x1205++) {
double x1206 = x115[x1205];
fprintf((FILE *)x1202, "%lf\n", x1206);

}
double x1200 = x116 - x1;
double x1201 = x1199 - x116;
fprintf((FILE *)x1202, "run time: %lf %lf\n", x1200, x1201);
fclose((FILE*)x1202);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

