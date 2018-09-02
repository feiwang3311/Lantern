
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
        char* tmp = (char*) mallocAddr;
        tmp += bytes;
        mallocAddr = (void*) tmp;
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
int32_t x1 = open("test_data",0);
int32_t x2 = fsize(x1);
printf("data has %d chars\n",x2);
int32_t* x5 = (int32_t*)myMalloc(x2 * sizeof(int32_t));
char* x3 = (char *)mmap(0, x2, PROT_READ, MAP_FILE | MAP_SHARED, x1, 0);
for(int x7=0; x7 < x2; x7++) {
char x8 = x3[x7];
int32_t x9 = (int32_t ) x8;
int32_t x10 = x9 - 96;
x5[x7] = x10;

}
float* x14 = (float*)myMalloc(1300 * sizeof(float));
for(int x16=0; x16 < 1300; x16++) {
float x17 = d(gen);
float x18 = x17 * 0.01f;
x14[x16] = x18;

}
float* x22 = (float*)myMalloc(2500 * sizeof(float));
for(int x24=0; x24 < 2500; x24++) {
float x25 = d(gen);
float x26 = x25 * 0.01f;
x22[x24] = x26;

}
float* x30 = (float*)myMalloc(1300 * sizeof(float));
for(int x31=0; x31 < 1300; x31++) {
float x32 = d(gen);
float x33 = x32 * 0.01f;
x30[x31] = x33;

}
float* x37 = (float*)myMalloc(50 * sizeof(float));
for(int x39=0; x39 < 50; x39++) {
x37[x39] = 0.0f;

}
float* x43 = (float*)myMalloc(26 * sizeof(float));
for(int x45=0; x45 < 26; x45++) {
x43[x45] = 0.0f;

}
float* x49 = (float*)myMalloc(50 * sizeof(float));
for(int x50=0; x50 < 50; x50++) {
x49[x50] = 0.0f;

}
float* x54 = (float*)myMalloc(50 * sizeof(float));
for(int x55=0; x55 < 50; x55++) {
x54[x55] = 0.0f;

}
float* x59 = (float*)myMalloc(1300 * sizeof(float));
for(int x60=0; x60 < 1300; x60++) {
x59[x60] = 0.0f;

}
float* x64 = (float*)myMalloc(2500 * sizeof(float));
for(int x65=0; x65 < 2500; x65++) {
x64[x65] = 0.0f;

}
float* x69 = (float*)myMalloc(1300 * sizeof(float));
for(int x70=0; x70 < 1300; x70++) {
x69[x70] = 0.0f;

}
float* x74 = (float*)myMalloc(50 * sizeof(float));
for(int x75=0; x75 < 50; x75++) {
x74[x75] = 0.0f;

}
float* x79 = (float*)myMalloc(26 * sizeof(float));
for(int x80=0; x80 < 26; x80++) {
x79[x80] = 0.0f;

}
float* x84 = (float*)myMalloc(50 * sizeof(float));
for(int x85=0; x85 < 50; x85++) {
x84[x85] = 0.0f;

}
float* x89 = (float*)myMalloc(1300 * sizeof(float));
for(int x90=0; x90 < 1300; x90++) {
x89[x90] = 0.0f;

}
float* x94 = (float*)myMalloc(2500 * sizeof(float));
for(int x95=0; x95 < 2500; x95++) {
x94[x95] = 0.0f;

}
float* x99 = (float*)myMalloc(1300 * sizeof(float));
for(int x100=0; x100 < 1300; x100++) {
x99[x100] = 0.0f;

}
float* x104 = (float*)myMalloc(50 * sizeof(float));
for(int x105=0; x105 < 50; x105++) {
x104[x105] = 0.0f;

}
float* x109 = (float*)myMalloc(26 * sizeof(float));
for(int x110=0; x110 < 26; x110++) {
x109[x110] = 0.0f;

}
int64_t x114 = (long)mallocAddr;
int32_t x115 = 0;
x115 -= 20;
clock_t begin_0, end_0; double time_spent_0;
begin_0 = clock();
for(int x120=0; x120 < 2001; x120++) {
int32_t* x134 = (int32_t*)myMalloc(20 * sizeof(int32_t));
int32_t* x135 = (int32_t*)myMalloc(20 * sizeof(int32_t));
function<void(int32_t,function<void(float**)>,float**)> x174 = [&](int32_t x175,function<void(float**)> x176,float** x177) {
float** x180 = x177;
float* x181 = x180[0];
float* x182 = x180[1];
float* x183 = x180[2];
float* x184 = x180[3];
int32_t x178 = x175;
bool x185 = x178 < 20;
if (x185) {
float** x660 = (float**)myMalloc(4 * sizeof(float*));
x660[0] = x181;
x660[1] = x182;
x660[2] = x183;
x660[3] = x184;
int32_t x186 = x178 + 1;
function<void(float**)> x179 = x176;
function<void(float**)> x187 = [&](float** x188) {
float* x189 = x188[0];
float* x190 = x188[1];
float* x191 = x188[2];
float* x192 = x188[3];
float* x193 = (float*)myMalloc(26 * sizeof(float));
for(int x194=0; x194 < 26; x194++) {
x193[x194] = 0.0f;

}
int32_t x198 = x134[x178];
x193[x198] = 1.0f;
float* x200 = (float*)myMalloc(26 * sizeof(float));
for(int x201=0; x201 < 26; x201++) {
x200[x201] = 0.0f;

}
float* x205 = (float*)myMalloc(26 * sizeof(float));
for(int x206=0; x206 < 26; x206++) {
x205[x206] = 0.0f;

}
int32_t x210 = x135[x178];
x205[x210] = 1.0f;
float* x212 = (float*)myMalloc(26 * sizeof(float));
for(int x213=0; x213 < 26; x213++) {
x212[x213] = 0.0f;

}
// dot WrappedArray(50, 26) - WrappedArray(26)
int32_t x218 = 0;
float* x219 = (float*)myMalloc(50 * sizeof(float));
for(int x220=0; x220 < 50; x220++) {
float x221 = 0.0f;
for(int x222=0; x222 < 26; x222++) {
int32_t x223 = x218;
float x224 = x14[x223];
float x225 = x193[x222];
float x226 = x224 * x225;
x221 += x226;
x218 += 1;

}
float x231 = x221;
x219[x220] = x231;

}
float* x235 = (float*)myMalloc(50 * sizeof(float));
for(int x236=0; x236 < 50; x236++) {
x235[x236] = 0.0f;

}
// dot WrappedArray(50, 50) - WrappedArray(50)
int32_t x241 = 0;
float* x242 = (float*)myMalloc(50 * sizeof(float));
for(int x243=0; x243 < 50; x243++) {
float x244 = 0.0f;
for(int x245=0; x245 < 50; x245++) {
int32_t x246 = x241;
float x247 = x22[x246];
float x248 = x191[x245];
float x249 = x247 * x248;
x244 += x249;
x241 += 1;

}
float x254 = x244;
x242[x243] = x254;

}
float* x258 = (float*)myMalloc(50 * sizeof(float));
for(int x259=0; x259 < 50; x259++) {
x258[x259] = 0.0f;

}
float* x263 = (float*)myMalloc(50 * sizeof(float));
for(int x264=0; x264 < 50; x264++) {
float x265 = x219[x264];
float x266 = x242[x264];
float x267 = x265 + x266;
x263[x264] = x267;

}
float* x271 = (float*)myMalloc(50 * sizeof(float));
for(int x272=0; x272 < 50; x272++) {
x271[x272] = 0.0f;

}
float* x276 = (float*)myMalloc(50 * sizeof(float));
for(int x277=0; x277 < 50; x277++) {
float x278 = x263[x277];
float x279 = x37[x277];
float x280 = x278 + x279;
x276[x277] = x280;

}
float* x284 = (float*)myMalloc(50 * sizeof(float));
for(int x285=0; x285 < 50; x285++) {
x284[x285] = 0.0f;

}
float* x289 = (float*)myMalloc(50 * sizeof(float));
for(int x290=0; x290 < 50; x290++) {
float x291 = x276[x290];
double x292 = (double)x291;
double x293 = tanh(x292);
float x294 = (float)x293;
x289[x290] = x294;

}
float* x298 = (float*)myMalloc(50 * sizeof(float));
for(int x299=0; x299 < 50; x299++) {
x298[x299] = 0.0f;

}
// dot WrappedArray(26, 50) - WrappedArray(50)
int32_t x304 = 0;
float* x305 = (float*)myMalloc(26 * sizeof(float));
for(int x306=0; x306 < 26; x306++) {
float x307 = 0.0f;
for(int x308=0; x308 < 50; x308++) {
int32_t x309 = x304;
float x310 = x30[x309];
float x311 = x289[x308];
float x312 = x310 * x311;
x307 += x312;
x304 += 1;

}
float x317 = x307;
x305[x306] = x317;

}
float* x321 = (float*)myMalloc(26 * sizeof(float));
for(int x322=0; x322 < 26; x322++) {
x321[x322] = 0.0f;

}
float* x326 = (float*)myMalloc(26 * sizeof(float));
for(int x327=0; x327 < 26; x327++) {
float x328 = x305[x327];
float x329 = x43[x327];
float x330 = x328 + x329;
x326[x327] = x330;

}
float* x334 = (float*)myMalloc(26 * sizeof(float));
for(int x335=0; x335 < 26; x335++) {
x334[x335] = 0.0f;

}
float* x339 = (float*)myMalloc(26 * sizeof(float));
for(int x340=0; x340 < 26; x340++) {
float x341 = x326[x340];
double x342 = (double)x341;
double x343 = exp(x342);
float x344 = (float)x343;
x339[x340] = x344;

}
float* x348 = (float*)myMalloc(26 * sizeof(float));
for(int x349=0; x349 < 26; x349++) {
x348[x349] = 0.0f;

}
float x353 = 0.0f;
for(int x354=0; x354 < 26; x354++) {
float x355 = x353;
float x356 = x339[x354];
float x357 = x355 + x356;
x353 = x357;

}
float x361 = x353;
float* x362 = (float*)myMalloc(1 * sizeof(float));
x362[0] = x361;
float* x364 = (float*)myMalloc(1 * sizeof(float));
for(int x365=0; x365 < 1; x365++) {
x364[x365] = 0.0f;

}
float x369 = x362[0];
float* x370 = (float*)myMalloc(26 * sizeof(float));
for(int x371=0; x371 < 26; x371++) {
float x372 = x339[x371];
float x373 = x372 / x369;
x370[x371] = x373;

}
float* x377 = (float*)myMalloc(26 * sizeof(float));
for(int x378=0; x378 < 26; x378++) {
x377[x378] = 0.0f;

}
// dot WrappedArray(26) - WrappedArray(26)
int32_t x383 = 0;
float* x384 = (float*)myMalloc(1 * sizeof(float));
float x385 = 0.0f;
for(int x386=0; x386 < 26; x386++) {
int32_t x387 = x383;
float x388 = x370[x387];
float x389 = x205[x386];
float x390 = x388 * x389;
x385 += x390;
x383 += 1;

}
float x395 = x385;
x384[0] = x395;
float* x397 = (float*)myMalloc(1 * sizeof(float));
for(int x398=0; x398 < 1; x398++) {
x397[x398] = 0.0f;

}
float* x402 = (float*)myMalloc(1 * sizeof(float));
float x403 = x384[0];
double x404 = (double)x403;
double x405 = log(x404);
float x406 = (float)x405;
x402[0] = x406;
float* x408 = (float*)myMalloc(1 * sizeof(float));
for(int x409=0; x409 < 1; x409++) {
x408[x409] = 0.0f;

}
float* x413 = (float*)myMalloc(1 * sizeof(float));
float x414 = x402[0];
float x415 = x189[0];
float x416 = x415 - x414;
x413[0] = x416;
float* x418 = (float*)myMalloc(1 * sizeof(float));
for(int x419=0; x419 < 1; x419++) {
x418[x419] = 0.0f;

}
float** x423 = (float**)myMalloc(4 * sizeof(float*));
x423[0] = x413;
x423[1] = x418;
x423[2] = x289;
x423[3] = x298;
x179(x423);
// += tensor of dim 0
float x430 = x418[0];
float x431 = x190[0];
float x432 = x431 + x430;
x190[0] = x432;
float x434 = x418[0];
float x435 = x408[0];
float x436 = x435 - x434;
x408[0] = x436;
float x438 = x397[0];
float x439 = x408[0];
float x440 = x384[0];
float x441 = x439 / x440;
float x442 = x438 + x441;
x397[0] = x442;
float x444 = x397[0];
// Generate code for addMul
for(int x446=0; x446 < 26; x446++) {
float x447 = x377[x446];
float x448 = x205[x446];
float x449 = x444 * x448;
float x450 = x447 + x449;
x377[x446] = x450;

}
float x454 = x397[0];
// Generate code for addMul
for(int x456=0; x456 < 26; x456++) {
float x457 = x212[x456];
float x458 = x370[x456];
float x459 = x454 * x458;
float x460 = x457 + x459;
x212[x456] = x460;

}
for(int x464=0; x464 < 26; x464++) {
float x465 = x348[x464];
float x466 = x377[x464];
float x467 = x362[0];
float x468 = x466 / x467;
float x469 = x465 + x468;
x348[x464] = x469;

}
for(int x473=0; x473 < 26; x473++) {
float x474 = x364[0];
float x475 = x339[x473];
float x476 = x377[x473];
float x478 = x362[0];
float x477 = x475 * x476;
float x479 = x478 * x478;
float x480 = x477 / x479;
float x481 = x474 - x480;
x364[0] = x481;

}
// += tensor of dim 0
float x486 = x364[0];
for(int x487=0; x487 < 26; x487++) {
float x488 = x348[x487];
float x489 = x488 + x486;
x348[x487] = x489;

}
// backpropage exp
for(int x494=0; x494 < 26; x494++) {
float x495 = x334[x494];
float x496 = x339[x494];
float x497 = x348[x494];
float x498 = x496 * x497;
float x499 = x495 + x498;
x334[x494] = x499;

}
// backpropagate +
for(int x504=0; x504 < 26; x504++) {
float x505 = x321[x504];
float x506 = x334[x504];
float x507 = x505 + x506;
x321[x504] = x507;

}
for(int x511=0; x511 < 26; x511++) {
float x512 = x79[x511];
float x513 = x334[x511];
float x514 = x512 + x513;
x79[x511] = x514;

}
// add_cartesian
int32_t x519 = 0;
for(int x520=0; x520 < 26; x520++) {
for(int x521=0; x521 < 50; x521++) {
int32_t x522 = x519;
int32_t x523 = x522 + x521;
float x524 = x69[x523];
float x525 = x289[x521];
float x526 = x321[x520];
float x527 = x525 * x526;
float x528 = x524 + x527;
x69[x523] = x528;

}
x519 += 50;

}
int32_t x535 = 0;
for(int x536=0; x536 < 26; x536++) {
for(int x537=0; x537 < 50; x537++) {
float x538 = x298[x537];
int32_t x539 = x535;
int32_t x540 = x539 + x537;
float x541 = x30[x540];
float x542 = x321[x536];
float x543 = x541 * x542;
float x544 = x538 + x543;
x298[x537] = x544;

}
x535 += 50;

}
// backpropagate tanh
for(int x552=0; x552 < 50; x552++) {
float x553 = x284[x552];
float x554 = x289[x552];
float x557 = x298[x552];
float x555 = x554 * x554;
float x556 = 1.0f - x555;
float x558 = x556 * x557;
float x559 = x553 + x558;
x284[x552] = x559;

}
// backpropagate +
for(int x564=0; x564 < 50; x564++) {
float x565 = x271[x564];
float x566 = x284[x564];
float x567 = x565 + x566;
x271[x564] = x567;

}
for(int x571=0; x571 < 50; x571++) {
float x572 = x74[x571];
float x573 = x284[x571];
float x574 = x572 + x573;
x74[x571] = x574;

}
// backpropagate +
for(int x579=0; x579 < 50; x579++) {
float x580 = x235[x579];
float x581 = x271[x579];
float x582 = x580 + x581;
x235[x579] = x582;

}
for(int x586=0; x586 < 50; x586++) {
float x587 = x258[x586];
float x588 = x271[x586];
float x589 = x587 + x588;
x258[x586] = x589;

}
// add_cartesian
int32_t x594 = 0;
for(int x595=0; x595 < 50; x595++) {
for(int x596=0; x596 < 50; x596++) {
int32_t x597 = x594;
int32_t x598 = x597 + x596;
float x599 = x64[x598];
float x600 = x191[x596];
float x601 = x258[x595];
float x602 = x600 * x601;
float x603 = x599 + x602;
x64[x598] = x603;

}
x594 += 50;

}
int32_t x610 = 0;
for(int x611=0; x611 < 50; x611++) {
for(int x612=0; x612 < 50; x612++) {
float x613 = x192[x612];
int32_t x614 = x610;
int32_t x615 = x614 + x612;
float x616 = x22[x615];
float x617 = x258[x611];
float x618 = x616 * x617;
float x619 = x613 + x618;
x192[x612] = x619;

}
x610 += 50;

}
// add_cartesian
int32_t x627 = 0;
for(int x628=0; x628 < 50; x628++) {
for(int x629=0; x629 < 26; x629++) {
int32_t x630 = x627;
int32_t x631 = x630 + x629;
float x632 = x59[x631];
float x633 = x193[x629];
float x634 = x235[x628];
float x635 = x633 * x634;
float x636 = x632 + x635;
x59[x631] = x636;

}
x627 += 26;

}
int32_t x643 = 0;
for(int x644=0; x644 < 50; x644++) {
for(int x645=0; x645 < 26; x645++) {
float x646 = x200[x645];
int32_t x647 = x643;
int32_t x648 = x647 + x645;
float x649 = x14[x648];
float x650 = x235[x644];
float x651 = x649 * x650;
float x652 = x646 + x651;
x200[x645] = x652;

}
x643 += 26;

}
};
x174(x186,x187,x660);
} else {
float** x668 = (float**)myMalloc(4 * sizeof(float*));
x668[0] = x181;
x668[1] = x182;
x668[2] = x183;
x668[3] = x184;
function<void(float**)> x179 = x176;
x179(x668);
}
};
x115 += 20;
int32_t x122 = x115;
int32_t x123 = x122 + 20;
int32_t x124 = x123 + 1;
bool x125 = x124 >= x2;
if (x125) {
x115 = 0;
for(int x127=0; x127 < 50; x127++) {
float x128 = x49[x127];
x49[x127] = 0.0f;

}
} else {
}
for(int x137=0; x137 < 20; x137++) {
int32_t x139 = x115;
int32_t x140 = x139 + x137;
int32_t x141 = x5[x140];
int32_t x138 = 19 - x137;
x134[x138] = x141;
int32_t x143 = x140 + 1;
int32_t x144 = x5[x143];
x135[x138] = x144;

}
float* x148 = (float*)myMalloc(1 * sizeof(float));
for(int x150=0; x150 < 1; x150++) {
x148[x150] = 0.0f;

}
float* x154 = (float*)myMalloc(1 * sizeof(float));
for(int x155=0; x155 < 1; x155++) {
x154[x155] = 0.0f;

}
float* x159 = (float*)myMalloc(1 * sizeof(float));
for(int x160=0; x160 < 1; x160++) {
x159[x160] = 0.0f;

}
float* x164 = (float*)myMalloc(1 * sizeof(float));
for(int x165=0; x165 < 1; x165++) {
x164[x165] = 0.0f;

}
float* x169 = (float*)myMalloc(1 * sizeof(float));
for(int x170=0; x170 < 1; x170++) {
x169[x170] = 0.0f;

}
float** x693 = (float**)myMalloc(4 * sizeof(float*));
x693[0] = x164;
x693[1] = x169;
x693[2] = x49;
x693[3] = x84;
function<void(float**)> x677 = [&](float** x678) {
float* x679 = x678[0];
float* x680 = x678[1];
float* x681 = x678[2];
float* x682 = x678[3];
for(int x683=0; x683 < 50; x683++) {
float x684 = x681[x683];
x54[x683] = x684;

}
float x688 = x680[0];
x680[0] = 1.0f;
float x690 = x679[0];
x159[0] = x690;
};
x174(0,x677,x693);
float x700 = x159[0];
int32_t x701 = x120 % 100;
bool x702 = x701 == 0;
if (x702) {
printf("iter %d, loss %f\n",x120,x700);
end_0 = clock(); printf("Time elapsed: %f\n", (double)(end_0 - begin_0) / CLOCKS_PER_SEC);
} else {
}
for(int x707=0; x707 < 1300; x707++) {
float x708 = x59[x707];
bool x709 = x708 > 5.0f;
if (x709) {
x59[x707] = 5.0f;
} else {
}
float x713 = x59[x707];
bool x714 = x713 < -5.0f;
if (x714) {
x59[x707] = -5.0f;
} else {
}

}
float* x720 = (float*)myMalloc(1300 * sizeof(float));
for(int x721=0; x721 < 1300; x721++) {
float x722 = x59[x721];
float x723 = x59[x721];
float x724 = x722 * x723;
x720[x721] = x724;

}
for(int x728=0; x728 < 1300; x728++) {
float x729 = x89[x728];
float x730 = x720[x728];
float x731 = x729 + x730;
x89[x728] = x731;

}
float* x735 = (float*)myMalloc(1300 * sizeof(float));
for(int x736=0; x736 < 1300; x736++) {
float x737 = x59[x736];
float x738 = x737 * 0.1f;
x735[x736] = x738;

}
float* x742 = (float*)myMalloc(1300 * sizeof(float));
for(int x743=0; x743 < 1300; x743++) {
float x744 = x89[x743];
float x745 = x744 + 1.0E-8f;
x742[x743] = x745;

}
float* x749 = (float*)myMalloc(1300 * sizeof(float));
for(int x750=0; x750 < 1300; x750++) {
float x751 = x742[x750];
double x752 = (double)x751;
double x753 = sqrt(x752);
float x754 = (float)x753;
x749[x750] = x754;

}
float* x758 = (float*)myMalloc(1300 * sizeof(float));
for(int x759=0; x759 < 1300; x759++) {
float x760 = x735[x759];
float x761 = x749[x759];
float x762 = x760 / x761;
x758[x759] = x762;

}
for(int x766=0; x766 < 1300; x766++) {
float x767 = x14[x766];
float x768 = x758[x766];
float x769 = x767 - x768;
x14[x766] = x769;

}
for(int x773=0; x773 < 1300; x773++) {
float x774 = x59[x773];
x59[x773] = 0.0f;

}
for(int x778=0; x778 < 2500; x778++) {
float x779 = x64[x778];
bool x780 = x779 > 5.0f;
if (x780) {
x64[x778] = 5.0f;
} else {
}
float x784 = x64[x778];
bool x785 = x784 < -5.0f;
if (x785) {
x64[x778] = -5.0f;
} else {
}

}
float* x791 = (float*)myMalloc(2500 * sizeof(float));
for(int x792=0; x792 < 2500; x792++) {
float x793 = x64[x792];
float x794 = x64[x792];
float x795 = x793 * x794;
x791[x792] = x795;

}
for(int x799=0; x799 < 2500; x799++) {
float x800 = x94[x799];
float x801 = x791[x799];
float x802 = x800 + x801;
x94[x799] = x802;

}
float* x806 = (float*)myMalloc(2500 * sizeof(float));
for(int x807=0; x807 < 2500; x807++) {
float x808 = x64[x807];
float x809 = x808 * 0.1f;
x806[x807] = x809;

}
float* x813 = (float*)myMalloc(2500 * sizeof(float));
for(int x814=0; x814 < 2500; x814++) {
float x815 = x94[x814];
float x816 = x815 + 1.0E-8f;
x813[x814] = x816;

}
float* x820 = (float*)myMalloc(2500 * sizeof(float));
for(int x821=0; x821 < 2500; x821++) {
float x822 = x813[x821];
double x823 = (double)x822;
double x824 = sqrt(x823);
float x825 = (float)x824;
x820[x821] = x825;

}
float* x829 = (float*)myMalloc(2500 * sizeof(float));
for(int x830=0; x830 < 2500; x830++) {
float x831 = x806[x830];
float x832 = x820[x830];
float x833 = x831 / x832;
x829[x830] = x833;

}
for(int x837=0; x837 < 2500; x837++) {
float x838 = x22[x837];
float x839 = x829[x837];
float x840 = x838 - x839;
x22[x837] = x840;

}
for(int x844=0; x844 < 2500; x844++) {
float x845 = x64[x844];
x64[x844] = 0.0f;

}
for(int x849=0; x849 < 1300; x849++) {
float x850 = x69[x849];
bool x851 = x850 > 5.0f;
if (x851) {
x69[x849] = 5.0f;
} else {
}
float x855 = x69[x849];
bool x856 = x855 < -5.0f;
if (x856) {
x69[x849] = -5.0f;
} else {
}

}
float* x862 = (float*)myMalloc(1300 * sizeof(float));
for(int x863=0; x863 < 1300; x863++) {
float x864 = x69[x863];
float x865 = x69[x863];
float x866 = x864 * x865;
x862[x863] = x866;

}
for(int x870=0; x870 < 1300; x870++) {
float x871 = x99[x870];
float x872 = x862[x870];
float x873 = x871 + x872;
x99[x870] = x873;

}
float* x877 = (float*)myMalloc(1300 * sizeof(float));
for(int x878=0; x878 < 1300; x878++) {
float x879 = x69[x878];
float x880 = x879 * 0.1f;
x877[x878] = x880;

}
float* x884 = (float*)myMalloc(1300 * sizeof(float));
for(int x885=0; x885 < 1300; x885++) {
float x886 = x99[x885];
float x887 = x886 + 1.0E-8f;
x884[x885] = x887;

}
float* x891 = (float*)myMalloc(1300 * sizeof(float));
for(int x892=0; x892 < 1300; x892++) {
float x893 = x884[x892];
double x894 = (double)x893;
double x895 = sqrt(x894);
float x896 = (float)x895;
x891[x892] = x896;

}
float* x900 = (float*)myMalloc(1300 * sizeof(float));
for(int x901=0; x901 < 1300; x901++) {
float x902 = x877[x901];
float x903 = x891[x901];
float x904 = x902 / x903;
x900[x901] = x904;

}
for(int x908=0; x908 < 1300; x908++) {
float x909 = x30[x908];
float x910 = x900[x908];
float x911 = x909 - x910;
x30[x908] = x911;

}
for(int x915=0; x915 < 1300; x915++) {
float x916 = x69[x915];
x69[x915] = 0.0f;

}
for(int x920=0; x920 < 50; x920++) {
float x921 = x74[x920];
bool x922 = x921 > 5.0f;
if (x922) {
x74[x920] = 5.0f;
} else {
}
float x926 = x74[x920];
bool x927 = x926 < -5.0f;
if (x927) {
x74[x920] = -5.0f;
} else {
}

}
float* x933 = (float*)myMalloc(50 * sizeof(float));
for(int x934=0; x934 < 50; x934++) {
float x935 = x74[x934];
float x936 = x74[x934];
float x937 = x935 * x936;
x933[x934] = x937;

}
for(int x941=0; x941 < 50; x941++) {
float x942 = x104[x941];
float x943 = x933[x941];
float x944 = x942 + x943;
x104[x941] = x944;

}
float* x948 = (float*)myMalloc(50 * sizeof(float));
for(int x949=0; x949 < 50; x949++) {
float x950 = x74[x949];
float x951 = x950 * 0.1f;
x948[x949] = x951;

}
float* x955 = (float*)myMalloc(50 * sizeof(float));
for(int x956=0; x956 < 50; x956++) {
float x957 = x104[x956];
float x958 = x957 + 1.0E-8f;
x955[x956] = x958;

}
float* x962 = (float*)myMalloc(50 * sizeof(float));
for(int x963=0; x963 < 50; x963++) {
float x964 = x955[x963];
double x965 = (double)x964;
double x966 = sqrt(x965);
float x967 = (float)x966;
x962[x963] = x967;

}
float* x971 = (float*)myMalloc(50 * sizeof(float));
for(int x972=0; x972 < 50; x972++) {
float x973 = x948[x972];
float x974 = x962[x972];
float x975 = x973 / x974;
x971[x972] = x975;

}
for(int x979=0; x979 < 50; x979++) {
float x980 = x37[x979];
float x981 = x971[x979];
float x982 = x980 - x981;
x37[x979] = x982;

}
for(int x986=0; x986 < 50; x986++) {
float x987 = x74[x986];
x74[x986] = 0.0f;

}
for(int x991=0; x991 < 26; x991++) {
float x992 = x79[x991];
bool x993 = x992 > 5.0f;
if (x993) {
x79[x991] = 5.0f;
} else {
}
float x997 = x79[x991];
bool x998 = x997 < -5.0f;
if (x998) {
x79[x991] = -5.0f;
} else {
}

}
float* x1004 = (float*)myMalloc(26 * sizeof(float));
for(int x1005=0; x1005 < 26; x1005++) {
float x1006 = x79[x1005];
float x1007 = x79[x1005];
float x1008 = x1006 * x1007;
x1004[x1005] = x1008;

}
for(int x1012=0; x1012 < 26; x1012++) {
float x1013 = x109[x1012];
float x1014 = x1004[x1012];
float x1015 = x1013 + x1014;
x109[x1012] = x1015;

}
float* x1019 = (float*)myMalloc(26 * sizeof(float));
for(int x1020=0; x1020 < 26; x1020++) {
float x1021 = x79[x1020];
float x1022 = x1021 * 0.1f;
x1019[x1020] = x1022;

}
float* x1026 = (float*)myMalloc(26 * sizeof(float));
for(int x1027=0; x1027 < 26; x1027++) {
float x1028 = x109[x1027];
float x1029 = x1028 + 1.0E-8f;
x1026[x1027] = x1029;

}
float* x1033 = (float*)myMalloc(26 * sizeof(float));
for(int x1034=0; x1034 < 26; x1034++) {
float x1035 = x1026[x1034];
double x1036 = (double)x1035;
double x1037 = sqrt(x1036);
float x1038 = (float)x1037;
x1033[x1034] = x1038;

}
float* x1042 = (float*)myMalloc(26 * sizeof(float));
for(int x1043=0; x1043 < 26; x1043++) {
float x1044 = x1019[x1043];
float x1045 = x1033[x1043];
float x1046 = x1044 / x1045;
x1042[x1043] = x1046;

}
for(int x1050=0; x1050 < 26; x1050++) {
float x1051 = x43[x1050];
float x1052 = x1042[x1050];
float x1053 = x1051 - x1052;
x43[x1050] = x1053;

}
for(int x1057=0; x1057 < 26; x1057++) {
float x1058 = x79[x1057];
x79[x1057] = 0.0f;

}
for(int x1062=0; x1062 < 50; x1062++) {
float x1063 = x84[x1062];
x84[x1062] = 0.0f;

}
for(int x1067=0; x1067 < 50; x1067++) {
float x1068 = x54[x1067];
x49[x1067] = x1068;

}
mallocAddr = (void*)x114;

}
}
/*****************************************
  End of C Generated Code                  
*******************************************/

