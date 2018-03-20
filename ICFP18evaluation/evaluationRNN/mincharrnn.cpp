
      #include <fcntl.h>
      #include <errno.h>
      #include <err.h>
      #include <sys/mman.h>
      #include <sys/stat.h>
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
double* x14 = (double*)myMalloc(1300 * sizeof(double));
for(int x16=0; x16 < 1300; x16++) {
double x17 = d(gen);
double x18 = x17 * 0.01;
x14[x16] = x18;

}
double* x22 = (double*)myMalloc(2500 * sizeof(double));
for(int x24=0; x24 < 2500; x24++) {
double x25 = d(gen);
double x26 = x25 * 0.01;
x22[x24] = x26;

}
double* x30 = (double*)myMalloc(1300 * sizeof(double));
for(int x31=0; x31 < 1300; x31++) {
double x32 = d(gen);
double x33 = x32 * 0.01;
x30[x31] = x33;

}
double* x37 = (double*)myMalloc(50 * sizeof(double));
for(int x39=0; x39 < 50; x39++) {
x37[x39] = 0.0;

}
double* x43 = (double*)myMalloc(26 * sizeof(double));
for(int x45=0; x45 < 26; x45++) {
x43[x45] = 0.0;

}
double* x49 = (double*)myMalloc(50 * sizeof(double));
for(int x50=0; x50 < 50; x50++) {
x49[x50] = 0.0;

}
double* x54 = (double*)myMalloc(50 * sizeof(double));
for(int x55=0; x55 < 50; x55++) {
x54[x55] = 0.0;

}
double* x59 = (double*)myMalloc(1300 * sizeof(double));
for(int x60=0; x60 < 1300; x60++) {
x59[x60] = 0.0;

}
double* x64 = (double*)myMalloc(2500 * sizeof(double));
for(int x65=0; x65 < 2500; x65++) {
x64[x65] = 0.0;

}
double* x69 = (double*)myMalloc(1300 * sizeof(double));
for(int x70=0; x70 < 1300; x70++) {
x69[x70] = 0.0;

}
double* x74 = (double*)myMalloc(50 * sizeof(double));
for(int x75=0; x75 < 50; x75++) {
x74[x75] = 0.0;

}
double* x79 = (double*)myMalloc(26 * sizeof(double));
for(int x80=0; x80 < 26; x80++) {
x79[x80] = 0.0;

}
double* x84 = (double*)myMalloc(50 * sizeof(double));
for(int x85=0; x85 < 50; x85++) {
x84[x85] = 0.0;

}
double* x89 = (double*)myMalloc(1 * sizeof(double));
for(int x91=0; x91 < 1; x91++) {
x89[x91] = 0.1;

}
double* x95 = (double*)myMalloc(1 * sizeof(double));
for(int x96=0; x96 < 1; x96++) {
x95[x96] = 1.0E-8;

}
double* x100 = (double*)myMalloc(1300 * sizeof(double));
for(int x101=0; x101 < 1300; x101++) {
x100[x101] = 0.0;

}
double* x105 = (double*)myMalloc(2500 * sizeof(double));
for(int x106=0; x106 < 2500; x106++) {
x105[x106] = 0.0;

}
double* x110 = (double*)myMalloc(1300 * sizeof(double));
for(int x111=0; x111 < 1300; x111++) {
x110[x111] = 0.0;

}
double* x115 = (double*)myMalloc(50 * sizeof(double));
for(int x116=0; x116 < 50; x116++) {
x115[x116] = 0.0;

}
double* x120 = (double*)myMalloc(26 * sizeof(double));
for(int x121=0; x121 < 26; x121++) {
x120[x121] = 0.0;

}
int64_t x125 = (long)mallocAddr;
int32_t x126 = 0;
x126 -= 20;
clock_t begin_0, end_0; double time_spent_0;
begin_0 = clock();
double x130 = 60.0;
for(int x132=0; x132 < 2001; x132++) {
double* x168 = (double*)myMalloc(1 * sizeof(double));
int32_t* x145 = (int32_t*)myMalloc(20 * sizeof(int32_t));
int32_t* x146 = (int32_t*)myMalloc(20 * sizeof(int32_t));
function<void(int32_t,double**)> x183 = [&](int32_t x184,double** x185) {
int32_t x186 = x184;
bool x192 = x186 < 20;
if (x192) {
double* x193 = (double*)myMalloc(26 * sizeof(double));
for(int x194=0; x194 < 26; x194++) {
x193[x194] = 0.0;

}
int32_t x198 = x145[x186];
x193[x198] = 1.0;
double* x200 = (double*)myMalloc(26 * sizeof(double));
for(int x201=0; x201 < 26; x201++) {
x200[x201] = 0.0;

}
double* x205 = (double*)myMalloc(26 * sizeof(double));
for(int x206=0; x206 < 26; x206++) {
x205[x206] = 0.0;

}
int32_t x210 = x146[x186];
x205[x210] = 1.0;
double* x212 = (double*)myMalloc(26 * sizeof(double));
for(int x213=0; x213 < 26; x213++) {
x212[x213] = 0.0;

}
double* x217 = (double*)myMalloc(50 * sizeof(double));
for(int x218=0; x218 < 50; x218++) {
double x219 = 0.0;
int32_t x221 = 26 * x218;
for(int x220=0; x220 < 26; x220++) {
int32_t x222 = x220 + x221;
double x223 = x14[x222];
double x224 = x193[x220];
double x225 = x223 * x224;
x219 += x225;

}
double x229 = x219;
x217[x218] = x229;

}
double* x233 = (double*)myMalloc(50 * sizeof(double));
for(int x234=0; x234 < 50; x234++) {
x233[x234] = 0.0;

}
double* x238 = (double*)myMalloc(50 * sizeof(double));
double** x187 = x185;
double* x190 = x187[2];
for(int x239=0; x239 < 50; x239++) {
double x240 = 0.0;
int32_t x242 = 50 * x239;
for(int x241=0; x241 < 50; x241++) {
int32_t x243 = x241 + x242;
double x244 = x22[x243];
double x245 = x190[x241];
double x246 = x244 * x245;
x240 += x246;

}
double x250 = x240;
x238[x239] = x250;

}
double* x254 = (double*)myMalloc(50 * sizeof(double));
for(int x255=0; x255 < 50; x255++) {
x254[x255] = 0.0;

}
double* x259 = (double*)myMalloc(50 * sizeof(double));
for(int x260=0; x260 < 50; x260++) {
double x261 = x217[x260];
double x262 = x238[x260];
double x263 = x261 + x262;
x259[x260] = x263;

}
double* x267 = (double*)myMalloc(50 * sizeof(double));
for(int x268=0; x268 < 50; x268++) {
x267[x268] = 0.0;

}
double* x272 = (double*)myMalloc(50 * sizeof(double));
for(int x273=0; x273 < 50; x273++) {
double x274 = x259[x273];
double x275 = x37[x273];
double x276 = x274 + x275;
x272[x273] = x276;

}
double* x280 = (double*)myMalloc(50 * sizeof(double));
for(int x281=0; x281 < 50; x281++) {
x280[x281] = 0.0;

}
double* x285 = (double*)myMalloc(50 * sizeof(double));
for(int x286=0; x286 < 50; x286++) {
double x287 = x272[x286];
double x288 = tanh(x287);
x285[x286] = x288;

}
double* x292 = (double*)myMalloc(50 * sizeof(double));
for(int x293=0; x293 < 50; x293++) {
x292[x293] = 0.0;

}
double* x297 = (double*)myMalloc(26 * sizeof(double));
for(int x298=0; x298 < 26; x298++) {
double x299 = 0.0;
int32_t x301 = 50 * x298;
for(int x300=0; x300 < 50; x300++) {
int32_t x302 = x300 + x301;
double x303 = x30[x302];
double x304 = x285[x300];
double x305 = x303 * x304;
x299 += x305;

}
double x309 = x299;
x297[x298] = x309;

}
double* x313 = (double*)myMalloc(26 * sizeof(double));
for(int x314=0; x314 < 26; x314++) {
x313[x314] = 0.0;

}
double* x318 = (double*)myMalloc(26 * sizeof(double));
for(int x319=0; x319 < 26; x319++) {
double x320 = x297[x319];
double x321 = x43[x319];
double x322 = x320 + x321;
x318[x319] = x322;

}
double* x326 = (double*)myMalloc(26 * sizeof(double));
for(int x327=0; x327 < 26; x327++) {
x326[x327] = 0.0;

}
double* x331 = (double*)myMalloc(26 * sizeof(double));
for(int x332=0; x332 < 26; x332++) {
double x333 = x318[x332];
double x334 = exp(x333);
x331[x332] = x334;

}
double* x338 = (double*)myMalloc(26 * sizeof(double));
for(int x339=0; x339 < 26; x339++) {
x338[x339] = 0.0;

}
double x343 = 0.0;
for(int x344=0; x344 < 26; x344++) {
double x345 = x331[x344];
x343 += x345;

}
double* x349 = (double*)myMalloc(1 * sizeof(double));
double x350 = x343;
x349[0] = x350;
double* x352 = (double*)myMalloc(1 * sizeof(double));
for(int x353=0; x353 < 1; x353++) {
x352[x353] = 0.0;

}
double* x357 = (double*)myMalloc(26 * sizeof(double));
for(int x358=0; x358 < 26; x358++) {
double x359 = x331[x358];
double x360 = x349[0];
double x361 = x359 / x360;
x357[x358] = x361;

}
double* x365 = (double*)myMalloc(26 * sizeof(double));
for(int x366=0; x366 < 26; x366++) {
x365[x366] = 0.0;

}
double* x370 = (double*)myMalloc(1 * sizeof(double));
for(int x371=0; x371 < 1; x371++) {
double x372 = 0.0;
int32_t x374 = 26 * x371;
for(int x373=0; x373 < 26; x373++) {
int32_t x375 = x373 + x374;
double x376 = x357[x375];
double x377 = x205[x373];
double x378 = x376 * x377;
x372 += x378;

}
double x382 = x372;
x370[x371] = x382;

}
double* x386 = (double*)myMalloc(1 * sizeof(double));
for(int x387=0; x387 < 1; x387++) {
x386[x387] = 0.0;

}
double* x391 = (double*)myMalloc(1 * sizeof(double));
for(int x392=0; x392 < 1; x392++) {
double x393 = x370[x392];
double x394 = log(x393);
x391[x392] = x394;

}
double* x398 = (double*)myMalloc(1 * sizeof(double));
for(int x399=0; x399 < 1; x399++) {
x398[x399] = 0.0;

}
double* x403 = (double*)myMalloc(1 * sizeof(double));
double* x188 = x187[0];
for(int x404=0; x404 < 1; x404++) {
double x406 = x391[x404];
double x405 = x188[x404];
double x407 = x405 - x406;
x403[x404] = x407;

}
double* x411 = (double*)myMalloc(1 * sizeof(double));
for(int x412=0; x412 < 1; x412++) {
x411[x412] = 0.0;

}
double** x417 = (double**)myMalloc(4 * sizeof(double*));
x417[0] = x403;
x417[1] = x411;
x417[2] = x285;
x417[3] = x292;
int32_t x416 = x186 + 1;
x183(x416,x417);
double* x189 = x187[1];
for(int x424=0; x424 < 1; x424++) {
double x426 = x411[x424];
double x425 = x189[x424];
double x427 = x425 + x426;
x189[x424] = x427;

}
for(int x431=0; x431 < 1; x431++) {
double x432 = x398[x431];
double x433 = x411[x431];
double x434 = x432 - x433;
x398[x431] = x434;

}
for(int x438=0; x438 < 1; x438++) {
double x439 = x386[0];
double x440 = x398[0];
double x441 = x370[0];
double x442 = x440 / x441;
double x443 = x439 + x442;
x386[0] = x443;

}
for(int x447=0; x447 < 1; x447++) {
int32_t x449 = 26 * x447;
for(int x448=0; x448 < 26; x448++) {
int32_t x450 = x449 + x448;
double x451 = x365[x450];
double x452 = x205[x448];
double x453 = x386[x447];
double x454 = x452 * x453;
double x455 = x451 + x454;
x365[x450] = x455;

}

}
for(int x461=0; x461 < 1; x461++) {
int32_t x464 = 26 * x461;
for(int x462=0; x462 < 26; x462++) {
double x463 = x212[x462];
int32_t x465 = x464 + x462;
double x466 = x357[x465];
double x467 = x386[x461];
double x468 = x466 * x467;
double x469 = x463 + x468;
x212[x462] = x469;

}

}
for(int x475=0; x475 < 26; x475++) {
double x476 = x338[x475];
double x477 = x365[x475];
double x478 = x349[0];
double x479 = x477 / x478;
double x480 = x476 + x479;
x338[x475] = x480;

}
for(int x484=0; x484 < 26; x484++) {
double x485 = x352[0];
double x486 = x331[x484];
double x487 = x365[x484];
double x489 = x349[0];
double x488 = x486 * x487;
double x490 = x489 * x489;
double x491 = x488 / x490;
double x492 = x485 - x491;
x352[0] = x492;

}
for(int x496=0; x496 < 26; x496++) {
double x497 = x338[x496];
double x498 = x352[0];
double x499 = x497 + x498;
x338[x496] = x499;

}
for(int x503=0; x503 < 26; x503++) {
double x504 = x326[x503];
double x505 = x331[x503];
double x506 = x338[x503];
double x507 = x505 * x506;
double x508 = x504 + x507;
x326[x503] = x508;

}
for(int x512=0; x512 < 26; x512++) {
double x513 = x313[x512];
double x514 = x326[x512];
double x515 = x513 + x514;
x313[x512] = x515;

}
for(int x519=0; x519 < 26; x519++) {
double x520 = x79[x519];
double x521 = x326[x519];
double x522 = x520 + x521;
x79[x519] = x522;

}
for(int x526=0; x526 < 26; x526++) {
int32_t x528 = 50 * x526;
for(int x527=0; x527 < 50; x527++) {
int32_t x529 = x528 + x527;
double x530 = x69[x529];
double x531 = x285[x527];
double x532 = x313[x526];
double x533 = x531 * x532;
double x534 = x530 + x533;
x69[x529] = x534;

}

}
for(int x540=0; x540 < 26; x540++) {
int32_t x543 = 50 * x540;
for(int x541=0; x541 < 50; x541++) {
double x542 = x292[x541];
int32_t x544 = x543 + x541;
double x545 = x30[x544];
double x546 = x313[x540];
double x547 = x545 * x546;
double x548 = x542 + x547;
x292[x541] = x548;

}

}
for(int x554=0; x554 < 50; x554++) {
double x555 = x280[x554];
double x556 = x285[x554];
double x559 = x292[x554];
double x557 = x556 * x556;
double x558 = 1.0 - x557;
double x560 = x558 * x559;
double x561 = x555 + x560;
x280[x554] = x561;

}
for(int x565=0; x565 < 50; x565++) {
double x566 = x267[x565];
double x567 = x280[x565];
double x568 = x566 + x567;
x267[x565] = x568;

}
for(int x572=0; x572 < 50; x572++) {
double x573 = x74[x572];
double x574 = x280[x572];
double x575 = x573 + x574;
x74[x572] = x575;

}
for(int x579=0; x579 < 50; x579++) {
double x580 = x233[x579];
double x581 = x267[x579];
double x582 = x580 + x581;
x233[x579] = x582;

}
for(int x586=0; x586 < 50; x586++) {
double x587 = x254[x586];
double x588 = x267[x586];
double x589 = x587 + x588;
x254[x586] = x589;

}
for(int x593=0; x593 < 50; x593++) {
int32_t x595 = 50 * x593;
for(int x594=0; x594 < 50; x594++) {
int32_t x596 = x595 + x594;
double x597 = x64[x596];
double x599 = x254[x593];
double x598 = x190[x594];
double x600 = x598 * x599;
double x601 = x597 + x600;
x64[x596] = x601;

}

}
double* x191 = x187[3];
for(int x607=0; x607 < 50; x607++) {
int32_t x610 = 50 * x607;
for(int x608=0; x608 < 50; x608++) {
int32_t x611 = x610 + x608;
double x612 = x22[x611];
double x613 = x254[x607];
double x609 = x191[x608];
double x614 = x612 * x613;
double x615 = x609 + x614;
x191[x608] = x615;

}

}
for(int x621=0; x621 < 50; x621++) {
int32_t x623 = 26 * x621;
for(int x622=0; x622 < 26; x622++) {
int32_t x624 = x623 + x622;
double x625 = x59[x624];
double x626 = x193[x622];
double x627 = x233[x621];
double x628 = x626 * x627;
double x629 = x625 + x628;
x59[x624] = x629;

}

}
for(int x635=0; x635 < 50; x635++) {
int32_t x638 = 26 * x635;
for(int x636=0; x636 < 26; x636++) {
double x637 = x200[x636];
int32_t x639 = x638 + x636;
double x640 = x14[x639];
double x641 = x233[x635];
double x642 = x640 * x641;
double x643 = x637 + x642;
x200[x636] = x643;

}

}
} else {
double** x187 = x185;
double* x190 = x187[2];
for(int x650=0; x650 < 50; x650++) {
double x651 = x190[x650];
x54[x650] = x651;

}
double* x189 = x187[1];
for(int x655=0; x655 < 1; x655++) {
x189[x655] = 1.0;

}
double* x188 = x187[0];
for(int x659=0; x659 < 1; x659++) {
double x660 = x188[x659];
x168[x659] = x660;

}
}
};
x126 += 20;
int32_t x134 = x126;
int32_t x135 = x134 + 20;
int32_t x136 = x135 + 1;
bool x137 = x136 >= x2;
if (x137) {
x126 = 0;
for(int x139=0; x139 < 50; x139++) {
x49[x139] = 0.0;

}
} else {
}
for(int x148=0; x148 < 20; x148++) {
int32_t x149 = x126;
int32_t x150 = x149 + x148;
int32_t x151 = x5[x150];
x145[x148] = x151;
int32_t x153 = x150 + 1;
int32_t x154 = x5[x153];
x146[x148] = x154;

}
double* x158 = (double*)myMalloc(1 * sizeof(double));
for(int x159=0; x159 < 1; x159++) {
x158[x159] = 0.0;

}
double* x163 = (double*)myMalloc(1 * sizeof(double));
for(int x164=0; x164 < 1; x164++) {
x163[x164] = 0.0;

}
for(int x169=0; x169 < 1; x169++) {
x168[x169] = 0.0;

}
double* x173 = (double*)myMalloc(1 * sizeof(double));
for(int x174=0; x174 < 1; x174++) {
x173[x174] = 0.0;

}
double* x178 = (double*)myMalloc(1 * sizeof(double));
for(int x179=0; x179 < 1; x179++) {
x178[x179] = 0.0;

}
double** x667 = (double**)myMalloc(4 * sizeof(double*));
x667[0] = x173;
x667[1] = x178;
x667[2] = x49;
x667[3] = x84;
x183(0,x667);
double x674 = x168[0];
double x675 = x130;
double x676 = x675 * 0.9;
double x677 = x674 * 0.1;
double x678 = x676 + x677;
x130 = x678;
int32_t x680 = x132 % 100;
bool x681 = x680 == 0;
if (x681) {
double x682 = x130;
printf("iter %d, loss %f\n",x132,x682);
} else {
}
for(int x686=0; x686 < 1300; x686++) {
double x687 = x59[x686];
bool x688 = x687 > 5.0;
if (x688) {
x59[x686] = 5.0;
} else {
}
bool x692 = x687 < -5.0;
if (x692) {
x59[x686] = -5.0;
} else {
}

}
double* x698 = (double*)myMalloc(1300 * sizeof(double));
for(int x699=0; x699 < 1300; x699++) {
double x700 = x59[x699];
double x701 = x700 * x700;
x698[x699] = x701;

}
for(int x705=0; x705 < 1300; x705++) {
double x706 = x100[x705];
double x707 = x698[x705];
double x708 = x706 + x707;
x100[x705] = x708;

}
double* x712 = (double*)myMalloc(1300 * sizeof(double));
for(int x713=0; x713 < 1300; x713++) {
double x714 = x59[x713];
double x715 = x89[0];
double x716 = x714 * x715;
x712[x713] = x716;

}
double* x720 = (double*)myMalloc(1300 * sizeof(double));
for(int x721=0; x721 < 1300; x721++) {
double x722 = x100[x721];
double x723 = x95[0];
double x724 = x722 + x723;
x720[x721] = x724;

}
double* x728 = (double*)myMalloc(1300 * sizeof(double));
for(int x729=0; x729 < 1300; x729++) {
double x730 = x720[x729];
double x731 = sqrt(x730);
x728[x729] = x731;

}
double* x735 = (double*)myMalloc(1300 * sizeof(double));
for(int x736=0; x736 < 1300; x736++) {
double x737 = x712[x736];
double x738 = x728[x736];
double x739 = x737 / x738;
x735[x736] = x739;

}
for(int x743=0; x743 < 1300; x743++) {
double x744 = x14[x743];
double x745 = x735[x743];
double x746 = x744 - x745;
x14[x743] = x746;

}
for(int x750=0; x750 < 1300; x750++) {
x59[x750] = 0.0;

}
for(int x754=0; x754 < 2500; x754++) {
double x755 = x64[x754];
bool x756 = x755 > 5.0;
if (x756) {
x64[x754] = 5.0;
} else {
}
bool x760 = x755 < -5.0;
if (x760) {
x64[x754] = -5.0;
} else {
}

}
double* x766 = (double*)myMalloc(2500 * sizeof(double));
for(int x767=0; x767 < 2500; x767++) {
double x768 = x64[x767];
double x769 = x768 * x768;
x766[x767] = x769;

}
for(int x773=0; x773 < 2500; x773++) {
double x774 = x105[x773];
double x775 = x766[x773];
double x776 = x774 + x775;
x105[x773] = x776;

}
double* x780 = (double*)myMalloc(2500 * sizeof(double));
for(int x781=0; x781 < 2500; x781++) {
double x782 = x64[x781];
double x783 = x89[0];
double x784 = x782 * x783;
x780[x781] = x784;

}
double* x788 = (double*)myMalloc(2500 * sizeof(double));
for(int x789=0; x789 < 2500; x789++) {
double x790 = x105[x789];
double x791 = x95[0];
double x792 = x790 + x791;
x788[x789] = x792;

}
double* x796 = (double*)myMalloc(2500 * sizeof(double));
for(int x797=0; x797 < 2500; x797++) {
double x798 = x788[x797];
double x799 = sqrt(x798);
x796[x797] = x799;

}
double* x803 = (double*)myMalloc(2500 * sizeof(double));
for(int x804=0; x804 < 2500; x804++) {
double x805 = x780[x804];
double x806 = x796[x804];
double x807 = x805 / x806;
x803[x804] = x807;

}
for(int x811=0; x811 < 2500; x811++) {
double x812 = x22[x811];
double x813 = x803[x811];
double x814 = x812 - x813;
x22[x811] = x814;

}
for(int x818=0; x818 < 2500; x818++) {
x64[x818] = 0.0;

}
for(int x822=0; x822 < 1300; x822++) {
double x823 = x69[x822];
bool x824 = x823 > 5.0;
if (x824) {
x69[x822] = 5.0;
} else {
}
bool x828 = x823 < -5.0;
if (x828) {
x69[x822] = -5.0;
} else {
}

}
double* x834 = (double*)myMalloc(1300 * sizeof(double));
for(int x835=0; x835 < 1300; x835++) {
double x836 = x69[x835];
double x837 = x836 * x836;
x834[x835] = x837;

}
for(int x841=0; x841 < 1300; x841++) {
double x842 = x110[x841];
double x843 = x834[x841];
double x844 = x842 + x843;
x110[x841] = x844;

}
double* x848 = (double*)myMalloc(1300 * sizeof(double));
for(int x849=0; x849 < 1300; x849++) {
double x850 = x69[x849];
double x851 = x89[0];
double x852 = x850 * x851;
x848[x849] = x852;

}
double* x856 = (double*)myMalloc(1300 * sizeof(double));
for(int x857=0; x857 < 1300; x857++) {
double x858 = x110[x857];
double x859 = x95[0];
double x860 = x858 + x859;
x856[x857] = x860;

}
double* x864 = (double*)myMalloc(1300 * sizeof(double));
for(int x865=0; x865 < 1300; x865++) {
double x866 = x856[x865];
double x867 = sqrt(x866);
x864[x865] = x867;

}
double* x871 = (double*)myMalloc(1300 * sizeof(double));
for(int x872=0; x872 < 1300; x872++) {
double x873 = x848[x872];
double x874 = x864[x872];
double x875 = x873 / x874;
x871[x872] = x875;

}
for(int x879=0; x879 < 1300; x879++) {
double x880 = x30[x879];
double x881 = x871[x879];
double x882 = x880 - x881;
x30[x879] = x882;

}
for(int x886=0; x886 < 1300; x886++) {
x69[x886] = 0.0;

}
for(int x890=0; x890 < 50; x890++) {
double x891 = x74[x890];
bool x892 = x891 > 5.0;
if (x892) {
x74[x890] = 5.0;
} else {
}
bool x896 = x891 < -5.0;
if (x896) {
x74[x890] = -5.0;
} else {
}

}
double* x902 = (double*)myMalloc(50 * sizeof(double));
for(int x903=0; x903 < 50; x903++) {
double x904 = x74[x903];
double x905 = x904 * x904;
x902[x903] = x905;

}
for(int x909=0; x909 < 50; x909++) {
double x910 = x115[x909];
double x911 = x902[x909];
double x912 = x910 + x911;
x115[x909] = x912;

}
double* x916 = (double*)myMalloc(50 * sizeof(double));
for(int x917=0; x917 < 50; x917++) {
double x918 = x74[x917];
double x919 = x89[0];
double x920 = x918 * x919;
x916[x917] = x920;

}
double* x924 = (double*)myMalloc(50 * sizeof(double));
for(int x925=0; x925 < 50; x925++) {
double x926 = x115[x925];
double x927 = x95[0];
double x928 = x926 + x927;
x924[x925] = x928;

}
double* x932 = (double*)myMalloc(50 * sizeof(double));
for(int x933=0; x933 < 50; x933++) {
double x934 = x924[x933];
double x935 = sqrt(x934);
x932[x933] = x935;

}
double* x939 = (double*)myMalloc(50 * sizeof(double));
for(int x940=0; x940 < 50; x940++) {
double x941 = x916[x940];
double x942 = x932[x940];
double x943 = x941 / x942;
x939[x940] = x943;

}
for(int x947=0; x947 < 50; x947++) {
double x948 = x37[x947];
double x949 = x939[x947];
double x950 = x948 - x949;
x37[x947] = x950;

}
for(int x954=0; x954 < 50; x954++) {
x74[x954] = 0.0;

}
for(int x958=0; x958 < 26; x958++) {
double x959 = x79[x958];
bool x960 = x959 > 5.0;
if (x960) {
x79[x958] = 5.0;
} else {
}
bool x964 = x959 < -5.0;
if (x964) {
x79[x958] = -5.0;
} else {
}

}
double* x970 = (double*)myMalloc(26 * sizeof(double));
for(int x971=0; x971 < 26; x971++) {
double x972 = x79[x971];
double x973 = x972 * x972;
x970[x971] = x973;

}
for(int x977=0; x977 < 26; x977++) {
double x978 = x120[x977];
double x979 = x970[x977];
double x980 = x978 + x979;
x120[x977] = x980;

}
double* x984 = (double*)myMalloc(26 * sizeof(double));
for(int x985=0; x985 < 26; x985++) {
double x986 = x79[x985];
double x987 = x89[0];
double x988 = x986 * x987;
x984[x985] = x988;

}
double* x992 = (double*)myMalloc(26 * sizeof(double));
for(int x993=0; x993 < 26; x993++) {
double x994 = x120[x993];
double x995 = x95[0];
double x996 = x994 + x995;
x992[x993] = x996;

}
double* x1000 = (double*)myMalloc(26 * sizeof(double));
for(int x1001=0; x1001 < 26; x1001++) {
double x1002 = x992[x1001];
double x1003 = sqrt(x1002);
x1000[x1001] = x1003;

}
double* x1007 = (double*)myMalloc(26 * sizeof(double));
for(int x1008=0; x1008 < 26; x1008++) {
double x1009 = x984[x1008];
double x1010 = x1000[x1008];
double x1011 = x1009 / x1010;
x1007[x1008] = x1011;

}
for(int x1015=0; x1015 < 26; x1015++) {
double x1016 = x43[x1015];
double x1017 = x1007[x1015];
double x1018 = x1016 - x1017;
x43[x1015] = x1018;

}
for(int x1022=0; x1022 < 26; x1022++) {
x79[x1022] = 0.0;

}
for(int x1026=0; x1026 < 50; x1026++) {
x84[x1026] = 0.0;

}
for(int x1030=0; x1030 < 50; x1030++) {
double x1031 = x54[x1030];
x49[x1030] = x1031;

}
mallocAddr = (void*)x125;

}
}
/*****************************************
  End of C Generated Code                  
*******************************************/

