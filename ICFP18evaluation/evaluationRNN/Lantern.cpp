
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
double* x90 = (double*)myMalloc(1 * sizeof(double));
for(int x92=0; x92 < 1; x92++) {
x90[x92] = 0.1;

}
double* x96 = (double*)myMalloc(1 * sizeof(double));
for(int x97=0; x97 < 1; x97++) {
x96[x97] = 1.0E-8;

}
double* x101 = (double*)myMalloc(1300 * sizeof(double));
for(int x102=0; x102 < 1300; x102++) {
x101[x102] = 0.0;

}
double* x106 = (double*)myMalloc(2500 * sizeof(double));
for(int x107=0; x107 < 2500; x107++) {
x106[x107] = 0.0;

}
double* x111 = (double*)myMalloc(1300 * sizeof(double));
for(int x112=0; x112 < 1300; x112++) {
x111[x112] = 0.0;

}
double* x116 = (double*)myMalloc(50 * sizeof(double));
for(int x117=0; x117 < 50; x117++) {
x116[x117] = 0.0;

}
double* x121 = (double*)myMalloc(26 * sizeof(double));
for(int x122=0; x122 < 26; x122++) {
x121[x122] = 0.0;

}
double* x126 = (double*)myMalloc(51 * sizeof(double));
double x127 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x128 = (long)mallocAddr;
int32_t x129 = 0;
x129 -= 20;
double x131 = 70.0;
for(int x133=0; x133 < 5001; x133++) {
double* x169 = (double*)myMalloc(1 * sizeof(double));
int32_t* x146 = (int32_t*)myMalloc(20 * sizeof(int32_t));
int32_t* x147 = (int32_t*)myMalloc(20 * sizeof(int32_t));
function<void(int32_t,double**)> x184 = [&](int32_t x185,double** x186) {
int32_t x187 = x185;
bool x193 = x187 < 20;
if (x193) {
double* x194 = (double*)myMalloc(26 * sizeof(double));
for(int x195=0; x195 < 26; x195++) {
x194[x195] = 0.0;

}
int32_t x199 = x146[x187];
x194[x199] = 1.0;
double* x201 = (double*)myMalloc(26 * sizeof(double));
for(int x202=0; x202 < 26; x202++) {
x201[x202] = 0.0;

}
double* x206 = (double*)myMalloc(26 * sizeof(double));
for(int x207=0; x207 < 26; x207++) {
x206[x207] = 0.0;

}
int32_t x211 = x147[x187];
x206[x211] = 1.0;
double* x213 = (double*)myMalloc(26 * sizeof(double));
for(int x214=0; x214 < 26; x214++) {
x213[x214] = 0.0;

}
double* x218 = (double*)myMalloc(50 * sizeof(double));
for(int x219=0; x219 < 50; x219++) {
double x220 = 0.0;
int32_t x222 = 26 * x219;
for(int x221=0; x221 < 26; x221++) {
int32_t x223 = x221 + x222;
double x224 = x15[x223];
double x225 = x194[x221];
double x226 = x224 * x225;
x220 += x226;

}
double x230 = x220;
x218[x219] = x230;

}
double* x234 = (double*)myMalloc(50 * sizeof(double));
for(int x235=0; x235 < 50; x235++) {
x234[x235] = 0.0;

}
double* x239 = (double*)myMalloc(50 * sizeof(double));
double** x188 = x186;
double* x191 = x188[2];
for(int x240=0; x240 < 50; x240++) {
double x241 = 0.0;
int32_t x243 = 50 * x240;
for(int x242=0; x242 < 50; x242++) {
int32_t x244 = x242 + x243;
double x245 = x23[x244];
double x246 = x191[x242];
double x247 = x245 * x246;
x241 += x247;

}
double x251 = x241;
x239[x240] = x251;

}
double* x255 = (double*)myMalloc(50 * sizeof(double));
for(int x256=0; x256 < 50; x256++) {
x255[x256] = 0.0;

}
double* x260 = (double*)myMalloc(50 * sizeof(double));
for(int x261=0; x261 < 50; x261++) {
double x262 = x218[x261];
double x263 = x239[x261];
double x264 = x262 + x263;
x260[x261] = x264;

}
double* x268 = (double*)myMalloc(50 * sizeof(double));
for(int x269=0; x269 < 50; x269++) {
x268[x269] = 0.0;

}
double* x273 = (double*)myMalloc(50 * sizeof(double));
for(int x274=0; x274 < 50; x274++) {
double x275 = x260[x274];
double x276 = x38[x274];
double x277 = x275 + x276;
x273[x274] = x277;

}
double* x281 = (double*)myMalloc(50 * sizeof(double));
for(int x282=0; x282 < 50; x282++) {
x281[x282] = 0.0;

}
double* x286 = (double*)myMalloc(50 * sizeof(double));
for(int x287=0; x287 < 50; x287++) {
double x288 = x273[x287];
double x289 = tanh(x288);
x286[x287] = x289;

}
double* x293 = (double*)myMalloc(50 * sizeof(double));
for(int x294=0; x294 < 50; x294++) {
x293[x294] = 0.0;

}
double* x298 = (double*)myMalloc(26 * sizeof(double));
for(int x299=0; x299 < 26; x299++) {
double x300 = 0.0;
int32_t x302 = 50 * x299;
for(int x301=0; x301 < 50; x301++) {
int32_t x303 = x301 + x302;
double x304 = x31[x303];
double x305 = x286[x301];
double x306 = x304 * x305;
x300 += x306;

}
double x310 = x300;
x298[x299] = x310;

}
double* x314 = (double*)myMalloc(26 * sizeof(double));
for(int x315=0; x315 < 26; x315++) {
x314[x315] = 0.0;

}
double* x319 = (double*)myMalloc(26 * sizeof(double));
for(int x320=0; x320 < 26; x320++) {
double x321 = x298[x320];
double x322 = x44[x320];
double x323 = x321 + x322;
x319[x320] = x323;

}
double* x327 = (double*)myMalloc(26 * sizeof(double));
for(int x328=0; x328 < 26; x328++) {
x327[x328] = 0.0;

}
double* x332 = (double*)myMalloc(26 * sizeof(double));
for(int x333=0; x333 < 26; x333++) {
double x334 = x319[x333];
double x335 = exp(x334);
x332[x333] = x335;

}
double* x339 = (double*)myMalloc(26 * sizeof(double));
for(int x340=0; x340 < 26; x340++) {
x339[x340] = 0.0;

}
double x344 = 0.0;
for(int x345=0; x345 < 26; x345++) {
double x346 = x332[x345];
x344 += x346;

}
double* x350 = (double*)myMalloc(1 * sizeof(double));
double x351 = x344;
x350[0] = x351;
double* x353 = (double*)myMalloc(1 * sizeof(double));
for(int x354=0; x354 < 1; x354++) {
x353[x354] = 0.0;

}
double* x358 = (double*)myMalloc(26 * sizeof(double));
for(int x359=0; x359 < 26; x359++) {
double x360 = x332[x359];
double x361 = x350[0];
double x362 = x360 / x361;
x358[x359] = x362;

}
double* x366 = (double*)myMalloc(26 * sizeof(double));
for(int x367=0; x367 < 26; x367++) {
x366[x367] = 0.0;

}
double* x371 = (double*)myMalloc(1 * sizeof(double));
for(int x372=0; x372 < 1; x372++) {
double x373 = 0.0;
int32_t x375 = 26 * x372;
for(int x374=0; x374 < 26; x374++) {
int32_t x376 = x374 + x375;
double x377 = x358[x376];
double x378 = x206[x374];
double x379 = x377 * x378;
x373 += x379;

}
double x383 = x373;
x371[x372] = x383;

}
double* x387 = (double*)myMalloc(1 * sizeof(double));
for(int x388=0; x388 < 1; x388++) {
x387[x388] = 0.0;

}
double* x392 = (double*)myMalloc(1 * sizeof(double));
for(int x393=0; x393 < 1; x393++) {
double x394 = x371[x393];
double x395 = log(x394);
x392[x393] = x395;

}
double* x399 = (double*)myMalloc(1 * sizeof(double));
for(int x400=0; x400 < 1; x400++) {
x399[x400] = 0.0;

}
double* x404 = (double*)myMalloc(1 * sizeof(double));
double* x189 = x188[0];
for(int x405=0; x405 < 1; x405++) {
double x407 = x392[x405];
double x406 = x189[x405];
double x408 = x406 - x407;
x404[x405] = x408;

}
double* x412 = (double*)myMalloc(1 * sizeof(double));
for(int x413=0; x413 < 1; x413++) {
x412[x413] = 0.0;

}
double** x418 = (double**)myMalloc(4 * sizeof(double*));
x418[0] = x404;
x418[1] = x412;
x418[2] = x286;
x418[3] = x293;
int32_t x417 = x187 + 1;
x184(x417,x418);
double* x190 = x188[1];
for(int x425=0; x425 < 1; x425++) {
double x427 = x412[x425];
double x426 = x190[x425];
double x428 = x426 + x427;
x190[x425] = x428;

}
for(int x432=0; x432 < 1; x432++) {
double x433 = x399[x432];
double x434 = x412[x432];
double x435 = x433 - x434;
x399[x432] = x435;

}
for(int x439=0; x439 < 1; x439++) {
double x440 = x387[0];
double x441 = x399[0];
double x442 = x371[0];
double x443 = x441 / x442;
double x444 = x440 + x443;
x387[0] = x444;

}
for(int x448=0; x448 < 1; x448++) {
int32_t x450 = 26 * x448;
for(int x449=0; x449 < 26; x449++) {
int32_t x451 = x450 + x449;
double x452 = x366[x451];
double x453 = x206[x449];
double x454 = x387[x448];
double x455 = x453 * x454;
double x456 = x452 + x455;
x366[x451] = x456;

}

}
for(int x462=0; x462 < 1; x462++) {
int32_t x465 = 26 * x462;
for(int x463=0; x463 < 26; x463++) {
double x464 = x213[x463];
int32_t x466 = x465 + x463;
double x467 = x358[x466];
double x468 = x387[x462];
double x469 = x467 * x468;
double x470 = x464 + x469;
x213[x463] = x470;

}

}
for(int x476=0; x476 < 26; x476++) {
double x477 = x339[x476];
double x478 = x366[x476];
double x479 = x350[0];
double x480 = x478 / x479;
double x481 = x477 + x480;
x339[x476] = x481;

}
for(int x485=0; x485 < 26; x485++) {
double x486 = x353[0];
double x487 = x332[x485];
double x488 = x366[x485];
double x490 = x350[0];
double x489 = x487 * x488;
double x491 = x490 * x490;
double x492 = x489 / x491;
double x493 = x486 - x492;
x353[0] = x493;

}
for(int x497=0; x497 < 26; x497++) {
double x498 = x339[x497];
double x499 = x353[0];
double x500 = x498 + x499;
x339[x497] = x500;

}
for(int x504=0; x504 < 26; x504++) {
double x505 = x327[x504];
double x506 = x332[x504];
double x507 = x339[x504];
double x508 = x506 * x507;
double x509 = x505 + x508;
x327[x504] = x509;

}
for(int x513=0; x513 < 26; x513++) {
double x514 = x314[x513];
double x515 = x327[x513];
double x516 = x514 + x515;
x314[x513] = x516;

}
for(int x520=0; x520 < 26; x520++) {
double x521 = x80[x520];
double x522 = x327[x520];
double x523 = x521 + x522;
x80[x520] = x523;

}
for(int x527=0; x527 < 26; x527++) {
int32_t x529 = 50 * x527;
for(int x528=0; x528 < 50; x528++) {
int32_t x530 = x529 + x528;
double x531 = x70[x530];
double x532 = x286[x528];
double x533 = x314[x527];
double x534 = x532 * x533;
double x535 = x531 + x534;
x70[x530] = x535;

}

}
for(int x541=0; x541 < 26; x541++) {
int32_t x544 = 50 * x541;
for(int x542=0; x542 < 50; x542++) {
double x543 = x293[x542];
int32_t x545 = x544 + x542;
double x546 = x31[x545];
double x547 = x314[x541];
double x548 = x546 * x547;
double x549 = x543 + x548;
x293[x542] = x549;

}

}
for(int x555=0; x555 < 50; x555++) {
double x556 = x281[x555];
double x557 = x286[x555];
double x560 = x293[x555];
double x558 = x557 * x557;
double x559 = 1.0 - x558;
double x561 = x559 * x560;
double x562 = x556 + x561;
x281[x555] = x562;

}
for(int x566=0; x566 < 50; x566++) {
double x567 = x268[x566];
double x568 = x281[x566];
double x569 = x567 + x568;
x268[x566] = x569;

}
for(int x573=0; x573 < 50; x573++) {
double x574 = x75[x573];
double x575 = x281[x573];
double x576 = x574 + x575;
x75[x573] = x576;

}
for(int x580=0; x580 < 50; x580++) {
double x581 = x234[x580];
double x582 = x268[x580];
double x583 = x581 + x582;
x234[x580] = x583;

}
for(int x587=0; x587 < 50; x587++) {
double x588 = x255[x587];
double x589 = x268[x587];
double x590 = x588 + x589;
x255[x587] = x590;

}
for(int x594=0; x594 < 50; x594++) {
int32_t x596 = 50 * x594;
for(int x595=0; x595 < 50; x595++) {
int32_t x597 = x596 + x595;
double x598 = x65[x597];
double x600 = x255[x594];
double x599 = x191[x595];
double x601 = x599 * x600;
double x602 = x598 + x601;
x65[x597] = x602;

}

}
double* x192 = x188[3];
for(int x608=0; x608 < 50; x608++) {
int32_t x611 = 50 * x608;
for(int x609=0; x609 < 50; x609++) {
int32_t x612 = x611 + x609;
double x613 = x23[x612];
double x614 = x255[x608];
double x610 = x192[x609];
double x615 = x613 * x614;
double x616 = x610 + x615;
x192[x609] = x616;

}

}
for(int x622=0; x622 < 50; x622++) {
int32_t x624 = 26 * x622;
for(int x623=0; x623 < 26; x623++) {
int32_t x625 = x624 + x623;
double x626 = x60[x625];
double x627 = x194[x623];
double x628 = x234[x622];
double x629 = x627 * x628;
double x630 = x626 + x629;
x60[x625] = x630;

}

}
for(int x636=0; x636 < 50; x636++) {
int32_t x639 = 26 * x636;
for(int x637=0; x637 < 26; x637++) {
double x638 = x201[x637];
int32_t x640 = x639 + x637;
double x641 = x15[x640];
double x642 = x234[x636];
double x643 = x641 * x642;
double x644 = x638 + x643;
x201[x637] = x644;

}

}
} else {
double** x188 = x186;
double* x191 = x188[2];
for(int x651=0; x651 < 50; x651++) {
double x652 = x191[x651];
x55[x651] = x652;

}
double* x190 = x188[1];
for(int x656=0; x656 < 1; x656++) {
x190[x656] = 1.0;

}
double* x189 = x188[0];
for(int x660=0; x660 < 1; x660++) {
double x661 = x189[x660];
x169[x660] = x661;

}
}
};
x129 += 20;
int32_t x135 = x129;
int32_t x136 = x135 + 20;
int32_t x137 = x136 + 1;
bool x138 = x137 >= x3;
if (x138) {
x129 = 0;
for(int x140=0; x140 < 50; x140++) {
x50[x140] = 0.0;

}
} else {
}
for(int x149=0; x149 < 20; x149++) {
int32_t x150 = x129;
int32_t x151 = x150 + x149;
int32_t x152 = x6[x151];
x146[x149] = x152;
int32_t x154 = x151 + 1;
int32_t x155 = x6[x154];
x147[x149] = x155;

}
double* x159 = (double*)myMalloc(1 * sizeof(double));
for(int x160=0; x160 < 1; x160++) {
x159[x160] = 0.0;

}
double* x164 = (double*)myMalloc(1 * sizeof(double));
for(int x165=0; x165 < 1; x165++) {
x164[x165] = 0.0;

}
for(int x170=0; x170 < 1; x170++) {
x169[x170] = 0.0;

}
double* x174 = (double*)myMalloc(1 * sizeof(double));
for(int x175=0; x175 < 1; x175++) {
x174[x175] = 0.0;

}
double* x179 = (double*)myMalloc(1 * sizeof(double));
for(int x180=0; x180 < 1; x180++) {
x179[x180] = 0.0;

}
double** x668 = (double**)myMalloc(4 * sizeof(double*));
x668[0] = x174;
x668[1] = x179;
x668[2] = x50;
x668[3] = x85;
x184(0,x668);
double x675 = x169[0];
double x676 = x131;
double x677 = x676 * 0.9;
double x678 = x675 * 0.1;
double x679 = x677 + x678;
x131 = x679;
int32_t x681 = x133 % 100;
bool x682 = x681 == 0;
if (x682) {
double x683 = x131;
printf("iter %d, loss %f\n",x133,x683);
int32_t x685 = x133 / 100;
x126[x685] = x683;
} else {
}
for(int x689=0; x689 < 1300; x689++) {
double x690 = x60[x689];
bool x691 = x690 > 5.0;
if (x691) {
x60[x689] = 5.0;
} else {
}
bool x695 = x690 < -5.0;
if (x695) {
x60[x689] = -5.0;
} else {
}

}
double* x701 = (double*)myMalloc(1300 * sizeof(double));
for(int x702=0; x702 < 1300; x702++) {
double x703 = x60[x702];
double x704 = x703 * x703;
x701[x702] = x704;

}
for(int x708=0; x708 < 1300; x708++) {
double x709 = x101[x708];
double x710 = x701[x708];
double x711 = x709 + x710;
x101[x708] = x711;

}
double* x715 = (double*)myMalloc(1300 * sizeof(double));
for(int x716=0; x716 < 1300; x716++) {
double x717 = x60[x716];
double x718 = x90[0];
double x719 = x717 * x718;
x715[x716] = x719;

}
double* x723 = (double*)myMalloc(1300 * sizeof(double));
for(int x724=0; x724 < 1300; x724++) {
double x725 = x101[x724];
double x726 = x96[0];
double x727 = x725 + x726;
x723[x724] = x727;

}
double* x731 = (double*)myMalloc(1300 * sizeof(double));
for(int x732=0; x732 < 1300; x732++) {
double x733 = x723[x732];
double x734 = sqrt(x733);
x731[x732] = x734;

}
double* x738 = (double*)myMalloc(1300 * sizeof(double));
for(int x739=0; x739 < 1300; x739++) {
double x740 = x715[x739];
double x741 = x731[x739];
double x742 = x740 / x741;
x738[x739] = x742;

}
for(int x746=0; x746 < 1300; x746++) {
double x747 = x15[x746];
double x748 = x738[x746];
double x749 = x747 - x748;
x15[x746] = x749;

}
for(int x753=0; x753 < 1300; x753++) {
x60[x753] = 0.0;

}
for(int x757=0; x757 < 2500; x757++) {
double x758 = x65[x757];
bool x759 = x758 > 5.0;
if (x759) {
x65[x757] = 5.0;
} else {
}
bool x763 = x758 < -5.0;
if (x763) {
x65[x757] = -5.0;
} else {
}

}
double* x769 = (double*)myMalloc(2500 * sizeof(double));
for(int x770=0; x770 < 2500; x770++) {
double x771 = x65[x770];
double x772 = x771 * x771;
x769[x770] = x772;

}
for(int x776=0; x776 < 2500; x776++) {
double x777 = x106[x776];
double x778 = x769[x776];
double x779 = x777 + x778;
x106[x776] = x779;

}
double* x783 = (double*)myMalloc(2500 * sizeof(double));
for(int x784=0; x784 < 2500; x784++) {
double x785 = x65[x784];
double x786 = x90[0];
double x787 = x785 * x786;
x783[x784] = x787;

}
double* x791 = (double*)myMalloc(2500 * sizeof(double));
for(int x792=0; x792 < 2500; x792++) {
double x793 = x106[x792];
double x794 = x96[0];
double x795 = x793 + x794;
x791[x792] = x795;

}
double* x799 = (double*)myMalloc(2500 * sizeof(double));
for(int x800=0; x800 < 2500; x800++) {
double x801 = x791[x800];
double x802 = sqrt(x801);
x799[x800] = x802;

}
double* x806 = (double*)myMalloc(2500 * sizeof(double));
for(int x807=0; x807 < 2500; x807++) {
double x808 = x783[x807];
double x809 = x799[x807];
double x810 = x808 / x809;
x806[x807] = x810;

}
for(int x814=0; x814 < 2500; x814++) {
double x815 = x23[x814];
double x816 = x806[x814];
double x817 = x815 - x816;
x23[x814] = x817;

}
for(int x821=0; x821 < 2500; x821++) {
x65[x821] = 0.0;

}
for(int x825=0; x825 < 1300; x825++) {
double x826 = x70[x825];
bool x827 = x826 > 5.0;
if (x827) {
x70[x825] = 5.0;
} else {
}
bool x831 = x826 < -5.0;
if (x831) {
x70[x825] = -5.0;
} else {
}

}
double* x837 = (double*)myMalloc(1300 * sizeof(double));
for(int x838=0; x838 < 1300; x838++) {
double x839 = x70[x838];
double x840 = x839 * x839;
x837[x838] = x840;

}
for(int x844=0; x844 < 1300; x844++) {
double x845 = x111[x844];
double x846 = x837[x844];
double x847 = x845 + x846;
x111[x844] = x847;

}
double* x851 = (double*)myMalloc(1300 * sizeof(double));
for(int x852=0; x852 < 1300; x852++) {
double x853 = x70[x852];
double x854 = x90[0];
double x855 = x853 * x854;
x851[x852] = x855;

}
double* x859 = (double*)myMalloc(1300 * sizeof(double));
for(int x860=0; x860 < 1300; x860++) {
double x861 = x111[x860];
double x862 = x96[0];
double x863 = x861 + x862;
x859[x860] = x863;

}
double* x867 = (double*)myMalloc(1300 * sizeof(double));
for(int x868=0; x868 < 1300; x868++) {
double x869 = x859[x868];
double x870 = sqrt(x869);
x867[x868] = x870;

}
double* x874 = (double*)myMalloc(1300 * sizeof(double));
for(int x875=0; x875 < 1300; x875++) {
double x876 = x851[x875];
double x877 = x867[x875];
double x878 = x876 / x877;
x874[x875] = x878;

}
for(int x882=0; x882 < 1300; x882++) {
double x883 = x31[x882];
double x884 = x874[x882];
double x885 = x883 - x884;
x31[x882] = x885;

}
for(int x889=0; x889 < 1300; x889++) {
x70[x889] = 0.0;

}
for(int x893=0; x893 < 50; x893++) {
double x894 = x75[x893];
bool x895 = x894 > 5.0;
if (x895) {
x75[x893] = 5.0;
} else {
}
bool x899 = x894 < -5.0;
if (x899) {
x75[x893] = -5.0;
} else {
}

}
double* x905 = (double*)myMalloc(50 * sizeof(double));
for(int x906=0; x906 < 50; x906++) {
double x907 = x75[x906];
double x908 = x907 * x907;
x905[x906] = x908;

}
for(int x912=0; x912 < 50; x912++) {
double x913 = x116[x912];
double x914 = x905[x912];
double x915 = x913 + x914;
x116[x912] = x915;

}
double* x919 = (double*)myMalloc(50 * sizeof(double));
for(int x920=0; x920 < 50; x920++) {
double x921 = x75[x920];
double x922 = x90[0];
double x923 = x921 * x922;
x919[x920] = x923;

}
double* x927 = (double*)myMalloc(50 * sizeof(double));
for(int x928=0; x928 < 50; x928++) {
double x929 = x116[x928];
double x930 = x96[0];
double x931 = x929 + x930;
x927[x928] = x931;

}
double* x935 = (double*)myMalloc(50 * sizeof(double));
for(int x936=0; x936 < 50; x936++) {
double x937 = x927[x936];
double x938 = sqrt(x937);
x935[x936] = x938;

}
double* x942 = (double*)myMalloc(50 * sizeof(double));
for(int x943=0; x943 < 50; x943++) {
double x944 = x919[x943];
double x945 = x935[x943];
double x946 = x944 / x945;
x942[x943] = x946;

}
for(int x950=0; x950 < 50; x950++) {
double x951 = x38[x950];
double x952 = x942[x950];
double x953 = x951 - x952;
x38[x950] = x953;

}
for(int x957=0; x957 < 50; x957++) {
x75[x957] = 0.0;

}
for(int x961=0; x961 < 26; x961++) {
double x962 = x80[x961];
bool x963 = x962 > 5.0;
if (x963) {
x80[x961] = 5.0;
} else {
}
bool x967 = x962 < -5.0;
if (x967) {
x80[x961] = -5.0;
} else {
}

}
double* x973 = (double*)myMalloc(26 * sizeof(double));
for(int x974=0; x974 < 26; x974++) {
double x975 = x80[x974];
double x976 = x975 * x975;
x973[x974] = x976;

}
for(int x980=0; x980 < 26; x980++) {
double x981 = x121[x980];
double x982 = x973[x980];
double x983 = x981 + x982;
x121[x980] = x983;

}
double* x987 = (double*)myMalloc(26 * sizeof(double));
for(int x988=0; x988 < 26; x988++) {
double x989 = x80[x988];
double x990 = x90[0];
double x991 = x989 * x990;
x987[x988] = x991;

}
double* x995 = (double*)myMalloc(26 * sizeof(double));
for(int x996=0; x996 < 26; x996++) {
double x997 = x121[x996];
double x998 = x96[0];
double x999 = x997 + x998;
x995[x996] = x999;

}
double* x1003 = (double*)myMalloc(26 * sizeof(double));
for(int x1004=0; x1004 < 26; x1004++) {
double x1005 = x995[x1004];
double x1006 = sqrt(x1005);
x1003[x1004] = x1006;

}
double* x1010 = (double*)myMalloc(26 * sizeof(double));
for(int x1011=0; x1011 < 26; x1011++) {
double x1012 = x987[x1011];
double x1013 = x1003[x1011];
double x1014 = x1012 / x1013;
x1010[x1011] = x1014;

}
for(int x1018=0; x1018 < 26; x1018++) {
double x1019 = x44[x1018];
double x1020 = x1010[x1018];
double x1021 = x1019 - x1020;
x44[x1018] = x1021;

}
for(int x1025=0; x1025 < 26; x1025++) {
x80[x1025] = 0.0;

}
for(int x1029=0; x1029 < 50; x1029++) {
x85[x1029] = 0.0;

}
for(int x1033=0; x1033 < 50; x1033++) {
double x1034 = x55[x1033];
x50[x1033] = x1034;

}
mallocAddr = (void*)x128;

}
double x1041 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x1044 = (long)fopen(x0, "w");
fprintf((FILE *)x1044, "unit: %s\n", "100 iteration");
for(int x1047=0; x1047 < 51; x1047++) {
double x1048 = x126[x1047];
fprintf((FILE *)x1044, "%lf\n", x1048);

}
double x1042 = x127 - x1;
double x1043 = x1041 - x127;
fprintf((FILE *)x1044, "run time: %lf %lf\n", x1042, x1043);
fclose((FILE*)x1044);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

