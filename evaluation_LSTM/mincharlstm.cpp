
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
double* x14 = (double*)myMalloc(2500 * sizeof(double));
for(int x16=0; x16 < 2500; x16++) {
double x17 = d(gen);
double x18 = x17 * 0.01;
x14[x16] = x18;

}
double* x22 = (double*)myMalloc(1300 * sizeof(double));
for(int x24=0; x24 < 1300; x24++) {
double x25 = d(gen);
double x26 = x25 * 0.01;
x22[x24] = x26;

}
double* x30 = (double*)myMalloc(50 * sizeof(double));
for(int x32=0; x32 < 50; x32++) {
x30[x32] = 0.0;

}
double* x36 = (double*)myMalloc(2500 * sizeof(double));
for(int x37=0; x37 < 2500; x37++) {
double x38 = d(gen);
double x39 = x38 * 0.01;
x36[x37] = x39;

}
double* x43 = (double*)myMalloc(1300 * sizeof(double));
for(int x44=0; x44 < 1300; x44++) {
double x45 = d(gen);
double x46 = x45 * 0.01;
x43[x44] = x46;

}
double* x50 = (double*)myMalloc(50 * sizeof(double));
for(int x51=0; x51 < 50; x51++) {
x50[x51] = 0.0;

}
double* x55 = (double*)myMalloc(2500 * sizeof(double));
for(int x56=0; x56 < 2500; x56++) {
double x57 = d(gen);
double x58 = x57 * 0.01;
x55[x56] = x58;

}
double* x62 = (double*)myMalloc(1300 * sizeof(double));
for(int x63=0; x63 < 1300; x63++) {
double x64 = d(gen);
double x65 = x64 * 0.01;
x62[x63] = x65;

}
double* x69 = (double*)myMalloc(50 * sizeof(double));
for(int x70=0; x70 < 50; x70++) {
x69[x70] = 0.0;

}
double* x74 = (double*)myMalloc(2500 * sizeof(double));
for(int x75=0; x75 < 2500; x75++) {
double x76 = d(gen);
double x77 = x76 * 0.01;
x74[x75] = x77;

}
double* x81 = (double*)myMalloc(1300 * sizeof(double));
for(int x82=0; x82 < 1300; x82++) {
double x83 = d(gen);
double x84 = x83 * 0.01;
x81[x82] = x84;

}
double* x88 = (double*)myMalloc(50 * sizeof(double));
for(int x89=0; x89 < 50; x89++) {
x88[x89] = 0.0;

}
double* x93 = (double*)myMalloc(1300 * sizeof(double));
for(int x94=0; x94 < 1300; x94++) {
double x95 = d(gen);
double x96 = x95 * 0.01;
x93[x94] = x96;

}
double* x100 = (double*)myMalloc(26 * sizeof(double));
for(int x102=0; x102 < 26; x102++) {
x100[x102] = 0.0;

}
double* x106 = (double*)myMalloc(50 * sizeof(double));
for(int x107=0; x107 < 50; x107++) {
x106[x107] = 0.0;

}
double* x111 = (double*)myMalloc(50 * sizeof(double));
for(int x112=0; x112 < 50; x112++) {
x111[x112] = 0.0;

}
double* x116 = (double*)myMalloc(50 * sizeof(double));
for(int x117=0; x117 < 50; x117++) {
x116[x117] = 0.0;

}
double* x121 = (double*)myMalloc(50 * sizeof(double));
for(int x122=0; x122 < 50; x122++) {
x121[x122] = 0.0;

}
double* x126 = (double*)myMalloc(2500 * sizeof(double));
for(int x127=0; x127 < 2500; x127++) {
x126[x127] = 0.0;

}
double* x131 = (double*)myMalloc(1300 * sizeof(double));
for(int x132=0; x132 < 1300; x132++) {
x131[x132] = 0.0;

}
double* x136 = (double*)myMalloc(50 * sizeof(double));
for(int x137=0; x137 < 50; x137++) {
x136[x137] = 0.0;

}
double* x141 = (double*)myMalloc(2500 * sizeof(double));
for(int x142=0; x142 < 2500; x142++) {
x141[x142] = 0.0;

}
double* x146 = (double*)myMalloc(1300 * sizeof(double));
for(int x147=0; x147 < 1300; x147++) {
x146[x147] = 0.0;

}
double* x151 = (double*)myMalloc(50 * sizeof(double));
for(int x152=0; x152 < 50; x152++) {
x151[x152] = 0.0;

}
double* x156 = (double*)myMalloc(2500 * sizeof(double));
for(int x157=0; x157 < 2500; x157++) {
x156[x157] = 0.0;

}
double* x161 = (double*)myMalloc(1300 * sizeof(double));
for(int x162=0; x162 < 1300; x162++) {
x161[x162] = 0.0;

}
double* x166 = (double*)myMalloc(50 * sizeof(double));
for(int x167=0; x167 < 50; x167++) {
x166[x167] = 0.0;

}
double* x171 = (double*)myMalloc(2500 * sizeof(double));
for(int x172=0; x172 < 2500; x172++) {
x171[x172] = 0.0;

}
double* x176 = (double*)myMalloc(1300 * sizeof(double));
for(int x177=0; x177 < 1300; x177++) {
x176[x177] = 0.0;

}
double* x181 = (double*)myMalloc(50 * sizeof(double));
for(int x182=0; x182 < 50; x182++) {
x181[x182] = 0.0;

}
double* x186 = (double*)myMalloc(1300 * sizeof(double));
for(int x187=0; x187 < 1300; x187++) {
x186[x187] = 0.0;

}
double* x191 = (double*)myMalloc(26 * sizeof(double));
for(int x192=0; x192 < 26; x192++) {
x191[x192] = 0.0;

}
double* x196 = (double*)myMalloc(50 * sizeof(double));
for(int x197=0; x197 < 50; x197++) {
x196[x197] = 0.0;

}
double* x201 = (double*)myMalloc(50 * sizeof(double));
for(int x202=0; x202 < 50; x202++) {
x201[x202] = 0.0;

}
double* x206 = (double*)myMalloc(1 * sizeof(double));
for(int x208=0; x208 < 1; x208++) {
x206[x208] = 0.1;

}
double* x212 = (double*)myMalloc(1 * sizeof(double));
for(int x213=0; x213 < 1; x213++) {
x212[x213] = 1.0E-8;

}
double* x217 = (double*)myMalloc(2500 * sizeof(double));
for(int x218=0; x218 < 2500; x218++) {
x217[x218] = 0.0;

}
double* x222 = (double*)myMalloc(1300 * sizeof(double));
for(int x223=0; x223 < 1300; x223++) {
x222[x223] = 0.0;

}
double* x227 = (double*)myMalloc(50 * sizeof(double));
for(int x228=0; x228 < 50; x228++) {
x227[x228] = 0.0;

}
double* x232 = (double*)myMalloc(2500 * sizeof(double));
for(int x233=0; x233 < 2500; x233++) {
x232[x233] = 0.0;

}
double* x237 = (double*)myMalloc(1300 * sizeof(double));
for(int x238=0; x238 < 1300; x238++) {
x237[x238] = 0.0;

}
double* x242 = (double*)myMalloc(50 * sizeof(double));
for(int x243=0; x243 < 50; x243++) {
x242[x243] = 0.0;

}
double* x247 = (double*)myMalloc(2500 * sizeof(double));
for(int x248=0; x248 < 2500; x248++) {
x247[x248] = 0.0;

}
double* x252 = (double*)myMalloc(1300 * sizeof(double));
for(int x253=0; x253 < 1300; x253++) {
x252[x253] = 0.0;

}
double* x257 = (double*)myMalloc(50 * sizeof(double));
for(int x258=0; x258 < 50; x258++) {
x257[x258] = 0.0;

}
double* x262 = (double*)myMalloc(2500 * sizeof(double));
for(int x263=0; x263 < 2500; x263++) {
x262[x263] = 0.0;

}
double* x267 = (double*)myMalloc(1300 * sizeof(double));
for(int x268=0; x268 < 1300; x268++) {
x267[x268] = 0.0;

}
double* x272 = (double*)myMalloc(50 * sizeof(double));
for(int x273=0; x273 < 50; x273++) {
x272[x273] = 0.0;

}
double* x277 = (double*)myMalloc(1300 * sizeof(double));
for(int x278=0; x278 < 1300; x278++) {
x277[x278] = 0.0;

}
double* x282 = (double*)myMalloc(26 * sizeof(double));
for(int x283=0; x283 < 26; x283++) {
x282[x283] = 0.0;

}
int64_t x287 = (long)mallocAddr;
int32_t x288 = 0;
x288 -= 20;
clock_t begin_0, end_0; double time_spent_0;
begin_0 = clock();
for(int x293=0; x293 < 2001; x293++) {
double* x329 = (double*)myMalloc(1 * sizeof(double));
int32_t* x306 = (int32_t*)myMalloc(20 * sizeof(int32_t));
int32_t* x307 = (int32_t*)myMalloc(20 * sizeof(int32_t));
function<void(int32_t,double**)> x344 = [&](int32_t x345,double** x346) {
int32_t x347 = x345;
bool x355 = x347 < 20;
if (x355) {
double* x356 = (double*)myMalloc(26 * sizeof(double));
for(int x357=0; x357 < 26; x357++) {
x356[x357] = 0.0;

}
int32_t x361 = x306[x347];
x356[x361] = 1.0;
double* x363 = (double*)myMalloc(26 * sizeof(double));
for(int x364=0; x364 < 26; x364++) {
x363[x364] = 0.0;

}
double* x368 = (double*)myMalloc(26 * sizeof(double));
for(int x369=0; x369 < 26; x369++) {
x368[x369] = 0.0;

}
int32_t x373 = x307[x347];
x368[x373] = 1.0;
double* x375 = (double*)myMalloc(26 * sizeof(double));
for(int x376=0; x376 < 26; x376++) {
x375[x376] = 0.0;

}
double* x380 = (double*)myMalloc(50 * sizeof(double));
double** x348 = x346;
double* x351 = x348[2];
for(int x381=0; x381 < 50; x381++) {
double x382 = 0.0;
int32_t x384 = 50 * x381;
for(int x383=0; x383 < 50; x383++) {
int32_t x385 = x383 + x384;
double x386 = x14[x385];
double x387 = x351[x383];
double x388 = x386 * x387;
x382 += x388;

}
double x392 = x382;
x380[x381] = x392;

}
double* x396 = (double*)myMalloc(50 * sizeof(double));
for(int x397=0; x397 < 50; x397++) {
x396[x397] = 0.0;

}
double* x401 = (double*)myMalloc(50 * sizeof(double));
for(int x402=0; x402 < 50; x402++) {
double x403 = 0.0;
int32_t x405 = 26 * x402;
for(int x404=0; x404 < 26; x404++) {
int32_t x406 = x404 + x405;
double x407 = x22[x406];
double x408 = x356[x404];
double x409 = x407 * x408;
x403 += x409;

}
double x413 = x403;
x401[x402] = x413;

}
double* x417 = (double*)myMalloc(50 * sizeof(double));
for(int x418=0; x418 < 50; x418++) {
x417[x418] = 0.0;

}
double* x422 = (double*)myMalloc(50 * sizeof(double));
for(int x423=0; x423 < 50; x423++) {
double x424 = x380[x423];
double x425 = x401[x423];
double x426 = x424 + x425;
x422[x423] = x426;

}
double* x430 = (double*)myMalloc(50 * sizeof(double));
for(int x431=0; x431 < 50; x431++) {
x430[x431] = 0.0;

}
double* x435 = (double*)myMalloc(50 * sizeof(double));
for(int x436=0; x436 < 50; x436++) {
double x437 = x422[x436];
double x438 = x30[x436];
double x439 = x437 + x438;
x435[x436] = x439;

}
double* x443 = (double*)myMalloc(50 * sizeof(double));
for(int x444=0; x444 < 50; x444++) {
x443[x444] = 0.0;

}
double* x448 = (double*)myMalloc(50 * sizeof(double));
for(int x449=0; x449 < 50; x449++) {
double x450 = x435[x449];
double x451 = -1.0 * x450;
double x452 = exp(x451);
double x453 = x452 + 1.0;
double x454 = 1.0 / x453;
x448[x449] = x454;

}
double* x458 = (double*)myMalloc(50 * sizeof(double));
for(int x459=0; x459 < 50; x459++) {
x458[x459] = 0.0;

}
double* x463 = (double*)myMalloc(50 * sizeof(double));
for(int x464=0; x464 < 50; x464++) {
double x465 = 0.0;
int32_t x467 = 50 * x464;
for(int x466=0; x466 < 50; x466++) {
int32_t x468 = x466 + x467;
double x469 = x36[x468];
double x470 = x351[x466];
double x471 = x469 * x470;
x465 += x471;

}
double x475 = x465;
x463[x464] = x475;

}
double* x479 = (double*)myMalloc(50 * sizeof(double));
for(int x480=0; x480 < 50; x480++) {
x479[x480] = 0.0;

}
double* x484 = (double*)myMalloc(50 * sizeof(double));
for(int x485=0; x485 < 50; x485++) {
double x486 = 0.0;
int32_t x488 = 26 * x485;
for(int x487=0; x487 < 26; x487++) {
int32_t x489 = x487 + x488;
double x490 = x43[x489];
double x491 = x356[x487];
double x492 = x490 * x491;
x486 += x492;

}
double x496 = x486;
x484[x485] = x496;

}
double* x500 = (double*)myMalloc(50 * sizeof(double));
for(int x501=0; x501 < 50; x501++) {
x500[x501] = 0.0;

}
double* x505 = (double*)myMalloc(50 * sizeof(double));
for(int x506=0; x506 < 50; x506++) {
double x507 = x463[x506];
double x508 = x484[x506];
double x509 = x507 + x508;
x505[x506] = x509;

}
double* x513 = (double*)myMalloc(50 * sizeof(double));
for(int x514=0; x514 < 50; x514++) {
x513[x514] = 0.0;

}
double* x518 = (double*)myMalloc(50 * sizeof(double));
for(int x519=0; x519 < 50; x519++) {
double x520 = x505[x519];
double x521 = x50[x519];
double x522 = x520 + x521;
x518[x519] = x522;

}
double* x526 = (double*)myMalloc(50 * sizeof(double));
for(int x527=0; x527 < 50; x527++) {
x526[x527] = 0.0;

}
double* x531 = (double*)myMalloc(50 * sizeof(double));
for(int x532=0; x532 < 50; x532++) {
double x533 = x518[x532];
double x534 = -1.0 * x533;
double x535 = exp(x534);
double x536 = x535 + 1.0;
double x537 = 1.0 / x536;
x531[x532] = x537;

}
double* x541 = (double*)myMalloc(50 * sizeof(double));
for(int x542=0; x542 < 50; x542++) {
x541[x542] = 0.0;

}
double* x546 = (double*)myMalloc(50 * sizeof(double));
for(int x547=0; x547 < 50; x547++) {
double x548 = 0.0;
int32_t x550 = 50 * x547;
for(int x549=0; x549 < 50; x549++) {
int32_t x551 = x549 + x550;
double x552 = x74[x551];
double x553 = x351[x549];
double x554 = x552 * x553;
x548 += x554;

}
double x558 = x548;
x546[x547] = x558;

}
double* x562 = (double*)myMalloc(50 * sizeof(double));
for(int x563=0; x563 < 50; x563++) {
x562[x563] = 0.0;

}
double* x567 = (double*)myMalloc(50 * sizeof(double));
for(int x568=0; x568 < 50; x568++) {
double x569 = 0.0;
int32_t x571 = 26 * x568;
for(int x570=0; x570 < 26; x570++) {
int32_t x572 = x570 + x571;
double x573 = x81[x572];
double x574 = x356[x570];
double x575 = x573 * x574;
x569 += x575;

}
double x579 = x569;
x567[x568] = x579;

}
double* x583 = (double*)myMalloc(50 * sizeof(double));
for(int x584=0; x584 < 50; x584++) {
x583[x584] = 0.0;

}
double* x588 = (double*)myMalloc(50 * sizeof(double));
for(int x589=0; x589 < 50; x589++) {
double x590 = x546[x589];
double x591 = x567[x589];
double x592 = x590 + x591;
x588[x589] = x592;

}
double* x596 = (double*)myMalloc(50 * sizeof(double));
for(int x597=0; x597 < 50; x597++) {
x596[x597] = 0.0;

}
double* x601 = (double*)myMalloc(50 * sizeof(double));
for(int x602=0; x602 < 50; x602++) {
double x603 = x588[x602];
double x604 = x88[x602];
double x605 = x603 + x604;
x601[x602] = x605;

}
double* x609 = (double*)myMalloc(50 * sizeof(double));
for(int x610=0; x610 < 50; x610++) {
x609[x610] = 0.0;

}
double* x614 = (double*)myMalloc(50 * sizeof(double));
for(int x615=0; x615 < 50; x615++) {
double x616 = x601[x615];
double x617 = -1.0 * x616;
double x618 = exp(x617);
double x619 = x618 + 1.0;
double x620 = 1.0 / x619;
x614[x615] = x620;

}
double* x624 = (double*)myMalloc(50 * sizeof(double));
for(int x625=0; x625 < 50; x625++) {
x624[x625] = 0.0;

}
double* x629 = (double*)myMalloc(50 * sizeof(double));
for(int x630=0; x630 < 50; x630++) {
double x631 = 0.0;
int32_t x633 = 50 * x630;
for(int x632=0; x632 < 50; x632++) {
int32_t x634 = x632 + x633;
double x635 = x55[x634];
double x636 = x351[x632];
double x637 = x635 * x636;
x631 += x637;

}
double x641 = x631;
x629[x630] = x641;

}
double* x645 = (double*)myMalloc(50 * sizeof(double));
for(int x646=0; x646 < 50; x646++) {
x645[x646] = 0.0;

}
double* x650 = (double*)myMalloc(50 * sizeof(double));
for(int x651=0; x651 < 50; x651++) {
double x652 = 0.0;
int32_t x654 = 26 * x651;
for(int x653=0; x653 < 26; x653++) {
int32_t x655 = x653 + x654;
double x656 = x62[x655];
double x657 = x356[x653];
double x658 = x656 * x657;
x652 += x658;

}
double x662 = x652;
x650[x651] = x662;

}
double* x666 = (double*)myMalloc(50 * sizeof(double));
for(int x667=0; x667 < 50; x667++) {
x666[x667] = 0.0;

}
double* x671 = (double*)myMalloc(50 * sizeof(double));
for(int x672=0; x672 < 50; x672++) {
double x673 = x629[x672];
double x674 = x650[x672];
double x675 = x673 + x674;
x671[x672] = x675;

}
double* x679 = (double*)myMalloc(50 * sizeof(double));
for(int x680=0; x680 < 50; x680++) {
x679[x680] = 0.0;

}
double* x684 = (double*)myMalloc(50 * sizeof(double));
for(int x685=0; x685 < 50; x685++) {
double x686 = x671[x685];
double x687 = x69[x685];
double x688 = x686 + x687;
x684[x685] = x688;

}
double* x692 = (double*)myMalloc(50 * sizeof(double));
for(int x693=0; x693 < 50; x693++) {
x692[x693] = 0.0;

}
double* x697 = (double*)myMalloc(50 * sizeof(double));
for(int x698=0; x698 < 50; x698++) {
double x699 = x684[x698];
double x700 = tanh(x699);
x697[x698] = x700;

}
double* x704 = (double*)myMalloc(50 * sizeof(double));
for(int x705=0; x705 < 50; x705++) {
x704[x705] = 0.0;

}
double* x709 = (double*)myMalloc(50 * sizeof(double));
double* x353 = x348[4];
for(int x710=0; x710 < 50; x710++) {
double x711 = x448[x710];
double x712 = x353[x710];
double x713 = x711 * x712;
x709[x710] = x713;

}
double* x717 = (double*)myMalloc(50 * sizeof(double));
for(int x718=0; x718 < 50; x718++) {
x717[x718] = 0.0;

}
double* x722 = (double*)myMalloc(50 * sizeof(double));
for(int x723=0; x723 < 50; x723++) {
double x724 = x531[x723];
double x725 = x697[x723];
double x726 = x724 * x725;
x722[x723] = x726;

}
double* x730 = (double*)myMalloc(50 * sizeof(double));
for(int x731=0; x731 < 50; x731++) {
x730[x731] = 0.0;

}
double* x735 = (double*)myMalloc(50 * sizeof(double));
for(int x736=0; x736 < 50; x736++) {
double x737 = x709[x736];
double x738 = x722[x736];
double x739 = x737 + x738;
x735[x736] = x739;

}
double* x743 = (double*)myMalloc(50 * sizeof(double));
for(int x744=0; x744 < 50; x744++) {
x743[x744] = 0.0;

}
double* x748 = (double*)myMalloc(50 * sizeof(double));
for(int x749=0; x749 < 50; x749++) {
double x750 = x735[x749];
double x751 = tanh(x750);
x748[x749] = x751;

}
double* x755 = (double*)myMalloc(50 * sizeof(double));
for(int x756=0; x756 < 50; x756++) {
x755[x756] = 0.0;

}
double* x760 = (double*)myMalloc(50 * sizeof(double));
for(int x761=0; x761 < 50; x761++) {
double x762 = x614[x761];
double x763 = x748[x761];
double x764 = x762 * x763;
x760[x761] = x764;

}
double* x768 = (double*)myMalloc(50 * sizeof(double));
for(int x769=0; x769 < 50; x769++) {
x768[x769] = 0.0;

}
double* x773 = (double*)myMalloc(26 * sizeof(double));
for(int x774=0; x774 < 26; x774++) {
double x775 = 0.0;
int32_t x777 = 50 * x774;
for(int x776=0; x776 < 50; x776++) {
int32_t x778 = x776 + x777;
double x779 = x93[x778];
double x780 = x760[x776];
double x781 = x779 * x780;
x775 += x781;

}
double x785 = x775;
x773[x774] = x785;

}
double* x789 = (double*)myMalloc(26 * sizeof(double));
for(int x790=0; x790 < 26; x790++) {
x789[x790] = 0.0;

}
double* x794 = (double*)myMalloc(26 * sizeof(double));
for(int x795=0; x795 < 26; x795++) {
double x796 = x773[x795];
double x797 = x100[x795];
double x798 = x796 + x797;
x794[x795] = x798;

}
double* x802 = (double*)myMalloc(26 * sizeof(double));
for(int x803=0; x803 < 26; x803++) {
x802[x803] = 0.0;

}
double* x807 = (double*)myMalloc(26 * sizeof(double));
for(int x808=0; x808 < 26; x808++) {
double x809 = x794[x808];
double x810 = exp(x809);
x807[x808] = x810;

}
double* x814 = (double*)myMalloc(26 * sizeof(double));
for(int x815=0; x815 < 26; x815++) {
x814[x815] = 0.0;

}
double x819 = 0.0;
for(int x820=0; x820 < 26; x820++) {
double x821 = x807[x820];
x819 += x821;

}
double* x825 = (double*)myMalloc(1 * sizeof(double));
double x826 = x819;
x825[0] = x826;
double* x828 = (double*)myMalloc(1 * sizeof(double));
for(int x829=0; x829 < 1; x829++) {
x828[x829] = 0.0;

}
double* x833 = (double*)myMalloc(26 * sizeof(double));
for(int x834=0; x834 < 26; x834++) {
double x835 = x807[x834];
double x836 = x825[0];
double x837 = x835 / x836;
x833[x834] = x837;

}
double* x841 = (double*)myMalloc(26 * sizeof(double));
for(int x842=0; x842 < 26; x842++) {
x841[x842] = 0.0;

}
double* x846 = (double*)myMalloc(1 * sizeof(double));
for(int x847=0; x847 < 1; x847++) {
double x848 = 0.0;
int32_t x850 = 26 * x847;
for(int x849=0; x849 < 26; x849++) {
int32_t x851 = x849 + x850;
double x852 = x833[x851];
double x853 = x368[x849];
double x854 = x852 * x853;
x848 += x854;

}
double x858 = x848;
x846[x847] = x858;

}
double* x862 = (double*)myMalloc(1 * sizeof(double));
for(int x863=0; x863 < 1; x863++) {
x862[x863] = 0.0;

}
double* x867 = (double*)myMalloc(1 * sizeof(double));
for(int x868=0; x868 < 1; x868++) {
double x869 = x846[x868];
double x870 = log(x869);
x867[x868] = x870;

}
double* x874 = (double*)myMalloc(1 * sizeof(double));
for(int x875=0; x875 < 1; x875++) {
x874[x875] = 0.0;

}
double* x879 = (double*)myMalloc(1 * sizeof(double));
double* x349 = x348[0];
for(int x880=0; x880 < 1; x880++) {
double x882 = x867[x880];
double x881 = x349[x880];
double x883 = x881 - x882;
x879[x880] = x883;

}
double* x887 = (double*)myMalloc(1 * sizeof(double));
for(int x888=0; x888 < 1; x888++) {
x887[x888] = 0.0;

}
double** x893 = (double**)myMalloc(6 * sizeof(double*));
x893[0] = x879;
x893[1] = x887;
x893[2] = x760;
x893[3] = x768;
x893[4] = x735;
x893[5] = x743;
int32_t x892 = x347 + 1;
x344(x892,x893);
double* x350 = x348[1];
for(int x902=0; x902 < 1; x902++) {
double x904 = x887[x902];
double x903 = x350[x902];
double x905 = x903 + x904;
x350[x902] = x905;

}
for(int x909=0; x909 < 1; x909++) {
double x910 = x874[x909];
double x911 = x887[x909];
double x912 = x910 - x911;
x874[x909] = x912;

}
for(int x916=0; x916 < 1; x916++) {
double x917 = x862[0];
double x918 = x874[0];
double x919 = x846[0];
double x920 = x918 / x919;
double x921 = x917 + x920;
x862[0] = x921;

}
for(int x925=0; x925 < 1; x925++) {
int32_t x927 = 26 * x925;
for(int x926=0; x926 < 26; x926++) {
int32_t x928 = x927 + x926;
double x929 = x841[x928];
double x930 = x368[x926];
double x931 = x862[x925];
double x932 = x930 * x931;
double x933 = x929 + x932;
x841[x928] = x933;

}

}
for(int x939=0; x939 < 1; x939++) {
int32_t x942 = 26 * x939;
for(int x940=0; x940 < 26; x940++) {
double x941 = x375[x940];
int32_t x943 = x942 + x940;
double x944 = x833[x943];
double x945 = x862[x939];
double x946 = x944 * x945;
double x947 = x941 + x946;
x375[x940] = x947;

}

}
for(int x953=0; x953 < 26; x953++) {
double x954 = x814[x953];
double x955 = x841[x953];
double x956 = x825[0];
double x957 = x955 / x956;
double x958 = x954 + x957;
x814[x953] = x958;

}
for(int x962=0; x962 < 26; x962++) {
double x963 = x828[0];
double x964 = x807[x962];
double x965 = x841[x962];
double x967 = x825[0];
double x966 = x964 * x965;
double x968 = x967 * x967;
double x969 = x966 / x968;
double x970 = x963 - x969;
x828[0] = x970;

}
for(int x974=0; x974 < 26; x974++) {
double x975 = x814[x974];
double x976 = x828[0];
double x977 = x975 + x976;
x814[x974] = x977;

}
for(int x981=0; x981 < 26; x981++) {
double x982 = x802[x981];
double x983 = x807[x981];
double x984 = x814[x981];
double x985 = x983 * x984;
double x986 = x982 + x985;
x802[x981] = x986;

}
for(int x990=0; x990 < 26; x990++) {
double x991 = x789[x990];
double x992 = x802[x990];
double x993 = x991 + x992;
x789[x990] = x993;

}
for(int x997=0; x997 < 26; x997++) {
double x998 = x191[x997];
double x999 = x802[x997];
double x1000 = x998 + x999;
x191[x997] = x1000;

}
for(int x1004=0; x1004 < 26; x1004++) {
int32_t x1006 = 50 * x1004;
for(int x1005=0; x1005 < 50; x1005++) {
int32_t x1007 = x1006 + x1005;
double x1008 = x186[x1007];
double x1009 = x760[x1005];
double x1010 = x789[x1004];
double x1011 = x1009 * x1010;
double x1012 = x1008 + x1011;
x186[x1007] = x1012;

}

}
for(int x1018=0; x1018 < 26; x1018++) {
int32_t x1021 = 50 * x1018;
for(int x1019=0; x1019 < 50; x1019++) {
double x1020 = x768[x1019];
int32_t x1022 = x1021 + x1019;
double x1023 = x93[x1022];
double x1024 = x789[x1018];
double x1025 = x1023 * x1024;
double x1026 = x1020 + x1025;
x768[x1019] = x1026;

}

}
for(int x1032=0; x1032 < 50; x1032++) {
double x1033 = x624[x1032];
double x1034 = x748[x1032];
double x1035 = x768[x1032];
double x1036 = x1034 * x1035;
double x1037 = x1033 + x1036;
x624[x1032] = x1037;

}
for(int x1041=0; x1041 < 50; x1041++) {
double x1042 = x755[x1041];
double x1043 = x614[x1041];
double x1044 = x768[x1041];
double x1045 = x1043 * x1044;
double x1046 = x1042 + x1045;
x755[x1041] = x1046;

}
for(int x1050=0; x1050 < 50; x1050++) {
double x1051 = x743[x1050];
double x1052 = x748[x1050];
double x1055 = x755[x1050];
double x1053 = x1052 * x1052;
double x1054 = 1.0 - x1053;
double x1056 = x1054 * x1055;
double x1057 = x1051 + x1056;
x743[x1050] = x1057;

}
for(int x1061=0; x1061 < 50; x1061++) {
double x1062 = x717[x1061];
double x1063 = x743[x1061];
double x1064 = x1062 + x1063;
x717[x1061] = x1064;

}
for(int x1068=0; x1068 < 50; x1068++) {
double x1069 = x730[x1068];
double x1070 = x743[x1068];
double x1071 = x1069 + x1070;
x730[x1068] = x1071;

}
for(int x1075=0; x1075 < 50; x1075++) {
double x1076 = x541[x1075];
double x1077 = x697[x1075];
double x1078 = x730[x1075];
double x1079 = x1077 * x1078;
double x1080 = x1076 + x1079;
x541[x1075] = x1080;

}
for(int x1084=0; x1084 < 50; x1084++) {
double x1085 = x704[x1084];
double x1086 = x531[x1084];
double x1087 = x730[x1084];
double x1088 = x1086 * x1087;
double x1089 = x1085 + x1088;
x704[x1084] = x1089;

}
for(int x1093=0; x1093 < 50; x1093++) {
double x1094 = x458[x1093];
double x1096 = x717[x1093];
double x1095 = x353[x1093];
double x1097 = x1095 * x1096;
double x1098 = x1094 + x1097;
x458[x1093] = x1098;

}
double* x354 = x348[5];
for(int x1102=0; x1102 < 50; x1102++) {
double x1104 = x448[x1102];
double x1105 = x717[x1102];
double x1103 = x354[x1102];
double x1106 = x1104 * x1105;
double x1107 = x1103 + x1106;
x354[x1102] = x1107;

}
for(int x1111=0; x1111 < 50; x1111++) {
double x1112 = x692[x1111];
double x1113 = x697[x1111];
double x1116 = x704[x1111];
double x1114 = x1113 * x1113;
double x1115 = 1.0 - x1114;
double x1117 = x1115 * x1116;
double x1118 = x1112 + x1117;
x692[x1111] = x1118;

}
for(int x1122=0; x1122 < 50; x1122++) {
double x1123 = x679[x1122];
double x1124 = x692[x1122];
double x1125 = x1123 + x1124;
x679[x1122] = x1125;

}
for(int x1129=0; x1129 < 50; x1129++) {
double x1130 = x166[x1129];
double x1131 = x692[x1129];
double x1132 = x1130 + x1131;
x166[x1129] = x1132;

}
for(int x1136=0; x1136 < 50; x1136++) {
double x1137 = x645[x1136];
double x1138 = x679[x1136];
double x1139 = x1137 + x1138;
x645[x1136] = x1139;

}
for(int x1143=0; x1143 < 50; x1143++) {
double x1144 = x666[x1143];
double x1145 = x679[x1143];
double x1146 = x1144 + x1145;
x666[x1143] = x1146;

}
for(int x1150=0; x1150 < 50; x1150++) {
int32_t x1152 = 26 * x1150;
for(int x1151=0; x1151 < 26; x1151++) {
int32_t x1153 = x1152 + x1151;
double x1154 = x161[x1153];
double x1155 = x356[x1151];
double x1156 = x666[x1150];
double x1157 = x1155 * x1156;
double x1158 = x1154 + x1157;
x161[x1153] = x1158;

}

}
for(int x1164=0; x1164 < 50; x1164++) {
int32_t x1167 = 26 * x1164;
for(int x1165=0; x1165 < 26; x1165++) {
double x1166 = x363[x1165];
int32_t x1168 = x1167 + x1165;
double x1169 = x62[x1168];
double x1170 = x666[x1164];
double x1171 = x1169 * x1170;
double x1172 = x1166 + x1171;
x363[x1165] = x1172;

}

}
for(int x1178=0; x1178 < 50; x1178++) {
int32_t x1180 = 50 * x1178;
for(int x1179=0; x1179 < 50; x1179++) {
int32_t x1181 = x1180 + x1179;
double x1182 = x156[x1181];
double x1184 = x645[x1178];
double x1183 = x351[x1179];
double x1185 = x1183 * x1184;
double x1186 = x1182 + x1185;
x156[x1181] = x1186;

}

}
double* x352 = x348[3];
for(int x1192=0; x1192 < 50; x1192++) {
int32_t x1195 = 50 * x1192;
for(int x1193=0; x1193 < 50; x1193++) {
int32_t x1196 = x1195 + x1193;
double x1197 = x55[x1196];
double x1198 = x645[x1192];
double x1194 = x352[x1193];
double x1199 = x1197 * x1198;
double x1200 = x1194 + x1199;
x352[x1193] = x1200;

}

}
for(int x1206=0; x1206 < 50; x1206++) {
double x1207 = x609[x1206];
double x1208 = x614[x1206];
double x1211 = x624[x1206];
double x1209 = 1.0 - x1208;
double x1210 = x1209 * x1208;
double x1212 = x1210 * x1211;
double x1213 = x1207 + x1212;
x609[x1206] = x1213;

}
for(int x1217=0; x1217 < 50; x1217++) {
double x1218 = x596[x1217];
double x1219 = x609[x1217];
double x1220 = x1218 + x1219;
x596[x1217] = x1220;

}
for(int x1224=0; x1224 < 50; x1224++) {
double x1225 = x181[x1224];
double x1226 = x609[x1224];
double x1227 = x1225 + x1226;
x181[x1224] = x1227;

}
for(int x1231=0; x1231 < 50; x1231++) {
double x1232 = x562[x1231];
double x1233 = x596[x1231];
double x1234 = x1232 + x1233;
x562[x1231] = x1234;

}
for(int x1238=0; x1238 < 50; x1238++) {
double x1239 = x583[x1238];
double x1240 = x596[x1238];
double x1241 = x1239 + x1240;
x583[x1238] = x1241;

}
for(int x1245=0; x1245 < 50; x1245++) {
int32_t x1247 = 26 * x1245;
for(int x1246=0; x1246 < 26; x1246++) {
int32_t x1248 = x1247 + x1246;
double x1249 = x176[x1248];
double x1250 = x356[x1246];
double x1251 = x583[x1245];
double x1252 = x1250 * x1251;
double x1253 = x1249 + x1252;
x176[x1248] = x1253;

}

}
for(int x1259=0; x1259 < 50; x1259++) {
int32_t x1262 = 26 * x1259;
for(int x1260=0; x1260 < 26; x1260++) {
double x1261 = x363[x1260];
int32_t x1263 = x1262 + x1260;
double x1264 = x81[x1263];
double x1265 = x583[x1259];
double x1266 = x1264 * x1265;
double x1267 = x1261 + x1266;
x363[x1260] = x1267;

}

}
for(int x1273=0; x1273 < 50; x1273++) {
int32_t x1275 = 50 * x1273;
for(int x1274=0; x1274 < 50; x1274++) {
int32_t x1276 = x1275 + x1274;
double x1277 = x171[x1276];
double x1279 = x562[x1273];
double x1278 = x351[x1274];
double x1280 = x1278 * x1279;
double x1281 = x1277 + x1280;
x171[x1276] = x1281;

}

}
for(int x1287=0; x1287 < 50; x1287++) {
int32_t x1290 = 50 * x1287;
for(int x1288=0; x1288 < 50; x1288++) {
int32_t x1291 = x1290 + x1288;
double x1292 = x74[x1291];
double x1293 = x562[x1287];
double x1289 = x352[x1288];
double x1294 = x1292 * x1293;
double x1295 = x1289 + x1294;
x352[x1288] = x1295;

}

}
for(int x1301=0; x1301 < 50; x1301++) {
double x1302 = x526[x1301];
double x1303 = x531[x1301];
double x1306 = x541[x1301];
double x1304 = 1.0 - x1303;
double x1305 = x1304 * x1303;
double x1307 = x1305 * x1306;
double x1308 = x1302 + x1307;
x526[x1301] = x1308;

}
for(int x1312=0; x1312 < 50; x1312++) {
double x1313 = x513[x1312];
double x1314 = x526[x1312];
double x1315 = x1313 + x1314;
x513[x1312] = x1315;

}
for(int x1319=0; x1319 < 50; x1319++) {
double x1320 = x151[x1319];
double x1321 = x526[x1319];
double x1322 = x1320 + x1321;
x151[x1319] = x1322;

}
for(int x1326=0; x1326 < 50; x1326++) {
double x1327 = x479[x1326];
double x1328 = x513[x1326];
double x1329 = x1327 + x1328;
x479[x1326] = x1329;

}
for(int x1333=0; x1333 < 50; x1333++) {
double x1334 = x500[x1333];
double x1335 = x513[x1333];
double x1336 = x1334 + x1335;
x500[x1333] = x1336;

}
for(int x1340=0; x1340 < 50; x1340++) {
int32_t x1342 = 26 * x1340;
for(int x1341=0; x1341 < 26; x1341++) {
int32_t x1343 = x1342 + x1341;
double x1344 = x146[x1343];
double x1345 = x356[x1341];
double x1346 = x500[x1340];
double x1347 = x1345 * x1346;
double x1348 = x1344 + x1347;
x146[x1343] = x1348;

}

}
for(int x1354=0; x1354 < 50; x1354++) {
int32_t x1357 = 26 * x1354;
for(int x1355=0; x1355 < 26; x1355++) {
double x1356 = x363[x1355];
int32_t x1358 = x1357 + x1355;
double x1359 = x43[x1358];
double x1360 = x500[x1354];
double x1361 = x1359 * x1360;
double x1362 = x1356 + x1361;
x363[x1355] = x1362;

}

}
for(int x1368=0; x1368 < 50; x1368++) {
int32_t x1370 = 50 * x1368;
for(int x1369=0; x1369 < 50; x1369++) {
int32_t x1371 = x1370 + x1369;
double x1372 = x141[x1371];
double x1374 = x479[x1368];
double x1373 = x351[x1369];
double x1375 = x1373 * x1374;
double x1376 = x1372 + x1375;
x141[x1371] = x1376;

}

}
for(int x1382=0; x1382 < 50; x1382++) {
int32_t x1385 = 50 * x1382;
for(int x1383=0; x1383 < 50; x1383++) {
int32_t x1386 = x1385 + x1383;
double x1387 = x36[x1386];
double x1388 = x479[x1382];
double x1384 = x352[x1383];
double x1389 = x1387 * x1388;
double x1390 = x1384 + x1389;
x352[x1383] = x1390;

}

}
for(int x1396=0; x1396 < 50; x1396++) {
double x1397 = x443[x1396];
double x1398 = x448[x1396];
double x1401 = x458[x1396];
double x1399 = 1.0 - x1398;
double x1400 = x1399 * x1398;
double x1402 = x1400 * x1401;
double x1403 = x1397 + x1402;
x443[x1396] = x1403;

}
for(int x1407=0; x1407 < 50; x1407++) {
double x1408 = x430[x1407];
double x1409 = x443[x1407];
double x1410 = x1408 + x1409;
x430[x1407] = x1410;

}
for(int x1414=0; x1414 < 50; x1414++) {
double x1415 = x136[x1414];
double x1416 = x443[x1414];
double x1417 = x1415 + x1416;
x136[x1414] = x1417;

}
for(int x1421=0; x1421 < 50; x1421++) {
double x1422 = x396[x1421];
double x1423 = x430[x1421];
double x1424 = x1422 + x1423;
x396[x1421] = x1424;

}
for(int x1428=0; x1428 < 50; x1428++) {
double x1429 = x417[x1428];
double x1430 = x430[x1428];
double x1431 = x1429 + x1430;
x417[x1428] = x1431;

}
for(int x1435=0; x1435 < 50; x1435++) {
int32_t x1437 = 26 * x1435;
for(int x1436=0; x1436 < 26; x1436++) {
int32_t x1438 = x1437 + x1436;
double x1439 = x131[x1438];
double x1440 = x356[x1436];
double x1441 = x417[x1435];
double x1442 = x1440 * x1441;
double x1443 = x1439 + x1442;
x131[x1438] = x1443;

}

}
for(int x1449=0; x1449 < 50; x1449++) {
int32_t x1452 = 26 * x1449;
for(int x1450=0; x1450 < 26; x1450++) {
double x1451 = x363[x1450];
int32_t x1453 = x1452 + x1450;
double x1454 = x22[x1453];
double x1455 = x417[x1449];
double x1456 = x1454 * x1455;
double x1457 = x1451 + x1456;
x363[x1450] = x1457;

}

}
for(int x1463=0; x1463 < 50; x1463++) {
int32_t x1465 = 50 * x1463;
for(int x1464=0; x1464 < 50; x1464++) {
int32_t x1466 = x1465 + x1464;
double x1467 = x126[x1466];
double x1469 = x396[x1463];
double x1468 = x351[x1464];
double x1470 = x1468 * x1469;
double x1471 = x1467 + x1470;
x126[x1466] = x1471;

}

}
for(int x1477=0; x1477 < 50; x1477++) {
int32_t x1480 = 50 * x1477;
for(int x1478=0; x1478 < 50; x1478++) {
int32_t x1481 = x1480 + x1478;
double x1482 = x14[x1481];
double x1483 = x396[x1477];
double x1479 = x352[x1478];
double x1484 = x1482 * x1483;
double x1485 = x1479 + x1484;
x352[x1478] = x1485;

}

}
} else {
double** x348 = x346;
double* x351 = x348[2];
for(int x1492=0; x1492 < 50; x1492++) {
double x1493 = x351[x1492];
x116[x1492] = x1493;

}
double* x353 = x348[4];
for(int x1497=0; x1497 < 50; x1497++) {
double x1498 = x353[x1497];
x121[x1497] = x1498;

}
double* x350 = x348[1];
for(int x1502=0; x1502 < 1; x1502++) {
x350[x1502] = 1.0;

}
double* x349 = x348[0];
for(int x1506=0; x1506 < 1; x1506++) {
double x1507 = x349[x1506];
x329[x1506] = x1507;

}
}
};
x288 += 20;
int32_t x295 = x288;
int32_t x296 = x295 + 20;
int32_t x297 = x296 + 1;
bool x298 = x297 >= x2;
if (x298) {
x288 = 0;
for(int x300=0; x300 < 50; x300++) {
x106[x300] = 0.0;

}
} else {
}
for(int x309=0; x309 < 20; x309++) {
int32_t x310 = x288;
int32_t x311 = x310 + x309;
int32_t x312 = x5[x311];
x306[x309] = x312;
int32_t x314 = x311 + 1;
int32_t x315 = x5[x314];
x307[x309] = x315;

}
double* x319 = (double*)myMalloc(1 * sizeof(double));
for(int x320=0; x320 < 1; x320++) {
x319[x320] = 0.0;

}
double* x324 = (double*)myMalloc(1 * sizeof(double));
for(int x325=0; x325 < 1; x325++) {
x324[x325] = 0.0;

}
for(int x330=0; x330 < 1; x330++) {
x329[x330] = 0.0;

}
double* x334 = (double*)myMalloc(1 * sizeof(double));
for(int x335=0; x335 < 1; x335++) {
x334[x335] = 0.0;

}
double* x339 = (double*)myMalloc(1 * sizeof(double));
for(int x340=0; x340 < 1; x340++) {
x339[x340] = 0.0;

}
double** x1514 = (double**)myMalloc(6 * sizeof(double*));
x1514[0] = x334;
x1514[1] = x339;
x1514[2] = x106;
x1514[3] = x196;
x1514[4] = x111;
x1514[5] = x201;
x344(0,x1514);
double x1523 = x329[0];
int32_t x1524 = x293 % 100;
bool x1525 = x1524 == 0;
if (x1525) {
printf("iter %d, loss %f\n",x293,x1523);
} else {
}
for(int x1529=0; x1529 < 2500; x1529++) {
double x1530 = x126[x1529];
bool x1531 = x1530 > 5.0;
if (x1531) {
x126[x1529] = 5.0;
} else {
}
bool x1535 = x1530 < -5.0;
if (x1535) {
x126[x1529] = -5.0;
} else {
}

}
double* x1541 = (double*)myMalloc(2500 * sizeof(double));
for(int x1542=0; x1542 < 2500; x1542++) {
double x1543 = x126[x1542];
double x1544 = x1543 * x1543;
x1541[x1542] = x1544;

}
for(int x1548=0; x1548 < 2500; x1548++) {
double x1549 = x217[x1548];
double x1550 = x1541[x1548];
double x1551 = x1549 + x1550;
x217[x1548] = x1551;

}
double* x1555 = (double*)myMalloc(2500 * sizeof(double));
for(int x1556=0; x1556 < 2500; x1556++) {
double x1557 = x126[x1556];
double x1558 = x206[0];
double x1559 = x1557 * x1558;
x1555[x1556] = x1559;

}
double* x1563 = (double*)myMalloc(2500 * sizeof(double));
for(int x1564=0; x1564 < 2500; x1564++) {
double x1565 = x217[x1564];
double x1566 = x212[0];
double x1567 = x1565 + x1566;
x1563[x1564] = x1567;

}
double* x1571 = (double*)myMalloc(2500 * sizeof(double));
for(int x1572=0; x1572 < 2500; x1572++) {
double x1573 = x1563[x1572];
double x1574 = sqrt(x1573);
x1571[x1572] = x1574;

}
double* x1578 = (double*)myMalloc(2500 * sizeof(double));
for(int x1579=0; x1579 < 2500; x1579++) {
double x1580 = x1555[x1579];
double x1581 = x1571[x1579];
double x1582 = x1580 / x1581;
x1578[x1579] = x1582;

}
for(int x1586=0; x1586 < 2500; x1586++) {
double x1587 = x14[x1586];
double x1588 = x1578[x1586];
double x1589 = x1587 - x1588;
x14[x1586] = x1589;

}
for(int x1593=0; x1593 < 2500; x1593++) {
x126[x1593] = 0.0;

}
for(int x1597=0; x1597 < 1300; x1597++) {
double x1598 = x131[x1597];
bool x1599 = x1598 > 5.0;
if (x1599) {
x131[x1597] = 5.0;
} else {
}
bool x1603 = x1598 < -5.0;
if (x1603) {
x131[x1597] = -5.0;
} else {
}

}
double* x1609 = (double*)myMalloc(1300 * sizeof(double));
for(int x1610=0; x1610 < 1300; x1610++) {
double x1611 = x131[x1610];
double x1612 = x1611 * x1611;
x1609[x1610] = x1612;

}
for(int x1616=0; x1616 < 1300; x1616++) {
double x1617 = x222[x1616];
double x1618 = x1609[x1616];
double x1619 = x1617 + x1618;
x222[x1616] = x1619;

}
double* x1623 = (double*)myMalloc(1300 * sizeof(double));
for(int x1624=0; x1624 < 1300; x1624++) {
double x1625 = x131[x1624];
double x1626 = x206[0];
double x1627 = x1625 * x1626;
x1623[x1624] = x1627;

}
double* x1631 = (double*)myMalloc(1300 * sizeof(double));
for(int x1632=0; x1632 < 1300; x1632++) {
double x1633 = x222[x1632];
double x1634 = x212[0];
double x1635 = x1633 + x1634;
x1631[x1632] = x1635;

}
double* x1639 = (double*)myMalloc(1300 * sizeof(double));
for(int x1640=0; x1640 < 1300; x1640++) {
double x1641 = x1631[x1640];
double x1642 = sqrt(x1641);
x1639[x1640] = x1642;

}
double* x1646 = (double*)myMalloc(1300 * sizeof(double));
for(int x1647=0; x1647 < 1300; x1647++) {
double x1648 = x1623[x1647];
double x1649 = x1639[x1647];
double x1650 = x1648 / x1649;
x1646[x1647] = x1650;

}
for(int x1654=0; x1654 < 1300; x1654++) {
double x1655 = x22[x1654];
double x1656 = x1646[x1654];
double x1657 = x1655 - x1656;
x22[x1654] = x1657;

}
for(int x1661=0; x1661 < 1300; x1661++) {
x131[x1661] = 0.0;

}
for(int x1665=0; x1665 < 50; x1665++) {
double x1666 = x136[x1665];
bool x1667 = x1666 > 5.0;
if (x1667) {
x136[x1665] = 5.0;
} else {
}
bool x1671 = x1666 < -5.0;
if (x1671) {
x136[x1665] = -5.0;
} else {
}

}
double* x1677 = (double*)myMalloc(50 * sizeof(double));
for(int x1678=0; x1678 < 50; x1678++) {
double x1679 = x136[x1678];
double x1680 = x1679 * x1679;
x1677[x1678] = x1680;

}
for(int x1684=0; x1684 < 50; x1684++) {
double x1685 = x227[x1684];
double x1686 = x1677[x1684];
double x1687 = x1685 + x1686;
x227[x1684] = x1687;

}
double* x1691 = (double*)myMalloc(50 * sizeof(double));
for(int x1692=0; x1692 < 50; x1692++) {
double x1693 = x136[x1692];
double x1694 = x206[0];
double x1695 = x1693 * x1694;
x1691[x1692] = x1695;

}
double* x1699 = (double*)myMalloc(50 * sizeof(double));
for(int x1700=0; x1700 < 50; x1700++) {
double x1701 = x227[x1700];
double x1702 = x212[0];
double x1703 = x1701 + x1702;
x1699[x1700] = x1703;

}
double* x1707 = (double*)myMalloc(50 * sizeof(double));
for(int x1708=0; x1708 < 50; x1708++) {
double x1709 = x1699[x1708];
double x1710 = sqrt(x1709);
x1707[x1708] = x1710;

}
double* x1714 = (double*)myMalloc(50 * sizeof(double));
for(int x1715=0; x1715 < 50; x1715++) {
double x1716 = x1691[x1715];
double x1717 = x1707[x1715];
double x1718 = x1716 / x1717;
x1714[x1715] = x1718;

}
for(int x1722=0; x1722 < 50; x1722++) {
double x1723 = x30[x1722];
double x1724 = x1714[x1722];
double x1725 = x1723 - x1724;
x30[x1722] = x1725;

}
for(int x1729=0; x1729 < 50; x1729++) {
x136[x1729] = 0.0;

}
for(int x1733=0; x1733 < 2500; x1733++) {
double x1734 = x141[x1733];
bool x1735 = x1734 > 5.0;
if (x1735) {
x141[x1733] = 5.0;
} else {
}
bool x1739 = x1734 < -5.0;
if (x1739) {
x141[x1733] = -5.0;
} else {
}

}
double* x1745 = (double*)myMalloc(2500 * sizeof(double));
for(int x1746=0; x1746 < 2500; x1746++) {
double x1747 = x141[x1746];
double x1748 = x1747 * x1747;
x1745[x1746] = x1748;

}
for(int x1752=0; x1752 < 2500; x1752++) {
double x1753 = x232[x1752];
double x1754 = x1745[x1752];
double x1755 = x1753 + x1754;
x232[x1752] = x1755;

}
double* x1759 = (double*)myMalloc(2500 * sizeof(double));
for(int x1760=0; x1760 < 2500; x1760++) {
double x1761 = x141[x1760];
double x1762 = x206[0];
double x1763 = x1761 * x1762;
x1759[x1760] = x1763;

}
double* x1767 = (double*)myMalloc(2500 * sizeof(double));
for(int x1768=0; x1768 < 2500; x1768++) {
double x1769 = x232[x1768];
double x1770 = x212[0];
double x1771 = x1769 + x1770;
x1767[x1768] = x1771;

}
double* x1775 = (double*)myMalloc(2500 * sizeof(double));
for(int x1776=0; x1776 < 2500; x1776++) {
double x1777 = x1767[x1776];
double x1778 = sqrt(x1777);
x1775[x1776] = x1778;

}
double* x1782 = (double*)myMalloc(2500 * sizeof(double));
for(int x1783=0; x1783 < 2500; x1783++) {
double x1784 = x1759[x1783];
double x1785 = x1775[x1783];
double x1786 = x1784 / x1785;
x1782[x1783] = x1786;

}
for(int x1790=0; x1790 < 2500; x1790++) {
double x1791 = x36[x1790];
double x1792 = x1782[x1790];
double x1793 = x1791 - x1792;
x36[x1790] = x1793;

}
for(int x1797=0; x1797 < 2500; x1797++) {
x141[x1797] = 0.0;

}
for(int x1801=0; x1801 < 1300; x1801++) {
double x1802 = x146[x1801];
bool x1803 = x1802 > 5.0;
if (x1803) {
x146[x1801] = 5.0;
} else {
}
bool x1807 = x1802 < -5.0;
if (x1807) {
x146[x1801] = -5.0;
} else {
}

}
double* x1813 = (double*)myMalloc(1300 * sizeof(double));
for(int x1814=0; x1814 < 1300; x1814++) {
double x1815 = x146[x1814];
double x1816 = x1815 * x1815;
x1813[x1814] = x1816;

}
for(int x1820=0; x1820 < 1300; x1820++) {
double x1821 = x237[x1820];
double x1822 = x1813[x1820];
double x1823 = x1821 + x1822;
x237[x1820] = x1823;

}
double* x1827 = (double*)myMalloc(1300 * sizeof(double));
for(int x1828=0; x1828 < 1300; x1828++) {
double x1829 = x146[x1828];
double x1830 = x206[0];
double x1831 = x1829 * x1830;
x1827[x1828] = x1831;

}
double* x1835 = (double*)myMalloc(1300 * sizeof(double));
for(int x1836=0; x1836 < 1300; x1836++) {
double x1837 = x237[x1836];
double x1838 = x212[0];
double x1839 = x1837 + x1838;
x1835[x1836] = x1839;

}
double* x1843 = (double*)myMalloc(1300 * sizeof(double));
for(int x1844=0; x1844 < 1300; x1844++) {
double x1845 = x1835[x1844];
double x1846 = sqrt(x1845);
x1843[x1844] = x1846;

}
double* x1850 = (double*)myMalloc(1300 * sizeof(double));
for(int x1851=0; x1851 < 1300; x1851++) {
double x1852 = x1827[x1851];
double x1853 = x1843[x1851];
double x1854 = x1852 / x1853;
x1850[x1851] = x1854;

}
for(int x1858=0; x1858 < 1300; x1858++) {
double x1859 = x43[x1858];
double x1860 = x1850[x1858];
double x1861 = x1859 - x1860;
x43[x1858] = x1861;

}
for(int x1865=0; x1865 < 1300; x1865++) {
x146[x1865] = 0.0;

}
for(int x1869=0; x1869 < 50; x1869++) {
double x1870 = x151[x1869];
bool x1871 = x1870 > 5.0;
if (x1871) {
x151[x1869] = 5.0;
} else {
}
bool x1875 = x1870 < -5.0;
if (x1875) {
x151[x1869] = -5.0;
} else {
}

}
double* x1881 = (double*)myMalloc(50 * sizeof(double));
for(int x1882=0; x1882 < 50; x1882++) {
double x1883 = x151[x1882];
double x1884 = x1883 * x1883;
x1881[x1882] = x1884;

}
for(int x1888=0; x1888 < 50; x1888++) {
double x1889 = x242[x1888];
double x1890 = x1881[x1888];
double x1891 = x1889 + x1890;
x242[x1888] = x1891;

}
double* x1895 = (double*)myMalloc(50 * sizeof(double));
for(int x1896=0; x1896 < 50; x1896++) {
double x1897 = x151[x1896];
double x1898 = x206[0];
double x1899 = x1897 * x1898;
x1895[x1896] = x1899;

}
double* x1903 = (double*)myMalloc(50 * sizeof(double));
for(int x1904=0; x1904 < 50; x1904++) {
double x1905 = x242[x1904];
double x1906 = x212[0];
double x1907 = x1905 + x1906;
x1903[x1904] = x1907;

}
double* x1911 = (double*)myMalloc(50 * sizeof(double));
for(int x1912=0; x1912 < 50; x1912++) {
double x1913 = x1903[x1912];
double x1914 = sqrt(x1913);
x1911[x1912] = x1914;

}
double* x1918 = (double*)myMalloc(50 * sizeof(double));
for(int x1919=0; x1919 < 50; x1919++) {
double x1920 = x1895[x1919];
double x1921 = x1911[x1919];
double x1922 = x1920 / x1921;
x1918[x1919] = x1922;

}
for(int x1926=0; x1926 < 50; x1926++) {
double x1927 = x50[x1926];
double x1928 = x1918[x1926];
double x1929 = x1927 - x1928;
x50[x1926] = x1929;

}
for(int x1933=0; x1933 < 50; x1933++) {
x151[x1933] = 0.0;

}
for(int x1937=0; x1937 < 2500; x1937++) {
double x1938 = x156[x1937];
bool x1939 = x1938 > 5.0;
if (x1939) {
x156[x1937] = 5.0;
} else {
}
bool x1943 = x1938 < -5.0;
if (x1943) {
x156[x1937] = -5.0;
} else {
}

}
double* x1949 = (double*)myMalloc(2500 * sizeof(double));
for(int x1950=0; x1950 < 2500; x1950++) {
double x1951 = x156[x1950];
double x1952 = x1951 * x1951;
x1949[x1950] = x1952;

}
for(int x1956=0; x1956 < 2500; x1956++) {
double x1957 = x247[x1956];
double x1958 = x1949[x1956];
double x1959 = x1957 + x1958;
x247[x1956] = x1959;

}
double* x1963 = (double*)myMalloc(2500 * sizeof(double));
for(int x1964=0; x1964 < 2500; x1964++) {
double x1965 = x156[x1964];
double x1966 = x206[0];
double x1967 = x1965 * x1966;
x1963[x1964] = x1967;

}
double* x1971 = (double*)myMalloc(2500 * sizeof(double));
for(int x1972=0; x1972 < 2500; x1972++) {
double x1973 = x247[x1972];
double x1974 = x212[0];
double x1975 = x1973 + x1974;
x1971[x1972] = x1975;

}
double* x1979 = (double*)myMalloc(2500 * sizeof(double));
for(int x1980=0; x1980 < 2500; x1980++) {
double x1981 = x1971[x1980];
double x1982 = sqrt(x1981);
x1979[x1980] = x1982;

}
double* x1986 = (double*)myMalloc(2500 * sizeof(double));
for(int x1987=0; x1987 < 2500; x1987++) {
double x1988 = x1963[x1987];
double x1989 = x1979[x1987];
double x1990 = x1988 / x1989;
x1986[x1987] = x1990;

}
for(int x1994=0; x1994 < 2500; x1994++) {
double x1995 = x55[x1994];
double x1996 = x1986[x1994];
double x1997 = x1995 - x1996;
x55[x1994] = x1997;

}
for(int x2001=0; x2001 < 2500; x2001++) {
x156[x2001] = 0.0;

}
for(int x2005=0; x2005 < 1300; x2005++) {
double x2006 = x161[x2005];
bool x2007 = x2006 > 5.0;
if (x2007) {
x161[x2005] = 5.0;
} else {
}
bool x2011 = x2006 < -5.0;
if (x2011) {
x161[x2005] = -5.0;
} else {
}

}
double* x2017 = (double*)myMalloc(1300 * sizeof(double));
for(int x2018=0; x2018 < 1300; x2018++) {
double x2019 = x161[x2018];
double x2020 = x2019 * x2019;
x2017[x2018] = x2020;

}
for(int x2024=0; x2024 < 1300; x2024++) {
double x2025 = x252[x2024];
double x2026 = x2017[x2024];
double x2027 = x2025 + x2026;
x252[x2024] = x2027;

}
double* x2031 = (double*)myMalloc(1300 * sizeof(double));
for(int x2032=0; x2032 < 1300; x2032++) {
double x2033 = x161[x2032];
double x2034 = x206[0];
double x2035 = x2033 * x2034;
x2031[x2032] = x2035;

}
double* x2039 = (double*)myMalloc(1300 * sizeof(double));
for(int x2040=0; x2040 < 1300; x2040++) {
double x2041 = x252[x2040];
double x2042 = x212[0];
double x2043 = x2041 + x2042;
x2039[x2040] = x2043;

}
double* x2047 = (double*)myMalloc(1300 * sizeof(double));
for(int x2048=0; x2048 < 1300; x2048++) {
double x2049 = x2039[x2048];
double x2050 = sqrt(x2049);
x2047[x2048] = x2050;

}
double* x2054 = (double*)myMalloc(1300 * sizeof(double));
for(int x2055=0; x2055 < 1300; x2055++) {
double x2056 = x2031[x2055];
double x2057 = x2047[x2055];
double x2058 = x2056 / x2057;
x2054[x2055] = x2058;

}
for(int x2062=0; x2062 < 1300; x2062++) {
double x2063 = x62[x2062];
double x2064 = x2054[x2062];
double x2065 = x2063 - x2064;
x62[x2062] = x2065;

}
for(int x2069=0; x2069 < 1300; x2069++) {
x161[x2069] = 0.0;

}
for(int x2073=0; x2073 < 50; x2073++) {
double x2074 = x166[x2073];
bool x2075 = x2074 > 5.0;
if (x2075) {
x166[x2073] = 5.0;
} else {
}
bool x2079 = x2074 < -5.0;
if (x2079) {
x166[x2073] = -5.0;
} else {
}

}
double* x2085 = (double*)myMalloc(50 * sizeof(double));
for(int x2086=0; x2086 < 50; x2086++) {
double x2087 = x166[x2086];
double x2088 = x2087 * x2087;
x2085[x2086] = x2088;

}
for(int x2092=0; x2092 < 50; x2092++) {
double x2093 = x257[x2092];
double x2094 = x2085[x2092];
double x2095 = x2093 + x2094;
x257[x2092] = x2095;

}
double* x2099 = (double*)myMalloc(50 * sizeof(double));
for(int x2100=0; x2100 < 50; x2100++) {
double x2101 = x166[x2100];
double x2102 = x206[0];
double x2103 = x2101 * x2102;
x2099[x2100] = x2103;

}
double* x2107 = (double*)myMalloc(50 * sizeof(double));
for(int x2108=0; x2108 < 50; x2108++) {
double x2109 = x257[x2108];
double x2110 = x212[0];
double x2111 = x2109 + x2110;
x2107[x2108] = x2111;

}
double* x2115 = (double*)myMalloc(50 * sizeof(double));
for(int x2116=0; x2116 < 50; x2116++) {
double x2117 = x2107[x2116];
double x2118 = sqrt(x2117);
x2115[x2116] = x2118;

}
double* x2122 = (double*)myMalloc(50 * sizeof(double));
for(int x2123=0; x2123 < 50; x2123++) {
double x2124 = x2099[x2123];
double x2125 = x2115[x2123];
double x2126 = x2124 / x2125;
x2122[x2123] = x2126;

}
for(int x2130=0; x2130 < 50; x2130++) {
double x2131 = x69[x2130];
double x2132 = x2122[x2130];
double x2133 = x2131 - x2132;
x69[x2130] = x2133;

}
for(int x2137=0; x2137 < 50; x2137++) {
x166[x2137] = 0.0;

}
for(int x2141=0; x2141 < 2500; x2141++) {
double x2142 = x171[x2141];
bool x2143 = x2142 > 5.0;
if (x2143) {
x171[x2141] = 5.0;
} else {
}
bool x2147 = x2142 < -5.0;
if (x2147) {
x171[x2141] = -5.0;
} else {
}

}
double* x2153 = (double*)myMalloc(2500 * sizeof(double));
for(int x2154=0; x2154 < 2500; x2154++) {
double x2155 = x171[x2154];
double x2156 = x2155 * x2155;
x2153[x2154] = x2156;

}
for(int x2160=0; x2160 < 2500; x2160++) {
double x2161 = x262[x2160];
double x2162 = x2153[x2160];
double x2163 = x2161 + x2162;
x262[x2160] = x2163;

}
double* x2167 = (double*)myMalloc(2500 * sizeof(double));
for(int x2168=0; x2168 < 2500; x2168++) {
double x2169 = x171[x2168];
double x2170 = x206[0];
double x2171 = x2169 * x2170;
x2167[x2168] = x2171;

}
double* x2175 = (double*)myMalloc(2500 * sizeof(double));
for(int x2176=0; x2176 < 2500; x2176++) {
double x2177 = x262[x2176];
double x2178 = x212[0];
double x2179 = x2177 + x2178;
x2175[x2176] = x2179;

}
double* x2183 = (double*)myMalloc(2500 * sizeof(double));
for(int x2184=0; x2184 < 2500; x2184++) {
double x2185 = x2175[x2184];
double x2186 = sqrt(x2185);
x2183[x2184] = x2186;

}
double* x2190 = (double*)myMalloc(2500 * sizeof(double));
for(int x2191=0; x2191 < 2500; x2191++) {
double x2192 = x2167[x2191];
double x2193 = x2183[x2191];
double x2194 = x2192 / x2193;
x2190[x2191] = x2194;

}
for(int x2198=0; x2198 < 2500; x2198++) {
double x2199 = x74[x2198];
double x2200 = x2190[x2198];
double x2201 = x2199 - x2200;
x74[x2198] = x2201;

}
for(int x2205=0; x2205 < 2500; x2205++) {
x171[x2205] = 0.0;

}
for(int x2209=0; x2209 < 1300; x2209++) {
double x2210 = x176[x2209];
bool x2211 = x2210 > 5.0;
if (x2211) {
x176[x2209] = 5.0;
} else {
}
bool x2215 = x2210 < -5.0;
if (x2215) {
x176[x2209] = -5.0;
} else {
}

}
double* x2221 = (double*)myMalloc(1300 * sizeof(double));
for(int x2222=0; x2222 < 1300; x2222++) {
double x2223 = x176[x2222];
double x2224 = x2223 * x2223;
x2221[x2222] = x2224;

}
for(int x2228=0; x2228 < 1300; x2228++) {
double x2229 = x267[x2228];
double x2230 = x2221[x2228];
double x2231 = x2229 + x2230;
x267[x2228] = x2231;

}
double* x2235 = (double*)myMalloc(1300 * sizeof(double));
for(int x2236=0; x2236 < 1300; x2236++) {
double x2237 = x176[x2236];
double x2238 = x206[0];
double x2239 = x2237 * x2238;
x2235[x2236] = x2239;

}
double* x2243 = (double*)myMalloc(1300 * sizeof(double));
for(int x2244=0; x2244 < 1300; x2244++) {
double x2245 = x267[x2244];
double x2246 = x212[0];
double x2247 = x2245 + x2246;
x2243[x2244] = x2247;

}
double* x2251 = (double*)myMalloc(1300 * sizeof(double));
for(int x2252=0; x2252 < 1300; x2252++) {
double x2253 = x2243[x2252];
double x2254 = sqrt(x2253);
x2251[x2252] = x2254;

}
double* x2258 = (double*)myMalloc(1300 * sizeof(double));
for(int x2259=0; x2259 < 1300; x2259++) {
double x2260 = x2235[x2259];
double x2261 = x2251[x2259];
double x2262 = x2260 / x2261;
x2258[x2259] = x2262;

}
for(int x2266=0; x2266 < 1300; x2266++) {
double x2267 = x81[x2266];
double x2268 = x2258[x2266];
double x2269 = x2267 - x2268;
x81[x2266] = x2269;

}
for(int x2273=0; x2273 < 1300; x2273++) {
x176[x2273] = 0.0;

}
for(int x2277=0; x2277 < 50; x2277++) {
double x2278 = x181[x2277];
bool x2279 = x2278 > 5.0;
if (x2279) {
x181[x2277] = 5.0;
} else {
}
bool x2283 = x2278 < -5.0;
if (x2283) {
x181[x2277] = -5.0;
} else {
}

}
double* x2289 = (double*)myMalloc(50 * sizeof(double));
for(int x2290=0; x2290 < 50; x2290++) {
double x2291 = x181[x2290];
double x2292 = x2291 * x2291;
x2289[x2290] = x2292;

}
for(int x2296=0; x2296 < 50; x2296++) {
double x2297 = x272[x2296];
double x2298 = x2289[x2296];
double x2299 = x2297 + x2298;
x272[x2296] = x2299;

}
double* x2303 = (double*)myMalloc(50 * sizeof(double));
for(int x2304=0; x2304 < 50; x2304++) {
double x2305 = x181[x2304];
double x2306 = x206[0];
double x2307 = x2305 * x2306;
x2303[x2304] = x2307;

}
double* x2311 = (double*)myMalloc(50 * sizeof(double));
for(int x2312=0; x2312 < 50; x2312++) {
double x2313 = x272[x2312];
double x2314 = x212[0];
double x2315 = x2313 + x2314;
x2311[x2312] = x2315;

}
double* x2319 = (double*)myMalloc(50 * sizeof(double));
for(int x2320=0; x2320 < 50; x2320++) {
double x2321 = x2311[x2320];
double x2322 = sqrt(x2321);
x2319[x2320] = x2322;

}
double* x2326 = (double*)myMalloc(50 * sizeof(double));
for(int x2327=0; x2327 < 50; x2327++) {
double x2328 = x2303[x2327];
double x2329 = x2319[x2327];
double x2330 = x2328 / x2329;
x2326[x2327] = x2330;

}
for(int x2334=0; x2334 < 50; x2334++) {
double x2335 = x88[x2334];
double x2336 = x2326[x2334];
double x2337 = x2335 - x2336;
x88[x2334] = x2337;

}
for(int x2341=0; x2341 < 50; x2341++) {
x181[x2341] = 0.0;

}
for(int x2345=0; x2345 < 1300; x2345++) {
double x2346 = x186[x2345];
bool x2347 = x2346 > 5.0;
if (x2347) {
x186[x2345] = 5.0;
} else {
}
bool x2351 = x2346 < -5.0;
if (x2351) {
x186[x2345] = -5.0;
} else {
}

}
double* x2357 = (double*)myMalloc(1300 * sizeof(double));
for(int x2358=0; x2358 < 1300; x2358++) {
double x2359 = x186[x2358];
double x2360 = x2359 * x2359;
x2357[x2358] = x2360;

}
for(int x2364=0; x2364 < 1300; x2364++) {
double x2365 = x277[x2364];
double x2366 = x2357[x2364];
double x2367 = x2365 + x2366;
x277[x2364] = x2367;

}
double* x2371 = (double*)myMalloc(1300 * sizeof(double));
for(int x2372=0; x2372 < 1300; x2372++) {
double x2373 = x186[x2372];
double x2374 = x206[0];
double x2375 = x2373 * x2374;
x2371[x2372] = x2375;

}
double* x2379 = (double*)myMalloc(1300 * sizeof(double));
for(int x2380=0; x2380 < 1300; x2380++) {
double x2381 = x277[x2380];
double x2382 = x212[0];
double x2383 = x2381 + x2382;
x2379[x2380] = x2383;

}
double* x2387 = (double*)myMalloc(1300 * sizeof(double));
for(int x2388=0; x2388 < 1300; x2388++) {
double x2389 = x2379[x2388];
double x2390 = sqrt(x2389);
x2387[x2388] = x2390;

}
double* x2394 = (double*)myMalloc(1300 * sizeof(double));
for(int x2395=0; x2395 < 1300; x2395++) {
double x2396 = x2371[x2395];
double x2397 = x2387[x2395];
double x2398 = x2396 / x2397;
x2394[x2395] = x2398;

}
for(int x2402=0; x2402 < 1300; x2402++) {
double x2403 = x93[x2402];
double x2404 = x2394[x2402];
double x2405 = x2403 - x2404;
x93[x2402] = x2405;

}
for(int x2409=0; x2409 < 1300; x2409++) {
x186[x2409] = 0.0;

}
for(int x2413=0; x2413 < 26; x2413++) {
double x2414 = x191[x2413];
bool x2415 = x2414 > 5.0;
if (x2415) {
x191[x2413] = 5.0;
} else {
}
bool x2419 = x2414 < -5.0;
if (x2419) {
x191[x2413] = -5.0;
} else {
}

}
double* x2425 = (double*)myMalloc(26 * sizeof(double));
for(int x2426=0; x2426 < 26; x2426++) {
double x2427 = x191[x2426];
double x2428 = x2427 * x2427;
x2425[x2426] = x2428;

}
for(int x2432=0; x2432 < 26; x2432++) {
double x2433 = x282[x2432];
double x2434 = x2425[x2432];
double x2435 = x2433 + x2434;
x282[x2432] = x2435;

}
double* x2439 = (double*)myMalloc(26 * sizeof(double));
for(int x2440=0; x2440 < 26; x2440++) {
double x2441 = x191[x2440];
double x2442 = x206[0];
double x2443 = x2441 * x2442;
x2439[x2440] = x2443;

}
double* x2447 = (double*)myMalloc(26 * sizeof(double));
for(int x2448=0; x2448 < 26; x2448++) {
double x2449 = x282[x2448];
double x2450 = x212[0];
double x2451 = x2449 + x2450;
x2447[x2448] = x2451;

}
double* x2455 = (double*)myMalloc(26 * sizeof(double));
for(int x2456=0; x2456 < 26; x2456++) {
double x2457 = x2447[x2456];
double x2458 = sqrt(x2457);
x2455[x2456] = x2458;

}
double* x2462 = (double*)myMalloc(26 * sizeof(double));
for(int x2463=0; x2463 < 26; x2463++) {
double x2464 = x2439[x2463];
double x2465 = x2455[x2463];
double x2466 = x2464 / x2465;
x2462[x2463] = x2466;

}
for(int x2470=0; x2470 < 26; x2470++) {
double x2471 = x100[x2470];
double x2472 = x2462[x2470];
double x2473 = x2471 - x2472;
x100[x2470] = x2473;

}
for(int x2477=0; x2477 < 26; x2477++) {
x191[x2477] = 0.0;

}
for(int x2481=0; x2481 < 50; x2481++) {
x196[x2481] = 0.0;

}
for(int x2485=0; x2485 < 50; x2485++) {
x201[x2485] = 0.0;

}
for(int x2489=0; x2489 < 50; x2489++) {
double x2490 = x116[x2489];
x106[x2489] = x2490;

}
for(int x2494=0; x2494 < 50; x2494++) {
double x2495 = x121[x2494];
x111[x2494] = x2495;

}
mallocAddr = (void*)x287;

}
}
/*****************************************
  End of C Generated Code                  
*******************************************/

