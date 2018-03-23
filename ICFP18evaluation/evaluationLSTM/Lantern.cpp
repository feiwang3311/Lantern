
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
double* x15 = (double*)myMalloc(2500 * sizeof(double));
for(int x17=0; x17 < 2500; x17++) {
double x18 = d(gen);
double x19 = x18 * 0.01;
x15[x17] = x19;

}
double* x23 = (double*)myMalloc(1300 * sizeof(double));
for(int x25=0; x25 < 1300; x25++) {
double x26 = d(gen);
double x27 = x26 * 0.01;
x23[x25] = x27;

}
double* x31 = (double*)myMalloc(50 * sizeof(double));
for(int x33=0; x33 < 50; x33++) {
x31[x33] = 0.0;

}
double* x37 = (double*)myMalloc(2500 * sizeof(double));
for(int x38=0; x38 < 2500; x38++) {
double x39 = d(gen);
double x40 = x39 * 0.01;
x37[x38] = x40;

}
double* x44 = (double*)myMalloc(1300 * sizeof(double));
for(int x45=0; x45 < 1300; x45++) {
double x46 = d(gen);
double x47 = x46 * 0.01;
x44[x45] = x47;

}
double* x51 = (double*)myMalloc(50 * sizeof(double));
for(int x52=0; x52 < 50; x52++) {
x51[x52] = 0.0;

}
double* x56 = (double*)myMalloc(2500 * sizeof(double));
for(int x57=0; x57 < 2500; x57++) {
double x58 = d(gen);
double x59 = x58 * 0.01;
x56[x57] = x59;

}
double* x63 = (double*)myMalloc(1300 * sizeof(double));
for(int x64=0; x64 < 1300; x64++) {
double x65 = d(gen);
double x66 = x65 * 0.01;
x63[x64] = x66;

}
double* x70 = (double*)myMalloc(50 * sizeof(double));
for(int x71=0; x71 < 50; x71++) {
x70[x71] = 0.0;

}
double* x75 = (double*)myMalloc(2500 * sizeof(double));
for(int x76=0; x76 < 2500; x76++) {
double x77 = d(gen);
double x78 = x77 * 0.01;
x75[x76] = x78;

}
double* x82 = (double*)myMalloc(1300 * sizeof(double));
for(int x83=0; x83 < 1300; x83++) {
double x84 = d(gen);
double x85 = x84 * 0.01;
x82[x83] = x85;

}
double* x89 = (double*)myMalloc(50 * sizeof(double));
for(int x90=0; x90 < 50; x90++) {
x89[x90] = 0.0;

}
double* x94 = (double*)myMalloc(1300 * sizeof(double));
for(int x95=0; x95 < 1300; x95++) {
double x96 = d(gen);
double x97 = x96 * 0.01;
x94[x95] = x97;

}
double* x101 = (double*)myMalloc(26 * sizeof(double));
for(int x103=0; x103 < 26; x103++) {
x101[x103] = 0.0;

}
double* x107 = (double*)myMalloc(50 * sizeof(double));
for(int x108=0; x108 < 50; x108++) {
x107[x108] = 0.0;

}
double* x112 = (double*)myMalloc(50 * sizeof(double));
for(int x113=0; x113 < 50; x113++) {
x112[x113] = 0.0;

}
double* x117 = (double*)myMalloc(50 * sizeof(double));
for(int x118=0; x118 < 50; x118++) {
x117[x118] = 0.0;

}
double* x122 = (double*)myMalloc(50 * sizeof(double));
for(int x123=0; x123 < 50; x123++) {
x122[x123] = 0.0;

}
double* x127 = (double*)myMalloc(2500 * sizeof(double));
for(int x128=0; x128 < 2500; x128++) {
x127[x128] = 0.0;

}
double* x132 = (double*)myMalloc(1300 * sizeof(double));
for(int x133=0; x133 < 1300; x133++) {
x132[x133] = 0.0;

}
double* x137 = (double*)myMalloc(50 * sizeof(double));
for(int x138=0; x138 < 50; x138++) {
x137[x138] = 0.0;

}
double* x142 = (double*)myMalloc(2500 * sizeof(double));
for(int x143=0; x143 < 2500; x143++) {
x142[x143] = 0.0;

}
double* x147 = (double*)myMalloc(1300 * sizeof(double));
for(int x148=0; x148 < 1300; x148++) {
x147[x148] = 0.0;

}
double* x152 = (double*)myMalloc(50 * sizeof(double));
for(int x153=0; x153 < 50; x153++) {
x152[x153] = 0.0;

}
double* x157 = (double*)myMalloc(2500 * sizeof(double));
for(int x158=0; x158 < 2500; x158++) {
x157[x158] = 0.0;

}
double* x162 = (double*)myMalloc(1300 * sizeof(double));
for(int x163=0; x163 < 1300; x163++) {
x162[x163] = 0.0;

}
double* x167 = (double*)myMalloc(50 * sizeof(double));
for(int x168=0; x168 < 50; x168++) {
x167[x168] = 0.0;

}
double* x172 = (double*)myMalloc(2500 * sizeof(double));
for(int x173=0; x173 < 2500; x173++) {
x172[x173] = 0.0;

}
double* x177 = (double*)myMalloc(1300 * sizeof(double));
for(int x178=0; x178 < 1300; x178++) {
x177[x178] = 0.0;

}
double* x182 = (double*)myMalloc(50 * sizeof(double));
for(int x183=0; x183 < 50; x183++) {
x182[x183] = 0.0;

}
double* x187 = (double*)myMalloc(1300 * sizeof(double));
for(int x188=0; x188 < 1300; x188++) {
x187[x188] = 0.0;

}
double* x192 = (double*)myMalloc(26 * sizeof(double));
for(int x193=0; x193 < 26; x193++) {
x192[x193] = 0.0;

}
double* x197 = (double*)myMalloc(50 * sizeof(double));
for(int x198=0; x198 < 50; x198++) {
x197[x198] = 0.0;

}
double* x202 = (double*)myMalloc(50 * sizeof(double));
for(int x203=0; x203 < 50; x203++) {
x202[x203] = 0.0;

}
double* x207 = (double*)myMalloc(2500 * sizeof(double));
for(int x208=0; x208 < 2500; x208++) {
x207[x208] = 0.0;

}
double* x212 = (double*)myMalloc(1300 * sizeof(double));
for(int x213=0; x213 < 1300; x213++) {
x212[x213] = 0.0;

}
double* x217 = (double*)myMalloc(50 * sizeof(double));
for(int x218=0; x218 < 50; x218++) {
x217[x218] = 0.0;

}
double* x222 = (double*)myMalloc(2500 * sizeof(double));
for(int x223=0; x223 < 2500; x223++) {
x222[x223] = 0.0;

}
double* x227 = (double*)myMalloc(1300 * sizeof(double));
for(int x228=0; x228 < 1300; x228++) {
x227[x228] = 0.0;

}
double* x232 = (double*)myMalloc(50 * sizeof(double));
for(int x233=0; x233 < 50; x233++) {
x232[x233] = 0.0;

}
double* x237 = (double*)myMalloc(2500 * sizeof(double));
for(int x238=0; x238 < 2500; x238++) {
x237[x238] = 0.0;

}
double* x242 = (double*)myMalloc(1300 * sizeof(double));
for(int x243=0; x243 < 1300; x243++) {
x242[x243] = 0.0;

}
double* x247 = (double*)myMalloc(50 * sizeof(double));
for(int x248=0; x248 < 50; x248++) {
x247[x248] = 0.0;

}
double* x252 = (double*)myMalloc(2500 * sizeof(double));
for(int x253=0; x253 < 2500; x253++) {
x252[x253] = 0.0;

}
double* x257 = (double*)myMalloc(1300 * sizeof(double));
for(int x258=0; x258 < 1300; x258++) {
x257[x258] = 0.0;

}
double* x262 = (double*)myMalloc(50 * sizeof(double));
for(int x263=0; x263 < 50; x263++) {
x262[x263] = 0.0;

}
double* x267 = (double*)myMalloc(1300 * sizeof(double));
for(int x268=0; x268 < 1300; x268++) {
x267[x268] = 0.0;

}
double* x272 = (double*)myMalloc(26 * sizeof(double));
for(int x273=0; x273 < 26; x273++) {
x272[x273] = 0.0;

}
double x277 = ((double)clock() / CLOCKS_PER_SEC);
double* x278 = (double*)myMalloc(51 * sizeof(double));
int64_t x279 = (long)mallocAddr;
int32_t x280 = 0;
x280 -= 20;
double x282 = 70.0;
for(int x284=0; x284 < 5001; x284++) {
double* x322 = (double*)myMalloc(1 * sizeof(double));
int32_t* x298 = (int32_t*)myMalloc(20 * sizeof(int32_t));
int32_t* x299 = (int32_t*)myMalloc(20 * sizeof(int32_t));
function<void(int32_t,double**)> x337 = [&](int32_t x338,double** x339) {
double** x341 = x339;
double* x342 = x341[0];
double* x343 = x341[1];
double* x344 = x341[2];
double* x345 = x341[3];
double* x346 = x341[4];
double* x347 = x341[5];
int32_t x340 = x338;
bool x348 = x340 < 20;
if (x348) {
double* x349 = (double*)myMalloc(26 * sizeof(double));
for(int x350=0; x350 < 26; x350++) {
x349[x350] = 0.0;

}
int32_t x354 = x298[x340];
x349[x354] = 1.0;
double* x356 = (double*)myMalloc(26 * sizeof(double));
for(int x357=0; x357 < 26; x357++) {
x356[x357] = 0.0;

}
double* x361 = (double*)myMalloc(26 * sizeof(double));
for(int x362=0; x362 < 26; x362++) {
x361[x362] = 0.0;

}
int32_t x366 = x299[x340];
x361[x366] = 1.0;
double* x368 = (double*)myMalloc(26 * sizeof(double));
for(int x369=0; x369 < 26; x369++) {
x368[x369] = 0.0;

}
// dot WrappedArray(50, 50) - WrappedArray(50)
int32_t x374 = 0;
double* x375 = (double*)myMalloc(50 * sizeof(double));
for(int x376=0; x376 < 50; x376++) {
double x377 = 0.0;
for(int x378=0; x378 < 50; x378++) {
int32_t x379 = x374;
double x380 = x15[x379];
double x381 = x344[x378];
double x382 = x380 * x381;
x377 += x382;
x374 += 1;

}
double x387 = x377;
x375[x376] = x387;

}
double* x391 = (double*)myMalloc(50 * sizeof(double));
for(int x392=0; x392 < 50; x392++) {
x391[x392] = 0.0;

}
// dot WrappedArray(50, 26) - WrappedArray(26)
int32_t x397 = 0;
double* x398 = (double*)myMalloc(50 * sizeof(double));
for(int x399=0; x399 < 50; x399++) {
double x400 = 0.0;
for(int x401=0; x401 < 26; x401++) {
int32_t x402 = x397;
double x403 = x23[x402];
double x404 = x349[x401];
double x405 = x403 * x404;
x400 += x405;
x397 += 1;

}
double x410 = x400;
x398[x399] = x410;

}
double* x414 = (double*)myMalloc(50 * sizeof(double));
for(int x415=0; x415 < 50; x415++) {
x414[x415] = 0.0;

}
double* x419 = (double*)myMalloc(50 * sizeof(double));
for(int x420=0; x420 < 50; x420++) {
double x421 = x375[x420];
double x422 = x398[x420];
double x423 = x421 + x422;
x419[x420] = x423;

}
double* x427 = (double*)myMalloc(50 * sizeof(double));
for(int x428=0; x428 < 50; x428++) {
x427[x428] = 0.0;

}
double* x432 = (double*)myMalloc(50 * sizeof(double));
for(int x433=0; x433 < 50; x433++) {
double x434 = x419[x433];
double x435 = x31[x433];
double x436 = x434 + x435;
x432[x433] = x436;

}
double* x440 = (double*)myMalloc(50 * sizeof(double));
for(int x441=0; x441 < 50; x441++) {
x440[x441] = 0.0;

}
double* x445 = (double*)myMalloc(50 * sizeof(double));
for(int x446=0; x446 < 50; x446++) {
double x447 = x432[x446];
double x448 = -1.0 * x447;
double x449 = exp(x448);
double x450 = x449 + 1.0;
double x451 = 1.0 / x450;
x445[x446] = x451;

}
double* x455 = (double*)myMalloc(50 * sizeof(double));
for(int x456=0; x456 < 50; x456++) {
x455[x456] = 0.0;

}
// dot WrappedArray(50, 50) - WrappedArray(50)
int32_t x461 = 0;
double* x462 = (double*)myMalloc(50 * sizeof(double));
for(int x463=0; x463 < 50; x463++) {
double x464 = 0.0;
for(int x465=0; x465 < 50; x465++) {
int32_t x466 = x461;
double x467 = x37[x466];
double x468 = x344[x465];
double x469 = x467 * x468;
x464 += x469;
x461 += 1;

}
double x474 = x464;
x462[x463] = x474;

}
double* x478 = (double*)myMalloc(50 * sizeof(double));
for(int x479=0; x479 < 50; x479++) {
x478[x479] = 0.0;

}
// dot WrappedArray(50, 26) - WrappedArray(26)
int32_t x484 = 0;
double* x485 = (double*)myMalloc(50 * sizeof(double));
for(int x486=0; x486 < 50; x486++) {
double x487 = 0.0;
for(int x488=0; x488 < 26; x488++) {
int32_t x489 = x484;
double x490 = x44[x489];
double x491 = x349[x488];
double x492 = x490 * x491;
x487 += x492;
x484 += 1;

}
double x497 = x487;
x485[x486] = x497;

}
double* x501 = (double*)myMalloc(50 * sizeof(double));
for(int x502=0; x502 < 50; x502++) {
x501[x502] = 0.0;

}
double* x506 = (double*)myMalloc(50 * sizeof(double));
for(int x507=0; x507 < 50; x507++) {
double x508 = x462[x507];
double x509 = x485[x507];
double x510 = x508 + x509;
x506[x507] = x510;

}
double* x514 = (double*)myMalloc(50 * sizeof(double));
for(int x515=0; x515 < 50; x515++) {
x514[x515] = 0.0;

}
double* x519 = (double*)myMalloc(50 * sizeof(double));
for(int x520=0; x520 < 50; x520++) {
double x521 = x506[x520];
double x522 = x51[x520];
double x523 = x521 + x522;
x519[x520] = x523;

}
double* x527 = (double*)myMalloc(50 * sizeof(double));
for(int x528=0; x528 < 50; x528++) {
x527[x528] = 0.0;

}
double* x532 = (double*)myMalloc(50 * sizeof(double));
for(int x533=0; x533 < 50; x533++) {
double x534 = x519[x533];
double x535 = -1.0 * x534;
double x536 = exp(x535);
double x537 = x536 + 1.0;
double x538 = 1.0 / x537;
x532[x533] = x538;

}
double* x542 = (double*)myMalloc(50 * sizeof(double));
for(int x543=0; x543 < 50; x543++) {
x542[x543] = 0.0;

}
// dot WrappedArray(50, 50) - WrappedArray(50)
int32_t x548 = 0;
double* x549 = (double*)myMalloc(50 * sizeof(double));
for(int x550=0; x550 < 50; x550++) {
double x551 = 0.0;
for(int x552=0; x552 < 50; x552++) {
int32_t x553 = x548;
double x554 = x75[x553];
double x555 = x344[x552];
double x556 = x554 * x555;
x551 += x556;
x548 += 1;

}
double x561 = x551;
x549[x550] = x561;

}
double* x565 = (double*)myMalloc(50 * sizeof(double));
for(int x566=0; x566 < 50; x566++) {
x565[x566] = 0.0;

}
// dot WrappedArray(50, 26) - WrappedArray(26)
int32_t x571 = 0;
double* x572 = (double*)myMalloc(50 * sizeof(double));
for(int x573=0; x573 < 50; x573++) {
double x574 = 0.0;
for(int x575=0; x575 < 26; x575++) {
int32_t x576 = x571;
double x577 = x82[x576];
double x578 = x349[x575];
double x579 = x577 * x578;
x574 += x579;
x571 += 1;

}
double x584 = x574;
x572[x573] = x584;

}
double* x588 = (double*)myMalloc(50 * sizeof(double));
for(int x589=0; x589 < 50; x589++) {
x588[x589] = 0.0;

}
double* x593 = (double*)myMalloc(50 * sizeof(double));
for(int x594=0; x594 < 50; x594++) {
double x595 = x549[x594];
double x596 = x572[x594];
double x597 = x595 + x596;
x593[x594] = x597;

}
double* x601 = (double*)myMalloc(50 * sizeof(double));
for(int x602=0; x602 < 50; x602++) {
x601[x602] = 0.0;

}
double* x606 = (double*)myMalloc(50 * sizeof(double));
for(int x607=0; x607 < 50; x607++) {
double x608 = x593[x607];
double x609 = x89[x607];
double x610 = x608 + x609;
x606[x607] = x610;

}
double* x614 = (double*)myMalloc(50 * sizeof(double));
for(int x615=0; x615 < 50; x615++) {
x614[x615] = 0.0;

}
double* x619 = (double*)myMalloc(50 * sizeof(double));
for(int x620=0; x620 < 50; x620++) {
double x621 = x606[x620];
double x622 = -1.0 * x621;
double x623 = exp(x622);
double x624 = x623 + 1.0;
double x625 = 1.0 / x624;
x619[x620] = x625;

}
double* x629 = (double*)myMalloc(50 * sizeof(double));
for(int x630=0; x630 < 50; x630++) {
x629[x630] = 0.0;

}
// dot WrappedArray(50, 50) - WrappedArray(50)
int32_t x635 = 0;
double* x636 = (double*)myMalloc(50 * sizeof(double));
for(int x637=0; x637 < 50; x637++) {
double x638 = 0.0;
for(int x639=0; x639 < 50; x639++) {
int32_t x640 = x635;
double x641 = x56[x640];
double x642 = x344[x639];
double x643 = x641 * x642;
x638 += x643;
x635 += 1;

}
double x648 = x638;
x636[x637] = x648;

}
double* x652 = (double*)myMalloc(50 * sizeof(double));
for(int x653=0; x653 < 50; x653++) {
x652[x653] = 0.0;

}
// dot WrappedArray(50, 26) - WrappedArray(26)
int32_t x658 = 0;
double* x659 = (double*)myMalloc(50 * sizeof(double));
for(int x660=0; x660 < 50; x660++) {
double x661 = 0.0;
for(int x662=0; x662 < 26; x662++) {
int32_t x663 = x658;
double x664 = x63[x663];
double x665 = x349[x662];
double x666 = x664 * x665;
x661 += x666;
x658 += 1;

}
double x671 = x661;
x659[x660] = x671;

}
double* x675 = (double*)myMalloc(50 * sizeof(double));
for(int x676=0; x676 < 50; x676++) {
x675[x676] = 0.0;

}
double* x680 = (double*)myMalloc(50 * sizeof(double));
for(int x681=0; x681 < 50; x681++) {
double x682 = x636[x681];
double x683 = x659[x681];
double x684 = x682 + x683;
x680[x681] = x684;

}
double* x688 = (double*)myMalloc(50 * sizeof(double));
for(int x689=0; x689 < 50; x689++) {
x688[x689] = 0.0;

}
double* x693 = (double*)myMalloc(50 * sizeof(double));
for(int x694=0; x694 < 50; x694++) {
double x695 = x680[x694];
double x696 = x70[x694];
double x697 = x695 + x696;
x693[x694] = x697;

}
double* x701 = (double*)myMalloc(50 * sizeof(double));
for(int x702=0; x702 < 50; x702++) {
x701[x702] = 0.0;

}
double* x706 = (double*)myMalloc(50 * sizeof(double));
for(int x707=0; x707 < 50; x707++) {
double x708 = x693[x707];
double x709 = tanh(x708);
x706[x707] = x709;

}
double* x713 = (double*)myMalloc(50 * sizeof(double));
for(int x714=0; x714 < 50; x714++) {
x713[x714] = 0.0;

}
double* x718 = (double*)myMalloc(50 * sizeof(double));
for(int x719=0; x719 < 50; x719++) {
double x720 = x445[x719];
double x721 = x346[x719];
double x722 = x720 * x721;
x718[x719] = x722;

}
double* x726 = (double*)myMalloc(50 * sizeof(double));
for(int x727=0; x727 < 50; x727++) {
x726[x727] = 0.0;

}
double* x731 = (double*)myMalloc(50 * sizeof(double));
for(int x732=0; x732 < 50; x732++) {
double x733 = x532[x732];
double x734 = x706[x732];
double x735 = x733 * x734;
x731[x732] = x735;

}
double* x739 = (double*)myMalloc(50 * sizeof(double));
for(int x740=0; x740 < 50; x740++) {
x739[x740] = 0.0;

}
double* x744 = (double*)myMalloc(50 * sizeof(double));
for(int x745=0; x745 < 50; x745++) {
double x746 = x718[x745];
double x747 = x731[x745];
double x748 = x746 + x747;
x744[x745] = x748;

}
double* x752 = (double*)myMalloc(50 * sizeof(double));
for(int x753=0; x753 < 50; x753++) {
x752[x753] = 0.0;

}
double* x757 = (double*)myMalloc(50 * sizeof(double));
for(int x758=0; x758 < 50; x758++) {
double x759 = x744[x758];
double x760 = tanh(x759);
x757[x758] = x760;

}
double* x764 = (double*)myMalloc(50 * sizeof(double));
for(int x765=0; x765 < 50; x765++) {
x764[x765] = 0.0;

}
double* x769 = (double*)myMalloc(50 * sizeof(double));
for(int x770=0; x770 < 50; x770++) {
double x771 = x619[x770];
double x772 = x757[x770];
double x773 = x771 * x772;
x769[x770] = x773;

}
double* x777 = (double*)myMalloc(50 * sizeof(double));
for(int x778=0; x778 < 50; x778++) {
x777[x778] = 0.0;

}
// dot WrappedArray(26, 50) - WrappedArray(50)
int32_t x783 = 0;
double* x784 = (double*)myMalloc(26 * sizeof(double));
for(int x785=0; x785 < 26; x785++) {
double x786 = 0.0;
for(int x787=0; x787 < 50; x787++) {
int32_t x788 = x783;
double x789 = x94[x788];
double x790 = x769[x787];
double x791 = x789 * x790;
x786 += x791;
x783 += 1;

}
double x796 = x786;
x784[x785] = x796;

}
double* x800 = (double*)myMalloc(26 * sizeof(double));
for(int x801=0; x801 < 26; x801++) {
x800[x801] = 0.0;

}
double* x805 = (double*)myMalloc(26 * sizeof(double));
for(int x806=0; x806 < 26; x806++) {
double x807 = x784[x806];
double x808 = x101[x806];
double x809 = x807 + x808;
x805[x806] = x809;

}
double* x813 = (double*)myMalloc(26 * sizeof(double));
for(int x814=0; x814 < 26; x814++) {
x813[x814] = 0.0;

}
double* x818 = (double*)myMalloc(26 * sizeof(double));
for(int x819=0; x819 < 26; x819++) {
double x820 = x805[x819];
double x821 = exp(x820);
x818[x819] = x821;

}
double* x825 = (double*)myMalloc(26 * sizeof(double));
for(int x826=0; x826 < 26; x826++) {
x825[x826] = 0.0;

}
// Here
double x831 = 0.0;
for(int x832=0; x832 < 26; x832++) {
double x833 = x831;
double x834 = x818[x832];
double x835 = x833 + x834;
x831 = x835;

}
double x839 = x831;
double* x840 = (double*)myMalloc(1 * sizeof(double));
x840[0] = x839;
double* x842 = (double*)myMalloc(1 * sizeof(double));
for(int x843=0; x843 < 1; x843++) {
x842[x843] = 0.0;

}
double x847 = x840[0];
double* x848 = (double*)myMalloc(26 * sizeof(double));
for(int x849=0; x849 < 26; x849++) {
double x850 = x818[x849];
double x851 = x850 / x847;
x848[x849] = x851;

}
double* x855 = (double*)myMalloc(26 * sizeof(double));
for(int x856=0; x856 < 26; x856++) {
x855[x856] = 0.0;

}
// dot WrappedArray(26) - WrappedArray(26)
int32_t x861 = 0;
double* x862 = (double*)myMalloc(1 * sizeof(double));
for(int x863=0; x863 < 1; x863++) {
double x864 = 0.0;
for(int x865=0; x865 < 26; x865++) {
int32_t x866 = x861;
double x867 = x848[x866];
double x868 = x361[x865];
double x869 = x867 * x868;
x864 += x869;
x861 += 1;

}
double x874 = x864;
x862[x863] = x874;

}
double* x878 = (double*)myMalloc(1 * sizeof(double));
for(int x879=0; x879 < 1; x879++) {
x878[x879] = 0.0;

}
double* x883 = (double*)myMalloc(1 * sizeof(double));
for(int x884=0; x884 < 1; x884++) {
double x885 = x862[x884];
double x886 = log(x885);
x883[x884] = x886;

}
double* x890 = (double*)myMalloc(1 * sizeof(double));
for(int x891=0; x891 < 1; x891++) {
x890[x891] = 0.0;

}
double* x895 = (double*)myMalloc(1 * sizeof(double));
for(int x896=0; x896 < 1; x896++) {
double x897 = x883[x896];
double x898 = x342[0];
double x899 = x898 - x897;
x895[x896] = x899;

}
double* x903 = (double*)myMalloc(1 * sizeof(double));
for(int x904=0; x904 < 1; x904++) {
x903[x904] = 0.0;

}
double** x909 = (double**)myMalloc(6 * sizeof(double*));
x909[0] = x895;
x909[1] = x903;
x909[2] = x769;
x909[3] = x777;
x909[4] = x744;
x909[5] = x752;
int32_t x1016 = 0;
int32_t x1017 = x1016;
int32_t x1018 = x1017;
int32_t x1019 = 0;
int32_t x1020 = x1019;
int32_t x1021 = x1020;
int32_t x1022 = 0;
int32_t x1023 = x1022;
int32_t x1024 = x1023;
int32_t x1055 = 0;
int32_t x1056 = x1055;
int32_t x1057 = x1056;
int32_t x1058 = 0;
int32_t x1059 = x1058;
int32_t x1060 = x1059;
int32_t x1061 = 0;
int32_t x1062 = x1061;
int32_t x1063 = x1062;
int32_t x1217 = 0;
int32_t x1218 = x1217;
int32_t x1219 = x1218;
int32_t x1220 = 0;
int32_t x1221 = x1220;
int32_t x1222 = x1221;
int32_t x1223 = 0;
int32_t x1224 = x1223;
int32_t x1225 = x1224;
int32_t x1256 = 0;
int32_t x1257 = x1256;
int32_t x1258 = x1257;
int32_t x1259 = 0;
int32_t x1260 = x1259;
int32_t x1261 = x1260;
int32_t x1262 = 0;
int32_t x1263 = x1262;
int32_t x1264 = x1263;
int32_t x1295 = 0;
int32_t x1296 = x1295;
int32_t x1297 = x1296;
int32_t x1298 = 0;
int32_t x1299 = x1298;
int32_t x1300 = x1299;
int32_t x1301 = 0;
int32_t x1302 = x1301;
int32_t x1303 = x1302;
int32_t x1334 = 0;
int32_t x1335 = x1334;
int32_t x1336 = x1335;
int32_t x1337 = 0;
int32_t x1338 = x1337;
int32_t x1339 = x1338;
int32_t x1340 = 0;
int32_t x1341 = x1340;
int32_t x1342 = x1341;
int32_t x1414 = 0;
int32_t x1415 = x1414;
int32_t x1416 = x1415;
int32_t x1417 = 0;
int32_t x1418 = x1417;
int32_t x1419 = x1418;
int32_t x1420 = 0;
int32_t x1421 = x1420;
int32_t x1422 = x1421;
int32_t x1453 = 0;
int32_t x1454 = x1453;
int32_t x1455 = x1454;
int32_t x1456 = 0;
int32_t x1457 = x1456;
int32_t x1458 = x1457;
int32_t x1459 = 0;
int32_t x1460 = x1459;
int32_t x1461 = x1460;
int32_t x1492 = 0;
int32_t x1493 = x1492;
int32_t x1494 = x1493;
int32_t x1495 = 0;
int32_t x1496 = x1495;
int32_t x1497 = x1496;
int32_t x1498 = 0;
int32_t x1499 = x1498;
int32_t x1500 = x1499;
int32_t x1531 = 0;
int32_t x1532 = x1531;
int32_t x1533 = x1532;
int32_t x1534 = 0;
int32_t x1535 = x1534;
int32_t x1536 = x1535;
int32_t x1537 = 0;
int32_t x1538 = x1537;
int32_t x1539 = x1538;
int32_t x1611 = 0;
int32_t x1612 = x1611;
int32_t x1613 = x1612;
int32_t x1614 = 0;
int32_t x1615 = x1614;
int32_t x1616 = x1615;
int32_t x1617 = 0;
int32_t x1618 = x1617;
int32_t x1619 = x1618;
int32_t x1650 = 0;
int32_t x1651 = x1650;
int32_t x1652 = x1651;
int32_t x1653 = 0;
int32_t x1654 = x1653;
int32_t x1655 = x1654;
int32_t x1656 = 0;
int32_t x1657 = x1656;
int32_t x1658 = x1657;
int32_t x1689 = 0;
int32_t x1690 = x1689;
int32_t x1691 = x1690;
int32_t x1692 = 0;
int32_t x1693 = x1692;
int32_t x1694 = x1693;
int32_t x1695 = 0;
int32_t x1696 = x1695;
int32_t x1697 = x1696;
int32_t x1728 = 0;
int32_t x1729 = x1728;
int32_t x1730 = x1729;
int32_t x1731 = 0;
int32_t x1732 = x1731;
int32_t x1733 = x1732;
int32_t x1734 = 0;
int32_t x1735 = x1734;
int32_t x1736 = x1735;
int32_t x1808 = 0;
int32_t x1809 = x1808;
int32_t x1810 = x1809;
int32_t x1811 = 0;
int32_t x1812 = x1811;
int32_t x1813 = x1812;
int32_t x1814 = 0;
int32_t x1815 = x1814;
int32_t x1816 = x1815;
int32_t x1847 = 0;
int32_t x1848 = x1847;
int32_t x1849 = x1848;
int32_t x1850 = 0;
int32_t x1851 = x1850;
int32_t x1852 = x1851;
int32_t x1853 = 0;
int32_t x1854 = x1853;
int32_t x1855 = x1854;
int32_t x1886 = 0;
int32_t x1887 = x1886;
int32_t x1888 = x1887;
int32_t x1889 = 0;
int32_t x1890 = x1889;
int32_t x1891 = x1890;
int32_t x1892 = 0;
int32_t x1893 = x1892;
int32_t x1894 = x1893;
int32_t x1925 = 0;
int32_t x1926 = x1925;
int32_t x1927 = x1926;
int32_t x1928 = 0;
int32_t x1929 = x1928;
int32_t x1930 = x1929;
int32_t x1931 = 0;
int32_t x1932 = x1931;
int32_t x1933 = x1932;
int32_t x908 = x340 + 1;
x337(x908,x909);
// += tensor of dim 0
double x919 = x903[0];
for(int x920=0; x920 < 1; x920++) {
double x921 = x343[x920];
double x922 = x921 + x919;
x343[x920] = x922;

}
double x926 = x903[0];
for(int x927=0; x927 < 1; x927++) {
double x928 = x890[x927];
double x929 = x928 - x926;
x890[x927] = x929;

}
for(int x933=0; x933 < 1; x933++) {
double x934 = x878[0];
double x935 = x890[0];
double x936 = x862[0];
double x937 = x935 / x936;
double x938 = x934 + x937;
x878[0] = x938;

}
double x942 = x878[0];
// Generate code for addMul
for(int x944=0; x944 < 26; x944++) {
double x945 = x855[x944];
double x946 = x361[x944];
double x947 = x942 * x946;
double x948 = x945 + x947;
x855[x944] = x948;

}
double x952 = x878[0];
// Generate code for addMul
for(int x954=0; x954 < 26; x954++) {
double x955 = x368[x954];
double x956 = x848[x954];
double x957 = x952 * x956;
double x958 = x955 + x957;
x368[x954] = x958;

}
for(int x962=0; x962 < 26; x962++) {
double x963 = x825[x962];
double x964 = x855[x962];
double x965 = x840[0];
double x966 = x964 / x965;
double x967 = x963 + x966;
x825[x962] = x967;

}
for(int x971=0; x971 < 26; x971++) {
double x972 = x842[0];
double x973 = x818[x971];
double x974 = x855[x971];
double x976 = x840[0];
double x975 = x973 * x974;
double x977 = x976 * x976;
double x978 = x975 / x977;
double x979 = x972 - x978;
x842[0] = x979;

}
// += tensor of dim 0
double x984 = x842[0];
for(int x985=0; x985 < 26; x985++) {
double x986 = x825[x985];
double x987 = x986 + x984;
x825[x985] = x987;

}
// backpropage exp
for(int x992=0; x992 < 26; x992++) {
double x993 = x813[x992];
double x994 = x818[x992];
double x995 = x825[x992];
double x996 = x994 * x995;
double x997 = x993 + x996;
x813[x992] = x997;

}
// backpropagate +
for(int x1002=0; x1002 < 26; x1002++) {
double x1003 = x800[x1002];
double x1004 = x813[x1002];
double x1005 = x1003 + x1004;
x800[x1002] = x1005;

}
for(int x1009=0; x1009 < 26; x1009++) {
double x1010 = x192[x1009];
double x1011 = x813[x1009];
double x1012 = x1010 + x1011;
x192[x1009] = x1012;

}
for(int x1025=0; x1025 < 26; x1025++) {
int32_t x1026 = x1024;
int32_t x1027 = x1026;
for(int x1028=0; x1028 < 50; x1028++) {
int32_t x1029 = x1027;
int32_t x1030 = x1029;
int32_t x1031 = x1021;
int32_t x1032 = x1031;
for(int x1033=0; x1033 < 1; x1033++) {
int32_t x1034 = x1018;
double x1035 = x187[x1034];
int32_t x1036 = x1032;
double x1037 = x800[x1036];
int32_t x1038 = x1030;
double x1039 = x769[x1038];
double x1040 = x1037 * x1039;
double x1041 = x1035 + x1040;
x187[x1034] = x1041;
x1032 += 1;
x1030 += 50;

}
x1018 += 1;
x1027 += 1;

}
x1021 += 1;
x1024 *= 0;

}
for(int x1064=0; x1064 < 1; x1064++) {
int32_t x1065 = x1063;
int32_t x1066 = x1065;
for(int x1067=0; x1067 < 50; x1067++) {
int32_t x1068 = x1066;
int32_t x1069 = x1068;
int32_t x1070 = x1060;
int32_t x1071 = x1070;
for(int x1072=0; x1072 < 26; x1072++) {
int32_t x1073 = x1057;
double x1074 = x777[x1073];
int32_t x1075 = x1071;
double x1076 = x800[x1075];
int32_t x1077 = x1069;
double x1078 = x94[x1077];
double x1079 = x1076 * x1078;
double x1080 = x1074 + x1079;
x777[x1073] = x1080;
x1071 += 1;
x1069 += 50;

}
x1057 += 1;
x1066 += 1;

}
x1060 += 26;
x1063 *= 0;

}
for(int x1094=0; x1094 < 50; x1094++) {
double x1095 = x629[x1094];
double x1096 = x757[x1094];
double x1097 = x777[x1094];
double x1098 = x1096 * x1097;
double x1099 = x1095 + x1098;
x629[x1094] = x1099;

}
for(int x1103=0; x1103 < 50; x1103++) {
double x1104 = x764[x1103];
double x1105 = x619[x1103];
double x1106 = x777[x1103];
double x1107 = x1105 * x1106;
double x1108 = x1104 + x1107;
x764[x1103] = x1108;

}
// backpropagate tanh
for(int x1113=0; x1113 < 50; x1113++) {
double x1114 = x752[x1113];
double x1115 = x757[x1113];
double x1118 = x764[x1113];
double x1116 = x1115 * x1115;
double x1117 = 1.0 - x1116;
double x1119 = x1117 * x1118;
double x1120 = x1114 + x1119;
x752[x1113] = x1120;

}
// backpropagate +
for(int x1125=0; x1125 < 50; x1125++) {
double x1126 = x726[x1125];
double x1127 = x752[x1125];
double x1128 = x1126 + x1127;
x726[x1125] = x1128;

}
for(int x1132=0; x1132 < 50; x1132++) {
double x1133 = x739[x1132];
double x1134 = x752[x1132];
double x1135 = x1133 + x1134;
x739[x1132] = x1135;

}
for(int x1139=0; x1139 < 50; x1139++) {
double x1140 = x542[x1139];
double x1141 = x706[x1139];
double x1142 = x739[x1139];
double x1143 = x1141 * x1142;
double x1144 = x1140 + x1143;
x542[x1139] = x1144;

}
for(int x1148=0; x1148 < 50; x1148++) {
double x1149 = x713[x1148];
double x1150 = x532[x1148];
double x1151 = x739[x1148];
double x1152 = x1150 * x1151;
double x1153 = x1149 + x1152;
x713[x1148] = x1153;

}
for(int x1157=0; x1157 < 50; x1157++) {
double x1158 = x455[x1157];
double x1159 = x346[x1157];
double x1160 = x726[x1157];
double x1161 = x1159 * x1160;
double x1162 = x1158 + x1161;
x455[x1157] = x1162;

}
for(int x1166=0; x1166 < 50; x1166++) {
double x1167 = x347[x1166];
double x1168 = x445[x1166];
double x1169 = x726[x1166];
double x1170 = x1168 * x1169;
double x1171 = x1167 + x1170;
x347[x1166] = x1171;

}
// backpropagate tanh
for(int x1176=0; x1176 < 50; x1176++) {
double x1177 = x701[x1176];
double x1178 = x706[x1176];
double x1181 = x713[x1176];
double x1179 = x1178 * x1178;
double x1180 = 1.0 - x1179;
double x1182 = x1180 * x1181;
double x1183 = x1177 + x1182;
x701[x1176] = x1183;

}
// backpropagate +
for(int x1188=0; x1188 < 50; x1188++) {
double x1189 = x688[x1188];
double x1190 = x701[x1188];
double x1191 = x1189 + x1190;
x688[x1188] = x1191;

}
for(int x1195=0; x1195 < 50; x1195++) {
double x1196 = x167[x1195];
double x1197 = x701[x1195];
double x1198 = x1196 + x1197;
x167[x1195] = x1198;

}
// backpropagate +
for(int x1203=0; x1203 < 50; x1203++) {
double x1204 = x652[x1203];
double x1205 = x688[x1203];
double x1206 = x1204 + x1205;
x652[x1203] = x1206;

}
for(int x1210=0; x1210 < 50; x1210++) {
double x1211 = x675[x1210];
double x1212 = x688[x1210];
double x1213 = x1211 + x1212;
x675[x1210] = x1213;

}
for(int x1226=0; x1226 < 50; x1226++) {
int32_t x1227 = x1225;
int32_t x1228 = x1227;
for(int x1229=0; x1229 < 26; x1229++) {
int32_t x1230 = x1228;
int32_t x1231 = x1230;
int32_t x1232 = x1222;
int32_t x1233 = x1232;
for(int x1234=0; x1234 < 1; x1234++) {
int32_t x1235 = x1219;
double x1236 = x162[x1235];
int32_t x1237 = x1233;
double x1238 = x675[x1237];
int32_t x1239 = x1231;
double x1240 = x349[x1239];
double x1241 = x1238 * x1240;
double x1242 = x1236 + x1241;
x162[x1235] = x1242;
x1233 += 1;
x1231 += 26;

}
x1219 += 1;
x1228 += 1;

}
x1222 += 1;
x1225 *= 0;

}
for(int x1265=0; x1265 < 1; x1265++) {
int32_t x1266 = x1264;
int32_t x1267 = x1266;
for(int x1268=0; x1268 < 26; x1268++) {
int32_t x1269 = x1267;
int32_t x1270 = x1269;
int32_t x1271 = x1261;
int32_t x1272 = x1271;
for(int x1273=0; x1273 < 50; x1273++) {
int32_t x1274 = x1258;
double x1275 = x356[x1274];
int32_t x1276 = x1272;
double x1277 = x675[x1276];
int32_t x1278 = x1270;
double x1279 = x63[x1278];
double x1280 = x1277 * x1279;
double x1281 = x1275 + x1280;
x356[x1274] = x1281;
x1272 += 1;
x1270 += 26;

}
x1258 += 1;
x1267 += 1;

}
x1261 += 50;
x1264 *= 0;

}
for(int x1304=0; x1304 < 50; x1304++) {
int32_t x1305 = x1303;
int32_t x1306 = x1305;
for(int x1307=0; x1307 < 50; x1307++) {
int32_t x1308 = x1306;
int32_t x1309 = x1308;
int32_t x1310 = x1300;
int32_t x1311 = x1310;
for(int x1312=0; x1312 < 1; x1312++) {
int32_t x1313 = x1297;
double x1314 = x157[x1313];
int32_t x1315 = x1311;
double x1316 = x652[x1315];
int32_t x1317 = x1309;
double x1318 = x344[x1317];
double x1319 = x1316 * x1318;
double x1320 = x1314 + x1319;
x157[x1313] = x1320;
x1311 += 1;
x1309 += 50;

}
x1297 += 1;
x1306 += 1;

}
x1300 += 1;
x1303 *= 0;

}
for(int x1343=0; x1343 < 1; x1343++) {
int32_t x1344 = x1342;
int32_t x1345 = x1344;
for(int x1346=0; x1346 < 50; x1346++) {
int32_t x1347 = x1345;
int32_t x1348 = x1347;
int32_t x1349 = x1339;
int32_t x1350 = x1349;
for(int x1351=0; x1351 < 50; x1351++) {
int32_t x1352 = x1336;
double x1353 = x345[x1352];
int32_t x1354 = x1350;
double x1355 = x652[x1354];
int32_t x1356 = x1348;
double x1357 = x56[x1356];
double x1358 = x1355 * x1357;
double x1359 = x1353 + x1358;
x345[x1352] = x1359;
x1350 += 1;
x1348 += 50;

}
x1336 += 1;
x1345 += 1;

}
x1339 += 50;
x1342 *= 0;

}
for(int x1373=0; x1373 < 50; x1373++) {
double x1374 = x614[x1373];
double x1375 = x619[x1373];
double x1378 = x629[x1373];
double x1376 = 1.0 - x1375;
double x1377 = x1376 * x1375;
double x1379 = x1377 * x1378;
double x1380 = x1374 + x1379;
x614[x1373] = x1380;

}
// backpropagate +
for(int x1385=0; x1385 < 50; x1385++) {
double x1386 = x601[x1385];
double x1387 = x614[x1385];
double x1388 = x1386 + x1387;
x601[x1385] = x1388;

}
for(int x1392=0; x1392 < 50; x1392++) {
double x1393 = x182[x1392];
double x1394 = x614[x1392];
double x1395 = x1393 + x1394;
x182[x1392] = x1395;

}
// backpropagate +
for(int x1400=0; x1400 < 50; x1400++) {
double x1401 = x565[x1400];
double x1402 = x601[x1400];
double x1403 = x1401 + x1402;
x565[x1400] = x1403;

}
for(int x1407=0; x1407 < 50; x1407++) {
double x1408 = x588[x1407];
double x1409 = x601[x1407];
double x1410 = x1408 + x1409;
x588[x1407] = x1410;

}
for(int x1423=0; x1423 < 50; x1423++) {
int32_t x1424 = x1422;
int32_t x1425 = x1424;
for(int x1426=0; x1426 < 26; x1426++) {
int32_t x1427 = x1425;
int32_t x1428 = x1427;
int32_t x1429 = x1419;
int32_t x1430 = x1429;
for(int x1431=0; x1431 < 1; x1431++) {
int32_t x1432 = x1416;
double x1433 = x177[x1432];
int32_t x1434 = x1430;
double x1435 = x588[x1434];
int32_t x1436 = x1428;
double x1437 = x349[x1436];
double x1438 = x1435 * x1437;
double x1439 = x1433 + x1438;
x177[x1432] = x1439;
x1430 += 1;
x1428 += 26;

}
x1416 += 1;
x1425 += 1;

}
x1419 += 1;
x1422 *= 0;

}
for(int x1462=0; x1462 < 1; x1462++) {
int32_t x1463 = x1461;
int32_t x1464 = x1463;
for(int x1465=0; x1465 < 26; x1465++) {
int32_t x1466 = x1464;
int32_t x1467 = x1466;
int32_t x1468 = x1458;
int32_t x1469 = x1468;
for(int x1470=0; x1470 < 50; x1470++) {
int32_t x1471 = x1455;
double x1472 = x356[x1471];
int32_t x1473 = x1469;
double x1474 = x588[x1473];
int32_t x1475 = x1467;
double x1476 = x82[x1475];
double x1477 = x1474 * x1476;
double x1478 = x1472 + x1477;
x356[x1471] = x1478;
x1469 += 1;
x1467 += 26;

}
x1455 += 1;
x1464 += 1;

}
x1458 += 50;
x1461 *= 0;

}
for(int x1501=0; x1501 < 50; x1501++) {
int32_t x1502 = x1500;
int32_t x1503 = x1502;
for(int x1504=0; x1504 < 50; x1504++) {
int32_t x1505 = x1503;
int32_t x1506 = x1505;
int32_t x1507 = x1497;
int32_t x1508 = x1507;
for(int x1509=0; x1509 < 1; x1509++) {
int32_t x1510 = x1494;
double x1511 = x172[x1510];
int32_t x1512 = x1508;
double x1513 = x565[x1512];
int32_t x1514 = x1506;
double x1515 = x344[x1514];
double x1516 = x1513 * x1515;
double x1517 = x1511 + x1516;
x172[x1510] = x1517;
x1508 += 1;
x1506 += 50;

}
x1494 += 1;
x1503 += 1;

}
x1497 += 1;
x1500 *= 0;

}
for(int x1540=0; x1540 < 1; x1540++) {
int32_t x1541 = x1539;
int32_t x1542 = x1541;
for(int x1543=0; x1543 < 50; x1543++) {
int32_t x1544 = x1542;
int32_t x1545 = x1544;
int32_t x1546 = x1536;
int32_t x1547 = x1546;
for(int x1548=0; x1548 < 50; x1548++) {
int32_t x1549 = x1533;
double x1550 = x345[x1549];
int32_t x1551 = x1547;
double x1552 = x565[x1551];
int32_t x1553 = x1545;
double x1554 = x75[x1553];
double x1555 = x1552 * x1554;
double x1556 = x1550 + x1555;
x345[x1549] = x1556;
x1547 += 1;
x1545 += 50;

}
x1533 += 1;
x1542 += 1;

}
x1536 += 50;
x1539 *= 0;

}
for(int x1570=0; x1570 < 50; x1570++) {
double x1571 = x527[x1570];
double x1572 = x532[x1570];
double x1575 = x542[x1570];
double x1573 = 1.0 - x1572;
double x1574 = x1573 * x1572;
double x1576 = x1574 * x1575;
double x1577 = x1571 + x1576;
x527[x1570] = x1577;

}
// backpropagate +
for(int x1582=0; x1582 < 50; x1582++) {
double x1583 = x514[x1582];
double x1584 = x527[x1582];
double x1585 = x1583 + x1584;
x514[x1582] = x1585;

}
for(int x1589=0; x1589 < 50; x1589++) {
double x1590 = x152[x1589];
double x1591 = x527[x1589];
double x1592 = x1590 + x1591;
x152[x1589] = x1592;

}
// backpropagate +
for(int x1597=0; x1597 < 50; x1597++) {
double x1598 = x478[x1597];
double x1599 = x514[x1597];
double x1600 = x1598 + x1599;
x478[x1597] = x1600;

}
for(int x1604=0; x1604 < 50; x1604++) {
double x1605 = x501[x1604];
double x1606 = x514[x1604];
double x1607 = x1605 + x1606;
x501[x1604] = x1607;

}
for(int x1620=0; x1620 < 50; x1620++) {
int32_t x1621 = x1619;
int32_t x1622 = x1621;
for(int x1623=0; x1623 < 26; x1623++) {
int32_t x1624 = x1622;
int32_t x1625 = x1624;
int32_t x1626 = x1616;
int32_t x1627 = x1626;
for(int x1628=0; x1628 < 1; x1628++) {
int32_t x1629 = x1613;
double x1630 = x147[x1629];
int32_t x1631 = x1627;
double x1632 = x501[x1631];
int32_t x1633 = x1625;
double x1634 = x349[x1633];
double x1635 = x1632 * x1634;
double x1636 = x1630 + x1635;
x147[x1629] = x1636;
x1627 += 1;
x1625 += 26;

}
x1613 += 1;
x1622 += 1;

}
x1616 += 1;
x1619 *= 0;

}
for(int x1659=0; x1659 < 1; x1659++) {
int32_t x1660 = x1658;
int32_t x1661 = x1660;
for(int x1662=0; x1662 < 26; x1662++) {
int32_t x1663 = x1661;
int32_t x1664 = x1663;
int32_t x1665 = x1655;
int32_t x1666 = x1665;
for(int x1667=0; x1667 < 50; x1667++) {
int32_t x1668 = x1652;
double x1669 = x356[x1668];
int32_t x1670 = x1666;
double x1671 = x501[x1670];
int32_t x1672 = x1664;
double x1673 = x44[x1672];
double x1674 = x1671 * x1673;
double x1675 = x1669 + x1674;
x356[x1668] = x1675;
x1666 += 1;
x1664 += 26;

}
x1652 += 1;
x1661 += 1;

}
x1655 += 50;
x1658 *= 0;

}
for(int x1698=0; x1698 < 50; x1698++) {
int32_t x1699 = x1697;
int32_t x1700 = x1699;
for(int x1701=0; x1701 < 50; x1701++) {
int32_t x1702 = x1700;
int32_t x1703 = x1702;
int32_t x1704 = x1694;
int32_t x1705 = x1704;
for(int x1706=0; x1706 < 1; x1706++) {
int32_t x1707 = x1691;
double x1708 = x142[x1707];
int32_t x1709 = x1705;
double x1710 = x478[x1709];
int32_t x1711 = x1703;
double x1712 = x344[x1711];
double x1713 = x1710 * x1712;
double x1714 = x1708 + x1713;
x142[x1707] = x1714;
x1705 += 1;
x1703 += 50;

}
x1691 += 1;
x1700 += 1;

}
x1694 += 1;
x1697 *= 0;

}
for(int x1737=0; x1737 < 1; x1737++) {
int32_t x1738 = x1736;
int32_t x1739 = x1738;
for(int x1740=0; x1740 < 50; x1740++) {
int32_t x1741 = x1739;
int32_t x1742 = x1741;
int32_t x1743 = x1733;
int32_t x1744 = x1743;
for(int x1745=0; x1745 < 50; x1745++) {
int32_t x1746 = x1730;
double x1747 = x345[x1746];
int32_t x1748 = x1744;
double x1749 = x478[x1748];
int32_t x1750 = x1742;
double x1751 = x37[x1750];
double x1752 = x1749 * x1751;
double x1753 = x1747 + x1752;
x345[x1746] = x1753;
x1744 += 1;
x1742 += 50;

}
x1730 += 1;
x1739 += 1;

}
x1733 += 50;
x1736 *= 0;

}
for(int x1767=0; x1767 < 50; x1767++) {
double x1768 = x440[x1767];
double x1769 = x445[x1767];
double x1772 = x455[x1767];
double x1770 = 1.0 - x1769;
double x1771 = x1770 * x1769;
double x1773 = x1771 * x1772;
double x1774 = x1768 + x1773;
x440[x1767] = x1774;

}
// backpropagate +
for(int x1779=0; x1779 < 50; x1779++) {
double x1780 = x427[x1779];
double x1781 = x440[x1779];
double x1782 = x1780 + x1781;
x427[x1779] = x1782;

}
for(int x1786=0; x1786 < 50; x1786++) {
double x1787 = x137[x1786];
double x1788 = x440[x1786];
double x1789 = x1787 + x1788;
x137[x1786] = x1789;

}
// backpropagate +
for(int x1794=0; x1794 < 50; x1794++) {
double x1795 = x391[x1794];
double x1796 = x427[x1794];
double x1797 = x1795 + x1796;
x391[x1794] = x1797;

}
for(int x1801=0; x1801 < 50; x1801++) {
double x1802 = x414[x1801];
double x1803 = x427[x1801];
double x1804 = x1802 + x1803;
x414[x1801] = x1804;

}
for(int x1817=0; x1817 < 50; x1817++) {
int32_t x1818 = x1816;
int32_t x1819 = x1818;
for(int x1820=0; x1820 < 26; x1820++) {
int32_t x1821 = x1819;
int32_t x1822 = x1821;
int32_t x1823 = x1813;
int32_t x1824 = x1823;
for(int x1825=0; x1825 < 1; x1825++) {
int32_t x1826 = x1810;
double x1827 = x132[x1826];
int32_t x1828 = x1824;
double x1829 = x414[x1828];
int32_t x1830 = x1822;
double x1831 = x349[x1830];
double x1832 = x1829 * x1831;
double x1833 = x1827 + x1832;
x132[x1826] = x1833;
x1824 += 1;
x1822 += 26;

}
x1810 += 1;
x1819 += 1;

}
x1813 += 1;
x1816 *= 0;

}
for(int x1856=0; x1856 < 1; x1856++) {
int32_t x1857 = x1855;
int32_t x1858 = x1857;
for(int x1859=0; x1859 < 26; x1859++) {
int32_t x1860 = x1858;
int32_t x1861 = x1860;
int32_t x1862 = x1852;
int32_t x1863 = x1862;
for(int x1864=0; x1864 < 50; x1864++) {
int32_t x1865 = x1849;
double x1866 = x356[x1865];
int32_t x1867 = x1863;
double x1868 = x414[x1867];
int32_t x1869 = x1861;
double x1870 = x23[x1869];
double x1871 = x1868 * x1870;
double x1872 = x1866 + x1871;
x356[x1865] = x1872;
x1863 += 1;
x1861 += 26;

}
x1849 += 1;
x1858 += 1;

}
x1852 += 50;
x1855 *= 0;

}
for(int x1895=0; x1895 < 50; x1895++) {
int32_t x1896 = x1894;
int32_t x1897 = x1896;
for(int x1898=0; x1898 < 50; x1898++) {
int32_t x1899 = x1897;
int32_t x1900 = x1899;
int32_t x1901 = x1891;
int32_t x1902 = x1901;
for(int x1903=0; x1903 < 1; x1903++) {
int32_t x1904 = x1888;
double x1905 = x127[x1904];
int32_t x1906 = x1902;
double x1907 = x391[x1906];
int32_t x1908 = x1900;
double x1909 = x344[x1908];
double x1910 = x1907 * x1909;
double x1911 = x1905 + x1910;
x127[x1904] = x1911;
x1902 += 1;
x1900 += 50;

}
x1888 += 1;
x1897 += 1;

}
x1891 += 1;
x1894 *= 0;

}
for(int x1934=0; x1934 < 1; x1934++) {
int32_t x1935 = x1933;
int32_t x1936 = x1935;
for(int x1937=0; x1937 < 50; x1937++) {
int32_t x1938 = x1936;
int32_t x1939 = x1938;
int32_t x1940 = x1930;
int32_t x1941 = x1940;
for(int x1942=0; x1942 < 50; x1942++) {
int32_t x1943 = x1927;
double x1944 = x345[x1943];
int32_t x1945 = x1941;
double x1946 = x391[x1945];
int32_t x1947 = x1939;
double x1948 = x15[x1947];
double x1949 = x1946 * x1948;
double x1950 = x1944 + x1949;
x345[x1943] = x1950;
x1941 += 1;
x1939 += 50;

}
x1927 += 1;
x1936 += 1;

}
x1930 += 50;
x1933 *= 0;

}
} else {
for(int x1965=0; x1965 < 50; x1965++) {
double x1966 = x344[x1965];
x117[x1965] = x1966;

}
for(int x1970=0; x1970 < 50; x1970++) {
double x1971 = x346[x1970];
x122[x1970] = x1971;

}
for(int x1975=0; x1975 < 1; x1975++) {
double x1976 = x343[x1975];
x343[x1975] = 1.0;

}
for(int x1980=0; x1980 < 1; x1980++) {
double x1981 = x342[x1980];
x322[x1980] = x1981;

}
}
};
x280 += 20;
int32_t x286 = x280;
int32_t x287 = x286 + 20;
int32_t x288 = x287 + 1;
bool x289 = x288 >= x3;
if (x289) {
x280 = 0;
for(int x291=0; x291 < 50; x291++) {
double x292 = x107[x291];
x107[x291] = 0.0;

}
} else {
}
for(int x301=0; x301 < 20; x301++) {
int32_t x302 = x280;
int32_t x303 = x302 + x301;
int32_t x304 = x6[x303];
x298[x301] = x304;
int32_t x306 = x303 + 1;
int32_t x307 = x6[x306];
x299[x301] = x307;

}
double* x311 = (double*)myMalloc(1 * sizeof(double));
for(int x313=0; x313 < 1; x313++) {
x311[x313] = 0.0;

}
double* x317 = (double*)myMalloc(1 * sizeof(double));
for(int x318=0; x318 < 1; x318++) {
x317[x318] = 0.0;

}
for(int x323=0; x323 < 1; x323++) {
x322[x323] = 0.0;

}
double* x327 = (double*)myMalloc(1 * sizeof(double));
for(int x328=0; x328 < 1; x328++) {
x327[x328] = 0.0;

}
double* x332 = (double*)myMalloc(1 * sizeof(double));
for(int x333=0; x333 < 1; x333++) {
x332[x333] = 0.0;

}
double** x1988 = (double**)myMalloc(6 * sizeof(double*));
x1988[0] = x327;
x1988[1] = x332;
x1988[2] = x107;
x1988[3] = x197;
x1988[4] = x112;
x1988[5] = x202;
x337(0,x1988);
double x1997 = x322[0];
double x1998 = x282;
double x1999 = x1998 * 0.9;
double x2000 = x1997 * 0.1;
double x2001 = x1999 + x2000;
x282 = x2001;
int32_t x2003 = x284 % 100;
bool x2004 = x2003 == 0;
if (x2004) {
double x2005 = x282;
printf("iter %d, loss %f\n",x284,x2005);
int32_t x2007 = x284 / 100;
x278[x2007] = x2005;
} else {
}
for(int x2011=0; x2011 < 2500; x2011++) {
double x2012 = x127[x2011];
bool x2013 = x2012 > 5.0;
if (x2013) {
x127[x2011] = 5.0;
} else {
}
double x2017 = x127[x2011];
bool x2018 = x2017 < -5.0;
if (x2018) {
x127[x2011] = -5.0;
} else {
}

}
double* x2024 = (double*)myMalloc(2500 * sizeof(double));
for(int x2025=0; x2025 < 2500; x2025++) {
double x2026 = x127[x2025];
double x2027 = x127[x2025];
double x2028 = x2026 * x2027;
x2024[x2025] = x2028;

}
for(int x2032=0; x2032 < 2500; x2032++) {
double x2033 = x207[x2032];
double x2034 = x2024[x2032];
double x2035 = x2033 + x2034;
x207[x2032] = x2035;

}
double* x2039 = (double*)myMalloc(2500 * sizeof(double));
for(int x2040=0; x2040 < 2500; x2040++) {
double x2041 = x127[x2040];
double x2042 = x2041 * 0.1;
x2039[x2040] = x2042;

}
double* x2046 = (double*)myMalloc(2500 * sizeof(double));
for(int x2047=0; x2047 < 2500; x2047++) {
double x2048 = x207[x2047];
double x2049 = x2048 + 1.0E-8;
x2046[x2047] = x2049;

}
double* x2053 = (double*)myMalloc(2500 * sizeof(double));
for(int x2054=0; x2054 < 2500; x2054++) {
double x2055 = x2046[x2054];
double x2056 = sqrt(x2055);
x2053[x2054] = x2056;

}
double* x2060 = (double*)myMalloc(2500 * sizeof(double));
for(int x2061=0; x2061 < 2500; x2061++) {
double x2062 = x2039[x2061];
double x2063 = x2053[x2061];
double x2064 = x2062 / x2063;
x2060[x2061] = x2064;

}
for(int x2068=0; x2068 < 2500; x2068++) {
double x2069 = x15[x2068];
double x2070 = x2060[x2068];
double x2071 = x2069 - x2070;
x15[x2068] = x2071;

}
for(int x2075=0; x2075 < 2500; x2075++) {
double x2076 = x127[x2075];
x127[x2075] = 0.0;

}
for(int x2080=0; x2080 < 1300; x2080++) {
double x2081 = x132[x2080];
bool x2082 = x2081 > 5.0;
if (x2082) {
x132[x2080] = 5.0;
} else {
}
double x2086 = x132[x2080];
bool x2087 = x2086 < -5.0;
if (x2087) {
x132[x2080] = -5.0;
} else {
}

}
double* x2093 = (double*)myMalloc(1300 * sizeof(double));
for(int x2094=0; x2094 < 1300; x2094++) {
double x2095 = x132[x2094];
double x2096 = x132[x2094];
double x2097 = x2095 * x2096;
x2093[x2094] = x2097;

}
for(int x2101=0; x2101 < 1300; x2101++) {
double x2102 = x212[x2101];
double x2103 = x2093[x2101];
double x2104 = x2102 + x2103;
x212[x2101] = x2104;

}
double* x2108 = (double*)myMalloc(1300 * sizeof(double));
for(int x2109=0; x2109 < 1300; x2109++) {
double x2110 = x132[x2109];
double x2111 = x2110 * 0.1;
x2108[x2109] = x2111;

}
double* x2115 = (double*)myMalloc(1300 * sizeof(double));
for(int x2116=0; x2116 < 1300; x2116++) {
double x2117 = x212[x2116];
double x2118 = x2117 + 1.0E-8;
x2115[x2116] = x2118;

}
double* x2122 = (double*)myMalloc(1300 * sizeof(double));
for(int x2123=0; x2123 < 1300; x2123++) {
double x2124 = x2115[x2123];
double x2125 = sqrt(x2124);
x2122[x2123] = x2125;

}
double* x2129 = (double*)myMalloc(1300 * sizeof(double));
for(int x2130=0; x2130 < 1300; x2130++) {
double x2131 = x2108[x2130];
double x2132 = x2122[x2130];
double x2133 = x2131 / x2132;
x2129[x2130] = x2133;

}
for(int x2137=0; x2137 < 1300; x2137++) {
double x2138 = x23[x2137];
double x2139 = x2129[x2137];
double x2140 = x2138 - x2139;
x23[x2137] = x2140;

}
for(int x2144=0; x2144 < 1300; x2144++) {
double x2145 = x132[x2144];
x132[x2144] = 0.0;

}
for(int x2149=0; x2149 < 50; x2149++) {
double x2150 = x137[x2149];
bool x2151 = x2150 > 5.0;
if (x2151) {
x137[x2149] = 5.0;
} else {
}
double x2155 = x137[x2149];
bool x2156 = x2155 < -5.0;
if (x2156) {
x137[x2149] = -5.0;
} else {
}

}
double* x2162 = (double*)myMalloc(50 * sizeof(double));
for(int x2163=0; x2163 < 50; x2163++) {
double x2164 = x137[x2163];
double x2165 = x137[x2163];
double x2166 = x2164 * x2165;
x2162[x2163] = x2166;

}
for(int x2170=0; x2170 < 50; x2170++) {
double x2171 = x217[x2170];
double x2172 = x2162[x2170];
double x2173 = x2171 + x2172;
x217[x2170] = x2173;

}
double* x2177 = (double*)myMalloc(50 * sizeof(double));
for(int x2178=0; x2178 < 50; x2178++) {
double x2179 = x137[x2178];
double x2180 = x2179 * 0.1;
x2177[x2178] = x2180;

}
double* x2184 = (double*)myMalloc(50 * sizeof(double));
for(int x2185=0; x2185 < 50; x2185++) {
double x2186 = x217[x2185];
double x2187 = x2186 + 1.0E-8;
x2184[x2185] = x2187;

}
double* x2191 = (double*)myMalloc(50 * sizeof(double));
for(int x2192=0; x2192 < 50; x2192++) {
double x2193 = x2184[x2192];
double x2194 = sqrt(x2193);
x2191[x2192] = x2194;

}
double* x2198 = (double*)myMalloc(50 * sizeof(double));
for(int x2199=0; x2199 < 50; x2199++) {
double x2200 = x2177[x2199];
double x2201 = x2191[x2199];
double x2202 = x2200 / x2201;
x2198[x2199] = x2202;

}
for(int x2206=0; x2206 < 50; x2206++) {
double x2207 = x31[x2206];
double x2208 = x2198[x2206];
double x2209 = x2207 - x2208;
x31[x2206] = x2209;

}
for(int x2213=0; x2213 < 50; x2213++) {
double x2214 = x137[x2213];
x137[x2213] = 0.0;

}
for(int x2218=0; x2218 < 2500; x2218++) {
double x2219 = x142[x2218];
bool x2220 = x2219 > 5.0;
if (x2220) {
x142[x2218] = 5.0;
} else {
}
double x2224 = x142[x2218];
bool x2225 = x2224 < -5.0;
if (x2225) {
x142[x2218] = -5.0;
} else {
}

}
double* x2231 = (double*)myMalloc(2500 * sizeof(double));
for(int x2232=0; x2232 < 2500; x2232++) {
double x2233 = x142[x2232];
double x2234 = x142[x2232];
double x2235 = x2233 * x2234;
x2231[x2232] = x2235;

}
for(int x2239=0; x2239 < 2500; x2239++) {
double x2240 = x222[x2239];
double x2241 = x2231[x2239];
double x2242 = x2240 + x2241;
x222[x2239] = x2242;

}
double* x2246 = (double*)myMalloc(2500 * sizeof(double));
for(int x2247=0; x2247 < 2500; x2247++) {
double x2248 = x142[x2247];
double x2249 = x2248 * 0.1;
x2246[x2247] = x2249;

}
double* x2253 = (double*)myMalloc(2500 * sizeof(double));
for(int x2254=0; x2254 < 2500; x2254++) {
double x2255 = x222[x2254];
double x2256 = x2255 + 1.0E-8;
x2253[x2254] = x2256;

}
double* x2260 = (double*)myMalloc(2500 * sizeof(double));
for(int x2261=0; x2261 < 2500; x2261++) {
double x2262 = x2253[x2261];
double x2263 = sqrt(x2262);
x2260[x2261] = x2263;

}
double* x2267 = (double*)myMalloc(2500 * sizeof(double));
for(int x2268=0; x2268 < 2500; x2268++) {
double x2269 = x2246[x2268];
double x2270 = x2260[x2268];
double x2271 = x2269 / x2270;
x2267[x2268] = x2271;

}
for(int x2275=0; x2275 < 2500; x2275++) {
double x2276 = x37[x2275];
double x2277 = x2267[x2275];
double x2278 = x2276 - x2277;
x37[x2275] = x2278;

}
for(int x2282=0; x2282 < 2500; x2282++) {
double x2283 = x142[x2282];
x142[x2282] = 0.0;

}
for(int x2287=0; x2287 < 1300; x2287++) {
double x2288 = x147[x2287];
bool x2289 = x2288 > 5.0;
if (x2289) {
x147[x2287] = 5.0;
} else {
}
double x2293 = x147[x2287];
bool x2294 = x2293 < -5.0;
if (x2294) {
x147[x2287] = -5.0;
} else {
}

}
double* x2300 = (double*)myMalloc(1300 * sizeof(double));
for(int x2301=0; x2301 < 1300; x2301++) {
double x2302 = x147[x2301];
double x2303 = x147[x2301];
double x2304 = x2302 * x2303;
x2300[x2301] = x2304;

}
for(int x2308=0; x2308 < 1300; x2308++) {
double x2309 = x227[x2308];
double x2310 = x2300[x2308];
double x2311 = x2309 + x2310;
x227[x2308] = x2311;

}
double* x2315 = (double*)myMalloc(1300 * sizeof(double));
for(int x2316=0; x2316 < 1300; x2316++) {
double x2317 = x147[x2316];
double x2318 = x2317 * 0.1;
x2315[x2316] = x2318;

}
double* x2322 = (double*)myMalloc(1300 * sizeof(double));
for(int x2323=0; x2323 < 1300; x2323++) {
double x2324 = x227[x2323];
double x2325 = x2324 + 1.0E-8;
x2322[x2323] = x2325;

}
double* x2329 = (double*)myMalloc(1300 * sizeof(double));
for(int x2330=0; x2330 < 1300; x2330++) {
double x2331 = x2322[x2330];
double x2332 = sqrt(x2331);
x2329[x2330] = x2332;

}
double* x2336 = (double*)myMalloc(1300 * sizeof(double));
for(int x2337=0; x2337 < 1300; x2337++) {
double x2338 = x2315[x2337];
double x2339 = x2329[x2337];
double x2340 = x2338 / x2339;
x2336[x2337] = x2340;

}
for(int x2344=0; x2344 < 1300; x2344++) {
double x2345 = x44[x2344];
double x2346 = x2336[x2344];
double x2347 = x2345 - x2346;
x44[x2344] = x2347;

}
for(int x2351=0; x2351 < 1300; x2351++) {
double x2352 = x147[x2351];
x147[x2351] = 0.0;

}
for(int x2356=0; x2356 < 50; x2356++) {
double x2357 = x152[x2356];
bool x2358 = x2357 > 5.0;
if (x2358) {
x152[x2356] = 5.0;
} else {
}
double x2362 = x152[x2356];
bool x2363 = x2362 < -5.0;
if (x2363) {
x152[x2356] = -5.0;
} else {
}

}
double* x2369 = (double*)myMalloc(50 * sizeof(double));
for(int x2370=0; x2370 < 50; x2370++) {
double x2371 = x152[x2370];
double x2372 = x152[x2370];
double x2373 = x2371 * x2372;
x2369[x2370] = x2373;

}
for(int x2377=0; x2377 < 50; x2377++) {
double x2378 = x232[x2377];
double x2379 = x2369[x2377];
double x2380 = x2378 + x2379;
x232[x2377] = x2380;

}
double* x2384 = (double*)myMalloc(50 * sizeof(double));
for(int x2385=0; x2385 < 50; x2385++) {
double x2386 = x152[x2385];
double x2387 = x2386 * 0.1;
x2384[x2385] = x2387;

}
double* x2391 = (double*)myMalloc(50 * sizeof(double));
for(int x2392=0; x2392 < 50; x2392++) {
double x2393 = x232[x2392];
double x2394 = x2393 + 1.0E-8;
x2391[x2392] = x2394;

}
double* x2398 = (double*)myMalloc(50 * sizeof(double));
for(int x2399=0; x2399 < 50; x2399++) {
double x2400 = x2391[x2399];
double x2401 = sqrt(x2400);
x2398[x2399] = x2401;

}
double* x2405 = (double*)myMalloc(50 * sizeof(double));
for(int x2406=0; x2406 < 50; x2406++) {
double x2407 = x2384[x2406];
double x2408 = x2398[x2406];
double x2409 = x2407 / x2408;
x2405[x2406] = x2409;

}
for(int x2413=0; x2413 < 50; x2413++) {
double x2414 = x51[x2413];
double x2415 = x2405[x2413];
double x2416 = x2414 - x2415;
x51[x2413] = x2416;

}
for(int x2420=0; x2420 < 50; x2420++) {
double x2421 = x152[x2420];
x152[x2420] = 0.0;

}
for(int x2425=0; x2425 < 2500; x2425++) {
double x2426 = x157[x2425];
bool x2427 = x2426 > 5.0;
if (x2427) {
x157[x2425] = 5.0;
} else {
}
double x2431 = x157[x2425];
bool x2432 = x2431 < -5.0;
if (x2432) {
x157[x2425] = -5.0;
} else {
}

}
double* x2438 = (double*)myMalloc(2500 * sizeof(double));
for(int x2439=0; x2439 < 2500; x2439++) {
double x2440 = x157[x2439];
double x2441 = x157[x2439];
double x2442 = x2440 * x2441;
x2438[x2439] = x2442;

}
for(int x2446=0; x2446 < 2500; x2446++) {
double x2447 = x237[x2446];
double x2448 = x2438[x2446];
double x2449 = x2447 + x2448;
x237[x2446] = x2449;

}
double* x2453 = (double*)myMalloc(2500 * sizeof(double));
for(int x2454=0; x2454 < 2500; x2454++) {
double x2455 = x157[x2454];
double x2456 = x2455 * 0.1;
x2453[x2454] = x2456;

}
double* x2460 = (double*)myMalloc(2500 * sizeof(double));
for(int x2461=0; x2461 < 2500; x2461++) {
double x2462 = x237[x2461];
double x2463 = x2462 + 1.0E-8;
x2460[x2461] = x2463;

}
double* x2467 = (double*)myMalloc(2500 * sizeof(double));
for(int x2468=0; x2468 < 2500; x2468++) {
double x2469 = x2460[x2468];
double x2470 = sqrt(x2469);
x2467[x2468] = x2470;

}
double* x2474 = (double*)myMalloc(2500 * sizeof(double));
for(int x2475=0; x2475 < 2500; x2475++) {
double x2476 = x2453[x2475];
double x2477 = x2467[x2475];
double x2478 = x2476 / x2477;
x2474[x2475] = x2478;

}
for(int x2482=0; x2482 < 2500; x2482++) {
double x2483 = x56[x2482];
double x2484 = x2474[x2482];
double x2485 = x2483 - x2484;
x56[x2482] = x2485;

}
for(int x2489=0; x2489 < 2500; x2489++) {
double x2490 = x157[x2489];
x157[x2489] = 0.0;

}
for(int x2494=0; x2494 < 1300; x2494++) {
double x2495 = x162[x2494];
bool x2496 = x2495 > 5.0;
if (x2496) {
x162[x2494] = 5.0;
} else {
}
double x2500 = x162[x2494];
bool x2501 = x2500 < -5.0;
if (x2501) {
x162[x2494] = -5.0;
} else {
}

}
double* x2507 = (double*)myMalloc(1300 * sizeof(double));
for(int x2508=0; x2508 < 1300; x2508++) {
double x2509 = x162[x2508];
double x2510 = x162[x2508];
double x2511 = x2509 * x2510;
x2507[x2508] = x2511;

}
for(int x2515=0; x2515 < 1300; x2515++) {
double x2516 = x242[x2515];
double x2517 = x2507[x2515];
double x2518 = x2516 + x2517;
x242[x2515] = x2518;

}
double* x2522 = (double*)myMalloc(1300 * sizeof(double));
for(int x2523=0; x2523 < 1300; x2523++) {
double x2524 = x162[x2523];
double x2525 = x2524 * 0.1;
x2522[x2523] = x2525;

}
double* x2529 = (double*)myMalloc(1300 * sizeof(double));
for(int x2530=0; x2530 < 1300; x2530++) {
double x2531 = x242[x2530];
double x2532 = x2531 + 1.0E-8;
x2529[x2530] = x2532;

}
double* x2536 = (double*)myMalloc(1300 * sizeof(double));
for(int x2537=0; x2537 < 1300; x2537++) {
double x2538 = x2529[x2537];
double x2539 = sqrt(x2538);
x2536[x2537] = x2539;

}
double* x2543 = (double*)myMalloc(1300 * sizeof(double));
for(int x2544=0; x2544 < 1300; x2544++) {
double x2545 = x2522[x2544];
double x2546 = x2536[x2544];
double x2547 = x2545 / x2546;
x2543[x2544] = x2547;

}
for(int x2551=0; x2551 < 1300; x2551++) {
double x2552 = x63[x2551];
double x2553 = x2543[x2551];
double x2554 = x2552 - x2553;
x63[x2551] = x2554;

}
for(int x2558=0; x2558 < 1300; x2558++) {
double x2559 = x162[x2558];
x162[x2558] = 0.0;

}
for(int x2563=0; x2563 < 50; x2563++) {
double x2564 = x167[x2563];
bool x2565 = x2564 > 5.0;
if (x2565) {
x167[x2563] = 5.0;
} else {
}
double x2569 = x167[x2563];
bool x2570 = x2569 < -5.0;
if (x2570) {
x167[x2563] = -5.0;
} else {
}

}
double* x2576 = (double*)myMalloc(50 * sizeof(double));
for(int x2577=0; x2577 < 50; x2577++) {
double x2578 = x167[x2577];
double x2579 = x167[x2577];
double x2580 = x2578 * x2579;
x2576[x2577] = x2580;

}
for(int x2584=0; x2584 < 50; x2584++) {
double x2585 = x247[x2584];
double x2586 = x2576[x2584];
double x2587 = x2585 + x2586;
x247[x2584] = x2587;

}
double* x2591 = (double*)myMalloc(50 * sizeof(double));
for(int x2592=0; x2592 < 50; x2592++) {
double x2593 = x167[x2592];
double x2594 = x2593 * 0.1;
x2591[x2592] = x2594;

}
double* x2598 = (double*)myMalloc(50 * sizeof(double));
for(int x2599=0; x2599 < 50; x2599++) {
double x2600 = x247[x2599];
double x2601 = x2600 + 1.0E-8;
x2598[x2599] = x2601;

}
double* x2605 = (double*)myMalloc(50 * sizeof(double));
for(int x2606=0; x2606 < 50; x2606++) {
double x2607 = x2598[x2606];
double x2608 = sqrt(x2607);
x2605[x2606] = x2608;

}
double* x2612 = (double*)myMalloc(50 * sizeof(double));
for(int x2613=0; x2613 < 50; x2613++) {
double x2614 = x2591[x2613];
double x2615 = x2605[x2613];
double x2616 = x2614 / x2615;
x2612[x2613] = x2616;

}
for(int x2620=0; x2620 < 50; x2620++) {
double x2621 = x70[x2620];
double x2622 = x2612[x2620];
double x2623 = x2621 - x2622;
x70[x2620] = x2623;

}
for(int x2627=0; x2627 < 50; x2627++) {
double x2628 = x167[x2627];
x167[x2627] = 0.0;

}
for(int x2632=0; x2632 < 2500; x2632++) {
double x2633 = x172[x2632];
bool x2634 = x2633 > 5.0;
if (x2634) {
x172[x2632] = 5.0;
} else {
}
double x2638 = x172[x2632];
bool x2639 = x2638 < -5.0;
if (x2639) {
x172[x2632] = -5.0;
} else {
}

}
double* x2645 = (double*)myMalloc(2500 * sizeof(double));
for(int x2646=0; x2646 < 2500; x2646++) {
double x2647 = x172[x2646];
double x2648 = x172[x2646];
double x2649 = x2647 * x2648;
x2645[x2646] = x2649;

}
for(int x2653=0; x2653 < 2500; x2653++) {
double x2654 = x252[x2653];
double x2655 = x2645[x2653];
double x2656 = x2654 + x2655;
x252[x2653] = x2656;

}
double* x2660 = (double*)myMalloc(2500 * sizeof(double));
for(int x2661=0; x2661 < 2500; x2661++) {
double x2662 = x172[x2661];
double x2663 = x2662 * 0.1;
x2660[x2661] = x2663;

}
double* x2667 = (double*)myMalloc(2500 * sizeof(double));
for(int x2668=0; x2668 < 2500; x2668++) {
double x2669 = x252[x2668];
double x2670 = x2669 + 1.0E-8;
x2667[x2668] = x2670;

}
double* x2674 = (double*)myMalloc(2500 * sizeof(double));
for(int x2675=0; x2675 < 2500; x2675++) {
double x2676 = x2667[x2675];
double x2677 = sqrt(x2676);
x2674[x2675] = x2677;

}
double* x2681 = (double*)myMalloc(2500 * sizeof(double));
for(int x2682=0; x2682 < 2500; x2682++) {
double x2683 = x2660[x2682];
double x2684 = x2674[x2682];
double x2685 = x2683 / x2684;
x2681[x2682] = x2685;

}
for(int x2689=0; x2689 < 2500; x2689++) {
double x2690 = x75[x2689];
double x2691 = x2681[x2689];
double x2692 = x2690 - x2691;
x75[x2689] = x2692;

}
for(int x2696=0; x2696 < 2500; x2696++) {
double x2697 = x172[x2696];
x172[x2696] = 0.0;

}
for(int x2701=0; x2701 < 1300; x2701++) {
double x2702 = x177[x2701];
bool x2703 = x2702 > 5.0;
if (x2703) {
x177[x2701] = 5.0;
} else {
}
double x2707 = x177[x2701];
bool x2708 = x2707 < -5.0;
if (x2708) {
x177[x2701] = -5.0;
} else {
}

}
double* x2714 = (double*)myMalloc(1300 * sizeof(double));
for(int x2715=0; x2715 < 1300; x2715++) {
double x2716 = x177[x2715];
double x2717 = x177[x2715];
double x2718 = x2716 * x2717;
x2714[x2715] = x2718;

}
for(int x2722=0; x2722 < 1300; x2722++) {
double x2723 = x257[x2722];
double x2724 = x2714[x2722];
double x2725 = x2723 + x2724;
x257[x2722] = x2725;

}
double* x2729 = (double*)myMalloc(1300 * sizeof(double));
for(int x2730=0; x2730 < 1300; x2730++) {
double x2731 = x177[x2730];
double x2732 = x2731 * 0.1;
x2729[x2730] = x2732;

}
double* x2736 = (double*)myMalloc(1300 * sizeof(double));
for(int x2737=0; x2737 < 1300; x2737++) {
double x2738 = x257[x2737];
double x2739 = x2738 + 1.0E-8;
x2736[x2737] = x2739;

}
double* x2743 = (double*)myMalloc(1300 * sizeof(double));
for(int x2744=0; x2744 < 1300; x2744++) {
double x2745 = x2736[x2744];
double x2746 = sqrt(x2745);
x2743[x2744] = x2746;

}
double* x2750 = (double*)myMalloc(1300 * sizeof(double));
for(int x2751=0; x2751 < 1300; x2751++) {
double x2752 = x2729[x2751];
double x2753 = x2743[x2751];
double x2754 = x2752 / x2753;
x2750[x2751] = x2754;

}
for(int x2758=0; x2758 < 1300; x2758++) {
double x2759 = x82[x2758];
double x2760 = x2750[x2758];
double x2761 = x2759 - x2760;
x82[x2758] = x2761;

}
for(int x2765=0; x2765 < 1300; x2765++) {
double x2766 = x177[x2765];
x177[x2765] = 0.0;

}
for(int x2770=0; x2770 < 50; x2770++) {
double x2771 = x182[x2770];
bool x2772 = x2771 > 5.0;
if (x2772) {
x182[x2770] = 5.0;
} else {
}
double x2776 = x182[x2770];
bool x2777 = x2776 < -5.0;
if (x2777) {
x182[x2770] = -5.0;
} else {
}

}
double* x2783 = (double*)myMalloc(50 * sizeof(double));
for(int x2784=0; x2784 < 50; x2784++) {
double x2785 = x182[x2784];
double x2786 = x182[x2784];
double x2787 = x2785 * x2786;
x2783[x2784] = x2787;

}
for(int x2791=0; x2791 < 50; x2791++) {
double x2792 = x262[x2791];
double x2793 = x2783[x2791];
double x2794 = x2792 + x2793;
x262[x2791] = x2794;

}
double* x2798 = (double*)myMalloc(50 * sizeof(double));
for(int x2799=0; x2799 < 50; x2799++) {
double x2800 = x182[x2799];
double x2801 = x2800 * 0.1;
x2798[x2799] = x2801;

}
double* x2805 = (double*)myMalloc(50 * sizeof(double));
for(int x2806=0; x2806 < 50; x2806++) {
double x2807 = x262[x2806];
double x2808 = x2807 + 1.0E-8;
x2805[x2806] = x2808;

}
double* x2812 = (double*)myMalloc(50 * sizeof(double));
for(int x2813=0; x2813 < 50; x2813++) {
double x2814 = x2805[x2813];
double x2815 = sqrt(x2814);
x2812[x2813] = x2815;

}
double* x2819 = (double*)myMalloc(50 * sizeof(double));
for(int x2820=0; x2820 < 50; x2820++) {
double x2821 = x2798[x2820];
double x2822 = x2812[x2820];
double x2823 = x2821 / x2822;
x2819[x2820] = x2823;

}
for(int x2827=0; x2827 < 50; x2827++) {
double x2828 = x89[x2827];
double x2829 = x2819[x2827];
double x2830 = x2828 - x2829;
x89[x2827] = x2830;

}
for(int x2834=0; x2834 < 50; x2834++) {
double x2835 = x182[x2834];
x182[x2834] = 0.0;

}
for(int x2839=0; x2839 < 1300; x2839++) {
double x2840 = x187[x2839];
bool x2841 = x2840 > 5.0;
if (x2841) {
x187[x2839] = 5.0;
} else {
}
double x2845 = x187[x2839];
bool x2846 = x2845 < -5.0;
if (x2846) {
x187[x2839] = -5.0;
} else {
}

}
double* x2852 = (double*)myMalloc(1300 * sizeof(double));
for(int x2853=0; x2853 < 1300; x2853++) {
double x2854 = x187[x2853];
double x2855 = x187[x2853];
double x2856 = x2854 * x2855;
x2852[x2853] = x2856;

}
for(int x2860=0; x2860 < 1300; x2860++) {
double x2861 = x267[x2860];
double x2862 = x2852[x2860];
double x2863 = x2861 + x2862;
x267[x2860] = x2863;

}
double* x2867 = (double*)myMalloc(1300 * sizeof(double));
for(int x2868=0; x2868 < 1300; x2868++) {
double x2869 = x187[x2868];
double x2870 = x2869 * 0.1;
x2867[x2868] = x2870;

}
double* x2874 = (double*)myMalloc(1300 * sizeof(double));
for(int x2875=0; x2875 < 1300; x2875++) {
double x2876 = x267[x2875];
double x2877 = x2876 + 1.0E-8;
x2874[x2875] = x2877;

}
double* x2881 = (double*)myMalloc(1300 * sizeof(double));
for(int x2882=0; x2882 < 1300; x2882++) {
double x2883 = x2874[x2882];
double x2884 = sqrt(x2883);
x2881[x2882] = x2884;

}
double* x2888 = (double*)myMalloc(1300 * sizeof(double));
for(int x2889=0; x2889 < 1300; x2889++) {
double x2890 = x2867[x2889];
double x2891 = x2881[x2889];
double x2892 = x2890 / x2891;
x2888[x2889] = x2892;

}
for(int x2896=0; x2896 < 1300; x2896++) {
double x2897 = x94[x2896];
double x2898 = x2888[x2896];
double x2899 = x2897 - x2898;
x94[x2896] = x2899;

}
for(int x2903=0; x2903 < 1300; x2903++) {
double x2904 = x187[x2903];
x187[x2903] = 0.0;

}
for(int x2908=0; x2908 < 26; x2908++) {
double x2909 = x192[x2908];
bool x2910 = x2909 > 5.0;
if (x2910) {
x192[x2908] = 5.0;
} else {
}
double x2914 = x192[x2908];
bool x2915 = x2914 < -5.0;
if (x2915) {
x192[x2908] = -5.0;
} else {
}

}
double* x2921 = (double*)myMalloc(26 * sizeof(double));
for(int x2922=0; x2922 < 26; x2922++) {
double x2923 = x192[x2922];
double x2924 = x192[x2922];
double x2925 = x2923 * x2924;
x2921[x2922] = x2925;

}
for(int x2929=0; x2929 < 26; x2929++) {
double x2930 = x272[x2929];
double x2931 = x2921[x2929];
double x2932 = x2930 + x2931;
x272[x2929] = x2932;

}
double* x2936 = (double*)myMalloc(26 * sizeof(double));
for(int x2937=0; x2937 < 26; x2937++) {
double x2938 = x192[x2937];
double x2939 = x2938 * 0.1;
x2936[x2937] = x2939;

}
double* x2943 = (double*)myMalloc(26 * sizeof(double));
for(int x2944=0; x2944 < 26; x2944++) {
double x2945 = x272[x2944];
double x2946 = x2945 + 1.0E-8;
x2943[x2944] = x2946;

}
double* x2950 = (double*)myMalloc(26 * sizeof(double));
for(int x2951=0; x2951 < 26; x2951++) {
double x2952 = x2943[x2951];
double x2953 = sqrt(x2952);
x2950[x2951] = x2953;

}
double* x2957 = (double*)myMalloc(26 * sizeof(double));
for(int x2958=0; x2958 < 26; x2958++) {
double x2959 = x2936[x2958];
double x2960 = x2950[x2958];
double x2961 = x2959 / x2960;
x2957[x2958] = x2961;

}
for(int x2965=0; x2965 < 26; x2965++) {
double x2966 = x101[x2965];
double x2967 = x2957[x2965];
double x2968 = x2966 - x2967;
x101[x2965] = x2968;

}
for(int x2972=0; x2972 < 26; x2972++) {
double x2973 = x192[x2972];
x192[x2972] = 0.0;

}
for(int x2977=0; x2977 < 50; x2977++) {
double x2978 = x197[x2977];
x197[x2977] = 0.0;

}
for(int x2982=0; x2982 < 50; x2982++) {
double x2983 = x202[x2982];
x202[x2982] = 0.0;

}
for(int x2987=0; x2987 < 50; x2987++) {
double x2988 = x117[x2987];
x107[x2987] = x2988;

}
for(int x2992=0; x2992 < 50; x2992++) {
double x2993 = x122[x2992];
x112[x2992] = x2993;

}
mallocAddr = (void*)x279;

}
double x3000 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x3003 = (long)fopen(x0, "w");
fprintf((FILE *)x3003, "unit: %s\n", "100 iteration");
for(int x3006=0; x3006 < 51; x3006++) {
double x3007 = x278[x3006];
fprintf((FILE *)x3003, "%lf\n", x3007);

}
double x3001 = x277 - x1;
double x3002 = x3000 - x277;
fprintf((FILE *)x3003, "run time: %lf %lf\n", x3001, x3002);
fclose((FILE*)x3003);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

