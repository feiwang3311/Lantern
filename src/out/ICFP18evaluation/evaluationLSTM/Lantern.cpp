
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
float* x15 = (float*)myMalloc(2500 * sizeof(float));
for(int x17=0; x17 < 2500; x17++) {
float x18 = d(gen);
float x19 = x18 * 0.01f;
x15[x17] = x19;

}
float* x23 = (float*)myMalloc(1300 * sizeof(float));
for(int x25=0; x25 < 1300; x25++) {
float x26 = d(gen);
float x27 = x26 * 0.01f;
x23[x25] = x27;

}
float* x31 = (float*)myMalloc(50 * sizeof(float));
for(int x33=0; x33 < 50; x33++) {
x31[x33] = 0.0f;

}
float* x37 = (float*)myMalloc(2500 * sizeof(float));
for(int x38=0; x38 < 2500; x38++) {
float x39 = d(gen);
float x40 = x39 * 0.01f;
x37[x38] = x40;

}
float* x44 = (float*)myMalloc(1300 * sizeof(float));
for(int x45=0; x45 < 1300; x45++) {
float x46 = d(gen);
float x47 = x46 * 0.01f;
x44[x45] = x47;

}
float* x51 = (float*)myMalloc(50 * sizeof(float));
for(int x52=0; x52 < 50; x52++) {
x51[x52] = 0.0f;

}
float* x56 = (float*)myMalloc(2500 * sizeof(float));
for(int x57=0; x57 < 2500; x57++) {
float x58 = d(gen);
float x59 = x58 * 0.01f;
x56[x57] = x59;

}
float* x63 = (float*)myMalloc(1300 * sizeof(float));
for(int x64=0; x64 < 1300; x64++) {
float x65 = d(gen);
float x66 = x65 * 0.01f;
x63[x64] = x66;

}
float* x70 = (float*)myMalloc(50 * sizeof(float));
for(int x71=0; x71 < 50; x71++) {
x70[x71] = 0.0f;

}
float* x75 = (float*)myMalloc(2500 * sizeof(float));
for(int x76=0; x76 < 2500; x76++) {
float x77 = d(gen);
float x78 = x77 * 0.01f;
x75[x76] = x78;

}
float* x82 = (float*)myMalloc(1300 * sizeof(float));
for(int x83=0; x83 < 1300; x83++) {
float x84 = d(gen);
float x85 = x84 * 0.01f;
x82[x83] = x85;

}
float* x89 = (float*)myMalloc(50 * sizeof(float));
for(int x90=0; x90 < 50; x90++) {
x89[x90] = 0.0f;

}
float* x94 = (float*)myMalloc(1300 * sizeof(float));
for(int x95=0; x95 < 1300; x95++) {
float x96 = d(gen);
float x97 = x96 * 0.01f;
x94[x95] = x97;

}
float* x101 = (float*)myMalloc(26 * sizeof(float));
for(int x103=0; x103 < 26; x103++) {
x101[x103] = 0.0f;

}
float* x107 = (float*)myMalloc(50 * sizeof(float));
for(int x108=0; x108 < 50; x108++) {
x107[x108] = 0.0f;

}
float* x112 = (float*)myMalloc(50 * sizeof(float));
for(int x113=0; x113 < 50; x113++) {
x112[x113] = 0.0f;

}
float* x117 = (float*)myMalloc(50 * sizeof(float));
for(int x118=0; x118 < 50; x118++) {
x117[x118] = 0.0f;

}
float* x122 = (float*)myMalloc(50 * sizeof(float));
for(int x123=0; x123 < 50; x123++) {
x122[x123] = 0.0f;

}
float* x127 = (float*)myMalloc(2500 * sizeof(float));
for(int x128=0; x128 < 2500; x128++) {
x127[x128] = 0.0f;

}
float* x132 = (float*)myMalloc(1300 * sizeof(float));
for(int x133=0; x133 < 1300; x133++) {
x132[x133] = 0.0f;

}
float* x137 = (float*)myMalloc(50 * sizeof(float));
for(int x138=0; x138 < 50; x138++) {
x137[x138] = 0.0f;

}
float* x142 = (float*)myMalloc(2500 * sizeof(float));
for(int x143=0; x143 < 2500; x143++) {
x142[x143] = 0.0f;

}
float* x147 = (float*)myMalloc(1300 * sizeof(float));
for(int x148=0; x148 < 1300; x148++) {
x147[x148] = 0.0f;

}
float* x152 = (float*)myMalloc(50 * sizeof(float));
for(int x153=0; x153 < 50; x153++) {
x152[x153] = 0.0f;

}
float* x157 = (float*)myMalloc(2500 * sizeof(float));
for(int x158=0; x158 < 2500; x158++) {
x157[x158] = 0.0f;

}
float* x162 = (float*)myMalloc(1300 * sizeof(float));
for(int x163=0; x163 < 1300; x163++) {
x162[x163] = 0.0f;

}
float* x167 = (float*)myMalloc(50 * sizeof(float));
for(int x168=0; x168 < 50; x168++) {
x167[x168] = 0.0f;

}
float* x172 = (float*)myMalloc(2500 * sizeof(float));
for(int x173=0; x173 < 2500; x173++) {
x172[x173] = 0.0f;

}
float* x177 = (float*)myMalloc(1300 * sizeof(float));
for(int x178=0; x178 < 1300; x178++) {
x177[x178] = 0.0f;

}
float* x182 = (float*)myMalloc(50 * sizeof(float));
for(int x183=0; x183 < 50; x183++) {
x182[x183] = 0.0f;

}
float* x187 = (float*)myMalloc(1300 * sizeof(float));
for(int x188=0; x188 < 1300; x188++) {
x187[x188] = 0.0f;

}
float* x192 = (float*)myMalloc(26 * sizeof(float));
for(int x193=0; x193 < 26; x193++) {
x192[x193] = 0.0f;

}
float* x197 = (float*)myMalloc(50 * sizeof(float));
for(int x198=0; x198 < 50; x198++) {
x197[x198] = 0.0f;

}
float* x202 = (float*)myMalloc(50 * sizeof(float));
for(int x203=0; x203 < 50; x203++) {
x202[x203] = 0.0f;

}
float* x207 = (float*)myMalloc(2500 * sizeof(float));
for(int x208=0; x208 < 2500; x208++) {
x207[x208] = 0.0f;

}
float* x212 = (float*)myMalloc(1300 * sizeof(float));
for(int x213=0; x213 < 1300; x213++) {
x212[x213] = 0.0f;

}
float* x217 = (float*)myMalloc(50 * sizeof(float));
for(int x218=0; x218 < 50; x218++) {
x217[x218] = 0.0f;

}
float* x222 = (float*)myMalloc(2500 * sizeof(float));
for(int x223=0; x223 < 2500; x223++) {
x222[x223] = 0.0f;

}
float* x227 = (float*)myMalloc(1300 * sizeof(float));
for(int x228=0; x228 < 1300; x228++) {
x227[x228] = 0.0f;

}
float* x232 = (float*)myMalloc(50 * sizeof(float));
for(int x233=0; x233 < 50; x233++) {
x232[x233] = 0.0f;

}
float* x237 = (float*)myMalloc(2500 * sizeof(float));
for(int x238=0; x238 < 2500; x238++) {
x237[x238] = 0.0f;

}
float* x242 = (float*)myMalloc(1300 * sizeof(float));
for(int x243=0; x243 < 1300; x243++) {
x242[x243] = 0.0f;

}
float* x247 = (float*)myMalloc(50 * sizeof(float));
for(int x248=0; x248 < 50; x248++) {
x247[x248] = 0.0f;

}
float* x252 = (float*)myMalloc(2500 * sizeof(float));
for(int x253=0; x253 < 2500; x253++) {
x252[x253] = 0.0f;

}
float* x257 = (float*)myMalloc(1300 * sizeof(float));
for(int x258=0; x258 < 1300; x258++) {
x257[x258] = 0.0f;

}
float* x262 = (float*)myMalloc(50 * sizeof(float));
for(int x263=0; x263 < 50; x263++) {
x262[x263] = 0.0f;

}
float* x267 = (float*)myMalloc(1300 * sizeof(float));
for(int x268=0; x268 < 1300; x268++) {
x267[x268] = 0.0f;

}
float* x272 = (float*)myMalloc(26 * sizeof(float));
for(int x273=0; x273 < 26; x273++) {
x272[x273] = 0.0f;

}
double x277 = ((double)clock() / CLOCKS_PER_SEC);
double* x278 = (double*)myMalloc(51 * sizeof(double));
int64_t x279 = (long)mallocAddr;
int32_t x280 = 0;
x280 -= 20;
double x282 = 70.0;
for(int x284=0; x284 < 5001; x284++) {
float* x322 = (float*)myMalloc(1 * sizeof(float));
int32_t* x298 = (int32_t*)myMalloc(20 * sizeof(int32_t));
int32_t* x299 = (int32_t*)myMalloc(20 * sizeof(int32_t));
function<void(int32_t,float**)> x337 = [&](int32_t x338,float** x339) {
float** x341 = x339;
float* x342 = x341[0];
float* x343 = x341[1];
float* x344 = x341[2];
float* x345 = x341[3];
float* x346 = x341[4];
float* x347 = x341[5];
int32_t x340 = x338;
bool x348 = x340 < 20;
if (x348) {
float* x349 = (float*)myMalloc(26 * sizeof(float));
for(int x350=0; x350 < 26; x350++) {
x349[x350] = 0.0f;

}
int32_t x354 = x298[x340];
x349[x354] = 1.0f;
float* x356 = (float*)myMalloc(26 * sizeof(float));
for(int x357=0; x357 < 26; x357++) {
x356[x357] = 0.0f;

}
float* x361 = (float*)myMalloc(26 * sizeof(float));
for(int x362=0; x362 < 26; x362++) {
x361[x362] = 0.0f;

}
int32_t x366 = x299[x340];
x361[x366] = 1.0f;
float* x368 = (float*)myMalloc(26 * sizeof(float));
for(int x369=0; x369 < 26; x369++) {
x368[x369] = 0.0f;

}
// dot WrappedArray(50, 50) - WrappedArray(50)
int32_t x374 = 0;
float* x375 = (float*)myMalloc(50 * sizeof(float));
for(int x376=0; x376 < 50; x376++) {
float x377 = 0.0f;
for(int x378=0; x378 < 50; x378++) {
int32_t x379 = x374;
float x380 = x15[x379];
float x381 = x344[x378];
float x382 = x380 * x381;
x377 += x382;
x374 += 1;

}
float x387 = x377;
x375[x376] = x387;

}
float* x391 = (float*)myMalloc(50 * sizeof(float));
for(int x392=0; x392 < 50; x392++) {
x391[x392] = 0.0f;

}
// dot WrappedArray(50, 26) - WrappedArray(26)
int32_t x397 = 0;
float* x398 = (float*)myMalloc(50 * sizeof(float));
for(int x399=0; x399 < 50; x399++) {
float x400 = 0.0f;
for(int x401=0; x401 < 26; x401++) {
int32_t x402 = x397;
float x403 = x23[x402];
float x404 = x349[x401];
float x405 = x403 * x404;
x400 += x405;
x397 += 1;

}
float x410 = x400;
x398[x399] = x410;

}
float* x414 = (float*)myMalloc(50 * sizeof(float));
for(int x415=0; x415 < 50; x415++) {
x414[x415] = 0.0f;

}
float* x419 = (float*)myMalloc(50 * sizeof(float));
for(int x420=0; x420 < 50; x420++) {
float x421 = x375[x420];
float x422 = x398[x420];
float x423 = x421 + x422;
x419[x420] = x423;

}
float* x427 = (float*)myMalloc(50 * sizeof(float));
for(int x428=0; x428 < 50; x428++) {
x427[x428] = 0.0f;

}
float* x432 = (float*)myMalloc(50 * sizeof(float));
for(int x433=0; x433 < 50; x433++) {
float x434 = x419[x433];
float x435 = x31[x433];
float x436 = x434 + x435;
x432[x433] = x436;

}
float* x440 = (float*)myMalloc(50 * sizeof(float));
for(int x441=0; x441 < 50; x441++) {
x440[x441] = 0.0f;

}
float* x445 = (float*)myMalloc(50 * sizeof(float));
for(int x446=0; x446 < 50; x446++) {
float x447 = x432[x446];
float x448 = -1.0f * x447;
double x449 = (double)x448;
double x450 = exp(x449);
float x451 = (float)x450;
float x452 = x451 + 1.0f;
float x453 = 1.0f / x452;
x445[x446] = x453;

}
float* x457 = (float*)myMalloc(50 * sizeof(float));
for(int x458=0; x458 < 50; x458++) {
x457[x458] = 0.0f;

}
// dot WrappedArray(50, 50) - WrappedArray(50)
int32_t x463 = 0;
float* x464 = (float*)myMalloc(50 * sizeof(float));
for(int x465=0; x465 < 50; x465++) {
float x466 = 0.0f;
for(int x467=0; x467 < 50; x467++) {
int32_t x468 = x463;
float x469 = x37[x468];
float x470 = x344[x467];
float x471 = x469 * x470;
x466 += x471;
x463 += 1;

}
float x476 = x466;
x464[x465] = x476;

}
float* x480 = (float*)myMalloc(50 * sizeof(float));
for(int x481=0; x481 < 50; x481++) {
x480[x481] = 0.0f;

}
// dot WrappedArray(50, 26) - WrappedArray(26)
int32_t x486 = 0;
float* x487 = (float*)myMalloc(50 * sizeof(float));
for(int x488=0; x488 < 50; x488++) {
float x489 = 0.0f;
for(int x490=0; x490 < 26; x490++) {
int32_t x491 = x486;
float x492 = x44[x491];
float x493 = x349[x490];
float x494 = x492 * x493;
x489 += x494;
x486 += 1;

}
float x499 = x489;
x487[x488] = x499;

}
float* x503 = (float*)myMalloc(50 * sizeof(float));
for(int x504=0; x504 < 50; x504++) {
x503[x504] = 0.0f;

}
float* x508 = (float*)myMalloc(50 * sizeof(float));
for(int x509=0; x509 < 50; x509++) {
float x510 = x464[x509];
float x511 = x487[x509];
float x512 = x510 + x511;
x508[x509] = x512;

}
float* x516 = (float*)myMalloc(50 * sizeof(float));
for(int x517=0; x517 < 50; x517++) {
x516[x517] = 0.0f;

}
float* x521 = (float*)myMalloc(50 * sizeof(float));
for(int x522=0; x522 < 50; x522++) {
float x523 = x508[x522];
float x524 = x51[x522];
float x525 = x523 + x524;
x521[x522] = x525;

}
float* x529 = (float*)myMalloc(50 * sizeof(float));
for(int x530=0; x530 < 50; x530++) {
x529[x530] = 0.0f;

}
float* x534 = (float*)myMalloc(50 * sizeof(float));
for(int x535=0; x535 < 50; x535++) {
float x536 = x521[x535];
float x537 = -1.0f * x536;
double x538 = (double)x537;
double x539 = exp(x538);
float x540 = (float)x539;
float x541 = x540 + 1.0f;
float x542 = 1.0f / x541;
x534[x535] = x542;

}
float* x546 = (float*)myMalloc(50 * sizeof(float));
for(int x547=0; x547 < 50; x547++) {
x546[x547] = 0.0f;

}
// dot WrappedArray(50, 50) - WrappedArray(50)
int32_t x552 = 0;
float* x553 = (float*)myMalloc(50 * sizeof(float));
for(int x554=0; x554 < 50; x554++) {
float x555 = 0.0f;
for(int x556=0; x556 < 50; x556++) {
int32_t x557 = x552;
float x558 = x75[x557];
float x559 = x344[x556];
float x560 = x558 * x559;
x555 += x560;
x552 += 1;

}
float x565 = x555;
x553[x554] = x565;

}
float* x569 = (float*)myMalloc(50 * sizeof(float));
for(int x570=0; x570 < 50; x570++) {
x569[x570] = 0.0f;

}
// dot WrappedArray(50, 26) - WrappedArray(26)
int32_t x575 = 0;
float* x576 = (float*)myMalloc(50 * sizeof(float));
for(int x577=0; x577 < 50; x577++) {
float x578 = 0.0f;
for(int x579=0; x579 < 26; x579++) {
int32_t x580 = x575;
float x581 = x82[x580];
float x582 = x349[x579];
float x583 = x581 * x582;
x578 += x583;
x575 += 1;

}
float x588 = x578;
x576[x577] = x588;

}
float* x592 = (float*)myMalloc(50 * sizeof(float));
for(int x593=0; x593 < 50; x593++) {
x592[x593] = 0.0f;

}
float* x597 = (float*)myMalloc(50 * sizeof(float));
for(int x598=0; x598 < 50; x598++) {
float x599 = x553[x598];
float x600 = x576[x598];
float x601 = x599 + x600;
x597[x598] = x601;

}
float* x605 = (float*)myMalloc(50 * sizeof(float));
for(int x606=0; x606 < 50; x606++) {
x605[x606] = 0.0f;

}
float* x610 = (float*)myMalloc(50 * sizeof(float));
for(int x611=0; x611 < 50; x611++) {
float x612 = x597[x611];
float x613 = x89[x611];
float x614 = x612 + x613;
x610[x611] = x614;

}
float* x618 = (float*)myMalloc(50 * sizeof(float));
for(int x619=0; x619 < 50; x619++) {
x618[x619] = 0.0f;

}
float* x623 = (float*)myMalloc(50 * sizeof(float));
for(int x624=0; x624 < 50; x624++) {
float x625 = x610[x624];
float x626 = -1.0f * x625;
double x627 = (double)x626;
double x628 = exp(x627);
float x629 = (float)x628;
float x630 = x629 + 1.0f;
float x631 = 1.0f / x630;
x623[x624] = x631;

}
float* x635 = (float*)myMalloc(50 * sizeof(float));
for(int x636=0; x636 < 50; x636++) {
x635[x636] = 0.0f;

}
// dot WrappedArray(50, 50) - WrappedArray(50)
int32_t x641 = 0;
float* x642 = (float*)myMalloc(50 * sizeof(float));
for(int x643=0; x643 < 50; x643++) {
float x644 = 0.0f;
for(int x645=0; x645 < 50; x645++) {
int32_t x646 = x641;
float x647 = x56[x646];
float x648 = x344[x645];
float x649 = x647 * x648;
x644 += x649;
x641 += 1;

}
float x654 = x644;
x642[x643] = x654;

}
float* x658 = (float*)myMalloc(50 * sizeof(float));
for(int x659=0; x659 < 50; x659++) {
x658[x659] = 0.0f;

}
// dot WrappedArray(50, 26) - WrappedArray(26)
int32_t x664 = 0;
float* x665 = (float*)myMalloc(50 * sizeof(float));
for(int x666=0; x666 < 50; x666++) {
float x667 = 0.0f;
for(int x668=0; x668 < 26; x668++) {
int32_t x669 = x664;
float x670 = x63[x669];
float x671 = x349[x668];
float x672 = x670 * x671;
x667 += x672;
x664 += 1;

}
float x677 = x667;
x665[x666] = x677;

}
float* x681 = (float*)myMalloc(50 * sizeof(float));
for(int x682=0; x682 < 50; x682++) {
x681[x682] = 0.0f;

}
float* x686 = (float*)myMalloc(50 * sizeof(float));
for(int x687=0; x687 < 50; x687++) {
float x688 = x642[x687];
float x689 = x665[x687];
float x690 = x688 + x689;
x686[x687] = x690;

}
float* x694 = (float*)myMalloc(50 * sizeof(float));
for(int x695=0; x695 < 50; x695++) {
x694[x695] = 0.0f;

}
float* x699 = (float*)myMalloc(50 * sizeof(float));
for(int x700=0; x700 < 50; x700++) {
float x701 = x686[x700];
float x702 = x70[x700];
float x703 = x701 + x702;
x699[x700] = x703;

}
float* x707 = (float*)myMalloc(50 * sizeof(float));
for(int x708=0; x708 < 50; x708++) {
x707[x708] = 0.0f;

}
float* x712 = (float*)myMalloc(50 * sizeof(float));
for(int x713=0; x713 < 50; x713++) {
float x714 = x699[x713];
double x715 = (double)x714;
double x716 = tanh(x715);
float x717 = (float)x716;
x712[x713] = x717;

}
float* x721 = (float*)myMalloc(50 * sizeof(float));
for(int x722=0; x722 < 50; x722++) {
x721[x722] = 0.0f;

}
float* x726 = (float*)myMalloc(50 * sizeof(float));
for(int x727=0; x727 < 50; x727++) {
float x728 = x445[x727];
float x729 = x346[x727];
float x730 = x728 * x729;
x726[x727] = x730;

}
float* x734 = (float*)myMalloc(50 * sizeof(float));
for(int x735=0; x735 < 50; x735++) {
x734[x735] = 0.0f;

}
float* x739 = (float*)myMalloc(50 * sizeof(float));
for(int x740=0; x740 < 50; x740++) {
float x741 = x534[x740];
float x742 = x712[x740];
float x743 = x741 * x742;
x739[x740] = x743;

}
float* x747 = (float*)myMalloc(50 * sizeof(float));
for(int x748=0; x748 < 50; x748++) {
x747[x748] = 0.0f;

}
float* x752 = (float*)myMalloc(50 * sizeof(float));
for(int x753=0; x753 < 50; x753++) {
float x754 = x726[x753];
float x755 = x739[x753];
float x756 = x754 + x755;
x752[x753] = x756;

}
float* x760 = (float*)myMalloc(50 * sizeof(float));
for(int x761=0; x761 < 50; x761++) {
x760[x761] = 0.0f;

}
float* x765 = (float*)myMalloc(50 * sizeof(float));
for(int x766=0; x766 < 50; x766++) {
float x767 = x752[x766];
double x768 = (double)x767;
double x769 = tanh(x768);
float x770 = (float)x769;
x765[x766] = x770;

}
float* x774 = (float*)myMalloc(50 * sizeof(float));
for(int x775=0; x775 < 50; x775++) {
x774[x775] = 0.0f;

}
float* x779 = (float*)myMalloc(50 * sizeof(float));
for(int x780=0; x780 < 50; x780++) {
float x781 = x623[x780];
float x782 = x765[x780];
float x783 = x781 * x782;
x779[x780] = x783;

}
float* x787 = (float*)myMalloc(50 * sizeof(float));
for(int x788=0; x788 < 50; x788++) {
x787[x788] = 0.0f;

}
// dot WrappedArray(26, 50) - WrappedArray(50)
int32_t x793 = 0;
float* x794 = (float*)myMalloc(26 * sizeof(float));
for(int x795=0; x795 < 26; x795++) {
float x796 = 0.0f;
for(int x797=0; x797 < 50; x797++) {
int32_t x798 = x793;
float x799 = x94[x798];
float x800 = x779[x797];
float x801 = x799 * x800;
x796 += x801;
x793 += 1;

}
float x806 = x796;
x794[x795] = x806;

}
float* x810 = (float*)myMalloc(26 * sizeof(float));
for(int x811=0; x811 < 26; x811++) {
x810[x811] = 0.0f;

}
float* x815 = (float*)myMalloc(26 * sizeof(float));
for(int x816=0; x816 < 26; x816++) {
float x817 = x794[x816];
float x818 = x101[x816];
float x819 = x817 + x818;
x815[x816] = x819;

}
float* x823 = (float*)myMalloc(26 * sizeof(float));
for(int x824=0; x824 < 26; x824++) {
x823[x824] = 0.0f;

}
float* x828 = (float*)myMalloc(26 * sizeof(float));
for(int x829=0; x829 < 26; x829++) {
float x830 = x815[x829];
double x831 = (double)x830;
double x832 = exp(x831);
float x833 = (float)x832;
x828[x829] = x833;

}
float* x837 = (float*)myMalloc(26 * sizeof(float));
for(int x838=0; x838 < 26; x838++) {
x837[x838] = 0.0f;

}
float x842 = 0.0f;
for(int x843=0; x843 < 26; x843++) {
float x844 = x842;
float x845 = x828[x843];
float x846 = x844 + x845;
x842 = x846;

}
float x850 = x842;
float* x851 = (float*)myMalloc(1 * sizeof(float));
x851[0] = x850;
float* x853 = (float*)myMalloc(1 * sizeof(float));
for(int x854=0; x854 < 1; x854++) {
x853[x854] = 0.0f;

}
float x858 = x851[0];
float* x859 = (float*)myMalloc(26 * sizeof(float));
for(int x860=0; x860 < 26; x860++) {
float x861 = x828[x860];
float x862 = x861 / x858;
x859[x860] = x862;

}
float* x866 = (float*)myMalloc(26 * sizeof(float));
for(int x867=0; x867 < 26; x867++) {
x866[x867] = 0.0f;

}
// dot WrappedArray(26) - WrappedArray(26)
int32_t x872 = 0;
float* x873 = (float*)myMalloc(1 * sizeof(float));
float x874 = 0.0f;
for(int x875=0; x875 < 26; x875++) {
int32_t x876 = x872;
float x877 = x859[x876];
float x878 = x361[x875];
float x879 = x877 * x878;
x874 += x879;
x872 += 1;

}
float x884 = x874;
x873[0] = x884;
float* x886 = (float*)myMalloc(1 * sizeof(float));
for(int x887=0; x887 < 1; x887++) {
x886[x887] = 0.0f;

}
float* x891 = (float*)myMalloc(1 * sizeof(float));
float x892 = x873[0];
double x893 = (double)x892;
double x894 = log(x893);
float x895 = (float)x894;
x891[0] = x895;
float* x897 = (float*)myMalloc(1 * sizeof(float));
for(int x898=0; x898 < 1; x898++) {
x897[x898] = 0.0f;

}
float* x902 = (float*)myMalloc(1 * sizeof(float));
float x903 = x891[0];
float x904 = x342[0];
float x905 = x904 - x903;
x902[0] = x905;
float* x907 = (float*)myMalloc(1 * sizeof(float));
for(int x908=0; x908 < 1; x908++) {
x907[x908] = 0.0f;

}
float** x913 = (float**)myMalloc(6 * sizeof(float*));
x913[0] = x902;
x913[1] = x907;
x913[2] = x779;
x913[3] = x787;
x913[4] = x752;
x913[5] = x760;
int32_t x1012 = 0;
int32_t x1028 = 0;
int32_t x1168 = 0;
int32_t x1184 = 0;
int32_t x1201 = 0;
int32_t x1217 = 0;
int32_t x1275 = 0;
int32_t x1291 = 0;
int32_t x1308 = 0;
int32_t x1324 = 0;
int32_t x1382 = 0;
int32_t x1398 = 0;
int32_t x1415 = 0;
int32_t x1431 = 0;
int32_t x1489 = 0;
int32_t x1505 = 0;
int32_t x1522 = 0;
int32_t x1538 = 0;
int32_t x912 = x340 + 1;
x337(x912,x913);
// += tensor of dim 0
float x923 = x907[0];
float x924 = x343[0];
float x925 = x924 + x923;
x343[0] = x925;
float x927 = x907[0];
float x928 = x897[0];
float x929 = x928 - x927;
x897[0] = x929;
float x931 = x886[0];
float x932 = x897[0];
float x933 = x873[0];
float x934 = x932 / x933;
float x935 = x931 + x934;
x886[0] = x935;
float x937 = x886[0];
// Generate code for addMul
for(int x939=0; x939 < 26; x939++) {
float x940 = x866[x939];
float x941 = x361[x939];
float x942 = x937 * x941;
float x943 = x940 + x942;
x866[x939] = x943;

}
float x947 = x886[0];
// Generate code for addMul
for(int x949=0; x949 < 26; x949++) {
float x950 = x368[x949];
float x951 = x859[x949];
float x952 = x947 * x951;
float x953 = x950 + x952;
x368[x949] = x953;

}
for(int x957=0; x957 < 26; x957++) {
float x958 = x837[x957];
float x959 = x866[x957];
float x960 = x851[0];
float x961 = x959 / x960;
float x962 = x958 + x961;
x837[x957] = x962;

}
for(int x966=0; x966 < 26; x966++) {
float x967 = x853[0];
float x968 = x828[x966];
float x969 = x866[x966];
float x971 = x851[0];
float x970 = x968 * x969;
float x972 = x971 * x971;
float x973 = x970 / x972;
float x974 = x967 - x973;
x853[0] = x974;

}
// += tensor of dim 0
float x979 = x853[0];
for(int x980=0; x980 < 26; x980++) {
float x981 = x837[x980];
float x982 = x981 + x979;
x837[x980] = x982;

}
// backpropage exp
for(int x987=0; x987 < 26; x987++) {
float x988 = x823[x987];
float x989 = x828[x987];
float x990 = x837[x987];
float x991 = x989 * x990;
float x992 = x988 + x991;
x823[x987] = x992;

}
// backpropagate +
for(int x997=0; x997 < 26; x997++) {
float x998 = x810[x997];
float x999 = x823[x997];
float x1000 = x998 + x999;
x810[x997] = x1000;

}
for(int x1004=0; x1004 < 26; x1004++) {
float x1005 = x192[x1004];
float x1006 = x823[x1004];
float x1007 = x1005 + x1006;
x192[x1004] = x1007;

}
// add_cartesian
for(int x1013=0; x1013 < 26; x1013++) {
for(int x1014=0; x1014 < 50; x1014++) {
int32_t x1015 = x1012;
int32_t x1016 = x1015 + x1014;
float x1017 = x187[x1016];
float x1018 = x779[x1014];
float x1019 = x810[x1013];
float x1020 = x1018 * x1019;
float x1021 = x1017 + x1020;
x187[x1016] = x1021;

}
x1012 += 50;

}
for(int x1029=0; x1029 < 26; x1029++) {
for(int x1030=0; x1030 < 50; x1030++) {
float x1031 = x787[x1030];
int32_t x1032 = x1028;
int32_t x1033 = x1032 + x1030;
float x1034 = x94[x1033];
float x1035 = x810[x1029];
float x1036 = x1034 * x1035;
float x1037 = x1031 + x1036;
x787[x1030] = x1037;

}
x1028 += 50;

}
for(int x1044=0; x1044 < 50; x1044++) {
float x1045 = x635[x1044];
float x1046 = x765[x1044];
float x1047 = x787[x1044];
float x1048 = x1046 * x1047;
float x1049 = x1045 + x1048;
x635[x1044] = x1049;

}
for(int x1053=0; x1053 < 50; x1053++) {
float x1054 = x774[x1053];
float x1055 = x623[x1053];
float x1056 = x787[x1053];
float x1057 = x1055 * x1056;
float x1058 = x1054 + x1057;
x774[x1053] = x1058;

}
// backpropagate tanh
for(int x1063=0; x1063 < 50; x1063++) {
float x1064 = x760[x1063];
float x1065 = x765[x1063];
float x1068 = x774[x1063];
float x1066 = x1065 * x1065;
float x1067 = 1.0f - x1066;
float x1069 = x1067 * x1068;
float x1070 = x1064 + x1069;
x760[x1063] = x1070;

}
// backpropagate +
for(int x1075=0; x1075 < 50; x1075++) {
float x1076 = x734[x1075];
float x1077 = x760[x1075];
float x1078 = x1076 + x1077;
x734[x1075] = x1078;

}
for(int x1082=0; x1082 < 50; x1082++) {
float x1083 = x747[x1082];
float x1084 = x760[x1082];
float x1085 = x1083 + x1084;
x747[x1082] = x1085;

}
for(int x1089=0; x1089 < 50; x1089++) {
float x1090 = x546[x1089];
float x1091 = x712[x1089];
float x1092 = x747[x1089];
float x1093 = x1091 * x1092;
float x1094 = x1090 + x1093;
x546[x1089] = x1094;

}
for(int x1098=0; x1098 < 50; x1098++) {
float x1099 = x721[x1098];
float x1100 = x534[x1098];
float x1101 = x747[x1098];
float x1102 = x1100 * x1101;
float x1103 = x1099 + x1102;
x721[x1098] = x1103;

}
for(int x1107=0; x1107 < 50; x1107++) {
float x1108 = x457[x1107];
float x1109 = x346[x1107];
float x1110 = x734[x1107];
float x1111 = x1109 * x1110;
float x1112 = x1108 + x1111;
x457[x1107] = x1112;

}
for(int x1116=0; x1116 < 50; x1116++) {
float x1117 = x347[x1116];
float x1118 = x445[x1116];
float x1119 = x734[x1116];
float x1120 = x1118 * x1119;
float x1121 = x1117 + x1120;
x347[x1116] = x1121;

}
// backpropagate tanh
for(int x1126=0; x1126 < 50; x1126++) {
float x1127 = x707[x1126];
float x1128 = x712[x1126];
float x1131 = x721[x1126];
float x1129 = x1128 * x1128;
float x1130 = 1.0f - x1129;
float x1132 = x1130 * x1131;
float x1133 = x1127 + x1132;
x707[x1126] = x1133;

}
// backpropagate +
for(int x1138=0; x1138 < 50; x1138++) {
float x1139 = x694[x1138];
float x1140 = x707[x1138];
float x1141 = x1139 + x1140;
x694[x1138] = x1141;

}
for(int x1145=0; x1145 < 50; x1145++) {
float x1146 = x167[x1145];
float x1147 = x707[x1145];
float x1148 = x1146 + x1147;
x167[x1145] = x1148;

}
// backpropagate +
for(int x1153=0; x1153 < 50; x1153++) {
float x1154 = x658[x1153];
float x1155 = x694[x1153];
float x1156 = x1154 + x1155;
x658[x1153] = x1156;

}
for(int x1160=0; x1160 < 50; x1160++) {
float x1161 = x681[x1160];
float x1162 = x694[x1160];
float x1163 = x1161 + x1162;
x681[x1160] = x1163;

}
// add_cartesian
for(int x1169=0; x1169 < 50; x1169++) {
for(int x1170=0; x1170 < 26; x1170++) {
int32_t x1171 = x1168;
int32_t x1172 = x1171 + x1170;
float x1173 = x162[x1172];
float x1174 = x349[x1170];
float x1175 = x681[x1169];
float x1176 = x1174 * x1175;
float x1177 = x1173 + x1176;
x162[x1172] = x1177;

}
x1168 += 26;

}
for(int x1185=0; x1185 < 50; x1185++) {
for(int x1186=0; x1186 < 26; x1186++) {
float x1187 = x356[x1186];
int32_t x1188 = x1184;
int32_t x1189 = x1188 + x1186;
float x1190 = x63[x1189];
float x1191 = x681[x1185];
float x1192 = x1190 * x1191;
float x1193 = x1187 + x1192;
x356[x1186] = x1193;

}
x1184 += 26;

}
// add_cartesian
for(int x1202=0; x1202 < 50; x1202++) {
for(int x1203=0; x1203 < 50; x1203++) {
int32_t x1204 = x1201;
int32_t x1205 = x1204 + x1203;
float x1206 = x157[x1205];
float x1207 = x344[x1203];
float x1208 = x658[x1202];
float x1209 = x1207 * x1208;
float x1210 = x1206 + x1209;
x157[x1205] = x1210;

}
x1201 += 50;

}
for(int x1218=0; x1218 < 50; x1218++) {
for(int x1219=0; x1219 < 50; x1219++) {
float x1220 = x345[x1219];
int32_t x1221 = x1217;
int32_t x1222 = x1221 + x1219;
float x1223 = x56[x1222];
float x1224 = x658[x1218];
float x1225 = x1223 * x1224;
float x1226 = x1220 + x1225;
x345[x1219] = x1226;

}
x1217 += 50;

}
for(int x1233=0; x1233 < 50; x1233++) {
float x1234 = x618[x1233];
float x1235 = x623[x1233];
float x1238 = x635[x1233];
float x1236 = 1.0f - x1235;
float x1237 = x1236 * x1235;
float x1239 = x1237 * x1238;
float x1240 = x1234 + x1239;
x618[x1233] = x1240;

}
// backpropagate +
for(int x1245=0; x1245 < 50; x1245++) {
float x1246 = x605[x1245];
float x1247 = x618[x1245];
float x1248 = x1246 + x1247;
x605[x1245] = x1248;

}
for(int x1252=0; x1252 < 50; x1252++) {
float x1253 = x182[x1252];
float x1254 = x618[x1252];
float x1255 = x1253 + x1254;
x182[x1252] = x1255;

}
// backpropagate +
for(int x1260=0; x1260 < 50; x1260++) {
float x1261 = x569[x1260];
float x1262 = x605[x1260];
float x1263 = x1261 + x1262;
x569[x1260] = x1263;

}
for(int x1267=0; x1267 < 50; x1267++) {
float x1268 = x592[x1267];
float x1269 = x605[x1267];
float x1270 = x1268 + x1269;
x592[x1267] = x1270;

}
// add_cartesian
for(int x1276=0; x1276 < 50; x1276++) {
for(int x1277=0; x1277 < 26; x1277++) {
int32_t x1278 = x1275;
int32_t x1279 = x1278 + x1277;
float x1280 = x177[x1279];
float x1281 = x349[x1277];
float x1282 = x592[x1276];
float x1283 = x1281 * x1282;
float x1284 = x1280 + x1283;
x177[x1279] = x1284;

}
x1275 += 26;

}
for(int x1292=0; x1292 < 50; x1292++) {
for(int x1293=0; x1293 < 26; x1293++) {
float x1294 = x356[x1293];
int32_t x1295 = x1291;
int32_t x1296 = x1295 + x1293;
float x1297 = x82[x1296];
float x1298 = x592[x1292];
float x1299 = x1297 * x1298;
float x1300 = x1294 + x1299;
x356[x1293] = x1300;

}
x1291 += 26;

}
// add_cartesian
for(int x1309=0; x1309 < 50; x1309++) {
for(int x1310=0; x1310 < 50; x1310++) {
int32_t x1311 = x1308;
int32_t x1312 = x1311 + x1310;
float x1313 = x172[x1312];
float x1314 = x344[x1310];
float x1315 = x569[x1309];
float x1316 = x1314 * x1315;
float x1317 = x1313 + x1316;
x172[x1312] = x1317;

}
x1308 += 50;

}
for(int x1325=0; x1325 < 50; x1325++) {
for(int x1326=0; x1326 < 50; x1326++) {
float x1327 = x345[x1326];
int32_t x1328 = x1324;
int32_t x1329 = x1328 + x1326;
float x1330 = x75[x1329];
float x1331 = x569[x1325];
float x1332 = x1330 * x1331;
float x1333 = x1327 + x1332;
x345[x1326] = x1333;

}
x1324 += 50;

}
for(int x1340=0; x1340 < 50; x1340++) {
float x1341 = x529[x1340];
float x1342 = x534[x1340];
float x1345 = x546[x1340];
float x1343 = 1.0f - x1342;
float x1344 = x1343 * x1342;
float x1346 = x1344 * x1345;
float x1347 = x1341 + x1346;
x529[x1340] = x1347;

}
// backpropagate +
for(int x1352=0; x1352 < 50; x1352++) {
float x1353 = x516[x1352];
float x1354 = x529[x1352];
float x1355 = x1353 + x1354;
x516[x1352] = x1355;

}
for(int x1359=0; x1359 < 50; x1359++) {
float x1360 = x152[x1359];
float x1361 = x529[x1359];
float x1362 = x1360 + x1361;
x152[x1359] = x1362;

}
// backpropagate +
for(int x1367=0; x1367 < 50; x1367++) {
float x1368 = x480[x1367];
float x1369 = x516[x1367];
float x1370 = x1368 + x1369;
x480[x1367] = x1370;

}
for(int x1374=0; x1374 < 50; x1374++) {
float x1375 = x503[x1374];
float x1376 = x516[x1374];
float x1377 = x1375 + x1376;
x503[x1374] = x1377;

}
// add_cartesian
for(int x1383=0; x1383 < 50; x1383++) {
for(int x1384=0; x1384 < 26; x1384++) {
int32_t x1385 = x1382;
int32_t x1386 = x1385 + x1384;
float x1387 = x147[x1386];
float x1388 = x349[x1384];
float x1389 = x503[x1383];
float x1390 = x1388 * x1389;
float x1391 = x1387 + x1390;
x147[x1386] = x1391;

}
x1382 += 26;

}
for(int x1399=0; x1399 < 50; x1399++) {
for(int x1400=0; x1400 < 26; x1400++) {
float x1401 = x356[x1400];
int32_t x1402 = x1398;
int32_t x1403 = x1402 + x1400;
float x1404 = x44[x1403];
float x1405 = x503[x1399];
float x1406 = x1404 * x1405;
float x1407 = x1401 + x1406;
x356[x1400] = x1407;

}
x1398 += 26;

}
// add_cartesian
for(int x1416=0; x1416 < 50; x1416++) {
for(int x1417=0; x1417 < 50; x1417++) {
int32_t x1418 = x1415;
int32_t x1419 = x1418 + x1417;
float x1420 = x142[x1419];
float x1421 = x344[x1417];
float x1422 = x480[x1416];
float x1423 = x1421 * x1422;
float x1424 = x1420 + x1423;
x142[x1419] = x1424;

}
x1415 += 50;

}
for(int x1432=0; x1432 < 50; x1432++) {
for(int x1433=0; x1433 < 50; x1433++) {
float x1434 = x345[x1433];
int32_t x1435 = x1431;
int32_t x1436 = x1435 + x1433;
float x1437 = x37[x1436];
float x1438 = x480[x1432];
float x1439 = x1437 * x1438;
float x1440 = x1434 + x1439;
x345[x1433] = x1440;

}
x1431 += 50;

}
for(int x1447=0; x1447 < 50; x1447++) {
float x1448 = x440[x1447];
float x1449 = x445[x1447];
float x1452 = x457[x1447];
float x1450 = 1.0f - x1449;
float x1451 = x1450 * x1449;
float x1453 = x1451 * x1452;
float x1454 = x1448 + x1453;
x440[x1447] = x1454;

}
// backpropagate +
for(int x1459=0; x1459 < 50; x1459++) {
float x1460 = x427[x1459];
float x1461 = x440[x1459];
float x1462 = x1460 + x1461;
x427[x1459] = x1462;

}
for(int x1466=0; x1466 < 50; x1466++) {
float x1467 = x137[x1466];
float x1468 = x440[x1466];
float x1469 = x1467 + x1468;
x137[x1466] = x1469;

}
// backpropagate +
for(int x1474=0; x1474 < 50; x1474++) {
float x1475 = x391[x1474];
float x1476 = x427[x1474];
float x1477 = x1475 + x1476;
x391[x1474] = x1477;

}
for(int x1481=0; x1481 < 50; x1481++) {
float x1482 = x414[x1481];
float x1483 = x427[x1481];
float x1484 = x1482 + x1483;
x414[x1481] = x1484;

}
// add_cartesian
for(int x1490=0; x1490 < 50; x1490++) {
for(int x1491=0; x1491 < 26; x1491++) {
int32_t x1492 = x1489;
int32_t x1493 = x1492 + x1491;
float x1494 = x132[x1493];
float x1495 = x349[x1491];
float x1496 = x414[x1490];
float x1497 = x1495 * x1496;
float x1498 = x1494 + x1497;
x132[x1493] = x1498;

}
x1489 += 26;

}
for(int x1506=0; x1506 < 50; x1506++) {
for(int x1507=0; x1507 < 26; x1507++) {
float x1508 = x356[x1507];
int32_t x1509 = x1505;
int32_t x1510 = x1509 + x1507;
float x1511 = x23[x1510];
float x1512 = x414[x1506];
float x1513 = x1511 * x1512;
float x1514 = x1508 + x1513;
x356[x1507] = x1514;

}
x1505 += 26;

}
// add_cartesian
for(int x1523=0; x1523 < 50; x1523++) {
for(int x1524=0; x1524 < 50; x1524++) {
int32_t x1525 = x1522;
int32_t x1526 = x1525 + x1524;
float x1527 = x127[x1526];
float x1528 = x344[x1524];
float x1529 = x391[x1523];
float x1530 = x1528 * x1529;
float x1531 = x1527 + x1530;
x127[x1526] = x1531;

}
x1522 += 50;

}
for(int x1539=0; x1539 < 50; x1539++) {
for(int x1540=0; x1540 < 50; x1540++) {
float x1541 = x345[x1540];
int32_t x1542 = x1538;
int32_t x1543 = x1542 + x1540;
float x1544 = x15[x1543];
float x1545 = x391[x1539];
float x1546 = x1544 * x1545;
float x1547 = x1541 + x1546;
x345[x1540] = x1547;

}
x1538 += 50;

}
} else {
for(int x1555=0; x1555 < 50; x1555++) {
float x1556 = x344[x1555];
x117[x1555] = x1556;

}
for(int x1560=0; x1560 < 50; x1560++) {
float x1561 = x346[x1560];
x122[x1560] = x1561;

}
float x1565 = x343[0];
x343[0] = 1.0f;
float x1567 = x342[0];
x322[0] = x1567;
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
float x292 = x107[x291];
x107[x291] = 0.0f;

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
float* x311 = (float*)myMalloc(1 * sizeof(float));
for(int x313=0; x313 < 1; x313++) {
x311[x313] = 0.0f;

}
float* x317 = (float*)myMalloc(1 * sizeof(float));
for(int x318=0; x318 < 1; x318++) {
x317[x318] = 0.0f;

}
for(int x323=0; x323 < 1; x323++) {
x322[x323] = 0.0f;

}
float* x327 = (float*)myMalloc(1 * sizeof(float));
for(int x328=0; x328 < 1; x328++) {
x327[x328] = 0.0f;

}
float* x332 = (float*)myMalloc(1 * sizeof(float));
for(int x333=0; x333 < 1; x333++) {
x332[x333] = 0.0f;

}
float** x1572 = (float**)myMalloc(6 * sizeof(float*));
x1572[0] = x327;
x1572[1] = x332;
x1572[2] = x107;
x1572[3] = x197;
x1572[4] = x112;
x1572[5] = x202;
x337(0,x1572);
float x1581 = x322[0];
double x1582 = x282;
double x1583 = x1582 * 0.9;
double x1584 = (double)x1581;
double x1585 = x1584 * 0.1;
double x1586 = x1583 + x1585;
x282 = x1586;
int32_t x1588 = x284 % 100;
bool x1589 = x1588 == 0;
if (x1589) {
double x1590 = x282;
printf("iter %d, loss %f\n",x284,x1590);
int32_t x1592 = x284 / 100;
x278[x1592] = x1590;
} else {
}
for(int x1596=0; x1596 < 2500; x1596++) {
float x1597 = x127[x1596];
bool x1598 = x1597 > 5.0f;
if (x1598) {
x127[x1596] = 5.0f;
} else {
}
float x1602 = x127[x1596];
bool x1603 = x1602 < -5.0f;
if (x1603) {
x127[x1596] = -5.0f;
} else {
}

}
float* x1609 = (float*)myMalloc(2500 * sizeof(float));
for(int x1610=0; x1610 < 2500; x1610++) {
float x1611 = x127[x1610];
float x1612 = x127[x1610];
float x1613 = x1611 * x1612;
x1609[x1610] = x1613;

}
for(int x1617=0; x1617 < 2500; x1617++) {
float x1618 = x207[x1617];
float x1619 = x1609[x1617];
float x1620 = x1618 + x1619;
x207[x1617] = x1620;

}
float* x1624 = (float*)myMalloc(2500 * sizeof(float));
for(int x1625=0; x1625 < 2500; x1625++) {
float x1626 = x127[x1625];
float x1627 = x1626 * 0.1f;
x1624[x1625] = x1627;

}
float* x1631 = (float*)myMalloc(2500 * sizeof(float));
for(int x1632=0; x1632 < 2500; x1632++) {
float x1633 = x207[x1632];
float x1634 = x1633 + 1.0E-8f;
x1631[x1632] = x1634;

}
float* x1638 = (float*)myMalloc(2500 * sizeof(float));
for(int x1639=0; x1639 < 2500; x1639++) {
float x1640 = x1631[x1639];
double x1641 = (double)x1640;
double x1642 = sqrt(x1641);
float x1643 = (float)x1642;
x1638[x1639] = x1643;

}
float* x1647 = (float*)myMalloc(2500 * sizeof(float));
for(int x1648=0; x1648 < 2500; x1648++) {
float x1649 = x1624[x1648];
float x1650 = x1638[x1648];
float x1651 = x1649 / x1650;
x1647[x1648] = x1651;

}
for(int x1655=0; x1655 < 2500; x1655++) {
float x1656 = x15[x1655];
float x1657 = x1647[x1655];
float x1658 = x1656 - x1657;
x15[x1655] = x1658;

}
for(int x1662=0; x1662 < 2500; x1662++) {
float x1663 = x127[x1662];
x127[x1662] = 0.0f;

}
for(int x1667=0; x1667 < 1300; x1667++) {
float x1668 = x132[x1667];
bool x1669 = x1668 > 5.0f;
if (x1669) {
x132[x1667] = 5.0f;
} else {
}
float x1673 = x132[x1667];
bool x1674 = x1673 < -5.0f;
if (x1674) {
x132[x1667] = -5.0f;
} else {
}

}
float* x1680 = (float*)myMalloc(1300 * sizeof(float));
for(int x1681=0; x1681 < 1300; x1681++) {
float x1682 = x132[x1681];
float x1683 = x132[x1681];
float x1684 = x1682 * x1683;
x1680[x1681] = x1684;

}
for(int x1688=0; x1688 < 1300; x1688++) {
float x1689 = x212[x1688];
float x1690 = x1680[x1688];
float x1691 = x1689 + x1690;
x212[x1688] = x1691;

}
float* x1695 = (float*)myMalloc(1300 * sizeof(float));
for(int x1696=0; x1696 < 1300; x1696++) {
float x1697 = x132[x1696];
float x1698 = x1697 * 0.1f;
x1695[x1696] = x1698;

}
float* x1702 = (float*)myMalloc(1300 * sizeof(float));
for(int x1703=0; x1703 < 1300; x1703++) {
float x1704 = x212[x1703];
float x1705 = x1704 + 1.0E-8f;
x1702[x1703] = x1705;

}
float* x1709 = (float*)myMalloc(1300 * sizeof(float));
for(int x1710=0; x1710 < 1300; x1710++) {
float x1711 = x1702[x1710];
double x1712 = (double)x1711;
double x1713 = sqrt(x1712);
float x1714 = (float)x1713;
x1709[x1710] = x1714;

}
float* x1718 = (float*)myMalloc(1300 * sizeof(float));
for(int x1719=0; x1719 < 1300; x1719++) {
float x1720 = x1695[x1719];
float x1721 = x1709[x1719];
float x1722 = x1720 / x1721;
x1718[x1719] = x1722;

}
for(int x1726=0; x1726 < 1300; x1726++) {
float x1727 = x23[x1726];
float x1728 = x1718[x1726];
float x1729 = x1727 - x1728;
x23[x1726] = x1729;

}
for(int x1733=0; x1733 < 1300; x1733++) {
float x1734 = x132[x1733];
x132[x1733] = 0.0f;

}
for(int x1738=0; x1738 < 50; x1738++) {
float x1739 = x137[x1738];
bool x1740 = x1739 > 5.0f;
if (x1740) {
x137[x1738] = 5.0f;
} else {
}
float x1744 = x137[x1738];
bool x1745 = x1744 < -5.0f;
if (x1745) {
x137[x1738] = -5.0f;
} else {
}

}
float* x1751 = (float*)myMalloc(50 * sizeof(float));
for(int x1752=0; x1752 < 50; x1752++) {
float x1753 = x137[x1752];
float x1754 = x137[x1752];
float x1755 = x1753 * x1754;
x1751[x1752] = x1755;

}
for(int x1759=0; x1759 < 50; x1759++) {
float x1760 = x217[x1759];
float x1761 = x1751[x1759];
float x1762 = x1760 + x1761;
x217[x1759] = x1762;

}
float* x1766 = (float*)myMalloc(50 * sizeof(float));
for(int x1767=0; x1767 < 50; x1767++) {
float x1768 = x137[x1767];
float x1769 = x1768 * 0.1f;
x1766[x1767] = x1769;

}
float* x1773 = (float*)myMalloc(50 * sizeof(float));
for(int x1774=0; x1774 < 50; x1774++) {
float x1775 = x217[x1774];
float x1776 = x1775 + 1.0E-8f;
x1773[x1774] = x1776;

}
float* x1780 = (float*)myMalloc(50 * sizeof(float));
for(int x1781=0; x1781 < 50; x1781++) {
float x1782 = x1773[x1781];
double x1783 = (double)x1782;
double x1784 = sqrt(x1783);
float x1785 = (float)x1784;
x1780[x1781] = x1785;

}
float* x1789 = (float*)myMalloc(50 * sizeof(float));
for(int x1790=0; x1790 < 50; x1790++) {
float x1791 = x1766[x1790];
float x1792 = x1780[x1790];
float x1793 = x1791 / x1792;
x1789[x1790] = x1793;

}
for(int x1797=0; x1797 < 50; x1797++) {
float x1798 = x31[x1797];
float x1799 = x1789[x1797];
float x1800 = x1798 - x1799;
x31[x1797] = x1800;

}
for(int x1804=0; x1804 < 50; x1804++) {
float x1805 = x137[x1804];
x137[x1804] = 0.0f;

}
for(int x1809=0; x1809 < 2500; x1809++) {
float x1810 = x142[x1809];
bool x1811 = x1810 > 5.0f;
if (x1811) {
x142[x1809] = 5.0f;
} else {
}
float x1815 = x142[x1809];
bool x1816 = x1815 < -5.0f;
if (x1816) {
x142[x1809] = -5.0f;
} else {
}

}
float* x1822 = (float*)myMalloc(2500 * sizeof(float));
for(int x1823=0; x1823 < 2500; x1823++) {
float x1824 = x142[x1823];
float x1825 = x142[x1823];
float x1826 = x1824 * x1825;
x1822[x1823] = x1826;

}
for(int x1830=0; x1830 < 2500; x1830++) {
float x1831 = x222[x1830];
float x1832 = x1822[x1830];
float x1833 = x1831 + x1832;
x222[x1830] = x1833;

}
float* x1837 = (float*)myMalloc(2500 * sizeof(float));
for(int x1838=0; x1838 < 2500; x1838++) {
float x1839 = x142[x1838];
float x1840 = x1839 * 0.1f;
x1837[x1838] = x1840;

}
float* x1844 = (float*)myMalloc(2500 * sizeof(float));
for(int x1845=0; x1845 < 2500; x1845++) {
float x1846 = x222[x1845];
float x1847 = x1846 + 1.0E-8f;
x1844[x1845] = x1847;

}
float* x1851 = (float*)myMalloc(2500 * sizeof(float));
for(int x1852=0; x1852 < 2500; x1852++) {
float x1853 = x1844[x1852];
double x1854 = (double)x1853;
double x1855 = sqrt(x1854);
float x1856 = (float)x1855;
x1851[x1852] = x1856;

}
float* x1860 = (float*)myMalloc(2500 * sizeof(float));
for(int x1861=0; x1861 < 2500; x1861++) {
float x1862 = x1837[x1861];
float x1863 = x1851[x1861];
float x1864 = x1862 / x1863;
x1860[x1861] = x1864;

}
for(int x1868=0; x1868 < 2500; x1868++) {
float x1869 = x37[x1868];
float x1870 = x1860[x1868];
float x1871 = x1869 - x1870;
x37[x1868] = x1871;

}
for(int x1875=0; x1875 < 2500; x1875++) {
float x1876 = x142[x1875];
x142[x1875] = 0.0f;

}
for(int x1880=0; x1880 < 1300; x1880++) {
float x1881 = x147[x1880];
bool x1882 = x1881 > 5.0f;
if (x1882) {
x147[x1880] = 5.0f;
} else {
}
float x1886 = x147[x1880];
bool x1887 = x1886 < -5.0f;
if (x1887) {
x147[x1880] = -5.0f;
} else {
}

}
float* x1893 = (float*)myMalloc(1300 * sizeof(float));
for(int x1894=0; x1894 < 1300; x1894++) {
float x1895 = x147[x1894];
float x1896 = x147[x1894];
float x1897 = x1895 * x1896;
x1893[x1894] = x1897;

}
for(int x1901=0; x1901 < 1300; x1901++) {
float x1902 = x227[x1901];
float x1903 = x1893[x1901];
float x1904 = x1902 + x1903;
x227[x1901] = x1904;

}
float* x1908 = (float*)myMalloc(1300 * sizeof(float));
for(int x1909=0; x1909 < 1300; x1909++) {
float x1910 = x147[x1909];
float x1911 = x1910 * 0.1f;
x1908[x1909] = x1911;

}
float* x1915 = (float*)myMalloc(1300 * sizeof(float));
for(int x1916=0; x1916 < 1300; x1916++) {
float x1917 = x227[x1916];
float x1918 = x1917 + 1.0E-8f;
x1915[x1916] = x1918;

}
float* x1922 = (float*)myMalloc(1300 * sizeof(float));
for(int x1923=0; x1923 < 1300; x1923++) {
float x1924 = x1915[x1923];
double x1925 = (double)x1924;
double x1926 = sqrt(x1925);
float x1927 = (float)x1926;
x1922[x1923] = x1927;

}
float* x1931 = (float*)myMalloc(1300 * sizeof(float));
for(int x1932=0; x1932 < 1300; x1932++) {
float x1933 = x1908[x1932];
float x1934 = x1922[x1932];
float x1935 = x1933 / x1934;
x1931[x1932] = x1935;

}
for(int x1939=0; x1939 < 1300; x1939++) {
float x1940 = x44[x1939];
float x1941 = x1931[x1939];
float x1942 = x1940 - x1941;
x44[x1939] = x1942;

}
for(int x1946=0; x1946 < 1300; x1946++) {
float x1947 = x147[x1946];
x147[x1946] = 0.0f;

}
for(int x1951=0; x1951 < 50; x1951++) {
float x1952 = x152[x1951];
bool x1953 = x1952 > 5.0f;
if (x1953) {
x152[x1951] = 5.0f;
} else {
}
float x1957 = x152[x1951];
bool x1958 = x1957 < -5.0f;
if (x1958) {
x152[x1951] = -5.0f;
} else {
}

}
float* x1964 = (float*)myMalloc(50 * sizeof(float));
for(int x1965=0; x1965 < 50; x1965++) {
float x1966 = x152[x1965];
float x1967 = x152[x1965];
float x1968 = x1966 * x1967;
x1964[x1965] = x1968;

}
for(int x1972=0; x1972 < 50; x1972++) {
float x1973 = x232[x1972];
float x1974 = x1964[x1972];
float x1975 = x1973 + x1974;
x232[x1972] = x1975;

}
float* x1979 = (float*)myMalloc(50 * sizeof(float));
for(int x1980=0; x1980 < 50; x1980++) {
float x1981 = x152[x1980];
float x1982 = x1981 * 0.1f;
x1979[x1980] = x1982;

}
float* x1986 = (float*)myMalloc(50 * sizeof(float));
for(int x1987=0; x1987 < 50; x1987++) {
float x1988 = x232[x1987];
float x1989 = x1988 + 1.0E-8f;
x1986[x1987] = x1989;

}
float* x1993 = (float*)myMalloc(50 * sizeof(float));
for(int x1994=0; x1994 < 50; x1994++) {
float x1995 = x1986[x1994];
double x1996 = (double)x1995;
double x1997 = sqrt(x1996);
float x1998 = (float)x1997;
x1993[x1994] = x1998;

}
float* x2002 = (float*)myMalloc(50 * sizeof(float));
for(int x2003=0; x2003 < 50; x2003++) {
float x2004 = x1979[x2003];
float x2005 = x1993[x2003];
float x2006 = x2004 / x2005;
x2002[x2003] = x2006;

}
for(int x2010=0; x2010 < 50; x2010++) {
float x2011 = x51[x2010];
float x2012 = x2002[x2010];
float x2013 = x2011 - x2012;
x51[x2010] = x2013;

}
for(int x2017=0; x2017 < 50; x2017++) {
float x2018 = x152[x2017];
x152[x2017] = 0.0f;

}
for(int x2022=0; x2022 < 2500; x2022++) {
float x2023 = x157[x2022];
bool x2024 = x2023 > 5.0f;
if (x2024) {
x157[x2022] = 5.0f;
} else {
}
float x2028 = x157[x2022];
bool x2029 = x2028 < -5.0f;
if (x2029) {
x157[x2022] = -5.0f;
} else {
}

}
float* x2035 = (float*)myMalloc(2500 * sizeof(float));
for(int x2036=0; x2036 < 2500; x2036++) {
float x2037 = x157[x2036];
float x2038 = x157[x2036];
float x2039 = x2037 * x2038;
x2035[x2036] = x2039;

}
for(int x2043=0; x2043 < 2500; x2043++) {
float x2044 = x237[x2043];
float x2045 = x2035[x2043];
float x2046 = x2044 + x2045;
x237[x2043] = x2046;

}
float* x2050 = (float*)myMalloc(2500 * sizeof(float));
for(int x2051=0; x2051 < 2500; x2051++) {
float x2052 = x157[x2051];
float x2053 = x2052 * 0.1f;
x2050[x2051] = x2053;

}
float* x2057 = (float*)myMalloc(2500 * sizeof(float));
for(int x2058=0; x2058 < 2500; x2058++) {
float x2059 = x237[x2058];
float x2060 = x2059 + 1.0E-8f;
x2057[x2058] = x2060;

}
float* x2064 = (float*)myMalloc(2500 * sizeof(float));
for(int x2065=0; x2065 < 2500; x2065++) {
float x2066 = x2057[x2065];
double x2067 = (double)x2066;
double x2068 = sqrt(x2067);
float x2069 = (float)x2068;
x2064[x2065] = x2069;

}
float* x2073 = (float*)myMalloc(2500 * sizeof(float));
for(int x2074=0; x2074 < 2500; x2074++) {
float x2075 = x2050[x2074];
float x2076 = x2064[x2074];
float x2077 = x2075 / x2076;
x2073[x2074] = x2077;

}
for(int x2081=0; x2081 < 2500; x2081++) {
float x2082 = x56[x2081];
float x2083 = x2073[x2081];
float x2084 = x2082 - x2083;
x56[x2081] = x2084;

}
for(int x2088=0; x2088 < 2500; x2088++) {
float x2089 = x157[x2088];
x157[x2088] = 0.0f;

}
for(int x2093=0; x2093 < 1300; x2093++) {
float x2094 = x162[x2093];
bool x2095 = x2094 > 5.0f;
if (x2095) {
x162[x2093] = 5.0f;
} else {
}
float x2099 = x162[x2093];
bool x2100 = x2099 < -5.0f;
if (x2100) {
x162[x2093] = -5.0f;
} else {
}

}
float* x2106 = (float*)myMalloc(1300 * sizeof(float));
for(int x2107=0; x2107 < 1300; x2107++) {
float x2108 = x162[x2107];
float x2109 = x162[x2107];
float x2110 = x2108 * x2109;
x2106[x2107] = x2110;

}
for(int x2114=0; x2114 < 1300; x2114++) {
float x2115 = x242[x2114];
float x2116 = x2106[x2114];
float x2117 = x2115 + x2116;
x242[x2114] = x2117;

}
float* x2121 = (float*)myMalloc(1300 * sizeof(float));
for(int x2122=0; x2122 < 1300; x2122++) {
float x2123 = x162[x2122];
float x2124 = x2123 * 0.1f;
x2121[x2122] = x2124;

}
float* x2128 = (float*)myMalloc(1300 * sizeof(float));
for(int x2129=0; x2129 < 1300; x2129++) {
float x2130 = x242[x2129];
float x2131 = x2130 + 1.0E-8f;
x2128[x2129] = x2131;

}
float* x2135 = (float*)myMalloc(1300 * sizeof(float));
for(int x2136=0; x2136 < 1300; x2136++) {
float x2137 = x2128[x2136];
double x2138 = (double)x2137;
double x2139 = sqrt(x2138);
float x2140 = (float)x2139;
x2135[x2136] = x2140;

}
float* x2144 = (float*)myMalloc(1300 * sizeof(float));
for(int x2145=0; x2145 < 1300; x2145++) {
float x2146 = x2121[x2145];
float x2147 = x2135[x2145];
float x2148 = x2146 / x2147;
x2144[x2145] = x2148;

}
for(int x2152=0; x2152 < 1300; x2152++) {
float x2153 = x63[x2152];
float x2154 = x2144[x2152];
float x2155 = x2153 - x2154;
x63[x2152] = x2155;

}
for(int x2159=0; x2159 < 1300; x2159++) {
float x2160 = x162[x2159];
x162[x2159] = 0.0f;

}
for(int x2164=0; x2164 < 50; x2164++) {
float x2165 = x167[x2164];
bool x2166 = x2165 > 5.0f;
if (x2166) {
x167[x2164] = 5.0f;
} else {
}
float x2170 = x167[x2164];
bool x2171 = x2170 < -5.0f;
if (x2171) {
x167[x2164] = -5.0f;
} else {
}

}
float* x2177 = (float*)myMalloc(50 * sizeof(float));
for(int x2178=0; x2178 < 50; x2178++) {
float x2179 = x167[x2178];
float x2180 = x167[x2178];
float x2181 = x2179 * x2180;
x2177[x2178] = x2181;

}
for(int x2185=0; x2185 < 50; x2185++) {
float x2186 = x247[x2185];
float x2187 = x2177[x2185];
float x2188 = x2186 + x2187;
x247[x2185] = x2188;

}
float* x2192 = (float*)myMalloc(50 * sizeof(float));
for(int x2193=0; x2193 < 50; x2193++) {
float x2194 = x167[x2193];
float x2195 = x2194 * 0.1f;
x2192[x2193] = x2195;

}
float* x2199 = (float*)myMalloc(50 * sizeof(float));
for(int x2200=0; x2200 < 50; x2200++) {
float x2201 = x247[x2200];
float x2202 = x2201 + 1.0E-8f;
x2199[x2200] = x2202;

}
float* x2206 = (float*)myMalloc(50 * sizeof(float));
for(int x2207=0; x2207 < 50; x2207++) {
float x2208 = x2199[x2207];
double x2209 = (double)x2208;
double x2210 = sqrt(x2209);
float x2211 = (float)x2210;
x2206[x2207] = x2211;

}
float* x2215 = (float*)myMalloc(50 * sizeof(float));
for(int x2216=0; x2216 < 50; x2216++) {
float x2217 = x2192[x2216];
float x2218 = x2206[x2216];
float x2219 = x2217 / x2218;
x2215[x2216] = x2219;

}
for(int x2223=0; x2223 < 50; x2223++) {
float x2224 = x70[x2223];
float x2225 = x2215[x2223];
float x2226 = x2224 - x2225;
x70[x2223] = x2226;

}
for(int x2230=0; x2230 < 50; x2230++) {
float x2231 = x167[x2230];
x167[x2230] = 0.0f;

}
for(int x2235=0; x2235 < 2500; x2235++) {
float x2236 = x172[x2235];
bool x2237 = x2236 > 5.0f;
if (x2237) {
x172[x2235] = 5.0f;
} else {
}
float x2241 = x172[x2235];
bool x2242 = x2241 < -5.0f;
if (x2242) {
x172[x2235] = -5.0f;
} else {
}

}
float* x2248 = (float*)myMalloc(2500 * sizeof(float));
for(int x2249=0; x2249 < 2500; x2249++) {
float x2250 = x172[x2249];
float x2251 = x172[x2249];
float x2252 = x2250 * x2251;
x2248[x2249] = x2252;

}
for(int x2256=0; x2256 < 2500; x2256++) {
float x2257 = x252[x2256];
float x2258 = x2248[x2256];
float x2259 = x2257 + x2258;
x252[x2256] = x2259;

}
float* x2263 = (float*)myMalloc(2500 * sizeof(float));
for(int x2264=0; x2264 < 2500; x2264++) {
float x2265 = x172[x2264];
float x2266 = x2265 * 0.1f;
x2263[x2264] = x2266;

}
float* x2270 = (float*)myMalloc(2500 * sizeof(float));
for(int x2271=0; x2271 < 2500; x2271++) {
float x2272 = x252[x2271];
float x2273 = x2272 + 1.0E-8f;
x2270[x2271] = x2273;

}
float* x2277 = (float*)myMalloc(2500 * sizeof(float));
for(int x2278=0; x2278 < 2500; x2278++) {
float x2279 = x2270[x2278];
double x2280 = (double)x2279;
double x2281 = sqrt(x2280);
float x2282 = (float)x2281;
x2277[x2278] = x2282;

}
float* x2286 = (float*)myMalloc(2500 * sizeof(float));
for(int x2287=0; x2287 < 2500; x2287++) {
float x2288 = x2263[x2287];
float x2289 = x2277[x2287];
float x2290 = x2288 / x2289;
x2286[x2287] = x2290;

}
for(int x2294=0; x2294 < 2500; x2294++) {
float x2295 = x75[x2294];
float x2296 = x2286[x2294];
float x2297 = x2295 - x2296;
x75[x2294] = x2297;

}
for(int x2301=0; x2301 < 2500; x2301++) {
float x2302 = x172[x2301];
x172[x2301] = 0.0f;

}
for(int x2306=0; x2306 < 1300; x2306++) {
float x2307 = x177[x2306];
bool x2308 = x2307 > 5.0f;
if (x2308) {
x177[x2306] = 5.0f;
} else {
}
float x2312 = x177[x2306];
bool x2313 = x2312 < -5.0f;
if (x2313) {
x177[x2306] = -5.0f;
} else {
}

}
float* x2319 = (float*)myMalloc(1300 * sizeof(float));
for(int x2320=0; x2320 < 1300; x2320++) {
float x2321 = x177[x2320];
float x2322 = x177[x2320];
float x2323 = x2321 * x2322;
x2319[x2320] = x2323;

}
for(int x2327=0; x2327 < 1300; x2327++) {
float x2328 = x257[x2327];
float x2329 = x2319[x2327];
float x2330 = x2328 + x2329;
x257[x2327] = x2330;

}
float* x2334 = (float*)myMalloc(1300 * sizeof(float));
for(int x2335=0; x2335 < 1300; x2335++) {
float x2336 = x177[x2335];
float x2337 = x2336 * 0.1f;
x2334[x2335] = x2337;

}
float* x2341 = (float*)myMalloc(1300 * sizeof(float));
for(int x2342=0; x2342 < 1300; x2342++) {
float x2343 = x257[x2342];
float x2344 = x2343 + 1.0E-8f;
x2341[x2342] = x2344;

}
float* x2348 = (float*)myMalloc(1300 * sizeof(float));
for(int x2349=0; x2349 < 1300; x2349++) {
float x2350 = x2341[x2349];
double x2351 = (double)x2350;
double x2352 = sqrt(x2351);
float x2353 = (float)x2352;
x2348[x2349] = x2353;

}
float* x2357 = (float*)myMalloc(1300 * sizeof(float));
for(int x2358=0; x2358 < 1300; x2358++) {
float x2359 = x2334[x2358];
float x2360 = x2348[x2358];
float x2361 = x2359 / x2360;
x2357[x2358] = x2361;

}
for(int x2365=0; x2365 < 1300; x2365++) {
float x2366 = x82[x2365];
float x2367 = x2357[x2365];
float x2368 = x2366 - x2367;
x82[x2365] = x2368;

}
for(int x2372=0; x2372 < 1300; x2372++) {
float x2373 = x177[x2372];
x177[x2372] = 0.0f;

}
for(int x2377=0; x2377 < 50; x2377++) {
float x2378 = x182[x2377];
bool x2379 = x2378 > 5.0f;
if (x2379) {
x182[x2377] = 5.0f;
} else {
}
float x2383 = x182[x2377];
bool x2384 = x2383 < -5.0f;
if (x2384) {
x182[x2377] = -5.0f;
} else {
}

}
float* x2390 = (float*)myMalloc(50 * sizeof(float));
for(int x2391=0; x2391 < 50; x2391++) {
float x2392 = x182[x2391];
float x2393 = x182[x2391];
float x2394 = x2392 * x2393;
x2390[x2391] = x2394;

}
for(int x2398=0; x2398 < 50; x2398++) {
float x2399 = x262[x2398];
float x2400 = x2390[x2398];
float x2401 = x2399 + x2400;
x262[x2398] = x2401;

}
float* x2405 = (float*)myMalloc(50 * sizeof(float));
for(int x2406=0; x2406 < 50; x2406++) {
float x2407 = x182[x2406];
float x2408 = x2407 * 0.1f;
x2405[x2406] = x2408;

}
float* x2412 = (float*)myMalloc(50 * sizeof(float));
for(int x2413=0; x2413 < 50; x2413++) {
float x2414 = x262[x2413];
float x2415 = x2414 + 1.0E-8f;
x2412[x2413] = x2415;

}
float* x2419 = (float*)myMalloc(50 * sizeof(float));
for(int x2420=0; x2420 < 50; x2420++) {
float x2421 = x2412[x2420];
double x2422 = (double)x2421;
double x2423 = sqrt(x2422);
float x2424 = (float)x2423;
x2419[x2420] = x2424;

}
float* x2428 = (float*)myMalloc(50 * sizeof(float));
for(int x2429=0; x2429 < 50; x2429++) {
float x2430 = x2405[x2429];
float x2431 = x2419[x2429];
float x2432 = x2430 / x2431;
x2428[x2429] = x2432;

}
for(int x2436=0; x2436 < 50; x2436++) {
float x2437 = x89[x2436];
float x2438 = x2428[x2436];
float x2439 = x2437 - x2438;
x89[x2436] = x2439;

}
for(int x2443=0; x2443 < 50; x2443++) {
float x2444 = x182[x2443];
x182[x2443] = 0.0f;

}
for(int x2448=0; x2448 < 1300; x2448++) {
float x2449 = x187[x2448];
bool x2450 = x2449 > 5.0f;
if (x2450) {
x187[x2448] = 5.0f;
} else {
}
float x2454 = x187[x2448];
bool x2455 = x2454 < -5.0f;
if (x2455) {
x187[x2448] = -5.0f;
} else {
}

}
float* x2461 = (float*)myMalloc(1300 * sizeof(float));
for(int x2462=0; x2462 < 1300; x2462++) {
float x2463 = x187[x2462];
float x2464 = x187[x2462];
float x2465 = x2463 * x2464;
x2461[x2462] = x2465;

}
for(int x2469=0; x2469 < 1300; x2469++) {
float x2470 = x267[x2469];
float x2471 = x2461[x2469];
float x2472 = x2470 + x2471;
x267[x2469] = x2472;

}
float* x2476 = (float*)myMalloc(1300 * sizeof(float));
for(int x2477=0; x2477 < 1300; x2477++) {
float x2478 = x187[x2477];
float x2479 = x2478 * 0.1f;
x2476[x2477] = x2479;

}
float* x2483 = (float*)myMalloc(1300 * sizeof(float));
for(int x2484=0; x2484 < 1300; x2484++) {
float x2485 = x267[x2484];
float x2486 = x2485 + 1.0E-8f;
x2483[x2484] = x2486;

}
float* x2490 = (float*)myMalloc(1300 * sizeof(float));
for(int x2491=0; x2491 < 1300; x2491++) {
float x2492 = x2483[x2491];
double x2493 = (double)x2492;
double x2494 = sqrt(x2493);
float x2495 = (float)x2494;
x2490[x2491] = x2495;

}
float* x2499 = (float*)myMalloc(1300 * sizeof(float));
for(int x2500=0; x2500 < 1300; x2500++) {
float x2501 = x2476[x2500];
float x2502 = x2490[x2500];
float x2503 = x2501 / x2502;
x2499[x2500] = x2503;

}
for(int x2507=0; x2507 < 1300; x2507++) {
float x2508 = x94[x2507];
float x2509 = x2499[x2507];
float x2510 = x2508 - x2509;
x94[x2507] = x2510;

}
for(int x2514=0; x2514 < 1300; x2514++) {
float x2515 = x187[x2514];
x187[x2514] = 0.0f;

}
for(int x2519=0; x2519 < 26; x2519++) {
float x2520 = x192[x2519];
bool x2521 = x2520 > 5.0f;
if (x2521) {
x192[x2519] = 5.0f;
} else {
}
float x2525 = x192[x2519];
bool x2526 = x2525 < -5.0f;
if (x2526) {
x192[x2519] = -5.0f;
} else {
}

}
float* x2532 = (float*)myMalloc(26 * sizeof(float));
for(int x2533=0; x2533 < 26; x2533++) {
float x2534 = x192[x2533];
float x2535 = x192[x2533];
float x2536 = x2534 * x2535;
x2532[x2533] = x2536;

}
for(int x2540=0; x2540 < 26; x2540++) {
float x2541 = x272[x2540];
float x2542 = x2532[x2540];
float x2543 = x2541 + x2542;
x272[x2540] = x2543;

}
float* x2547 = (float*)myMalloc(26 * sizeof(float));
for(int x2548=0; x2548 < 26; x2548++) {
float x2549 = x192[x2548];
float x2550 = x2549 * 0.1f;
x2547[x2548] = x2550;

}
float* x2554 = (float*)myMalloc(26 * sizeof(float));
for(int x2555=0; x2555 < 26; x2555++) {
float x2556 = x272[x2555];
float x2557 = x2556 + 1.0E-8f;
x2554[x2555] = x2557;

}
float* x2561 = (float*)myMalloc(26 * sizeof(float));
for(int x2562=0; x2562 < 26; x2562++) {
float x2563 = x2554[x2562];
double x2564 = (double)x2563;
double x2565 = sqrt(x2564);
float x2566 = (float)x2565;
x2561[x2562] = x2566;

}
float* x2570 = (float*)myMalloc(26 * sizeof(float));
for(int x2571=0; x2571 < 26; x2571++) {
float x2572 = x2547[x2571];
float x2573 = x2561[x2571];
float x2574 = x2572 / x2573;
x2570[x2571] = x2574;

}
for(int x2578=0; x2578 < 26; x2578++) {
float x2579 = x101[x2578];
float x2580 = x2570[x2578];
float x2581 = x2579 - x2580;
x101[x2578] = x2581;

}
for(int x2585=0; x2585 < 26; x2585++) {
float x2586 = x192[x2585];
x192[x2585] = 0.0f;

}
for(int x2590=0; x2590 < 50; x2590++) {
float x2591 = x197[x2590];
x197[x2590] = 0.0f;

}
for(int x2595=0; x2595 < 50; x2595++) {
float x2596 = x202[x2595];
x202[x2595] = 0.0f;

}
for(int x2600=0; x2600 < 50; x2600++) {
float x2601 = x117[x2600];
x107[x2600] = x2601;

}
for(int x2605=0; x2605 < 50; x2605++) {
float x2606 = x122[x2605];
x112[x2605] = x2606;

}
mallocAddr = (void*)x279;

}
double x2613 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x2616 = (long)fopen(x0, "w");
fprintf((FILE *)x2616, "unit: %s\n", "100 iteration");
for(int x2619=0; x2619 < 51; x2619++) {
double x2620 = x278[x2619];
fprintf((FILE *)x2616, "%lf\n", x2620);

}
double x2614 = x277 - x1;
double x2615 = x2613 - x277;
fprintf((FILE *)x2616, "run time: %lf %lf\n", x2614, x2615);
fclose((FILE*)x2616);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

