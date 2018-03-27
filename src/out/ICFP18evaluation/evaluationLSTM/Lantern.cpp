
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
for(int x874=0; x874 < 1; x874++) {
float x875 = 0.0f;
for(int x876=0; x876 < 26; x876++) {
int32_t x877 = x872;
float x878 = x859[x877];
float x879 = x361[x876];
float x880 = x878 * x879;
x875 += x880;
x872 += 1;

}
float x885 = x875;
x873[x874] = x885;

}
float* x889 = (float*)myMalloc(1 * sizeof(float));
for(int x890=0; x890 < 1; x890++) {
x889[x890] = 0.0f;

}
float* x894 = (float*)myMalloc(1 * sizeof(float));
for(int x895=0; x895 < 1; x895++) {
float x896 = x873[x895];
double x897 = (double)x896;
double x898 = log(x897);
float x899 = (float)x898;
x894[x895] = x899;

}
float* x903 = (float*)myMalloc(1 * sizeof(float));
for(int x904=0; x904 < 1; x904++) {
x903[x904] = 0.0f;

}
float* x908 = (float*)myMalloc(1 * sizeof(float));
for(int x909=0; x909 < 1; x909++) {
float x910 = x894[x909];
float x911 = x342[0];
float x912 = x911 - x910;
x908[x909] = x912;

}
float* x916 = (float*)myMalloc(1 * sizeof(float));
for(int x917=0; x917 < 1; x917++) {
x916[x917] = 0.0f;

}
float** x922 = (float**)myMalloc(6 * sizeof(float*));
x922[0] = x908;
x922[1] = x916;
x922[2] = x779;
x922[3] = x787;
x922[4] = x752;
x922[5] = x760;
int32_t x1030 = 0;
int32_t x1046 = 0;
int32_t x1186 = 0;
int32_t x1202 = 0;
int32_t x1219 = 0;
int32_t x1235 = 0;
int32_t x1293 = 0;
int32_t x1309 = 0;
int32_t x1326 = 0;
int32_t x1342 = 0;
int32_t x1400 = 0;
int32_t x1416 = 0;
int32_t x1433 = 0;
int32_t x1449 = 0;
int32_t x1507 = 0;
int32_t x1523 = 0;
int32_t x1540 = 0;
int32_t x1556 = 0;
int32_t x921 = x340 + 1;
x337(x921,x922);
// += tensor of dim 0
float x932 = x916[0];
for(int x933=0; x933 < 1; x933++) {
float x934 = x343[x933];
float x935 = x934 + x932;
x343[x933] = x935;

}
float x939 = x916[0];
for(int x940=0; x940 < 1; x940++) {
float x941 = x903[x940];
float x942 = x941 - x939;
x903[x940] = x942;

}
for(int x946=0; x946 < 1; x946++) {
float x947 = x889[0];
float x948 = x903[0];
float x949 = x873[0];
float x950 = x948 / x949;
float x951 = x947 + x950;
x889[0] = x951;

}
float x955 = x889[0];
// Generate code for addMul
for(int x957=0; x957 < 26; x957++) {
float x958 = x866[x957];
float x959 = x361[x957];
float x960 = x955 * x959;
float x961 = x958 + x960;
x866[x957] = x961;

}
float x965 = x889[0];
// Generate code for addMul
for(int x967=0; x967 < 26; x967++) {
float x968 = x368[x967];
float x969 = x859[x967];
float x970 = x965 * x969;
float x971 = x968 + x970;
x368[x967] = x971;

}
for(int x975=0; x975 < 26; x975++) {
float x976 = x837[x975];
float x977 = x866[x975];
float x978 = x851[0];
float x979 = x977 / x978;
float x980 = x976 + x979;
x837[x975] = x980;

}
for(int x984=0; x984 < 26; x984++) {
float x985 = x853[0];
float x986 = x828[x984];
float x987 = x866[x984];
float x989 = x851[0];
float x988 = x986 * x987;
float x990 = x989 * x989;
float x991 = x988 / x990;
float x992 = x985 - x991;
x853[0] = x992;

}
// += tensor of dim 0
float x997 = x853[0];
for(int x998=0; x998 < 26; x998++) {
float x999 = x837[x998];
float x1000 = x999 + x997;
x837[x998] = x1000;

}
// backpropage exp
for(int x1005=0; x1005 < 26; x1005++) {
float x1006 = x823[x1005];
float x1007 = x828[x1005];
float x1008 = x837[x1005];
float x1009 = x1007 * x1008;
float x1010 = x1006 + x1009;
x823[x1005] = x1010;

}
// backpropagate +
for(int x1015=0; x1015 < 26; x1015++) {
float x1016 = x810[x1015];
float x1017 = x823[x1015];
float x1018 = x1016 + x1017;
x810[x1015] = x1018;

}
for(int x1022=0; x1022 < 26; x1022++) {
float x1023 = x192[x1022];
float x1024 = x823[x1022];
float x1025 = x1023 + x1024;
x192[x1022] = x1025;

}
// add_cartesian
for(int x1031=0; x1031 < 26; x1031++) {
for(int x1032=0; x1032 < 50; x1032++) {
int32_t x1033 = x1030;
int32_t x1034 = x1033 + x1032;
float x1035 = x187[x1034];
float x1036 = x779[x1032];
float x1037 = x810[x1031];
float x1038 = x1036 * x1037;
float x1039 = x1035 + x1038;
x187[x1034] = x1039;

}
x1030 += 50;

}
for(int x1047=0; x1047 < 26; x1047++) {
for(int x1048=0; x1048 < 50; x1048++) {
float x1049 = x787[x1048];
int32_t x1050 = x1046;
int32_t x1051 = x1050 + x1048;
float x1052 = x94[x1051];
float x1053 = x810[x1047];
float x1054 = x1052 * x1053;
float x1055 = x1049 + x1054;
x787[x1048] = x1055;

}
x1046 += 50;

}
for(int x1062=0; x1062 < 50; x1062++) {
float x1063 = x635[x1062];
float x1064 = x765[x1062];
float x1065 = x787[x1062];
float x1066 = x1064 * x1065;
float x1067 = x1063 + x1066;
x635[x1062] = x1067;

}
for(int x1071=0; x1071 < 50; x1071++) {
float x1072 = x774[x1071];
float x1073 = x623[x1071];
float x1074 = x787[x1071];
float x1075 = x1073 * x1074;
float x1076 = x1072 + x1075;
x774[x1071] = x1076;

}
// backpropagate tanh
for(int x1081=0; x1081 < 50; x1081++) {
float x1082 = x760[x1081];
float x1083 = x765[x1081];
float x1086 = x774[x1081];
float x1084 = x1083 * x1083;
float x1085 = 1.0f - x1084;
float x1087 = x1085 * x1086;
float x1088 = x1082 + x1087;
x760[x1081] = x1088;

}
// backpropagate +
for(int x1093=0; x1093 < 50; x1093++) {
float x1094 = x734[x1093];
float x1095 = x760[x1093];
float x1096 = x1094 + x1095;
x734[x1093] = x1096;

}
for(int x1100=0; x1100 < 50; x1100++) {
float x1101 = x747[x1100];
float x1102 = x760[x1100];
float x1103 = x1101 + x1102;
x747[x1100] = x1103;

}
for(int x1107=0; x1107 < 50; x1107++) {
float x1108 = x546[x1107];
float x1109 = x712[x1107];
float x1110 = x747[x1107];
float x1111 = x1109 * x1110;
float x1112 = x1108 + x1111;
x546[x1107] = x1112;

}
for(int x1116=0; x1116 < 50; x1116++) {
float x1117 = x721[x1116];
float x1118 = x534[x1116];
float x1119 = x747[x1116];
float x1120 = x1118 * x1119;
float x1121 = x1117 + x1120;
x721[x1116] = x1121;

}
for(int x1125=0; x1125 < 50; x1125++) {
float x1126 = x457[x1125];
float x1127 = x346[x1125];
float x1128 = x734[x1125];
float x1129 = x1127 * x1128;
float x1130 = x1126 + x1129;
x457[x1125] = x1130;

}
for(int x1134=0; x1134 < 50; x1134++) {
float x1135 = x347[x1134];
float x1136 = x445[x1134];
float x1137 = x734[x1134];
float x1138 = x1136 * x1137;
float x1139 = x1135 + x1138;
x347[x1134] = x1139;

}
// backpropagate tanh
for(int x1144=0; x1144 < 50; x1144++) {
float x1145 = x707[x1144];
float x1146 = x712[x1144];
float x1149 = x721[x1144];
float x1147 = x1146 * x1146;
float x1148 = 1.0f - x1147;
float x1150 = x1148 * x1149;
float x1151 = x1145 + x1150;
x707[x1144] = x1151;

}
// backpropagate +
for(int x1156=0; x1156 < 50; x1156++) {
float x1157 = x694[x1156];
float x1158 = x707[x1156];
float x1159 = x1157 + x1158;
x694[x1156] = x1159;

}
for(int x1163=0; x1163 < 50; x1163++) {
float x1164 = x167[x1163];
float x1165 = x707[x1163];
float x1166 = x1164 + x1165;
x167[x1163] = x1166;

}
// backpropagate +
for(int x1171=0; x1171 < 50; x1171++) {
float x1172 = x658[x1171];
float x1173 = x694[x1171];
float x1174 = x1172 + x1173;
x658[x1171] = x1174;

}
for(int x1178=0; x1178 < 50; x1178++) {
float x1179 = x681[x1178];
float x1180 = x694[x1178];
float x1181 = x1179 + x1180;
x681[x1178] = x1181;

}
// add_cartesian
for(int x1187=0; x1187 < 50; x1187++) {
for(int x1188=0; x1188 < 26; x1188++) {
int32_t x1189 = x1186;
int32_t x1190 = x1189 + x1188;
float x1191 = x162[x1190];
float x1192 = x349[x1188];
float x1193 = x681[x1187];
float x1194 = x1192 * x1193;
float x1195 = x1191 + x1194;
x162[x1190] = x1195;

}
x1186 += 26;

}
for(int x1203=0; x1203 < 50; x1203++) {
for(int x1204=0; x1204 < 26; x1204++) {
float x1205 = x356[x1204];
int32_t x1206 = x1202;
int32_t x1207 = x1206 + x1204;
float x1208 = x63[x1207];
float x1209 = x681[x1203];
float x1210 = x1208 * x1209;
float x1211 = x1205 + x1210;
x356[x1204] = x1211;

}
x1202 += 26;

}
// add_cartesian
for(int x1220=0; x1220 < 50; x1220++) {
for(int x1221=0; x1221 < 50; x1221++) {
int32_t x1222 = x1219;
int32_t x1223 = x1222 + x1221;
float x1224 = x157[x1223];
float x1225 = x344[x1221];
float x1226 = x658[x1220];
float x1227 = x1225 * x1226;
float x1228 = x1224 + x1227;
x157[x1223] = x1228;

}
x1219 += 50;

}
for(int x1236=0; x1236 < 50; x1236++) {
for(int x1237=0; x1237 < 50; x1237++) {
float x1238 = x345[x1237];
int32_t x1239 = x1235;
int32_t x1240 = x1239 + x1237;
float x1241 = x56[x1240];
float x1242 = x658[x1236];
float x1243 = x1241 * x1242;
float x1244 = x1238 + x1243;
x345[x1237] = x1244;

}
x1235 += 50;

}
for(int x1251=0; x1251 < 50; x1251++) {
float x1252 = x618[x1251];
float x1253 = x623[x1251];
float x1256 = x635[x1251];
float x1254 = 1.0f - x1253;
float x1255 = x1254 * x1253;
float x1257 = x1255 * x1256;
float x1258 = x1252 + x1257;
x618[x1251] = x1258;

}
// backpropagate +
for(int x1263=0; x1263 < 50; x1263++) {
float x1264 = x605[x1263];
float x1265 = x618[x1263];
float x1266 = x1264 + x1265;
x605[x1263] = x1266;

}
for(int x1270=0; x1270 < 50; x1270++) {
float x1271 = x182[x1270];
float x1272 = x618[x1270];
float x1273 = x1271 + x1272;
x182[x1270] = x1273;

}
// backpropagate +
for(int x1278=0; x1278 < 50; x1278++) {
float x1279 = x569[x1278];
float x1280 = x605[x1278];
float x1281 = x1279 + x1280;
x569[x1278] = x1281;

}
for(int x1285=0; x1285 < 50; x1285++) {
float x1286 = x592[x1285];
float x1287 = x605[x1285];
float x1288 = x1286 + x1287;
x592[x1285] = x1288;

}
// add_cartesian
for(int x1294=0; x1294 < 50; x1294++) {
for(int x1295=0; x1295 < 26; x1295++) {
int32_t x1296 = x1293;
int32_t x1297 = x1296 + x1295;
float x1298 = x177[x1297];
float x1299 = x349[x1295];
float x1300 = x592[x1294];
float x1301 = x1299 * x1300;
float x1302 = x1298 + x1301;
x177[x1297] = x1302;

}
x1293 += 26;

}
for(int x1310=0; x1310 < 50; x1310++) {
for(int x1311=0; x1311 < 26; x1311++) {
float x1312 = x356[x1311];
int32_t x1313 = x1309;
int32_t x1314 = x1313 + x1311;
float x1315 = x82[x1314];
float x1316 = x592[x1310];
float x1317 = x1315 * x1316;
float x1318 = x1312 + x1317;
x356[x1311] = x1318;

}
x1309 += 26;

}
// add_cartesian
for(int x1327=0; x1327 < 50; x1327++) {
for(int x1328=0; x1328 < 50; x1328++) {
int32_t x1329 = x1326;
int32_t x1330 = x1329 + x1328;
float x1331 = x172[x1330];
float x1332 = x344[x1328];
float x1333 = x569[x1327];
float x1334 = x1332 * x1333;
float x1335 = x1331 + x1334;
x172[x1330] = x1335;

}
x1326 += 50;

}
for(int x1343=0; x1343 < 50; x1343++) {
for(int x1344=0; x1344 < 50; x1344++) {
float x1345 = x345[x1344];
int32_t x1346 = x1342;
int32_t x1347 = x1346 + x1344;
float x1348 = x75[x1347];
float x1349 = x569[x1343];
float x1350 = x1348 * x1349;
float x1351 = x1345 + x1350;
x345[x1344] = x1351;

}
x1342 += 50;

}
for(int x1358=0; x1358 < 50; x1358++) {
float x1359 = x529[x1358];
float x1360 = x534[x1358];
float x1363 = x546[x1358];
float x1361 = 1.0f - x1360;
float x1362 = x1361 * x1360;
float x1364 = x1362 * x1363;
float x1365 = x1359 + x1364;
x529[x1358] = x1365;

}
// backpropagate +
for(int x1370=0; x1370 < 50; x1370++) {
float x1371 = x516[x1370];
float x1372 = x529[x1370];
float x1373 = x1371 + x1372;
x516[x1370] = x1373;

}
for(int x1377=0; x1377 < 50; x1377++) {
float x1378 = x152[x1377];
float x1379 = x529[x1377];
float x1380 = x1378 + x1379;
x152[x1377] = x1380;

}
// backpropagate +
for(int x1385=0; x1385 < 50; x1385++) {
float x1386 = x480[x1385];
float x1387 = x516[x1385];
float x1388 = x1386 + x1387;
x480[x1385] = x1388;

}
for(int x1392=0; x1392 < 50; x1392++) {
float x1393 = x503[x1392];
float x1394 = x516[x1392];
float x1395 = x1393 + x1394;
x503[x1392] = x1395;

}
// add_cartesian
for(int x1401=0; x1401 < 50; x1401++) {
for(int x1402=0; x1402 < 26; x1402++) {
int32_t x1403 = x1400;
int32_t x1404 = x1403 + x1402;
float x1405 = x147[x1404];
float x1406 = x349[x1402];
float x1407 = x503[x1401];
float x1408 = x1406 * x1407;
float x1409 = x1405 + x1408;
x147[x1404] = x1409;

}
x1400 += 26;

}
for(int x1417=0; x1417 < 50; x1417++) {
for(int x1418=0; x1418 < 26; x1418++) {
float x1419 = x356[x1418];
int32_t x1420 = x1416;
int32_t x1421 = x1420 + x1418;
float x1422 = x44[x1421];
float x1423 = x503[x1417];
float x1424 = x1422 * x1423;
float x1425 = x1419 + x1424;
x356[x1418] = x1425;

}
x1416 += 26;

}
// add_cartesian
for(int x1434=0; x1434 < 50; x1434++) {
for(int x1435=0; x1435 < 50; x1435++) {
int32_t x1436 = x1433;
int32_t x1437 = x1436 + x1435;
float x1438 = x142[x1437];
float x1439 = x344[x1435];
float x1440 = x480[x1434];
float x1441 = x1439 * x1440;
float x1442 = x1438 + x1441;
x142[x1437] = x1442;

}
x1433 += 50;

}
for(int x1450=0; x1450 < 50; x1450++) {
for(int x1451=0; x1451 < 50; x1451++) {
float x1452 = x345[x1451];
int32_t x1453 = x1449;
int32_t x1454 = x1453 + x1451;
float x1455 = x37[x1454];
float x1456 = x480[x1450];
float x1457 = x1455 * x1456;
float x1458 = x1452 + x1457;
x345[x1451] = x1458;

}
x1449 += 50;

}
for(int x1465=0; x1465 < 50; x1465++) {
float x1466 = x440[x1465];
float x1467 = x445[x1465];
float x1470 = x457[x1465];
float x1468 = 1.0f - x1467;
float x1469 = x1468 * x1467;
float x1471 = x1469 * x1470;
float x1472 = x1466 + x1471;
x440[x1465] = x1472;

}
// backpropagate +
for(int x1477=0; x1477 < 50; x1477++) {
float x1478 = x427[x1477];
float x1479 = x440[x1477];
float x1480 = x1478 + x1479;
x427[x1477] = x1480;

}
for(int x1484=0; x1484 < 50; x1484++) {
float x1485 = x137[x1484];
float x1486 = x440[x1484];
float x1487 = x1485 + x1486;
x137[x1484] = x1487;

}
// backpropagate +
for(int x1492=0; x1492 < 50; x1492++) {
float x1493 = x391[x1492];
float x1494 = x427[x1492];
float x1495 = x1493 + x1494;
x391[x1492] = x1495;

}
for(int x1499=0; x1499 < 50; x1499++) {
float x1500 = x414[x1499];
float x1501 = x427[x1499];
float x1502 = x1500 + x1501;
x414[x1499] = x1502;

}
// add_cartesian
for(int x1508=0; x1508 < 50; x1508++) {
for(int x1509=0; x1509 < 26; x1509++) {
int32_t x1510 = x1507;
int32_t x1511 = x1510 + x1509;
float x1512 = x132[x1511];
float x1513 = x349[x1509];
float x1514 = x414[x1508];
float x1515 = x1513 * x1514;
float x1516 = x1512 + x1515;
x132[x1511] = x1516;

}
x1507 += 26;

}
for(int x1524=0; x1524 < 50; x1524++) {
for(int x1525=0; x1525 < 26; x1525++) {
float x1526 = x356[x1525];
int32_t x1527 = x1523;
int32_t x1528 = x1527 + x1525;
float x1529 = x23[x1528];
float x1530 = x414[x1524];
float x1531 = x1529 * x1530;
float x1532 = x1526 + x1531;
x356[x1525] = x1532;

}
x1523 += 26;

}
// add_cartesian
for(int x1541=0; x1541 < 50; x1541++) {
for(int x1542=0; x1542 < 50; x1542++) {
int32_t x1543 = x1540;
int32_t x1544 = x1543 + x1542;
float x1545 = x127[x1544];
float x1546 = x344[x1542];
float x1547 = x391[x1541];
float x1548 = x1546 * x1547;
float x1549 = x1545 + x1548;
x127[x1544] = x1549;

}
x1540 += 50;

}
for(int x1557=0; x1557 < 50; x1557++) {
for(int x1558=0; x1558 < 50; x1558++) {
float x1559 = x345[x1558];
int32_t x1560 = x1556;
int32_t x1561 = x1560 + x1558;
float x1562 = x15[x1561];
float x1563 = x391[x1557];
float x1564 = x1562 * x1563;
float x1565 = x1559 + x1564;
x345[x1558] = x1565;

}
x1556 += 50;

}
} else {
for(int x1573=0; x1573 < 50; x1573++) {
float x1574 = x344[x1573];
x117[x1573] = x1574;

}
for(int x1578=0; x1578 < 50; x1578++) {
float x1579 = x346[x1578];
x122[x1578] = x1579;

}
for(int x1583=0; x1583 < 1; x1583++) {
float x1584 = x343[x1583];
x343[x1583] = 1.0f;

}
for(int x1588=0; x1588 < 1; x1588++) {
float x1589 = x342[x1588];
x322[x1588] = x1589;

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
float** x1596 = (float**)myMalloc(6 * sizeof(float*));
x1596[0] = x327;
x1596[1] = x332;
x1596[2] = x107;
x1596[3] = x197;
x1596[4] = x112;
x1596[5] = x202;
x337(0,x1596);
float x1605 = x322[0];
double x1606 = x282;
double x1607 = x1606 * 0.9;
double x1608 = (double)x1605;
double x1609 = x1608 * 0.1;
double x1610 = x1607 + x1609;
x282 = x1610;
int32_t x1612 = x284 % 100;
bool x1613 = x1612 == 0;
if (x1613) {
double x1614 = x282;
printf("iter %d, loss %f\n",x284,x1614);
int32_t x1616 = x284 / 100;
x278[x1616] = x1614;
} else {
}
for(int x1620=0; x1620 < 2500; x1620++) {
float x1621 = x127[x1620];
bool x1622 = x1621 > 5.0f;
if (x1622) {
x127[x1620] = 5.0f;
} else {
}
float x1626 = x127[x1620];
bool x1627 = x1626 < -5.0f;
if (x1627) {
x127[x1620] = -5.0f;
} else {
}

}
float* x1633 = (float*)myMalloc(2500 * sizeof(float));
for(int x1634=0; x1634 < 2500; x1634++) {
float x1635 = x127[x1634];
float x1636 = x127[x1634];
float x1637 = x1635 * x1636;
x1633[x1634] = x1637;

}
for(int x1641=0; x1641 < 2500; x1641++) {
float x1642 = x207[x1641];
float x1643 = x1633[x1641];
float x1644 = x1642 + x1643;
x207[x1641] = x1644;

}
float* x1648 = (float*)myMalloc(2500 * sizeof(float));
for(int x1649=0; x1649 < 2500; x1649++) {
float x1650 = x127[x1649];
float x1651 = x1650 * 0.1f;
x1648[x1649] = x1651;

}
float* x1655 = (float*)myMalloc(2500 * sizeof(float));
for(int x1656=0; x1656 < 2500; x1656++) {
float x1657 = x207[x1656];
float x1658 = x1657 + 1.0E-8f;
x1655[x1656] = x1658;

}
float* x1662 = (float*)myMalloc(2500 * sizeof(float));
for(int x1663=0; x1663 < 2500; x1663++) {
float x1664 = x1655[x1663];
double x1665 = (double)x1664;
double x1666 = sqrt(x1665);
float x1667 = (float)x1666;
x1662[x1663] = x1667;

}
float* x1671 = (float*)myMalloc(2500 * sizeof(float));
for(int x1672=0; x1672 < 2500; x1672++) {
float x1673 = x1648[x1672];
float x1674 = x1662[x1672];
float x1675 = x1673 / x1674;
x1671[x1672] = x1675;

}
for(int x1679=0; x1679 < 2500; x1679++) {
float x1680 = x15[x1679];
float x1681 = x1671[x1679];
float x1682 = x1680 - x1681;
x15[x1679] = x1682;

}
for(int x1686=0; x1686 < 2500; x1686++) {
float x1687 = x127[x1686];
x127[x1686] = 0.0f;

}
for(int x1691=0; x1691 < 1300; x1691++) {
float x1692 = x132[x1691];
bool x1693 = x1692 > 5.0f;
if (x1693) {
x132[x1691] = 5.0f;
} else {
}
float x1697 = x132[x1691];
bool x1698 = x1697 < -5.0f;
if (x1698) {
x132[x1691] = -5.0f;
} else {
}

}
float* x1704 = (float*)myMalloc(1300 * sizeof(float));
for(int x1705=0; x1705 < 1300; x1705++) {
float x1706 = x132[x1705];
float x1707 = x132[x1705];
float x1708 = x1706 * x1707;
x1704[x1705] = x1708;

}
for(int x1712=0; x1712 < 1300; x1712++) {
float x1713 = x212[x1712];
float x1714 = x1704[x1712];
float x1715 = x1713 + x1714;
x212[x1712] = x1715;

}
float* x1719 = (float*)myMalloc(1300 * sizeof(float));
for(int x1720=0; x1720 < 1300; x1720++) {
float x1721 = x132[x1720];
float x1722 = x1721 * 0.1f;
x1719[x1720] = x1722;

}
float* x1726 = (float*)myMalloc(1300 * sizeof(float));
for(int x1727=0; x1727 < 1300; x1727++) {
float x1728 = x212[x1727];
float x1729 = x1728 + 1.0E-8f;
x1726[x1727] = x1729;

}
float* x1733 = (float*)myMalloc(1300 * sizeof(float));
for(int x1734=0; x1734 < 1300; x1734++) {
float x1735 = x1726[x1734];
double x1736 = (double)x1735;
double x1737 = sqrt(x1736);
float x1738 = (float)x1737;
x1733[x1734] = x1738;

}
float* x1742 = (float*)myMalloc(1300 * sizeof(float));
for(int x1743=0; x1743 < 1300; x1743++) {
float x1744 = x1719[x1743];
float x1745 = x1733[x1743];
float x1746 = x1744 / x1745;
x1742[x1743] = x1746;

}
for(int x1750=0; x1750 < 1300; x1750++) {
float x1751 = x23[x1750];
float x1752 = x1742[x1750];
float x1753 = x1751 - x1752;
x23[x1750] = x1753;

}
for(int x1757=0; x1757 < 1300; x1757++) {
float x1758 = x132[x1757];
x132[x1757] = 0.0f;

}
for(int x1762=0; x1762 < 50; x1762++) {
float x1763 = x137[x1762];
bool x1764 = x1763 > 5.0f;
if (x1764) {
x137[x1762] = 5.0f;
} else {
}
float x1768 = x137[x1762];
bool x1769 = x1768 < -5.0f;
if (x1769) {
x137[x1762] = -5.0f;
} else {
}

}
float* x1775 = (float*)myMalloc(50 * sizeof(float));
for(int x1776=0; x1776 < 50; x1776++) {
float x1777 = x137[x1776];
float x1778 = x137[x1776];
float x1779 = x1777 * x1778;
x1775[x1776] = x1779;

}
for(int x1783=0; x1783 < 50; x1783++) {
float x1784 = x217[x1783];
float x1785 = x1775[x1783];
float x1786 = x1784 + x1785;
x217[x1783] = x1786;

}
float* x1790 = (float*)myMalloc(50 * sizeof(float));
for(int x1791=0; x1791 < 50; x1791++) {
float x1792 = x137[x1791];
float x1793 = x1792 * 0.1f;
x1790[x1791] = x1793;

}
float* x1797 = (float*)myMalloc(50 * sizeof(float));
for(int x1798=0; x1798 < 50; x1798++) {
float x1799 = x217[x1798];
float x1800 = x1799 + 1.0E-8f;
x1797[x1798] = x1800;

}
float* x1804 = (float*)myMalloc(50 * sizeof(float));
for(int x1805=0; x1805 < 50; x1805++) {
float x1806 = x1797[x1805];
double x1807 = (double)x1806;
double x1808 = sqrt(x1807);
float x1809 = (float)x1808;
x1804[x1805] = x1809;

}
float* x1813 = (float*)myMalloc(50 * sizeof(float));
for(int x1814=0; x1814 < 50; x1814++) {
float x1815 = x1790[x1814];
float x1816 = x1804[x1814];
float x1817 = x1815 / x1816;
x1813[x1814] = x1817;

}
for(int x1821=0; x1821 < 50; x1821++) {
float x1822 = x31[x1821];
float x1823 = x1813[x1821];
float x1824 = x1822 - x1823;
x31[x1821] = x1824;

}
for(int x1828=0; x1828 < 50; x1828++) {
float x1829 = x137[x1828];
x137[x1828] = 0.0f;

}
for(int x1833=0; x1833 < 2500; x1833++) {
float x1834 = x142[x1833];
bool x1835 = x1834 > 5.0f;
if (x1835) {
x142[x1833] = 5.0f;
} else {
}
float x1839 = x142[x1833];
bool x1840 = x1839 < -5.0f;
if (x1840) {
x142[x1833] = -5.0f;
} else {
}

}
float* x1846 = (float*)myMalloc(2500 * sizeof(float));
for(int x1847=0; x1847 < 2500; x1847++) {
float x1848 = x142[x1847];
float x1849 = x142[x1847];
float x1850 = x1848 * x1849;
x1846[x1847] = x1850;

}
for(int x1854=0; x1854 < 2500; x1854++) {
float x1855 = x222[x1854];
float x1856 = x1846[x1854];
float x1857 = x1855 + x1856;
x222[x1854] = x1857;

}
float* x1861 = (float*)myMalloc(2500 * sizeof(float));
for(int x1862=0; x1862 < 2500; x1862++) {
float x1863 = x142[x1862];
float x1864 = x1863 * 0.1f;
x1861[x1862] = x1864;

}
float* x1868 = (float*)myMalloc(2500 * sizeof(float));
for(int x1869=0; x1869 < 2500; x1869++) {
float x1870 = x222[x1869];
float x1871 = x1870 + 1.0E-8f;
x1868[x1869] = x1871;

}
float* x1875 = (float*)myMalloc(2500 * sizeof(float));
for(int x1876=0; x1876 < 2500; x1876++) {
float x1877 = x1868[x1876];
double x1878 = (double)x1877;
double x1879 = sqrt(x1878);
float x1880 = (float)x1879;
x1875[x1876] = x1880;

}
float* x1884 = (float*)myMalloc(2500 * sizeof(float));
for(int x1885=0; x1885 < 2500; x1885++) {
float x1886 = x1861[x1885];
float x1887 = x1875[x1885];
float x1888 = x1886 / x1887;
x1884[x1885] = x1888;

}
for(int x1892=0; x1892 < 2500; x1892++) {
float x1893 = x37[x1892];
float x1894 = x1884[x1892];
float x1895 = x1893 - x1894;
x37[x1892] = x1895;

}
for(int x1899=0; x1899 < 2500; x1899++) {
float x1900 = x142[x1899];
x142[x1899] = 0.0f;

}
for(int x1904=0; x1904 < 1300; x1904++) {
float x1905 = x147[x1904];
bool x1906 = x1905 > 5.0f;
if (x1906) {
x147[x1904] = 5.0f;
} else {
}
float x1910 = x147[x1904];
bool x1911 = x1910 < -5.0f;
if (x1911) {
x147[x1904] = -5.0f;
} else {
}

}
float* x1917 = (float*)myMalloc(1300 * sizeof(float));
for(int x1918=0; x1918 < 1300; x1918++) {
float x1919 = x147[x1918];
float x1920 = x147[x1918];
float x1921 = x1919 * x1920;
x1917[x1918] = x1921;

}
for(int x1925=0; x1925 < 1300; x1925++) {
float x1926 = x227[x1925];
float x1927 = x1917[x1925];
float x1928 = x1926 + x1927;
x227[x1925] = x1928;

}
float* x1932 = (float*)myMalloc(1300 * sizeof(float));
for(int x1933=0; x1933 < 1300; x1933++) {
float x1934 = x147[x1933];
float x1935 = x1934 * 0.1f;
x1932[x1933] = x1935;

}
float* x1939 = (float*)myMalloc(1300 * sizeof(float));
for(int x1940=0; x1940 < 1300; x1940++) {
float x1941 = x227[x1940];
float x1942 = x1941 + 1.0E-8f;
x1939[x1940] = x1942;

}
float* x1946 = (float*)myMalloc(1300 * sizeof(float));
for(int x1947=0; x1947 < 1300; x1947++) {
float x1948 = x1939[x1947];
double x1949 = (double)x1948;
double x1950 = sqrt(x1949);
float x1951 = (float)x1950;
x1946[x1947] = x1951;

}
float* x1955 = (float*)myMalloc(1300 * sizeof(float));
for(int x1956=0; x1956 < 1300; x1956++) {
float x1957 = x1932[x1956];
float x1958 = x1946[x1956];
float x1959 = x1957 / x1958;
x1955[x1956] = x1959;

}
for(int x1963=0; x1963 < 1300; x1963++) {
float x1964 = x44[x1963];
float x1965 = x1955[x1963];
float x1966 = x1964 - x1965;
x44[x1963] = x1966;

}
for(int x1970=0; x1970 < 1300; x1970++) {
float x1971 = x147[x1970];
x147[x1970] = 0.0f;

}
for(int x1975=0; x1975 < 50; x1975++) {
float x1976 = x152[x1975];
bool x1977 = x1976 > 5.0f;
if (x1977) {
x152[x1975] = 5.0f;
} else {
}
float x1981 = x152[x1975];
bool x1982 = x1981 < -5.0f;
if (x1982) {
x152[x1975] = -5.0f;
} else {
}

}
float* x1988 = (float*)myMalloc(50 * sizeof(float));
for(int x1989=0; x1989 < 50; x1989++) {
float x1990 = x152[x1989];
float x1991 = x152[x1989];
float x1992 = x1990 * x1991;
x1988[x1989] = x1992;

}
for(int x1996=0; x1996 < 50; x1996++) {
float x1997 = x232[x1996];
float x1998 = x1988[x1996];
float x1999 = x1997 + x1998;
x232[x1996] = x1999;

}
float* x2003 = (float*)myMalloc(50 * sizeof(float));
for(int x2004=0; x2004 < 50; x2004++) {
float x2005 = x152[x2004];
float x2006 = x2005 * 0.1f;
x2003[x2004] = x2006;

}
float* x2010 = (float*)myMalloc(50 * sizeof(float));
for(int x2011=0; x2011 < 50; x2011++) {
float x2012 = x232[x2011];
float x2013 = x2012 + 1.0E-8f;
x2010[x2011] = x2013;

}
float* x2017 = (float*)myMalloc(50 * sizeof(float));
for(int x2018=0; x2018 < 50; x2018++) {
float x2019 = x2010[x2018];
double x2020 = (double)x2019;
double x2021 = sqrt(x2020);
float x2022 = (float)x2021;
x2017[x2018] = x2022;

}
float* x2026 = (float*)myMalloc(50 * sizeof(float));
for(int x2027=0; x2027 < 50; x2027++) {
float x2028 = x2003[x2027];
float x2029 = x2017[x2027];
float x2030 = x2028 / x2029;
x2026[x2027] = x2030;

}
for(int x2034=0; x2034 < 50; x2034++) {
float x2035 = x51[x2034];
float x2036 = x2026[x2034];
float x2037 = x2035 - x2036;
x51[x2034] = x2037;

}
for(int x2041=0; x2041 < 50; x2041++) {
float x2042 = x152[x2041];
x152[x2041] = 0.0f;

}
for(int x2046=0; x2046 < 2500; x2046++) {
float x2047 = x157[x2046];
bool x2048 = x2047 > 5.0f;
if (x2048) {
x157[x2046] = 5.0f;
} else {
}
float x2052 = x157[x2046];
bool x2053 = x2052 < -5.0f;
if (x2053) {
x157[x2046] = -5.0f;
} else {
}

}
float* x2059 = (float*)myMalloc(2500 * sizeof(float));
for(int x2060=0; x2060 < 2500; x2060++) {
float x2061 = x157[x2060];
float x2062 = x157[x2060];
float x2063 = x2061 * x2062;
x2059[x2060] = x2063;

}
for(int x2067=0; x2067 < 2500; x2067++) {
float x2068 = x237[x2067];
float x2069 = x2059[x2067];
float x2070 = x2068 + x2069;
x237[x2067] = x2070;

}
float* x2074 = (float*)myMalloc(2500 * sizeof(float));
for(int x2075=0; x2075 < 2500; x2075++) {
float x2076 = x157[x2075];
float x2077 = x2076 * 0.1f;
x2074[x2075] = x2077;

}
float* x2081 = (float*)myMalloc(2500 * sizeof(float));
for(int x2082=0; x2082 < 2500; x2082++) {
float x2083 = x237[x2082];
float x2084 = x2083 + 1.0E-8f;
x2081[x2082] = x2084;

}
float* x2088 = (float*)myMalloc(2500 * sizeof(float));
for(int x2089=0; x2089 < 2500; x2089++) {
float x2090 = x2081[x2089];
double x2091 = (double)x2090;
double x2092 = sqrt(x2091);
float x2093 = (float)x2092;
x2088[x2089] = x2093;

}
float* x2097 = (float*)myMalloc(2500 * sizeof(float));
for(int x2098=0; x2098 < 2500; x2098++) {
float x2099 = x2074[x2098];
float x2100 = x2088[x2098];
float x2101 = x2099 / x2100;
x2097[x2098] = x2101;

}
for(int x2105=0; x2105 < 2500; x2105++) {
float x2106 = x56[x2105];
float x2107 = x2097[x2105];
float x2108 = x2106 - x2107;
x56[x2105] = x2108;

}
for(int x2112=0; x2112 < 2500; x2112++) {
float x2113 = x157[x2112];
x157[x2112] = 0.0f;

}
for(int x2117=0; x2117 < 1300; x2117++) {
float x2118 = x162[x2117];
bool x2119 = x2118 > 5.0f;
if (x2119) {
x162[x2117] = 5.0f;
} else {
}
float x2123 = x162[x2117];
bool x2124 = x2123 < -5.0f;
if (x2124) {
x162[x2117] = -5.0f;
} else {
}

}
float* x2130 = (float*)myMalloc(1300 * sizeof(float));
for(int x2131=0; x2131 < 1300; x2131++) {
float x2132 = x162[x2131];
float x2133 = x162[x2131];
float x2134 = x2132 * x2133;
x2130[x2131] = x2134;

}
for(int x2138=0; x2138 < 1300; x2138++) {
float x2139 = x242[x2138];
float x2140 = x2130[x2138];
float x2141 = x2139 + x2140;
x242[x2138] = x2141;

}
float* x2145 = (float*)myMalloc(1300 * sizeof(float));
for(int x2146=0; x2146 < 1300; x2146++) {
float x2147 = x162[x2146];
float x2148 = x2147 * 0.1f;
x2145[x2146] = x2148;

}
float* x2152 = (float*)myMalloc(1300 * sizeof(float));
for(int x2153=0; x2153 < 1300; x2153++) {
float x2154 = x242[x2153];
float x2155 = x2154 + 1.0E-8f;
x2152[x2153] = x2155;

}
float* x2159 = (float*)myMalloc(1300 * sizeof(float));
for(int x2160=0; x2160 < 1300; x2160++) {
float x2161 = x2152[x2160];
double x2162 = (double)x2161;
double x2163 = sqrt(x2162);
float x2164 = (float)x2163;
x2159[x2160] = x2164;

}
float* x2168 = (float*)myMalloc(1300 * sizeof(float));
for(int x2169=0; x2169 < 1300; x2169++) {
float x2170 = x2145[x2169];
float x2171 = x2159[x2169];
float x2172 = x2170 / x2171;
x2168[x2169] = x2172;

}
for(int x2176=0; x2176 < 1300; x2176++) {
float x2177 = x63[x2176];
float x2178 = x2168[x2176];
float x2179 = x2177 - x2178;
x63[x2176] = x2179;

}
for(int x2183=0; x2183 < 1300; x2183++) {
float x2184 = x162[x2183];
x162[x2183] = 0.0f;

}
for(int x2188=0; x2188 < 50; x2188++) {
float x2189 = x167[x2188];
bool x2190 = x2189 > 5.0f;
if (x2190) {
x167[x2188] = 5.0f;
} else {
}
float x2194 = x167[x2188];
bool x2195 = x2194 < -5.0f;
if (x2195) {
x167[x2188] = -5.0f;
} else {
}

}
float* x2201 = (float*)myMalloc(50 * sizeof(float));
for(int x2202=0; x2202 < 50; x2202++) {
float x2203 = x167[x2202];
float x2204 = x167[x2202];
float x2205 = x2203 * x2204;
x2201[x2202] = x2205;

}
for(int x2209=0; x2209 < 50; x2209++) {
float x2210 = x247[x2209];
float x2211 = x2201[x2209];
float x2212 = x2210 + x2211;
x247[x2209] = x2212;

}
float* x2216 = (float*)myMalloc(50 * sizeof(float));
for(int x2217=0; x2217 < 50; x2217++) {
float x2218 = x167[x2217];
float x2219 = x2218 * 0.1f;
x2216[x2217] = x2219;

}
float* x2223 = (float*)myMalloc(50 * sizeof(float));
for(int x2224=0; x2224 < 50; x2224++) {
float x2225 = x247[x2224];
float x2226 = x2225 + 1.0E-8f;
x2223[x2224] = x2226;

}
float* x2230 = (float*)myMalloc(50 * sizeof(float));
for(int x2231=0; x2231 < 50; x2231++) {
float x2232 = x2223[x2231];
double x2233 = (double)x2232;
double x2234 = sqrt(x2233);
float x2235 = (float)x2234;
x2230[x2231] = x2235;

}
float* x2239 = (float*)myMalloc(50 * sizeof(float));
for(int x2240=0; x2240 < 50; x2240++) {
float x2241 = x2216[x2240];
float x2242 = x2230[x2240];
float x2243 = x2241 / x2242;
x2239[x2240] = x2243;

}
for(int x2247=0; x2247 < 50; x2247++) {
float x2248 = x70[x2247];
float x2249 = x2239[x2247];
float x2250 = x2248 - x2249;
x70[x2247] = x2250;

}
for(int x2254=0; x2254 < 50; x2254++) {
float x2255 = x167[x2254];
x167[x2254] = 0.0f;

}
for(int x2259=0; x2259 < 2500; x2259++) {
float x2260 = x172[x2259];
bool x2261 = x2260 > 5.0f;
if (x2261) {
x172[x2259] = 5.0f;
} else {
}
float x2265 = x172[x2259];
bool x2266 = x2265 < -5.0f;
if (x2266) {
x172[x2259] = -5.0f;
} else {
}

}
float* x2272 = (float*)myMalloc(2500 * sizeof(float));
for(int x2273=0; x2273 < 2500; x2273++) {
float x2274 = x172[x2273];
float x2275 = x172[x2273];
float x2276 = x2274 * x2275;
x2272[x2273] = x2276;

}
for(int x2280=0; x2280 < 2500; x2280++) {
float x2281 = x252[x2280];
float x2282 = x2272[x2280];
float x2283 = x2281 + x2282;
x252[x2280] = x2283;

}
float* x2287 = (float*)myMalloc(2500 * sizeof(float));
for(int x2288=0; x2288 < 2500; x2288++) {
float x2289 = x172[x2288];
float x2290 = x2289 * 0.1f;
x2287[x2288] = x2290;

}
float* x2294 = (float*)myMalloc(2500 * sizeof(float));
for(int x2295=0; x2295 < 2500; x2295++) {
float x2296 = x252[x2295];
float x2297 = x2296 + 1.0E-8f;
x2294[x2295] = x2297;

}
float* x2301 = (float*)myMalloc(2500 * sizeof(float));
for(int x2302=0; x2302 < 2500; x2302++) {
float x2303 = x2294[x2302];
double x2304 = (double)x2303;
double x2305 = sqrt(x2304);
float x2306 = (float)x2305;
x2301[x2302] = x2306;

}
float* x2310 = (float*)myMalloc(2500 * sizeof(float));
for(int x2311=0; x2311 < 2500; x2311++) {
float x2312 = x2287[x2311];
float x2313 = x2301[x2311];
float x2314 = x2312 / x2313;
x2310[x2311] = x2314;

}
for(int x2318=0; x2318 < 2500; x2318++) {
float x2319 = x75[x2318];
float x2320 = x2310[x2318];
float x2321 = x2319 - x2320;
x75[x2318] = x2321;

}
for(int x2325=0; x2325 < 2500; x2325++) {
float x2326 = x172[x2325];
x172[x2325] = 0.0f;

}
for(int x2330=0; x2330 < 1300; x2330++) {
float x2331 = x177[x2330];
bool x2332 = x2331 > 5.0f;
if (x2332) {
x177[x2330] = 5.0f;
} else {
}
float x2336 = x177[x2330];
bool x2337 = x2336 < -5.0f;
if (x2337) {
x177[x2330] = -5.0f;
} else {
}

}
float* x2343 = (float*)myMalloc(1300 * sizeof(float));
for(int x2344=0; x2344 < 1300; x2344++) {
float x2345 = x177[x2344];
float x2346 = x177[x2344];
float x2347 = x2345 * x2346;
x2343[x2344] = x2347;

}
for(int x2351=0; x2351 < 1300; x2351++) {
float x2352 = x257[x2351];
float x2353 = x2343[x2351];
float x2354 = x2352 + x2353;
x257[x2351] = x2354;

}
float* x2358 = (float*)myMalloc(1300 * sizeof(float));
for(int x2359=0; x2359 < 1300; x2359++) {
float x2360 = x177[x2359];
float x2361 = x2360 * 0.1f;
x2358[x2359] = x2361;

}
float* x2365 = (float*)myMalloc(1300 * sizeof(float));
for(int x2366=0; x2366 < 1300; x2366++) {
float x2367 = x257[x2366];
float x2368 = x2367 + 1.0E-8f;
x2365[x2366] = x2368;

}
float* x2372 = (float*)myMalloc(1300 * sizeof(float));
for(int x2373=0; x2373 < 1300; x2373++) {
float x2374 = x2365[x2373];
double x2375 = (double)x2374;
double x2376 = sqrt(x2375);
float x2377 = (float)x2376;
x2372[x2373] = x2377;

}
float* x2381 = (float*)myMalloc(1300 * sizeof(float));
for(int x2382=0; x2382 < 1300; x2382++) {
float x2383 = x2358[x2382];
float x2384 = x2372[x2382];
float x2385 = x2383 / x2384;
x2381[x2382] = x2385;

}
for(int x2389=0; x2389 < 1300; x2389++) {
float x2390 = x82[x2389];
float x2391 = x2381[x2389];
float x2392 = x2390 - x2391;
x82[x2389] = x2392;

}
for(int x2396=0; x2396 < 1300; x2396++) {
float x2397 = x177[x2396];
x177[x2396] = 0.0f;

}
for(int x2401=0; x2401 < 50; x2401++) {
float x2402 = x182[x2401];
bool x2403 = x2402 > 5.0f;
if (x2403) {
x182[x2401] = 5.0f;
} else {
}
float x2407 = x182[x2401];
bool x2408 = x2407 < -5.0f;
if (x2408) {
x182[x2401] = -5.0f;
} else {
}

}
float* x2414 = (float*)myMalloc(50 * sizeof(float));
for(int x2415=0; x2415 < 50; x2415++) {
float x2416 = x182[x2415];
float x2417 = x182[x2415];
float x2418 = x2416 * x2417;
x2414[x2415] = x2418;

}
for(int x2422=0; x2422 < 50; x2422++) {
float x2423 = x262[x2422];
float x2424 = x2414[x2422];
float x2425 = x2423 + x2424;
x262[x2422] = x2425;

}
float* x2429 = (float*)myMalloc(50 * sizeof(float));
for(int x2430=0; x2430 < 50; x2430++) {
float x2431 = x182[x2430];
float x2432 = x2431 * 0.1f;
x2429[x2430] = x2432;

}
float* x2436 = (float*)myMalloc(50 * sizeof(float));
for(int x2437=0; x2437 < 50; x2437++) {
float x2438 = x262[x2437];
float x2439 = x2438 + 1.0E-8f;
x2436[x2437] = x2439;

}
float* x2443 = (float*)myMalloc(50 * sizeof(float));
for(int x2444=0; x2444 < 50; x2444++) {
float x2445 = x2436[x2444];
double x2446 = (double)x2445;
double x2447 = sqrt(x2446);
float x2448 = (float)x2447;
x2443[x2444] = x2448;

}
float* x2452 = (float*)myMalloc(50 * sizeof(float));
for(int x2453=0; x2453 < 50; x2453++) {
float x2454 = x2429[x2453];
float x2455 = x2443[x2453];
float x2456 = x2454 / x2455;
x2452[x2453] = x2456;

}
for(int x2460=0; x2460 < 50; x2460++) {
float x2461 = x89[x2460];
float x2462 = x2452[x2460];
float x2463 = x2461 - x2462;
x89[x2460] = x2463;

}
for(int x2467=0; x2467 < 50; x2467++) {
float x2468 = x182[x2467];
x182[x2467] = 0.0f;

}
for(int x2472=0; x2472 < 1300; x2472++) {
float x2473 = x187[x2472];
bool x2474 = x2473 > 5.0f;
if (x2474) {
x187[x2472] = 5.0f;
} else {
}
float x2478 = x187[x2472];
bool x2479 = x2478 < -5.0f;
if (x2479) {
x187[x2472] = -5.0f;
} else {
}

}
float* x2485 = (float*)myMalloc(1300 * sizeof(float));
for(int x2486=0; x2486 < 1300; x2486++) {
float x2487 = x187[x2486];
float x2488 = x187[x2486];
float x2489 = x2487 * x2488;
x2485[x2486] = x2489;

}
for(int x2493=0; x2493 < 1300; x2493++) {
float x2494 = x267[x2493];
float x2495 = x2485[x2493];
float x2496 = x2494 + x2495;
x267[x2493] = x2496;

}
float* x2500 = (float*)myMalloc(1300 * sizeof(float));
for(int x2501=0; x2501 < 1300; x2501++) {
float x2502 = x187[x2501];
float x2503 = x2502 * 0.1f;
x2500[x2501] = x2503;

}
float* x2507 = (float*)myMalloc(1300 * sizeof(float));
for(int x2508=0; x2508 < 1300; x2508++) {
float x2509 = x267[x2508];
float x2510 = x2509 + 1.0E-8f;
x2507[x2508] = x2510;

}
float* x2514 = (float*)myMalloc(1300 * sizeof(float));
for(int x2515=0; x2515 < 1300; x2515++) {
float x2516 = x2507[x2515];
double x2517 = (double)x2516;
double x2518 = sqrt(x2517);
float x2519 = (float)x2518;
x2514[x2515] = x2519;

}
float* x2523 = (float*)myMalloc(1300 * sizeof(float));
for(int x2524=0; x2524 < 1300; x2524++) {
float x2525 = x2500[x2524];
float x2526 = x2514[x2524];
float x2527 = x2525 / x2526;
x2523[x2524] = x2527;

}
for(int x2531=0; x2531 < 1300; x2531++) {
float x2532 = x94[x2531];
float x2533 = x2523[x2531];
float x2534 = x2532 - x2533;
x94[x2531] = x2534;

}
for(int x2538=0; x2538 < 1300; x2538++) {
float x2539 = x187[x2538];
x187[x2538] = 0.0f;

}
for(int x2543=0; x2543 < 26; x2543++) {
float x2544 = x192[x2543];
bool x2545 = x2544 > 5.0f;
if (x2545) {
x192[x2543] = 5.0f;
} else {
}
float x2549 = x192[x2543];
bool x2550 = x2549 < -5.0f;
if (x2550) {
x192[x2543] = -5.0f;
} else {
}

}
float* x2556 = (float*)myMalloc(26 * sizeof(float));
for(int x2557=0; x2557 < 26; x2557++) {
float x2558 = x192[x2557];
float x2559 = x192[x2557];
float x2560 = x2558 * x2559;
x2556[x2557] = x2560;

}
for(int x2564=0; x2564 < 26; x2564++) {
float x2565 = x272[x2564];
float x2566 = x2556[x2564];
float x2567 = x2565 + x2566;
x272[x2564] = x2567;

}
float* x2571 = (float*)myMalloc(26 * sizeof(float));
for(int x2572=0; x2572 < 26; x2572++) {
float x2573 = x192[x2572];
float x2574 = x2573 * 0.1f;
x2571[x2572] = x2574;

}
float* x2578 = (float*)myMalloc(26 * sizeof(float));
for(int x2579=0; x2579 < 26; x2579++) {
float x2580 = x272[x2579];
float x2581 = x2580 + 1.0E-8f;
x2578[x2579] = x2581;

}
float* x2585 = (float*)myMalloc(26 * sizeof(float));
for(int x2586=0; x2586 < 26; x2586++) {
float x2587 = x2578[x2586];
double x2588 = (double)x2587;
double x2589 = sqrt(x2588);
float x2590 = (float)x2589;
x2585[x2586] = x2590;

}
float* x2594 = (float*)myMalloc(26 * sizeof(float));
for(int x2595=0; x2595 < 26; x2595++) {
float x2596 = x2571[x2595];
float x2597 = x2585[x2595];
float x2598 = x2596 / x2597;
x2594[x2595] = x2598;

}
for(int x2602=0; x2602 < 26; x2602++) {
float x2603 = x101[x2602];
float x2604 = x2594[x2602];
float x2605 = x2603 - x2604;
x101[x2602] = x2605;

}
for(int x2609=0; x2609 < 26; x2609++) {
float x2610 = x192[x2609];
x192[x2609] = 0.0f;

}
for(int x2614=0; x2614 < 50; x2614++) {
float x2615 = x197[x2614];
x197[x2614] = 0.0f;

}
for(int x2619=0; x2619 < 50; x2619++) {
float x2620 = x202[x2619];
x202[x2619] = 0.0f;

}
for(int x2624=0; x2624 < 50; x2624++) {
float x2625 = x117[x2624];
x107[x2624] = x2625;

}
for(int x2629=0; x2629 < 50; x2629++) {
float x2630 = x122[x2629];
x112[x2629] = x2630;

}
mallocAddr = (void*)x279;

}
double x2637 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x2640 = (long)fopen(x0, "w");
fprintf((FILE *)x2640, "unit: %s\n", "100 iteration");
for(int x2643=0; x2643 < 51; x2643++) {
double x2644 = x278[x2643];
fprintf((FILE *)x2640, "%lf\n", x2644);

}
double x2638 = x277 - x1;
double x2639 = x2637 - x277;
fprintf((FILE *)x2640, "run time: %lf %lf\n", x2638, x2639);
fclose((FILE*)x2640);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

