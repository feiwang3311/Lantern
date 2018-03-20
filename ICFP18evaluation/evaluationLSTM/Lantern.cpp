
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
double* x207 = (double*)myMalloc(1 * sizeof(double));
for(int x209=0; x209 < 1; x209++) {
x207[x209] = 0.1;

}
double* x213 = (double*)myMalloc(1 * sizeof(double));
for(int x214=0; x214 < 1; x214++) {
x213[x214] = 1.0E-8;

}
double* x218 = (double*)myMalloc(2500 * sizeof(double));
for(int x219=0; x219 < 2500; x219++) {
x218[x219] = 0.0;

}
double* x223 = (double*)myMalloc(1300 * sizeof(double));
for(int x224=0; x224 < 1300; x224++) {
x223[x224] = 0.0;

}
double* x228 = (double*)myMalloc(50 * sizeof(double));
for(int x229=0; x229 < 50; x229++) {
x228[x229] = 0.0;

}
double* x233 = (double*)myMalloc(2500 * sizeof(double));
for(int x234=0; x234 < 2500; x234++) {
x233[x234] = 0.0;

}
double* x238 = (double*)myMalloc(1300 * sizeof(double));
for(int x239=0; x239 < 1300; x239++) {
x238[x239] = 0.0;

}
double* x243 = (double*)myMalloc(50 * sizeof(double));
for(int x244=0; x244 < 50; x244++) {
x243[x244] = 0.0;

}
double* x248 = (double*)myMalloc(2500 * sizeof(double));
for(int x249=0; x249 < 2500; x249++) {
x248[x249] = 0.0;

}
double* x253 = (double*)myMalloc(1300 * sizeof(double));
for(int x254=0; x254 < 1300; x254++) {
x253[x254] = 0.0;

}
double* x258 = (double*)myMalloc(50 * sizeof(double));
for(int x259=0; x259 < 50; x259++) {
x258[x259] = 0.0;

}
double* x263 = (double*)myMalloc(2500 * sizeof(double));
for(int x264=0; x264 < 2500; x264++) {
x263[x264] = 0.0;

}
double* x268 = (double*)myMalloc(1300 * sizeof(double));
for(int x269=0; x269 < 1300; x269++) {
x268[x269] = 0.0;

}
double* x273 = (double*)myMalloc(50 * sizeof(double));
for(int x274=0; x274 < 50; x274++) {
x273[x274] = 0.0;

}
double* x278 = (double*)myMalloc(1300 * sizeof(double));
for(int x279=0; x279 < 1300; x279++) {
x278[x279] = 0.0;

}
double* x283 = (double*)myMalloc(26 * sizeof(double));
for(int x284=0; x284 < 26; x284++) {
x283[x284] = 0.0;

}
double x288 = ((double)clock() / CLOCKS_PER_SEC);
double* x289 = (double*)myMalloc(51 * sizeof(double));
int64_t x290 = (long)mallocAddr;
int32_t x291 = 0;
x291 -= 20;
double x293 = 70.0;
for(int x295=0; x295 < 5001; x295++) {
double* x331 = (double*)myMalloc(1 * sizeof(double));
int32_t* x308 = (int32_t*)myMalloc(20 * sizeof(int32_t));
int32_t* x309 = (int32_t*)myMalloc(20 * sizeof(int32_t));
function<void(int32_t,double**)> x346 = [&](int32_t x347,double** x348) {
int32_t x349 = x347;
bool x357 = x349 < 20;
if (x357) {
double* x358 = (double*)myMalloc(26 * sizeof(double));
for(int x359=0; x359 < 26; x359++) {
x358[x359] = 0.0;

}
int32_t x363 = x308[x349];
x358[x363] = 1.0;
double* x365 = (double*)myMalloc(26 * sizeof(double));
for(int x366=0; x366 < 26; x366++) {
x365[x366] = 0.0;

}
double* x370 = (double*)myMalloc(26 * sizeof(double));
for(int x371=0; x371 < 26; x371++) {
x370[x371] = 0.0;

}
int32_t x375 = x309[x349];
x370[x375] = 1.0;
double* x377 = (double*)myMalloc(26 * sizeof(double));
for(int x378=0; x378 < 26; x378++) {
x377[x378] = 0.0;

}
double* x382 = (double*)myMalloc(50 * sizeof(double));
double** x350 = x348;
double* x353 = x350[2];
for(int x383=0; x383 < 50; x383++) {
double x384 = 0.0;
int32_t x386 = 50 * x383;
for(int x385=0; x385 < 50; x385++) {
int32_t x387 = x385 + x386;
double x388 = x15[x387];
double x389 = x353[x385];
double x390 = x388 * x389;
x384 += x390;

}
double x394 = x384;
x382[x383] = x394;

}
double* x398 = (double*)myMalloc(50 * sizeof(double));
for(int x399=0; x399 < 50; x399++) {
x398[x399] = 0.0;

}
double* x403 = (double*)myMalloc(50 * sizeof(double));
for(int x404=0; x404 < 50; x404++) {
double x405 = 0.0;
int32_t x407 = 26 * x404;
for(int x406=0; x406 < 26; x406++) {
int32_t x408 = x406 + x407;
double x409 = x23[x408];
double x410 = x358[x406];
double x411 = x409 * x410;
x405 += x411;

}
double x415 = x405;
x403[x404] = x415;

}
double* x419 = (double*)myMalloc(50 * sizeof(double));
for(int x420=0; x420 < 50; x420++) {
x419[x420] = 0.0;

}
double* x424 = (double*)myMalloc(50 * sizeof(double));
for(int x425=0; x425 < 50; x425++) {
double x426 = x382[x425];
double x427 = x403[x425];
double x428 = x426 + x427;
x424[x425] = x428;

}
double* x432 = (double*)myMalloc(50 * sizeof(double));
for(int x433=0; x433 < 50; x433++) {
x432[x433] = 0.0;

}
double* x437 = (double*)myMalloc(50 * sizeof(double));
for(int x438=0; x438 < 50; x438++) {
double x439 = x424[x438];
double x440 = x31[x438];
double x441 = x439 + x440;
x437[x438] = x441;

}
double* x445 = (double*)myMalloc(50 * sizeof(double));
for(int x446=0; x446 < 50; x446++) {
x445[x446] = 0.0;

}
double* x450 = (double*)myMalloc(50 * sizeof(double));
for(int x451=0; x451 < 50; x451++) {
double x452 = x437[x451];
double x453 = -1.0 * x452;
double x454 = exp(x453);
double x455 = x454 + 1.0;
double x456 = 1.0 / x455;
x450[x451] = x456;

}
double* x460 = (double*)myMalloc(50 * sizeof(double));
for(int x461=0; x461 < 50; x461++) {
x460[x461] = 0.0;

}
double* x465 = (double*)myMalloc(50 * sizeof(double));
for(int x466=0; x466 < 50; x466++) {
double x467 = 0.0;
int32_t x469 = 50 * x466;
for(int x468=0; x468 < 50; x468++) {
int32_t x470 = x468 + x469;
double x471 = x37[x470];
double x472 = x353[x468];
double x473 = x471 * x472;
x467 += x473;

}
double x477 = x467;
x465[x466] = x477;

}
double* x481 = (double*)myMalloc(50 * sizeof(double));
for(int x482=0; x482 < 50; x482++) {
x481[x482] = 0.0;

}
double* x486 = (double*)myMalloc(50 * sizeof(double));
for(int x487=0; x487 < 50; x487++) {
double x488 = 0.0;
int32_t x490 = 26 * x487;
for(int x489=0; x489 < 26; x489++) {
int32_t x491 = x489 + x490;
double x492 = x44[x491];
double x493 = x358[x489];
double x494 = x492 * x493;
x488 += x494;

}
double x498 = x488;
x486[x487] = x498;

}
double* x502 = (double*)myMalloc(50 * sizeof(double));
for(int x503=0; x503 < 50; x503++) {
x502[x503] = 0.0;

}
double* x507 = (double*)myMalloc(50 * sizeof(double));
for(int x508=0; x508 < 50; x508++) {
double x509 = x465[x508];
double x510 = x486[x508];
double x511 = x509 + x510;
x507[x508] = x511;

}
double* x515 = (double*)myMalloc(50 * sizeof(double));
for(int x516=0; x516 < 50; x516++) {
x515[x516] = 0.0;

}
double* x520 = (double*)myMalloc(50 * sizeof(double));
for(int x521=0; x521 < 50; x521++) {
double x522 = x507[x521];
double x523 = x51[x521];
double x524 = x522 + x523;
x520[x521] = x524;

}
double* x528 = (double*)myMalloc(50 * sizeof(double));
for(int x529=0; x529 < 50; x529++) {
x528[x529] = 0.0;

}
double* x533 = (double*)myMalloc(50 * sizeof(double));
for(int x534=0; x534 < 50; x534++) {
double x535 = x520[x534];
double x536 = -1.0 * x535;
double x537 = exp(x536);
double x538 = x537 + 1.0;
double x539 = 1.0 / x538;
x533[x534] = x539;

}
double* x543 = (double*)myMalloc(50 * sizeof(double));
for(int x544=0; x544 < 50; x544++) {
x543[x544] = 0.0;

}
double* x548 = (double*)myMalloc(50 * sizeof(double));
for(int x549=0; x549 < 50; x549++) {
double x550 = 0.0;
int32_t x552 = 50 * x549;
for(int x551=0; x551 < 50; x551++) {
int32_t x553 = x551 + x552;
double x554 = x75[x553];
double x555 = x353[x551];
double x556 = x554 * x555;
x550 += x556;

}
double x560 = x550;
x548[x549] = x560;

}
double* x564 = (double*)myMalloc(50 * sizeof(double));
for(int x565=0; x565 < 50; x565++) {
x564[x565] = 0.0;

}
double* x569 = (double*)myMalloc(50 * sizeof(double));
for(int x570=0; x570 < 50; x570++) {
double x571 = 0.0;
int32_t x573 = 26 * x570;
for(int x572=0; x572 < 26; x572++) {
int32_t x574 = x572 + x573;
double x575 = x82[x574];
double x576 = x358[x572];
double x577 = x575 * x576;
x571 += x577;

}
double x581 = x571;
x569[x570] = x581;

}
double* x585 = (double*)myMalloc(50 * sizeof(double));
for(int x586=0; x586 < 50; x586++) {
x585[x586] = 0.0;

}
double* x590 = (double*)myMalloc(50 * sizeof(double));
for(int x591=0; x591 < 50; x591++) {
double x592 = x548[x591];
double x593 = x569[x591];
double x594 = x592 + x593;
x590[x591] = x594;

}
double* x598 = (double*)myMalloc(50 * sizeof(double));
for(int x599=0; x599 < 50; x599++) {
x598[x599] = 0.0;

}
double* x603 = (double*)myMalloc(50 * sizeof(double));
for(int x604=0; x604 < 50; x604++) {
double x605 = x590[x604];
double x606 = x89[x604];
double x607 = x605 + x606;
x603[x604] = x607;

}
double* x611 = (double*)myMalloc(50 * sizeof(double));
for(int x612=0; x612 < 50; x612++) {
x611[x612] = 0.0;

}
double* x616 = (double*)myMalloc(50 * sizeof(double));
for(int x617=0; x617 < 50; x617++) {
double x618 = x603[x617];
double x619 = -1.0 * x618;
double x620 = exp(x619);
double x621 = x620 + 1.0;
double x622 = 1.0 / x621;
x616[x617] = x622;

}
double* x626 = (double*)myMalloc(50 * sizeof(double));
for(int x627=0; x627 < 50; x627++) {
x626[x627] = 0.0;

}
double* x631 = (double*)myMalloc(50 * sizeof(double));
for(int x632=0; x632 < 50; x632++) {
double x633 = 0.0;
int32_t x635 = 50 * x632;
for(int x634=0; x634 < 50; x634++) {
int32_t x636 = x634 + x635;
double x637 = x56[x636];
double x638 = x353[x634];
double x639 = x637 * x638;
x633 += x639;

}
double x643 = x633;
x631[x632] = x643;

}
double* x647 = (double*)myMalloc(50 * sizeof(double));
for(int x648=0; x648 < 50; x648++) {
x647[x648] = 0.0;

}
double* x652 = (double*)myMalloc(50 * sizeof(double));
for(int x653=0; x653 < 50; x653++) {
double x654 = 0.0;
int32_t x656 = 26 * x653;
for(int x655=0; x655 < 26; x655++) {
int32_t x657 = x655 + x656;
double x658 = x63[x657];
double x659 = x358[x655];
double x660 = x658 * x659;
x654 += x660;

}
double x664 = x654;
x652[x653] = x664;

}
double* x668 = (double*)myMalloc(50 * sizeof(double));
for(int x669=0; x669 < 50; x669++) {
x668[x669] = 0.0;

}
double* x673 = (double*)myMalloc(50 * sizeof(double));
for(int x674=0; x674 < 50; x674++) {
double x675 = x631[x674];
double x676 = x652[x674];
double x677 = x675 + x676;
x673[x674] = x677;

}
double* x681 = (double*)myMalloc(50 * sizeof(double));
for(int x682=0; x682 < 50; x682++) {
x681[x682] = 0.0;

}
double* x686 = (double*)myMalloc(50 * sizeof(double));
for(int x687=0; x687 < 50; x687++) {
double x688 = x673[x687];
double x689 = x70[x687];
double x690 = x688 + x689;
x686[x687] = x690;

}
double* x694 = (double*)myMalloc(50 * sizeof(double));
for(int x695=0; x695 < 50; x695++) {
x694[x695] = 0.0;

}
double* x699 = (double*)myMalloc(50 * sizeof(double));
for(int x700=0; x700 < 50; x700++) {
double x701 = x686[x700];
double x702 = tanh(x701);
x699[x700] = x702;

}
double* x706 = (double*)myMalloc(50 * sizeof(double));
for(int x707=0; x707 < 50; x707++) {
x706[x707] = 0.0;

}
double* x711 = (double*)myMalloc(50 * sizeof(double));
double* x355 = x350[4];
for(int x712=0; x712 < 50; x712++) {
double x713 = x450[x712];
double x714 = x355[x712];
double x715 = x713 * x714;
x711[x712] = x715;

}
double* x719 = (double*)myMalloc(50 * sizeof(double));
for(int x720=0; x720 < 50; x720++) {
x719[x720] = 0.0;

}
double* x724 = (double*)myMalloc(50 * sizeof(double));
for(int x725=0; x725 < 50; x725++) {
double x726 = x533[x725];
double x727 = x699[x725];
double x728 = x726 * x727;
x724[x725] = x728;

}
double* x732 = (double*)myMalloc(50 * sizeof(double));
for(int x733=0; x733 < 50; x733++) {
x732[x733] = 0.0;

}
double* x737 = (double*)myMalloc(50 * sizeof(double));
for(int x738=0; x738 < 50; x738++) {
double x739 = x711[x738];
double x740 = x724[x738];
double x741 = x739 + x740;
x737[x738] = x741;

}
double* x745 = (double*)myMalloc(50 * sizeof(double));
for(int x746=0; x746 < 50; x746++) {
x745[x746] = 0.0;

}
double* x750 = (double*)myMalloc(50 * sizeof(double));
for(int x751=0; x751 < 50; x751++) {
double x752 = x737[x751];
double x753 = tanh(x752);
x750[x751] = x753;

}
double* x757 = (double*)myMalloc(50 * sizeof(double));
for(int x758=0; x758 < 50; x758++) {
x757[x758] = 0.0;

}
double* x762 = (double*)myMalloc(50 * sizeof(double));
for(int x763=0; x763 < 50; x763++) {
double x764 = x616[x763];
double x765 = x750[x763];
double x766 = x764 * x765;
x762[x763] = x766;

}
double* x770 = (double*)myMalloc(50 * sizeof(double));
for(int x771=0; x771 < 50; x771++) {
x770[x771] = 0.0;

}
double* x775 = (double*)myMalloc(26 * sizeof(double));
for(int x776=0; x776 < 26; x776++) {
double x777 = 0.0;
int32_t x779 = 50 * x776;
for(int x778=0; x778 < 50; x778++) {
int32_t x780 = x778 + x779;
double x781 = x94[x780];
double x782 = x762[x778];
double x783 = x781 * x782;
x777 += x783;

}
double x787 = x777;
x775[x776] = x787;

}
double* x791 = (double*)myMalloc(26 * sizeof(double));
for(int x792=0; x792 < 26; x792++) {
x791[x792] = 0.0;

}
double* x796 = (double*)myMalloc(26 * sizeof(double));
for(int x797=0; x797 < 26; x797++) {
double x798 = x775[x797];
double x799 = x101[x797];
double x800 = x798 + x799;
x796[x797] = x800;

}
double* x804 = (double*)myMalloc(26 * sizeof(double));
for(int x805=0; x805 < 26; x805++) {
x804[x805] = 0.0;

}
double* x809 = (double*)myMalloc(26 * sizeof(double));
for(int x810=0; x810 < 26; x810++) {
double x811 = x796[x810];
double x812 = exp(x811);
x809[x810] = x812;

}
double* x816 = (double*)myMalloc(26 * sizeof(double));
for(int x817=0; x817 < 26; x817++) {
x816[x817] = 0.0;

}
double x821 = 0.0;
for(int x822=0; x822 < 26; x822++) {
double x823 = x809[x822];
x821 += x823;

}
double* x827 = (double*)myMalloc(1 * sizeof(double));
double x828 = x821;
x827[0] = x828;
double* x830 = (double*)myMalloc(1 * sizeof(double));
for(int x831=0; x831 < 1; x831++) {
x830[x831] = 0.0;

}
double* x835 = (double*)myMalloc(26 * sizeof(double));
for(int x836=0; x836 < 26; x836++) {
double x837 = x809[x836];
double x838 = x827[0];
double x839 = x837 / x838;
x835[x836] = x839;

}
double* x843 = (double*)myMalloc(26 * sizeof(double));
for(int x844=0; x844 < 26; x844++) {
x843[x844] = 0.0;

}
double* x848 = (double*)myMalloc(1 * sizeof(double));
for(int x849=0; x849 < 1; x849++) {
double x850 = 0.0;
int32_t x852 = 26 * x849;
for(int x851=0; x851 < 26; x851++) {
int32_t x853 = x851 + x852;
double x854 = x835[x853];
double x855 = x370[x851];
double x856 = x854 * x855;
x850 += x856;

}
double x860 = x850;
x848[x849] = x860;

}
double* x864 = (double*)myMalloc(1 * sizeof(double));
for(int x865=0; x865 < 1; x865++) {
x864[x865] = 0.0;

}
double* x869 = (double*)myMalloc(1 * sizeof(double));
for(int x870=0; x870 < 1; x870++) {
double x871 = x848[x870];
double x872 = log(x871);
x869[x870] = x872;

}
double* x876 = (double*)myMalloc(1 * sizeof(double));
for(int x877=0; x877 < 1; x877++) {
x876[x877] = 0.0;

}
double* x881 = (double*)myMalloc(1 * sizeof(double));
double* x351 = x350[0];
for(int x882=0; x882 < 1; x882++) {
double x884 = x869[x882];
double x883 = x351[x882];
double x885 = x883 - x884;
x881[x882] = x885;

}
double* x889 = (double*)myMalloc(1 * sizeof(double));
for(int x890=0; x890 < 1; x890++) {
x889[x890] = 0.0;

}
double** x895 = (double**)myMalloc(6 * sizeof(double*));
x895[0] = x881;
x895[1] = x889;
x895[2] = x762;
x895[3] = x770;
x895[4] = x737;
x895[5] = x745;
int32_t x894 = x349 + 1;
x346(x894,x895);
double* x352 = x350[1];
for(int x904=0; x904 < 1; x904++) {
double x906 = x889[x904];
double x905 = x352[x904];
double x907 = x905 + x906;
x352[x904] = x907;

}
for(int x911=0; x911 < 1; x911++) {
double x912 = x876[x911];
double x913 = x889[x911];
double x914 = x912 - x913;
x876[x911] = x914;

}
for(int x918=0; x918 < 1; x918++) {
double x919 = x864[0];
double x920 = x876[0];
double x921 = x848[0];
double x922 = x920 / x921;
double x923 = x919 + x922;
x864[0] = x923;

}
for(int x927=0; x927 < 1; x927++) {
int32_t x929 = 26 * x927;
for(int x928=0; x928 < 26; x928++) {
int32_t x930 = x929 + x928;
double x931 = x843[x930];
double x932 = x370[x928];
double x933 = x864[x927];
double x934 = x932 * x933;
double x935 = x931 + x934;
x843[x930] = x935;

}

}
for(int x941=0; x941 < 1; x941++) {
int32_t x944 = 26 * x941;
for(int x942=0; x942 < 26; x942++) {
double x943 = x377[x942];
int32_t x945 = x944 + x942;
double x946 = x835[x945];
double x947 = x864[x941];
double x948 = x946 * x947;
double x949 = x943 + x948;
x377[x942] = x949;

}

}
for(int x955=0; x955 < 26; x955++) {
double x956 = x816[x955];
double x957 = x843[x955];
double x958 = x827[0];
double x959 = x957 / x958;
double x960 = x956 + x959;
x816[x955] = x960;

}
for(int x964=0; x964 < 26; x964++) {
double x965 = x830[0];
double x966 = x809[x964];
double x967 = x843[x964];
double x969 = x827[0];
double x968 = x966 * x967;
double x970 = x969 * x969;
double x971 = x968 / x970;
double x972 = x965 - x971;
x830[0] = x972;

}
for(int x976=0; x976 < 26; x976++) {
double x977 = x816[x976];
double x978 = x830[0];
double x979 = x977 + x978;
x816[x976] = x979;

}
for(int x983=0; x983 < 26; x983++) {
double x984 = x804[x983];
double x985 = x809[x983];
double x986 = x816[x983];
double x987 = x985 * x986;
double x988 = x984 + x987;
x804[x983] = x988;

}
for(int x992=0; x992 < 26; x992++) {
double x993 = x791[x992];
double x994 = x804[x992];
double x995 = x993 + x994;
x791[x992] = x995;

}
for(int x999=0; x999 < 26; x999++) {
double x1000 = x192[x999];
double x1001 = x804[x999];
double x1002 = x1000 + x1001;
x192[x999] = x1002;

}
for(int x1006=0; x1006 < 26; x1006++) {
int32_t x1008 = 50 * x1006;
for(int x1007=0; x1007 < 50; x1007++) {
int32_t x1009 = x1008 + x1007;
double x1010 = x187[x1009];
double x1011 = x762[x1007];
double x1012 = x791[x1006];
double x1013 = x1011 * x1012;
double x1014 = x1010 + x1013;
x187[x1009] = x1014;

}

}
for(int x1020=0; x1020 < 26; x1020++) {
int32_t x1023 = 50 * x1020;
for(int x1021=0; x1021 < 50; x1021++) {
double x1022 = x770[x1021];
int32_t x1024 = x1023 + x1021;
double x1025 = x94[x1024];
double x1026 = x791[x1020];
double x1027 = x1025 * x1026;
double x1028 = x1022 + x1027;
x770[x1021] = x1028;

}

}
for(int x1034=0; x1034 < 50; x1034++) {
double x1035 = x626[x1034];
double x1036 = x750[x1034];
double x1037 = x770[x1034];
double x1038 = x1036 * x1037;
double x1039 = x1035 + x1038;
x626[x1034] = x1039;

}
for(int x1043=0; x1043 < 50; x1043++) {
double x1044 = x757[x1043];
double x1045 = x616[x1043];
double x1046 = x770[x1043];
double x1047 = x1045 * x1046;
double x1048 = x1044 + x1047;
x757[x1043] = x1048;

}
for(int x1052=0; x1052 < 50; x1052++) {
double x1053 = x745[x1052];
double x1054 = x750[x1052];
double x1057 = x757[x1052];
double x1055 = x1054 * x1054;
double x1056 = 1.0 - x1055;
double x1058 = x1056 * x1057;
double x1059 = x1053 + x1058;
x745[x1052] = x1059;

}
for(int x1063=0; x1063 < 50; x1063++) {
double x1064 = x719[x1063];
double x1065 = x745[x1063];
double x1066 = x1064 + x1065;
x719[x1063] = x1066;

}
for(int x1070=0; x1070 < 50; x1070++) {
double x1071 = x732[x1070];
double x1072 = x745[x1070];
double x1073 = x1071 + x1072;
x732[x1070] = x1073;

}
for(int x1077=0; x1077 < 50; x1077++) {
double x1078 = x543[x1077];
double x1079 = x699[x1077];
double x1080 = x732[x1077];
double x1081 = x1079 * x1080;
double x1082 = x1078 + x1081;
x543[x1077] = x1082;

}
for(int x1086=0; x1086 < 50; x1086++) {
double x1087 = x706[x1086];
double x1088 = x533[x1086];
double x1089 = x732[x1086];
double x1090 = x1088 * x1089;
double x1091 = x1087 + x1090;
x706[x1086] = x1091;

}
for(int x1095=0; x1095 < 50; x1095++) {
double x1096 = x460[x1095];
double x1098 = x719[x1095];
double x1097 = x355[x1095];
double x1099 = x1097 * x1098;
double x1100 = x1096 + x1099;
x460[x1095] = x1100;

}
double* x356 = x350[5];
for(int x1104=0; x1104 < 50; x1104++) {
double x1106 = x450[x1104];
double x1107 = x719[x1104];
double x1105 = x356[x1104];
double x1108 = x1106 * x1107;
double x1109 = x1105 + x1108;
x356[x1104] = x1109;

}
for(int x1113=0; x1113 < 50; x1113++) {
double x1114 = x694[x1113];
double x1115 = x699[x1113];
double x1118 = x706[x1113];
double x1116 = x1115 * x1115;
double x1117 = 1.0 - x1116;
double x1119 = x1117 * x1118;
double x1120 = x1114 + x1119;
x694[x1113] = x1120;

}
for(int x1124=0; x1124 < 50; x1124++) {
double x1125 = x681[x1124];
double x1126 = x694[x1124];
double x1127 = x1125 + x1126;
x681[x1124] = x1127;

}
for(int x1131=0; x1131 < 50; x1131++) {
double x1132 = x167[x1131];
double x1133 = x694[x1131];
double x1134 = x1132 + x1133;
x167[x1131] = x1134;

}
for(int x1138=0; x1138 < 50; x1138++) {
double x1139 = x647[x1138];
double x1140 = x681[x1138];
double x1141 = x1139 + x1140;
x647[x1138] = x1141;

}
for(int x1145=0; x1145 < 50; x1145++) {
double x1146 = x668[x1145];
double x1147 = x681[x1145];
double x1148 = x1146 + x1147;
x668[x1145] = x1148;

}
for(int x1152=0; x1152 < 50; x1152++) {
int32_t x1154 = 26 * x1152;
for(int x1153=0; x1153 < 26; x1153++) {
int32_t x1155 = x1154 + x1153;
double x1156 = x162[x1155];
double x1157 = x358[x1153];
double x1158 = x668[x1152];
double x1159 = x1157 * x1158;
double x1160 = x1156 + x1159;
x162[x1155] = x1160;

}

}
for(int x1166=0; x1166 < 50; x1166++) {
int32_t x1169 = 26 * x1166;
for(int x1167=0; x1167 < 26; x1167++) {
double x1168 = x365[x1167];
int32_t x1170 = x1169 + x1167;
double x1171 = x63[x1170];
double x1172 = x668[x1166];
double x1173 = x1171 * x1172;
double x1174 = x1168 + x1173;
x365[x1167] = x1174;

}

}
for(int x1180=0; x1180 < 50; x1180++) {
int32_t x1182 = 50 * x1180;
for(int x1181=0; x1181 < 50; x1181++) {
int32_t x1183 = x1182 + x1181;
double x1184 = x157[x1183];
double x1186 = x647[x1180];
double x1185 = x353[x1181];
double x1187 = x1185 * x1186;
double x1188 = x1184 + x1187;
x157[x1183] = x1188;

}

}
double* x354 = x350[3];
for(int x1194=0; x1194 < 50; x1194++) {
int32_t x1197 = 50 * x1194;
for(int x1195=0; x1195 < 50; x1195++) {
int32_t x1198 = x1197 + x1195;
double x1199 = x56[x1198];
double x1200 = x647[x1194];
double x1196 = x354[x1195];
double x1201 = x1199 * x1200;
double x1202 = x1196 + x1201;
x354[x1195] = x1202;

}

}
for(int x1208=0; x1208 < 50; x1208++) {
double x1209 = x611[x1208];
double x1210 = x616[x1208];
double x1213 = x626[x1208];
double x1211 = 1.0 - x1210;
double x1212 = x1211 * x1210;
double x1214 = x1212 * x1213;
double x1215 = x1209 + x1214;
x611[x1208] = x1215;

}
for(int x1219=0; x1219 < 50; x1219++) {
double x1220 = x598[x1219];
double x1221 = x611[x1219];
double x1222 = x1220 + x1221;
x598[x1219] = x1222;

}
for(int x1226=0; x1226 < 50; x1226++) {
double x1227 = x182[x1226];
double x1228 = x611[x1226];
double x1229 = x1227 + x1228;
x182[x1226] = x1229;

}
for(int x1233=0; x1233 < 50; x1233++) {
double x1234 = x564[x1233];
double x1235 = x598[x1233];
double x1236 = x1234 + x1235;
x564[x1233] = x1236;

}
for(int x1240=0; x1240 < 50; x1240++) {
double x1241 = x585[x1240];
double x1242 = x598[x1240];
double x1243 = x1241 + x1242;
x585[x1240] = x1243;

}
for(int x1247=0; x1247 < 50; x1247++) {
int32_t x1249 = 26 * x1247;
for(int x1248=0; x1248 < 26; x1248++) {
int32_t x1250 = x1249 + x1248;
double x1251 = x177[x1250];
double x1252 = x358[x1248];
double x1253 = x585[x1247];
double x1254 = x1252 * x1253;
double x1255 = x1251 + x1254;
x177[x1250] = x1255;

}

}
for(int x1261=0; x1261 < 50; x1261++) {
int32_t x1264 = 26 * x1261;
for(int x1262=0; x1262 < 26; x1262++) {
double x1263 = x365[x1262];
int32_t x1265 = x1264 + x1262;
double x1266 = x82[x1265];
double x1267 = x585[x1261];
double x1268 = x1266 * x1267;
double x1269 = x1263 + x1268;
x365[x1262] = x1269;

}

}
for(int x1275=0; x1275 < 50; x1275++) {
int32_t x1277 = 50 * x1275;
for(int x1276=0; x1276 < 50; x1276++) {
int32_t x1278 = x1277 + x1276;
double x1279 = x172[x1278];
double x1281 = x564[x1275];
double x1280 = x353[x1276];
double x1282 = x1280 * x1281;
double x1283 = x1279 + x1282;
x172[x1278] = x1283;

}

}
for(int x1289=0; x1289 < 50; x1289++) {
int32_t x1292 = 50 * x1289;
for(int x1290=0; x1290 < 50; x1290++) {
int32_t x1293 = x1292 + x1290;
double x1294 = x75[x1293];
double x1295 = x564[x1289];
double x1291 = x354[x1290];
double x1296 = x1294 * x1295;
double x1297 = x1291 + x1296;
x354[x1290] = x1297;

}

}
for(int x1303=0; x1303 < 50; x1303++) {
double x1304 = x528[x1303];
double x1305 = x533[x1303];
double x1308 = x543[x1303];
double x1306 = 1.0 - x1305;
double x1307 = x1306 * x1305;
double x1309 = x1307 * x1308;
double x1310 = x1304 + x1309;
x528[x1303] = x1310;

}
for(int x1314=0; x1314 < 50; x1314++) {
double x1315 = x515[x1314];
double x1316 = x528[x1314];
double x1317 = x1315 + x1316;
x515[x1314] = x1317;

}
for(int x1321=0; x1321 < 50; x1321++) {
double x1322 = x152[x1321];
double x1323 = x528[x1321];
double x1324 = x1322 + x1323;
x152[x1321] = x1324;

}
for(int x1328=0; x1328 < 50; x1328++) {
double x1329 = x481[x1328];
double x1330 = x515[x1328];
double x1331 = x1329 + x1330;
x481[x1328] = x1331;

}
for(int x1335=0; x1335 < 50; x1335++) {
double x1336 = x502[x1335];
double x1337 = x515[x1335];
double x1338 = x1336 + x1337;
x502[x1335] = x1338;

}
for(int x1342=0; x1342 < 50; x1342++) {
int32_t x1344 = 26 * x1342;
for(int x1343=0; x1343 < 26; x1343++) {
int32_t x1345 = x1344 + x1343;
double x1346 = x147[x1345];
double x1347 = x358[x1343];
double x1348 = x502[x1342];
double x1349 = x1347 * x1348;
double x1350 = x1346 + x1349;
x147[x1345] = x1350;

}

}
for(int x1356=0; x1356 < 50; x1356++) {
int32_t x1359 = 26 * x1356;
for(int x1357=0; x1357 < 26; x1357++) {
double x1358 = x365[x1357];
int32_t x1360 = x1359 + x1357;
double x1361 = x44[x1360];
double x1362 = x502[x1356];
double x1363 = x1361 * x1362;
double x1364 = x1358 + x1363;
x365[x1357] = x1364;

}

}
for(int x1370=0; x1370 < 50; x1370++) {
int32_t x1372 = 50 * x1370;
for(int x1371=0; x1371 < 50; x1371++) {
int32_t x1373 = x1372 + x1371;
double x1374 = x142[x1373];
double x1376 = x481[x1370];
double x1375 = x353[x1371];
double x1377 = x1375 * x1376;
double x1378 = x1374 + x1377;
x142[x1373] = x1378;

}

}
for(int x1384=0; x1384 < 50; x1384++) {
int32_t x1387 = 50 * x1384;
for(int x1385=0; x1385 < 50; x1385++) {
int32_t x1388 = x1387 + x1385;
double x1389 = x37[x1388];
double x1390 = x481[x1384];
double x1386 = x354[x1385];
double x1391 = x1389 * x1390;
double x1392 = x1386 + x1391;
x354[x1385] = x1392;

}

}
for(int x1398=0; x1398 < 50; x1398++) {
double x1399 = x445[x1398];
double x1400 = x450[x1398];
double x1403 = x460[x1398];
double x1401 = 1.0 - x1400;
double x1402 = x1401 * x1400;
double x1404 = x1402 * x1403;
double x1405 = x1399 + x1404;
x445[x1398] = x1405;

}
for(int x1409=0; x1409 < 50; x1409++) {
double x1410 = x432[x1409];
double x1411 = x445[x1409];
double x1412 = x1410 + x1411;
x432[x1409] = x1412;

}
for(int x1416=0; x1416 < 50; x1416++) {
double x1417 = x137[x1416];
double x1418 = x445[x1416];
double x1419 = x1417 + x1418;
x137[x1416] = x1419;

}
for(int x1423=0; x1423 < 50; x1423++) {
double x1424 = x398[x1423];
double x1425 = x432[x1423];
double x1426 = x1424 + x1425;
x398[x1423] = x1426;

}
for(int x1430=0; x1430 < 50; x1430++) {
double x1431 = x419[x1430];
double x1432 = x432[x1430];
double x1433 = x1431 + x1432;
x419[x1430] = x1433;

}
for(int x1437=0; x1437 < 50; x1437++) {
int32_t x1439 = 26 * x1437;
for(int x1438=0; x1438 < 26; x1438++) {
int32_t x1440 = x1439 + x1438;
double x1441 = x132[x1440];
double x1442 = x358[x1438];
double x1443 = x419[x1437];
double x1444 = x1442 * x1443;
double x1445 = x1441 + x1444;
x132[x1440] = x1445;

}

}
for(int x1451=0; x1451 < 50; x1451++) {
int32_t x1454 = 26 * x1451;
for(int x1452=0; x1452 < 26; x1452++) {
double x1453 = x365[x1452];
int32_t x1455 = x1454 + x1452;
double x1456 = x23[x1455];
double x1457 = x419[x1451];
double x1458 = x1456 * x1457;
double x1459 = x1453 + x1458;
x365[x1452] = x1459;

}

}
for(int x1465=0; x1465 < 50; x1465++) {
int32_t x1467 = 50 * x1465;
for(int x1466=0; x1466 < 50; x1466++) {
int32_t x1468 = x1467 + x1466;
double x1469 = x127[x1468];
double x1471 = x398[x1465];
double x1470 = x353[x1466];
double x1472 = x1470 * x1471;
double x1473 = x1469 + x1472;
x127[x1468] = x1473;

}

}
for(int x1479=0; x1479 < 50; x1479++) {
int32_t x1482 = 50 * x1479;
for(int x1480=0; x1480 < 50; x1480++) {
int32_t x1483 = x1482 + x1480;
double x1484 = x15[x1483];
double x1485 = x398[x1479];
double x1481 = x354[x1480];
double x1486 = x1484 * x1485;
double x1487 = x1481 + x1486;
x354[x1480] = x1487;

}

}
} else {
double** x350 = x348;
double* x353 = x350[2];
for(int x1494=0; x1494 < 50; x1494++) {
double x1495 = x353[x1494];
x117[x1494] = x1495;

}
double* x355 = x350[4];
for(int x1499=0; x1499 < 50; x1499++) {
double x1500 = x355[x1499];
x122[x1499] = x1500;

}
double* x352 = x350[1];
for(int x1504=0; x1504 < 1; x1504++) {
x352[x1504] = 1.0;

}
double* x351 = x350[0];
for(int x1508=0; x1508 < 1; x1508++) {
double x1509 = x351[x1508];
x331[x1508] = x1509;

}
}
};
x291 += 20;
int32_t x297 = x291;
int32_t x298 = x297 + 20;
int32_t x299 = x298 + 1;
bool x300 = x299 >= x3;
if (x300) {
x291 = 0;
for(int x302=0; x302 < 50; x302++) {
x107[x302] = 0.0;

}
} else {
}
for(int x311=0; x311 < 20; x311++) {
int32_t x312 = x291;
int32_t x313 = x312 + x311;
int32_t x314 = x6[x313];
x308[x311] = x314;
int32_t x316 = x313 + 1;
int32_t x317 = x6[x316];
x309[x311] = x317;

}
double* x321 = (double*)myMalloc(1 * sizeof(double));
for(int x322=0; x322 < 1; x322++) {
x321[x322] = 0.0;

}
double* x326 = (double*)myMalloc(1 * sizeof(double));
for(int x327=0; x327 < 1; x327++) {
x326[x327] = 0.0;

}
for(int x332=0; x332 < 1; x332++) {
x331[x332] = 0.0;

}
double* x336 = (double*)myMalloc(1 * sizeof(double));
for(int x337=0; x337 < 1; x337++) {
x336[x337] = 0.0;

}
double* x341 = (double*)myMalloc(1 * sizeof(double));
for(int x342=0; x342 < 1; x342++) {
x341[x342] = 0.0;

}
double** x1516 = (double**)myMalloc(6 * sizeof(double*));
x1516[0] = x336;
x1516[1] = x341;
x1516[2] = x107;
x1516[3] = x197;
x1516[4] = x112;
x1516[5] = x202;
x346(0,x1516);
double x1525 = x331[0];
double x1526 = x293;
double x1527 = x1526 * 0.9;
double x1528 = x1525 * 0.1;
double x1529 = x1527 + x1528;
x293 = x1529;
int32_t x1531 = x295 % 100;
bool x1532 = x1531 == 0;
if (x1532) {
double x1533 = x293;
printf("iter %d, loss %f\n",x295,x1533);
int32_t x1535 = x295 / 100;
x289[x1535] = x1533;
} else {
}
for(int x1539=0; x1539 < 2500; x1539++) {
double x1540 = x127[x1539];
bool x1541 = x1540 > 5.0;
if (x1541) {
x127[x1539] = 5.0;
} else {
}
bool x1545 = x1540 < -5.0;
if (x1545) {
x127[x1539] = -5.0;
} else {
}

}
double* x1551 = (double*)myMalloc(2500 * sizeof(double));
for(int x1552=0; x1552 < 2500; x1552++) {
double x1553 = x127[x1552];
double x1554 = x1553 * x1553;
x1551[x1552] = x1554;

}
for(int x1558=0; x1558 < 2500; x1558++) {
double x1559 = x218[x1558];
double x1560 = x1551[x1558];
double x1561 = x1559 + x1560;
x218[x1558] = x1561;

}
double* x1565 = (double*)myMalloc(2500 * sizeof(double));
for(int x1566=0; x1566 < 2500; x1566++) {
double x1567 = x127[x1566];
double x1568 = x207[0];
double x1569 = x1567 * x1568;
x1565[x1566] = x1569;

}
double* x1573 = (double*)myMalloc(2500 * sizeof(double));
for(int x1574=0; x1574 < 2500; x1574++) {
double x1575 = x218[x1574];
double x1576 = x213[0];
double x1577 = x1575 + x1576;
x1573[x1574] = x1577;

}
double* x1581 = (double*)myMalloc(2500 * sizeof(double));
for(int x1582=0; x1582 < 2500; x1582++) {
double x1583 = x1573[x1582];
double x1584 = sqrt(x1583);
x1581[x1582] = x1584;

}
double* x1588 = (double*)myMalloc(2500 * sizeof(double));
for(int x1589=0; x1589 < 2500; x1589++) {
double x1590 = x1565[x1589];
double x1591 = x1581[x1589];
double x1592 = x1590 / x1591;
x1588[x1589] = x1592;

}
for(int x1596=0; x1596 < 2500; x1596++) {
double x1597 = x15[x1596];
double x1598 = x1588[x1596];
double x1599 = x1597 - x1598;
x15[x1596] = x1599;

}
for(int x1603=0; x1603 < 2500; x1603++) {
x127[x1603] = 0.0;

}
for(int x1607=0; x1607 < 1300; x1607++) {
double x1608 = x132[x1607];
bool x1609 = x1608 > 5.0;
if (x1609) {
x132[x1607] = 5.0;
} else {
}
bool x1613 = x1608 < -5.0;
if (x1613) {
x132[x1607] = -5.0;
} else {
}

}
double* x1619 = (double*)myMalloc(1300 * sizeof(double));
for(int x1620=0; x1620 < 1300; x1620++) {
double x1621 = x132[x1620];
double x1622 = x1621 * x1621;
x1619[x1620] = x1622;

}
for(int x1626=0; x1626 < 1300; x1626++) {
double x1627 = x223[x1626];
double x1628 = x1619[x1626];
double x1629 = x1627 + x1628;
x223[x1626] = x1629;

}
double* x1633 = (double*)myMalloc(1300 * sizeof(double));
for(int x1634=0; x1634 < 1300; x1634++) {
double x1635 = x132[x1634];
double x1636 = x207[0];
double x1637 = x1635 * x1636;
x1633[x1634] = x1637;

}
double* x1641 = (double*)myMalloc(1300 * sizeof(double));
for(int x1642=0; x1642 < 1300; x1642++) {
double x1643 = x223[x1642];
double x1644 = x213[0];
double x1645 = x1643 + x1644;
x1641[x1642] = x1645;

}
double* x1649 = (double*)myMalloc(1300 * sizeof(double));
for(int x1650=0; x1650 < 1300; x1650++) {
double x1651 = x1641[x1650];
double x1652 = sqrt(x1651);
x1649[x1650] = x1652;

}
double* x1656 = (double*)myMalloc(1300 * sizeof(double));
for(int x1657=0; x1657 < 1300; x1657++) {
double x1658 = x1633[x1657];
double x1659 = x1649[x1657];
double x1660 = x1658 / x1659;
x1656[x1657] = x1660;

}
for(int x1664=0; x1664 < 1300; x1664++) {
double x1665 = x23[x1664];
double x1666 = x1656[x1664];
double x1667 = x1665 - x1666;
x23[x1664] = x1667;

}
for(int x1671=0; x1671 < 1300; x1671++) {
x132[x1671] = 0.0;

}
for(int x1675=0; x1675 < 50; x1675++) {
double x1676 = x137[x1675];
bool x1677 = x1676 > 5.0;
if (x1677) {
x137[x1675] = 5.0;
} else {
}
bool x1681 = x1676 < -5.0;
if (x1681) {
x137[x1675] = -5.0;
} else {
}

}
double* x1687 = (double*)myMalloc(50 * sizeof(double));
for(int x1688=0; x1688 < 50; x1688++) {
double x1689 = x137[x1688];
double x1690 = x1689 * x1689;
x1687[x1688] = x1690;

}
for(int x1694=0; x1694 < 50; x1694++) {
double x1695 = x228[x1694];
double x1696 = x1687[x1694];
double x1697 = x1695 + x1696;
x228[x1694] = x1697;

}
double* x1701 = (double*)myMalloc(50 * sizeof(double));
for(int x1702=0; x1702 < 50; x1702++) {
double x1703 = x137[x1702];
double x1704 = x207[0];
double x1705 = x1703 * x1704;
x1701[x1702] = x1705;

}
double* x1709 = (double*)myMalloc(50 * sizeof(double));
for(int x1710=0; x1710 < 50; x1710++) {
double x1711 = x228[x1710];
double x1712 = x213[0];
double x1713 = x1711 + x1712;
x1709[x1710] = x1713;

}
double* x1717 = (double*)myMalloc(50 * sizeof(double));
for(int x1718=0; x1718 < 50; x1718++) {
double x1719 = x1709[x1718];
double x1720 = sqrt(x1719);
x1717[x1718] = x1720;

}
double* x1724 = (double*)myMalloc(50 * sizeof(double));
for(int x1725=0; x1725 < 50; x1725++) {
double x1726 = x1701[x1725];
double x1727 = x1717[x1725];
double x1728 = x1726 / x1727;
x1724[x1725] = x1728;

}
for(int x1732=0; x1732 < 50; x1732++) {
double x1733 = x31[x1732];
double x1734 = x1724[x1732];
double x1735 = x1733 - x1734;
x31[x1732] = x1735;

}
for(int x1739=0; x1739 < 50; x1739++) {
x137[x1739] = 0.0;

}
for(int x1743=0; x1743 < 2500; x1743++) {
double x1744 = x142[x1743];
bool x1745 = x1744 > 5.0;
if (x1745) {
x142[x1743] = 5.0;
} else {
}
bool x1749 = x1744 < -5.0;
if (x1749) {
x142[x1743] = -5.0;
} else {
}

}
double* x1755 = (double*)myMalloc(2500 * sizeof(double));
for(int x1756=0; x1756 < 2500; x1756++) {
double x1757 = x142[x1756];
double x1758 = x1757 * x1757;
x1755[x1756] = x1758;

}
for(int x1762=0; x1762 < 2500; x1762++) {
double x1763 = x233[x1762];
double x1764 = x1755[x1762];
double x1765 = x1763 + x1764;
x233[x1762] = x1765;

}
double* x1769 = (double*)myMalloc(2500 * sizeof(double));
for(int x1770=0; x1770 < 2500; x1770++) {
double x1771 = x142[x1770];
double x1772 = x207[0];
double x1773 = x1771 * x1772;
x1769[x1770] = x1773;

}
double* x1777 = (double*)myMalloc(2500 * sizeof(double));
for(int x1778=0; x1778 < 2500; x1778++) {
double x1779 = x233[x1778];
double x1780 = x213[0];
double x1781 = x1779 + x1780;
x1777[x1778] = x1781;

}
double* x1785 = (double*)myMalloc(2500 * sizeof(double));
for(int x1786=0; x1786 < 2500; x1786++) {
double x1787 = x1777[x1786];
double x1788 = sqrt(x1787);
x1785[x1786] = x1788;

}
double* x1792 = (double*)myMalloc(2500 * sizeof(double));
for(int x1793=0; x1793 < 2500; x1793++) {
double x1794 = x1769[x1793];
double x1795 = x1785[x1793];
double x1796 = x1794 / x1795;
x1792[x1793] = x1796;

}
for(int x1800=0; x1800 < 2500; x1800++) {
double x1801 = x37[x1800];
double x1802 = x1792[x1800];
double x1803 = x1801 - x1802;
x37[x1800] = x1803;

}
for(int x1807=0; x1807 < 2500; x1807++) {
x142[x1807] = 0.0;

}
for(int x1811=0; x1811 < 1300; x1811++) {
double x1812 = x147[x1811];
bool x1813 = x1812 > 5.0;
if (x1813) {
x147[x1811] = 5.0;
} else {
}
bool x1817 = x1812 < -5.0;
if (x1817) {
x147[x1811] = -5.0;
} else {
}

}
double* x1823 = (double*)myMalloc(1300 * sizeof(double));
for(int x1824=0; x1824 < 1300; x1824++) {
double x1825 = x147[x1824];
double x1826 = x1825 * x1825;
x1823[x1824] = x1826;

}
for(int x1830=0; x1830 < 1300; x1830++) {
double x1831 = x238[x1830];
double x1832 = x1823[x1830];
double x1833 = x1831 + x1832;
x238[x1830] = x1833;

}
double* x1837 = (double*)myMalloc(1300 * sizeof(double));
for(int x1838=0; x1838 < 1300; x1838++) {
double x1839 = x147[x1838];
double x1840 = x207[0];
double x1841 = x1839 * x1840;
x1837[x1838] = x1841;

}
double* x1845 = (double*)myMalloc(1300 * sizeof(double));
for(int x1846=0; x1846 < 1300; x1846++) {
double x1847 = x238[x1846];
double x1848 = x213[0];
double x1849 = x1847 + x1848;
x1845[x1846] = x1849;

}
double* x1853 = (double*)myMalloc(1300 * sizeof(double));
for(int x1854=0; x1854 < 1300; x1854++) {
double x1855 = x1845[x1854];
double x1856 = sqrt(x1855);
x1853[x1854] = x1856;

}
double* x1860 = (double*)myMalloc(1300 * sizeof(double));
for(int x1861=0; x1861 < 1300; x1861++) {
double x1862 = x1837[x1861];
double x1863 = x1853[x1861];
double x1864 = x1862 / x1863;
x1860[x1861] = x1864;

}
for(int x1868=0; x1868 < 1300; x1868++) {
double x1869 = x44[x1868];
double x1870 = x1860[x1868];
double x1871 = x1869 - x1870;
x44[x1868] = x1871;

}
for(int x1875=0; x1875 < 1300; x1875++) {
x147[x1875] = 0.0;

}
for(int x1879=0; x1879 < 50; x1879++) {
double x1880 = x152[x1879];
bool x1881 = x1880 > 5.0;
if (x1881) {
x152[x1879] = 5.0;
} else {
}
bool x1885 = x1880 < -5.0;
if (x1885) {
x152[x1879] = -5.0;
} else {
}

}
double* x1891 = (double*)myMalloc(50 * sizeof(double));
for(int x1892=0; x1892 < 50; x1892++) {
double x1893 = x152[x1892];
double x1894 = x1893 * x1893;
x1891[x1892] = x1894;

}
for(int x1898=0; x1898 < 50; x1898++) {
double x1899 = x243[x1898];
double x1900 = x1891[x1898];
double x1901 = x1899 + x1900;
x243[x1898] = x1901;

}
double* x1905 = (double*)myMalloc(50 * sizeof(double));
for(int x1906=0; x1906 < 50; x1906++) {
double x1907 = x152[x1906];
double x1908 = x207[0];
double x1909 = x1907 * x1908;
x1905[x1906] = x1909;

}
double* x1913 = (double*)myMalloc(50 * sizeof(double));
for(int x1914=0; x1914 < 50; x1914++) {
double x1915 = x243[x1914];
double x1916 = x213[0];
double x1917 = x1915 + x1916;
x1913[x1914] = x1917;

}
double* x1921 = (double*)myMalloc(50 * sizeof(double));
for(int x1922=0; x1922 < 50; x1922++) {
double x1923 = x1913[x1922];
double x1924 = sqrt(x1923);
x1921[x1922] = x1924;

}
double* x1928 = (double*)myMalloc(50 * sizeof(double));
for(int x1929=0; x1929 < 50; x1929++) {
double x1930 = x1905[x1929];
double x1931 = x1921[x1929];
double x1932 = x1930 / x1931;
x1928[x1929] = x1932;

}
for(int x1936=0; x1936 < 50; x1936++) {
double x1937 = x51[x1936];
double x1938 = x1928[x1936];
double x1939 = x1937 - x1938;
x51[x1936] = x1939;

}
for(int x1943=0; x1943 < 50; x1943++) {
x152[x1943] = 0.0;

}
for(int x1947=0; x1947 < 2500; x1947++) {
double x1948 = x157[x1947];
bool x1949 = x1948 > 5.0;
if (x1949) {
x157[x1947] = 5.0;
} else {
}
bool x1953 = x1948 < -5.0;
if (x1953) {
x157[x1947] = -5.0;
} else {
}

}
double* x1959 = (double*)myMalloc(2500 * sizeof(double));
for(int x1960=0; x1960 < 2500; x1960++) {
double x1961 = x157[x1960];
double x1962 = x1961 * x1961;
x1959[x1960] = x1962;

}
for(int x1966=0; x1966 < 2500; x1966++) {
double x1967 = x248[x1966];
double x1968 = x1959[x1966];
double x1969 = x1967 + x1968;
x248[x1966] = x1969;

}
double* x1973 = (double*)myMalloc(2500 * sizeof(double));
for(int x1974=0; x1974 < 2500; x1974++) {
double x1975 = x157[x1974];
double x1976 = x207[0];
double x1977 = x1975 * x1976;
x1973[x1974] = x1977;

}
double* x1981 = (double*)myMalloc(2500 * sizeof(double));
for(int x1982=0; x1982 < 2500; x1982++) {
double x1983 = x248[x1982];
double x1984 = x213[0];
double x1985 = x1983 + x1984;
x1981[x1982] = x1985;

}
double* x1989 = (double*)myMalloc(2500 * sizeof(double));
for(int x1990=0; x1990 < 2500; x1990++) {
double x1991 = x1981[x1990];
double x1992 = sqrt(x1991);
x1989[x1990] = x1992;

}
double* x1996 = (double*)myMalloc(2500 * sizeof(double));
for(int x1997=0; x1997 < 2500; x1997++) {
double x1998 = x1973[x1997];
double x1999 = x1989[x1997];
double x2000 = x1998 / x1999;
x1996[x1997] = x2000;

}
for(int x2004=0; x2004 < 2500; x2004++) {
double x2005 = x56[x2004];
double x2006 = x1996[x2004];
double x2007 = x2005 - x2006;
x56[x2004] = x2007;

}
for(int x2011=0; x2011 < 2500; x2011++) {
x157[x2011] = 0.0;

}
for(int x2015=0; x2015 < 1300; x2015++) {
double x2016 = x162[x2015];
bool x2017 = x2016 > 5.0;
if (x2017) {
x162[x2015] = 5.0;
} else {
}
bool x2021 = x2016 < -5.0;
if (x2021) {
x162[x2015] = -5.0;
} else {
}

}
double* x2027 = (double*)myMalloc(1300 * sizeof(double));
for(int x2028=0; x2028 < 1300; x2028++) {
double x2029 = x162[x2028];
double x2030 = x2029 * x2029;
x2027[x2028] = x2030;

}
for(int x2034=0; x2034 < 1300; x2034++) {
double x2035 = x253[x2034];
double x2036 = x2027[x2034];
double x2037 = x2035 + x2036;
x253[x2034] = x2037;

}
double* x2041 = (double*)myMalloc(1300 * sizeof(double));
for(int x2042=0; x2042 < 1300; x2042++) {
double x2043 = x162[x2042];
double x2044 = x207[0];
double x2045 = x2043 * x2044;
x2041[x2042] = x2045;

}
double* x2049 = (double*)myMalloc(1300 * sizeof(double));
for(int x2050=0; x2050 < 1300; x2050++) {
double x2051 = x253[x2050];
double x2052 = x213[0];
double x2053 = x2051 + x2052;
x2049[x2050] = x2053;

}
double* x2057 = (double*)myMalloc(1300 * sizeof(double));
for(int x2058=0; x2058 < 1300; x2058++) {
double x2059 = x2049[x2058];
double x2060 = sqrt(x2059);
x2057[x2058] = x2060;

}
double* x2064 = (double*)myMalloc(1300 * sizeof(double));
for(int x2065=0; x2065 < 1300; x2065++) {
double x2066 = x2041[x2065];
double x2067 = x2057[x2065];
double x2068 = x2066 / x2067;
x2064[x2065] = x2068;

}
for(int x2072=0; x2072 < 1300; x2072++) {
double x2073 = x63[x2072];
double x2074 = x2064[x2072];
double x2075 = x2073 - x2074;
x63[x2072] = x2075;

}
for(int x2079=0; x2079 < 1300; x2079++) {
x162[x2079] = 0.0;

}
for(int x2083=0; x2083 < 50; x2083++) {
double x2084 = x167[x2083];
bool x2085 = x2084 > 5.0;
if (x2085) {
x167[x2083] = 5.0;
} else {
}
bool x2089 = x2084 < -5.0;
if (x2089) {
x167[x2083] = -5.0;
} else {
}

}
double* x2095 = (double*)myMalloc(50 * sizeof(double));
for(int x2096=0; x2096 < 50; x2096++) {
double x2097 = x167[x2096];
double x2098 = x2097 * x2097;
x2095[x2096] = x2098;

}
for(int x2102=0; x2102 < 50; x2102++) {
double x2103 = x258[x2102];
double x2104 = x2095[x2102];
double x2105 = x2103 + x2104;
x258[x2102] = x2105;

}
double* x2109 = (double*)myMalloc(50 * sizeof(double));
for(int x2110=0; x2110 < 50; x2110++) {
double x2111 = x167[x2110];
double x2112 = x207[0];
double x2113 = x2111 * x2112;
x2109[x2110] = x2113;

}
double* x2117 = (double*)myMalloc(50 * sizeof(double));
for(int x2118=0; x2118 < 50; x2118++) {
double x2119 = x258[x2118];
double x2120 = x213[0];
double x2121 = x2119 + x2120;
x2117[x2118] = x2121;

}
double* x2125 = (double*)myMalloc(50 * sizeof(double));
for(int x2126=0; x2126 < 50; x2126++) {
double x2127 = x2117[x2126];
double x2128 = sqrt(x2127);
x2125[x2126] = x2128;

}
double* x2132 = (double*)myMalloc(50 * sizeof(double));
for(int x2133=0; x2133 < 50; x2133++) {
double x2134 = x2109[x2133];
double x2135 = x2125[x2133];
double x2136 = x2134 / x2135;
x2132[x2133] = x2136;

}
for(int x2140=0; x2140 < 50; x2140++) {
double x2141 = x70[x2140];
double x2142 = x2132[x2140];
double x2143 = x2141 - x2142;
x70[x2140] = x2143;

}
for(int x2147=0; x2147 < 50; x2147++) {
x167[x2147] = 0.0;

}
for(int x2151=0; x2151 < 2500; x2151++) {
double x2152 = x172[x2151];
bool x2153 = x2152 > 5.0;
if (x2153) {
x172[x2151] = 5.0;
} else {
}
bool x2157 = x2152 < -5.0;
if (x2157) {
x172[x2151] = -5.0;
} else {
}

}
double* x2163 = (double*)myMalloc(2500 * sizeof(double));
for(int x2164=0; x2164 < 2500; x2164++) {
double x2165 = x172[x2164];
double x2166 = x2165 * x2165;
x2163[x2164] = x2166;

}
for(int x2170=0; x2170 < 2500; x2170++) {
double x2171 = x263[x2170];
double x2172 = x2163[x2170];
double x2173 = x2171 + x2172;
x263[x2170] = x2173;

}
double* x2177 = (double*)myMalloc(2500 * sizeof(double));
for(int x2178=0; x2178 < 2500; x2178++) {
double x2179 = x172[x2178];
double x2180 = x207[0];
double x2181 = x2179 * x2180;
x2177[x2178] = x2181;

}
double* x2185 = (double*)myMalloc(2500 * sizeof(double));
for(int x2186=0; x2186 < 2500; x2186++) {
double x2187 = x263[x2186];
double x2188 = x213[0];
double x2189 = x2187 + x2188;
x2185[x2186] = x2189;

}
double* x2193 = (double*)myMalloc(2500 * sizeof(double));
for(int x2194=0; x2194 < 2500; x2194++) {
double x2195 = x2185[x2194];
double x2196 = sqrt(x2195);
x2193[x2194] = x2196;

}
double* x2200 = (double*)myMalloc(2500 * sizeof(double));
for(int x2201=0; x2201 < 2500; x2201++) {
double x2202 = x2177[x2201];
double x2203 = x2193[x2201];
double x2204 = x2202 / x2203;
x2200[x2201] = x2204;

}
for(int x2208=0; x2208 < 2500; x2208++) {
double x2209 = x75[x2208];
double x2210 = x2200[x2208];
double x2211 = x2209 - x2210;
x75[x2208] = x2211;

}
for(int x2215=0; x2215 < 2500; x2215++) {
x172[x2215] = 0.0;

}
for(int x2219=0; x2219 < 1300; x2219++) {
double x2220 = x177[x2219];
bool x2221 = x2220 > 5.0;
if (x2221) {
x177[x2219] = 5.0;
} else {
}
bool x2225 = x2220 < -5.0;
if (x2225) {
x177[x2219] = -5.0;
} else {
}

}
double* x2231 = (double*)myMalloc(1300 * sizeof(double));
for(int x2232=0; x2232 < 1300; x2232++) {
double x2233 = x177[x2232];
double x2234 = x2233 * x2233;
x2231[x2232] = x2234;

}
for(int x2238=0; x2238 < 1300; x2238++) {
double x2239 = x268[x2238];
double x2240 = x2231[x2238];
double x2241 = x2239 + x2240;
x268[x2238] = x2241;

}
double* x2245 = (double*)myMalloc(1300 * sizeof(double));
for(int x2246=0; x2246 < 1300; x2246++) {
double x2247 = x177[x2246];
double x2248 = x207[0];
double x2249 = x2247 * x2248;
x2245[x2246] = x2249;

}
double* x2253 = (double*)myMalloc(1300 * sizeof(double));
for(int x2254=0; x2254 < 1300; x2254++) {
double x2255 = x268[x2254];
double x2256 = x213[0];
double x2257 = x2255 + x2256;
x2253[x2254] = x2257;

}
double* x2261 = (double*)myMalloc(1300 * sizeof(double));
for(int x2262=0; x2262 < 1300; x2262++) {
double x2263 = x2253[x2262];
double x2264 = sqrt(x2263);
x2261[x2262] = x2264;

}
double* x2268 = (double*)myMalloc(1300 * sizeof(double));
for(int x2269=0; x2269 < 1300; x2269++) {
double x2270 = x2245[x2269];
double x2271 = x2261[x2269];
double x2272 = x2270 / x2271;
x2268[x2269] = x2272;

}
for(int x2276=0; x2276 < 1300; x2276++) {
double x2277 = x82[x2276];
double x2278 = x2268[x2276];
double x2279 = x2277 - x2278;
x82[x2276] = x2279;

}
for(int x2283=0; x2283 < 1300; x2283++) {
x177[x2283] = 0.0;

}
for(int x2287=0; x2287 < 50; x2287++) {
double x2288 = x182[x2287];
bool x2289 = x2288 > 5.0;
if (x2289) {
x182[x2287] = 5.0;
} else {
}
bool x2293 = x2288 < -5.0;
if (x2293) {
x182[x2287] = -5.0;
} else {
}

}
double* x2299 = (double*)myMalloc(50 * sizeof(double));
for(int x2300=0; x2300 < 50; x2300++) {
double x2301 = x182[x2300];
double x2302 = x2301 * x2301;
x2299[x2300] = x2302;

}
for(int x2306=0; x2306 < 50; x2306++) {
double x2307 = x273[x2306];
double x2308 = x2299[x2306];
double x2309 = x2307 + x2308;
x273[x2306] = x2309;

}
double* x2313 = (double*)myMalloc(50 * sizeof(double));
for(int x2314=0; x2314 < 50; x2314++) {
double x2315 = x182[x2314];
double x2316 = x207[0];
double x2317 = x2315 * x2316;
x2313[x2314] = x2317;

}
double* x2321 = (double*)myMalloc(50 * sizeof(double));
for(int x2322=0; x2322 < 50; x2322++) {
double x2323 = x273[x2322];
double x2324 = x213[0];
double x2325 = x2323 + x2324;
x2321[x2322] = x2325;

}
double* x2329 = (double*)myMalloc(50 * sizeof(double));
for(int x2330=0; x2330 < 50; x2330++) {
double x2331 = x2321[x2330];
double x2332 = sqrt(x2331);
x2329[x2330] = x2332;

}
double* x2336 = (double*)myMalloc(50 * sizeof(double));
for(int x2337=0; x2337 < 50; x2337++) {
double x2338 = x2313[x2337];
double x2339 = x2329[x2337];
double x2340 = x2338 / x2339;
x2336[x2337] = x2340;

}
for(int x2344=0; x2344 < 50; x2344++) {
double x2345 = x89[x2344];
double x2346 = x2336[x2344];
double x2347 = x2345 - x2346;
x89[x2344] = x2347;

}
for(int x2351=0; x2351 < 50; x2351++) {
x182[x2351] = 0.0;

}
for(int x2355=0; x2355 < 1300; x2355++) {
double x2356 = x187[x2355];
bool x2357 = x2356 > 5.0;
if (x2357) {
x187[x2355] = 5.0;
} else {
}
bool x2361 = x2356 < -5.0;
if (x2361) {
x187[x2355] = -5.0;
} else {
}

}
double* x2367 = (double*)myMalloc(1300 * sizeof(double));
for(int x2368=0; x2368 < 1300; x2368++) {
double x2369 = x187[x2368];
double x2370 = x2369 * x2369;
x2367[x2368] = x2370;

}
for(int x2374=0; x2374 < 1300; x2374++) {
double x2375 = x278[x2374];
double x2376 = x2367[x2374];
double x2377 = x2375 + x2376;
x278[x2374] = x2377;

}
double* x2381 = (double*)myMalloc(1300 * sizeof(double));
for(int x2382=0; x2382 < 1300; x2382++) {
double x2383 = x187[x2382];
double x2384 = x207[0];
double x2385 = x2383 * x2384;
x2381[x2382] = x2385;

}
double* x2389 = (double*)myMalloc(1300 * sizeof(double));
for(int x2390=0; x2390 < 1300; x2390++) {
double x2391 = x278[x2390];
double x2392 = x213[0];
double x2393 = x2391 + x2392;
x2389[x2390] = x2393;

}
double* x2397 = (double*)myMalloc(1300 * sizeof(double));
for(int x2398=0; x2398 < 1300; x2398++) {
double x2399 = x2389[x2398];
double x2400 = sqrt(x2399);
x2397[x2398] = x2400;

}
double* x2404 = (double*)myMalloc(1300 * sizeof(double));
for(int x2405=0; x2405 < 1300; x2405++) {
double x2406 = x2381[x2405];
double x2407 = x2397[x2405];
double x2408 = x2406 / x2407;
x2404[x2405] = x2408;

}
for(int x2412=0; x2412 < 1300; x2412++) {
double x2413 = x94[x2412];
double x2414 = x2404[x2412];
double x2415 = x2413 - x2414;
x94[x2412] = x2415;

}
for(int x2419=0; x2419 < 1300; x2419++) {
x187[x2419] = 0.0;

}
for(int x2423=0; x2423 < 26; x2423++) {
double x2424 = x192[x2423];
bool x2425 = x2424 > 5.0;
if (x2425) {
x192[x2423] = 5.0;
} else {
}
bool x2429 = x2424 < -5.0;
if (x2429) {
x192[x2423] = -5.0;
} else {
}

}
double* x2435 = (double*)myMalloc(26 * sizeof(double));
for(int x2436=0; x2436 < 26; x2436++) {
double x2437 = x192[x2436];
double x2438 = x2437 * x2437;
x2435[x2436] = x2438;

}
for(int x2442=0; x2442 < 26; x2442++) {
double x2443 = x283[x2442];
double x2444 = x2435[x2442];
double x2445 = x2443 + x2444;
x283[x2442] = x2445;

}
double* x2449 = (double*)myMalloc(26 * sizeof(double));
for(int x2450=0; x2450 < 26; x2450++) {
double x2451 = x192[x2450];
double x2452 = x207[0];
double x2453 = x2451 * x2452;
x2449[x2450] = x2453;

}
double* x2457 = (double*)myMalloc(26 * sizeof(double));
for(int x2458=0; x2458 < 26; x2458++) {
double x2459 = x283[x2458];
double x2460 = x213[0];
double x2461 = x2459 + x2460;
x2457[x2458] = x2461;

}
double* x2465 = (double*)myMalloc(26 * sizeof(double));
for(int x2466=0; x2466 < 26; x2466++) {
double x2467 = x2457[x2466];
double x2468 = sqrt(x2467);
x2465[x2466] = x2468;

}
double* x2472 = (double*)myMalloc(26 * sizeof(double));
for(int x2473=0; x2473 < 26; x2473++) {
double x2474 = x2449[x2473];
double x2475 = x2465[x2473];
double x2476 = x2474 / x2475;
x2472[x2473] = x2476;

}
for(int x2480=0; x2480 < 26; x2480++) {
double x2481 = x101[x2480];
double x2482 = x2472[x2480];
double x2483 = x2481 - x2482;
x101[x2480] = x2483;

}
for(int x2487=0; x2487 < 26; x2487++) {
x192[x2487] = 0.0;

}
for(int x2491=0; x2491 < 50; x2491++) {
x197[x2491] = 0.0;

}
for(int x2495=0; x2495 < 50; x2495++) {
x202[x2495] = 0.0;

}
for(int x2499=0; x2499 < 50; x2499++) {
double x2500 = x117[x2499];
x107[x2499] = x2500;

}
for(int x2504=0; x2504 < 50; x2504++) {
double x2505 = x122[x2504];
x112[x2504] = x2505;

}
mallocAddr = (void*)x290;

}
double x2512 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x2515 = (long)fopen(x0, "w");
fprintf((FILE *)x2515, "unit: %s\n", "100 iteration");
for(int x2518=0; x2518 < 51; x2518++) {
double x2519 = x289[x2518];
fprintf((FILE *)x2515, "%lf\n", x2519);

}
double x2513 = x288 - x1;
double x2514 = x2512 - x288;
fprintf((FILE *)x2515, "run time: %lf %lf\n", x2513, x2514);
fclose((FILE*)x2515);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

