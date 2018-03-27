
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
printf("Here we go!! Go MNIST!!!!\n");
srand(42);
struct timeval begin_0, end_0, diff_0;
gettimeofday(&begin_0, NULL);
float* x5 = (float*)myMalloc(250 * sizeof(float));
for(int x7=0; x7 < 250; x7++) {
float x8 = (float)rand()/RAND_MAX;
float x9 = x8 - 0.5f;
float x10 = x9 * 0.2f;
x5[x7] = x10;

}
float* x14 = (float*)myMalloc(250 * sizeof(float));
for(int x15=0; x15 < 250; x15++) {
x14[x15] = 0.0f;

}
float* x19 = (float*)myMalloc(5000 * sizeof(float));
for(int x21=0; x21 < 5000; x21++) {
float x22 = (float)rand()/RAND_MAX;
float x23 = x22 - 0.5f;
float x24 = x23 * 0.06324556f;
x19[x21] = x24;

}
float* x28 = (float*)myMalloc(5000 * sizeof(float));
for(int x29=0; x29 < 5000; x29++) {
x28[x29] = 0.0f;

}
float* x33 = (float*)myMalloc(16000 * sizeof(float));
for(int x35=0; x35 < 16000; x35++) {
float x36 = (float)rand()/RAND_MAX;
float x37 = x36 - 0.5f;
float x38 = x37 * 0.0559017f;
x33[x35] = x38;

}
float* x42 = (float*)myMalloc(50 * sizeof(float));
for(int x44=0; x44 < 50; x44++) {
float x45 = (float)rand()/RAND_MAX;
float x46 = x45 - 0.5f;
float x47 = x46 * 0.0559017f;
x42[x44] = x47;

}
float* x51 = (float*)myMalloc(16000 * sizeof(float));
for(int x52=0; x52 < 16000; x52++) {
x51[x52] = 0.0f;

}
float* x56 = (float*)myMalloc(50 * sizeof(float));
for(int x57=0; x57 < 50; x57++) {
x56[x57] = 0.0f;

}
float* x61 = (float*)myMalloc(500 * sizeof(float));
for(int x63=0; x63 < 500; x63++) {
float x64 = (float)rand()/RAND_MAX;
float x65 = x64 - 0.5f;
float x66 = x65 * 0.14142136f;
x61[x63] = x66;

}
float* x70 = (float*)myMalloc(10 * sizeof(float));
for(int x72=0; x72 < 10; x72++) {
float x73 = (float)rand()/RAND_MAX;
float x74 = x73 - 0.5f;
float x75 = x74 * 0.14142136f;
x70[x72] = x75;

}
float* x79 = (float*)myMalloc(500 * sizeof(float));
for(int x80=0; x80 < 500; x80++) {
x79[x80] = 0.0f;

}
float* x84 = (float*)myMalloc(10 * sizeof(float));
for(int x85=0; x85 < 10; x85++) {
x84[x85] = 0.0f;

}
int64_t* x89 = (int64_t*)myMalloc(2 * sizeof(int64_t));
int64_t* x90 = (int64_t*)myMalloc(2 * sizeof(int64_t));
printf("Start normalize\n");
int32_t x102 = 0;
int32_t x103 = x102;
int32_t x104 = x103;
int32_t x96 = open("../data/bin/mnist_train_target.bin",0);
int64_t x97 = fsize(x96);
int64_t x99 = x97 / 4LL;
int32_t x100 = (int32_t)x99;
int* x98 = (int32_t*)mmap(0, x97, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x96, 0);
int32_t x91 = open("../data/bin/mnist_train.bin",0);
int64_t x92 = fsize(x91);
float* x93 = (float*)mmap(0, x92, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x91, 0);
for(int x106=0; x106 < x100; x106++) {
int32_t x107 = x104;
int32_t x109 = x98[x106];
float* x108 = x93+x107;
for(int x111=0; x111 < 784; x111++) {
float x112 = x108[x111];
float x113 = x112 - 0.1307f;
float x114 = x113 / 0.3081f;
x108[x111] = x114;

}
x104 += 784;

}
int32_t x121 = x104;
int64_t x94 = x92 / 4LL;
int32_t x95 = (int32_t)x94;
bool x122 = x121 == x95;
if (x122) {
} else {
printf("Data length doesn't match\n");
exit(0);
}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x130 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
float x131 = (float)x130;
float x132 = x131 / 1000000.0f;
printf("Data normalized (all prepare time) in %lf sec\n",x132);
double* x134 = (double*)myMalloc(10 * sizeof(double));
int64_t x135 = (long)mallocAddr;
int32_t x988 = x100 / 10;
double x993 = (double)x100;
int64_t x1017 = (int64_t)x100;
float x1021 = (float)x100;
for(int x136=0; x136 < 10; x136++) {
struct timeval begin_1, end_1, diff_1;
int32_t x138 = 0;
int32_t x139 = x138;
int32_t x140 = x139;
float x141 = 0.0f;
float x142 = x141;
float x143 = x142;
int32_t x144 = x136 + 1;
printf("Start training epoch %d\n",x144);
gettimeofday(&begin_1, NULL);
int32_t x147 = 0;
int32_t x148 = x147;
int32_t x149 = x148;
for(int x150=0; x150 < x100; x150++) {
int32_t x151 = x149;
int32_t x153 = x98[x150];
x140 += 1;
float* x155 = (float*)myMalloc(1 * sizeof(float));
x155[0] = 0.0f;
float* x157 = (float*)myMalloc(1 * sizeof(float));
x157[0] = 0.0f;
float* x159 = (float*)myMalloc(1 * sizeof(float));
for(int x161=0; x161 < 1; x161++) {
x159[x161] = 0.0f;

}
float* x165 = (float*)myMalloc(1 * sizeof(float));
for(int x166=0; x166 < 1; x166++) {
x165[x166] = 0.0f;

}
float* x170 = (float*)myMalloc(5760 * sizeof(float));
for(int x172=0; x172 < 5760; x172++) {
x170[x172] = 0.0f;

}
int32_t x176 = 0;
int32_t x177 = 0;
float* x152 = x93+x151;
for(int x178=0; x178 < 10; x178++) {
int32_t x179 = x177;
int32_t x180 = x179;
int32_t x181 = 0;
int32_t x182 = x176;
float* x183 = x170+x182;
for(int x184=0; x184 < 1; x184++) {
int32_t x185 = x181;
int32_t x187 = x180;
float* x188 = x5+x187;
int32_t x189 = 0;
int32_t x190 = 0;
float* x186 = x152+x185;
for(int x192=0; x192 < 24; x192++) {
int32_t x193 = x190;
int32_t x194 = x193;
for(int x195=0; x195 < 24; x195++) {
int32_t x196 = 0;
int32_t x197 = x194;
int32_t x198 = x197;
float x199 = 0.0f;
for(int x201=0; x201 < 5; x201++) {
int32_t x202 = x198;
int32_t x204 = x196;
float* x205 = x188+x204;
float* x203 = x186+x202;
for(int x206=0; x206 < 5; x206++) {
float x207 = x203[x206];
float x208 = x205[x206];
float x209 = x207 * x208;
x199 += x209;

}
x196 += 5;
x198 += 28;

}
int32_t x217 = x189;
float x218 = x183[x217];
float x219 = x199;
float x220 = x218 + x219;
x183[x217] = x220;
x189 += 1;
x194 += 1;

}
x190 += 28;

}
x180 += 25;
x181 += 784;

}
x177 += 25;
x176 += 576;

}
float* x237 = (float*)myMalloc(5760 * sizeof(float));
for(int x238=0; x238 < 5760; x238++) {
x237[x238] = 0.0f;

}
float* x242 = (float*)myMalloc(1440 * sizeof(float));
for(int x244=0; x244 < 1440; x244++) {
x242[x244] = -1.0E10f;

}
int32_t* x248 = (int32_t*)myMalloc(1440 * sizeof(int32_t));
int32_t x249 = 0;
int32_t x250 = 0;
for(int x251=0; x251 < 10; x251++) {
int32_t x252 = x249;
int32_t x253 = x252;
for(int x255=0; x255 < 12; x255++) {
for(int x257=0; x257 < 2; x257++) {
int32_t x258 = x253;
int32_t x259 = x258;
for(int x260=0; x260 < 12; x260++) {
int32_t x261 = x250;
float x262 = x170[x261];
int32_t x263 = x259;
float x264 = x242[x263];
bool x265 = x262 > x264;
if (x265) {
float x266 = x170[x261];
x242[x263] = x266;
x248[x263] = x261;
} else {
}
x250 += 1;
int32_t x272 = x250;
float x273 = x170[x272];
float x274 = x242[x263];
bool x275 = x273 > x274;
if (x275) {
float x276 = x170[x272];
x242[x263] = x276;
x248[x263] = x272;
} else {
}
x250 += 1;
x259 += 1;

}

}
x253 += 12;

}
x249 += 144;

}
float* x293 = (float*)myMalloc(1440 * sizeof(float));
for(int x294=0; x294 < 1440; x294++) {
x293[x294] = 0.0f;

}
float* x298 = (float*)myMalloc(1440 * sizeof(float));
for(int x299=0; x299 < 1440; x299++) {
float x300 = x242[x299];
bool x301 = x300 < 0.0f;
if (x301) {
x298[x299] = 0.0f;
} else {
float x304 = x242[x299];
x298[x299] = x304;
}

}
float* x310 = (float*)myMalloc(1440 * sizeof(float));
for(int x311=0; x311 < 1440; x311++) {
x310[x311] = 0.0f;

}
float* x315 = (float*)myMalloc(1280 * sizeof(float));
for(int x317=0; x317 < 1280; x317++) {
x315[x317] = 0.0f;

}
int32_t x321 = 0;
int32_t x322 = 0;
for(int x324=0; x324 < 20; x324++) {
int32_t x325 = x322;
int32_t x326 = x325;
int32_t x327 = 0;
int32_t x328 = x321;
float* x329 = x315+x328;
for(int x330=0; x330 < 10; x330++) {
int32_t x331 = x327;
float* x332 = x298+x331;
int32_t x333 = x326;
float* x334 = x19+x333;
int32_t x335 = 0;
int32_t x336 = 0;
for(int x338=0; x338 < 8; x338++) {
int32_t x339 = x336;
int32_t x340 = x339;
for(int x341=0; x341 < 8; x341++) {
int32_t x342 = 0;
int32_t x343 = x340;
int32_t x344 = x343;
float x345 = 0.0f;
for(int x346=0; x346 < 5; x346++) {
int32_t x347 = x344;
float* x348 = x332+x347;
int32_t x349 = x342;
float* x350 = x334+x349;
for(int x351=0; x351 < 5; x351++) {
float x352 = x348[x351];
float x353 = x350[x351];
float x354 = x352 * x353;
x345 += x354;

}
x342 += 5;
x344 += 12;

}
int32_t x362 = x335;
float x363 = x329[x362];
float x364 = x345;
float x365 = x363 + x364;
x329[x362] = x365;
x335 += 1;
x340 += 1;

}
x336 += 12;

}
x326 += 25;
x327 += 144;

}
x322 += 250;
x321 += 64;

}
float* x382 = (float*)myMalloc(1280 * sizeof(float));
for(int x383=0; x383 < 1280; x383++) {
x382[x383] = 0.0f;

}
float* x387 = (float*)myMalloc(320 * sizeof(float));
for(int x389=0; x389 < 320; x389++) {
x387[x389] = -1.0E10f;

}
int32_t* x393 = (int32_t*)myMalloc(320 * sizeof(int32_t));
int32_t x394 = 0;
int32_t x395 = 0;
for(int x396=0; x396 < 20; x396++) {
int32_t x397 = x394;
int32_t x398 = x397;
for(int x400=0; x400 < 4; x400++) {
for(int x401=0; x401 < 2; x401++) {
int32_t x402 = x398;
int32_t x403 = x402;
for(int x404=0; x404 < 4; x404++) {
int32_t x405 = x395;
float x406 = x315[x405];
int32_t x407 = x403;
float x408 = x387[x407];
bool x409 = x406 > x408;
if (x409) {
float x410 = x315[x405];
x387[x407] = x410;
x393[x407] = x405;
} else {
}
x395 += 1;
int32_t x416 = x395;
float x417 = x315[x416];
float x418 = x387[x407];
bool x419 = x417 > x418;
if (x419) {
float x420 = x315[x416];
x387[x407] = x420;
x393[x407] = x416;
} else {
}
x395 += 1;
x403 += 1;

}

}
x398 += 4;

}
x394 += 16;

}
float* x437 = (float*)myMalloc(320 * sizeof(float));
for(int x438=0; x438 < 320; x438++) {
x437[x438] = 0.0f;

}
float* x442 = (float*)myMalloc(320 * sizeof(float));
for(int x443=0; x443 < 320; x443++) {
float x444 = x387[x443];
bool x445 = x444 < 0.0f;
if (x445) {
x442[x443] = 0.0f;
} else {
float x448 = x387[x443];
x442[x443] = x448;
}

}
float* x454 = (float*)myMalloc(320 * sizeof(float));
for(int x455=0; x455 < 320; x455++) {
x454[x455] = 0.0f;

}
// dot WrappedArray(50, 320) - WrappedArray(320)
int32_t x460 = 0;
float* x461 = (float*)myMalloc(50 * sizeof(float));
for(int x462=0; x462 < 50; x462++) {
float x463 = 0.0f;
for(int x464=0; x464 < 320; x464++) {
int32_t x465 = x460;
float x466 = x33[x465];
float x467 = x442[x464];
float x468 = x466 * x467;
x463 += x468;
x460 += 1;

}
float x473 = x463;
x461[x462] = x473;

}
float* x477 = (float*)myMalloc(50 * sizeof(float));
for(int x478=0; x478 < 50; x478++) {
x477[x478] = 0.0f;

}
float* x482 = (float*)myMalloc(50 * sizeof(float));
for(int x483=0; x483 < 50; x483++) {
float x484 = x461[x483];
float x485 = x42[x483];
float x486 = x484 + x485;
x482[x483] = x486;

}
float* x490 = (float*)myMalloc(50 * sizeof(float));
for(int x491=0; x491 < 50; x491++) {
x490[x491] = 0.0f;

}
float* x495 = (float*)myMalloc(50 * sizeof(float));
for(int x496=0; x496 < 50; x496++) {
float x497 = x482[x496];
bool x498 = x497 < 0.0f;
if (x498) {
x495[x496] = 0.0f;
} else {
float x501 = x482[x496];
x495[x496] = x501;
}

}
float* x507 = (float*)myMalloc(50 * sizeof(float));
for(int x508=0; x508 < 50; x508++) {
x507[x508] = 0.0f;

}
float* x512 = (float*)myMalloc(50 * sizeof(float));
float* x513 = (float*)myMalloc(50 * sizeof(float));
for(int x514=0; x514 < 50; x514++) {
float x515 = (float)rand()/RAND_MAX;
bool x516 = x515 > 0.5f;
if (x516) {
float x517 = x495[x514];
float x518 = x517 * 2.0f;
x512[x514] = x518;
x513[x514] = 2.0f;
} else {
x512[x514] = 0.0f;
x513[x514] = 0.0f;
}

}
float* x528 = (float*)myMalloc(50 * sizeof(float));
for(int x529=0; x529 < 50; x529++) {
x528[x529] = 0.0f;

}
// dot WrappedArray(10, 50) - WrappedArray(50)
int32_t x534 = 0;
float* x535 = (float*)myMalloc(10 * sizeof(float));
for(int x536=0; x536 < 10; x536++) {
float x537 = 0.0f;
for(int x538=0; x538 < 50; x538++) {
int32_t x539 = x534;
float x540 = x61[x539];
float x541 = x512[x538];
float x542 = x540 * x541;
x537 += x542;
x534 += 1;

}
float x547 = x537;
x535[x536] = x547;

}
float* x551 = (float*)myMalloc(10 * sizeof(float));
for(int x552=0; x552 < 10; x552++) {
x551[x552] = 0.0f;

}
float* x556 = (float*)myMalloc(10 * sizeof(float));
for(int x557=0; x557 < 10; x557++) {
float x558 = x535[x557];
float x559 = x70[x557];
float x560 = x558 + x559;
x556[x557] = x560;

}
float* x564 = (float*)myMalloc(10 * sizeof(float));
for(int x565=0; x565 < 10; x565++) {
x564[x565] = 0.0f;

}
float x569 = -1.0E10f;
for(int x570=0; x570 < 10; x570++) {
float x571 = x569;
float x572 = x556[x570];
bool x573 = x572 > x571;
float x574;
if (x573) {
x574 = x572;
} else {
x574 = x571;
}
x569 = x574;

}
float x578 = x569;
float x579 = 0.0f;
for(int x580=0; x580 < 10; x580++) {
float x581 = x579;
float x582 = x556[x580];
float x583 = x569;
float x584 = x582 - x583;
double x585 = (double)x584;
double x586 = exp(x585);
float x587 = (float)x586;
float x588 = x581 + x587;
x579 = x588;

}
float x592 = x579;
float* x597 = (float*)myMalloc(10 * sizeof(float));
double x593 = (double)x592;
double x594 = log(x593);
float x595 = (float)x594;
float x596 = x578 + x595;
for(int x598=0; x598 < 10; x598++) {
float x599 = x556[x598];
float x600 = x599 - x596;
x597[x598] = x600;

}
float* x604 = (float*)myMalloc(10 * sizeof(float));
for(int x605=0; x605 < 10; x605++) {
x604[x605] = 0.0f;

}
float x609 = x597[x153];
float* x611 = (float*)myMalloc(1 * sizeof(float));
float x610 = -1.0f * x609;
x611[0] = x610;
float* x613 = (float*)myMalloc(1 * sizeof(float));
for(int x614=0; x614 < 1; x614++) {
x613[x614] = 0.0f;

}
for(int x618=0; x618 < 1; x618++) {
float x619 = x613[x618];
x613[x618] = 1.0f;

}
for(int x623=0; x623 < 1; x623++) {
float x624 = x611[x623];
x165[x623] = x624;

}
float x628 = x613[0];
float x629 = -1.0f * x628;
x604[x153] = x629;
float x631 = 0.0f;
for(int x632=0; x632 < 10; x632++) {
float x633 = x631;
float x634 = x604[x632];
float x635 = x633 + x634;
x631 = x635;

}
float x639 = x631;
float* x640 = (float*)myMalloc(1 * sizeof(float));
x640[0] = x639;
float x642 = x640[0];
for(int x643=0; x643 < 10; x643++) {
float x644 = x604[x643];
float x645 = x597[x643];
double x646 = (double)x645;
double x647 = exp(x646);
float x648 = (float)x647;
float x649 = x648 * x642;
float x650 = x644 - x649;
x564[x643] = x650;

}
// backpropagate +
for(int x655=0; x655 < 10; x655++) {
float x656 = x551[x655];
float x657 = x564[x655];
float x658 = x656 + x657;
x551[x655] = x658;

}
for(int x662=0; x662 < 10; x662++) {
float x663 = x84[x662];
float x664 = x564[x662];
float x665 = x663 + x664;
x84[x662] = x665;

}
// add_cartesian
int32_t x670 = 0;
for(int x671=0; x671 < 10; x671++) {
for(int x672=0; x672 < 50; x672++) {
int32_t x673 = x670;
int32_t x674 = x673 + x672;
float x675 = x79[x674];
float x676 = x512[x672];
float x677 = x551[x671];
float x678 = x676 * x677;
float x679 = x675 + x678;
x79[x674] = x679;

}
x670 += 50;

}
int32_t x686 = 0;
for(int x687=0; x687 < 10; x687++) {
for(int x688=0; x688 < 50; x688++) {
float x689 = x528[x688];
int32_t x690 = x686;
int32_t x691 = x690 + x688;
float x692 = x61[x691];
float x693 = x551[x687];
float x694 = x692 * x693;
float x695 = x689 + x694;
x528[x688] = x695;

}
x686 += 50;

}
float* x702 = (float*)myMalloc(50 * sizeof(float));
for(int x703=0; x703 < 50; x703++) {
float x704 = x513[x703];
float x705 = x528[x703];
float x706 = x704 * x705;
x702[x703] = x706;

}
for(int x710=0; x710 < 50; x710++) {
float x711 = x507[x710];
float x712 = x702[x710];
float x713 = x711 + x712;
x507[x710] = x713;

}
for(int x717=0; x717 < 50; x717++) {
float x718 = x482[x717];
bool x719 = x718 < 0.0f;
float x722;
if (x719) {
x722 = 0.0f;
} else {
float x720 = x507[x717];
x722 = x720;
}
x490[x717] = x722;

}
// backpropagate +
for(int x727=0; x727 < 50; x727++) {
float x728 = x477[x727];
float x729 = x490[x727];
float x730 = x728 + x729;
x477[x727] = x730;

}
for(int x734=0; x734 < 50; x734++) {
float x735 = x56[x734];
float x736 = x490[x734];
float x737 = x735 + x736;
x56[x734] = x737;

}
// add_cartesian
int32_t x742 = 0;
for(int x743=0; x743 < 50; x743++) {
for(int x744=0; x744 < 320; x744++) {
int32_t x745 = x742;
int32_t x746 = x745 + x744;
float x747 = x51[x746];
float x748 = x442[x744];
float x749 = x477[x743];
float x750 = x748 * x749;
float x751 = x747 + x750;
x51[x746] = x751;

}
x742 += 320;

}
int32_t x758 = 0;
for(int x759=0; x759 < 50; x759++) {
for(int x760=0; x760 < 320; x760++) {
float x761 = x454[x760];
int32_t x762 = x758;
int32_t x763 = x762 + x760;
float x764 = x33[x763];
float x765 = x477[x759];
float x766 = x764 * x765;
float x767 = x761 + x766;
x454[x760] = x767;

}
x758 += 320;

}
for(int x774=0; x774 < 320; x774++) {
float x775 = x387[x774];
bool x776 = x775 < 0.0f;
float x779;
if (x776) {
x779 = 0.0f;
} else {
float x777 = x454[x774];
x779 = x777;
}
x437[x774] = x779;

}
for(int x783=0; x783 < 320; x783++) {
int32_t x784 = x393[x783];
float x785 = x437[x783];
x382[x784] = x785;

}
int32_t x789 = 0;
int32_t x790 = 0;
for(int x791=0; x791 < 20; x791++) {
int32_t x792 = 0;
for(int x793=0; x793 < 8; x793++) {
int32_t x794 = x792;
int32_t x795 = x794;
for(int x796=0; x796 < 8; x796++) {
int32_t x797 = x789;
float x798 = x382[x797];
int32_t x799 = x795;
int32_t x800 = x799;
int32_t x801 = x790;
int32_t x802 = x801;
for(int x803=0; x803 < 10; x803++) {
int32_t x804 = x800;
int32_t x805 = x804;
for(int x806=0; x806 < 5; x806++) {
for(int x807=0; x807 < 5; x807++) {
int32_t x808 = x805;
int32_t x809 = x808 + x807;
float x810 = x310[x809];
int32_t x811 = x802;
float x812 = x19[x811];
float x813 = x798 * x812;
float x814 = x810 + x813;
x310[x809] = x814;
float x816 = x28[x811];
float x817 = x298[x809];
float x818 = x798 * x817;
float x819 = x816 + x818;
x28[x811] = x819;
x802 += 1;

}
x805 += 12;

}
x800 += 144;

}
x795 += 1;
x789 += 1;

}
x792 += 12;

}
x790 += 250;

}
for(int x840=0; x840 < 1440; x840++) {
float x841 = x242[x840];
bool x842 = x841 < 0.0f;
float x845;
if (x842) {
x845 = 0.0f;
} else {
float x843 = x310[x840];
x845 = x843;
}
x293[x840] = x845;

}
for(int x849=0; x849 < 1440; x849++) {
int32_t x850 = x248[x849];
float x851 = x293[x849];
x237[x850] = x851;

}
int32_t x855 = 0;
int32_t x856 = 0;
for(int x857=0; x857 < 10; x857++) {
int32_t x858 = 0;
for(int x859=0; x859 < 24; x859++) {
int32_t x860 = x858;
int32_t x861 = x860;
for(int x862=0; x862 < 24; x862++) {
int32_t x863 = x855;
float x864 = x237[x863];
int32_t x865 = x861;
int32_t x866 = x865;
int32_t x867 = x856;
int32_t x868 = x867;
for(int x869=0; x869 < 1; x869++) {
int32_t x870 = x866;
int32_t x871 = x870;
for(int x872=0; x872 < 5; x872++) {
for(int x873=0; x873 < 5; x873++) {
int32_t x874 = x868;
float x875 = x14[x874];
int32_t x876 = x871;
int32_t x877 = x876 + x873;
float x878 = x152[x877];
float x879 = x864 * x878;
float x880 = x875 + x879;
x14[x874] = x880;
x868 += 1;

}
x871 += 28;

}
x866 += 784;

}
x861 += 1;
x855 += 1;

}
x858 += 28;

}
x856 += 25;

}
float x901 = x165[0];
x143 += x901;
// Generate code for addMul
for(int x904=0; x904 < 250; x904++) {
float x905 = x5[x904];
float x906 = x14[x904];
float x907 = -5.0E-4f * x906;
float x908 = x905 + x907;
x5[x904] = x908;

}
for(int x912=0; x912 < 250; x912++) {
float x913 = x14[x912];
x14[x912] = 0.0f;

}
// Generate code for addMul
for(int x918=0; x918 < 5000; x918++) {
float x919 = x19[x918];
float x920 = x28[x918];
float x921 = -5.0E-4f * x920;
float x922 = x919 + x921;
x19[x918] = x922;

}
for(int x926=0; x926 < 5000; x926++) {
float x927 = x28[x926];
x28[x926] = 0.0f;

}
// Generate code for addMul
for(int x932=0; x932 < 16000; x932++) {
float x933 = x33[x932];
float x934 = x51[x932];
float x935 = -5.0E-4f * x934;
float x936 = x933 + x935;
x33[x932] = x936;

}
for(int x940=0; x940 < 16000; x940++) {
float x941 = x51[x940];
x51[x940] = 0.0f;

}
// Generate code for addMul
for(int x946=0; x946 < 50; x946++) {
float x947 = x42[x946];
float x948 = x56[x946];
float x949 = -5.0E-4f * x948;
float x950 = x947 + x949;
x42[x946] = x950;

}
for(int x954=0; x954 < 50; x954++) {
float x955 = x56[x954];
x56[x954] = 0.0f;

}
// Generate code for addMul
for(int x960=0; x960 < 500; x960++) {
float x961 = x61[x960];
float x962 = x79[x960];
float x963 = -5.0E-4f * x962;
float x964 = x961 + x963;
x61[x960] = x964;

}
for(int x968=0; x968 < 500; x968++) {
float x969 = x79[x968];
x79[x968] = 0.0f;

}
// Generate code for addMul
for(int x974=0; x974 < 10; x974++) {
float x975 = x70[x974];
float x976 = x84[x974];
float x977 = -5.0E-4f * x976;
float x978 = x975 + x977;
x70[x974] = x978;

}
for(int x982=0; x982 < 10; x982++) {
float x983 = x84[x982];
x84[x982] = 0.0f;

}
int32_t x987 = x140;
int32_t x989 = x987 % x988;
bool x990 = x989 == 0;
if (x990) {
float x995 = x143;
double x991 = (double)x987;
double x992 = 100.0 * x991;
double x994 = x992 / x993;
float x996 = (float)x987;
float x997 = x995 / x996;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x136,x987,x100,x994,x997);
fflush(stdout);
} else {
}
mallocAddr = (void*)x135;
x149 += 784;

}
int32_t x1006 = x149;
bool x1007 = x1006 == x95;
if (x1007) {
} else {
printf("Data length doesn't match\n");
exit(0);
}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1015 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1016 = x1015 / 1000LL;
int64_t x1018 = x1015 / x1017;
printf("Training completed in %ldms (%ld us/images)\n",x1016,x1018);
float x1020 = x143;
float x1022 = x1020 / x1021;
double x1023 = (double)x1022;
x134[x136] = x1023;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1029 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
int64_t x1034 = (long)fopen(x0, "w");
fprintf((FILE *)x1034, "unit: %s\n", "1 epoch");
for(int x1036=0; x1036 < 10; x1036++) {
double x1037 = x134[x1036];
fprintf((FILE *)x1034, "%lf\n", x1037);

}
float x1030 = (float)x1029;
float x1031 = x1030 / 1000000.0f;
float x1032 = x1031 - x132;
float x1033 = x1032 / 10.0f;
fprintf((FILE *)x1034, "run time: %lf %lf\n", x132, x1033);
fclose((FILE*)x1034);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

