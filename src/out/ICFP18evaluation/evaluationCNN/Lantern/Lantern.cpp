
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
        mallocAddr = (void *)((char *)mallocAddr + bytes);
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
int32_t x979 = x100 / 10;
double x984 = (double)x100;
int64_t x1008 = (int64_t)x100;
float x1012 = (float)x100;
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
int32_t x184 = x181;
int32_t x186 = x180;
float* x187 = x5+x186;
int32_t x188 = 0;
int32_t x189 = 0;
float* x185 = x152+x184;
for(int x191=0; x191 < 24; x191++) {
int32_t x192 = x189;
int32_t x193 = x192;
for(int x194=0; x194 < 24; x194++) {
int32_t x195 = 0;
int32_t x196 = x193;
int32_t x197 = x196;
float x198 = 0.0f;
for(int x200=0; x200 < 5; x200++) {
int32_t x201 = x197;
int32_t x203 = x195;
float* x204 = x187+x203;
float* x202 = x185+x201;
for(int x205=0; x205 < 5; x205++) {
float x206 = x202[x205];
float x207 = x204[x205];
float x208 = x206 * x207;
x198 += x208;

}
x195 += 5;
x197 += 28;

}
int32_t x216 = x188;
float x217 = x183[x216];
float x218 = x198;
float x219 = x217 + x218;
x183[x216] = x219;
x188 += 1;
x193 += 1;

}
x189 += 28;

}
x180 += 25;
x181 += 784;
x177 += 25;
x176 += 576;

}
float* x234 = (float*)myMalloc(5760 * sizeof(float));
for(int x235=0; x235 < 5760; x235++) {
x234[x235] = 0.0f;

}
float* x239 = (float*)myMalloc(1440 * sizeof(float));
for(int x241=0; x241 < 1440; x241++) {
x239[x241] = -1.0E10f;

}
int32_t* x245 = (int32_t*)myMalloc(1440 * sizeof(int32_t));
int32_t x246 = 0;
int32_t x247 = 0;
for(int x248=0; x248 < 10; x248++) {
int32_t x249 = x246;
int32_t x250 = x249;
for(int x252=0; x252 < 12; x252++) {
for(int x254=0; x254 < 2; x254++) {
int32_t x255 = x250;
int32_t x256 = x255;
for(int x257=0; x257 < 12; x257++) {
int32_t x258 = x247;
float x259 = x170[x258];
int32_t x260 = x256;
float x261 = x239[x260];
bool x262 = x259 > x261;
if (x262) {
float x263 = x170[x258];
x239[x260] = x263;
x245[x260] = x258;
} else {
}
x247 += 1;
int32_t x269 = x247;
float x270 = x170[x269];
float x271 = x239[x260];
bool x272 = x270 > x271;
if (x272) {
float x273 = x170[x269];
x239[x260] = x273;
x245[x260] = x269;
} else {
}
x247 += 1;
x256 += 1;

}

}
x250 += 12;

}
x246 += 144;

}
float* x290 = (float*)myMalloc(1440 * sizeof(float));
for(int x291=0; x291 < 1440; x291++) {
x290[x291] = 0.0f;

}
float* x295 = (float*)myMalloc(1440 * sizeof(float));
for(int x296=0; x296 < 1440; x296++) {
float x297 = x239[x296];
bool x298 = x297 < 0.0f;
if (x298) {
x295[x296] = 0.0f;
} else {
float x301 = x239[x296];
x295[x296] = x301;
}

}
float* x307 = (float*)myMalloc(1440 * sizeof(float));
for(int x308=0; x308 < 1440; x308++) {
x307[x308] = 0.0f;

}
float* x312 = (float*)myMalloc(1280 * sizeof(float));
for(int x314=0; x314 < 1280; x314++) {
x312[x314] = 0.0f;

}
int32_t x318 = 0;
int32_t x319 = 0;
for(int x321=0; x321 < 20; x321++) {
int32_t x322 = x319;
int32_t x323 = x322;
int32_t x324 = 0;
int32_t x325 = x318;
float* x326 = x312+x325;
for(int x327=0; x327 < 10; x327++) {
int32_t x328 = x324;
float* x329 = x295+x328;
int32_t x330 = x323;
float* x331 = x19+x330;
int32_t x332 = 0;
int32_t x333 = 0;
for(int x335=0; x335 < 8; x335++) {
int32_t x336 = x333;
int32_t x337 = x336;
for(int x338=0; x338 < 8; x338++) {
int32_t x339 = 0;
int32_t x340 = x337;
int32_t x341 = x340;
float x342 = 0.0f;
for(int x343=0; x343 < 5; x343++) {
int32_t x344 = x341;
float* x345 = x329+x344;
int32_t x346 = x339;
float* x347 = x331+x346;
for(int x348=0; x348 < 5; x348++) {
float x349 = x345[x348];
float x350 = x347[x348];
float x351 = x349 * x350;
x342 += x351;

}
x339 += 5;
x341 += 12;

}
int32_t x359 = x332;
float x360 = x326[x359];
float x361 = x342;
float x362 = x360 + x361;
x326[x359] = x362;
x332 += 1;
x337 += 1;

}
x333 += 12;

}
x323 += 25;
x324 += 144;

}
x319 += 250;
x318 += 64;

}
float* x379 = (float*)myMalloc(1280 * sizeof(float));
for(int x380=0; x380 < 1280; x380++) {
x379[x380] = 0.0f;

}
float* x384 = (float*)myMalloc(320 * sizeof(float));
for(int x386=0; x386 < 320; x386++) {
x384[x386] = -1.0E10f;

}
int32_t* x390 = (int32_t*)myMalloc(320 * sizeof(int32_t));
int32_t x391 = 0;
int32_t x392 = 0;
for(int x393=0; x393 < 20; x393++) {
int32_t x394 = x391;
int32_t x395 = x394;
for(int x397=0; x397 < 4; x397++) {
for(int x398=0; x398 < 2; x398++) {
int32_t x399 = x395;
int32_t x400 = x399;
for(int x401=0; x401 < 4; x401++) {
int32_t x402 = x392;
float x403 = x312[x402];
int32_t x404 = x400;
float x405 = x384[x404];
bool x406 = x403 > x405;
if (x406) {
float x407 = x312[x402];
x384[x404] = x407;
x390[x404] = x402;
} else {
}
x392 += 1;
int32_t x413 = x392;
float x414 = x312[x413];
float x415 = x384[x404];
bool x416 = x414 > x415;
if (x416) {
float x417 = x312[x413];
x384[x404] = x417;
x390[x404] = x413;
} else {
}
x392 += 1;
x400 += 1;

}

}
x395 += 4;

}
x391 += 16;

}
float* x434 = (float*)myMalloc(320 * sizeof(float));
for(int x435=0; x435 < 320; x435++) {
x434[x435] = 0.0f;

}
float* x439 = (float*)myMalloc(320 * sizeof(float));
for(int x440=0; x440 < 320; x440++) {
float x441 = x384[x440];
bool x442 = x441 < 0.0f;
if (x442) {
x439[x440] = 0.0f;
} else {
float x445 = x384[x440];
x439[x440] = x445;
}

}
float* x451 = (float*)myMalloc(320 * sizeof(float));
for(int x452=0; x452 < 320; x452++) {
x451[x452] = 0.0f;

}
// dot WrappedArray(50, 320) - WrappedArray(320)
int32_t x457 = 0;
float* x458 = (float*)myMalloc(50 * sizeof(float));
for(int x459=0; x459 < 50; x459++) {
float x460 = 0.0f;
for(int x461=0; x461 < 320; x461++) {
int32_t x462 = x457;
float x463 = x33[x462];
float x464 = x439[x461];
float x465 = x463 * x464;
x460 += x465;
x457 += 1;

}
float x470 = x460;
x458[x459] = x470;

}
float* x474 = (float*)myMalloc(50 * sizeof(float));
for(int x475=0; x475 < 50; x475++) {
x474[x475] = 0.0f;

}
float* x479 = (float*)myMalloc(50 * sizeof(float));
for(int x480=0; x480 < 50; x480++) {
float x481 = x458[x480];
float x482 = x42[x480];
float x483 = x481 + x482;
x479[x480] = x483;

}
float* x487 = (float*)myMalloc(50 * sizeof(float));
for(int x488=0; x488 < 50; x488++) {
x487[x488] = 0.0f;

}
float* x492 = (float*)myMalloc(50 * sizeof(float));
for(int x493=0; x493 < 50; x493++) {
float x494 = x479[x493];
bool x495 = x494 < 0.0f;
if (x495) {
x492[x493] = 0.0f;
} else {
float x498 = x479[x493];
x492[x493] = x498;
}

}
float* x504 = (float*)myMalloc(50 * sizeof(float));
for(int x505=0; x505 < 50; x505++) {
x504[x505] = 0.0f;

}
float* x509 = (float*)myMalloc(50 * sizeof(float));
float* x510 = (float*)myMalloc(50 * sizeof(float));
for(int x511=0; x511 < 50; x511++) {
float x512 = (float)rand()/RAND_MAX;
bool x513 = x512 > 0.5f;
if (x513) {
float x514 = x492[x511];
float x515 = x514 * 2.0f;
x509[x511] = x515;
x510[x511] = 2.0f;
} else {
x509[x511] = 0.0f;
x510[x511] = 0.0f;
}

}
float* x525 = (float*)myMalloc(50 * sizeof(float));
for(int x526=0; x526 < 50; x526++) {
x525[x526] = 0.0f;

}
// dot WrappedArray(10, 50) - WrappedArray(50)
int32_t x531 = 0;
float* x532 = (float*)myMalloc(10 * sizeof(float));
for(int x533=0; x533 < 10; x533++) {
float x534 = 0.0f;
for(int x535=0; x535 < 50; x535++) {
int32_t x536 = x531;
float x537 = x61[x536];
float x538 = x509[x535];
float x539 = x537 * x538;
x534 += x539;
x531 += 1;

}
float x544 = x534;
x532[x533] = x544;

}
float* x548 = (float*)myMalloc(10 * sizeof(float));
for(int x549=0; x549 < 10; x549++) {
x548[x549] = 0.0f;

}
float* x553 = (float*)myMalloc(10 * sizeof(float));
for(int x554=0; x554 < 10; x554++) {
float x555 = x532[x554];
float x556 = x70[x554];
float x557 = x555 + x556;
x553[x554] = x557;

}
float* x561 = (float*)myMalloc(10 * sizeof(float));
for(int x562=0; x562 < 10; x562++) {
x561[x562] = 0.0f;

}
float x566 = -1.0E10f;
for(int x567=0; x567 < 10; x567++) {
float x568 = x566;
float x569 = x553[x567];
bool x570 = x569 > x568;
float x571;
if (x570) {
x571 = x569;
} else {
x571 = x568;
}
x566 = x571;

}
float x575 = x566;
float x576 = 0.0f;
for(int x577=0; x577 < 10; x577++) {
float x578 = x576;
float x579 = x553[x577];
float x580 = x566;
float x581 = x579 - x580;
double x582 = (double)x581;
double x583 = exp(x582);
float x584 = (float)x583;
float x585 = x578 + x584;
x576 = x585;

}
float x589 = x576;
float* x594 = (float*)myMalloc(10 * sizeof(float));
double x590 = (double)x589;
double x591 = log(x590);
float x592 = (float)x591;
float x593 = x575 + x592;
for(int x595=0; x595 < 10; x595++) {
float x596 = x553[x595];
float x597 = x596 - x593;
x594[x595] = x597;

}
float* x601 = (float*)myMalloc(10 * sizeof(float));
for(int x602=0; x602 < 10; x602++) {
x601[x602] = 0.0f;

}
float x606 = x594[x153];
float* x608 = (float*)myMalloc(1 * sizeof(float));
float x607 = -1.0f * x606;
x608[0] = x607;
float* x610 = (float*)myMalloc(1 * sizeof(float));
for(int x611=0; x611 < 1; x611++) {
x610[x611] = 0.0f;

}
float x615 = x610[0];
x610[0] = 1.0f;
float x617 = x608[0];
x165[0] = x617;
float x619 = x610[0];
float x620 = -1.0f * x619;
x601[x153] = x620;
float x622 = 0.0f;
for(int x623=0; x623 < 10; x623++) {
float x624 = x622;
float x625 = x601[x623];
float x626 = x624 + x625;
x622 = x626;

}
float x630 = x622;
float* x631 = (float*)myMalloc(1 * sizeof(float));
x631[0] = x630;
float x633 = x631[0];
for(int x634=0; x634 < 10; x634++) {
float x635 = x601[x634];
float x636 = x594[x634];
double x637 = (double)x636;
double x638 = exp(x637);
float x639 = (float)x638;
float x640 = x639 * x633;
float x641 = x635 - x640;
x561[x634] = x641;

}
// backpropagate +
for(int x646=0; x646 < 10; x646++) {
float x647 = x548[x646];
float x648 = x561[x646];
float x649 = x647 + x648;
x548[x646] = x649;

}
for(int x653=0; x653 < 10; x653++) {
float x654 = x84[x653];
float x655 = x561[x653];
float x656 = x654 + x655;
x84[x653] = x656;

}
// add_cartesian
int32_t x661 = 0;
for(int x662=0; x662 < 10; x662++) {
for(int x663=0; x663 < 50; x663++) {
int32_t x664 = x661;
int32_t x665 = x664 + x663;
float x666 = x79[x665];
float x667 = x509[x663];
float x668 = x548[x662];
float x669 = x667 * x668;
float x670 = x666 + x669;
x79[x665] = x670;

}
x661 += 50;

}
int32_t x677 = 0;
for(int x678=0; x678 < 10; x678++) {
for(int x679=0; x679 < 50; x679++) {
float x680 = x525[x679];
int32_t x681 = x677;
int32_t x682 = x681 + x679;
float x683 = x61[x682];
float x684 = x548[x678];
float x685 = x683 * x684;
float x686 = x680 + x685;
x525[x679] = x686;

}
x677 += 50;

}
float* x693 = (float*)myMalloc(50 * sizeof(float));
for(int x694=0; x694 < 50; x694++) {
float x695 = x510[x694];
float x696 = x525[x694];
float x697 = x695 * x696;
x693[x694] = x697;

}
for(int x701=0; x701 < 50; x701++) {
float x702 = x504[x701];
float x703 = x693[x701];
float x704 = x702 + x703;
x504[x701] = x704;

}
for(int x708=0; x708 < 50; x708++) {
float x709 = x479[x708];
bool x710 = x709 < 0.0f;
float x713;
if (x710) {
x713 = 0.0f;
} else {
float x711 = x504[x708];
x713 = x711;
}
x487[x708] = x713;

}
// backpropagate +
for(int x718=0; x718 < 50; x718++) {
float x719 = x474[x718];
float x720 = x487[x718];
float x721 = x719 + x720;
x474[x718] = x721;

}
for(int x725=0; x725 < 50; x725++) {
float x726 = x56[x725];
float x727 = x487[x725];
float x728 = x726 + x727;
x56[x725] = x728;

}
// add_cartesian
int32_t x733 = 0;
for(int x734=0; x734 < 50; x734++) {
for(int x735=0; x735 < 320; x735++) {
int32_t x736 = x733;
int32_t x737 = x736 + x735;
float x738 = x51[x737];
float x739 = x439[x735];
float x740 = x474[x734];
float x741 = x739 * x740;
float x742 = x738 + x741;
x51[x737] = x742;

}
x733 += 320;

}
int32_t x749 = 0;
for(int x750=0; x750 < 50; x750++) {
for(int x751=0; x751 < 320; x751++) {
float x752 = x451[x751];
int32_t x753 = x749;
int32_t x754 = x753 + x751;
float x755 = x33[x754];
float x756 = x474[x750];
float x757 = x755 * x756;
float x758 = x752 + x757;
x451[x751] = x758;

}
x749 += 320;

}
for(int x765=0; x765 < 320; x765++) {
float x766 = x384[x765];
bool x767 = x766 < 0.0f;
float x770;
if (x767) {
x770 = 0.0f;
} else {
float x768 = x451[x765];
x770 = x768;
}
x434[x765] = x770;

}
for(int x774=0; x774 < 320; x774++) {
int32_t x775 = x390[x774];
float x776 = x434[x774];
x379[x775] = x776;

}
int32_t x780 = 0;
int32_t x781 = 0;
for(int x782=0; x782 < 20; x782++) {
int32_t x783 = 0;
for(int x784=0; x784 < 8; x784++) {
int32_t x785 = x783;
int32_t x786 = x785;
for(int x787=0; x787 < 8; x787++) {
int32_t x788 = x780;
float x789 = x379[x788];
int32_t x790 = x786;
int32_t x791 = x790;
int32_t x792 = x781;
int32_t x793 = x792;
for(int x794=0; x794 < 10; x794++) {
int32_t x795 = x791;
int32_t x796 = x795;
for(int x797=0; x797 < 5; x797++) {
for(int x798=0; x798 < 5; x798++) {
int32_t x799 = x796;
int32_t x800 = x799 + x798;
float x801 = x307[x800];
int32_t x802 = x793;
float x803 = x19[x802];
float x804 = x789 * x803;
float x805 = x801 + x804;
x307[x800] = x805;
float x807 = x28[x802];
float x808 = x295[x800];
float x809 = x789 * x808;
float x810 = x807 + x809;
x28[x802] = x810;
x793 += 1;

}
x796 += 12;

}
x791 += 144;

}
x786 += 1;
x780 += 1;

}
x783 += 12;

}
x781 += 250;

}
for(int x831=0; x831 < 1440; x831++) {
float x832 = x239[x831];
bool x833 = x832 < 0.0f;
float x836;
if (x833) {
x836 = 0.0f;
} else {
float x834 = x307[x831];
x836 = x834;
}
x290[x831] = x836;

}
for(int x840=0; x840 < 1440; x840++) {
int32_t x841 = x245[x840];
float x842 = x290[x840];
x234[x841] = x842;

}
int32_t x846 = 0;
int32_t x847 = 0;
for(int x848=0; x848 < 10; x848++) {
int32_t x849 = 0;
for(int x850=0; x850 < 24; x850++) {
int32_t x851 = x849;
int32_t x852 = x851;
for(int x853=0; x853 < 24; x853++) {
int32_t x854 = x846;
float x855 = x234[x854];
int32_t x856 = x852;
int32_t x857 = x856;
int32_t x858 = x847;
int32_t x859 = x858;
for(int x860=0; x860 < 1; x860++) {
int32_t x861 = x857;
int32_t x862 = x861;
for(int x863=0; x863 < 5; x863++) {
for(int x864=0; x864 < 5; x864++) {
int32_t x865 = x859;
float x866 = x14[x865];
int32_t x867 = x862;
int32_t x868 = x867 + x864;
float x869 = x152[x868];
float x870 = x855 * x869;
float x871 = x866 + x870;
x14[x865] = x871;
x859 += 1;

}
x862 += 28;

}
x857 += 784;

}
x852 += 1;
x846 += 1;

}
x849 += 28;

}
x847 += 25;

}
float x892 = x165[0];
x143 += x892;
// Generate code for addMul
for(int x895=0; x895 < 250; x895++) {
float x896 = x5[x895];
float x897 = x14[x895];
float x898 = -5.0E-4f * x897;
float x899 = x896 + x898;
x5[x895] = x899;

}
for(int x903=0; x903 < 250; x903++) {
float x904 = x14[x903];
x14[x903] = 0.0f;

}
// Generate code for addMul
for(int x909=0; x909 < 5000; x909++) {
float x910 = x19[x909];
float x911 = x28[x909];
float x912 = -5.0E-4f * x911;
float x913 = x910 + x912;
x19[x909] = x913;

}
for(int x917=0; x917 < 5000; x917++) {
float x918 = x28[x917];
x28[x917] = 0.0f;

}
// Generate code for addMul
for(int x923=0; x923 < 16000; x923++) {
float x924 = x33[x923];
float x925 = x51[x923];
float x926 = -5.0E-4f * x925;
float x927 = x924 + x926;
x33[x923] = x927;

}
for(int x931=0; x931 < 16000; x931++) {
float x932 = x51[x931];
x51[x931] = 0.0f;

}
// Generate code for addMul
for(int x937=0; x937 < 50; x937++) {
float x938 = x42[x937];
float x939 = x56[x937];
float x940 = -5.0E-4f * x939;
float x941 = x938 + x940;
x42[x937] = x941;

}
for(int x945=0; x945 < 50; x945++) {
float x946 = x56[x945];
x56[x945] = 0.0f;

}
// Generate code for addMul
for(int x951=0; x951 < 500; x951++) {
float x952 = x61[x951];
float x953 = x79[x951];
float x954 = -5.0E-4f * x953;
float x955 = x952 + x954;
x61[x951] = x955;

}
for(int x959=0; x959 < 500; x959++) {
float x960 = x79[x959];
x79[x959] = 0.0f;

}
// Generate code for addMul
for(int x965=0; x965 < 10; x965++) {
float x966 = x70[x965];
float x967 = x84[x965];
float x968 = -5.0E-4f * x967;
float x969 = x966 + x968;
x70[x965] = x969;

}
for(int x973=0; x973 < 10; x973++) {
float x974 = x84[x973];
x84[x973] = 0.0f;

}
int32_t x978 = x140;
int32_t x980 = x978 % x979;
bool x981 = x980 == 0;
if (x981) {
float x986 = x143;
double x982 = (double)x978;
double x983 = 100.0 * x982;
double x985 = x983 / x984;
float x987 = (float)x978;
float x988 = x986 / x987;
printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x136,x978,x100,x985,x988);
fflush(stdout);
} else {
}
mallocAddr = (void*)x135;
x149 += 784;

}
int32_t x997 = x149;
bool x998 = x997 == x95;
if (x998) {
} else {
printf("Data length doesn't match\n");
exit(0);
}
gettimeofday(&end_1, NULL);
timeval_subtract(&diff_1, &end_1, &begin_1);;
int64_t x1006 = ((diff_1.tv_sec * 1000000L) + (diff_1.tv_usec));
int64_t x1007 = x1006 / 1000LL;
int64_t x1009 = x1006 / x1008;
printf("Training completed in %ldms (%ld us/images)\n",x1007,x1009);
float x1011 = x143;
float x1013 = x1011 / x1012;
double x1014 = (double)x1013;
x134[x136] = x1014;

}
gettimeofday(&end_0, NULL);
timeval_subtract(&diff_0, &end_0, &begin_0);;
int64_t x1020 = ((diff_0.tv_sec * 1000000L) + (diff_0.tv_usec));
int64_t x1025 = (long)fopen(x0, "w");
fprintf((FILE *)x1025, "unit: %s\n", "1 epoch");
for(int x1027=0; x1027 < 10; x1027++) {
double x1028 = x134[x1027];
fprintf((FILE *)x1025, "%lf\n", x1028);

}
float x1021 = (float)x1020;
float x1022 = x1021 / 1000000.0f;
float x1023 = x1022 - x132;
float x1024 = x1023 / 10.0f;
fprintf((FILE *)x1025, "run time: %lf %lf\n", x132, x1024);
fclose((FILE*)x1025);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

