
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
int64_t x1 = (long)fopen("senti/small_glove.txt", "r");
float** x2 = (float**)myMalloc(5265 * sizeof(float*));
for(int x4=0; x4 < 5265; x4++) {
float* x5 = (float*)myMalloc(300 * sizeof(float));
x2[x4] = x5;
for(int x8=0; x8 < 300; x8++) {
float* x9 = x2[x4];
if (fscanf((FILE *)x1,"%f", &x9[x8])!=1) perror("Error reading file");

}

}
fclose((FILE*)x1);
int64_t x16 = (long)fopen("senti/array_tree.txt", "r");
int** x17 = (int**)myMalloc(4404 * sizeof(int*));
int32_t* x18 = (int32_t*)myMalloc(1 * sizeof(int32_t));
for(int x20=0; x20 < 1101; x20++) {
if (fscanf((FILE *)x16,"%d", &x18[0])!=1) perror("Error reading file");
int32_t x24 = x20 * 4;
for(int x23=0; x23 < 4; x23++) {
int32_t x26 = x18[0];
int32_t* x27 = (int32_t*)myMalloc(x26 * sizeof(int32_t));
int32_t x25 = x24 + x23;
x17[x25] = x27;
int32_t x29 = x18[0];
for(int x31=0; x31 < x29; x31++) {
int* x32 = x17[x25];
if (fscanf((FILE *)x16,"%d", &x32[x31])!=1) perror("Error reading file");

}

}

}
float* x40 = (float*)myMalloc(30000 * sizeof(float));
for(int x42=0; x42 < 30000; x42++) {
float x43 = (float)rand()/RAND_MAX;
float x44 = x43 - 0.5f;
float x45 = x44 * 0.01f;
x40[x42] = x45;

}
float* x49 = (float*)myMalloc(100 * sizeof(float));
for(int x51=0; x51 < 100; x51++) {
x49[x51] = 0.0f;

}
float* x55 = (float*)myMalloc(10000 * sizeof(float));
for(int x57=0; x57 < 10000; x57++) {
float x58 = (float)rand()/RAND_MAX;
float x59 = x58 - 0.5f;
float x60 = x59 * 0.01f;
x55[x57] = x60;

}
float* x64 = (float*)myMalloc(10000 * sizeof(float));
for(int x65=0; x65 < 10000; x65++) {
float x66 = (float)rand()/RAND_MAX;
float x67 = x66 - 0.5f;
float x68 = x67 * 0.01f;
x64[x65] = x68;

}
float* x72 = (float*)myMalloc(100 * sizeof(float));
for(int x73=0; x73 < 100; x73++) {
x72[x73] = 0.0f;

}
float* x77 = (float*)myMalloc(500 * sizeof(float));
for(int x79=0; x79 < 500; x79++) {
float x80 = (float)rand()/RAND_MAX;
float x81 = x80 - 0.5f;
float x82 = x81 * 0.01f;
x77[x79] = x82;

}
float* x86 = (float*)myMalloc(5 * sizeof(float));
for(int x88=0; x88 < 5; x88++) {
x86[x88] = 0.0f;

}
float* x92 = (float*)myMalloc(30000 * sizeof(float));
for(int x93=0; x93 < 30000; x93++) {
x92[x93] = 0.0f;

}
float* x97 = (float*)myMalloc(100 * sizeof(float));
for(int x98=0; x98 < 100; x98++) {
x97[x98] = 0.0f;

}
float* x102 = (float*)myMalloc(10000 * sizeof(float));
for(int x103=0; x103 < 10000; x103++) {
x102[x103] = 0.0f;

}
float* x107 = (float*)myMalloc(10000 * sizeof(float));
for(int x108=0; x108 < 10000; x108++) {
x107[x108] = 0.0f;

}
float* x112 = (float*)myMalloc(100 * sizeof(float));
for(int x113=0; x113 < 100; x113++) {
x112[x113] = 0.0f;

}
float* x117 = (float*)myMalloc(500 * sizeof(float));
for(int x118=0; x118 < 500; x118++) {
x117[x118] = 0.0f;

}
float* x122 = (float*)myMalloc(5 * sizeof(float));
for(int x123=0; x123 < 5; x123++) {
x122[x123] = 0.0f;

}
float* x127 = (float*)myMalloc(30000 * sizeof(float));
for(int x128=0; x128 < 30000; x128++) {
x127[x128] = 0.0f;

}
float* x132 = (float*)myMalloc(100 * sizeof(float));
for(int x133=0; x133 < 100; x133++) {
x132[x133] = 0.0f;

}
float* x137 = (float*)myMalloc(10000 * sizeof(float));
for(int x138=0; x138 < 10000; x138++) {
x137[x138] = 0.0f;

}
float* x142 = (float*)myMalloc(10000 * sizeof(float));
for(int x143=0; x143 < 10000; x143++) {
x142[x143] = 0.0f;

}
float* x147 = (float*)myMalloc(100 * sizeof(float));
for(int x148=0; x148 < 100; x148++) {
x147[x148] = 0.0f;

}
float* x152 = (float*)myMalloc(500 * sizeof(float));
for(int x153=0; x153 < 500; x153++) {
x152[x153] = 0.0f;

}
float* x157 = (float*)myMalloc(5 * sizeof(float));
for(int x158=0; x158 < 5; x158++) {
x157[x158] = 0.0f;

}
int64_t x162 = (long)mallocAddr;
for(int x164=0; x164 < 10; x164++) {
double x165 = 0.0;
for(int x166=0; x166 < 1101; x166++) {
int32_t x167 = x166 % 1101;
int32_t x168 = x167 * 4;
int* x169 = x17[x168];
int32_t x170 = x168 + 1;
int* x171 = x17[x170];
int32_t x172 = x168 + 2;
int* x173 = x17[x172];
int32_t x174 = x168 + 3;
int* x175 = x17[x174];
function<void(int32_t,function<void(float**)>,float**)> x212 = [&](int32_t x213,function<void(float**)> x214,float** x215) {
float** x218 = x215;
float* x219 = x218[0];
float* x220 = x218[1];
float* x221 = x218[2];
float* x222 = x218[3];
int32_t x216 = x213;
bool x223 = x216 >= 0;
if (x223) {
int32_t x224 = x173[x216];
float** x857 = (float**)myMalloc(4 * sizeof(float*));
x857[0] = x219;
x857[1] = x220;
x857[2] = x221;
x857[3] = x222;
function<void(float**)> x217 = x214;
function<void(float**)> x225 = [&](float** x226) {
float* x227 = x226[0];
float* x228 = x226[1];
float* x229 = x226[2];
float* x230 = x226[3];
int32_t x231 = x175[x216];
float** x849 = (float**)myMalloc(4 * sizeof(float*));
x849[0] = x219;
x849[1] = x220;
x849[2] = x221;
x849[3] = x222;
function<void(float**)> x232 = [&](float** x233) {
float* x234 = x233[0];
float* x235 = x233[1];
float* x236 = x233[2];
float* x237 = x233[3];
float* x238 = (float*)myMalloc(5 * sizeof(float));
for(int x239=0; x239 < 5; x239++) {
x238[x239] = 0.0f;

}
int32_t x243 = x169[x216];
x238[x243] = 1.0f;
float* x245 = (float*)myMalloc(5 * sizeof(float));
for(int x246=0; x246 < 5; x246++) {
x245[x246] = 0.0f;

}
int32_t x250 = x173[x216];
bool x251 = x250 < 0;
if (x251) {
int32_t x526 = x171[x216];
float* x527 = x2[x526];
float* x528 = (float*)myMalloc(300 * sizeof(float));
for(int x529=0; x529 < 300; x529++) {
x528[x529] = 0.0f;

}
// dot List(100, 300) - WrappedArray(300)
int32_t x534 = 0;
float* x535 = (float*)myMalloc(100 * sizeof(float));
for(int x536=0; x536 < 100; x536++) {
float x537 = 0.0f;
for(int x538=0; x538 < 300; x538++) {
int32_t x539 = x534;
float x540 = x40[x539];
float x541 = x527[x538];
float x542 = x540 * x541;
x537 += x542;
x534 += 1;

}
float x547 = x537;
x535[x536] = x547;

}
float* x551 = (float*)myMalloc(100 * sizeof(float));
for(int x552=0; x552 < 100; x552++) {
x551[x552] = 0.0f;

}
float* x556 = (float*)myMalloc(100 * sizeof(float));
for(int x557=0; x557 < 100; x557++) {
float x558 = x535[x557];
float x559 = x49[x557];
float x560 = x558 + x559;
x556[x557] = x560;

}
float* x564 = (float*)myMalloc(100 * sizeof(float));
for(int x565=0; x565 < 100; x565++) {
x564[x565] = 0.0f;

}
float* x569 = (float*)myMalloc(100 * sizeof(float));
for(int x570=0; x570 < 100; x570++) {
float x571 = x556[x570];
double x572 = (double)x571;
double x573 = tanh(x572);
float x574 = (float)x573;
x569[x570] = x574;

}
float* x578 = (float*)myMalloc(100 * sizeof(float));
for(int x579=0; x579 < 100; x579++) {
x578[x579] = 0.0f;

}
float** x583 = (float**)myMalloc(2 * sizeof(float*));
x583[0] = x569;
x583[1] = x578;
function<void(float**)> x252 = [&](float** x253) {
float* x254 = x253[0];
float* x255 = x253[1];
// dot List(5, 100) - WrappedArray(100)
int32_t x257 = 0;
float* x258 = (float*)myMalloc(5 * sizeof(float));
for(int x259=0; x259 < 5; x259++) {
float x260 = 0.0f;
for(int x261=0; x261 < 100; x261++) {
int32_t x262 = x257;
float x263 = x77[x262];
float x264 = x254[x261];
float x265 = x263 * x264;
x260 += x265;
x257 += 1;

}
float x270 = x260;
x258[x259] = x270;

}
float* x274 = (float*)myMalloc(5 * sizeof(float));
for(int x275=0; x275 < 5; x275++) {
x274[x275] = 0.0f;

}
float* x279 = (float*)myMalloc(5 * sizeof(float));
for(int x280=0; x280 < 5; x280++) {
float x281 = x258[x280];
float x282 = x86[x280];
float x283 = x281 + x282;
x279[x280] = x283;

}
float* x287 = (float*)myMalloc(5 * sizeof(float));
for(int x288=0; x288 < 5; x288++) {
x287[x288] = 0.0f;

}
float* x292 = (float*)myMalloc(5 * sizeof(float));
for(int x293=0; x293 < 5; x293++) {
float x294 = x279[x293];
double x295 = (double)x294;
double x296 = exp(x295);
float x297 = (float)x296;
x292[x293] = x297;

}
float* x301 = (float*)myMalloc(5 * sizeof(float));
for(int x302=0; x302 < 5; x302++) {
x301[x302] = 0.0f;

}
float x306 = 0.0f;
for(int x307=0; x307 < 5; x307++) {
float x308 = x306;
float x309 = x292[x307];
float x310 = x308 + x309;
x306 = x310;

}
float x314 = x306;
float* x315 = (float*)myMalloc(1 * sizeof(float));
x315[0] = x314;
float* x317 = (float*)myMalloc(1 * sizeof(float));
for(int x318=0; x318 < 1; x318++) {
x317[x318] = 0.0f;

}
float x322 = x315[0];
float* x323 = (float*)myMalloc(5 * sizeof(float));
for(int x324=0; x324 < 5; x324++) {
float x325 = x292[x324];
float x326 = x325 / x322;
x323[x324] = x326;

}
float* x330 = (float*)myMalloc(5 * sizeof(float));
for(int x331=0; x331 < 5; x331++) {
x330[x331] = 0.0f;

}
float x335 = x227[0];
float* x336 = (float*)myMalloc(1 * sizeof(float));
float x337 = x234[0];
float x338 = x337 + x335;
x336[0] = x338;
float* x340 = (float*)myMalloc(1 * sizeof(float));
for(int x341=0; x341 < 1; x341++) {
x340[x341] = 0.0f;

}
// dot WrappedArray(5) - WrappedArray(5)
int32_t x346 = 0;
float* x347 = (float*)myMalloc(1 * sizeof(float));
float x348 = 0.0f;
for(int x349=0; x349 < 5; x349++) {
int32_t x350 = x346;
float x351 = x323[x350];
float x352 = x238[x349];
float x353 = x351 * x352;
x348 += x353;
x346 += 1;

}
float x358 = x348;
x347[0] = x358;
float* x360 = (float*)myMalloc(1 * sizeof(float));
for(int x361=0; x361 < 1; x361++) {
x360[x361] = 0.0f;

}
float* x365 = (float*)myMalloc(1 * sizeof(float));
float x366 = x347[0];
double x367 = (double)x366;
double x368 = log(x367);
float x369 = (float)x368;
x365[0] = x369;
float* x371 = (float*)myMalloc(1 * sizeof(float));
for(int x372=0; x372 < 1; x372++) {
x371[x372] = 0.0f;

}
float* x376 = (float*)myMalloc(1 * sizeof(float));
float x377 = x365[0];
float x378 = x336[0];
float x379 = x378 - x377;
x376[0] = x379;
float* x381 = (float*)myMalloc(1 * sizeof(float));
for(int x382=0; x382 < 1; x382++) {
x381[x382] = 0.0f;

}
float** x386 = (float**)myMalloc(4 * sizeof(float*));
x386[0] = x376;
x386[1] = x381;
x386[2] = x254;
x386[3] = x255;
x217(x386);
// += tensor of dim 0
float x393 = x381[0];
float x394 = x340[0];
float x395 = x394 + x393;
x340[0] = x395;
float x397 = x381[0];
float x398 = x371[0];
float x399 = x398 - x397;
x371[0] = x399;
float x401 = x360[0];
float x402 = x371[0];
float x403 = x347[0];
float x404 = x402 / x403;
float x405 = x401 + x404;
x360[0] = x405;
float x407 = x360[0];
// Generate code for addMul
for(int x409=0; x409 < 5; x409++) {
float x410 = x330[x409];
float x411 = x238[x409];
float x412 = x407 * x411;
float x413 = x410 + x412;
x330[x409] = x413;

}
float x417 = x360[0];
// Generate code for addMul
for(int x419=0; x419 < 5; x419++) {
float x420 = x245[x419];
float x421 = x323[x419];
float x422 = x417 * x421;
float x423 = x420 + x422;
x245[x419] = x423;

}
// backpropagate +
// += tensor of dim 0
float x429 = x340[0];
float x430 = x228[0];
float x431 = x430 + x429;
x228[0] = x431;
// += tensor of dim 0
float x434 = x340[0];
float x435 = x235[0];
float x436 = x435 + x434;
x235[0] = x436;
for(int x438=0; x438 < 5; x438++) {
float x439 = x301[x438];
float x440 = x330[x438];
float x441 = x315[0];
float x442 = x440 / x441;
float x443 = x439 + x442;
x301[x438] = x443;

}
for(int x447=0; x447 < 5; x447++) {
float x448 = x317[0];
float x449 = x292[x447];
float x450 = x330[x447];
float x452 = x315[0];
float x451 = x449 * x450;
float x453 = x452 * x452;
float x454 = x451 / x453;
float x455 = x448 - x454;
x317[0] = x455;

}
// += tensor of dim 0
float x460 = x317[0];
for(int x461=0; x461 < 5; x461++) {
float x462 = x301[x461];
float x463 = x462 + x460;
x301[x461] = x463;

}
// backpropage exp
for(int x468=0; x468 < 5; x468++) {
float x469 = x287[x468];
float x470 = x292[x468];
float x471 = x301[x468];
float x472 = x470 * x471;
float x473 = x469 + x472;
x287[x468] = x473;

}
// backpropagate +
for(int x478=0; x478 < 5; x478++) {
float x479 = x274[x478];
float x480 = x287[x478];
float x481 = x479 + x480;
x274[x478] = x481;

}
for(int x485=0; x485 < 5; x485++) {
float x486 = x122[x485];
float x487 = x287[x485];
float x488 = x486 + x487;
x122[x485] = x488;

}
// add_cartesian
int32_t x493 = 0;
for(int x494=0; x494 < 5; x494++) {
for(int x495=0; x495 < 100; x495++) {
int32_t x496 = x493;
int32_t x497 = x496 + x495;
float x498 = x117[x497];
float x499 = x254[x495];
float x500 = x274[x494];
float x501 = x499 * x500;
float x502 = x498 + x501;
x117[x497] = x502;

}
x493 += 100;

}
int32_t x509 = 0;
for(int x510=0; x510 < 5; x510++) {
for(int x511=0; x511 < 100; x511++) {
float x512 = x255[x511];
int32_t x513 = x509;
int32_t x514 = x513 + x511;
float x515 = x77[x514];
float x516 = x274[x510];
float x517 = x515 * x516;
float x518 = x512 + x517;
x255[x511] = x518;

}
x509 += 100;

}
};
x252(x583);
// backpropagate tanh
for(int x588=0; x588 < 100; x588++) {
float x589 = x564[x588];
float x590 = x569[x588];
float x593 = x578[x588];
float x591 = x590 * x590;
float x592 = 1.0f - x591;
float x594 = x592 * x593;
float x595 = x589 + x594;
x564[x588] = x595;

}
// backpropagate +
for(int x600=0; x600 < 100; x600++) {
float x601 = x551[x600];
float x602 = x564[x600];
float x603 = x601 + x602;
x551[x600] = x603;

}
for(int x607=0; x607 < 100; x607++) {
float x608 = x97[x607];
float x609 = x564[x607];
float x610 = x608 + x609;
x97[x607] = x610;

}
// add_cartesian
int32_t x615 = 0;
for(int x616=0; x616 < 100; x616++) {
for(int x617=0; x617 < 300; x617++) {
int32_t x618 = x615;
int32_t x619 = x618 + x617;
float x620 = x92[x619];
float x621 = x527[x617];
float x622 = x551[x616];
float x623 = x621 * x622;
float x624 = x620 + x623;
x92[x619] = x624;

}
x615 += 300;

}
int32_t x631 = 0;
for(int x632=0; x632 < 100; x632++) {
for(int x633=0; x633 < 300; x633++) {
float x634 = x528[x633];
int32_t x635 = x631;
int32_t x636 = x635 + x633;
float x637 = x40[x636];
float x638 = x551[x632];
float x639 = x637 * x638;
float x640 = x634 + x639;
x528[x633] = x640;

}
x631 += 300;

}
} else {
// dot List(100, 100) - WrappedArray(100)
int32_t x649 = 0;
float* x650 = (float*)myMalloc(100 * sizeof(float));
for(int x651=0; x651 < 100; x651++) {
float x652 = 0.0f;
for(int x653=0; x653 < 100; x653++) {
int32_t x654 = x649;
float x655 = x55[x654];
float x656 = x229[x653];
float x657 = x655 * x656;
x652 += x657;
x649 += 1;

}
float x662 = x652;
x650[x651] = x662;

}
float* x666 = (float*)myMalloc(100 * sizeof(float));
for(int x667=0; x667 < 100; x667++) {
x666[x667] = 0.0f;

}
// dot List(100, 100) - WrappedArray(100)
int32_t x672 = 0;
float* x673 = (float*)myMalloc(100 * sizeof(float));
for(int x674=0; x674 < 100; x674++) {
float x675 = 0.0f;
for(int x676=0; x676 < 100; x676++) {
int32_t x677 = x672;
float x678 = x64[x677];
float x679 = x236[x676];
float x680 = x678 * x679;
x675 += x680;
x672 += 1;

}
float x685 = x675;
x673[x674] = x685;

}
float* x689 = (float*)myMalloc(100 * sizeof(float));
for(int x690=0; x690 < 100; x690++) {
x689[x690] = 0.0f;

}
float* x694 = (float*)myMalloc(100 * sizeof(float));
for(int x695=0; x695 < 100; x695++) {
float x696 = x650[x695];
float x697 = x673[x695];
float x698 = x696 + x697;
x694[x695] = x698;

}
float* x702 = (float*)myMalloc(100 * sizeof(float));
for(int x703=0; x703 < 100; x703++) {
x702[x703] = 0.0f;

}
float* x707 = (float*)myMalloc(100 * sizeof(float));
for(int x708=0; x708 < 100; x708++) {
float x709 = x694[x708];
float x710 = x72[x708];
float x711 = x709 + x710;
x707[x708] = x711;

}
float* x715 = (float*)myMalloc(100 * sizeof(float));
for(int x716=0; x716 < 100; x716++) {
x715[x716] = 0.0f;

}
float* x720 = (float*)myMalloc(100 * sizeof(float));
for(int x721=0; x721 < 100; x721++) {
float x722 = x707[x721];
double x723 = (double)x722;
double x724 = tanh(x723);
float x725 = (float)x724;
x720[x721] = x725;

}
float* x729 = (float*)myMalloc(100 * sizeof(float));
for(int x730=0; x730 < 100; x730++) {
x729[x730] = 0.0f;

}
float** x734 = (float**)myMalloc(2 * sizeof(float*));
x734[0] = x720;
x734[1] = x729;
function<void(float**)> x252 = [&](float** x253) {
float* x254 = x253[0];
float* x255 = x253[1];
// dot List(5, 100) - WrappedArray(100)
int32_t x257 = 0;
float* x258 = (float*)myMalloc(5 * sizeof(float));
for(int x259=0; x259 < 5; x259++) {
float x260 = 0.0f;
for(int x261=0; x261 < 100; x261++) {
int32_t x262 = x257;
float x263 = x77[x262];
float x264 = x254[x261];
float x265 = x263 * x264;
x260 += x265;
x257 += 1;

}
float x270 = x260;
x258[x259] = x270;

}
float* x274 = (float*)myMalloc(5 * sizeof(float));
for(int x275=0; x275 < 5; x275++) {
x274[x275] = 0.0f;

}
float* x279 = (float*)myMalloc(5 * sizeof(float));
for(int x280=0; x280 < 5; x280++) {
float x281 = x258[x280];
float x282 = x86[x280];
float x283 = x281 + x282;
x279[x280] = x283;

}
float* x287 = (float*)myMalloc(5 * sizeof(float));
for(int x288=0; x288 < 5; x288++) {
x287[x288] = 0.0f;

}
float* x292 = (float*)myMalloc(5 * sizeof(float));
for(int x293=0; x293 < 5; x293++) {
float x294 = x279[x293];
double x295 = (double)x294;
double x296 = exp(x295);
float x297 = (float)x296;
x292[x293] = x297;

}
float* x301 = (float*)myMalloc(5 * sizeof(float));
for(int x302=0; x302 < 5; x302++) {
x301[x302] = 0.0f;

}
float x306 = 0.0f;
for(int x307=0; x307 < 5; x307++) {
float x308 = x306;
float x309 = x292[x307];
float x310 = x308 + x309;
x306 = x310;

}
float x314 = x306;
float* x315 = (float*)myMalloc(1 * sizeof(float));
x315[0] = x314;
float* x317 = (float*)myMalloc(1 * sizeof(float));
for(int x318=0; x318 < 1; x318++) {
x317[x318] = 0.0f;

}
float x322 = x315[0];
float* x323 = (float*)myMalloc(5 * sizeof(float));
for(int x324=0; x324 < 5; x324++) {
float x325 = x292[x324];
float x326 = x325 / x322;
x323[x324] = x326;

}
float* x330 = (float*)myMalloc(5 * sizeof(float));
for(int x331=0; x331 < 5; x331++) {
x330[x331] = 0.0f;

}
float x335 = x227[0];
float* x336 = (float*)myMalloc(1 * sizeof(float));
float x337 = x234[0];
float x338 = x337 + x335;
x336[0] = x338;
float* x340 = (float*)myMalloc(1 * sizeof(float));
for(int x341=0; x341 < 1; x341++) {
x340[x341] = 0.0f;

}
// dot WrappedArray(5) - WrappedArray(5)
int32_t x346 = 0;
float* x347 = (float*)myMalloc(1 * sizeof(float));
float x348 = 0.0f;
for(int x349=0; x349 < 5; x349++) {
int32_t x350 = x346;
float x351 = x323[x350];
float x352 = x238[x349];
float x353 = x351 * x352;
x348 += x353;
x346 += 1;

}
float x358 = x348;
x347[0] = x358;
float* x360 = (float*)myMalloc(1 * sizeof(float));
for(int x361=0; x361 < 1; x361++) {
x360[x361] = 0.0f;

}
float* x365 = (float*)myMalloc(1 * sizeof(float));
float x366 = x347[0];
double x367 = (double)x366;
double x368 = log(x367);
float x369 = (float)x368;
x365[0] = x369;
float* x371 = (float*)myMalloc(1 * sizeof(float));
for(int x372=0; x372 < 1; x372++) {
x371[x372] = 0.0f;

}
float* x376 = (float*)myMalloc(1 * sizeof(float));
float x377 = x365[0];
float x378 = x336[0];
float x379 = x378 - x377;
x376[0] = x379;
float* x381 = (float*)myMalloc(1 * sizeof(float));
for(int x382=0; x382 < 1; x382++) {
x381[x382] = 0.0f;

}
float** x386 = (float**)myMalloc(4 * sizeof(float*));
x386[0] = x376;
x386[1] = x381;
x386[2] = x254;
x386[3] = x255;
x217(x386);
// += tensor of dim 0
float x393 = x381[0];
float x394 = x340[0];
float x395 = x394 + x393;
x340[0] = x395;
float x397 = x381[0];
float x398 = x371[0];
float x399 = x398 - x397;
x371[0] = x399;
float x401 = x360[0];
float x402 = x371[0];
float x403 = x347[0];
float x404 = x402 / x403;
float x405 = x401 + x404;
x360[0] = x405;
float x407 = x360[0];
// Generate code for addMul
for(int x409=0; x409 < 5; x409++) {
float x410 = x330[x409];
float x411 = x238[x409];
float x412 = x407 * x411;
float x413 = x410 + x412;
x330[x409] = x413;

}
float x417 = x360[0];
// Generate code for addMul
for(int x419=0; x419 < 5; x419++) {
float x420 = x245[x419];
float x421 = x323[x419];
float x422 = x417 * x421;
float x423 = x420 + x422;
x245[x419] = x423;

}
// backpropagate +
// += tensor of dim 0
float x429 = x340[0];
float x430 = x228[0];
float x431 = x430 + x429;
x228[0] = x431;
// += tensor of dim 0
float x434 = x340[0];
float x435 = x235[0];
float x436 = x435 + x434;
x235[0] = x436;
for(int x438=0; x438 < 5; x438++) {
float x439 = x301[x438];
float x440 = x330[x438];
float x441 = x315[0];
float x442 = x440 / x441;
float x443 = x439 + x442;
x301[x438] = x443;

}
for(int x447=0; x447 < 5; x447++) {
float x448 = x317[0];
float x449 = x292[x447];
float x450 = x330[x447];
float x452 = x315[0];
float x451 = x449 * x450;
float x453 = x452 * x452;
float x454 = x451 / x453;
float x455 = x448 - x454;
x317[0] = x455;

}
// += tensor of dim 0
float x460 = x317[0];
for(int x461=0; x461 < 5; x461++) {
float x462 = x301[x461];
float x463 = x462 + x460;
x301[x461] = x463;

}
// backpropage exp
for(int x468=0; x468 < 5; x468++) {
float x469 = x287[x468];
float x470 = x292[x468];
float x471 = x301[x468];
float x472 = x470 * x471;
float x473 = x469 + x472;
x287[x468] = x473;

}
// backpropagate +
for(int x478=0; x478 < 5; x478++) {
float x479 = x274[x478];
float x480 = x287[x478];
float x481 = x479 + x480;
x274[x478] = x481;

}
for(int x485=0; x485 < 5; x485++) {
float x486 = x122[x485];
float x487 = x287[x485];
float x488 = x486 + x487;
x122[x485] = x488;

}
// add_cartesian
int32_t x493 = 0;
for(int x494=0; x494 < 5; x494++) {
for(int x495=0; x495 < 100; x495++) {
int32_t x496 = x493;
int32_t x497 = x496 + x495;
float x498 = x117[x497];
float x499 = x254[x495];
float x500 = x274[x494];
float x501 = x499 * x500;
float x502 = x498 + x501;
x117[x497] = x502;

}
x493 += 100;

}
int32_t x509 = 0;
for(int x510=0; x510 < 5; x510++) {
for(int x511=0; x511 < 100; x511++) {
float x512 = x255[x511];
int32_t x513 = x509;
int32_t x514 = x513 + x511;
float x515 = x77[x514];
float x516 = x274[x510];
float x517 = x515 * x516;
float x518 = x512 + x517;
x255[x511] = x518;

}
x509 += 100;

}
};
x252(x734);
// backpropagate tanh
for(int x739=0; x739 < 100; x739++) {
float x740 = x715[x739];
float x741 = x720[x739];
float x744 = x729[x739];
float x742 = x741 * x741;
float x743 = 1.0f - x742;
float x745 = x743 * x744;
float x746 = x740 + x745;
x715[x739] = x746;

}
// backpropagate +
for(int x751=0; x751 < 100; x751++) {
float x752 = x702[x751];
float x753 = x715[x751];
float x754 = x752 + x753;
x702[x751] = x754;

}
for(int x758=0; x758 < 100; x758++) {
float x759 = x112[x758];
float x760 = x715[x758];
float x761 = x759 + x760;
x112[x758] = x761;

}
// backpropagate +
for(int x766=0; x766 < 100; x766++) {
float x767 = x666[x766];
float x768 = x702[x766];
float x769 = x767 + x768;
x666[x766] = x769;

}
for(int x773=0; x773 < 100; x773++) {
float x774 = x689[x773];
float x775 = x702[x773];
float x776 = x774 + x775;
x689[x773] = x776;

}
// add_cartesian
int32_t x781 = 0;
for(int x782=0; x782 < 100; x782++) {
for(int x783=0; x783 < 100; x783++) {
int32_t x784 = x781;
int32_t x785 = x784 + x783;
float x786 = x107[x785];
float x787 = x236[x783];
float x788 = x689[x782];
float x789 = x787 * x788;
float x790 = x786 + x789;
x107[x785] = x790;

}
x781 += 100;

}
int32_t x797 = 0;
for(int x798=0; x798 < 100; x798++) {
for(int x799=0; x799 < 100; x799++) {
float x800 = x237[x799];
int32_t x801 = x797;
int32_t x802 = x801 + x799;
float x803 = x64[x802];
float x804 = x689[x798];
float x805 = x803 * x804;
float x806 = x800 + x805;
x237[x799] = x806;

}
x797 += 100;

}
// add_cartesian
int32_t x814 = 0;
for(int x815=0; x815 < 100; x815++) {
for(int x816=0; x816 < 100; x816++) {
int32_t x817 = x814;
int32_t x818 = x817 + x816;
float x819 = x102[x818];
float x820 = x229[x816];
float x821 = x666[x815];
float x822 = x820 * x821;
float x823 = x819 + x822;
x102[x818] = x823;

}
x814 += 100;

}
int32_t x830 = 0;
for(int x831=0; x831 < 100; x831++) {
for(int x832=0; x832 < 100; x832++) {
float x833 = x230[x832];
int32_t x834 = x830;
int32_t x835 = x834 + x832;
float x836 = x55[x835];
float x837 = x666[x831];
float x838 = x836 * x837;
float x839 = x833 + x838;
x230[x832] = x839;

}
x830 += 100;

}
}
};
x212(x231,x232,x849);
};
x212(x224,x225,x857);
} else {
float** x865 = (float**)myMalloc(4 * sizeof(float*));
x865[0] = x219;
x865[1] = x220;
x865[2] = x221;
x865[3] = x222;
function<void(float**)> x217 = x214;
x217(x865);
}
};
float* x176 = (float*)myMalloc(1 * sizeof(float));
for(int x178=0; x178 < 1; x178++) {
x176[x178] = 0.0f;

}
float* x182 = (float*)myMalloc(1 * sizeof(float));
for(int x183=0; x183 < 1; x183++) {
x182[x183] = 0.0f;

}
float* x187 = (float*)myMalloc(1 * sizeof(float));
for(int x188=0; x188 < 1; x188++) {
x187[x188] = 0.0f;

}
float* x192 = (float*)myMalloc(1 * sizeof(float));
for(int x193=0; x193 < 1; x193++) {
x192[x193] = 0.0f;

}
float* x197 = (float*)myMalloc(1 * sizeof(float));
for(int x198=0; x198 < 1; x198++) {
x197[x198] = 0.0f;

}
float* x202 = (float*)myMalloc(100 * sizeof(float));
for(int x203=0; x203 < 100; x203++) {
x202[x203] = 0.0f;

}
float* x207 = (float*)myMalloc(100 * sizeof(float));
for(int x208=0; x208 < 100; x208++) {
x207[x208] = 0.0f;

}
float** x885 = (float**)myMalloc(4 * sizeof(float*));
x885[0] = x192;
x885[1] = x197;
x885[2] = x202;
x885[3] = x207;
function<void(float**)> x874 = [&](float** x875) {
float* x876 = x875[0];
float* x877 = x875[1];
float* x878 = x875[2];
float* x879 = x875[3];
float x880 = x877[0];
x877[0] = 1.0f;
float x882 = x876[0];
x187[0] = x882;
};
x212(0,x874,x885);
float x892 = x187[0];
double x893 = x165;
double x894 = (double)x166;
double x895 = x893 * x894;
int32_t x896 = x166 + 1;
double x897 = (double)x896;
double x898 = x895 / x897;
float x899 = (float)x896;
float x900 = x892 / x899;
double x901 = (double)x900;
double x902 = x898 + x901;
x165 = x902;
for(int x904=0; x904 < 30000; x904++) {
float x905 = x92[x904];
bool x906 = x905 > 1.0f;
if (x906) {
x92[x904] = 1.0f;
} else {
}
float x910 = x92[x904];
bool x911 = x910 < -1.0f;
if (x911) {
x92[x904] = -1.0f;
} else {
}

}
float* x917 = (float*)myMalloc(30000 * sizeof(float));
for(int x918=0; x918 < 30000; x918++) {
float x919 = x92[x918];
float x920 = x92[x918];
float x921 = x919 * x920;
x917[x918] = x921;

}
for(int x925=0; x925 < 30000; x925++) {
float x926 = x127[x925];
float x927 = x917[x925];
float x928 = x926 + x927;
x127[x925] = x928;

}
float* x932 = (float*)myMalloc(30000 * sizeof(float));
for(int x933=0; x933 < 30000; x933++) {
float x934 = x92[x933];
float x935 = x934 * 0.05f;
x932[x933] = x935;

}
float* x939 = (float*)myMalloc(30000 * sizeof(float));
for(int x940=0; x940 < 30000; x940++) {
float x941 = x127[x940];
float x942 = x941 + 1.0E-8f;
x939[x940] = x942;

}
float* x946 = (float*)myMalloc(30000 * sizeof(float));
for(int x947=0; x947 < 30000; x947++) {
float x948 = x939[x947];
double x949 = (double)x948;
double x950 = sqrt(x949);
float x951 = (float)x950;
x946[x947] = x951;

}
float* x955 = (float*)myMalloc(30000 * sizeof(float));
for(int x956=0; x956 < 30000; x956++) {
float x957 = x932[x956];
float x958 = x946[x956];
float x959 = x957 / x958;
x955[x956] = x959;

}
for(int x963=0; x963 < 30000; x963++) {
float x964 = x40[x963];
float x965 = x955[x963];
float x966 = x964 - x965;
x40[x963] = x966;

}
for(int x970=0; x970 < 30000; x970++) {
float x971 = x92[x970];
x92[x970] = 0.0f;

}
for(int x975=0; x975 < 100; x975++) {
float x976 = x97[x975];
bool x977 = x976 > 1.0f;
if (x977) {
x97[x975] = 1.0f;
} else {
}
float x981 = x97[x975];
bool x982 = x981 < -1.0f;
if (x982) {
x97[x975] = -1.0f;
} else {
}

}
float* x988 = (float*)myMalloc(100 * sizeof(float));
for(int x989=0; x989 < 100; x989++) {
float x990 = x97[x989];
float x991 = x97[x989];
float x992 = x990 * x991;
x988[x989] = x992;

}
for(int x996=0; x996 < 100; x996++) {
float x997 = x132[x996];
float x998 = x988[x996];
float x999 = x997 + x998;
x132[x996] = x999;

}
float* x1003 = (float*)myMalloc(100 * sizeof(float));
for(int x1004=0; x1004 < 100; x1004++) {
float x1005 = x97[x1004];
float x1006 = x1005 * 0.05f;
x1003[x1004] = x1006;

}
float* x1010 = (float*)myMalloc(100 * sizeof(float));
for(int x1011=0; x1011 < 100; x1011++) {
float x1012 = x132[x1011];
float x1013 = x1012 + 1.0E-8f;
x1010[x1011] = x1013;

}
float* x1017 = (float*)myMalloc(100 * sizeof(float));
for(int x1018=0; x1018 < 100; x1018++) {
float x1019 = x1010[x1018];
double x1020 = (double)x1019;
double x1021 = sqrt(x1020);
float x1022 = (float)x1021;
x1017[x1018] = x1022;

}
float* x1026 = (float*)myMalloc(100 * sizeof(float));
for(int x1027=0; x1027 < 100; x1027++) {
float x1028 = x1003[x1027];
float x1029 = x1017[x1027];
float x1030 = x1028 / x1029;
x1026[x1027] = x1030;

}
for(int x1034=0; x1034 < 100; x1034++) {
float x1035 = x49[x1034];
float x1036 = x1026[x1034];
float x1037 = x1035 - x1036;
x49[x1034] = x1037;

}
for(int x1041=0; x1041 < 100; x1041++) {
float x1042 = x97[x1041];
x97[x1041] = 0.0f;

}
for(int x1046=0; x1046 < 10000; x1046++) {
float x1047 = x102[x1046];
bool x1048 = x1047 > 1.0f;
if (x1048) {
x102[x1046] = 1.0f;
} else {
}
float x1052 = x102[x1046];
bool x1053 = x1052 < -1.0f;
if (x1053) {
x102[x1046] = -1.0f;
} else {
}

}
float* x1059 = (float*)myMalloc(10000 * sizeof(float));
for(int x1060=0; x1060 < 10000; x1060++) {
float x1061 = x102[x1060];
float x1062 = x102[x1060];
float x1063 = x1061 * x1062;
x1059[x1060] = x1063;

}
for(int x1067=0; x1067 < 10000; x1067++) {
float x1068 = x137[x1067];
float x1069 = x1059[x1067];
float x1070 = x1068 + x1069;
x137[x1067] = x1070;

}
float* x1074 = (float*)myMalloc(10000 * sizeof(float));
for(int x1075=0; x1075 < 10000; x1075++) {
float x1076 = x102[x1075];
float x1077 = x1076 * 0.05f;
x1074[x1075] = x1077;

}
float* x1081 = (float*)myMalloc(10000 * sizeof(float));
for(int x1082=0; x1082 < 10000; x1082++) {
float x1083 = x137[x1082];
float x1084 = x1083 + 1.0E-8f;
x1081[x1082] = x1084;

}
float* x1088 = (float*)myMalloc(10000 * sizeof(float));
for(int x1089=0; x1089 < 10000; x1089++) {
float x1090 = x1081[x1089];
double x1091 = (double)x1090;
double x1092 = sqrt(x1091);
float x1093 = (float)x1092;
x1088[x1089] = x1093;

}
float* x1097 = (float*)myMalloc(10000 * sizeof(float));
for(int x1098=0; x1098 < 10000; x1098++) {
float x1099 = x1074[x1098];
float x1100 = x1088[x1098];
float x1101 = x1099 / x1100;
x1097[x1098] = x1101;

}
for(int x1105=0; x1105 < 10000; x1105++) {
float x1106 = x55[x1105];
float x1107 = x1097[x1105];
float x1108 = x1106 - x1107;
x55[x1105] = x1108;

}
for(int x1112=0; x1112 < 10000; x1112++) {
float x1113 = x102[x1112];
x102[x1112] = 0.0f;

}
for(int x1117=0; x1117 < 10000; x1117++) {
float x1118 = x107[x1117];
bool x1119 = x1118 > 1.0f;
if (x1119) {
x107[x1117] = 1.0f;
} else {
}
float x1123 = x107[x1117];
bool x1124 = x1123 < -1.0f;
if (x1124) {
x107[x1117] = -1.0f;
} else {
}

}
float* x1130 = (float*)myMalloc(10000 * sizeof(float));
for(int x1131=0; x1131 < 10000; x1131++) {
float x1132 = x107[x1131];
float x1133 = x107[x1131];
float x1134 = x1132 * x1133;
x1130[x1131] = x1134;

}
for(int x1138=0; x1138 < 10000; x1138++) {
float x1139 = x142[x1138];
float x1140 = x1130[x1138];
float x1141 = x1139 + x1140;
x142[x1138] = x1141;

}
float* x1145 = (float*)myMalloc(10000 * sizeof(float));
for(int x1146=0; x1146 < 10000; x1146++) {
float x1147 = x107[x1146];
float x1148 = x1147 * 0.05f;
x1145[x1146] = x1148;

}
float* x1152 = (float*)myMalloc(10000 * sizeof(float));
for(int x1153=0; x1153 < 10000; x1153++) {
float x1154 = x142[x1153];
float x1155 = x1154 + 1.0E-8f;
x1152[x1153] = x1155;

}
float* x1159 = (float*)myMalloc(10000 * sizeof(float));
for(int x1160=0; x1160 < 10000; x1160++) {
float x1161 = x1152[x1160];
double x1162 = (double)x1161;
double x1163 = sqrt(x1162);
float x1164 = (float)x1163;
x1159[x1160] = x1164;

}
float* x1168 = (float*)myMalloc(10000 * sizeof(float));
for(int x1169=0; x1169 < 10000; x1169++) {
float x1170 = x1145[x1169];
float x1171 = x1159[x1169];
float x1172 = x1170 / x1171;
x1168[x1169] = x1172;

}
for(int x1176=0; x1176 < 10000; x1176++) {
float x1177 = x64[x1176];
float x1178 = x1168[x1176];
float x1179 = x1177 - x1178;
x64[x1176] = x1179;

}
for(int x1183=0; x1183 < 10000; x1183++) {
float x1184 = x107[x1183];
x107[x1183] = 0.0f;

}
for(int x1188=0; x1188 < 100; x1188++) {
float x1189 = x112[x1188];
bool x1190 = x1189 > 1.0f;
if (x1190) {
x112[x1188] = 1.0f;
} else {
}
float x1194 = x112[x1188];
bool x1195 = x1194 < -1.0f;
if (x1195) {
x112[x1188] = -1.0f;
} else {
}

}
float* x1201 = (float*)myMalloc(100 * sizeof(float));
for(int x1202=0; x1202 < 100; x1202++) {
float x1203 = x112[x1202];
float x1204 = x112[x1202];
float x1205 = x1203 * x1204;
x1201[x1202] = x1205;

}
for(int x1209=0; x1209 < 100; x1209++) {
float x1210 = x147[x1209];
float x1211 = x1201[x1209];
float x1212 = x1210 + x1211;
x147[x1209] = x1212;

}
float* x1216 = (float*)myMalloc(100 * sizeof(float));
for(int x1217=0; x1217 < 100; x1217++) {
float x1218 = x112[x1217];
float x1219 = x1218 * 0.05f;
x1216[x1217] = x1219;

}
float* x1223 = (float*)myMalloc(100 * sizeof(float));
for(int x1224=0; x1224 < 100; x1224++) {
float x1225 = x147[x1224];
float x1226 = x1225 + 1.0E-8f;
x1223[x1224] = x1226;

}
float* x1230 = (float*)myMalloc(100 * sizeof(float));
for(int x1231=0; x1231 < 100; x1231++) {
float x1232 = x1223[x1231];
double x1233 = (double)x1232;
double x1234 = sqrt(x1233);
float x1235 = (float)x1234;
x1230[x1231] = x1235;

}
float* x1239 = (float*)myMalloc(100 * sizeof(float));
for(int x1240=0; x1240 < 100; x1240++) {
float x1241 = x1216[x1240];
float x1242 = x1230[x1240];
float x1243 = x1241 / x1242;
x1239[x1240] = x1243;

}
for(int x1247=0; x1247 < 100; x1247++) {
float x1248 = x72[x1247];
float x1249 = x1239[x1247];
float x1250 = x1248 - x1249;
x72[x1247] = x1250;

}
for(int x1254=0; x1254 < 100; x1254++) {
float x1255 = x112[x1254];
x112[x1254] = 0.0f;

}
for(int x1259=0; x1259 < 500; x1259++) {
float x1260 = x117[x1259];
bool x1261 = x1260 > 1.0f;
if (x1261) {
x117[x1259] = 1.0f;
} else {
}
float x1265 = x117[x1259];
bool x1266 = x1265 < -1.0f;
if (x1266) {
x117[x1259] = -1.0f;
} else {
}

}
float* x1272 = (float*)myMalloc(500 * sizeof(float));
for(int x1273=0; x1273 < 500; x1273++) {
float x1274 = x117[x1273];
float x1275 = x117[x1273];
float x1276 = x1274 * x1275;
x1272[x1273] = x1276;

}
for(int x1280=0; x1280 < 500; x1280++) {
float x1281 = x152[x1280];
float x1282 = x1272[x1280];
float x1283 = x1281 + x1282;
x152[x1280] = x1283;

}
float* x1287 = (float*)myMalloc(500 * sizeof(float));
for(int x1288=0; x1288 < 500; x1288++) {
float x1289 = x117[x1288];
float x1290 = x1289 * 0.05f;
x1287[x1288] = x1290;

}
float* x1294 = (float*)myMalloc(500 * sizeof(float));
for(int x1295=0; x1295 < 500; x1295++) {
float x1296 = x152[x1295];
float x1297 = x1296 + 1.0E-8f;
x1294[x1295] = x1297;

}
float* x1301 = (float*)myMalloc(500 * sizeof(float));
for(int x1302=0; x1302 < 500; x1302++) {
float x1303 = x1294[x1302];
double x1304 = (double)x1303;
double x1305 = sqrt(x1304);
float x1306 = (float)x1305;
x1301[x1302] = x1306;

}
float* x1310 = (float*)myMalloc(500 * sizeof(float));
for(int x1311=0; x1311 < 500; x1311++) {
float x1312 = x1287[x1311];
float x1313 = x1301[x1311];
float x1314 = x1312 / x1313;
x1310[x1311] = x1314;

}
for(int x1318=0; x1318 < 500; x1318++) {
float x1319 = x77[x1318];
float x1320 = x1310[x1318];
float x1321 = x1319 - x1320;
x77[x1318] = x1321;

}
for(int x1325=0; x1325 < 500; x1325++) {
float x1326 = x117[x1325];
x117[x1325] = 0.0f;

}
for(int x1330=0; x1330 < 5; x1330++) {
float x1331 = x122[x1330];
bool x1332 = x1331 > 1.0f;
if (x1332) {
x122[x1330] = 1.0f;
} else {
}
float x1336 = x122[x1330];
bool x1337 = x1336 < -1.0f;
if (x1337) {
x122[x1330] = -1.0f;
} else {
}

}
float* x1343 = (float*)myMalloc(5 * sizeof(float));
for(int x1344=0; x1344 < 5; x1344++) {
float x1345 = x122[x1344];
float x1346 = x122[x1344];
float x1347 = x1345 * x1346;
x1343[x1344] = x1347;

}
for(int x1351=0; x1351 < 5; x1351++) {
float x1352 = x157[x1351];
float x1353 = x1343[x1351];
float x1354 = x1352 + x1353;
x157[x1351] = x1354;

}
float* x1358 = (float*)myMalloc(5 * sizeof(float));
for(int x1359=0; x1359 < 5; x1359++) {
float x1360 = x122[x1359];
float x1361 = x1360 * 0.05f;
x1358[x1359] = x1361;

}
float* x1365 = (float*)myMalloc(5 * sizeof(float));
for(int x1366=0; x1366 < 5; x1366++) {
float x1367 = x157[x1366];
float x1368 = x1367 + 1.0E-8f;
x1365[x1366] = x1368;

}
float* x1372 = (float*)myMalloc(5 * sizeof(float));
for(int x1373=0; x1373 < 5; x1373++) {
float x1374 = x1365[x1373];
double x1375 = (double)x1374;
double x1376 = sqrt(x1375);
float x1377 = (float)x1376;
x1372[x1373] = x1377;

}
float* x1381 = (float*)myMalloc(5 * sizeof(float));
for(int x1382=0; x1382 < 5; x1382++) {
float x1383 = x1358[x1382];
float x1384 = x1372[x1382];
float x1385 = x1383 / x1384;
x1381[x1382] = x1385;

}
for(int x1389=0; x1389 < 5; x1389++) {
float x1390 = x86[x1389];
float x1391 = x1381[x1389];
float x1392 = x1390 - x1391;
x86[x1389] = x1392;

}
for(int x1396=0; x1396 < 5; x1396++) {
float x1397 = x122[x1396];
x122[x1396] = 0.0f;

}
mallocAddr = (void*)x162;

}
double x1404 = x165;
printf("epoc %d, ave_loss %f\n",x164,x1404);

}
}
/*****************************************
  End of C Generated Code                  
*******************************************/

