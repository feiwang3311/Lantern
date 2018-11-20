#include <assert.h>
#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <functional>
#include <math.h>
#include <memory>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cblas.h>
#include <algorithm>
#include <numeric>

using namespace std;
#ifndef MAP_FILE
#define MAP_FILE MAP_SHARED
#endif

long fsize(int fd) {
  struct stat stat;
  int res = fstat(fd, &stat);
  return stat.st_size;
}

int printll(char *s) {
  while (*s != '\n' && *s != ',' && *s != '\t') {
    putchar(*s++);
  }
  return 0;
}

long hash(char *str0, int len) {
  unsigned char *str = (unsigned char *)str0;
  unsigned long hash = 5381;
  int c;

  while ((c = *str++) && len--)
    hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

  return hash;
}

long HEAP_SIZE_CPU = 1073741826; // 1048576; // 536870912; // 268435456; // 2097152; 1610612739; // 4294967304; //
void *mallocBase = calloc(HEAP_SIZE_CPU, 1);
void *mallocAddr = mallocBase;
void *waterMark = mallocBase;
void *myMalloc(size_t bytes) {
  void *res = mallocAddr;
  mallocAddr = (void *)((char *)mallocAddr + bytes);
  if ((long)mallocAddr >= (long)mallocBase + HEAP_SIZE_CPU)
    fprintf(stderr, "CPU memory breached limit of HEAP_SIZE_CPU\n");
  return res;
}

long HEAP_SIZE = 8589934608; //  4294967304; // this is for GPU

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
  long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
  result->tv_sec = diff / 1000000;
  result->tv_usec = diff % 1000000;
  return (diff < 0);
}



void Snippet(char *);

std::random_device rd{};
std::mt19937 gen{rd()};
std::normal_distribution<> d{0, 0.01};

int main(int argc, char *argv[]) {
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
int64_t x3 = fsize(x2);
int32_t x4 = (int32_t)x3;
int* x7 = (int32_t*)myMalloc(x4 * sizeof(int32_t));;
int64_t x5 = (int64_t)x4;
char* x6 = (char*)mmap(0, x5, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x2, 0);
for(int x9=0; x9 < x4; x9++) {
char x10 = x6[x9];
int32_t x11 = (int32_t ) x10;
int32_t x12 = x11 - 96;
x7[x9] = x12;

}
float* x16 = (float*)myMalloc(1300 * sizeof(float));;
for(int x18=0; x18 < 1300; x18++) {
float x19 = (float)rand()/RAND_MAX;
float x20 = x19 - 0.5f;
float x21 = x20 * 0.19611613f;
x16[x18] = x21;

}
float* x26 = (float*)myMalloc(1300 * sizeof(float));;
float* x27 = (float*)myMalloc(2500 * sizeof(float));;
for(int x29=0; x29 < 2500; x29++) {
float x30 = (float)rand()/RAND_MAX;
float x31 = x30 - 0.5f;
float x32 = x31 * 0.14142136f;
x27[x29] = x32;

}
float* x36 = (float*)myMalloc(2500 * sizeof(float));;
float* x37 = (float*)myMalloc(50 * sizeof(float));;
float* x38 = (float*)myMalloc(50 * sizeof(float));;
float* x39 = (float*)myMalloc(1300 * sizeof(float));;
for(int x40=0; x40 < 1300; x40++) {
float x41 = (float)rand()/RAND_MAX;
float x42 = x41 - 0.5f;
float x43 = x42 * 0.14142136f;
x39[x40] = x43;

}
float* x47 = (float*)myMalloc(1300 * sizeof(float));;
float* x48 = (float*)myMalloc(26 * sizeof(float));;
float* x49 = (float*)myMalloc(26 * sizeof(float));;
float* x50 = (float*)myMalloc(26 * sizeof(float));;
float* x51 = (float*)myMalloc(1300 * sizeof(float));;
float* x52 = (float*)myMalloc(2500 * sizeof(float));;
float* x53 = (float*)myMalloc(50 * sizeof(float));;
float* x54 = (float*)myMalloc(1300 * sizeof(float));;
double* x55 = (double*)myMalloc(51 * sizeof(double));;
double x56 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x57 = (long)mallocAddr;
int32_t x58 = 0;
x58 -= 400;
bool x345 = true || true;
bool x346 = x345 || true;
bool x665 = true || false;
for(int x61=0; x61 < 5001; x61++) {
float* x86 = (float*)myMalloc(1 * sizeof(float));;
float* x87 = (float*)myMalloc(10400 * sizeof(float));;
float* x104 = (float*)myMalloc(10400 * sizeof(float));;
int* x71 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
function<void(int32_t,float**)> x369 = [&](int32_t x370,float** x371) {
float** x373 = x371;
float* x374 = x373[0];
float* x375 = x373[1];
float* x376 = x373[2];
float* x377 = x373[3];
int32_t x372 = x370;
bool x378 = x372 < 20;
if (x378) {
int32_t x379 = x372 * 520;
float* x380 = x87+x379;
float* x381 = x104+x379;
float* x382 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x380,26,x16,50,0,x382,50);
float* x384 = (float*)myMalloc(1000 * sizeof(float));;
float* x385 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x376,50,x27,50,0,x385,50);
float* x387 = (float*)myMalloc(1000 * sizeof(float));;
float* x388 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x389 = 0;
int32_t x390 = 0;
int32_t x391 = 0;
for(int x392=0; x392 < 20; x392++) {
int32_t x393 = x390;
int32_t x394 = x391;
int32_t x395 = x389;
int32_t x396 = x395;
int32_t x397 = x393;
int32_t x398 = x394;
for(int x399=0; x399 < 50; x399++) {
int32_t x400 = x396;
int32_t x401 = x397;
float x402 = x382[x401];
int32_t x403 = x398;
float x404 = x385[x403];
float x405 = x402 + x404;
x388[x400] = x405;
x396 += 1;
x397 += 1;
x398 += 1;

}
x389 += 50;
x390 += 50;
x391 += 50;

}
float* x417 = (float*)myMalloc(1000 * sizeof(float));;
float* x418 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x419 = 0;
int32_t x420 = 0;
int32_t x421 = 0;
for(int x422=0; x422 < 20; x422++) {
int32_t x423 = x420;
int32_t x424 = x421;
int32_t x425 = x419;
int32_t x426 = x425;
int32_t x427 = x423;
int32_t x428 = x424;
for(int x429=0; x429 < 50; x429++) {
int32_t x430 = x426;
int32_t x431 = x427;
float x432 = x388[x431];
int32_t x433 = x428;
float x434 = x37[x433];
float x435 = x432 + x434;
x418[x430] = x435;
x426 += 1;
x427 += 1;
x428 += 1;

}
x419 += 50;
x420 += 50;

}
float* x446 = (float*)myMalloc(1000 * sizeof(float));;
float* x447 = (float*)myMalloc(1000 * sizeof(float));;
for(int x448=0; x448 < 1000; x448++) {
float x449 = x418[x448];
double x450 = (double)x449;
double x451 = tanh(x450);
float x452 = (float)x451;
x447[x448] = x452;

}
float* x456 = (float*)myMalloc(1000 * sizeof(float));;
float* x457 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x447,50,x39,26,0,x457,26);
float* x459 = (float*)myMalloc(520 * sizeof(float));;
int32_t x460 = 0;
int32_t x461 = 0;
int32_t x462 = 0;
for(int x463=0; x463 < 20; x463++) {
int32_t x464 = x461;
int32_t x465 = x462;
int32_t x466 = x460;
int32_t x467 = x466;
int32_t x468 = x464;
int32_t x469 = x465;
for(int x470=0; x470 < 26; x470++) {
int32_t x471 = x468;
float x472 = x457[x471];
int32_t x473 = x469;
float x474 = x48[x473];
float x475 = x472 + x474;
x457[x471] = x475;
x467 += 1;
x468 += 1;
x469 += 1;

}
x460 += 26;
x461 += 26;

}
int* x486 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x487=0; x487 < 20; x487++) {
int32_t x488 = x487 * 20;
int32_t x489 = x372 + x488;
int32_t x490 = x71[x489];
x486[x487] = x490;

}
float* x494 = (float*)myMalloc(20 * sizeof(float));;
int32_t x495 = 0;
for(int x496=0; x496 < 20; x496++) {
float x497 = -3.4028235E38f;
for(int x498=0; x498 < 26; x498++) {
int32_t x499 = x495;
float x500 = x457[x499];
float x501 = x497;
bool x502 = x500 > x501;
if (x502) {
float x503 = x457[x499];
x497 = x503;
} else {
}
x495 += 1;

}
float x510 = x497;
x494[x496] = x510;

}
float* x514 = (float*)myMalloc(520 * sizeof(float));;
int32_t x515 = 0;
for(int x516=0; x516 < 20; x516++) {
for(int x517=0; x517 < 26; x517++) {
int32_t x518 = x515;
float x519 = x457[x518];
float x520 = x494[x516];
float x521 = x519 - x520;
double x522 = (double)x521;
double x523 = exp(x522);
float x524 = (float)x523;
x514[x518] = x524;
x515 += 1;

}

}
float* x531 = (float*)myMalloc(20 * sizeof(float));;
for(int x532=0; x532 < 20; x532++) {
int32_t x533 = x532;
int32_t x534 = x532 * 26;
int32_t x535 = x534;
for(int x536=0; x536 < 26; x536++) {
for(int x537=0; x537 < 1; x537++) {
int32_t x538 = x533;
int32_t x539 = x538 + x537;
float x540 = x531[x539];
int32_t x541 = x535;
int32_t x542 = x541 + x537;
float x543 = x514[x542];
float x544 = x540 + x543;
x531[x539] = x544;

}
x535 += 1;

}

}
x515 = 0;
for(int x554=0; x554 < 20; x554++) {
float x555 = x494[x554];
float x556 = x531[x554];
double x557 = (double)x556;
double x558 = log(x557);
float x559 = (float)x558;
float x560 = x555 + x559;
for(int x561=0; x561 < 26; x561++) {
int32_t x562 = x515;
float x563 = x457[x562];
float x564 = x563 - x560;
x514[x562] = x564;
x515 += 1;

}

}
float* x571 = (float*)myMalloc(520 * sizeof(float));;
// nllLoss forward in CPU
float* x573 = (float*)myMalloc(20 * sizeof(float));;
int32_t x574 = 0;
for(int x575=0; x575 < 20; x575++) {
int32_t x576 = x574;
int32_t x577 = x486[x575];
int32_t x578 = x576 + x577;
float x579 = x514[x578];
float x580 = -1.0f * x579;
x573[x575] = x580;
x574 += 26;

}
float* x585 = (float*)myMalloc(20 * sizeof(float));;
float x586 = 0.0f;
for(int x587=0; x587 < 20; x587++) {
float x588 = x586;
float x589 = x573[x587];
float x590 = x588 + x589;
x586 = x590;

}
float x594 = x586;
float* x595 = (float*)myMalloc(1 * sizeof(float));;
for(int x596=0; x596 < 1; x596++) {
x595[x596] = x594;

}
float* x600 = (float*)myMalloc(1 * sizeof(float));;
if (x346) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x605 = (float*)myMalloc(1 * sizeof(float));;
int32_t x606 = 0;
int32_t x607 = 0;
int32_t x608 = 0;
for(int x609=0; x609 < 1; x609++) {
int32_t x610 = x606;
int32_t x611 = x607;
float x612 = x374[x611];
int32_t x613 = x608;
float x614 = x595[x613];
float x615 = x612 + x614;
x605[x610] = x615;
x606 += 1;

}
float* x620 = (float*)myMalloc(1 * sizeof(float));;
float** x622 = (float**)myMalloc(4 * sizeof(float*));;
x622[0] = x605;
x622[1] = x620;
x622[2] = x447;
x622[3] = x456;
int32_t x634 = 0;
int32_t x635 = 0;
int32_t x636 = 0;
int32_t x651 = 0;
int32_t x652 = 0;
int32_t x653 = 0;
int32_t x671 = 0;
int32_t x672 = 0;
int32_t x673 = 0;
int32_t x687 = 0;
float* x700 = (float*)myMalloc(20 * sizeof(float));;
int32_t x722 = 0;
int32_t x742 = 0;
int32_t x743 = 0;
int32_t x744 = 0;
int32_t x783 = 0;
int32_t x784 = 0;
int32_t x785 = 0;
int32_t x810 = 0;
int32_t x811 = 0;
int32_t x812 = 0;
int32_t x837 = 0;
int32_t x838 = 0;
int32_t x839 = 0;
int32_t x864 = 0;
int32_t x865 = 0;
int32_t x866 = 0;
int32_t x621 = x372 + 1;
x369(x621,x622);
// back prop for + op
if (x346) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x637=0; x637 < 1; x637++) {
int32_t x638 = x635;
float x639 = x375[x638];
int32_t x640 = x636;
float x641 = x620[x640];
float x642 = x639 + x641;
x375[x638] = x642;
x634 += 1;

}
if (x346) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x654=0; x654 < 1; x654++) {
int32_t x655 = x652;
float x656 = x600[x655];
int32_t x657 = x653;
float x658 = x620[x657];
float x659 = x656 + x658;
x600[x655] = x659;
x651 += 1;

}
// 'sum' gradient.
if (x665) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",20,1);
assert(false && "");
}
for(int x674=0; x674 < 20; x674++) {
int32_t x675 = x672;
float x676 = x585[x675];
int32_t x677 = x673;
float x678 = x600[x677];
float x679 = x676 + x678;
x585[x675] = x679;
x671 += 1;
x672 += 1;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
for(int x688=0; x688 < 20; x688++) {
int32_t x689 = x687;
int32_t x690 = x486[x688];
int32_t x691 = x689 + x690;
float x692 = x571[x691];
float x693 = x585[x688];
float x694 = -1.0f * x693;
float x695 = x692 + x694;
x571[x691] = x695;
x687 += 26;

}
for(int x701=0; x701 < 20; x701++) {
int32_t x702 = x701;
int32_t x703 = x701 * 26;
int32_t x704 = x703;
for(int x705=0; x705 < 26; x705++) {
for(int x706=0; x706 < 1; x706++) {
int32_t x707 = x702;
int32_t x708 = x707 + x706;
float x709 = x700[x708];
int32_t x710 = x704;
int32_t x711 = x710 + x706;
float x712 = x571[x711];
float x713 = x709 + x712;
x700[x708] = x713;

}
x704 += 1;

}

}
for(int x723=0; x723 < 20; x723++) {
for(int x724=0; x724 < 26; x724++) {
int32_t x725 = x722;
float x726 = x459[x725];
float x727 = x571[x725];
float x728 = x514[x725];
float x732 = x700[x723];
double x729 = (double)x728;
double x730 = exp(x729);
float x731 = (float)x730;
float x733 = x731 * x732;
float x734 = x727 - x733;
float x735 = x726 + x734;
x459[x725] = x735;
x722 += 1;

}

}
for(int x745=0; x745 < 20; x745++) {
int32_t x746 = x743;
int32_t x747 = x744;
int32_t x748 = x742;
int32_t x749 = x748;
int32_t x750 = x746;
int32_t x751 = x747;
for(int x752=0; x752 < 26; x752++) {
int32_t x753 = x750;
float x754 = x49[x753];
int32_t x755 = x751;
float x756 = x459[x755];
float x757 = x754 + x756;
x49[x753] = x757;
x749 += 1;
x750 += 1;
x751 += 1;

}
x742 += 26;
x744 += 26;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x459,26,x39,26,1,x456,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x447,50,x459,26,1,x47,26);
for(int x771=0; x771 < 1000; x771++) {
float x772 = x446[x771];
float x773 = x447[x771];
float x776 = x456[x771];
float x774 = x773 * x773;
float x775 = 1.0f - x774;
float x777 = x775 * x776;
float x778 = x772 + x777;
x446[x771] = x778;

}
// back prop for + op
for(int x786=0; x786 < 20; x786++) {
int32_t x787 = x784;
int32_t x788 = x785;
int32_t x789 = x783;
int32_t x790 = x789;
int32_t x791 = x787;
int32_t x792 = x788;
for(int x793=0; x793 < 50; x793++) {
int32_t x794 = x791;
float x795 = x417[x794];
int32_t x796 = x792;
float x797 = x446[x796];
float x798 = x795 + x797;
x417[x794] = x798;
x790 += 1;
x791 += 1;
x792 += 1;

}
x783 += 50;
x784 += 50;
x785 += 50;

}
for(int x813=0; x813 < 20; x813++) {
int32_t x814 = x811;
int32_t x815 = x812;
int32_t x816 = x810;
int32_t x817 = x816;
int32_t x818 = x814;
int32_t x819 = x815;
for(int x820=0; x820 < 50; x820++) {
int32_t x821 = x818;
float x822 = x38[x821];
int32_t x823 = x819;
float x824 = x446[x823];
float x825 = x822 + x824;
x38[x821] = x825;
x817 += 1;
x818 += 1;
x819 += 1;

}
x810 += 50;
x812 += 50;

}
// back prop for + op
for(int x840=0; x840 < 20; x840++) {
int32_t x841 = x838;
int32_t x842 = x839;
int32_t x843 = x837;
int32_t x844 = x843;
int32_t x845 = x841;
int32_t x846 = x842;
for(int x847=0; x847 < 50; x847++) {
int32_t x848 = x845;
float x849 = x384[x848];
int32_t x850 = x846;
float x851 = x417[x850];
float x852 = x849 + x851;
x384[x848] = x852;
x844 += 1;
x845 += 1;
x846 += 1;

}
x837 += 50;
x838 += 50;
x839 += 50;

}
for(int x867=0; x867 < 20; x867++) {
int32_t x868 = x865;
int32_t x869 = x866;
int32_t x870 = x864;
int32_t x871 = x870;
int32_t x872 = x868;
int32_t x873 = x869;
for(int x874=0; x874 < 50; x874++) {
int32_t x875 = x872;
float x876 = x387[x875];
int32_t x877 = x873;
float x878 = x417[x877];
float x879 = x876 + x878;
x387[x875] = x879;
x871 += 1;
x872 += 1;
x873 += 1;

}
x864 += 50;
x865 += 50;
x866 += 50;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x387,50,x27,50,1,x377,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x376,50,x387,50,1,x36,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x384,50,x16,50,1,x381,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x380,26,x384,50,1,x26,50);
} else {
float x898 = 0.0f;
for(int x899=0; x899 < 1; x899++) {
float x900 = x898;
float x901 = x374[x899];
float x902 = x900 + x901;
x898 = x902;

}
float x906 = x898;
float* x907 = (float*)myMalloc(1 * sizeof(float));;
for(int x908=0; x908 < 1; x908++) {
x907[x908] = x906;

}
float* x912 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x914=0; x914 < 1; x914++) {
x912[x914] = 1.0f;

}
// backend is lantern.TensorDsl$BackendCPU@855a742
for(int x919=0; x919 < 1; x919++) {
float x920 = x907[x919];
x86[x919] = x920;

}
// 'sum' gradient.
if (x346) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
int32_t x929 = 0;
int32_t x930 = 0;
int32_t x931 = 0;
for(int x932=0; x932 < 1; x932++) {
int32_t x933 = x930;
float x934 = x375[x933];
int32_t x935 = x931;
float x936 = x912[x935];
float x937 = x934 + x936;
x375[x933] = x937;
x929 += 1;

}
}
};
x58 += 400;
int32_t x63 = x58;
int32_t x64 = x63 + 400;
int32_t x65 = x64 + 1;
bool x66 = x65 >= x4;
if (x66) {
x58 = 0;
} else {
}
int* x70 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
for(int x73=0; x73 < 400; x73++) {
int32_t x74 = x58;
int32_t x75 = x74 + x73;
int32_t x76 = x7[x75];
x70[x73] = x76;
int32_t x78 = x75 + 1;
int32_t x79 = x7[x78];
x71[x73] = x79;

}
float* x83 = (float*)myMalloc(1 * sizeof(float));;
float* x84 = (float*)myMalloc(1 * sizeof(float));;
// allocate memory to save the final loss in CPU Tensor
for(int x89=0; x89 < 20; x89++) {
int32_t x91 = x89 * 26;
int32_t x92 = x91 * 20;
for(int x90=0; x90 < 20; x90++) {
int32_t x95 = x90 * 20;
int32_t x96 = x95 + x89;
int32_t x97 = x70[x96];
int32_t x93 = x90 * 26;
int32_t x94 = x92 + x93;
int32_t x98 = x94 + x97;
x87[x98] = 1.0f;

}

}
float* x105 = (float*)myMalloc(1 * sizeof(float));;
float* x106 = (float*)myMalloc(1 * sizeof(float));;
float* x107 = (float*)myMalloc(1000 * sizeof(float));;
float* x108 = (float*)myMalloc(1000 * sizeof(float));;
float** x1266 = (float**)myMalloc(4 * sizeof(float*));;
x1266[0] = x105;
x1266[1] = x106;
x1266[2] = x107;
x1266[3] = x108;
function<void(int32_t,float**)> x109 = [&](int32_t x110,float** x111) {
float** x113 = x111;
float* x114 = x113[0];
float* x115 = x113[1];
float* x116 = x113[2];
float* x117 = x113[3];
int32_t x112 = x110;
bool x118 = x112 < 20;
if (x118) {
int32_t x119 = x112 * 520;
float* x120 = x87+x119;
float* x121 = x104+x119;
float* x122 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x120,26,x16,50,0,x122,50);
float* x124 = (float*)myMalloc(1000 * sizeof(float));;
float* x125 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x116,50,x27,50,0,x125,50);
float* x127 = (float*)myMalloc(1000 * sizeof(float));;
float* x128 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x129 = 0;
int32_t x130 = 0;
int32_t x131 = 0;
for(int x132=0; x132 < 20; x132++) {
int32_t x133 = x130;
int32_t x134 = x131;
int32_t x135 = x129;
int32_t x136 = x135;
int32_t x137 = x133;
int32_t x138 = x134;
for(int x140=0; x140 < 50; x140++) {
int32_t x141 = x136;
int32_t x142 = x137;
float x143 = x122[x142];
int32_t x144 = x138;
float x145 = x125[x144];
float x146 = x143 + x145;
x128[x141] = x146;
x136 += 1;
x137 += 1;
x138 += 1;

}
x129 += 50;
x130 += 50;
x131 += 50;

}
float* x158 = (float*)myMalloc(1000 * sizeof(float));;
float* x159 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x160 = 0;
int32_t x161 = 0;
int32_t x162 = 0;
for(int x163=0; x163 < 20; x163++) {
int32_t x164 = x161;
int32_t x165 = x162;
int32_t x166 = x160;
int32_t x167 = x166;
int32_t x168 = x164;
int32_t x169 = x165;
for(int x170=0; x170 < 50; x170++) {
int32_t x171 = x167;
int32_t x172 = x168;
float x173 = x128[x172];
int32_t x174 = x169;
float x175 = x37[x174];
float x176 = x173 + x175;
x159[x171] = x176;
x167 += 1;
x168 += 1;
x169 += 1;

}
x160 += 50;
x161 += 50;

}
float* x187 = (float*)myMalloc(1000 * sizeof(float));;
float* x188 = (float*)myMalloc(1000 * sizeof(float));;
for(int x190=0; x190 < 1000; x190++) {
float x191 = x159[x190];
double x192 = (double)x191;
double x193 = tanh(x192);
float x194 = (float)x193;
x188[x190] = x194;

}
float* x198 = (float*)myMalloc(1000 * sizeof(float));;
float* x199 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x188,50,x39,26,0,x199,26);
float* x201 = (float*)myMalloc(520 * sizeof(float));;
int32_t x202 = 0;
int32_t x203 = 0;
int32_t x204 = 0;
for(int x205=0; x205 < 20; x205++) {
int32_t x206 = x203;
int32_t x207 = x204;
int32_t x208 = x202;
int32_t x209 = x208;
int32_t x210 = x206;
int32_t x211 = x207;
for(int x213=0; x213 < 26; x213++) {
int32_t x214 = x210;
float x215 = x199[x214];
int32_t x216 = x211;
float x217 = x48[x216];
float x218 = x215 + x217;
x199[x214] = x218;
x209 += 1;
x210 += 1;
x211 += 1;

}
x202 += 26;
x203 += 26;

}
int* x229 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x230=0; x230 < 20; x230++) {
int32_t x231 = x230 * 20;
int32_t x232 = x112 + x231;
int32_t x233 = x71[x232];
x229[x230] = x233;

}
float* x237 = (float*)myMalloc(20 * sizeof(float));;
int32_t x238 = 0;
for(int x239=0; x239 < 20; x239++) {
float x240 = -3.4028235E38f;
for(int x241=0; x241 < 26; x241++) {
int32_t x242 = x238;
float x243 = x199[x242];
float x244 = x240;
bool x245 = x243 > x244;
if (x245) {
float x246 = x199[x242];
x240 = x246;
} else {
}
x238 += 1;

}
float x253 = x240;
x237[x239] = x253;

}
float* x257 = (float*)myMalloc(520 * sizeof(float));;
int32_t x258 = 0;
for(int x259=0; x259 < 20; x259++) {
for(int x260=0; x260 < 26; x260++) {
int32_t x261 = x258;
float x262 = x199[x261];
float x263 = x237[x259];
float x264 = x262 - x263;
double x265 = (double)x264;
double x266 = exp(x265);
float x267 = (float)x266;
x257[x261] = x267;
x258 += 1;

}

}
float* x274 = (float*)myMalloc(20 * sizeof(float));;
for(int x275=0; x275 < 20; x275++) {
int32_t x276 = x275;
int32_t x277 = x275 * 26;
int32_t x278 = x277;
for(int x279=0; x279 < 26; x279++) {
for(int x281=0; x281 < 1; x281++) {
int32_t x282 = x276;
int32_t x283 = x282 + x281;
float x284 = x274[x283];
int32_t x285 = x278;
int32_t x286 = x285 + x281;
float x287 = x257[x286];
float x288 = x284 + x287;
x274[x283] = x288;

}
x278 += 1;

}

}
x258 = 0;
for(int x298=0; x298 < 20; x298++) {
float x299 = x237[x298];
float x300 = x274[x298];
double x301 = (double)x300;
double x302 = log(x301);
float x303 = (float)x302;
float x304 = x299 + x303;
for(int x305=0; x305 < 26; x305++) {
int32_t x306 = x258;
float x307 = x199[x306];
float x308 = x307 - x304;
x257[x306] = x308;
x258 += 1;

}

}
float* x315 = (float*)myMalloc(520 * sizeof(float));;
// nllLoss forward in CPU
float* x317 = (float*)myMalloc(20 * sizeof(float));;
int32_t x318 = 0;
for(int x319=0; x319 < 20; x319++) {
int32_t x320 = x318;
int32_t x321 = x229[x319];
int32_t x322 = x320 + x321;
float x323 = x257[x322];
float x324 = -1.0f * x323;
x317[x319] = x324;
x318 += 26;

}
float* x329 = (float*)myMalloc(20 * sizeof(float));;
float x330 = 0.0f;
for(int x331=0; x331 < 20; x331++) {
float x332 = x330;
float x333 = x317[x331];
float x334 = x332 + x333;
x330 = x334;

}
float x338 = x330;
float* x339 = (float*)myMalloc(1 * sizeof(float));;
for(int x340=0; x340 < 1; x340++) {
x339[x340] = x338;

}
float* x344 = (float*)myMalloc(1 * sizeof(float));;
if (x346) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x352 = (float*)myMalloc(1 * sizeof(float));;
int32_t x353 = 0;
int32_t x354 = 0;
int32_t x355 = 0;
for(int x356=0; x356 < 1; x356++) {
int32_t x357 = x353;
int32_t x358 = x354;
float x359 = x114[x358];
int32_t x360 = x355;
float x361 = x339[x360];
float x362 = x359 + x361;
x352[x357] = x362;
x353 += 1;

}
float* x367 = (float*)myMalloc(1 * sizeof(float));;
float** x945 = (float**)myMalloc(4 * sizeof(float*));;
x945[0] = x352;
x945[1] = x367;
x945[2] = x188;
x945[3] = x198;
int32_t x368 = x112 + 1;
x369(x368,x945);
// back prop for + op
if (x346) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
int32_t x957 = 0;
int32_t x958 = 0;
int32_t x959 = 0;
for(int x960=0; x960 < 1; x960++) {
int32_t x961 = x958;
float x962 = x115[x961];
int32_t x963 = x959;
float x964 = x367[x963];
float x965 = x962 + x964;
x115[x961] = x965;
x957 += 1;

}
if (x346) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
int32_t x974 = 0;
int32_t x975 = 0;
int32_t x976 = 0;
for(int x977=0; x977 < 1; x977++) {
int32_t x978 = x975;
float x979 = x344[x978];
int32_t x980 = x976;
float x981 = x367[x980];
float x982 = x979 + x981;
x344[x978] = x982;
x974 += 1;

}
// 'sum' gradient.
if (x665) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",20,1);
assert(false && "");
}
int32_t x992 = 0;
int32_t x993 = 0;
int32_t x994 = 0;
for(int x995=0; x995 < 20; x995++) {
int32_t x996 = x993;
float x997 = x329[x996];
int32_t x998 = x994;
float x999 = x344[x998];
float x1000 = x997 + x999;
x329[x996] = x1000;
x992 += 1;
x993 += 1;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x1008 = 0;
for(int x1009=0; x1009 < 20; x1009++) {
int32_t x1010 = x1008;
int32_t x1011 = x229[x1009];
int32_t x1012 = x1010 + x1011;
float x1013 = x315[x1012];
float x1014 = x329[x1009];
float x1015 = -1.0f * x1014;
float x1016 = x1013 + x1015;
x315[x1012] = x1016;
x1008 += 26;

}
float* x1021 = (float*)myMalloc(20 * sizeof(float));;
for(int x1022=0; x1022 < 20; x1022++) {
int32_t x1023 = x1022;
int32_t x1024 = x1022 * 26;
int32_t x1025 = x1024;
for(int x1026=0; x1026 < 26; x1026++) {
for(int x1027=0; x1027 < 1; x1027++) {
int32_t x1028 = x1023;
int32_t x1029 = x1028 + x1027;
float x1030 = x1021[x1029];
int32_t x1031 = x1025;
int32_t x1032 = x1031 + x1027;
float x1033 = x315[x1032];
float x1034 = x1030 + x1033;
x1021[x1029] = x1034;

}
x1025 += 1;

}

}
int32_t x1043 = 0;
for(int x1044=0; x1044 < 20; x1044++) {
for(int x1045=0; x1045 < 26; x1045++) {
int32_t x1046 = x1043;
float x1047 = x201[x1046];
float x1048 = x315[x1046];
float x1049 = x257[x1046];
float x1053 = x1021[x1044];
double x1050 = (double)x1049;
double x1051 = exp(x1050);
float x1052 = (float)x1051;
float x1054 = x1052 * x1053;
float x1055 = x1048 - x1054;
float x1056 = x1047 + x1055;
x201[x1046] = x1056;
x1043 += 1;

}

}
int32_t x1063 = 0;
int32_t x1064 = 0;
int32_t x1065 = 0;
for(int x1066=0; x1066 < 20; x1066++) {
int32_t x1067 = x1064;
int32_t x1068 = x1065;
int32_t x1069 = x1063;
int32_t x1070 = x1069;
int32_t x1071 = x1067;
int32_t x1072 = x1068;
for(int x1073=0; x1073 < 26; x1073++) {
int32_t x1074 = x1071;
float x1075 = x49[x1074];
int32_t x1076 = x1072;
float x1077 = x201[x1076];
float x1078 = x1075 + x1077;
x49[x1074] = x1078;
x1070 += 1;
x1071 += 1;
x1072 += 1;

}
x1063 += 26;
x1065 += 26;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x201,26,x39,26,1,x198,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x188,50,x201,26,1,x47,26);
for(int x1092=0; x1092 < 1000; x1092++) {
float x1093 = x187[x1092];
float x1094 = x188[x1092];
float x1097 = x198[x1092];
float x1095 = x1094 * x1094;
float x1096 = 1.0f - x1095;
float x1098 = x1096 * x1097;
float x1099 = x1093 + x1098;
x187[x1092] = x1099;

}
// back prop for + op
int32_t x1104 = 0;
int32_t x1105 = 0;
int32_t x1106 = 0;
for(int x1107=0; x1107 < 20; x1107++) {
int32_t x1108 = x1105;
int32_t x1109 = x1106;
int32_t x1110 = x1104;
int32_t x1111 = x1110;
int32_t x1112 = x1108;
int32_t x1113 = x1109;
for(int x1114=0; x1114 < 50; x1114++) {
int32_t x1115 = x1112;
float x1116 = x158[x1115];
int32_t x1117 = x1113;
float x1118 = x187[x1117];
float x1119 = x1116 + x1118;
x158[x1115] = x1119;
x1111 += 1;
x1112 += 1;
x1113 += 1;

}
x1104 += 50;
x1105 += 50;
x1106 += 50;

}
int32_t x1131 = 0;
int32_t x1132 = 0;
int32_t x1133 = 0;
for(int x1134=0; x1134 < 20; x1134++) {
int32_t x1135 = x1132;
int32_t x1136 = x1133;
int32_t x1137 = x1131;
int32_t x1138 = x1137;
int32_t x1139 = x1135;
int32_t x1140 = x1136;
for(int x1141=0; x1141 < 50; x1141++) {
int32_t x1142 = x1139;
float x1143 = x38[x1142];
int32_t x1144 = x1140;
float x1145 = x187[x1144];
float x1146 = x1143 + x1145;
x38[x1142] = x1146;
x1138 += 1;
x1139 += 1;
x1140 += 1;

}
x1131 += 50;
x1133 += 50;

}
// back prop for + op
int32_t x1158 = 0;
int32_t x1159 = 0;
int32_t x1160 = 0;
for(int x1161=0; x1161 < 20; x1161++) {
int32_t x1162 = x1159;
int32_t x1163 = x1160;
int32_t x1164 = x1158;
int32_t x1165 = x1164;
int32_t x1166 = x1162;
int32_t x1167 = x1163;
for(int x1168=0; x1168 < 50; x1168++) {
int32_t x1169 = x1166;
float x1170 = x124[x1169];
int32_t x1171 = x1167;
float x1172 = x158[x1171];
float x1173 = x1170 + x1172;
x124[x1169] = x1173;
x1165 += 1;
x1166 += 1;
x1167 += 1;

}
x1158 += 50;
x1159 += 50;
x1160 += 50;

}
int32_t x1185 = 0;
int32_t x1186 = 0;
int32_t x1187 = 0;
for(int x1188=0; x1188 < 20; x1188++) {
int32_t x1189 = x1186;
int32_t x1190 = x1187;
int32_t x1191 = x1185;
int32_t x1192 = x1191;
int32_t x1193 = x1189;
int32_t x1194 = x1190;
for(int x1195=0; x1195 < 50; x1195++) {
int32_t x1196 = x1193;
float x1197 = x127[x1196];
int32_t x1198 = x1194;
float x1199 = x158[x1198];
float x1200 = x1197 + x1199;
x127[x1196] = x1200;
x1192 += 1;
x1193 += 1;
x1194 += 1;

}
x1185 += 50;
x1186 += 50;
x1187 += 50;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x127,50,x27,50,1,x117,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x116,50,x127,50,1,x36,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x124,50,x16,50,1,x121,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x120,26,x124,50,1,x26,50);
} else {
float x1219 = 0.0f;
for(int x1220=0; x1220 < 1; x1220++) {
float x1221 = x1219;
float x1222 = x114[x1220];
float x1223 = x1221 + x1222;
x1219 = x1223;

}
float x1227 = x1219;
float* x1228 = (float*)myMalloc(1 * sizeof(float));;
for(int x1229=0; x1229 < 1; x1229++) {
x1228[x1229] = x1227;

}
float* x1233 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x1235=0; x1235 < 1; x1235++) {
x1233[x1235] = 1.0f;

}
// backend is lantern.TensorDsl$BackendCPU@855a742
for(int x1240=0; x1240 < 1; x1240++) {
float x1241 = x1228[x1240];
x86[x1240] = x1241;

}
// 'sum' gradient.
if (x346) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
int32_t x1250 = 0;
int32_t x1251 = 0;
int32_t x1252 = 0;
for(int x1253=0; x1253 < 1; x1253++) {
int32_t x1254 = x1251;
float x1255 = x115[x1254];
int32_t x1256 = x1252;
float x1257 = x1233[x1256];
float x1258 = x1255 + x1257;
x115[x1254] = x1258;
x1250 += 1;

}
}
};
x109(0,x1266);
float x1273 = x86[0];
int32_t x1274 = x61 % 100;
bool x1275 = x1274 == 0;
if (x1275) {
printf("iter %d, loss %f\n",x61,x1273);
int32_t x1277 = x61 / 100;
double x1278 = (double)x1273;
x55[x1277] = x1278;
} else {
}
for(int x1282=0; x1282 < 26; x1282++) {
float x1283 = x49[x1282];
float x1284 = x1283;
float x1285 = x1284;
bool x1286 = x1285 > 5.0f;
if (x1286) {
x1284 = 5.0f;
} else {
}
float x1290 = x1284;
bool x1291 = x1290 < -5.0f;
if (x1291) {
x1284 = -5.0f;
} else {
}
float x1295 = x50[x1282];
float x1296 = x1284;
float x1297 = x1296 * x1296;
float x1298 = x1295 + x1297;
x50[x1282] = x1298;
float x1300 = x48[x1282];
float x1302 = x50[x1282];
float x1301 = 0.1f * x1296;
double x1303 = (double)x1302;
double x1304 = x1303 + 9.99999993922529E-9;
double x1305 = sqrt(x1304);
float x1306 = (float)x1305;
float x1307 = x1301 / x1306;
float x1308 = x1300 - x1307;
x48[x1282] = x1308;
x49[x1282] = 0.0f;

}
for(int x1313=0; x1313 < 1300; x1313++) {
float x1314 = x47[x1313];
float x1315 = x1314;
float x1316 = x1315;
bool x1317 = x1316 > 5.0f;
if (x1317) {
x1315 = 5.0f;
} else {
}
float x1321 = x1315;
bool x1322 = x1321 < -5.0f;
if (x1322) {
x1315 = -5.0f;
} else {
}
float x1326 = x51[x1313];
float x1327 = x1315;
float x1328 = x1327 * x1327;
float x1329 = x1326 + x1328;
x51[x1313] = x1329;
float x1331 = x39[x1313];
float x1333 = x51[x1313];
float x1332 = 0.1f * x1327;
double x1334 = (double)x1333;
double x1335 = x1334 + 9.99999993922529E-9;
double x1336 = sqrt(x1335);
float x1337 = (float)x1336;
float x1338 = x1332 / x1337;
float x1339 = x1331 - x1338;
x39[x1313] = x1339;
x47[x1313] = 0.0f;

}
for(int x1344=0; x1344 < 2500; x1344++) {
float x1345 = x36[x1344];
float x1346 = x1345;
float x1347 = x1346;
bool x1348 = x1347 > 5.0f;
if (x1348) {
x1346 = 5.0f;
} else {
}
float x1352 = x1346;
bool x1353 = x1352 < -5.0f;
if (x1353) {
x1346 = -5.0f;
} else {
}
float x1357 = x52[x1344];
float x1358 = x1346;
float x1359 = x1358 * x1358;
float x1360 = x1357 + x1359;
x52[x1344] = x1360;
float x1362 = x27[x1344];
float x1364 = x52[x1344];
float x1363 = 0.1f * x1358;
double x1365 = (double)x1364;
double x1366 = x1365 + 9.99999993922529E-9;
double x1367 = sqrt(x1366);
float x1368 = (float)x1367;
float x1369 = x1363 / x1368;
float x1370 = x1362 - x1369;
x27[x1344] = x1370;
x36[x1344] = 0.0f;

}
for(int x1375=0; x1375 < 50; x1375++) {
float x1376 = x38[x1375];
float x1377 = x1376;
float x1378 = x1377;
bool x1379 = x1378 > 5.0f;
if (x1379) {
x1377 = 5.0f;
} else {
}
float x1383 = x1377;
bool x1384 = x1383 < -5.0f;
if (x1384) {
x1377 = -5.0f;
} else {
}
float x1388 = x53[x1375];
float x1389 = x1377;
float x1390 = x1389 * x1389;
float x1391 = x1388 + x1390;
x53[x1375] = x1391;
float x1393 = x37[x1375];
float x1395 = x53[x1375];
float x1394 = 0.1f * x1389;
double x1396 = (double)x1395;
double x1397 = x1396 + 9.99999993922529E-9;
double x1398 = sqrt(x1397);
float x1399 = (float)x1398;
float x1400 = x1394 / x1399;
float x1401 = x1393 - x1400;
x37[x1375] = x1401;
x38[x1375] = 0.0f;

}
for(int x1406=0; x1406 < 1300; x1406++) {
float x1407 = x26[x1406];
float x1408 = x1407;
float x1409 = x1408;
bool x1410 = x1409 > 5.0f;
if (x1410) {
x1408 = 5.0f;
} else {
}
float x1414 = x1408;
bool x1415 = x1414 < -5.0f;
if (x1415) {
x1408 = -5.0f;
} else {
}
float x1419 = x54[x1406];
float x1420 = x1408;
float x1421 = x1420 * x1420;
float x1422 = x1419 + x1421;
x54[x1406] = x1422;
float x1424 = x16[x1406];
float x1426 = x54[x1406];
float x1425 = 0.1f * x1420;
double x1427 = (double)x1426;
double x1428 = x1427 + 9.99999993922529E-9;
double x1429 = sqrt(x1428);
float x1430 = (float)x1429;
float x1431 = x1425 / x1430;
float x1432 = x1424 - x1431;
x16[x1406] = x1432;
x26[x1406] = 0.0f;

}
int64_t x1437 = (long)mallocAddr;
int64_t x1438 = x1437 - x57;
memset((void*)x57, 0, x1438);
mallocAddr = (void*)x57;

}
double x1443 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x1446 = (long)fopen(x0, "w");
fprintf((FILE *)x1446, "unit: %s\n", "100 iteration");
for(int x1449=0; x1449 < 51; x1449++) {
double x1450 = x55[x1449];
fprintf((FILE *)x1446, "%lf\n", x1450);

}
double x1444 = x56 - x1;
double x1445 = x1443 - x56;
fprintf((FILE *)x1446, "run time: %lf %lf\n", x1444, x1445);
fclose((FILE*)x1446);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

