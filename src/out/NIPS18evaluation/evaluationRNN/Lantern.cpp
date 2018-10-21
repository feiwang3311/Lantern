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

using namespace std;
#ifndef MAP_FILE
#define MAP_FILE MAP_SHARED
#endif

int fsize(int fd) {
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

int HEAP_SIZE = 1073741826; // 1048576; // 2147483652; // 536870912; // 268435456; // 2097152;
void *mallocBase = calloc(HEAP_SIZE, 1);
void *mallocAddr = mallocBase;
void *waterMark = mallocBase;
void *myMalloc(size_t bytes) {
  void *res = mallocAddr;
  mallocAddr = (void *)((char *)mallocAddr + bytes);
  return res;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
  long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
  result->tv_sec = diff / 1000000;
  result->tv_usec = diff % 1000000;
  return (diff < 0);
}



void Snippet(char *);

std::random_device rd{};
std::mt19937 gen{rd()};
std::normal_distribution<> d{0, 1};

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
int32_t x3 = fsize(x2);
int* x5 = (int32_t*)myMalloc(x3 * sizeof(int32_t));;
char* x4 = (char*)mmap(0, x3, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x2, 0);
for(int x7=0; x7 < x3; x7++) {
char x8 = x4[x7];
int32_t x9 = (int32_t ) x8;
int32_t x10 = x9 - 96;
x5[x7] = x10;

}
float* x14 = (float*)myMalloc(1300 * sizeof(float));;
for(int x16=0; x16 < 1300; x16++) {
float x17 = (float)rand()/RAND_MAX;
float x18 = x17 - 0.5f;
float x19 = x18 * 0.19611613f;
x14[x16] = x19;

}
float* x23 = (float*)myMalloc(1300 * sizeof(float));;
float* x24 = (float*)myMalloc(2500 * sizeof(float));;
for(int x26=0; x26 < 2500; x26++) {
float x27 = (float)rand()/RAND_MAX;
float x28 = x27 - 0.5f;
float x29 = x28 * 0.14142136f;
x24[x26] = x29;

}
float* x33 = (float*)myMalloc(2500 * sizeof(float));;
float* x34 = (float*)myMalloc(50 * sizeof(float));;
float* x35 = (float*)myMalloc(50 * sizeof(float));;
float* x36 = (float*)myMalloc(1300 * sizeof(float));;
for(int x37=0; x37 < 1300; x37++) {
float x38 = (float)rand()/RAND_MAX;
float x39 = x38 - 0.5f;
float x40 = x39 * 0.14142136f;
x36[x37] = x40;

}
float* x44 = (float*)myMalloc(1300 * sizeof(float));;
float* x45 = (float*)myMalloc(26 * sizeof(float));;
float* x46 = (float*)myMalloc(26 * sizeof(float));;
float* x47 = (float*)myMalloc(26 * sizeof(float));;
float* x48 = (float*)myMalloc(1300 * sizeof(float));;
float* x49 = (float*)myMalloc(2500 * sizeof(float));;
float* x50 = (float*)myMalloc(50 * sizeof(float));;
float* x51 = (float*)myMalloc(1300 * sizeof(float));;
double* x52 = (double*)myMalloc(51 * sizeof(double));;
double x53 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x54 = (long)mallocAddr;
int32_t x55 = 0;
x55 -= 400;
for(int x58=0; x58 < 5001; x58++) {
float* x82 = (float*)myMalloc(1 * sizeof(float));;
float* x83 = (float*)myMalloc(10400 * sizeof(float));;
float* x100 = (float*)myMalloc(10400 * sizeof(float));;
int* x68 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
function<void(int32_t,float**)> x351 = [&](int32_t x352,float** x353) {
float** x355 = x353;
float* x356 = x355[0];
float* x357 = x355[1];
float* x358 = x355[2];
float* x359 = x355[3];
int32_t x354 = x352;
bool x360 = x354 < 20;
if (x360) {
int32_t x361 = x354 * 520;
float* x362 = x83+x361;
float* x363 = x100+x361;
// dot: WrappedArray(20, 26), List(26, 50)
float* x365 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x362,26,x14,50,0,x365,50);
float* x367 = (float*)myMalloc(1000 * sizeof(float));;
// dot: List(20, 50), List(50, 50)
float* x369 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x358,50,x24,50,0,x369,50);
float* x371 = (float*)myMalloc(1000 * sizeof(float));;
float* x372 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x373 = 0;
int32_t x374 = 0;
int32_t x375 = 0;
for(int x376=0; x376 < 20; x376++) {
int32_t x377 = x374;
int32_t x378 = x375;
int32_t x379 = x373;
int32_t x380 = x379;
int32_t x381 = x377;
int32_t x382 = x378;
for(int x383=0; x383 < 50; x383++) {
int32_t x384 = x380;
int32_t x385 = x381;
float x386 = x365[x385];
int32_t x387 = x382;
float x388 = x369[x387];
float x389 = x386 + x388;
x372[x384] = x389;
x380 += 1;
x381 += 1;
x382 += 1;

}
x373 += 50;
x374 += 50;
x375 += 50;

}
float* x401 = (float*)myMalloc(1000 * sizeof(float));;
float* x402 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x403 = 0;
int32_t x404 = 0;
int32_t x405 = 0;
for(int x406=0; x406 < 20; x406++) {
int32_t x407 = x404;
int32_t x408 = x405;
int32_t x409 = x403;
int32_t x410 = x409;
int32_t x411 = x407;
int32_t x412 = x408;
for(int x413=0; x413 < 50; x413++) {
int32_t x414 = x410;
int32_t x415 = x411;
float x416 = x372[x415];
int32_t x417 = x412;
float x418 = x34[x417];
float x419 = x416 + x418;
x402[x414] = x419;
x410 += 1;
x411 += 1;
x412 += 1;

}
x403 += 50;
x404 += 50;

}
float* x430 = (float*)myMalloc(1000 * sizeof(float));;
float* x431 = (float*)myMalloc(1000 * sizeof(float));;
for(int x432=0; x432 < 1000; x432++) {
float x433 = x402[x432];
double x434 = (double)x433;
double x435 = tanh(x434);
float x436 = (float)x435;
x431[x432] = x436;

}
float* x440 = (float*)myMalloc(1000 * sizeof(float));;
// dot: List(20, 50), List(50, 26)
float* x442 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x431,50,x36,26,0,x442,26);
float* x444 = (float*)myMalloc(520 * sizeof(float));;
float* x445 = (float*)myMalloc(520 * sizeof(float));;
int32_t x446 = 0;
int32_t x447 = 0;
int32_t x448 = 0;
for(int x449=0; x449 < 20; x449++) {
int32_t x450 = x447;
int32_t x451 = x448;
int32_t x452 = x446;
int32_t x453 = x452;
int32_t x454 = x450;
int32_t x455 = x451;
for(int x456=0; x456 < 26; x456++) {
int32_t x457 = x453;
int32_t x458 = x454;
float x459 = x442[x458];
int32_t x460 = x455;
float x461 = x45[x460];
float x462 = x459 + x461;
x445[x457] = x462;
x453 += 1;
x454 += 1;
x455 += 1;

}
x446 += 26;
x447 += 26;

}
float* x473 = (float*)myMalloc(520 * sizeof(float));;
int* x474 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x475=0; x475 < 20; x475++) {
int32_t x476 = x475 * 20;
int32_t x477 = x354 + x476;
int32_t x478 = x68[x477];
x474[x475] = x478;

}
float* x482 = (float*)myMalloc(20 * sizeof(float));;
int32_t x483 = 0;
for(int x484=0; x484 < 20; x484++) {
float x485 = -3.4028235E38f;
for(int x486=0; x486 < 26; x486++) {
int32_t x487 = x483;
float x488 = x445[x487];
float x489 = x485;
bool x490 = x488 > x489;
if (x490) {
float x491 = x445[x487];
x485 = x491;
} else {
}
x483 += 1;

}
float x498 = x485;
x482[x484] = x498;

}
float* x502 = (float*)myMalloc(520 * sizeof(float));;
int32_t x503 = 0;
for(int x504=0; x504 < 20; x504++) {
for(int x505=0; x505 < 26; x505++) {
int32_t x506 = x503;
float x507 = x445[x506];
float x508 = x482[x504];
float x509 = x507 - x508;
double x510 = (double)x509;
double x511 = exp(x510);
float x512 = (float)x511;
x502[x506] = x512;
x503 += 1;

}

}
float* x519 = (float*)myMalloc(20 * sizeof(float));;
for(int x520=0; x520 < 20; x520++) {
int32_t x521 = x520;
int32_t x522 = x520 * 26;
int32_t x523 = x522;
for(int x524=0; x524 < 26; x524++) {
int32_t x525 = x521;
float x526 = x519[x525];
int32_t x527 = x523;
float x528 = x502[x527];
float x529 = x526 + x528;
x519[x525] = x529;
x523 += 1;

}

}
x503 = 0;
for(int x537=0; x537 < 20; x537++) {
float x538 = x482[x537];
float x539 = x519[x537];
double x540 = (double)x539;
double x541 = log(x540);
float x542 = (float)x541;
float x543 = x538 + x542;
for(int x544=0; x544 < 26; x544++) {
int32_t x545 = x503;
float x546 = x445[x545];
float x547 = x546 - x543;
x502[x545] = x547;
x503 += 1;

}

}
float* x554 = (float*)myMalloc(520 * sizeof(float));;
float* x555 = (float*)myMalloc(20 * sizeof(float));;
int32_t x556 = 0;
for(int x557=0; x557 < 20; x557++) {
int32_t x558 = x556;
int32_t x559 = x474[x557];
int32_t x560 = x558 + x559;
float x561 = x502[x560];
float x562 = -1.0f * x561;
x555[x557] = x562;
x556 += 26;

}
float* x567 = (float*)myMalloc(20 * sizeof(float));;
float x568 = 0.0f;
for(int x569=0; x569 < 20; x569++) {
float x570 = x568;
float x571 = x555[x569];
float x572 = x570 + x571;
x568 = x572;

}
float x576 = x568;
float* x577 = (float*)myMalloc(1 * sizeof(float));;
x577[0] = x576;
float* x579 = (float*)myMalloc(1 * sizeof(float));;
float* x580 = (float*)myMalloc(1 * sizeof(float));;
int32_t x581 = 0;
int32_t x582 = 0;
int32_t x583 = 0;
int32_t x584 = x581;
int32_t x585 = x582;
float x586 = x356[x585];
int32_t x587 = x583;
float x588 = x577[x587];
float x589 = x586 + x588;
x580[x584] = x589;
x581 += 1;
float* x592 = (float*)myMalloc(1 * sizeof(float));;
float** x594 = (float**)myMalloc(4 * sizeof(float*));;
x594[0] = x580;
x594[1] = x592;
x594[2] = x431;
x594[3] = x440;
int32_t x601 = 0;
int32_t x602 = 0;
int32_t x603 = 0;
int32_t x604 = x601;
int32_t x607 = x602;
int32_t x609 = x603;
x603 += 1;
int32_t x628 = 0;
float* x641 = (float*)myMalloc(20 * sizeof(float));;
int32_t x658 = 0;
int32_t x678 = 0;
int32_t x679 = 0;
int32_t x680 = 0;
int32_t x726 = 0;
int32_t x727 = 0;
int32_t x728 = 0;
int32_t x761 = 0;
int32_t x762 = 0;
int32_t x763 = 0;
int32_t x593 = x354 + 1;
x351(x593,x594);
float x605 = x357[x604];
float x606 = x356[x604];
float x608 = x577[x607];
float x610 = x592[x609];
float x611 = x605 + x610;
x357[x604] = x611;
float x613 = x579[x607];
float x614 = x356[x604];
float x615 = x577[x607];
float x616 = x592[x609];
float x617 = x613 + x616;
x579[x607] = x617;
// += tensor of dim 0
float x621 = x579[0];
for(int x622=0; x622 < 20; x622++) {
float x623 = x567[x622];
float x624 = x623 + x621;
x567[x622] = x624;

}
for(int x629=0; x629 < 20; x629++) {
int32_t x630 = x628;
int32_t x631 = x474[x629];
int32_t x632 = x630 + x631;
float x633 = x554[x632];
float x634 = x567[x629];
float x635 = -1.0f * x634;
float x636 = x633 + x635;
x554[x632] = x636;
x628 += 26;

}
for(int x642=0; x642 < 20; x642++) {
int32_t x643 = x642;
int32_t x644 = x642 * 26;
int32_t x645 = x644;
for(int x646=0; x646 < 26; x646++) {
int32_t x647 = x643;
float x648 = x641[x647];
int32_t x649 = x645;
float x650 = x554[x649];
float x651 = x648 + x650;
x641[x647] = x651;
x645 += 1;

}

}
for(int x659=0; x659 < 20; x659++) {
for(int x660=0; x660 < 26; x660++) {
int32_t x661 = x658;
float x662 = x473[x661];
float x663 = x554[x661];
float x664 = x502[x661];
float x668 = x641[x659];
double x665 = (double)x664;
double x666 = exp(x665);
float x667 = (float)x666;
float x669 = x667 * x668;
float x670 = x663 - x669;
float x671 = x662 + x670;
x473[x661] = x671;
x658 += 1;

}

}
for(int x681=0; x681 < 20; x681++) {
int32_t x682 = x678;
int32_t x683 = x679;
int32_t x684 = x680;
int32_t x685 = x682;
int32_t x686 = x683;
int32_t x687 = x684;
for(int x688=0; x688 < 26; x688++) {
int32_t x689 = x685;
float x690 = x444[x689];
float x691 = x442[x689];
int32_t x692 = x686;
float x693 = x45[x692];
int32_t x694 = x687;
float x695 = x473[x694];
float x696 = x690 + x695;
x444[x689] = x696;
float x698 = x46[x692];
float x699 = x442[x689];
float x700 = x45[x692];
float x701 = x473[x694];
float x702 = x698 + x701;
x46[x692] = x702;
x687 += 1;
x685 += 1;
x686 += 1;

}
x680 += 26;
x678 += 26;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x444,26,x36,26,1,x440,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x431,50,x444,26,1,x44,26);
for(int x715=0; x715 < 1000; x715++) {
float x716 = x430[x715];
float x717 = x431[x715];
float x720 = x440[x715];
float x718 = x717 * x717;
float x719 = 1.0f - x718;
float x721 = x719 * x720;
float x722 = x716 + x721;
x430[x715] = x722;

}
for(int x729=0; x729 < 20; x729++) {
int32_t x730 = x726;
int32_t x731 = x727;
int32_t x732 = x728;
int32_t x733 = x730;
int32_t x734 = x731;
int32_t x735 = x732;
for(int x736=0; x736 < 50; x736++) {
int32_t x737 = x733;
float x738 = x401[x737];
float x739 = x372[x737];
int32_t x740 = x734;
float x741 = x34[x740];
int32_t x742 = x735;
float x743 = x430[x742];
float x744 = x738 + x743;
x401[x737] = x744;
float x746 = x35[x740];
float x747 = x372[x737];
float x748 = x34[x740];
float x749 = x430[x742];
float x750 = x746 + x749;
x35[x740] = x750;
x735 += 1;
x733 += 1;
x734 += 1;

}
x728 += 50;
x726 += 50;

}
for(int x764=0; x764 < 20; x764++) {
int32_t x765 = x761;
int32_t x766 = x762;
int32_t x767 = x763;
int32_t x768 = x765;
int32_t x769 = x766;
int32_t x770 = x767;
for(int x771=0; x771 < 50; x771++) {
int32_t x772 = x768;
float x773 = x367[x772];
float x774 = x365[x772];
int32_t x775 = x769;
float x776 = x369[x775];
int32_t x777 = x770;
float x778 = x401[x777];
float x779 = x773 + x778;
x367[x772] = x779;
float x781 = x371[x775];
float x782 = x365[x772];
float x783 = x369[x775];
float x784 = x401[x777];
float x785 = x781 + x784;
x371[x775] = x785;
x770 += 1;
x768 += 1;
x769 += 1;

}
x763 += 50;
x761 += 50;
x762 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x371,50,x24,50,1,x359,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x358,50,x371,50,1,x33,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x367,50,x14,50,1,x363,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x362,26,x367,50,1,x23,50);
} else {
float x802 = 0.0f;
float x803 = x802;
float x804 = x356[0];
float x805 = x803 + x804;
x802 = x805;
float x807 = x802;
float* x808 = (float*)myMalloc(1 * sizeof(float));;
x808[0] = x807;
float* x810 = (float*)myMalloc(1 * sizeof(float));;
x810[0] = 1.0f;
float x812 = x808[0];
x82[0] = x812;
// += tensor of dim 0
float x815 = x810[0];
float x816 = x357[0];
float x817 = x816 + x815;
x357[0] = x817;
}
};
x55 += 400;
int32_t x60 = x55;
int32_t x61 = x60 + 400;
int32_t x62 = x61 + 1;
bool x63 = x62 >= x3;
if (x63) {
x55 = 0;
} else {
}
int* x67 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
for(int x70=0; x70 < 400; x70++) {
int32_t x71 = x55;
int32_t x72 = x71 + x70;
int32_t x73 = x5[x72];
x67[x70] = x73;
int32_t x75 = x72 + 1;
int32_t x76 = x5[x75];
x68[x70] = x76;

}
float* x80 = (float*)myMalloc(1 * sizeof(float));;
float* x81 = (float*)myMalloc(1 * sizeof(float));;
for(int x85=0; x85 < 20; x85++) {
int32_t x87 = x85 * 26;
int32_t x88 = x87 * 20;
for(int x86=0; x86 < 20; x86++) {
int32_t x91 = x86 * 20;
int32_t x92 = x91 + x85;
int32_t x93 = x67[x92];
int32_t x89 = x86 * 26;
int32_t x90 = x88 + x89;
int32_t x94 = x90 + x93;
x83[x94] = 1.0f;

}

}
float* x101 = (float*)myMalloc(1 * sizeof(float));;
float* x102 = (float*)myMalloc(1 * sizeof(float));;
float* x103 = (float*)myMalloc(1000 * sizeof(float));;
float* x104 = (float*)myMalloc(1000 * sizeof(float));;
float** x1050 = (float**)myMalloc(4 * sizeof(float*));;
x1050[0] = x101;
x1050[1] = x102;
x1050[2] = x103;
x1050[3] = x104;
function<void(int32_t,float**)> x105 = [&](int32_t x106,float** x107) {
float** x109 = x107;
float* x110 = x109[0];
float* x111 = x109[1];
float* x112 = x109[2];
float* x113 = x109[3];
int32_t x108 = x106;
bool x114 = x108 < 20;
if (x114) {
int32_t x115 = x108 * 520;
float* x116 = x83+x115;
float* x117 = x100+x115;
// dot: WrappedArray(20, 26), List(26, 50)
float* x119 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x116,26,x14,50,0,x119,50);
float* x121 = (float*)myMalloc(1000 * sizeof(float));;
// dot: WrappedArray(20, 50), List(50, 50)
float* x123 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x112,50,x24,50,0,x123,50);
float* x125 = (float*)myMalloc(1000 * sizeof(float));;
float* x126 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x127 = 0;
int32_t x128 = 0;
int32_t x129 = 0;
for(int x130=0; x130 < 20; x130++) {
int32_t x131 = x128;
int32_t x132 = x129;
int32_t x133 = x127;
int32_t x134 = x133;
int32_t x135 = x131;
int32_t x136 = x132;
for(int x138=0; x138 < 50; x138++) {
int32_t x139 = x134;
int32_t x140 = x135;
float x141 = x119[x140];
int32_t x142 = x136;
float x143 = x123[x142];
float x144 = x141 + x143;
x126[x139] = x144;
x134 += 1;
x135 += 1;
x136 += 1;

}
x127 += 50;
x128 += 50;
x129 += 50;

}
float* x156 = (float*)myMalloc(1000 * sizeof(float));;
float* x157 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x158 = 0;
int32_t x159 = 0;
int32_t x160 = 0;
for(int x161=0; x161 < 20; x161++) {
int32_t x162 = x159;
int32_t x163 = x160;
int32_t x164 = x158;
int32_t x165 = x164;
int32_t x166 = x162;
int32_t x167 = x163;
for(int x168=0; x168 < 50; x168++) {
int32_t x169 = x165;
int32_t x170 = x166;
float x171 = x126[x170];
int32_t x172 = x167;
float x173 = x34[x172];
float x174 = x171 + x173;
x157[x169] = x174;
x165 += 1;
x166 += 1;
x167 += 1;

}
x158 += 50;
x159 += 50;

}
float* x185 = (float*)myMalloc(1000 * sizeof(float));;
float* x186 = (float*)myMalloc(1000 * sizeof(float));;
for(int x188=0; x188 < 1000; x188++) {
float x189 = x157[x188];
double x190 = (double)x189;
double x191 = tanh(x190);
float x192 = (float)x191;
x186[x188] = x192;

}
float* x196 = (float*)myMalloc(1000 * sizeof(float));;
// dot: List(20, 50), List(50, 26)
float* x198 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x186,50,x36,26,0,x198,26);
float* x200 = (float*)myMalloc(520 * sizeof(float));;
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
int32_t x214 = x209;
int32_t x215 = x210;
float x216 = x198[x215];
int32_t x217 = x211;
float x218 = x45[x217];
float x219 = x216 + x218;
x201[x214] = x219;
x209 += 1;
x210 += 1;
x211 += 1;

}
x202 += 26;
x203 += 26;

}
float* x230 = (float*)myMalloc(520 * sizeof(float));;
int* x231 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x232=0; x232 < 20; x232++) {
int32_t x233 = x232 * 20;
int32_t x234 = x108 + x233;
int32_t x235 = x68[x234];
x231[x232] = x235;

}
float* x239 = (float*)myMalloc(20 * sizeof(float));;
int32_t x240 = 0;
for(int x241=0; x241 < 20; x241++) {
float x242 = -3.4028235E38f;
for(int x243=0; x243 < 26; x243++) {
int32_t x244 = x240;
float x245 = x201[x244];
float x246 = x242;
bool x247 = x245 > x246;
if (x247) {
float x248 = x201[x244];
x242 = x248;
} else {
}
x240 += 1;

}
float x255 = x242;
x239[x241] = x255;

}
float* x259 = (float*)myMalloc(520 * sizeof(float));;
int32_t x260 = 0;
for(int x261=0; x261 < 20; x261++) {
for(int x262=0; x262 < 26; x262++) {
int32_t x263 = x260;
float x264 = x201[x263];
float x265 = x239[x261];
float x266 = x264 - x265;
double x267 = (double)x266;
double x268 = exp(x267);
float x269 = (float)x268;
x259[x263] = x269;
x260 += 1;

}

}
float* x276 = (float*)myMalloc(20 * sizeof(float));;
for(int x277=0; x277 < 20; x277++) {
int32_t x278 = x277;
int32_t x279 = x277 * 26;
int32_t x280 = x279;
for(int x281=0; x281 < 26; x281++) {
int32_t x282 = x278;
float x283 = x276[x282];
int32_t x284 = x280;
float x285 = x259[x284];
float x286 = x283 + x285;
x276[x282] = x286;
x280 += 1;

}

}
x260 = 0;
for(int x294=0; x294 < 20; x294++) {
float x295 = x239[x294];
float x296 = x276[x294];
double x297 = (double)x296;
double x298 = log(x297);
float x299 = (float)x298;
float x300 = x295 + x299;
for(int x301=0; x301 < 26; x301++) {
int32_t x302 = x260;
float x303 = x201[x302];
float x304 = x303 - x300;
x259[x302] = x304;
x260 += 1;

}

}
float* x311 = (float*)myMalloc(520 * sizeof(float));;
float* x312 = (float*)myMalloc(20 * sizeof(float));;
int32_t x313 = 0;
for(int x314=0; x314 < 20; x314++) {
int32_t x315 = x313;
int32_t x316 = x231[x314];
int32_t x317 = x315 + x316;
float x318 = x259[x317];
float x319 = -1.0f * x318;
x312[x314] = x319;
x313 += 26;

}
float* x324 = (float*)myMalloc(20 * sizeof(float));;
float x325 = 0.0f;
for(int x326=0; x326 < 20; x326++) {
float x327 = x325;
float x328 = x312[x326];
float x329 = x327 + x328;
x325 = x329;

}
float x333 = x325;
float* x334 = (float*)myMalloc(1 * sizeof(float));;
x334[0] = x333;
float* x336 = (float*)myMalloc(1 * sizeof(float));;
float* x337 = (float*)myMalloc(1 * sizeof(float));;
int32_t x338 = 0;
int32_t x339 = 0;
int32_t x340 = 0;
int32_t x341 = x338;
int32_t x342 = x339;
float x343 = x110[x342];
int32_t x344 = x340;
float x345 = x334[x344];
float x346 = x343 + x345;
x337[x341] = x346;
x338 += 1;
float* x349 = (float*)myMalloc(1 * sizeof(float));;
float** x822 = (float**)myMalloc(4 * sizeof(float*));;
x822[0] = x337;
x822[1] = x349;
x822[2] = x186;
x822[3] = x196;
int32_t x350 = x108 + 1;
x351(x350,x822);
int32_t x829 = 0;
int32_t x830 = 0;
int32_t x831 = 0;
int32_t x832 = x829;
float x833 = x111[x832];
float x834 = x110[x832];
int32_t x835 = x830;
float x836 = x334[x835];
int32_t x837 = x831;
float x838 = x349[x837];
float x839 = x833 + x838;
x111[x832] = x839;
float x841 = x336[x835];
float x842 = x110[x832];
float x843 = x334[x835];
float x844 = x349[x837];
float x845 = x841 + x844;
x336[x835] = x845;
x831 += 1;
// += tensor of dim 0
float x849 = x336[0];
for(int x850=0; x850 < 20; x850++) {
float x851 = x324[x850];
float x852 = x851 + x849;
x324[x850] = x852;

}
int32_t x856 = 0;
for(int x857=0; x857 < 20; x857++) {
int32_t x858 = x856;
int32_t x859 = x231[x857];
int32_t x860 = x858 + x859;
float x861 = x311[x860];
float x862 = x324[x857];
float x863 = -1.0f * x862;
float x864 = x861 + x863;
x311[x860] = x864;
x856 += 26;

}
float* x869 = (float*)myMalloc(20 * sizeof(float));;
for(int x870=0; x870 < 20; x870++) {
int32_t x871 = x870;
int32_t x872 = x870 * 26;
int32_t x873 = x872;
for(int x874=0; x874 < 26; x874++) {
int32_t x875 = x871;
float x876 = x869[x875];
int32_t x877 = x873;
float x878 = x311[x877];
float x879 = x876 + x878;
x869[x875] = x879;
x873 += 1;

}

}
int32_t x886 = 0;
for(int x887=0; x887 < 20; x887++) {
for(int x888=0; x888 < 26; x888++) {
int32_t x889 = x886;
float x890 = x230[x889];
float x891 = x311[x889];
float x892 = x259[x889];
float x896 = x869[x887];
double x893 = (double)x892;
double x894 = exp(x893);
float x895 = (float)x894;
float x897 = x895 * x896;
float x898 = x891 - x897;
float x899 = x890 + x898;
x230[x889] = x899;
x886 += 1;

}

}
int32_t x906 = 0;
int32_t x907 = 0;
int32_t x908 = 0;
for(int x909=0; x909 < 20; x909++) {
int32_t x910 = x906;
int32_t x911 = x907;
int32_t x912 = x908;
int32_t x913 = x910;
int32_t x914 = x911;
int32_t x915 = x912;
for(int x916=0; x916 < 26; x916++) {
int32_t x917 = x913;
float x918 = x200[x917];
float x919 = x198[x917];
int32_t x920 = x914;
float x921 = x45[x920];
int32_t x922 = x915;
float x923 = x230[x922];
float x924 = x918 + x923;
x200[x917] = x924;
float x926 = x46[x920];
float x927 = x198[x917];
float x928 = x45[x920];
float x929 = x230[x922];
float x930 = x926 + x929;
x46[x920] = x930;
x915 += 1;
x913 += 1;
x914 += 1;

}
x908 += 26;
x906 += 26;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x200,26,x36,26,1,x196,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x186,50,x200,26,1,x44,26);
for(int x943=0; x943 < 1000; x943++) {
float x944 = x185[x943];
float x945 = x186[x943];
float x948 = x196[x943];
float x946 = x945 * x945;
float x947 = 1.0f - x946;
float x949 = x947 * x948;
float x950 = x944 + x949;
x185[x943] = x950;

}
int32_t x954 = 0;
int32_t x955 = 0;
int32_t x956 = 0;
for(int x957=0; x957 < 20; x957++) {
int32_t x958 = x954;
int32_t x959 = x955;
int32_t x960 = x956;
int32_t x961 = x958;
int32_t x962 = x959;
int32_t x963 = x960;
for(int x964=0; x964 < 50; x964++) {
int32_t x965 = x961;
float x966 = x156[x965];
float x967 = x126[x965];
int32_t x968 = x962;
float x969 = x34[x968];
int32_t x970 = x963;
float x971 = x185[x970];
float x972 = x966 + x971;
x156[x965] = x972;
float x974 = x35[x968];
float x975 = x126[x965];
float x976 = x34[x968];
float x977 = x185[x970];
float x978 = x974 + x977;
x35[x968] = x978;
x963 += 1;
x961 += 1;
x962 += 1;

}
x956 += 50;
x954 += 50;

}
int32_t x989 = 0;
int32_t x990 = 0;
int32_t x991 = 0;
for(int x992=0; x992 < 20; x992++) {
int32_t x993 = x989;
int32_t x994 = x990;
int32_t x995 = x991;
int32_t x996 = x993;
int32_t x997 = x994;
int32_t x998 = x995;
for(int x999=0; x999 < 50; x999++) {
int32_t x1000 = x996;
float x1001 = x121[x1000];
float x1002 = x119[x1000];
int32_t x1003 = x997;
float x1004 = x123[x1003];
int32_t x1005 = x998;
float x1006 = x156[x1005];
float x1007 = x1001 + x1006;
x121[x1000] = x1007;
float x1009 = x125[x1003];
float x1010 = x119[x1000];
float x1011 = x123[x1003];
float x1012 = x156[x1005];
float x1013 = x1009 + x1012;
x125[x1003] = x1013;
x998 += 1;
x996 += 1;
x997 += 1;

}
x991 += 50;
x989 += 50;
x990 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x125,50,x24,50,1,x113,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x112,50,x125,50,1,x33,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x121,50,x14,50,1,x117,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x116,26,x121,50,1,x23,50);
} else {
float x1030 = 0.0f;
float x1031 = x1030;
float x1032 = x110[0];
float x1033 = x1031 + x1032;
x1030 = x1033;
float x1035 = x1030;
float* x1036 = (float*)myMalloc(1 * sizeof(float));;
x1036[0] = x1035;
float* x1038 = (float*)myMalloc(1 * sizeof(float));;
x1038[0] = 1.0f;
float x1040 = x1036[0];
x82[0] = x1040;
// += tensor of dim 0
float x1043 = x1038[0];
float x1044 = x111[0];
float x1045 = x1044 + x1043;
x111[0] = x1045;
}
};
x105(0,x1050);
float x1057 = x82[0];
int32_t x1058 = x58 % 100;
bool x1059 = x1058 == 0;
if (x1059) {
printf("iter %d, loss %f\n",x58,x1057);
int32_t x1061 = x58 / 100;
double x1062 = (double)x1057;
x52[x1061] = x1062;
} else {
}
for(int x1066=0; x1066 < 26; x1066++) {
float x1067 = x46[x1066];
float x1068 = x1067;
float x1069 = x1068;
bool x1070 = x1069 > 5.0f;
if (x1070) {
x1068 = 5.0f;
} else {
}
float x1074 = x1068;
bool x1075 = x1074 < -5.0f;
if (x1075) {
x1068 = -5.0f;
} else {
}
float x1079 = x47[x1066];
float x1080 = x1068;
float x1081 = x1080 * x1080;
float x1082 = x1079 + x1081;
x47[x1066] = x1082;
float x1084 = x45[x1066];
float x1086 = x47[x1066];
float x1085 = 0.1f * x1080;
double x1087 = (double)x1086;
double x1088 = x1087 + 9.99999993922529E-9;
double x1089 = sqrt(x1088);
float x1090 = (float)x1089;
float x1091 = x1085 / x1090;
float x1092 = x1084 - x1091;
x45[x1066] = x1092;
x46[x1066] = 0.0f;

}
for(int x1097=0; x1097 < 1300; x1097++) {
float x1098 = x44[x1097];
float x1099 = x1098;
float x1100 = x1099;
bool x1101 = x1100 > 5.0f;
if (x1101) {
x1099 = 5.0f;
} else {
}
float x1105 = x1099;
bool x1106 = x1105 < -5.0f;
if (x1106) {
x1099 = -5.0f;
} else {
}
float x1110 = x48[x1097];
float x1111 = x1099;
float x1112 = x1111 * x1111;
float x1113 = x1110 + x1112;
x48[x1097] = x1113;
float x1115 = x36[x1097];
float x1117 = x48[x1097];
float x1116 = 0.1f * x1111;
double x1118 = (double)x1117;
double x1119 = x1118 + 9.99999993922529E-9;
double x1120 = sqrt(x1119);
float x1121 = (float)x1120;
float x1122 = x1116 / x1121;
float x1123 = x1115 - x1122;
x36[x1097] = x1123;
x44[x1097] = 0.0f;

}
for(int x1128=0; x1128 < 2500; x1128++) {
float x1129 = x33[x1128];
float x1130 = x1129;
float x1131 = x1130;
bool x1132 = x1131 > 5.0f;
if (x1132) {
x1130 = 5.0f;
} else {
}
float x1136 = x1130;
bool x1137 = x1136 < -5.0f;
if (x1137) {
x1130 = -5.0f;
} else {
}
float x1141 = x49[x1128];
float x1142 = x1130;
float x1143 = x1142 * x1142;
float x1144 = x1141 + x1143;
x49[x1128] = x1144;
float x1146 = x24[x1128];
float x1148 = x49[x1128];
float x1147 = 0.1f * x1142;
double x1149 = (double)x1148;
double x1150 = x1149 + 9.99999993922529E-9;
double x1151 = sqrt(x1150);
float x1152 = (float)x1151;
float x1153 = x1147 / x1152;
float x1154 = x1146 - x1153;
x24[x1128] = x1154;
x33[x1128] = 0.0f;

}
for(int x1159=0; x1159 < 50; x1159++) {
float x1160 = x35[x1159];
float x1161 = x1160;
float x1162 = x1161;
bool x1163 = x1162 > 5.0f;
if (x1163) {
x1161 = 5.0f;
} else {
}
float x1167 = x1161;
bool x1168 = x1167 < -5.0f;
if (x1168) {
x1161 = -5.0f;
} else {
}
float x1172 = x50[x1159];
float x1173 = x1161;
float x1174 = x1173 * x1173;
float x1175 = x1172 + x1174;
x50[x1159] = x1175;
float x1177 = x34[x1159];
float x1179 = x50[x1159];
float x1178 = 0.1f * x1173;
double x1180 = (double)x1179;
double x1181 = x1180 + 9.99999993922529E-9;
double x1182 = sqrt(x1181);
float x1183 = (float)x1182;
float x1184 = x1178 / x1183;
float x1185 = x1177 - x1184;
x34[x1159] = x1185;
x35[x1159] = 0.0f;

}
for(int x1190=0; x1190 < 1300; x1190++) {
float x1191 = x23[x1190];
float x1192 = x1191;
float x1193 = x1192;
bool x1194 = x1193 > 5.0f;
if (x1194) {
x1192 = 5.0f;
} else {
}
float x1198 = x1192;
bool x1199 = x1198 < -5.0f;
if (x1199) {
x1192 = -5.0f;
} else {
}
float x1203 = x51[x1190];
float x1204 = x1192;
float x1205 = x1204 * x1204;
float x1206 = x1203 + x1205;
x51[x1190] = x1206;
float x1208 = x14[x1190];
float x1210 = x51[x1190];
float x1209 = 0.1f * x1204;
double x1211 = (double)x1210;
double x1212 = x1211 + 9.99999993922529E-9;
double x1213 = sqrt(x1212);
float x1214 = (float)x1213;
float x1215 = x1209 / x1214;
float x1216 = x1208 - x1215;
x14[x1190] = x1216;
x23[x1190] = 0.0f;

}
int64_t x1221 = (long)mallocAddr;
int64_t x1222 = x1221 - x54;
memset((void*)x54, 0, x1222);
mallocAddr = (void*)x54;

}
double x1227 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x1230 = (long)fopen(x0, "w");
fprintf((FILE *)x1230, "unit: %s\n", "100 iteration");
for(int x1233=0; x1233 < 51; x1233++) {
double x1234 = x52[x1233];
fprintf((FILE *)x1230, "%lf\n", x1234);

}
double x1228 = x53 - x1;
double x1229 = x1227 - x53;
fprintf((FILE *)x1230, "run time: %lf %lf\n", x1228, x1229);
fclose((FILE*)x1230);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

