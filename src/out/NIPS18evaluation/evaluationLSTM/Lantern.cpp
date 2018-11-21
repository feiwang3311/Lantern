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
// Backend setup.
double x2 = ((double)clock() / CLOCKS_PER_SEC);
int32_t x3 = open("graham.txt",0);
int64_t x4 = fsize(x3);
int32_t x5 = (int32_t)x4;
printf("LSTM Test: >> data has %d chars\n",x5);
int* x9 = (int32_t*)myMalloc(x5 * sizeof(int32_t));;
int64_t x6 = (int64_t)x5;
char* x7 = (char*)mmap(0, x6, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x3, 0);
for(int x11=0; x11 < x5; x11++) {
char x12 = x7[x11];
int32_t x13 = (int32_t ) x12;
int32_t x14 = x13 - 96;
x9[x11] = x14;

}
float* x18 = (float*)myMalloc(1300 * sizeof(float));;
for(int x20=0; x20 < 1300; x20++) {
float x21 = (float)rand()/RAND_MAX;
float x22 = x21 - 0.5f;
float x23 = x22 * 0.19611613f;
x18[x20] = x23;

}
float* x28 = (float*)myMalloc(1300 * sizeof(float));;
float* x29 = (float*)myMalloc(2500 * sizeof(float));;
for(int x31=0; x31 < 2500; x31++) {
float x32 = (float)rand()/RAND_MAX;
float x33 = x32 - 0.5f;
float x34 = x33 * 0.14142136f;
x29[x31] = x34;

}
float* x38 = (float*)myMalloc(2500 * sizeof(float));;
float* x39 = (float*)myMalloc(50 * sizeof(float));;
float* x40 = (float*)myMalloc(50 * sizeof(float));;
float* x41 = (float*)myMalloc(1300 * sizeof(float));;
for(int x42=0; x42 < 1300; x42++) {
float x43 = (float)rand()/RAND_MAX;
float x44 = x43 - 0.5f;
float x45 = x44 * 0.19611613f;
x41[x42] = x45;

}
float* x49 = (float*)myMalloc(1300 * sizeof(float));;
float* x50 = (float*)myMalloc(2500 * sizeof(float));;
for(int x51=0; x51 < 2500; x51++) {
float x52 = (float)rand()/RAND_MAX;
float x53 = x52 - 0.5f;
float x54 = x53 * 0.14142136f;
x50[x51] = x54;

}
float* x58 = (float*)myMalloc(2500 * sizeof(float));;
float* x59 = (float*)myMalloc(50 * sizeof(float));;
float* x60 = (float*)myMalloc(50 * sizeof(float));;
float* x61 = (float*)myMalloc(1300 * sizeof(float));;
for(int x62=0; x62 < 1300; x62++) {
float x63 = (float)rand()/RAND_MAX;
float x64 = x63 - 0.5f;
float x65 = x64 * 0.19611613f;
x61[x62] = x65;

}
float* x69 = (float*)myMalloc(1300 * sizeof(float));;
float* x70 = (float*)myMalloc(2500 * sizeof(float));;
for(int x71=0; x71 < 2500; x71++) {
float x72 = (float)rand()/RAND_MAX;
float x73 = x72 - 0.5f;
float x74 = x73 * 0.14142136f;
x70[x71] = x74;

}
float* x78 = (float*)myMalloc(2500 * sizeof(float));;
float* x79 = (float*)myMalloc(50 * sizeof(float));;
float* x80 = (float*)myMalloc(50 * sizeof(float));;
float* x81 = (float*)myMalloc(1300 * sizeof(float));;
for(int x82=0; x82 < 1300; x82++) {
float x83 = (float)rand()/RAND_MAX;
float x84 = x83 - 0.5f;
float x85 = x84 * 0.19611613f;
x81[x82] = x85;

}
float* x89 = (float*)myMalloc(1300 * sizeof(float));;
float* x90 = (float*)myMalloc(2500 * sizeof(float));;
for(int x91=0; x91 < 2500; x91++) {
float x92 = (float)rand()/RAND_MAX;
float x93 = x92 - 0.5f;
float x94 = x93 * 0.14142136f;
x90[x91] = x94;

}
float* x98 = (float*)myMalloc(2500 * sizeof(float));;
float* x99 = (float*)myMalloc(50 * sizeof(float));;
float* x100 = (float*)myMalloc(50 * sizeof(float));;
float* x101 = (float*)myMalloc(1300 * sizeof(float));;
for(int x102=0; x102 < 1300; x102++) {
float x103 = (float)rand()/RAND_MAX;
float x104 = x103 - 0.5f;
float x105 = x104 * 0.14142136f;
x101[x102] = x105;

}
float* x109 = (float*)myMalloc(1300 * sizeof(float));;
float* x110 = (float*)myMalloc(26 * sizeof(float));;
float* x111 = (float*)myMalloc(26 * sizeof(float));;
float* x112 = (float*)myMalloc(1300 * sizeof(float));;
float* x113 = (float*)myMalloc(50 * sizeof(float));;
float* x114 = (float*)myMalloc(2500 * sizeof(float));;
float* x115 = (float*)myMalloc(50 * sizeof(float));;
float* x116 = (float*)myMalloc(2500 * sizeof(float));;
float* x117 = (float*)myMalloc(1300 * sizeof(float));;
float* x118 = (float*)myMalloc(1300 * sizeof(float));;
float* x119 = (float*)myMalloc(50 * sizeof(float));;
float* x120 = (float*)myMalloc(2500 * sizeof(float));;
float* x121 = (float*)myMalloc(26 * sizeof(float));;
float* x122 = (float*)myMalloc(1300 * sizeof(float));;
float* x123 = (float*)myMalloc(2500 * sizeof(float));;
float* x124 = (float*)myMalloc(1300 * sizeof(float));;
float* x125 = (float*)myMalloc(50 * sizeof(float));;
double x126 = ((double)clock() / CLOCKS_PER_SEC);
double* x127 = (double*)myMalloc(51 * sizeof(double));;
int64_t x128 = (long)mallocAddr;
int32_t x129 = 0;
x129 -= 400;
double x131 = 70.0;
bool x595 = true || true;
bool x596 = x595 || true;
bool x1067 = true || false;
for(int x133=0; x133 < 5001; x133++) {
float* x158 = (float*)myMalloc(1 * sizeof(float));;
float* x159 = (float*)myMalloc(10400 * sizeof(float));;
float* x176 = (float*)myMalloc(10400 * sizeof(float));;
int* x143 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
function<void(int32_t,float**)> x612 = [&](int32_t x613,float** x614) {
float** x616 = x614;
float* x617 = x616[0];
float* x618 = x616[1];
float* x619 = x616[2];
float* x620 = x616[3];
float* x621 = x616[4];
float* x622 = x616[5];
int32_t x615 = x613;
bool x623 = x615 < 20;
if (x623) {
int32_t x624 = x615 * 520;
float* x625 = x159+x624;
float* x626 = x176+x624;
float* x627 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x625,26,x18,50,0,x627,50);
float* x629 = (float*)myMalloc(1000 * sizeof(float));;
float* x630 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x619,50,x29,50,0,x630,50);
float* x632 = (float*)myMalloc(1000 * sizeof(float));;
float* x633 = (float*)myMalloc(1000 * sizeof(float));;
for(int x634=0; x634 < 20; x634++) {
int32_t x636 = 50 * x634;
for(int x635=0; x635 < 50; x635++) {
int32_t x638 = x636 + x635;
float x639 = x627[x638];
float x640 = x630[x638];
int32_t x637 = x635 + x636;
float x641 = x639 + x640;
x633[x637] = x641;

}

}
float* x647 = (float*)myMalloc(1000 * sizeof(float));;
float* x648 = (float*)myMalloc(1000 * sizeof(float));;
for(int x649=0; x649 < 20; x649++) {
int32_t x651 = 50 * x649;
for(int x650=0; x650 < 50; x650++) {
int32_t x653 = x651 + x650;
float x654 = x633[x653];
float x655 = x39[x650];
int32_t x652 = x650 + x651;
float x656 = x654 + x655;
x648[x652] = x656;

}

}
float* x662 = (float*)myMalloc(1000 * sizeof(float));;
float* x663 = (float*)myMalloc(1000 * sizeof(float));;
for(int x664=0; x664 < 1000; x664++) {
float x665 = x648[x664];
float x666 = -1.0f * x665;
double x667 = (double)x666;
double x668 = exp(x667);
float x669 = (float)x668;
float x670 = x669 + 1.0f;
float x671 = 1.0f / x670;
x663[x664] = x671;

}
float* x675 = (float*)myMalloc(1000 * sizeof(float));;
float* x676 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x625,26,x41,50,0,x676,50);
float* x678 = (float*)myMalloc(1000 * sizeof(float));;
float* x679 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x619,50,x50,50,0,x679,50);
float* x681 = (float*)myMalloc(1000 * sizeof(float));;
float* x682 = (float*)myMalloc(1000 * sizeof(float));;
for(int x683=0; x683 < 20; x683++) {
int32_t x685 = 50 * x683;
for(int x684=0; x684 < 50; x684++) {
int32_t x687 = x685 + x684;
float x688 = x676[x687];
float x689 = x679[x687];
int32_t x686 = x684 + x685;
float x690 = x688 + x689;
x682[x686] = x690;

}

}
float* x696 = (float*)myMalloc(1000 * sizeof(float));;
float* x697 = (float*)myMalloc(1000 * sizeof(float));;
for(int x698=0; x698 < 20; x698++) {
int32_t x700 = 50 * x698;
for(int x699=0; x699 < 50; x699++) {
int32_t x702 = x700 + x699;
float x703 = x682[x702];
float x704 = x59[x699];
int32_t x701 = x699 + x700;
float x705 = x703 + x704;
x697[x701] = x705;

}

}
float* x711 = (float*)myMalloc(1000 * sizeof(float));;
float* x712 = (float*)myMalloc(1000 * sizeof(float));;
for(int x713=0; x713 < 1000; x713++) {
float x714 = x697[x713];
float x715 = -1.0f * x714;
double x716 = (double)x715;
double x717 = exp(x716);
float x718 = (float)x717;
float x719 = x718 + 1.0f;
float x720 = 1.0f / x719;
x712[x713] = x720;

}
float* x724 = (float*)myMalloc(1000 * sizeof(float));;
float* x725 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x625,26,x81,50,0,x725,50);
float* x727 = (float*)myMalloc(1000 * sizeof(float));;
float* x728 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x619,50,x90,50,0,x728,50);
float* x730 = (float*)myMalloc(1000 * sizeof(float));;
float* x731 = (float*)myMalloc(1000 * sizeof(float));;
for(int x732=0; x732 < 20; x732++) {
int32_t x734 = 50 * x732;
for(int x733=0; x733 < 50; x733++) {
int32_t x736 = x734 + x733;
float x737 = x725[x736];
float x738 = x728[x736];
int32_t x735 = x733 + x734;
float x739 = x737 + x738;
x731[x735] = x739;

}

}
float* x745 = (float*)myMalloc(1000 * sizeof(float));;
float* x746 = (float*)myMalloc(1000 * sizeof(float));;
for(int x747=0; x747 < 20; x747++) {
int32_t x749 = 50 * x747;
for(int x748=0; x748 < 50; x748++) {
int32_t x751 = x749 + x748;
float x752 = x731[x751];
float x753 = x99[x748];
int32_t x750 = x748 + x749;
float x754 = x752 + x753;
x746[x750] = x754;

}

}
float* x760 = (float*)myMalloc(1000 * sizeof(float));;
float* x761 = (float*)myMalloc(1000 * sizeof(float));;
for(int x762=0; x762 < 1000; x762++) {
float x763 = x746[x762];
float x764 = -1.0f * x763;
double x765 = (double)x764;
double x766 = exp(x765);
float x767 = (float)x766;
float x768 = x767 + 1.0f;
float x769 = 1.0f / x768;
x761[x762] = x769;

}
float* x773 = (float*)myMalloc(1000 * sizeof(float));;
float* x774 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x625,26,x61,50,0,x774,50);
float* x776 = (float*)myMalloc(1000 * sizeof(float));;
float* x777 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x619,50,x70,50,0,x777,50);
float* x779 = (float*)myMalloc(1000 * sizeof(float));;
float* x780 = (float*)myMalloc(1000 * sizeof(float));;
for(int x781=0; x781 < 20; x781++) {
int32_t x783 = 50 * x781;
for(int x782=0; x782 < 50; x782++) {
int32_t x785 = x783 + x782;
float x786 = x774[x785];
float x787 = x777[x785];
int32_t x784 = x782 + x783;
float x788 = x786 + x787;
x780[x784] = x788;

}

}
float* x794 = (float*)myMalloc(1000 * sizeof(float));;
float* x795 = (float*)myMalloc(1000 * sizeof(float));;
for(int x796=0; x796 < 20; x796++) {
int32_t x798 = 50 * x796;
for(int x797=0; x797 < 50; x797++) {
int32_t x800 = x798 + x797;
float x801 = x780[x800];
float x802 = x79[x797];
int32_t x799 = x797 + x798;
float x803 = x801 + x802;
x795[x799] = x803;

}

}
float* x809 = (float*)myMalloc(1000 * sizeof(float));;
float* x810 = (float*)myMalloc(1000 * sizeof(float));;
for(int x811=0; x811 < 1000; x811++) {
float x812 = x795[x811];
double x813 = (double)x812;
double x814 = tanh(x813);
float x815 = (float)x814;
x810[x811] = x815;

}
float* x819 = (float*)myMalloc(1000 * sizeof(float));;
float* x820 = (float*)myMalloc(1000 * sizeof(float));;
for(int x821=0; x821 < 20; x821++) {
int32_t x823 = 50 * x821;
for(int x822=0; x822 < 50; x822++) {
int32_t x825 = x823 + x822;
float x826 = x663[x825];
float x827 = x621[x825];
int32_t x824 = x822 + x823;
float x828 = x826 * x827;
x820[x824] = x828;

}

}
float* x834 = (float*)myMalloc(1000 * sizeof(float));;
float* x835 = (float*)myMalloc(1000 * sizeof(float));;
for(int x836=0; x836 < 20; x836++) {
int32_t x838 = 50 * x836;
for(int x837=0; x837 < 50; x837++) {
int32_t x840 = x838 + x837;
float x841 = x712[x840];
float x842 = x810[x840];
int32_t x839 = x837 + x838;
float x843 = x841 * x842;
x835[x839] = x843;

}

}
float* x849 = (float*)myMalloc(1000 * sizeof(float));;
float* x850 = (float*)myMalloc(1000 * sizeof(float));;
for(int x851=0; x851 < 20; x851++) {
int32_t x853 = 50 * x851;
for(int x852=0; x852 < 50; x852++) {
int32_t x855 = x853 + x852;
float x856 = x820[x855];
float x857 = x835[x855];
int32_t x854 = x852 + x853;
float x858 = x856 + x857;
x850[x854] = x858;

}

}
float* x864 = (float*)myMalloc(1000 * sizeof(float));;
float* x865 = (float*)myMalloc(1000 * sizeof(float));;
for(int x866=0; x866 < 1000; x866++) {
float x867 = x850[x866];
double x868 = (double)x867;
double x869 = tanh(x868);
float x870 = (float)x869;
x865[x866] = x870;

}
float* x874 = (float*)myMalloc(1000 * sizeof(float));;
float* x875 = (float*)myMalloc(1000 * sizeof(float));;
for(int x876=0; x876 < 20; x876++) {
int32_t x878 = 50 * x876;
for(int x877=0; x877 < 50; x877++) {
int32_t x880 = x878 + x877;
float x881 = x761[x880];
float x882 = x865[x880];
int32_t x879 = x877 + x878;
float x883 = x881 * x882;
x875[x879] = x883;

}

}
float* x889 = (float*)myMalloc(1000 * sizeof(float));;
float* x890 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x875,50,x101,26,0,x890,26);
float* x892 = (float*)myMalloc(520 * sizeof(float));;
for(int x893=0; x893 < 20; x893++) {
int32_t x895 = 26 * x893;
for(int x894=0; x894 < 26; x894++) {
int32_t x896 = x895 + x894;
float x897 = x890[x896];
float x898 = x110[x894];
float x899 = x897 + x898;
x890[x896] = x899;

}

}
int* x905 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x906=0; x906 < 20; x906++) {
int32_t x907 = x906 * 20;
int32_t x908 = x615 + x907;
int32_t x909 = x143[x908];
x905[x906] = x909;

}
float* x913 = (float*)myMalloc(20 * sizeof(float));;
int32_t x914 = 0;
for(int x915=0; x915 < 20; x915++) {
float x916 = -3.4028235E38f;
for(int x917=0; x917 < 26; x917++) {
int32_t x918 = x914;
float x919 = x890[x918];
float x920 = x916;
bool x921 = x919 > x920;
if (x921) {
float x922 = x890[x918];
x916 = x922;
} else {
}
x914 += 1;

}
float x929 = x916;
x913[x915] = x929;

}
float* x933 = (float*)myMalloc(520 * sizeof(float));;
int32_t x934 = 0;
for(int x935=0; x935 < 20; x935++) {
for(int x936=0; x936 < 26; x936++) {
int32_t x937 = x934;
float x938 = x890[x937];
float x939 = x913[x935];
float x940 = x938 - x939;
double x941 = (double)x940;
double x942 = exp(x941);
float x943 = (float)x942;
x933[x937] = x943;
x934 += 1;

}

}
float* x950 = (float*)myMalloc(20 * sizeof(float));;
for(int x951=0; x951 < 20; x951++) {
int32_t x952 = x951;
int32_t x953 = x951 * 26;
int32_t x954 = x953;
for(int x955=0; x955 < 26; x955++) {
for(int x956=0; x956 < 1; x956++) {
int32_t x957 = x952;
int32_t x958 = x957 + x956;
float x959 = x950[x958];
int32_t x960 = x954;
int32_t x961 = x960 + x956;
float x962 = x933[x961];
float x963 = x959 + x962;
x950[x958] = x963;

}
x954 += 1;

}

}
x934 = 0;
for(int x973=0; x973 < 20; x973++) {
float x974 = x913[x973];
float x975 = x950[x973];
double x976 = (double)x975;
double x977 = log(x976);
float x978 = (float)x977;
float x979 = x974 + x978;
for(int x980=0; x980 < 26; x980++) {
int32_t x981 = x934;
float x982 = x890[x981];
float x983 = x982 - x979;
x933[x981] = x983;
x934 += 1;

}

}
float* x990 = (float*)myMalloc(520 * sizeof(float));;
// nllLoss forward in CPU
float* x992 = (float*)myMalloc(20 * sizeof(float));;
int32_t x993 = 0;
for(int x994=0; x994 < 20; x994++) {
int32_t x995 = x993;
int32_t x996 = x905[x994];
int32_t x997 = x995 + x996;
float x998 = x933[x997];
float x999 = -1.0f * x998;
x992[x994] = x999;
x993 += 26;

}
float* x1004 = (float*)myMalloc(20 * sizeof(float));;
float x1005 = 0.0f;
for(int x1006=0; x1006 < 20; x1006++) {
float x1007 = x1005;
float x1008 = x992[x1006];
float x1009 = x1007 + x1008;
x1005 = x1009;

}
float x1013 = x1005;
float* x1014 = (float*)myMalloc(1 * sizeof(float));;
for(int x1015=0; x1015 < 1; x1015++) {
x1014[x1015] = x1013;

}
float* x1019 = (float*)myMalloc(1 * sizeof(float));;
if (x596) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x1024 = (float*)myMalloc(1 * sizeof(float));;
for(int x1025=0; x1025 < 1; x1025++) {
float x1026 = x617[0];
float x1027 = x1014[0];
float x1028 = x1026 + x1027;
x1024[x1025] = x1028;

}
float* x1032 = (float*)myMalloc(1 * sizeof(float));;
float** x1034 = (float**)myMalloc(6 * sizeof(float*));;
x1034[0] = x1024;
x1034[1] = x1032;
x1034[2] = x875;
x1034[3] = x889;
x1034[4] = x850;
x1034[5] = x864;
int32_t x1082 = 0;
float* x1095 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1117 = 0;
int32_t x1033 = x615 + 1;
x612(x1033,x1034);
// back prop for + op
if (x596) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1048=0; x1048 < 1; x1048++) {
float x1049 = x618[0];
float x1050 = x1032[0];
float x1051 = x1049 + x1050;
x618[0] = x1051;

}
if (x596) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1059=0; x1059 < 1; x1059++) {
float x1060 = x1019[0];
float x1061 = x1032[0];
float x1062 = x1060 + x1061;
x1019[0] = x1062;

}
// 'sum' gradient.
if (x1067) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",20,1);
assert(false && "");
}
for(int x1073=0; x1073 < 20; x1073++) {
float x1074 = x1004[x1073];
float x1075 = x1019[0];
float x1076 = x1074 + x1075;
x1004[x1073] = x1076;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
for(int x1083=0; x1083 < 20; x1083++) {
int32_t x1084 = x1082;
int32_t x1085 = x905[x1083];
int32_t x1086 = x1084 + x1085;
float x1087 = x990[x1086];
float x1088 = x1004[x1083];
float x1089 = -1.0f * x1088;
float x1090 = x1087 + x1089;
x990[x1086] = x1090;
x1082 += 26;

}
for(int x1096=0; x1096 < 20; x1096++) {
int32_t x1097 = x1096;
int32_t x1098 = x1096 * 26;
int32_t x1099 = x1098;
for(int x1100=0; x1100 < 26; x1100++) {
for(int x1101=0; x1101 < 1; x1101++) {
int32_t x1102 = x1097;
int32_t x1103 = x1102 + x1101;
float x1104 = x1095[x1103];
int32_t x1105 = x1099;
int32_t x1106 = x1105 + x1101;
float x1107 = x990[x1106];
float x1108 = x1104 + x1107;
x1095[x1103] = x1108;

}
x1099 += 1;

}

}
for(int x1118=0; x1118 < 20; x1118++) {
for(int x1119=0; x1119 < 26; x1119++) {
int32_t x1120 = x1117;
float x1121 = x892[x1120];
float x1122 = x990[x1120];
float x1123 = x933[x1120];
float x1127 = x1095[x1118];
double x1124 = (double)x1123;
double x1125 = exp(x1124);
float x1126 = (float)x1125;
float x1128 = x1126 * x1127;
float x1129 = x1122 - x1128;
float x1130 = x1121 + x1129;
x892[x1120] = x1130;
x1117 += 1;

}

}
for(int x1137=0; x1137 < 20; x1137++) {
int32_t x1139 = 26 * x1137;
for(int x1138=0; x1138 < 26; x1138++) {
float x1141 = x111[x1138];
int32_t x1140 = x1139 + x1138;
float x1142 = x892[x1140];
float x1143 = x1141 + x1142;
x111[x1138] = x1143;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x892,26,x101,26,1,x889,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x875,50,x892,26,1,x109,26);
// backprop for * op
for(int x1153=0; x1153 < 20; x1153++) {
int32_t x1155 = 50 * x1153;
for(int x1154=0; x1154 < 50; x1154++) {
int32_t x1156 = x1155 + x1154;
float x1157 = x773[x1156];
float x1158 = x761[x1156];
float x1159 = x865[x1156];
float x1160 = x889[x1156];
float x1161 = x1160 * x1159;
float x1162 = x1157 + x1161;
x773[x1156] = x1162;
float x1164 = x874[x1156];
float x1165 = x761[x1156];
float x1166 = x865[x1156];
float x1167 = x889[x1156];
float x1168 = x1167 * x1165;
float x1169 = x1164 + x1168;
x874[x1156] = x1169;

}

}
for(int x1175=0; x1175 < 1000; x1175++) {
float x1176 = x864[x1175];
float x1177 = x865[x1175];
float x1180 = x874[x1175];
float x1178 = x1177 * x1177;
float x1179 = 1.0f - x1178;
float x1181 = x1179 * x1180;
float x1182 = x1176 + x1181;
x864[x1175] = x1182;

}
// back prop for + op
for(int x1187=0; x1187 < 20; x1187++) {
int32_t x1189 = 50 * x1187;
for(int x1188=0; x1188 < 50; x1188++) {
int32_t x1190 = x1189 + x1188;
float x1191 = x834[x1190];
float x1192 = x864[x1190];
float x1193 = x1191 + x1192;
x834[x1190] = x1193;

}

}
for(int x1199=0; x1199 < 20; x1199++) {
int32_t x1201 = 50 * x1199;
for(int x1200=0; x1200 < 50; x1200++) {
int32_t x1202 = x1201 + x1200;
float x1203 = x849[x1202];
float x1204 = x864[x1202];
float x1205 = x1203 + x1204;
x849[x1202] = x1205;

}

}
// backprop for * op
for(int x1212=0; x1212 < 20; x1212++) {
int32_t x1214 = 50 * x1212;
for(int x1213=0; x1213 < 50; x1213++) {
int32_t x1215 = x1214 + x1213;
float x1216 = x724[x1215];
float x1217 = x712[x1215];
float x1218 = x810[x1215];
float x1219 = x849[x1215];
float x1220 = x1219 * x1218;
float x1221 = x1216 + x1220;
x724[x1215] = x1221;
float x1223 = x819[x1215];
float x1224 = x712[x1215];
float x1225 = x810[x1215];
float x1226 = x849[x1215];
float x1227 = x1226 * x1224;
float x1228 = x1223 + x1227;
x819[x1215] = x1228;

}

}
// backprop for * op
for(int x1235=0; x1235 < 20; x1235++) {
int32_t x1237 = 50 * x1235;
for(int x1236=0; x1236 < 50; x1236++) {
int32_t x1238 = x1237 + x1236;
float x1239 = x675[x1238];
float x1240 = x663[x1238];
float x1241 = x621[x1238];
float x1242 = x834[x1238];
float x1243 = x1242 * x1241;
float x1244 = x1239 + x1243;
x675[x1238] = x1244;
float x1246 = x622[x1238];
float x1247 = x663[x1238];
float x1248 = x621[x1238];
float x1249 = x834[x1238];
float x1250 = x1249 * x1247;
float x1251 = x1246 + x1250;
x622[x1238] = x1251;

}

}
for(int x1257=0; x1257 < 1000; x1257++) {
float x1258 = x809[x1257];
float x1259 = x810[x1257];
float x1262 = x819[x1257];
float x1260 = x1259 * x1259;
float x1261 = 1.0f - x1260;
float x1263 = x1261 * x1262;
float x1264 = x1258 + x1263;
x809[x1257] = x1264;

}
// back prop for + op
for(int x1269=0; x1269 < 20; x1269++) {
int32_t x1271 = 50 * x1269;
for(int x1270=0; x1270 < 50; x1270++) {
int32_t x1272 = x1271 + x1270;
float x1273 = x794[x1272];
float x1274 = x809[x1272];
float x1275 = x1273 + x1274;
x794[x1272] = x1275;

}

}
for(int x1281=0; x1281 < 20; x1281++) {
int32_t x1283 = 50 * x1281;
for(int x1282=0; x1282 < 50; x1282++) {
float x1285 = x80[x1282];
int32_t x1284 = x1283 + x1282;
float x1286 = x809[x1284];
float x1287 = x1285 + x1286;
x80[x1282] = x1287;

}

}
// back prop for + op
for(int x1294=0; x1294 < 20; x1294++) {
int32_t x1296 = 50 * x1294;
for(int x1295=0; x1295 < 50; x1295++) {
int32_t x1297 = x1296 + x1295;
float x1298 = x776[x1297];
float x1299 = x794[x1297];
float x1300 = x1298 + x1299;
x776[x1297] = x1300;

}

}
for(int x1306=0; x1306 < 20; x1306++) {
int32_t x1308 = 50 * x1306;
for(int x1307=0; x1307 < 50; x1307++) {
int32_t x1309 = x1308 + x1307;
float x1310 = x779[x1309];
float x1311 = x794[x1309];
float x1312 = x1310 + x1311;
x779[x1309] = x1312;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x779,50,x70,50,1,x620,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x619,50,x779,50,1,x78,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x776,50,x61,50,1,x626,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x625,26,x776,50,1,x69,50);
for(int x1324=0; x1324 < 1000; x1324++) {
float x1325 = x760[x1324];
float x1326 = x761[x1324];
float x1329 = x773[x1324];
float x1327 = 1.0f - x1326;
float x1328 = x1327 * x1326;
float x1330 = x1328 * x1329;
float x1331 = x1325 + x1330;
x760[x1324] = x1331;

}
// back prop for + op
for(int x1336=0; x1336 < 20; x1336++) {
int32_t x1338 = 50 * x1336;
for(int x1337=0; x1337 < 50; x1337++) {
int32_t x1339 = x1338 + x1337;
float x1340 = x745[x1339];
float x1341 = x760[x1339];
float x1342 = x1340 + x1341;
x745[x1339] = x1342;

}

}
for(int x1348=0; x1348 < 20; x1348++) {
int32_t x1350 = 50 * x1348;
for(int x1349=0; x1349 < 50; x1349++) {
float x1352 = x100[x1349];
int32_t x1351 = x1350 + x1349;
float x1353 = x760[x1351];
float x1354 = x1352 + x1353;
x100[x1349] = x1354;

}

}
// back prop for + op
for(int x1361=0; x1361 < 20; x1361++) {
int32_t x1363 = 50 * x1361;
for(int x1362=0; x1362 < 50; x1362++) {
int32_t x1364 = x1363 + x1362;
float x1365 = x727[x1364];
float x1366 = x745[x1364];
float x1367 = x1365 + x1366;
x727[x1364] = x1367;

}

}
for(int x1373=0; x1373 < 20; x1373++) {
int32_t x1375 = 50 * x1373;
for(int x1374=0; x1374 < 50; x1374++) {
int32_t x1376 = x1375 + x1374;
float x1377 = x730[x1376];
float x1378 = x745[x1376];
float x1379 = x1377 + x1378;
x730[x1376] = x1379;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x730,50,x90,50,1,x620,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x619,50,x730,50,1,x98,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x727,50,x81,50,1,x626,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x625,26,x727,50,1,x89,50);
for(int x1391=0; x1391 < 1000; x1391++) {
float x1392 = x711[x1391];
float x1393 = x712[x1391];
float x1396 = x724[x1391];
float x1394 = 1.0f - x1393;
float x1395 = x1394 * x1393;
float x1397 = x1395 * x1396;
float x1398 = x1392 + x1397;
x711[x1391] = x1398;

}
// back prop for + op
for(int x1403=0; x1403 < 20; x1403++) {
int32_t x1405 = 50 * x1403;
for(int x1404=0; x1404 < 50; x1404++) {
int32_t x1406 = x1405 + x1404;
float x1407 = x696[x1406];
float x1408 = x711[x1406];
float x1409 = x1407 + x1408;
x696[x1406] = x1409;

}

}
for(int x1415=0; x1415 < 20; x1415++) {
int32_t x1417 = 50 * x1415;
for(int x1416=0; x1416 < 50; x1416++) {
float x1419 = x60[x1416];
int32_t x1418 = x1417 + x1416;
float x1420 = x711[x1418];
float x1421 = x1419 + x1420;
x60[x1416] = x1421;

}

}
// back prop for + op
for(int x1428=0; x1428 < 20; x1428++) {
int32_t x1430 = 50 * x1428;
for(int x1429=0; x1429 < 50; x1429++) {
int32_t x1431 = x1430 + x1429;
float x1432 = x678[x1431];
float x1433 = x696[x1431];
float x1434 = x1432 + x1433;
x678[x1431] = x1434;

}

}
for(int x1440=0; x1440 < 20; x1440++) {
int32_t x1442 = 50 * x1440;
for(int x1441=0; x1441 < 50; x1441++) {
int32_t x1443 = x1442 + x1441;
float x1444 = x681[x1443];
float x1445 = x696[x1443];
float x1446 = x1444 + x1445;
x681[x1443] = x1446;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x681,50,x50,50,1,x620,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x619,50,x681,50,1,x58,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x678,50,x41,50,1,x626,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x625,26,x678,50,1,x49,50);
for(int x1458=0; x1458 < 1000; x1458++) {
float x1459 = x662[x1458];
float x1460 = x663[x1458];
float x1463 = x675[x1458];
float x1461 = 1.0f - x1460;
float x1462 = x1461 * x1460;
float x1464 = x1462 * x1463;
float x1465 = x1459 + x1464;
x662[x1458] = x1465;

}
// back prop for + op
for(int x1470=0; x1470 < 20; x1470++) {
int32_t x1472 = 50 * x1470;
for(int x1471=0; x1471 < 50; x1471++) {
int32_t x1473 = x1472 + x1471;
float x1474 = x647[x1473];
float x1475 = x662[x1473];
float x1476 = x1474 + x1475;
x647[x1473] = x1476;

}

}
for(int x1482=0; x1482 < 20; x1482++) {
int32_t x1484 = 50 * x1482;
for(int x1483=0; x1483 < 50; x1483++) {
float x1486 = x40[x1483];
int32_t x1485 = x1484 + x1483;
float x1487 = x662[x1485];
float x1488 = x1486 + x1487;
x40[x1483] = x1488;

}

}
// back prop for + op
for(int x1495=0; x1495 < 20; x1495++) {
int32_t x1497 = 50 * x1495;
for(int x1496=0; x1496 < 50; x1496++) {
int32_t x1498 = x1497 + x1496;
float x1499 = x629[x1498];
float x1500 = x647[x1498];
float x1501 = x1499 + x1500;
x629[x1498] = x1501;

}

}
for(int x1507=0; x1507 < 20; x1507++) {
int32_t x1509 = 50 * x1507;
for(int x1508=0; x1508 < 50; x1508++) {
int32_t x1510 = x1509 + x1508;
float x1511 = x632[x1510];
float x1512 = x647[x1510];
float x1513 = x1511 + x1512;
x632[x1510] = x1513;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x632,50,x29,50,1,x620,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x619,50,x632,50,1,x38,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x629,50,x18,50,1,x626,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x625,26,x629,50,1,x28,50);
} else {
float x1526 = 0.0f;
for(int x1527=0; x1527 < 1; x1527++) {
float x1528 = x1526;
float x1529 = x617[x1527];
float x1530 = x1528 + x1529;
x1526 = x1530;

}
float x1534 = x1526;
float* x1535 = (float*)myMalloc(1 * sizeof(float));;
for(int x1536=0; x1536 < 1; x1536++) {
x1535[x1536] = x1534;

}
float* x1540 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x1542=0; x1542 < 1; x1542++) {
x1540[x1542] = 1.0f;

}
// backend is lantern.TensorDsl$BackendCPU@51182e06
for(int x1547=0; x1547 < 1; x1547++) {
float x1548 = x1535[x1547];
x158[x1547] = x1548;

}
// 'sum' gradient.
if (x596) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1557=0; x1557 < 1; x1557++) {
float x1558 = x618[0];
float x1559 = x1540[0];
float x1560 = x1558 + x1559;
x618[0] = x1560;

}
}
};
x129 += 400;
int32_t x135 = x129;
int32_t x136 = x135 + 400;
int32_t x137 = x136 + 1;
bool x138 = x137 >= x5;
if (x138) {
x129 = 0;
} else {
}
int* x142 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
for(int x145=0; x145 < 400; x145++) {
int32_t x146 = x129;
int32_t x147 = x146 + x145;
int32_t x148 = x9[x147];
x142[x145] = x148;
int32_t x150 = x147 + 1;
int32_t x151 = x9[x150];
x143[x145] = x151;

}
float* x155 = (float*)myMalloc(1 * sizeof(float));;
float* x156 = (float*)myMalloc(1 * sizeof(float));;
// allocate memory to save the final loss in CPU Tensor
for(int x161=0; x161 < 20; x161++) {
int32_t x163 = x161 * 26;
int32_t x164 = x163 * 20;
for(int x162=0; x162 < 20; x162++) {
int32_t x167 = x162 * 20;
int32_t x168 = x167 + x161;
int32_t x169 = x142[x168];
int32_t x165 = x162 * 26;
int32_t x166 = x164 + x165;
int32_t x170 = x166 + x169;
x159[x170] = 1.0f;

}

}
float* x177 = (float*)myMalloc(1 * sizeof(float));;
float* x178 = (float*)myMalloc(1 * sizeof(float));;
float* x179 = (float*)myMalloc(1000 * sizeof(float));;
float* x180 = (float*)myMalloc(1000 * sizeof(float));;
float* x181 = (float*)myMalloc(1000 * sizeof(float));;
float* x182 = (float*)myMalloc(1000 * sizeof(float));;
float** x2098 = (float**)myMalloc(6 * sizeof(float*));;
x2098[0] = x177;
x2098[1] = x178;
x2098[2] = x179;
x2098[3] = x180;
x2098[4] = x181;
x2098[5] = x182;
function<void(int32_t,float**)> x183 = [&](int32_t x184,float** x185) {
float** x187 = x185;
float* x188 = x187[0];
float* x189 = x187[1];
float* x190 = x187[2];
float* x191 = x187[3];
float* x192 = x187[4];
float* x193 = x187[5];
int32_t x186 = x184;
bool x194 = x186 < 20;
if (x194) {
int32_t x195 = x186 * 520;
float* x196 = x159+x195;
float* x197 = x176+x195;
float* x198 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x196,26,x18,50,0,x198,50);
float* x200 = (float*)myMalloc(1000 * sizeof(float));;
float* x201 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x190,50,x29,50,0,x201,50);
float* x203 = (float*)myMalloc(1000 * sizeof(float));;
float* x204 = (float*)myMalloc(1000 * sizeof(float));;
for(int x205=0; x205 < 20; x205++) {
int32_t x208 = 50 * x205;
for(int x207=0; x207 < 50; x207++) {
int32_t x210 = x208 + x207;
float x211 = x198[x210];
float x212 = x201[x210];
int32_t x209 = x207 + x208;
float x213 = x211 + x212;
x204[x209] = x213;

}

}
float* x219 = (float*)myMalloc(1000 * sizeof(float));;
float* x220 = (float*)myMalloc(1000 * sizeof(float));;
for(int x221=0; x221 < 20; x221++) {
int32_t x223 = 50 * x221;
for(int x222=0; x222 < 50; x222++) {
int32_t x225 = x223 + x222;
float x226 = x204[x225];
float x227 = x39[x222];
int32_t x224 = x222 + x223;
float x228 = x226 + x227;
x220[x224] = x228;

}

}
float* x234 = (float*)myMalloc(1000 * sizeof(float));;
float* x235 = (float*)myMalloc(1000 * sizeof(float));;
for(int x237=0; x237 < 1000; x237++) {
float x238 = x220[x237];
float x239 = -1.0f * x238;
double x240 = (double)x239;
double x241 = exp(x240);
float x242 = (float)x241;
float x243 = x242 + 1.0f;
float x244 = 1.0f / x243;
x235[x237] = x244;

}
float* x248 = (float*)myMalloc(1000 * sizeof(float));;
float* x249 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x196,26,x41,50,0,x249,50);
float* x251 = (float*)myMalloc(1000 * sizeof(float));;
float* x252 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x190,50,x50,50,0,x252,50);
float* x254 = (float*)myMalloc(1000 * sizeof(float));;
float* x255 = (float*)myMalloc(1000 * sizeof(float));;
for(int x256=0; x256 < 20; x256++) {
int32_t x258 = 50 * x256;
for(int x257=0; x257 < 50; x257++) {
int32_t x260 = x258 + x257;
float x261 = x249[x260];
float x262 = x252[x260];
int32_t x259 = x257 + x258;
float x263 = x261 + x262;
x255[x259] = x263;

}

}
float* x269 = (float*)myMalloc(1000 * sizeof(float));;
float* x270 = (float*)myMalloc(1000 * sizeof(float));;
for(int x271=0; x271 < 20; x271++) {
int32_t x273 = 50 * x271;
for(int x272=0; x272 < 50; x272++) {
int32_t x275 = x273 + x272;
float x276 = x255[x275];
float x277 = x59[x272];
int32_t x274 = x272 + x273;
float x278 = x276 + x277;
x270[x274] = x278;

}

}
float* x284 = (float*)myMalloc(1000 * sizeof(float));;
float* x285 = (float*)myMalloc(1000 * sizeof(float));;
for(int x286=0; x286 < 1000; x286++) {
float x287 = x270[x286];
float x288 = -1.0f * x287;
double x289 = (double)x288;
double x290 = exp(x289);
float x291 = (float)x290;
float x292 = x291 + 1.0f;
float x293 = 1.0f / x292;
x285[x286] = x293;

}
float* x297 = (float*)myMalloc(1000 * sizeof(float));;
float* x298 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x196,26,x81,50,0,x298,50);
float* x300 = (float*)myMalloc(1000 * sizeof(float));;
float* x301 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x190,50,x90,50,0,x301,50);
float* x303 = (float*)myMalloc(1000 * sizeof(float));;
float* x304 = (float*)myMalloc(1000 * sizeof(float));;
for(int x305=0; x305 < 20; x305++) {
int32_t x307 = 50 * x305;
for(int x306=0; x306 < 50; x306++) {
int32_t x309 = x307 + x306;
float x310 = x298[x309];
float x311 = x301[x309];
int32_t x308 = x306 + x307;
float x312 = x310 + x311;
x304[x308] = x312;

}

}
float* x318 = (float*)myMalloc(1000 * sizeof(float));;
float* x319 = (float*)myMalloc(1000 * sizeof(float));;
for(int x320=0; x320 < 20; x320++) {
int32_t x322 = 50 * x320;
for(int x321=0; x321 < 50; x321++) {
int32_t x324 = x322 + x321;
float x325 = x304[x324];
float x326 = x99[x321];
int32_t x323 = x321 + x322;
float x327 = x325 + x326;
x319[x323] = x327;

}

}
float* x333 = (float*)myMalloc(1000 * sizeof(float));;
float* x334 = (float*)myMalloc(1000 * sizeof(float));;
for(int x335=0; x335 < 1000; x335++) {
float x336 = x319[x335];
float x337 = -1.0f * x336;
double x338 = (double)x337;
double x339 = exp(x338);
float x340 = (float)x339;
float x341 = x340 + 1.0f;
float x342 = 1.0f / x341;
x334[x335] = x342;

}
float* x346 = (float*)myMalloc(1000 * sizeof(float));;
float* x347 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x196,26,x61,50,0,x347,50);
float* x349 = (float*)myMalloc(1000 * sizeof(float));;
float* x350 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x190,50,x70,50,0,x350,50);
float* x352 = (float*)myMalloc(1000 * sizeof(float));;
float* x353 = (float*)myMalloc(1000 * sizeof(float));;
for(int x354=0; x354 < 20; x354++) {
int32_t x356 = 50 * x354;
for(int x355=0; x355 < 50; x355++) {
int32_t x358 = x356 + x355;
float x359 = x347[x358];
float x360 = x350[x358];
int32_t x357 = x355 + x356;
float x361 = x359 + x360;
x353[x357] = x361;

}

}
float* x367 = (float*)myMalloc(1000 * sizeof(float));;
float* x368 = (float*)myMalloc(1000 * sizeof(float));;
for(int x369=0; x369 < 20; x369++) {
int32_t x371 = 50 * x369;
for(int x370=0; x370 < 50; x370++) {
int32_t x373 = x371 + x370;
float x374 = x353[x373];
float x375 = x79[x370];
int32_t x372 = x370 + x371;
float x376 = x374 + x375;
x368[x372] = x376;

}

}
float* x382 = (float*)myMalloc(1000 * sizeof(float));;
float* x383 = (float*)myMalloc(1000 * sizeof(float));;
for(int x384=0; x384 < 1000; x384++) {
float x385 = x368[x384];
double x386 = (double)x385;
double x387 = tanh(x386);
float x388 = (float)x387;
x383[x384] = x388;

}
float* x392 = (float*)myMalloc(1000 * sizeof(float));;
float* x393 = (float*)myMalloc(1000 * sizeof(float));;
for(int x394=0; x394 < 20; x394++) {
int32_t x396 = 50 * x394;
for(int x395=0; x395 < 50; x395++) {
int32_t x398 = x396 + x395;
float x399 = x235[x398];
float x400 = x192[x398];
int32_t x397 = x395 + x396;
float x401 = x399 * x400;
x393[x397] = x401;

}

}
float* x407 = (float*)myMalloc(1000 * sizeof(float));;
float* x408 = (float*)myMalloc(1000 * sizeof(float));;
for(int x409=0; x409 < 20; x409++) {
int32_t x411 = 50 * x409;
for(int x410=0; x410 < 50; x410++) {
int32_t x413 = x411 + x410;
float x414 = x285[x413];
float x415 = x383[x413];
int32_t x412 = x410 + x411;
float x416 = x414 * x415;
x408[x412] = x416;

}

}
float* x422 = (float*)myMalloc(1000 * sizeof(float));;
float* x423 = (float*)myMalloc(1000 * sizeof(float));;
for(int x424=0; x424 < 20; x424++) {
int32_t x426 = 50 * x424;
for(int x425=0; x425 < 50; x425++) {
int32_t x428 = x426 + x425;
float x429 = x393[x428];
float x430 = x408[x428];
int32_t x427 = x425 + x426;
float x431 = x429 + x430;
x423[x427] = x431;

}

}
float* x437 = (float*)myMalloc(1000 * sizeof(float));;
float* x438 = (float*)myMalloc(1000 * sizeof(float));;
for(int x439=0; x439 < 1000; x439++) {
float x440 = x423[x439];
double x441 = (double)x440;
double x442 = tanh(x441);
float x443 = (float)x442;
x438[x439] = x443;

}
float* x447 = (float*)myMalloc(1000 * sizeof(float));;
float* x448 = (float*)myMalloc(1000 * sizeof(float));;
for(int x449=0; x449 < 20; x449++) {
int32_t x451 = 50 * x449;
for(int x450=0; x450 < 50; x450++) {
int32_t x453 = x451 + x450;
float x454 = x334[x453];
float x455 = x438[x453];
int32_t x452 = x450 + x451;
float x456 = x454 * x455;
x448[x452] = x456;

}

}
float* x462 = (float*)myMalloc(1000 * sizeof(float));;
float* x463 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x448,50,x101,26,0,x463,26);
float* x465 = (float*)myMalloc(520 * sizeof(float));;
for(int x466=0; x466 < 20; x466++) {
int32_t x469 = 26 * x466;
for(int x468=0; x468 < 26; x468++) {
int32_t x470 = x469 + x468;
float x471 = x463[x470];
float x472 = x110[x468];
float x473 = x471 + x472;
x463[x470] = x473;

}

}
int* x479 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x480=0; x480 < 20; x480++) {
int32_t x481 = x480 * 20;
int32_t x482 = x186 + x481;
int32_t x483 = x143[x482];
x479[x480] = x483;

}
float* x487 = (float*)myMalloc(20 * sizeof(float));;
int32_t x488 = 0;
for(int x489=0; x489 < 20; x489++) {
float x490 = -3.4028235E38f;
for(int x491=0; x491 < 26; x491++) {
int32_t x492 = x488;
float x493 = x463[x492];
float x494 = x490;
bool x495 = x493 > x494;
if (x495) {
float x496 = x463[x492];
x490 = x496;
} else {
}
x488 += 1;

}
float x503 = x490;
x487[x489] = x503;

}
float* x507 = (float*)myMalloc(520 * sizeof(float));;
int32_t x508 = 0;
for(int x509=0; x509 < 20; x509++) {
for(int x510=0; x510 < 26; x510++) {
int32_t x511 = x508;
float x512 = x463[x511];
float x513 = x487[x509];
float x514 = x512 - x513;
double x515 = (double)x514;
double x516 = exp(x515);
float x517 = (float)x516;
x507[x511] = x517;
x508 += 1;

}

}
float* x524 = (float*)myMalloc(20 * sizeof(float));;
for(int x525=0; x525 < 20; x525++) {
int32_t x526 = x525;
int32_t x527 = x525 * 26;
int32_t x528 = x527;
for(int x529=0; x529 < 26; x529++) {
for(int x531=0; x531 < 1; x531++) {
int32_t x532 = x526;
int32_t x533 = x532 + x531;
float x534 = x524[x533];
int32_t x535 = x528;
int32_t x536 = x535 + x531;
float x537 = x507[x536];
float x538 = x534 + x537;
x524[x533] = x538;

}
x528 += 1;

}

}
x508 = 0;
for(int x548=0; x548 < 20; x548++) {
float x549 = x487[x548];
float x550 = x524[x548];
double x551 = (double)x550;
double x552 = log(x551);
float x553 = (float)x552;
float x554 = x549 + x553;
for(int x555=0; x555 < 26; x555++) {
int32_t x556 = x508;
float x557 = x463[x556];
float x558 = x557 - x554;
x507[x556] = x558;
x508 += 1;

}

}
float* x565 = (float*)myMalloc(520 * sizeof(float));;
// nllLoss forward in CPU
float* x567 = (float*)myMalloc(20 * sizeof(float));;
int32_t x568 = 0;
for(int x569=0; x569 < 20; x569++) {
int32_t x570 = x568;
int32_t x571 = x479[x569];
int32_t x572 = x570 + x571;
float x573 = x507[x572];
float x574 = -1.0f * x573;
x567[x569] = x574;
x568 += 26;

}
float* x579 = (float*)myMalloc(20 * sizeof(float));;
float x580 = 0.0f;
for(int x581=0; x581 < 20; x581++) {
float x582 = x580;
float x583 = x567[x581];
float x584 = x582 + x583;
x580 = x584;

}
float x588 = x580;
float* x589 = (float*)myMalloc(1 * sizeof(float));;
for(int x590=0; x590 < 1; x590++) {
x589[x590] = x588;

}
float* x594 = (float*)myMalloc(1 * sizeof(float));;
if (x596) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x602 = (float*)myMalloc(1 * sizeof(float));;
for(int x603=0; x603 < 1; x603++) {
float x604 = x188[0];
float x605 = x589[0];
float x606 = x604 + x605;
x602[x603] = x606;

}
float* x610 = (float*)myMalloc(1 * sizeof(float));;
float** x1567 = (float**)myMalloc(6 * sizeof(float*));;
x1567[0] = x602;
x1567[1] = x610;
x1567[2] = x448;
x1567[3] = x462;
x1567[4] = x423;
x1567[5] = x437;
int32_t x611 = x186 + 1;
x612(x611,x1567);
// back prop for + op
if (x596) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1581=0; x1581 < 1; x1581++) {
float x1582 = x189[0];
float x1583 = x610[0];
float x1584 = x1582 + x1583;
x189[0] = x1584;

}
if (x596) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1592=0; x1592 < 1; x1592++) {
float x1593 = x594[0];
float x1594 = x610[0];
float x1595 = x1593 + x1594;
x594[0] = x1595;

}
// 'sum' gradient.
if (x1067) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",20,1);
assert(false && "");
}
for(int x1604=0; x1604 < 20; x1604++) {
float x1605 = x579[x1604];
float x1606 = x594[0];
float x1607 = x1605 + x1606;
x579[x1604] = x1607;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x1613 = 0;
for(int x1614=0; x1614 < 20; x1614++) {
int32_t x1615 = x1613;
int32_t x1616 = x479[x1614];
int32_t x1617 = x1615 + x1616;
float x1618 = x565[x1617];
float x1619 = x579[x1614];
float x1620 = -1.0f * x1619;
float x1621 = x1618 + x1620;
x565[x1617] = x1621;
x1613 += 26;

}
float* x1626 = (float*)myMalloc(20 * sizeof(float));;
for(int x1627=0; x1627 < 20; x1627++) {
int32_t x1628 = x1627;
int32_t x1629 = x1627 * 26;
int32_t x1630 = x1629;
for(int x1631=0; x1631 < 26; x1631++) {
for(int x1632=0; x1632 < 1; x1632++) {
int32_t x1633 = x1628;
int32_t x1634 = x1633 + x1632;
float x1635 = x1626[x1634];
int32_t x1636 = x1630;
int32_t x1637 = x1636 + x1632;
float x1638 = x565[x1637];
float x1639 = x1635 + x1638;
x1626[x1634] = x1639;

}
x1630 += 1;

}

}
int32_t x1648 = 0;
for(int x1649=0; x1649 < 20; x1649++) {
for(int x1650=0; x1650 < 26; x1650++) {
int32_t x1651 = x1648;
float x1652 = x465[x1651];
float x1653 = x565[x1651];
float x1654 = x507[x1651];
float x1658 = x1626[x1649];
double x1655 = (double)x1654;
double x1656 = exp(x1655);
float x1657 = (float)x1656;
float x1659 = x1657 * x1658;
float x1660 = x1653 - x1659;
float x1661 = x1652 + x1660;
x465[x1651] = x1661;
x1648 += 1;

}

}
for(int x1668=0; x1668 < 20; x1668++) {
int32_t x1670 = 26 * x1668;
for(int x1669=0; x1669 < 26; x1669++) {
float x1672 = x111[x1669];
int32_t x1671 = x1670 + x1669;
float x1673 = x465[x1671];
float x1674 = x1672 + x1673;
x111[x1669] = x1674;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x465,26,x101,26,1,x462,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x448,50,x465,26,1,x109,26);
// backprop for * op
for(int x1684=0; x1684 < 20; x1684++) {
int32_t x1686 = 50 * x1684;
for(int x1685=0; x1685 < 50; x1685++) {
int32_t x1687 = x1686 + x1685;
float x1688 = x346[x1687];
float x1689 = x334[x1687];
float x1690 = x438[x1687];
float x1691 = x462[x1687];
float x1692 = x1691 * x1690;
float x1693 = x1688 + x1692;
x346[x1687] = x1693;
float x1695 = x447[x1687];
float x1696 = x334[x1687];
float x1697 = x438[x1687];
float x1698 = x462[x1687];
float x1699 = x1698 * x1696;
float x1700 = x1695 + x1699;
x447[x1687] = x1700;

}

}
for(int x1706=0; x1706 < 1000; x1706++) {
float x1707 = x437[x1706];
float x1708 = x438[x1706];
float x1711 = x447[x1706];
float x1709 = x1708 * x1708;
float x1710 = 1.0f - x1709;
float x1712 = x1710 * x1711;
float x1713 = x1707 + x1712;
x437[x1706] = x1713;

}
// back prop for + op
for(int x1718=0; x1718 < 20; x1718++) {
int32_t x1720 = 50 * x1718;
for(int x1719=0; x1719 < 50; x1719++) {
int32_t x1721 = x1720 + x1719;
float x1722 = x407[x1721];
float x1723 = x437[x1721];
float x1724 = x1722 + x1723;
x407[x1721] = x1724;

}

}
for(int x1730=0; x1730 < 20; x1730++) {
int32_t x1732 = 50 * x1730;
for(int x1731=0; x1731 < 50; x1731++) {
int32_t x1733 = x1732 + x1731;
float x1734 = x422[x1733];
float x1735 = x437[x1733];
float x1736 = x1734 + x1735;
x422[x1733] = x1736;

}

}
// backprop for * op
for(int x1743=0; x1743 < 20; x1743++) {
int32_t x1745 = 50 * x1743;
for(int x1744=0; x1744 < 50; x1744++) {
int32_t x1746 = x1745 + x1744;
float x1747 = x297[x1746];
float x1748 = x285[x1746];
float x1749 = x383[x1746];
float x1750 = x422[x1746];
float x1751 = x1750 * x1749;
float x1752 = x1747 + x1751;
x297[x1746] = x1752;
float x1754 = x392[x1746];
float x1755 = x285[x1746];
float x1756 = x383[x1746];
float x1757 = x422[x1746];
float x1758 = x1757 * x1755;
float x1759 = x1754 + x1758;
x392[x1746] = x1759;

}

}
// backprop for * op
for(int x1766=0; x1766 < 20; x1766++) {
int32_t x1768 = 50 * x1766;
for(int x1767=0; x1767 < 50; x1767++) {
int32_t x1769 = x1768 + x1767;
float x1770 = x248[x1769];
float x1771 = x235[x1769];
float x1772 = x192[x1769];
float x1773 = x407[x1769];
float x1774 = x1773 * x1772;
float x1775 = x1770 + x1774;
x248[x1769] = x1775;
float x1777 = x193[x1769];
float x1778 = x235[x1769];
float x1779 = x192[x1769];
float x1780 = x407[x1769];
float x1781 = x1780 * x1778;
float x1782 = x1777 + x1781;
x193[x1769] = x1782;

}

}
for(int x1788=0; x1788 < 1000; x1788++) {
float x1789 = x382[x1788];
float x1790 = x383[x1788];
float x1793 = x392[x1788];
float x1791 = x1790 * x1790;
float x1792 = 1.0f - x1791;
float x1794 = x1792 * x1793;
float x1795 = x1789 + x1794;
x382[x1788] = x1795;

}
// back prop for + op
for(int x1800=0; x1800 < 20; x1800++) {
int32_t x1802 = 50 * x1800;
for(int x1801=0; x1801 < 50; x1801++) {
int32_t x1803 = x1802 + x1801;
float x1804 = x367[x1803];
float x1805 = x382[x1803];
float x1806 = x1804 + x1805;
x367[x1803] = x1806;

}

}
for(int x1812=0; x1812 < 20; x1812++) {
int32_t x1814 = 50 * x1812;
for(int x1813=0; x1813 < 50; x1813++) {
float x1816 = x80[x1813];
int32_t x1815 = x1814 + x1813;
float x1817 = x382[x1815];
float x1818 = x1816 + x1817;
x80[x1813] = x1818;

}

}
// back prop for + op
for(int x1825=0; x1825 < 20; x1825++) {
int32_t x1827 = 50 * x1825;
for(int x1826=0; x1826 < 50; x1826++) {
int32_t x1828 = x1827 + x1826;
float x1829 = x349[x1828];
float x1830 = x367[x1828];
float x1831 = x1829 + x1830;
x349[x1828] = x1831;

}

}
for(int x1837=0; x1837 < 20; x1837++) {
int32_t x1839 = 50 * x1837;
for(int x1838=0; x1838 < 50; x1838++) {
int32_t x1840 = x1839 + x1838;
float x1841 = x352[x1840];
float x1842 = x367[x1840];
float x1843 = x1841 + x1842;
x352[x1840] = x1843;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x352,50,x70,50,1,x191,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x190,50,x352,50,1,x78,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x349,50,x61,50,1,x197,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x196,26,x349,50,1,x69,50);
for(int x1855=0; x1855 < 1000; x1855++) {
float x1856 = x333[x1855];
float x1857 = x334[x1855];
float x1860 = x346[x1855];
float x1858 = 1.0f - x1857;
float x1859 = x1858 * x1857;
float x1861 = x1859 * x1860;
float x1862 = x1856 + x1861;
x333[x1855] = x1862;

}
// back prop for + op
for(int x1867=0; x1867 < 20; x1867++) {
int32_t x1869 = 50 * x1867;
for(int x1868=0; x1868 < 50; x1868++) {
int32_t x1870 = x1869 + x1868;
float x1871 = x318[x1870];
float x1872 = x333[x1870];
float x1873 = x1871 + x1872;
x318[x1870] = x1873;

}

}
for(int x1879=0; x1879 < 20; x1879++) {
int32_t x1881 = 50 * x1879;
for(int x1880=0; x1880 < 50; x1880++) {
float x1883 = x100[x1880];
int32_t x1882 = x1881 + x1880;
float x1884 = x333[x1882];
float x1885 = x1883 + x1884;
x100[x1880] = x1885;

}

}
// back prop for + op
for(int x1892=0; x1892 < 20; x1892++) {
int32_t x1894 = 50 * x1892;
for(int x1893=0; x1893 < 50; x1893++) {
int32_t x1895 = x1894 + x1893;
float x1896 = x300[x1895];
float x1897 = x318[x1895];
float x1898 = x1896 + x1897;
x300[x1895] = x1898;

}

}
for(int x1904=0; x1904 < 20; x1904++) {
int32_t x1906 = 50 * x1904;
for(int x1905=0; x1905 < 50; x1905++) {
int32_t x1907 = x1906 + x1905;
float x1908 = x303[x1907];
float x1909 = x318[x1907];
float x1910 = x1908 + x1909;
x303[x1907] = x1910;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x303,50,x90,50,1,x191,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x190,50,x303,50,1,x98,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x300,50,x81,50,1,x197,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x196,26,x300,50,1,x89,50);
for(int x1922=0; x1922 < 1000; x1922++) {
float x1923 = x284[x1922];
float x1924 = x285[x1922];
float x1927 = x297[x1922];
float x1925 = 1.0f - x1924;
float x1926 = x1925 * x1924;
float x1928 = x1926 * x1927;
float x1929 = x1923 + x1928;
x284[x1922] = x1929;

}
// back prop for + op
for(int x1934=0; x1934 < 20; x1934++) {
int32_t x1936 = 50 * x1934;
for(int x1935=0; x1935 < 50; x1935++) {
int32_t x1937 = x1936 + x1935;
float x1938 = x269[x1937];
float x1939 = x284[x1937];
float x1940 = x1938 + x1939;
x269[x1937] = x1940;

}

}
for(int x1946=0; x1946 < 20; x1946++) {
int32_t x1948 = 50 * x1946;
for(int x1947=0; x1947 < 50; x1947++) {
float x1950 = x60[x1947];
int32_t x1949 = x1948 + x1947;
float x1951 = x284[x1949];
float x1952 = x1950 + x1951;
x60[x1947] = x1952;

}

}
// back prop for + op
for(int x1959=0; x1959 < 20; x1959++) {
int32_t x1961 = 50 * x1959;
for(int x1960=0; x1960 < 50; x1960++) {
int32_t x1962 = x1961 + x1960;
float x1963 = x251[x1962];
float x1964 = x269[x1962];
float x1965 = x1963 + x1964;
x251[x1962] = x1965;

}

}
for(int x1971=0; x1971 < 20; x1971++) {
int32_t x1973 = 50 * x1971;
for(int x1972=0; x1972 < 50; x1972++) {
int32_t x1974 = x1973 + x1972;
float x1975 = x254[x1974];
float x1976 = x269[x1974];
float x1977 = x1975 + x1976;
x254[x1974] = x1977;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x254,50,x50,50,1,x191,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x190,50,x254,50,1,x58,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x251,50,x41,50,1,x197,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x196,26,x251,50,1,x49,50);
for(int x1989=0; x1989 < 1000; x1989++) {
float x1990 = x234[x1989];
float x1991 = x235[x1989];
float x1994 = x248[x1989];
float x1992 = 1.0f - x1991;
float x1993 = x1992 * x1991;
float x1995 = x1993 * x1994;
float x1996 = x1990 + x1995;
x234[x1989] = x1996;

}
// back prop for + op
for(int x2001=0; x2001 < 20; x2001++) {
int32_t x2003 = 50 * x2001;
for(int x2002=0; x2002 < 50; x2002++) {
int32_t x2004 = x2003 + x2002;
float x2005 = x219[x2004];
float x2006 = x234[x2004];
float x2007 = x2005 + x2006;
x219[x2004] = x2007;

}

}
for(int x2013=0; x2013 < 20; x2013++) {
int32_t x2015 = 50 * x2013;
for(int x2014=0; x2014 < 50; x2014++) {
float x2017 = x40[x2014];
int32_t x2016 = x2015 + x2014;
float x2018 = x234[x2016];
float x2019 = x2017 + x2018;
x40[x2014] = x2019;

}

}
// back prop for + op
for(int x2026=0; x2026 < 20; x2026++) {
int32_t x2028 = 50 * x2026;
for(int x2027=0; x2027 < 50; x2027++) {
int32_t x2029 = x2028 + x2027;
float x2030 = x200[x2029];
float x2031 = x219[x2029];
float x2032 = x2030 + x2031;
x200[x2029] = x2032;

}

}
for(int x2038=0; x2038 < 20; x2038++) {
int32_t x2040 = 50 * x2038;
for(int x2039=0; x2039 < 50; x2039++) {
int32_t x2041 = x2040 + x2039;
float x2042 = x203[x2041];
float x2043 = x219[x2041];
float x2044 = x2042 + x2043;
x203[x2041] = x2044;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x203,50,x29,50,1,x191,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x190,50,x203,50,1,x38,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x200,50,x18,50,1,x197,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x196,26,x200,50,1,x28,50);
} else {
float x2057 = 0.0f;
for(int x2058=0; x2058 < 1; x2058++) {
float x2059 = x2057;
float x2060 = x188[x2058];
float x2061 = x2059 + x2060;
x2057 = x2061;

}
float x2065 = x2057;
float* x2066 = (float*)myMalloc(1 * sizeof(float));;
for(int x2067=0; x2067 < 1; x2067++) {
x2066[x2067] = x2065;

}
float* x2071 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x2073=0; x2073 < 1; x2073++) {
x2071[x2073] = 1.0f;

}
// backend is lantern.TensorDsl$BackendCPU@51182e06
for(int x2078=0; x2078 < 1; x2078++) {
float x2079 = x2066[x2078];
x158[x2078] = x2079;

}
// 'sum' gradient.
if (x596) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x2088=0; x2088 < 1; x2088++) {
float x2089 = x189[0];
float x2090 = x2071[0];
float x2091 = x2089 + x2090;
x189[0] = x2091;

}
}
};
x183(0,x2098);
float x2107 = x158[0];
int32_t x2108 = x133 % 100;
bool x2109 = x2108 == 0;
if (x2109) {
printf("iter %d, loss %f\n",x133,x2107);
int32_t x2111 = x133 / 100;
double x2112 = (double)x2107;
x127[x2111] = x2112;
} else {
}
for(int x2116=0; x2116 < 1300; x2116++) {
float x2117 = x49[x2116];
float x2118 = x2117;
float x2119 = x2118;
bool x2120 = x2119 > 5.0f;
if (x2120) {
x2118 = 5.0f;
} else {
}
float x2124 = x2118;
bool x2125 = x2124 < -5.0f;
if (x2125) {
x2118 = -5.0f;
} else {
}
float x2129 = x112[x2116];
float x2130 = x2118;
float x2131 = x2130 * x2130;
float x2132 = x2129 + x2131;
x112[x2116] = x2132;
float x2134 = x41[x2116];
float x2136 = x112[x2116];
float x2135 = 0.1f * x2130;
double x2137 = (double)x2136;
double x2138 = x2137 + 9.99999993922529E-9;
double x2139 = sqrt(x2138);
float x2140 = (float)x2139;
float x2141 = x2135 / x2140;
float x2142 = x2134 - x2141;
x41[x2116] = x2142;
x49[x2116] = 0.0f;

}
for(int x2147=0; x2147 < 50; x2147++) {
float x2148 = x60[x2147];
float x2149 = x2148;
float x2150 = x2149;
bool x2151 = x2150 > 5.0f;
if (x2151) {
x2149 = 5.0f;
} else {
}
float x2155 = x2149;
bool x2156 = x2155 < -5.0f;
if (x2156) {
x2149 = -5.0f;
} else {
}
float x2160 = x113[x2147];
float x2161 = x2149;
float x2162 = x2161 * x2161;
float x2163 = x2160 + x2162;
x113[x2147] = x2163;
float x2165 = x59[x2147];
float x2167 = x113[x2147];
float x2166 = 0.1f * x2161;
double x2168 = (double)x2167;
double x2169 = x2168 + 9.99999993922529E-9;
double x2170 = sqrt(x2169);
float x2171 = (float)x2170;
float x2172 = x2166 / x2171;
float x2173 = x2165 - x2172;
x59[x2147] = x2173;
x60[x2147] = 0.0f;

}
for(int x2178=0; x2178 < 2500; x2178++) {
float x2179 = x58[x2178];
float x2180 = x2179;
float x2181 = x2180;
bool x2182 = x2181 > 5.0f;
if (x2182) {
x2180 = 5.0f;
} else {
}
float x2186 = x2180;
bool x2187 = x2186 < -5.0f;
if (x2187) {
x2180 = -5.0f;
} else {
}
float x2191 = x114[x2178];
float x2192 = x2180;
float x2193 = x2192 * x2192;
float x2194 = x2191 + x2193;
x114[x2178] = x2194;
float x2196 = x50[x2178];
float x2198 = x114[x2178];
float x2197 = 0.1f * x2192;
double x2199 = (double)x2198;
double x2200 = x2199 + 9.99999993922529E-9;
double x2201 = sqrt(x2200);
float x2202 = (float)x2201;
float x2203 = x2197 / x2202;
float x2204 = x2196 - x2203;
x50[x2178] = x2204;
x58[x2178] = 0.0f;

}
for(int x2209=0; x2209 < 50; x2209++) {
float x2210 = x40[x2209];
float x2211 = x2210;
float x2212 = x2211;
bool x2213 = x2212 > 5.0f;
if (x2213) {
x2211 = 5.0f;
} else {
}
float x2217 = x2211;
bool x2218 = x2217 < -5.0f;
if (x2218) {
x2211 = -5.0f;
} else {
}
float x2222 = x115[x2209];
float x2223 = x2211;
float x2224 = x2223 * x2223;
float x2225 = x2222 + x2224;
x115[x2209] = x2225;
float x2227 = x39[x2209];
float x2229 = x115[x2209];
float x2228 = 0.1f * x2223;
double x2230 = (double)x2229;
double x2231 = x2230 + 9.99999993922529E-9;
double x2232 = sqrt(x2231);
float x2233 = (float)x2232;
float x2234 = x2228 / x2233;
float x2235 = x2227 - x2234;
x39[x2209] = x2235;
x40[x2209] = 0.0f;

}
for(int x2240=0; x2240 < 2500; x2240++) {
float x2241 = x38[x2240];
float x2242 = x2241;
float x2243 = x2242;
bool x2244 = x2243 > 5.0f;
if (x2244) {
x2242 = 5.0f;
} else {
}
float x2248 = x2242;
bool x2249 = x2248 < -5.0f;
if (x2249) {
x2242 = -5.0f;
} else {
}
float x2253 = x116[x2240];
float x2254 = x2242;
float x2255 = x2254 * x2254;
float x2256 = x2253 + x2255;
x116[x2240] = x2256;
float x2258 = x29[x2240];
float x2260 = x116[x2240];
float x2259 = 0.1f * x2254;
double x2261 = (double)x2260;
double x2262 = x2261 + 9.99999993922529E-9;
double x2263 = sqrt(x2262);
float x2264 = (float)x2263;
float x2265 = x2259 / x2264;
float x2266 = x2258 - x2265;
x29[x2240] = x2266;
x38[x2240] = 0.0f;

}
for(int x2271=0; x2271 < 1300; x2271++) {
float x2272 = x28[x2271];
float x2273 = x2272;
float x2274 = x2273;
bool x2275 = x2274 > 5.0f;
if (x2275) {
x2273 = 5.0f;
} else {
}
float x2279 = x2273;
bool x2280 = x2279 < -5.0f;
if (x2280) {
x2273 = -5.0f;
} else {
}
float x2284 = x117[x2271];
float x2285 = x2273;
float x2286 = x2285 * x2285;
float x2287 = x2284 + x2286;
x117[x2271] = x2287;
float x2289 = x18[x2271];
float x2291 = x117[x2271];
float x2290 = 0.1f * x2285;
double x2292 = (double)x2291;
double x2293 = x2292 + 9.99999993922529E-9;
double x2294 = sqrt(x2293);
float x2295 = (float)x2294;
float x2296 = x2290 / x2295;
float x2297 = x2289 - x2296;
x18[x2271] = x2297;
x28[x2271] = 0.0f;

}
for(int x2302=0; x2302 < 1300; x2302++) {
float x2303 = x69[x2302];
float x2304 = x2303;
float x2305 = x2304;
bool x2306 = x2305 > 5.0f;
if (x2306) {
x2304 = 5.0f;
} else {
}
float x2310 = x2304;
bool x2311 = x2310 < -5.0f;
if (x2311) {
x2304 = -5.0f;
} else {
}
float x2315 = x118[x2302];
float x2316 = x2304;
float x2317 = x2316 * x2316;
float x2318 = x2315 + x2317;
x118[x2302] = x2318;
float x2320 = x61[x2302];
float x2322 = x118[x2302];
float x2321 = 0.1f * x2316;
double x2323 = (double)x2322;
double x2324 = x2323 + 9.99999993922529E-9;
double x2325 = sqrt(x2324);
float x2326 = (float)x2325;
float x2327 = x2321 / x2326;
float x2328 = x2320 - x2327;
x61[x2302] = x2328;
x69[x2302] = 0.0f;

}
for(int x2333=0; x2333 < 50; x2333++) {
float x2334 = x80[x2333];
float x2335 = x2334;
float x2336 = x2335;
bool x2337 = x2336 > 5.0f;
if (x2337) {
x2335 = 5.0f;
} else {
}
float x2341 = x2335;
bool x2342 = x2341 < -5.0f;
if (x2342) {
x2335 = -5.0f;
} else {
}
float x2346 = x119[x2333];
float x2347 = x2335;
float x2348 = x2347 * x2347;
float x2349 = x2346 + x2348;
x119[x2333] = x2349;
float x2351 = x79[x2333];
float x2353 = x119[x2333];
float x2352 = 0.1f * x2347;
double x2354 = (double)x2353;
double x2355 = x2354 + 9.99999993922529E-9;
double x2356 = sqrt(x2355);
float x2357 = (float)x2356;
float x2358 = x2352 / x2357;
float x2359 = x2351 - x2358;
x79[x2333] = x2359;
x80[x2333] = 0.0f;

}
for(int x2364=0; x2364 < 2500; x2364++) {
float x2365 = x78[x2364];
float x2366 = x2365;
float x2367 = x2366;
bool x2368 = x2367 > 5.0f;
if (x2368) {
x2366 = 5.0f;
} else {
}
float x2372 = x2366;
bool x2373 = x2372 < -5.0f;
if (x2373) {
x2366 = -5.0f;
} else {
}
float x2377 = x120[x2364];
float x2378 = x2366;
float x2379 = x2378 * x2378;
float x2380 = x2377 + x2379;
x120[x2364] = x2380;
float x2382 = x70[x2364];
float x2384 = x120[x2364];
float x2383 = 0.1f * x2378;
double x2385 = (double)x2384;
double x2386 = x2385 + 9.99999993922529E-9;
double x2387 = sqrt(x2386);
float x2388 = (float)x2387;
float x2389 = x2383 / x2388;
float x2390 = x2382 - x2389;
x70[x2364] = x2390;
x78[x2364] = 0.0f;

}
for(int x2395=0; x2395 < 26; x2395++) {
float x2396 = x111[x2395];
float x2397 = x2396;
float x2398 = x2397;
bool x2399 = x2398 > 5.0f;
if (x2399) {
x2397 = 5.0f;
} else {
}
float x2403 = x2397;
bool x2404 = x2403 < -5.0f;
if (x2404) {
x2397 = -5.0f;
} else {
}
float x2408 = x121[x2395];
float x2409 = x2397;
float x2410 = x2409 * x2409;
float x2411 = x2408 + x2410;
x121[x2395] = x2411;
float x2413 = x110[x2395];
float x2415 = x121[x2395];
float x2414 = 0.1f * x2409;
double x2416 = (double)x2415;
double x2417 = x2416 + 9.99999993922529E-9;
double x2418 = sqrt(x2417);
float x2419 = (float)x2418;
float x2420 = x2414 / x2419;
float x2421 = x2413 - x2420;
x110[x2395] = x2421;
x111[x2395] = 0.0f;

}
for(int x2426=0; x2426 < 1300; x2426++) {
float x2427 = x109[x2426];
float x2428 = x2427;
float x2429 = x2428;
bool x2430 = x2429 > 5.0f;
if (x2430) {
x2428 = 5.0f;
} else {
}
float x2434 = x2428;
bool x2435 = x2434 < -5.0f;
if (x2435) {
x2428 = -5.0f;
} else {
}
float x2439 = x122[x2426];
float x2440 = x2428;
float x2441 = x2440 * x2440;
float x2442 = x2439 + x2441;
x122[x2426] = x2442;
float x2444 = x101[x2426];
float x2446 = x122[x2426];
float x2445 = 0.1f * x2440;
double x2447 = (double)x2446;
double x2448 = x2447 + 9.99999993922529E-9;
double x2449 = sqrt(x2448);
float x2450 = (float)x2449;
float x2451 = x2445 / x2450;
float x2452 = x2444 - x2451;
x101[x2426] = x2452;
x109[x2426] = 0.0f;

}
for(int x2457=0; x2457 < 2500; x2457++) {
float x2458 = x98[x2457];
float x2459 = x2458;
float x2460 = x2459;
bool x2461 = x2460 > 5.0f;
if (x2461) {
x2459 = 5.0f;
} else {
}
float x2465 = x2459;
bool x2466 = x2465 < -5.0f;
if (x2466) {
x2459 = -5.0f;
} else {
}
float x2470 = x123[x2457];
float x2471 = x2459;
float x2472 = x2471 * x2471;
float x2473 = x2470 + x2472;
x123[x2457] = x2473;
float x2475 = x90[x2457];
float x2477 = x123[x2457];
float x2476 = 0.1f * x2471;
double x2478 = (double)x2477;
double x2479 = x2478 + 9.99999993922529E-9;
double x2480 = sqrt(x2479);
float x2481 = (float)x2480;
float x2482 = x2476 / x2481;
float x2483 = x2475 - x2482;
x90[x2457] = x2483;
x98[x2457] = 0.0f;

}
for(int x2488=0; x2488 < 1300; x2488++) {
float x2489 = x89[x2488];
float x2490 = x2489;
float x2491 = x2490;
bool x2492 = x2491 > 5.0f;
if (x2492) {
x2490 = 5.0f;
} else {
}
float x2496 = x2490;
bool x2497 = x2496 < -5.0f;
if (x2497) {
x2490 = -5.0f;
} else {
}
float x2501 = x124[x2488];
float x2502 = x2490;
float x2503 = x2502 * x2502;
float x2504 = x2501 + x2503;
x124[x2488] = x2504;
float x2506 = x81[x2488];
float x2508 = x124[x2488];
float x2507 = 0.1f * x2502;
double x2509 = (double)x2508;
double x2510 = x2509 + 9.99999993922529E-9;
double x2511 = sqrt(x2510);
float x2512 = (float)x2511;
float x2513 = x2507 / x2512;
float x2514 = x2506 - x2513;
x81[x2488] = x2514;
x89[x2488] = 0.0f;

}
for(int x2519=0; x2519 < 50; x2519++) {
float x2520 = x100[x2519];
float x2521 = x2520;
float x2522 = x2521;
bool x2523 = x2522 > 5.0f;
if (x2523) {
x2521 = 5.0f;
} else {
}
float x2527 = x2521;
bool x2528 = x2527 < -5.0f;
if (x2528) {
x2521 = -5.0f;
} else {
}
float x2532 = x125[x2519];
float x2533 = x2521;
float x2534 = x2533 * x2533;
float x2535 = x2532 + x2534;
x125[x2519] = x2535;
float x2537 = x99[x2519];
float x2539 = x125[x2519];
float x2538 = 0.1f * x2533;
double x2540 = (double)x2539;
double x2541 = x2540 + 9.99999993922529E-9;
double x2542 = sqrt(x2541);
float x2543 = (float)x2542;
float x2544 = x2538 / x2543;
float x2545 = x2537 - x2544;
x99[x2519] = x2545;
x100[x2519] = 0.0f;

}
int64_t x2550 = (long)mallocAddr;
int64_t x2551 = x2550 - x128;
memset((void*)x128, 0, x2551);
mallocAddr = (void*)x128;

}
double x2556 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x2559 = (long)fopen(x0, "w");
fprintf((FILE *)x2559, "unit: %s\n", "100 iteration");
for(int x2562=0; x2562 < 51; x2562++) {
double x2563 = x127[x2562];
fprintf((FILE *)x2559, "%lf\n", x2563);

}
double x2557 = x126 - x2;
double x2558 = x2556 - x126;
fprintf((FILE *)x2559, "run time: %lf %lf\n", x2557, x2558);
fclose((FILE*)x2559);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

