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
bool x596 = true || true;
bool x597 = x596 || true;
bool x1065 = true || false;
for(int x133=0; x133 < 5001; x133++) {
float* x158 = (float*)myMalloc(1 * sizeof(float));;
float* x159 = (float*)myMalloc(10400 * sizeof(float));;
float* x176 = (float*)myMalloc(10400 * sizeof(float));;
int* x143 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
function<void(int32_t,float**)> x613 = [&](int32_t x614,float** x615) {
float** x617 = x615;
float* x618 = x617[0];
float* x619 = x617[1];
float* x620 = x617[2];
float* x621 = x617[3];
float* x622 = x617[4];
float* x623 = x617[5];
int32_t x616 = x614;
bool x624 = x616 < 20;
if (x624) {
int32_t x625 = x616 * 520;
float* x626 = x159+x625;
float* x627 = x176+x625;
float* x628 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x626,26,x18,50,0,x628,50);
float* x630 = (float*)myMalloc(1000 * sizeof(float));;
float* x631 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x620,50,x29,50,0,x631,50);
float* x633 = (float*)myMalloc(1000 * sizeof(float));;
float* x634 = (float*)myMalloc(1000 * sizeof(float));;
for(int x635=0; x635 < 20; x635++) {
int32_t x637 = 50 * x635;
for(int x636=0; x636 < 50; x636++) {
int32_t x639 = x637 + x636;
float x640 = x628[x639];
float x641 = x631[x639];
int32_t x638 = x636 + x637;
float x642 = x640 + x641;
x634[x638] = x642;

}

}
float* x648 = (float*)myMalloc(1000 * sizeof(float));;
float* x649 = (float*)myMalloc(1000 * sizeof(float));;
for(int x650=0; x650 < 20; x650++) {
int32_t x652 = 50 * x650;
for(int x651=0; x651 < 50; x651++) {
int32_t x654 = x652 + x651;
float x655 = x634[x654];
float x656 = x39[x651];
int32_t x653 = x651 + x652;
float x657 = x655 + x656;
x649[x653] = x657;

}

}
float* x663 = (float*)myMalloc(1000 * sizeof(float));;
float* x664 = (float*)myMalloc(1000 * sizeof(float));;
for(int x665=0; x665 < 1000; x665++) {
float x666 = x649[x665];
float x667 = -1.0f * x666;
double x668 = (double)x667;
double x669 = exp(x668);
float x670 = (float)x669;
float x671 = x670 + 1.0f;
float x672 = 1.0f / x671;
x664[x665] = x672;

}
float* x676 = (float*)myMalloc(1000 * sizeof(float));;
float* x677 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x626,26,x41,50,0,x677,50);
float* x679 = (float*)myMalloc(1000 * sizeof(float));;
float* x680 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x620,50,x50,50,0,x680,50);
float* x682 = (float*)myMalloc(1000 * sizeof(float));;
float* x683 = (float*)myMalloc(1000 * sizeof(float));;
for(int x684=0; x684 < 20; x684++) {
int32_t x686 = 50 * x684;
for(int x685=0; x685 < 50; x685++) {
int32_t x688 = x686 + x685;
float x689 = x677[x688];
float x690 = x680[x688];
int32_t x687 = x685 + x686;
float x691 = x689 + x690;
x683[x687] = x691;

}

}
float* x697 = (float*)myMalloc(1000 * sizeof(float));;
float* x698 = (float*)myMalloc(1000 * sizeof(float));;
for(int x699=0; x699 < 20; x699++) {
int32_t x701 = 50 * x699;
for(int x700=0; x700 < 50; x700++) {
int32_t x703 = x701 + x700;
float x704 = x683[x703];
float x705 = x59[x700];
int32_t x702 = x700 + x701;
float x706 = x704 + x705;
x698[x702] = x706;

}

}
float* x712 = (float*)myMalloc(1000 * sizeof(float));;
float* x713 = (float*)myMalloc(1000 * sizeof(float));;
for(int x714=0; x714 < 1000; x714++) {
float x715 = x698[x714];
float x716 = -1.0f * x715;
double x717 = (double)x716;
double x718 = exp(x717);
float x719 = (float)x718;
float x720 = x719 + 1.0f;
float x721 = 1.0f / x720;
x713[x714] = x721;

}
float* x725 = (float*)myMalloc(1000 * sizeof(float));;
float* x726 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x626,26,x81,50,0,x726,50);
float* x728 = (float*)myMalloc(1000 * sizeof(float));;
float* x729 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x620,50,x90,50,0,x729,50);
float* x731 = (float*)myMalloc(1000 * sizeof(float));;
float* x732 = (float*)myMalloc(1000 * sizeof(float));;
for(int x733=0; x733 < 20; x733++) {
int32_t x735 = 50 * x733;
for(int x734=0; x734 < 50; x734++) {
int32_t x737 = x735 + x734;
float x738 = x726[x737];
float x739 = x729[x737];
int32_t x736 = x734 + x735;
float x740 = x738 + x739;
x732[x736] = x740;

}

}
float* x746 = (float*)myMalloc(1000 * sizeof(float));;
float* x747 = (float*)myMalloc(1000 * sizeof(float));;
for(int x748=0; x748 < 20; x748++) {
int32_t x750 = 50 * x748;
for(int x749=0; x749 < 50; x749++) {
int32_t x752 = x750 + x749;
float x753 = x732[x752];
float x754 = x99[x749];
int32_t x751 = x749 + x750;
float x755 = x753 + x754;
x747[x751] = x755;

}

}
float* x761 = (float*)myMalloc(1000 * sizeof(float));;
float* x762 = (float*)myMalloc(1000 * sizeof(float));;
for(int x763=0; x763 < 1000; x763++) {
float x764 = x747[x763];
float x765 = -1.0f * x764;
double x766 = (double)x765;
double x767 = exp(x766);
float x768 = (float)x767;
float x769 = x768 + 1.0f;
float x770 = 1.0f / x769;
x762[x763] = x770;

}
float* x774 = (float*)myMalloc(1000 * sizeof(float));;
float* x775 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x626,26,x61,50,0,x775,50);
float* x777 = (float*)myMalloc(1000 * sizeof(float));;
float* x778 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x620,50,x70,50,0,x778,50);
float* x780 = (float*)myMalloc(1000 * sizeof(float));;
float* x781 = (float*)myMalloc(1000 * sizeof(float));;
for(int x782=0; x782 < 20; x782++) {
int32_t x784 = 50 * x782;
for(int x783=0; x783 < 50; x783++) {
int32_t x786 = x784 + x783;
float x787 = x775[x786];
float x788 = x778[x786];
int32_t x785 = x783 + x784;
float x789 = x787 + x788;
x781[x785] = x789;

}

}
float* x795 = (float*)myMalloc(1000 * sizeof(float));;
float* x796 = (float*)myMalloc(1000 * sizeof(float));;
for(int x797=0; x797 < 20; x797++) {
int32_t x799 = 50 * x797;
for(int x798=0; x798 < 50; x798++) {
int32_t x801 = x799 + x798;
float x802 = x781[x801];
float x803 = x79[x798];
int32_t x800 = x798 + x799;
float x804 = x802 + x803;
x796[x800] = x804;

}

}
float* x810 = (float*)myMalloc(1000 * sizeof(float));;
float* x811 = (float*)myMalloc(1000 * sizeof(float));;
for(int x812=0; x812 < 1000; x812++) {
float x813 = x796[x812];
double x814 = (double)x813;
double x815 = tanh(x814);
float x816 = (float)x815;
x811[x812] = x816;

}
float* x820 = (float*)myMalloc(1000 * sizeof(float));;
float* x821 = (float*)myMalloc(1000 * sizeof(float));;
for(int x822=0; x822 < 20; x822++) {
int32_t x824 = 50 * x822;
for(int x823=0; x823 < 50; x823++) {
int32_t x826 = x824 + x823;
float x827 = x664[x826];
float x828 = x622[x826];
int32_t x825 = x823 + x824;
float x829 = x827 * x828;
x821[x825] = x829;

}

}
float* x835 = (float*)myMalloc(1000 * sizeof(float));;
float* x836 = (float*)myMalloc(1000 * sizeof(float));;
for(int x837=0; x837 < 20; x837++) {
int32_t x839 = 50 * x837;
for(int x838=0; x838 < 50; x838++) {
int32_t x841 = x839 + x838;
float x842 = x713[x841];
float x843 = x811[x841];
int32_t x840 = x838 + x839;
float x844 = x842 * x843;
x836[x840] = x844;

}

}
float* x850 = (float*)myMalloc(1000 * sizeof(float));;
float* x851 = (float*)myMalloc(1000 * sizeof(float));;
for(int x852=0; x852 < 20; x852++) {
int32_t x854 = 50 * x852;
for(int x853=0; x853 < 50; x853++) {
int32_t x856 = x854 + x853;
float x857 = x821[x856];
float x858 = x836[x856];
int32_t x855 = x853 + x854;
float x859 = x857 + x858;
x851[x855] = x859;

}

}
float* x865 = (float*)myMalloc(1000 * sizeof(float));;
float* x866 = (float*)myMalloc(1000 * sizeof(float));;
for(int x867=0; x867 < 1000; x867++) {
float x868 = x851[x867];
double x869 = (double)x868;
double x870 = tanh(x869);
float x871 = (float)x870;
x866[x867] = x871;

}
float* x875 = (float*)myMalloc(1000 * sizeof(float));;
float* x876 = (float*)myMalloc(1000 * sizeof(float));;
for(int x877=0; x877 < 20; x877++) {
int32_t x879 = 50 * x877;
for(int x878=0; x878 < 50; x878++) {
int32_t x881 = x879 + x878;
float x882 = x762[x881];
float x883 = x866[x881];
int32_t x880 = x878 + x879;
float x884 = x882 * x883;
x876[x880] = x884;

}

}
float* x890 = (float*)myMalloc(1000 * sizeof(float));;
float* x891 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x876,50,x101,26,0,x891,26);
float* x893 = (float*)myMalloc(520 * sizeof(float));;
for(int x894=0; x894 < 20; x894++) {
int32_t x896 = 26 * x894;
for(int x895=0; x895 < 26; x895++) {
int32_t x897 = x896 + x895;
float x898 = x891[x897];
float x899 = x110[x895];
float x900 = x898 + x899;
x891[x897] = x900;

}

}
int* x906 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x907=0; x907 < 20; x907++) {
int32_t x908 = x907 * 20;
int32_t x909 = x616 + x908;
int32_t x910 = x143[x909];
x906[x907] = x910;

}
float* x914 = (float*)myMalloc(20 * sizeof(float));;
int32_t x915 = 0;
for(int x916=0; x916 < 20; x916++) {
float x917 = -3.4028235E38f;
for(int x918=0; x918 < 26; x918++) {
int32_t x919 = x915;
float x920 = x891[x919];
float x921 = x917;
bool x922 = x920 > x921;
if (x922) {
float x923 = x891[x919];
x917 = x923;
} else {
}
x915 += 1;

}
float x930 = x917;
x914[x916] = x930;

}
float* x934 = (float*)myMalloc(520 * sizeof(float));;
int32_t x935 = 0;
for(int x936=0; x936 < 20; x936++) {
for(int x937=0; x937 < 26; x937++) {
int32_t x938 = x935;
float x939 = x891[x938];
float x940 = x914[x936];
float x941 = x939 - x940;
double x942 = (double)x941;
double x943 = exp(x942);
float x944 = (float)x943;
x934[x938] = x944;
x935 += 1;

}

}
float* x951 = (float*)myMalloc(20 * sizeof(float));;
for(int x952=0; x952 < 20; x952++) {
int32_t x953 = x952;
int32_t x954 = x952 * 26;
int32_t x955 = x954;
for(int x956=0; x956 < 26; x956++) {
for(int x957=0; x957 < 1; x957++) {
int32_t x958 = x953;
int32_t x959 = x958 + x957;
float x960 = x951[x959];
int32_t x961 = x955;
int32_t x962 = x961 + x957;
float x963 = x934[x962];
float x964 = x960 + x963;
x951[x959] = x964;

}
x955 += 1;

}

}
x935 = 0;
for(int x974=0; x974 < 20; x974++) {
float x975 = x914[x974];
float x976 = x951[x974];
double x977 = (double)x976;
double x978 = log(x977);
float x979 = (float)x978;
float x980 = x975 + x979;
for(int x981=0; x981 < 26; x981++) {
int32_t x982 = x935;
float x983 = x891[x982];
float x984 = x983 - x980;
x934[x982] = x984;
x935 += 1;

}

}
float* x991 = (float*)myMalloc(520 * sizeof(float));;
// nllLoss forward in CPU
float* x993 = (float*)myMalloc(20 * sizeof(float));;
int32_t x994 = 0;
for(int x995=0; x995 < 20; x995++) {
int32_t x996 = x994;
int32_t x997 = x906[x995];
int32_t x998 = x996 + x997;
float x999 = x934[x998];
float x1000 = -1.0f * x999;
x993[x995] = x1000;
x994 += 26;

}
float* x1005 = (float*)myMalloc(20 * sizeof(float));;
float x1006 = 0.0f;
for(int x1007=0; x1007 < 20; x1007++) {
float x1008 = x1006;
float x1009 = x993[x1007];
float x1010 = x1008 + x1009;
x1006 = x1010;

}
float x1014 = x1006;
float* x1015 = (float*)myMalloc(1 * sizeof(float));;
for(int x1016=0; x1016 < 1; x1016++) {
x1015[x1016] = x1014;

}
float* x1020 = (float*)myMalloc(1 * sizeof(float));;
if (x597) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x1025 = (float*)myMalloc(1 * sizeof(float));;
for(int x1026=0; x1026 < 1; x1026++) {
float x1027 = x618[0];
float x1028 = x1015[0];
float x1029 = x1027 + x1028;
x1025[x1026] = x1029;

}
float* x1033 = (float*)myMalloc(1 * sizeof(float));;
float** x1035 = (float**)myMalloc(6 * sizeof(float*));;
x1035[0] = x1025;
x1035[1] = x1033;
x1035[2] = x876;
x1035[3] = x890;
x1035[4] = x851;
x1035[5] = x865;
int32_t x1080 = 0;
float* x1093 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1115 = 0;
int32_t x1034 = x616 + 1;
x613(x1034,x1035);
// back prop for + op
if (x597) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1049=0; x1049 < 1; x1049++) {
float x1050 = x619[0];
float x1051 = x618[0];
float x1052 = x1015[0];
float x1053 = x1033[x1049];
float x1054 = x1050 + x1053;
x619[0] = x1054;
float x1056 = x1020[0];
float x1057 = x618[0];
float x1058 = x1015[0];
float x1059 = x1033[x1049];
float x1060 = x1056 + x1059;
x1020[0] = x1060;

}
// 'sum' gradient.
if (x1065) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",20,1);
assert(false && "");
}
for(int x1071=0; x1071 < 20; x1071++) {
float x1072 = x1005[x1071];
float x1073 = x1020[0];
float x1074 = x1072 + x1073;
x1005[x1071] = x1074;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
for(int x1081=0; x1081 < 20; x1081++) {
int32_t x1082 = x1080;
int32_t x1083 = x906[x1081];
int32_t x1084 = x1082 + x1083;
float x1085 = x991[x1084];
float x1086 = x1005[x1081];
float x1087 = -1.0f * x1086;
float x1088 = x1085 + x1087;
x991[x1084] = x1088;
x1080 += 26;

}
for(int x1094=0; x1094 < 20; x1094++) {
int32_t x1095 = x1094;
int32_t x1096 = x1094 * 26;
int32_t x1097 = x1096;
for(int x1098=0; x1098 < 26; x1098++) {
for(int x1099=0; x1099 < 1; x1099++) {
int32_t x1100 = x1095;
int32_t x1101 = x1100 + x1099;
float x1102 = x1093[x1101];
int32_t x1103 = x1097;
int32_t x1104 = x1103 + x1099;
float x1105 = x991[x1104];
float x1106 = x1102 + x1105;
x1093[x1101] = x1106;

}
x1097 += 1;

}

}
for(int x1116=0; x1116 < 20; x1116++) {
for(int x1117=0; x1117 < 26; x1117++) {
int32_t x1118 = x1115;
float x1119 = x893[x1118];
float x1120 = x991[x1118];
float x1121 = x934[x1118];
float x1125 = x1093[x1116];
double x1122 = (double)x1121;
double x1123 = exp(x1122);
float x1124 = (float)x1123;
float x1126 = x1124 * x1125;
float x1127 = x1120 - x1126;
float x1128 = x1119 + x1127;
x893[x1118] = x1128;
x1115 += 1;

}

}
for(int x1135=0; x1135 < 20; x1135++) {
int32_t x1137 = 26 * x1135;
for(int x1136=0; x1136 < 26; x1136++) {
float x1139 = x111[x1136];
int32_t x1138 = x1137 + x1136;
float x1140 = x893[x1138];
float x1141 = x1139 + x1140;
x111[x1136] = x1141;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x893,26,x101,26,1,x890,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x876,50,x893,26,1,x109,26);
// backprop for * op
for(int x1151=0; x1151 < 20; x1151++) {
int32_t x1153 = 50 * x1151;
for(int x1152=0; x1152 < 50; x1152++) {
int32_t x1154 = x1153 + x1152;
float x1155 = x774[x1154];
float x1156 = x762[x1154];
float x1157 = x866[x1154];
float x1158 = x890[x1154];
float x1159 = x1158 * x1157;
float x1160 = x1155 + x1159;
x774[x1154] = x1160;
float x1162 = x875[x1154];
float x1163 = x762[x1154];
float x1164 = x866[x1154];
float x1165 = x890[x1154];
float x1166 = x1165 * x1163;
float x1167 = x1162 + x1166;
x875[x1154] = x1167;

}

}
for(int x1173=0; x1173 < 1000; x1173++) {
float x1174 = x865[x1173];
float x1175 = x866[x1173];
float x1178 = x875[x1173];
float x1176 = x1175 * x1175;
float x1177 = 1.0f - x1176;
float x1179 = x1177 * x1178;
float x1180 = x1174 + x1179;
x865[x1173] = x1180;

}
// back prop for + op
for(int x1185=0; x1185 < 20; x1185++) {
int32_t x1187 = 50 * x1185;
for(int x1186=0; x1186 < 50; x1186++) {
int32_t x1188 = x1187 + x1186;
float x1189 = x835[x1188];
float x1190 = x821[x1188];
float x1191 = x836[x1188];
float x1192 = x865[x1188];
float x1193 = x1189 + x1192;
x835[x1188] = x1193;
float x1195 = x850[x1188];
float x1196 = x821[x1188];
float x1197 = x836[x1188];
float x1198 = x865[x1188];
float x1199 = x1195 + x1198;
x850[x1188] = x1199;

}

}
// backprop for * op
for(int x1206=0; x1206 < 20; x1206++) {
int32_t x1208 = 50 * x1206;
for(int x1207=0; x1207 < 50; x1207++) {
int32_t x1209 = x1208 + x1207;
float x1210 = x725[x1209];
float x1211 = x713[x1209];
float x1212 = x811[x1209];
float x1213 = x850[x1209];
float x1214 = x1213 * x1212;
float x1215 = x1210 + x1214;
x725[x1209] = x1215;
float x1217 = x820[x1209];
float x1218 = x713[x1209];
float x1219 = x811[x1209];
float x1220 = x850[x1209];
float x1221 = x1220 * x1218;
float x1222 = x1217 + x1221;
x820[x1209] = x1222;

}

}
// backprop for * op
for(int x1229=0; x1229 < 20; x1229++) {
int32_t x1231 = 50 * x1229;
for(int x1230=0; x1230 < 50; x1230++) {
int32_t x1232 = x1231 + x1230;
float x1233 = x676[x1232];
float x1234 = x664[x1232];
float x1235 = x622[x1232];
float x1236 = x835[x1232];
float x1237 = x1236 * x1235;
float x1238 = x1233 + x1237;
x676[x1232] = x1238;
float x1240 = x623[x1232];
float x1241 = x664[x1232];
float x1242 = x622[x1232];
float x1243 = x835[x1232];
float x1244 = x1243 * x1241;
float x1245 = x1240 + x1244;
x623[x1232] = x1245;

}

}
for(int x1251=0; x1251 < 1000; x1251++) {
float x1252 = x810[x1251];
float x1253 = x811[x1251];
float x1256 = x820[x1251];
float x1254 = x1253 * x1253;
float x1255 = 1.0f - x1254;
float x1257 = x1255 * x1256;
float x1258 = x1252 + x1257;
x810[x1251] = x1258;

}
// back prop for + op
for(int x1263=0; x1263 < 20; x1263++) {
int32_t x1265 = 50 * x1263;
for(int x1264=0; x1264 < 50; x1264++) {
int32_t x1266 = x1265 + x1264;
float x1267 = x795[x1266];
float x1268 = x781[x1266];
float x1269 = x79[x1264];
float x1270 = x810[x1266];
float x1271 = x1267 + x1270;
x795[x1266] = x1271;
float x1273 = x80[x1264];
float x1274 = x781[x1266];
float x1275 = x79[x1264];
float x1276 = x810[x1266];
float x1277 = x1273 + x1276;
x80[x1264] = x1277;

}

}
// back prop for + op
for(int x1284=0; x1284 < 20; x1284++) {
int32_t x1286 = 50 * x1284;
for(int x1285=0; x1285 < 50; x1285++) {
int32_t x1287 = x1286 + x1285;
float x1288 = x777[x1287];
float x1289 = x775[x1287];
float x1290 = x778[x1287];
float x1291 = x795[x1287];
float x1292 = x1288 + x1291;
x777[x1287] = x1292;
float x1294 = x780[x1287];
float x1295 = x775[x1287];
float x1296 = x778[x1287];
float x1297 = x795[x1287];
float x1298 = x1294 + x1297;
x780[x1287] = x1298;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x780,50,x70,50,1,x621,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x620,50,x780,50,1,x78,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x777,50,x61,50,1,x627,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x626,26,x777,50,1,x69,50);
for(int x1310=0; x1310 < 1000; x1310++) {
float x1311 = x761[x1310];
float x1312 = x762[x1310];
float x1315 = x774[x1310];
float x1313 = 1.0f - x1312;
float x1314 = x1313 * x1312;
float x1316 = x1314 * x1315;
float x1317 = x1311 + x1316;
x761[x1310] = x1317;

}
// back prop for + op
for(int x1322=0; x1322 < 20; x1322++) {
int32_t x1324 = 50 * x1322;
for(int x1323=0; x1323 < 50; x1323++) {
int32_t x1325 = x1324 + x1323;
float x1326 = x746[x1325];
float x1327 = x732[x1325];
float x1328 = x99[x1323];
float x1329 = x761[x1325];
float x1330 = x1326 + x1329;
x746[x1325] = x1330;
float x1332 = x100[x1323];
float x1333 = x732[x1325];
float x1334 = x99[x1323];
float x1335 = x761[x1325];
float x1336 = x1332 + x1335;
x100[x1323] = x1336;

}

}
// back prop for + op
for(int x1343=0; x1343 < 20; x1343++) {
int32_t x1345 = 50 * x1343;
for(int x1344=0; x1344 < 50; x1344++) {
int32_t x1346 = x1345 + x1344;
float x1347 = x728[x1346];
float x1348 = x726[x1346];
float x1349 = x729[x1346];
float x1350 = x746[x1346];
float x1351 = x1347 + x1350;
x728[x1346] = x1351;
float x1353 = x731[x1346];
float x1354 = x726[x1346];
float x1355 = x729[x1346];
float x1356 = x746[x1346];
float x1357 = x1353 + x1356;
x731[x1346] = x1357;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x731,50,x90,50,1,x621,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x620,50,x731,50,1,x98,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x728,50,x81,50,1,x627,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x626,26,x728,50,1,x89,50);
for(int x1369=0; x1369 < 1000; x1369++) {
float x1370 = x712[x1369];
float x1371 = x713[x1369];
float x1374 = x725[x1369];
float x1372 = 1.0f - x1371;
float x1373 = x1372 * x1371;
float x1375 = x1373 * x1374;
float x1376 = x1370 + x1375;
x712[x1369] = x1376;

}
// back prop for + op
for(int x1381=0; x1381 < 20; x1381++) {
int32_t x1383 = 50 * x1381;
for(int x1382=0; x1382 < 50; x1382++) {
int32_t x1384 = x1383 + x1382;
float x1385 = x697[x1384];
float x1386 = x683[x1384];
float x1387 = x59[x1382];
float x1388 = x712[x1384];
float x1389 = x1385 + x1388;
x697[x1384] = x1389;
float x1391 = x60[x1382];
float x1392 = x683[x1384];
float x1393 = x59[x1382];
float x1394 = x712[x1384];
float x1395 = x1391 + x1394;
x60[x1382] = x1395;

}

}
// back prop for + op
for(int x1402=0; x1402 < 20; x1402++) {
int32_t x1404 = 50 * x1402;
for(int x1403=0; x1403 < 50; x1403++) {
int32_t x1405 = x1404 + x1403;
float x1406 = x679[x1405];
float x1407 = x677[x1405];
float x1408 = x680[x1405];
float x1409 = x697[x1405];
float x1410 = x1406 + x1409;
x679[x1405] = x1410;
float x1412 = x682[x1405];
float x1413 = x677[x1405];
float x1414 = x680[x1405];
float x1415 = x697[x1405];
float x1416 = x1412 + x1415;
x682[x1405] = x1416;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x682,50,x50,50,1,x621,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x620,50,x682,50,1,x58,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x679,50,x41,50,1,x627,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x626,26,x679,50,1,x49,50);
for(int x1428=0; x1428 < 1000; x1428++) {
float x1429 = x663[x1428];
float x1430 = x664[x1428];
float x1433 = x676[x1428];
float x1431 = 1.0f - x1430;
float x1432 = x1431 * x1430;
float x1434 = x1432 * x1433;
float x1435 = x1429 + x1434;
x663[x1428] = x1435;

}
// back prop for + op
for(int x1440=0; x1440 < 20; x1440++) {
int32_t x1442 = 50 * x1440;
for(int x1441=0; x1441 < 50; x1441++) {
int32_t x1443 = x1442 + x1441;
float x1444 = x648[x1443];
float x1445 = x634[x1443];
float x1446 = x39[x1441];
float x1447 = x663[x1443];
float x1448 = x1444 + x1447;
x648[x1443] = x1448;
float x1450 = x40[x1441];
float x1451 = x634[x1443];
float x1452 = x39[x1441];
float x1453 = x663[x1443];
float x1454 = x1450 + x1453;
x40[x1441] = x1454;

}

}
// back prop for + op
for(int x1461=0; x1461 < 20; x1461++) {
int32_t x1463 = 50 * x1461;
for(int x1462=0; x1462 < 50; x1462++) {
int32_t x1464 = x1463 + x1462;
float x1465 = x630[x1464];
float x1466 = x628[x1464];
float x1467 = x631[x1464];
float x1468 = x648[x1464];
float x1469 = x1465 + x1468;
x630[x1464] = x1469;
float x1471 = x633[x1464];
float x1472 = x628[x1464];
float x1473 = x631[x1464];
float x1474 = x648[x1464];
float x1475 = x1471 + x1474;
x633[x1464] = x1475;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x633,50,x29,50,1,x621,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x620,50,x633,50,1,x38,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x630,50,x18,50,1,x627,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x626,26,x630,50,1,x28,50);
} else {
float x1488 = 0.0f;
for(int x1489=0; x1489 < 1; x1489++) {
float x1490 = x1488;
float x1491 = x618[x1489];
float x1492 = x1490 + x1491;
x1488 = x1492;

}
float x1496 = x1488;
float* x1497 = (float*)myMalloc(1 * sizeof(float));;
for(int x1498=0; x1498 < 1; x1498++) {
x1497[x1498] = x1496;

}
float* x1502 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x1504=0; x1504 < 1; x1504++) {
x1502[x1504] = 1.0f;

}
// backend is lantern.TensorDslCPU$BackendCPU@35c19b35
for(int x1509=0; x1509 < 1; x1509++) {
float x1510 = x1497[x1509];
x158[x1509] = x1510;

}
// 'sum' gradient.
if (x597) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1519=0; x1519 < 1; x1519++) {
float x1520 = x619[0];
float x1521 = x1502[0];
float x1522 = x1520 + x1521;
x619[0] = x1522;

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
float** x2021 = (float**)myMalloc(6 * sizeof(float*));;
x2021[0] = x177;
x2021[1] = x178;
x2021[2] = x179;
x2021[3] = x180;
x2021[4] = x181;
x2021[5] = x182;
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
float* x221 = (float*)myMalloc(1000 * sizeof(float));;
for(int x222=0; x222 < 20; x222++) {
int32_t x224 = 50 * x222;
for(int x223=0; x223 < 50; x223++) {
int32_t x226 = x224 + x223;
float x227 = x204[x226];
float x228 = x39[x223];
int32_t x225 = x223 + x224;
float x229 = x227 + x228;
x221[x225] = x229;

}

}
float* x235 = (float*)myMalloc(1000 * sizeof(float));;
float* x236 = (float*)myMalloc(1000 * sizeof(float));;
for(int x238=0; x238 < 1000; x238++) {
float x239 = x221[x238];
float x240 = -1.0f * x239;
double x241 = (double)x240;
double x242 = exp(x241);
float x243 = (float)x242;
float x244 = x243 + 1.0f;
float x245 = 1.0f / x244;
x236[x238] = x245;

}
float* x249 = (float*)myMalloc(1000 * sizeof(float));;
float* x250 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x196,26,x41,50,0,x250,50);
float* x252 = (float*)myMalloc(1000 * sizeof(float));;
float* x253 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x190,50,x50,50,0,x253,50);
float* x255 = (float*)myMalloc(1000 * sizeof(float));;
float* x256 = (float*)myMalloc(1000 * sizeof(float));;
for(int x257=0; x257 < 20; x257++) {
int32_t x259 = 50 * x257;
for(int x258=0; x258 < 50; x258++) {
int32_t x261 = x259 + x258;
float x262 = x250[x261];
float x263 = x253[x261];
int32_t x260 = x258 + x259;
float x264 = x262 + x263;
x256[x260] = x264;

}

}
float* x270 = (float*)myMalloc(1000 * sizeof(float));;
float* x271 = (float*)myMalloc(1000 * sizeof(float));;
for(int x272=0; x272 < 20; x272++) {
int32_t x274 = 50 * x272;
for(int x273=0; x273 < 50; x273++) {
int32_t x276 = x274 + x273;
float x277 = x256[x276];
float x278 = x59[x273];
int32_t x275 = x273 + x274;
float x279 = x277 + x278;
x271[x275] = x279;

}

}
float* x285 = (float*)myMalloc(1000 * sizeof(float));;
float* x286 = (float*)myMalloc(1000 * sizeof(float));;
for(int x287=0; x287 < 1000; x287++) {
float x288 = x271[x287];
float x289 = -1.0f * x288;
double x290 = (double)x289;
double x291 = exp(x290);
float x292 = (float)x291;
float x293 = x292 + 1.0f;
float x294 = 1.0f / x293;
x286[x287] = x294;

}
float* x298 = (float*)myMalloc(1000 * sizeof(float));;
float* x299 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x196,26,x81,50,0,x299,50);
float* x301 = (float*)myMalloc(1000 * sizeof(float));;
float* x302 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x190,50,x90,50,0,x302,50);
float* x304 = (float*)myMalloc(1000 * sizeof(float));;
float* x305 = (float*)myMalloc(1000 * sizeof(float));;
for(int x306=0; x306 < 20; x306++) {
int32_t x308 = 50 * x306;
for(int x307=0; x307 < 50; x307++) {
int32_t x310 = x308 + x307;
float x311 = x299[x310];
float x312 = x302[x310];
int32_t x309 = x307 + x308;
float x313 = x311 + x312;
x305[x309] = x313;

}

}
float* x319 = (float*)myMalloc(1000 * sizeof(float));;
float* x320 = (float*)myMalloc(1000 * sizeof(float));;
for(int x321=0; x321 < 20; x321++) {
int32_t x323 = 50 * x321;
for(int x322=0; x322 < 50; x322++) {
int32_t x325 = x323 + x322;
float x326 = x305[x325];
float x327 = x99[x322];
int32_t x324 = x322 + x323;
float x328 = x326 + x327;
x320[x324] = x328;

}

}
float* x334 = (float*)myMalloc(1000 * sizeof(float));;
float* x335 = (float*)myMalloc(1000 * sizeof(float));;
for(int x336=0; x336 < 1000; x336++) {
float x337 = x320[x336];
float x338 = -1.0f * x337;
double x339 = (double)x338;
double x340 = exp(x339);
float x341 = (float)x340;
float x342 = x341 + 1.0f;
float x343 = 1.0f / x342;
x335[x336] = x343;

}
float* x347 = (float*)myMalloc(1000 * sizeof(float));;
float* x348 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x196,26,x61,50,0,x348,50);
float* x350 = (float*)myMalloc(1000 * sizeof(float));;
float* x351 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x190,50,x70,50,0,x351,50);
float* x353 = (float*)myMalloc(1000 * sizeof(float));;
float* x354 = (float*)myMalloc(1000 * sizeof(float));;
for(int x355=0; x355 < 20; x355++) {
int32_t x357 = 50 * x355;
for(int x356=0; x356 < 50; x356++) {
int32_t x359 = x357 + x356;
float x360 = x348[x359];
float x361 = x351[x359];
int32_t x358 = x356 + x357;
float x362 = x360 + x361;
x354[x358] = x362;

}

}
float* x368 = (float*)myMalloc(1000 * sizeof(float));;
float* x369 = (float*)myMalloc(1000 * sizeof(float));;
for(int x370=0; x370 < 20; x370++) {
int32_t x372 = 50 * x370;
for(int x371=0; x371 < 50; x371++) {
int32_t x374 = x372 + x371;
float x375 = x354[x374];
float x376 = x79[x371];
int32_t x373 = x371 + x372;
float x377 = x375 + x376;
x369[x373] = x377;

}

}
float* x383 = (float*)myMalloc(1000 * sizeof(float));;
float* x384 = (float*)myMalloc(1000 * sizeof(float));;
for(int x385=0; x385 < 1000; x385++) {
float x386 = x369[x385];
double x387 = (double)x386;
double x388 = tanh(x387);
float x389 = (float)x388;
x384[x385] = x389;

}
float* x393 = (float*)myMalloc(1000 * sizeof(float));;
float* x394 = (float*)myMalloc(1000 * sizeof(float));;
for(int x395=0; x395 < 20; x395++) {
int32_t x397 = 50 * x395;
for(int x396=0; x396 < 50; x396++) {
int32_t x399 = x397 + x396;
float x400 = x236[x399];
float x401 = x192[x399];
int32_t x398 = x396 + x397;
float x402 = x400 * x401;
x394[x398] = x402;

}

}
float* x408 = (float*)myMalloc(1000 * sizeof(float));;
float* x409 = (float*)myMalloc(1000 * sizeof(float));;
for(int x410=0; x410 < 20; x410++) {
int32_t x412 = 50 * x410;
for(int x411=0; x411 < 50; x411++) {
int32_t x414 = x412 + x411;
float x415 = x286[x414];
float x416 = x384[x414];
int32_t x413 = x411 + x412;
float x417 = x415 * x416;
x409[x413] = x417;

}

}
float* x423 = (float*)myMalloc(1000 * sizeof(float));;
float* x424 = (float*)myMalloc(1000 * sizeof(float));;
for(int x425=0; x425 < 20; x425++) {
int32_t x427 = 50 * x425;
for(int x426=0; x426 < 50; x426++) {
int32_t x429 = x427 + x426;
float x430 = x394[x429];
float x431 = x409[x429];
int32_t x428 = x426 + x427;
float x432 = x430 + x431;
x424[x428] = x432;

}

}
float* x438 = (float*)myMalloc(1000 * sizeof(float));;
float* x439 = (float*)myMalloc(1000 * sizeof(float));;
for(int x440=0; x440 < 1000; x440++) {
float x441 = x424[x440];
double x442 = (double)x441;
double x443 = tanh(x442);
float x444 = (float)x443;
x439[x440] = x444;

}
float* x448 = (float*)myMalloc(1000 * sizeof(float));;
float* x449 = (float*)myMalloc(1000 * sizeof(float));;
for(int x450=0; x450 < 20; x450++) {
int32_t x452 = 50 * x450;
for(int x451=0; x451 < 50; x451++) {
int32_t x454 = x452 + x451;
float x455 = x335[x454];
float x456 = x439[x454];
int32_t x453 = x451 + x452;
float x457 = x455 * x456;
x449[x453] = x457;

}

}
float* x463 = (float*)myMalloc(1000 * sizeof(float));;
float* x464 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x449,50,x101,26,0,x464,26);
float* x466 = (float*)myMalloc(520 * sizeof(float));;
for(int x467=0; x467 < 20; x467++) {
int32_t x470 = 26 * x467;
for(int x469=0; x469 < 26; x469++) {
int32_t x471 = x470 + x469;
float x472 = x464[x471];
float x473 = x110[x469];
float x474 = x472 + x473;
x464[x471] = x474;

}

}
int* x480 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x481=0; x481 < 20; x481++) {
int32_t x482 = x481 * 20;
int32_t x483 = x186 + x482;
int32_t x484 = x143[x483];
x480[x481] = x484;

}
float* x488 = (float*)myMalloc(20 * sizeof(float));;
int32_t x489 = 0;
for(int x490=0; x490 < 20; x490++) {
float x491 = -3.4028235E38f;
for(int x492=0; x492 < 26; x492++) {
int32_t x493 = x489;
float x494 = x464[x493];
float x495 = x491;
bool x496 = x494 > x495;
if (x496) {
float x497 = x464[x493];
x491 = x497;
} else {
}
x489 += 1;

}
float x504 = x491;
x488[x490] = x504;

}
float* x508 = (float*)myMalloc(520 * sizeof(float));;
int32_t x509 = 0;
for(int x510=0; x510 < 20; x510++) {
for(int x511=0; x511 < 26; x511++) {
int32_t x512 = x509;
float x513 = x464[x512];
float x514 = x488[x510];
float x515 = x513 - x514;
double x516 = (double)x515;
double x517 = exp(x516);
float x518 = (float)x517;
x508[x512] = x518;
x509 += 1;

}

}
float* x525 = (float*)myMalloc(20 * sizeof(float));;
for(int x526=0; x526 < 20; x526++) {
int32_t x527 = x526;
int32_t x528 = x526 * 26;
int32_t x529 = x528;
for(int x530=0; x530 < 26; x530++) {
for(int x532=0; x532 < 1; x532++) {
int32_t x533 = x527;
int32_t x534 = x533 + x532;
float x535 = x525[x534];
int32_t x536 = x529;
int32_t x537 = x536 + x532;
float x538 = x508[x537];
float x539 = x535 + x538;
x525[x534] = x539;

}
x529 += 1;

}

}
x509 = 0;
for(int x549=0; x549 < 20; x549++) {
float x550 = x488[x549];
float x551 = x525[x549];
double x552 = (double)x551;
double x553 = log(x552);
float x554 = (float)x553;
float x555 = x550 + x554;
for(int x556=0; x556 < 26; x556++) {
int32_t x557 = x509;
float x558 = x464[x557];
float x559 = x558 - x555;
x508[x557] = x559;
x509 += 1;

}

}
float* x566 = (float*)myMalloc(520 * sizeof(float));;
// nllLoss forward in CPU
float* x568 = (float*)myMalloc(20 * sizeof(float));;
int32_t x569 = 0;
for(int x570=0; x570 < 20; x570++) {
int32_t x571 = x569;
int32_t x572 = x480[x570];
int32_t x573 = x571 + x572;
float x574 = x508[x573];
float x575 = -1.0f * x574;
x568[x570] = x575;
x569 += 26;

}
float* x580 = (float*)myMalloc(20 * sizeof(float));;
float x581 = 0.0f;
for(int x582=0; x582 < 20; x582++) {
float x583 = x581;
float x584 = x568[x582];
float x585 = x583 + x584;
x581 = x585;

}
float x589 = x581;
float* x590 = (float*)myMalloc(1 * sizeof(float));;
for(int x591=0; x591 < 1; x591++) {
x590[x591] = x589;

}
float* x595 = (float*)myMalloc(1 * sizeof(float));;
if (x597) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x603 = (float*)myMalloc(1 * sizeof(float));;
for(int x604=0; x604 < 1; x604++) {
float x605 = x188[0];
float x606 = x590[0];
float x607 = x605 + x606;
x603[x604] = x607;

}
float* x611 = (float*)myMalloc(1 * sizeof(float));;
float** x1529 = (float**)myMalloc(6 * sizeof(float*));;
x1529[0] = x603;
x1529[1] = x611;
x1529[2] = x449;
x1529[3] = x463;
x1529[4] = x424;
x1529[5] = x438;
int32_t x612 = x186 + 1;
x613(x612,x1529);
// back prop for + op
if (x597) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1543=0; x1543 < 1; x1543++) {
float x1544 = x189[0];
float x1545 = x188[0];
float x1546 = x590[0];
float x1547 = x611[x1543];
float x1548 = x1544 + x1547;
x189[0] = x1548;
float x1550 = x595[0];
float x1551 = x188[0];
float x1552 = x590[0];
float x1553 = x611[x1543];
float x1554 = x1550 + x1553;
x595[0] = x1554;

}
// 'sum' gradient.
if (x1065) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",20,1);
assert(false && "");
}
for(int x1563=0; x1563 < 20; x1563++) {
float x1564 = x580[x1563];
float x1565 = x595[0];
float x1566 = x1564 + x1565;
x580[x1563] = x1566;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x1572 = 0;
for(int x1573=0; x1573 < 20; x1573++) {
int32_t x1574 = x1572;
int32_t x1575 = x480[x1573];
int32_t x1576 = x1574 + x1575;
float x1577 = x566[x1576];
float x1578 = x580[x1573];
float x1579 = -1.0f * x1578;
float x1580 = x1577 + x1579;
x566[x1576] = x1580;
x1572 += 26;

}
float* x1585 = (float*)myMalloc(20 * sizeof(float));;
for(int x1586=0; x1586 < 20; x1586++) {
int32_t x1587 = x1586;
int32_t x1588 = x1586 * 26;
int32_t x1589 = x1588;
for(int x1590=0; x1590 < 26; x1590++) {
for(int x1591=0; x1591 < 1; x1591++) {
int32_t x1592 = x1587;
int32_t x1593 = x1592 + x1591;
float x1594 = x1585[x1593];
int32_t x1595 = x1589;
int32_t x1596 = x1595 + x1591;
float x1597 = x566[x1596];
float x1598 = x1594 + x1597;
x1585[x1593] = x1598;

}
x1589 += 1;

}

}
int32_t x1607 = 0;
for(int x1608=0; x1608 < 20; x1608++) {
for(int x1609=0; x1609 < 26; x1609++) {
int32_t x1610 = x1607;
float x1611 = x466[x1610];
float x1612 = x566[x1610];
float x1613 = x508[x1610];
float x1617 = x1585[x1608];
double x1614 = (double)x1613;
double x1615 = exp(x1614);
float x1616 = (float)x1615;
float x1618 = x1616 * x1617;
float x1619 = x1612 - x1618;
float x1620 = x1611 + x1619;
x466[x1610] = x1620;
x1607 += 1;

}

}
for(int x1627=0; x1627 < 20; x1627++) {
int32_t x1629 = 26 * x1627;
for(int x1628=0; x1628 < 26; x1628++) {
float x1631 = x111[x1628];
int32_t x1630 = x1629 + x1628;
float x1632 = x466[x1630];
float x1633 = x1631 + x1632;
x111[x1628] = x1633;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x466,26,x101,26,1,x463,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x449,50,x466,26,1,x109,26);
// backprop for * op
for(int x1643=0; x1643 < 20; x1643++) {
int32_t x1645 = 50 * x1643;
for(int x1644=0; x1644 < 50; x1644++) {
int32_t x1646 = x1645 + x1644;
float x1647 = x347[x1646];
float x1648 = x335[x1646];
float x1649 = x439[x1646];
float x1650 = x463[x1646];
float x1651 = x1650 * x1649;
float x1652 = x1647 + x1651;
x347[x1646] = x1652;
float x1654 = x448[x1646];
float x1655 = x335[x1646];
float x1656 = x439[x1646];
float x1657 = x463[x1646];
float x1658 = x1657 * x1655;
float x1659 = x1654 + x1658;
x448[x1646] = x1659;

}

}
for(int x1665=0; x1665 < 1000; x1665++) {
float x1666 = x438[x1665];
float x1667 = x439[x1665];
float x1670 = x448[x1665];
float x1668 = x1667 * x1667;
float x1669 = 1.0f - x1668;
float x1671 = x1669 * x1670;
float x1672 = x1666 + x1671;
x438[x1665] = x1672;

}
// back prop for + op
for(int x1677=0; x1677 < 20; x1677++) {
int32_t x1679 = 50 * x1677;
for(int x1678=0; x1678 < 50; x1678++) {
int32_t x1680 = x1679 + x1678;
float x1681 = x408[x1680];
float x1682 = x394[x1680];
float x1683 = x409[x1680];
float x1684 = x438[x1680];
float x1685 = x1681 + x1684;
x408[x1680] = x1685;
float x1687 = x423[x1680];
float x1688 = x394[x1680];
float x1689 = x409[x1680];
float x1690 = x438[x1680];
float x1691 = x1687 + x1690;
x423[x1680] = x1691;

}

}
// backprop for * op
for(int x1698=0; x1698 < 20; x1698++) {
int32_t x1700 = 50 * x1698;
for(int x1699=0; x1699 < 50; x1699++) {
int32_t x1701 = x1700 + x1699;
float x1702 = x298[x1701];
float x1703 = x286[x1701];
float x1704 = x384[x1701];
float x1705 = x423[x1701];
float x1706 = x1705 * x1704;
float x1707 = x1702 + x1706;
x298[x1701] = x1707;
float x1709 = x393[x1701];
float x1710 = x286[x1701];
float x1711 = x384[x1701];
float x1712 = x423[x1701];
float x1713 = x1712 * x1710;
float x1714 = x1709 + x1713;
x393[x1701] = x1714;

}

}
// backprop for * op
for(int x1721=0; x1721 < 20; x1721++) {
int32_t x1723 = 50 * x1721;
for(int x1722=0; x1722 < 50; x1722++) {
int32_t x1724 = x1723 + x1722;
float x1725 = x249[x1724];
float x1726 = x236[x1724];
float x1727 = x192[x1724];
float x1728 = x408[x1724];
float x1729 = x1728 * x1727;
float x1730 = x1725 + x1729;
x249[x1724] = x1730;
float x1732 = x193[x1724];
float x1733 = x236[x1724];
float x1734 = x192[x1724];
float x1735 = x408[x1724];
float x1736 = x1735 * x1733;
float x1737 = x1732 + x1736;
x193[x1724] = x1737;

}

}
for(int x1743=0; x1743 < 1000; x1743++) {
float x1744 = x383[x1743];
float x1745 = x384[x1743];
float x1748 = x393[x1743];
float x1746 = x1745 * x1745;
float x1747 = 1.0f - x1746;
float x1749 = x1747 * x1748;
float x1750 = x1744 + x1749;
x383[x1743] = x1750;

}
// back prop for + op
for(int x1755=0; x1755 < 20; x1755++) {
int32_t x1757 = 50 * x1755;
for(int x1756=0; x1756 < 50; x1756++) {
int32_t x1758 = x1757 + x1756;
float x1759 = x368[x1758];
float x1760 = x354[x1758];
float x1761 = x79[x1756];
float x1762 = x383[x1758];
float x1763 = x1759 + x1762;
x368[x1758] = x1763;
float x1765 = x80[x1756];
float x1766 = x354[x1758];
float x1767 = x79[x1756];
float x1768 = x383[x1758];
float x1769 = x1765 + x1768;
x80[x1756] = x1769;

}

}
// back prop for + op
for(int x1776=0; x1776 < 20; x1776++) {
int32_t x1778 = 50 * x1776;
for(int x1777=0; x1777 < 50; x1777++) {
int32_t x1779 = x1778 + x1777;
float x1780 = x350[x1779];
float x1781 = x348[x1779];
float x1782 = x351[x1779];
float x1783 = x368[x1779];
float x1784 = x1780 + x1783;
x350[x1779] = x1784;
float x1786 = x353[x1779];
float x1787 = x348[x1779];
float x1788 = x351[x1779];
float x1789 = x368[x1779];
float x1790 = x1786 + x1789;
x353[x1779] = x1790;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x353,50,x70,50,1,x191,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x190,50,x353,50,1,x78,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x350,50,x61,50,1,x197,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x196,26,x350,50,1,x69,50);
for(int x1802=0; x1802 < 1000; x1802++) {
float x1803 = x334[x1802];
float x1804 = x335[x1802];
float x1807 = x347[x1802];
float x1805 = 1.0f - x1804;
float x1806 = x1805 * x1804;
float x1808 = x1806 * x1807;
float x1809 = x1803 + x1808;
x334[x1802] = x1809;

}
// back prop for + op
for(int x1814=0; x1814 < 20; x1814++) {
int32_t x1816 = 50 * x1814;
for(int x1815=0; x1815 < 50; x1815++) {
int32_t x1817 = x1816 + x1815;
float x1818 = x319[x1817];
float x1819 = x305[x1817];
float x1820 = x99[x1815];
float x1821 = x334[x1817];
float x1822 = x1818 + x1821;
x319[x1817] = x1822;
float x1824 = x100[x1815];
float x1825 = x305[x1817];
float x1826 = x99[x1815];
float x1827 = x334[x1817];
float x1828 = x1824 + x1827;
x100[x1815] = x1828;

}

}
// back prop for + op
for(int x1835=0; x1835 < 20; x1835++) {
int32_t x1837 = 50 * x1835;
for(int x1836=0; x1836 < 50; x1836++) {
int32_t x1838 = x1837 + x1836;
float x1839 = x301[x1838];
float x1840 = x299[x1838];
float x1841 = x302[x1838];
float x1842 = x319[x1838];
float x1843 = x1839 + x1842;
x301[x1838] = x1843;
float x1845 = x304[x1838];
float x1846 = x299[x1838];
float x1847 = x302[x1838];
float x1848 = x319[x1838];
float x1849 = x1845 + x1848;
x304[x1838] = x1849;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x304,50,x90,50,1,x191,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x190,50,x304,50,1,x98,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x301,50,x81,50,1,x197,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x196,26,x301,50,1,x89,50);
for(int x1861=0; x1861 < 1000; x1861++) {
float x1862 = x285[x1861];
float x1863 = x286[x1861];
float x1866 = x298[x1861];
float x1864 = 1.0f - x1863;
float x1865 = x1864 * x1863;
float x1867 = x1865 * x1866;
float x1868 = x1862 + x1867;
x285[x1861] = x1868;

}
// back prop for + op
for(int x1873=0; x1873 < 20; x1873++) {
int32_t x1875 = 50 * x1873;
for(int x1874=0; x1874 < 50; x1874++) {
int32_t x1876 = x1875 + x1874;
float x1877 = x270[x1876];
float x1878 = x256[x1876];
float x1879 = x59[x1874];
float x1880 = x285[x1876];
float x1881 = x1877 + x1880;
x270[x1876] = x1881;
float x1883 = x60[x1874];
float x1884 = x256[x1876];
float x1885 = x59[x1874];
float x1886 = x285[x1876];
float x1887 = x1883 + x1886;
x60[x1874] = x1887;

}

}
// back prop for + op
for(int x1894=0; x1894 < 20; x1894++) {
int32_t x1896 = 50 * x1894;
for(int x1895=0; x1895 < 50; x1895++) {
int32_t x1897 = x1896 + x1895;
float x1898 = x252[x1897];
float x1899 = x250[x1897];
float x1900 = x253[x1897];
float x1901 = x270[x1897];
float x1902 = x1898 + x1901;
x252[x1897] = x1902;
float x1904 = x255[x1897];
float x1905 = x250[x1897];
float x1906 = x253[x1897];
float x1907 = x270[x1897];
float x1908 = x1904 + x1907;
x255[x1897] = x1908;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x255,50,x50,50,1,x191,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x190,50,x255,50,1,x58,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x252,50,x41,50,1,x197,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x196,26,x252,50,1,x49,50);
for(int x1920=0; x1920 < 1000; x1920++) {
float x1921 = x235[x1920];
float x1922 = x236[x1920];
float x1925 = x249[x1920];
float x1923 = 1.0f - x1922;
float x1924 = x1923 * x1922;
float x1926 = x1924 * x1925;
float x1927 = x1921 + x1926;
x235[x1920] = x1927;

}
// back prop for + op
for(int x1932=0; x1932 < 20; x1932++) {
int32_t x1934 = 50 * x1932;
for(int x1933=0; x1933 < 50; x1933++) {
int32_t x1935 = x1934 + x1933;
float x1936 = x219[x1935];
float x1937 = x204[x1935];
float x1938 = x39[x1933];
float x1939 = x235[x1935];
float x1940 = x1936 + x1939;
x219[x1935] = x1940;
float x1942 = x40[x1933];
float x1943 = x204[x1935];
float x1944 = x39[x1933];
float x1945 = x235[x1935];
float x1946 = x1942 + x1945;
x40[x1933] = x1946;

}

}
// back prop for + op
for(int x1953=0; x1953 < 20; x1953++) {
int32_t x1955 = 50 * x1953;
for(int x1954=0; x1954 < 50; x1954++) {
int32_t x1956 = x1955 + x1954;
float x1957 = x200[x1956];
float x1958 = x198[x1956];
float x1959 = x201[x1956];
float x1960 = x219[x1956];
float x1961 = x1957 + x1960;
x200[x1956] = x1961;
float x1963 = x203[x1956];
float x1964 = x198[x1956];
float x1965 = x201[x1956];
float x1966 = x219[x1956];
float x1967 = x1963 + x1966;
x203[x1956] = x1967;

}

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x203,50,x29,50,1,x191,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x190,50,x203,50,1,x38,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x200,50,x18,50,1,x197,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x196,26,x200,50,1,x28,50);
} else {
float x1980 = 0.0f;
for(int x1981=0; x1981 < 1; x1981++) {
float x1982 = x1980;
float x1983 = x188[x1981];
float x1984 = x1982 + x1983;
x1980 = x1984;

}
float x1988 = x1980;
float* x1989 = (float*)myMalloc(1 * sizeof(float));;
for(int x1990=0; x1990 < 1; x1990++) {
x1989[x1990] = x1988;

}
float* x1994 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x1996=0; x1996 < 1; x1996++) {
x1994[x1996] = 1.0f;

}
// backend is lantern.TensorDslCPU$BackendCPU@35c19b35
for(int x2001=0; x2001 < 1; x2001++) {
float x2002 = x1989[x2001];
x158[x2001] = x2002;

}
// 'sum' gradient.
if (x597) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x2011=0; x2011 < 1; x2011++) {
float x2012 = x189[0];
float x2013 = x1994[0];
float x2014 = x2012 + x2013;
x189[0] = x2014;

}
}
};
x183(0,x2021);
float x2030 = x158[0];
int32_t x2031 = x133 % 100;
bool x2032 = x2031 == 0;
if (x2032) {
printf("iter %d, loss %f\n",x133,x2030);
int32_t x2034 = x133 / 100;
double x2035 = (double)x2030;
x127[x2034] = x2035;
} else {
}
for(int x2039=0; x2039 < 1300; x2039++) {
float x2040 = x49[x2039];
float x2041 = x2040;
float x2042 = x2041;
bool x2043 = x2042 > 5.0f;
if (x2043) {
x2041 = 5.0f;
} else {
}
float x2047 = x2041;
bool x2048 = x2047 < -5.0f;
if (x2048) {
x2041 = -5.0f;
} else {
}
float x2052 = x112[x2039];
float x2053 = x2041;
float x2054 = x2053 * x2053;
float x2055 = x2052 + x2054;
x112[x2039] = x2055;
float x2057 = x41[x2039];
float x2059 = x112[x2039];
float x2058 = 0.1f * x2053;
double x2060 = (double)x2059;
double x2061 = x2060 + 9.99999993922529E-9;
double x2062 = sqrt(x2061);
float x2063 = (float)x2062;
float x2064 = x2058 / x2063;
float x2065 = x2057 - x2064;
x41[x2039] = x2065;
x49[x2039] = 0.0f;

}
for(int x2070=0; x2070 < 50; x2070++) {
float x2071 = x60[x2070];
float x2072 = x2071;
float x2073 = x2072;
bool x2074 = x2073 > 5.0f;
if (x2074) {
x2072 = 5.0f;
} else {
}
float x2078 = x2072;
bool x2079 = x2078 < -5.0f;
if (x2079) {
x2072 = -5.0f;
} else {
}
float x2083 = x113[x2070];
float x2084 = x2072;
float x2085 = x2084 * x2084;
float x2086 = x2083 + x2085;
x113[x2070] = x2086;
float x2088 = x59[x2070];
float x2090 = x113[x2070];
float x2089 = 0.1f * x2084;
double x2091 = (double)x2090;
double x2092 = x2091 + 9.99999993922529E-9;
double x2093 = sqrt(x2092);
float x2094 = (float)x2093;
float x2095 = x2089 / x2094;
float x2096 = x2088 - x2095;
x59[x2070] = x2096;
x60[x2070] = 0.0f;

}
for(int x2101=0; x2101 < 2500; x2101++) {
float x2102 = x58[x2101];
float x2103 = x2102;
float x2104 = x2103;
bool x2105 = x2104 > 5.0f;
if (x2105) {
x2103 = 5.0f;
} else {
}
float x2109 = x2103;
bool x2110 = x2109 < -5.0f;
if (x2110) {
x2103 = -5.0f;
} else {
}
float x2114 = x114[x2101];
float x2115 = x2103;
float x2116 = x2115 * x2115;
float x2117 = x2114 + x2116;
x114[x2101] = x2117;
float x2119 = x50[x2101];
float x2121 = x114[x2101];
float x2120 = 0.1f * x2115;
double x2122 = (double)x2121;
double x2123 = x2122 + 9.99999993922529E-9;
double x2124 = sqrt(x2123);
float x2125 = (float)x2124;
float x2126 = x2120 / x2125;
float x2127 = x2119 - x2126;
x50[x2101] = x2127;
x58[x2101] = 0.0f;

}
for(int x2132=0; x2132 < 50; x2132++) {
float x2133 = x40[x2132];
float x2134 = x2133;
float x2135 = x2134;
bool x2136 = x2135 > 5.0f;
if (x2136) {
x2134 = 5.0f;
} else {
}
float x2140 = x2134;
bool x2141 = x2140 < -5.0f;
if (x2141) {
x2134 = -5.0f;
} else {
}
float x2145 = x115[x2132];
float x2146 = x2134;
float x2147 = x2146 * x2146;
float x2148 = x2145 + x2147;
x115[x2132] = x2148;
float x2150 = x39[x2132];
float x2152 = x115[x2132];
float x2151 = 0.1f * x2146;
double x2153 = (double)x2152;
double x2154 = x2153 + 9.99999993922529E-9;
double x2155 = sqrt(x2154);
float x2156 = (float)x2155;
float x2157 = x2151 / x2156;
float x2158 = x2150 - x2157;
x39[x2132] = x2158;
x40[x2132] = 0.0f;

}
for(int x2163=0; x2163 < 2500; x2163++) {
float x2164 = x38[x2163];
float x2165 = x2164;
float x2166 = x2165;
bool x2167 = x2166 > 5.0f;
if (x2167) {
x2165 = 5.0f;
} else {
}
float x2171 = x2165;
bool x2172 = x2171 < -5.0f;
if (x2172) {
x2165 = -5.0f;
} else {
}
float x2176 = x116[x2163];
float x2177 = x2165;
float x2178 = x2177 * x2177;
float x2179 = x2176 + x2178;
x116[x2163] = x2179;
float x2181 = x29[x2163];
float x2183 = x116[x2163];
float x2182 = 0.1f * x2177;
double x2184 = (double)x2183;
double x2185 = x2184 + 9.99999993922529E-9;
double x2186 = sqrt(x2185);
float x2187 = (float)x2186;
float x2188 = x2182 / x2187;
float x2189 = x2181 - x2188;
x29[x2163] = x2189;
x38[x2163] = 0.0f;

}
for(int x2194=0; x2194 < 1300; x2194++) {
float x2195 = x28[x2194];
float x2196 = x2195;
float x2197 = x2196;
bool x2198 = x2197 > 5.0f;
if (x2198) {
x2196 = 5.0f;
} else {
}
float x2202 = x2196;
bool x2203 = x2202 < -5.0f;
if (x2203) {
x2196 = -5.0f;
} else {
}
float x2207 = x117[x2194];
float x2208 = x2196;
float x2209 = x2208 * x2208;
float x2210 = x2207 + x2209;
x117[x2194] = x2210;
float x2212 = x18[x2194];
float x2214 = x117[x2194];
float x2213 = 0.1f * x2208;
double x2215 = (double)x2214;
double x2216 = x2215 + 9.99999993922529E-9;
double x2217 = sqrt(x2216);
float x2218 = (float)x2217;
float x2219 = x2213 / x2218;
float x2220 = x2212 - x2219;
x18[x2194] = x2220;
x28[x2194] = 0.0f;

}
for(int x2225=0; x2225 < 1300; x2225++) {
float x2226 = x69[x2225];
float x2227 = x2226;
float x2228 = x2227;
bool x2229 = x2228 > 5.0f;
if (x2229) {
x2227 = 5.0f;
} else {
}
float x2233 = x2227;
bool x2234 = x2233 < -5.0f;
if (x2234) {
x2227 = -5.0f;
} else {
}
float x2238 = x118[x2225];
float x2239 = x2227;
float x2240 = x2239 * x2239;
float x2241 = x2238 + x2240;
x118[x2225] = x2241;
float x2243 = x61[x2225];
float x2245 = x118[x2225];
float x2244 = 0.1f * x2239;
double x2246 = (double)x2245;
double x2247 = x2246 + 9.99999993922529E-9;
double x2248 = sqrt(x2247);
float x2249 = (float)x2248;
float x2250 = x2244 / x2249;
float x2251 = x2243 - x2250;
x61[x2225] = x2251;
x69[x2225] = 0.0f;

}
for(int x2256=0; x2256 < 50; x2256++) {
float x2257 = x80[x2256];
float x2258 = x2257;
float x2259 = x2258;
bool x2260 = x2259 > 5.0f;
if (x2260) {
x2258 = 5.0f;
} else {
}
float x2264 = x2258;
bool x2265 = x2264 < -5.0f;
if (x2265) {
x2258 = -5.0f;
} else {
}
float x2269 = x119[x2256];
float x2270 = x2258;
float x2271 = x2270 * x2270;
float x2272 = x2269 + x2271;
x119[x2256] = x2272;
float x2274 = x79[x2256];
float x2276 = x119[x2256];
float x2275 = 0.1f * x2270;
double x2277 = (double)x2276;
double x2278 = x2277 + 9.99999993922529E-9;
double x2279 = sqrt(x2278);
float x2280 = (float)x2279;
float x2281 = x2275 / x2280;
float x2282 = x2274 - x2281;
x79[x2256] = x2282;
x80[x2256] = 0.0f;

}
for(int x2287=0; x2287 < 2500; x2287++) {
float x2288 = x78[x2287];
float x2289 = x2288;
float x2290 = x2289;
bool x2291 = x2290 > 5.0f;
if (x2291) {
x2289 = 5.0f;
} else {
}
float x2295 = x2289;
bool x2296 = x2295 < -5.0f;
if (x2296) {
x2289 = -5.0f;
} else {
}
float x2300 = x120[x2287];
float x2301 = x2289;
float x2302 = x2301 * x2301;
float x2303 = x2300 + x2302;
x120[x2287] = x2303;
float x2305 = x70[x2287];
float x2307 = x120[x2287];
float x2306 = 0.1f * x2301;
double x2308 = (double)x2307;
double x2309 = x2308 + 9.99999993922529E-9;
double x2310 = sqrt(x2309);
float x2311 = (float)x2310;
float x2312 = x2306 / x2311;
float x2313 = x2305 - x2312;
x70[x2287] = x2313;
x78[x2287] = 0.0f;

}
for(int x2318=0; x2318 < 26; x2318++) {
float x2319 = x111[x2318];
float x2320 = x2319;
float x2321 = x2320;
bool x2322 = x2321 > 5.0f;
if (x2322) {
x2320 = 5.0f;
} else {
}
float x2326 = x2320;
bool x2327 = x2326 < -5.0f;
if (x2327) {
x2320 = -5.0f;
} else {
}
float x2331 = x121[x2318];
float x2332 = x2320;
float x2333 = x2332 * x2332;
float x2334 = x2331 + x2333;
x121[x2318] = x2334;
float x2336 = x110[x2318];
float x2338 = x121[x2318];
float x2337 = 0.1f * x2332;
double x2339 = (double)x2338;
double x2340 = x2339 + 9.99999993922529E-9;
double x2341 = sqrt(x2340);
float x2342 = (float)x2341;
float x2343 = x2337 / x2342;
float x2344 = x2336 - x2343;
x110[x2318] = x2344;
x111[x2318] = 0.0f;

}
for(int x2349=0; x2349 < 1300; x2349++) {
float x2350 = x109[x2349];
float x2351 = x2350;
float x2352 = x2351;
bool x2353 = x2352 > 5.0f;
if (x2353) {
x2351 = 5.0f;
} else {
}
float x2357 = x2351;
bool x2358 = x2357 < -5.0f;
if (x2358) {
x2351 = -5.0f;
} else {
}
float x2362 = x122[x2349];
float x2363 = x2351;
float x2364 = x2363 * x2363;
float x2365 = x2362 + x2364;
x122[x2349] = x2365;
float x2367 = x101[x2349];
float x2369 = x122[x2349];
float x2368 = 0.1f * x2363;
double x2370 = (double)x2369;
double x2371 = x2370 + 9.99999993922529E-9;
double x2372 = sqrt(x2371);
float x2373 = (float)x2372;
float x2374 = x2368 / x2373;
float x2375 = x2367 - x2374;
x101[x2349] = x2375;
x109[x2349] = 0.0f;

}
for(int x2380=0; x2380 < 2500; x2380++) {
float x2381 = x98[x2380];
float x2382 = x2381;
float x2383 = x2382;
bool x2384 = x2383 > 5.0f;
if (x2384) {
x2382 = 5.0f;
} else {
}
float x2388 = x2382;
bool x2389 = x2388 < -5.0f;
if (x2389) {
x2382 = -5.0f;
} else {
}
float x2393 = x123[x2380];
float x2394 = x2382;
float x2395 = x2394 * x2394;
float x2396 = x2393 + x2395;
x123[x2380] = x2396;
float x2398 = x90[x2380];
float x2400 = x123[x2380];
float x2399 = 0.1f * x2394;
double x2401 = (double)x2400;
double x2402 = x2401 + 9.99999993922529E-9;
double x2403 = sqrt(x2402);
float x2404 = (float)x2403;
float x2405 = x2399 / x2404;
float x2406 = x2398 - x2405;
x90[x2380] = x2406;
x98[x2380] = 0.0f;

}
for(int x2411=0; x2411 < 1300; x2411++) {
float x2412 = x89[x2411];
float x2413 = x2412;
float x2414 = x2413;
bool x2415 = x2414 > 5.0f;
if (x2415) {
x2413 = 5.0f;
} else {
}
float x2419 = x2413;
bool x2420 = x2419 < -5.0f;
if (x2420) {
x2413 = -5.0f;
} else {
}
float x2424 = x124[x2411];
float x2425 = x2413;
float x2426 = x2425 * x2425;
float x2427 = x2424 + x2426;
x124[x2411] = x2427;
float x2429 = x81[x2411];
float x2431 = x124[x2411];
float x2430 = 0.1f * x2425;
double x2432 = (double)x2431;
double x2433 = x2432 + 9.99999993922529E-9;
double x2434 = sqrt(x2433);
float x2435 = (float)x2434;
float x2436 = x2430 / x2435;
float x2437 = x2429 - x2436;
x81[x2411] = x2437;
x89[x2411] = 0.0f;

}
for(int x2442=0; x2442 < 50; x2442++) {
float x2443 = x100[x2442];
float x2444 = x2443;
float x2445 = x2444;
bool x2446 = x2445 > 5.0f;
if (x2446) {
x2444 = 5.0f;
} else {
}
float x2450 = x2444;
bool x2451 = x2450 < -5.0f;
if (x2451) {
x2444 = -5.0f;
} else {
}
float x2455 = x125[x2442];
float x2456 = x2444;
float x2457 = x2456 * x2456;
float x2458 = x2455 + x2457;
x125[x2442] = x2458;
float x2460 = x99[x2442];
float x2462 = x125[x2442];
float x2461 = 0.1f * x2456;
double x2463 = (double)x2462;
double x2464 = x2463 + 9.99999993922529E-9;
double x2465 = sqrt(x2464);
float x2466 = (float)x2465;
float x2467 = x2461 / x2466;
float x2468 = x2460 - x2467;
x99[x2442] = x2468;
x100[x2442] = 0.0f;

}
int64_t x2473 = (long)mallocAddr;
int64_t x2474 = x2473 - x128;
memset((void*)x128, 0, x2474);
mallocAddr = (void*)x128;

}
double x2479 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x2482 = (long)fopen(x0, "w");
fprintf((FILE *)x2482, "unit: %s\n", "100 iteration");
for(int x2485=0; x2485 < 51; x2485++) {
double x2486 = x127[x2485];
fprintf((FILE *)x2482, "%lf\n", x2486);

}
double x2480 = x126 - x2;
double x2481 = x2479 - x126;
fprintf((FILE *)x2482, "run time: %lf %lf\n", x2480, x2481);
fclose((FILE*)x2482);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

