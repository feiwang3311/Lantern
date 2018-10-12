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
void *mallocBase = malloc(HEAP_SIZE);
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
for(int x24=0; x24 < 1300; x24++) {
x23[x24] = 0.0f;

}
float* x28 = (float*)myMalloc(2500 * sizeof(float));;
for(int x30=0; x30 < 2500; x30++) {
float x31 = (float)rand()/RAND_MAX;
float x32 = x31 - 0.5f;
float x33 = x32 * 0.14142136f;
x28[x30] = x33;

}
float* x37 = (float*)myMalloc(2500 * sizeof(float));;
for(int x38=0; x38 < 2500; x38++) {
x37[x38] = 0.0f;

}
float* x42 = (float*)myMalloc(50 * sizeof(float));;
for(int x44=0; x44 < 50; x44++) {
x42[x44] = 0.0f;

}
float* x48 = (float*)myMalloc(50 * sizeof(float));;
for(int x49=0; x49 < 50; x49++) {
x48[x49] = 0.0f;

}
float* x53 = (float*)myMalloc(1300 * sizeof(float));;
for(int x54=0; x54 < 1300; x54++) {
float x55 = (float)rand()/RAND_MAX;
float x56 = x55 - 0.5f;
float x57 = x56 * 0.14142136f;
x53[x54] = x57;

}
float* x61 = (float*)myMalloc(1300 * sizeof(float));;
for(int x62=0; x62 < 1300; x62++) {
x61[x62] = 0.0f;

}
float* x66 = (float*)myMalloc(26 * sizeof(float));;
for(int x68=0; x68 < 26; x68++) {
x66[x68] = 0.0f;

}
float* x72 = (float*)myMalloc(26 * sizeof(float));;
for(int x73=0; x73 < 26; x73++) {
x72[x73] = 0.0f;

}
float* x77 = (float*)myMalloc(26 * sizeof(float));;
for(int x78=0; x78 < 26; x78++) {
x77[x78] = 0.0f;

}
float* x82 = (float*)myMalloc(1300 * sizeof(float));;
for(int x83=0; x83 < 1300; x83++) {
x82[x83] = 0.0f;

}
float* x87 = (float*)myMalloc(2500 * sizeof(float));;
for(int x88=0; x88 < 2500; x88++) {
x87[x88] = 0.0f;

}
float* x92 = (float*)myMalloc(1300 * sizeof(float));;
for(int x93=0; x93 < 1300; x93++) {
x92[x93] = 0.0f;

}
float* x97 = (float*)myMalloc(50 * sizeof(float));;
for(int x98=0; x98 < 50; x98++) {
x97[x98] = 0.0f;

}
double* x102 = (double*)myMalloc(51 * sizeof(double));;
double x103 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x104 = (long)mallocAddr;
int32_t x105 = 0;
x105 -= 400;
for(int x108=0; x108 < 5001; x108++) {
float* x141 = (float*)myMalloc(1 * sizeof(float));;
float* x146 = (float*)myMalloc(10400 * sizeof(float));;
float* x168 = (float*)myMalloc(10400 * sizeof(float));;
int* x118 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
function<void(int32_t,float**)> x552 = [&](int32_t x553,float** x554) {
float** x556 = x554;
float* x557 = x556[0];
float* x558 = x556[1];
float* x559 = x556[2];
float* x560 = x556[3];
int32_t x555 = x553;
bool x561 = x555 < 20;
if (x561) {
int32_t x562 = x555 * 520;
float* x563 = x146+x562;
float* x564 = x168+x562;
// dot: WrappedArray(20, 26), WrappedArray(26, 50)
float* x566 = (float*)myMalloc(1000 * sizeof(float));;
for(int x567=0; x567 < 20; x567++) {
int32_t x571 = x567 * 26;
int32_t x581 = x567 * 50;
for(int x568=0; x568 < 50; x568++) {
float x569 = 0.0f;
for(int x570=0; x570 < 26; x570++) {
int32_t x572 = x571 + x570;
float x573 = x563[x572];
int32_t x574 = x570 * 50;
int32_t x575 = x574 + x568;
float x576 = x14[x575];
float x577 = x573 * x576;
x569 += x577;

}
float x583 = x569;
int32_t x582 = x581 + x568;
x566[x582] = x583;

}

}
float* x589 = (float*)myMalloc(1000 * sizeof(float));;
for(int x590=0; x590 < 1000; x590++) {
x589[x590] = 0.0f;

}
// dot: List(20, 50), WrappedArray(50, 50)
float* x595 = (float*)myMalloc(1000 * sizeof(float));;
for(int x596=0; x596 < 20; x596++) {
int32_t x600 = x596 * 50;
for(int x597=0; x597 < 50; x597++) {
float x598 = 0.0f;
for(int x599=0; x599 < 50; x599++) {
int32_t x601 = x600 + x599;
float x602 = x559[x601];
int32_t x603 = x599 * 50;
int32_t x604 = x603 + x597;
float x605 = x28[x604];
float x606 = x602 * x605;
x598 += x606;

}
float x611 = x598;
int32_t x610 = x600 + x597;
x595[x610] = x611;

}

}
float* x617 = (float*)myMalloc(1000 * sizeof(float));;
for(int x618=0; x618 < 1000; x618++) {
x617[x618] = 0.0f;

}
float* x622 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x623 = 0;
int32_t x624 = 0;
int32_t x625 = 0;
for(int x626=0; x626 < 20; x626++) {
int32_t x627 = x624;
int32_t x628 = x625;
int32_t x629 = x623;
int32_t x630 = x629;
int32_t x631 = x627;
int32_t x632 = x628;
for(int x633=0; x633 < 50; x633++) {
int32_t x634 = x630;
int32_t x635 = x631;
float x636 = x566[x635];
int32_t x637 = x632;
float x638 = x595[x637];
float x639 = x636 + x638;
x622[x634] = x639;
x630 += 1;
x631 += 1;
x632 += 1;

}
x623 += 50;
x624 += 50;
x625 += 50;

}
float* x651 = (float*)myMalloc(1000 * sizeof(float));;
for(int x652=0; x652 < 1000; x652++) {
x651[x652] = 0.0f;

}
float* x656 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x657 = 0;
int32_t x658 = 0;
int32_t x659 = 0;
for(int x660=0; x660 < 20; x660++) {
int32_t x661 = x658;
int32_t x662 = x659;
int32_t x663 = x657;
int32_t x664 = x663;
int32_t x665 = x661;
int32_t x666 = x662;
for(int x667=0; x667 < 50; x667++) {
int32_t x668 = x664;
int32_t x669 = x665;
float x670 = x622[x669];
int32_t x671 = x666;
float x672 = x42[x671];
float x673 = x670 + x672;
x656[x668] = x673;
x664 += 1;
x665 += 1;
x666 += 1;

}
x657 += 50;
x658 += 50;

}
float* x684 = (float*)myMalloc(1000 * sizeof(float));;
for(int x685=0; x685 < 1000; x685++) {
x684[x685] = 0.0f;

}
float* x689 = (float*)myMalloc(1000 * sizeof(float));;
for(int x690=0; x690 < 1000; x690++) {
float x691 = x656[x690];
double x692 = (double)x691;
double x693 = tanh(x692);
float x694 = (float)x693;
x689[x690] = x694;

}
float* x698 = (float*)myMalloc(1000 * sizeof(float));;
for(int x699=0; x699 < 1000; x699++) {
x698[x699] = 0.0f;

}
// dot: List(20, 50), WrappedArray(50, 26)
float* x704 = (float*)myMalloc(520 * sizeof(float));;
for(int x705=0; x705 < 20; x705++) {
int32_t x709 = x705 * 50;
int32_t x719 = x705 * 26;
for(int x706=0; x706 < 26; x706++) {
float x707 = 0.0f;
for(int x708=0; x708 < 50; x708++) {
int32_t x710 = x709 + x708;
float x711 = x689[x710];
int32_t x712 = x708 * 26;
int32_t x713 = x712 + x706;
float x714 = x53[x713];
float x715 = x711 * x714;
x707 += x715;

}
float x721 = x707;
int32_t x720 = x719 + x706;
x704[x720] = x721;

}

}
float* x727 = (float*)myMalloc(520 * sizeof(float));;
for(int x728=0; x728 < 520; x728++) {
x727[x728] = 0.0f;

}
float* x732 = (float*)myMalloc(520 * sizeof(float));;
int32_t x733 = 0;
int32_t x734 = 0;
int32_t x735 = 0;
for(int x736=0; x736 < 20; x736++) {
int32_t x737 = x734;
int32_t x738 = x735;
int32_t x739 = x733;
int32_t x740 = x739;
int32_t x741 = x737;
int32_t x742 = x738;
for(int x743=0; x743 < 26; x743++) {
int32_t x744 = x740;
int32_t x745 = x741;
float x746 = x704[x745];
int32_t x747 = x742;
float x748 = x66[x747];
float x749 = x746 + x748;
x732[x744] = x749;
x740 += 1;
x741 += 1;
x742 += 1;

}
x733 += 26;
x734 += 26;

}
float* x760 = (float*)myMalloc(520 * sizeof(float));;
for(int x761=0; x761 < 520; x761++) {
x760[x761] = 0.0f;

}
int* x765 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x766=0; x766 < 20; x766++) {
int32_t x767 = x766 * 20;
int32_t x768 = x555 + x767;
int32_t x769 = x118[x768];
x765[x766] = x769;

}
float* x773 = (float*)myMalloc(20 * sizeof(float));;
int32_t x774 = 0;
for(int x775=0; x775 < 20; x775++) {
float x776 = -3.4028235E38f;
for(int x777=0; x777 < 26; x777++) {
int32_t x778 = x774;
float x779 = x732[x778];
float x780 = x776;
bool x781 = x779 > x780;
if (x781) {
float x782 = x732[x778];
x776 = x782;
} else {
}
x774 += 1;

}
float x789 = x776;
x773[x775] = x789;

}
float* x793 = (float*)myMalloc(520 * sizeof(float));;
for(int x794=0; x794 < 520; x794++) {
x793[x794] = 0.0f;

}
int32_t x798 = 0;
for(int x799=0; x799 < 20; x799++) {
for(int x800=0; x800 < 26; x800++) {
int32_t x801 = x798;
float x802 = x732[x801];
float x803 = x773[x799];
float x804 = x802 - x803;
double x805 = (double)x804;
double x806 = exp(x805);
float x807 = (float)x806;
x793[x801] = x807;
x798 += 1;

}

}
float* x814 = (float*)myMalloc(20 * sizeof(float));;
for(int x815=0; x815 < 20; x815++) {
x814[x815] = 0.0f;

}
for(int x819=0; x819 < 20; x819++) {
int32_t x820 = x819;
int32_t x821 = x819 * 26;
int32_t x822 = x821;
for(int x823=0; x823 < 26; x823++) {
int32_t x824 = x820;
float x825 = x814[x824];
int32_t x826 = x822;
float x827 = x793[x826];
float x828 = x825 + x827;
x814[x824] = x828;
x822 += 1;

}

}
x798 = 0;
for(int x836=0; x836 < 20; x836++) {
float x837 = x773[x836];
float x838 = x814[x836];
double x839 = (double)x838;
double x840 = log(x839);
float x841 = (float)x840;
float x842 = x837 + x841;
for(int x843=0; x843 < 26; x843++) {
int32_t x844 = x798;
float x845 = x732[x844];
float x846 = x845 - x842;
x793[x844] = x846;
x798 += 1;

}

}
float* x853 = (float*)myMalloc(520 * sizeof(float));;
for(int x854=0; x854 < 520; x854++) {
x853[x854] = 0.0f;

}
float* x858 = (float*)myMalloc(20 * sizeof(float));;
int32_t x859 = 0;
for(int x860=0; x860 < 20; x860++) {
int32_t x861 = x859;
int32_t x862 = x765[x860];
int32_t x863 = x861 + x862;
float x864 = x793[x863];
float x865 = -1.0f * x864;
x858[x860] = x865;
x859 += 26;

}
float* x870 = (float*)myMalloc(20 * sizeof(float));;
for(int x871=0; x871 < 20; x871++) {
x870[x871] = 0.0f;

}
float x875 = 0.0f;
for(int x876=0; x876 < 20; x876++) {
float x877 = x875;
float x878 = x858[x876];
float x879 = x877 + x878;
x875 = x879;

}
float x883 = x875;
float* x884 = (float*)myMalloc(1 * sizeof(float));;
x884[0] = x883;
float* x886 = (float*)myMalloc(1 * sizeof(float));;
for(int x887=0; x887 < 1; x887++) {
x886[x887] = 0.0f;

}
float* x891 = (float*)myMalloc(1 * sizeof(float));;
int32_t x892 = 0;
int32_t x893 = 0;
int32_t x894 = 0;
int32_t x895 = x892;
int32_t x896 = x893;
float x897 = x557[x896];
int32_t x898 = x894;
float x899 = x884[x898];
float x900 = x897 + x899;
x891[x895] = x900;
x892 += 1;
float* x903 = (float*)myMalloc(1 * sizeof(float));;
for(int x904=0; x904 < 1; x904++) {
x903[x904] = 0.0f;

}
float** x909 = (float**)myMalloc(4 * sizeof(float*));;
x909[0] = x891;
x909[1] = x903;
x909[2] = x689;
x909[3] = x698;
int32_t x916 = 0;
int32_t x917 = 0;
int32_t x918 = 0;
int32_t x919 = x916;
int32_t x922 = x917;
int32_t x924 = x918;
x918 += 1;
int32_t x943 = 0;
float* x956 = (float*)myMalloc(20 * sizeof(float));;
int32_t x977 = 0;
int32_t x997 = 0;
int32_t x998 = 0;
int32_t x999 = 0;
int32_t x1070 = 0;
int32_t x1071 = 0;
int32_t x1072 = 0;
int32_t x1105 = 0;
int32_t x1106 = 0;
int32_t x1107 = 0;
int32_t x908 = x555 + 1;
x552(x908,x909);
float x920 = x558[x919];
float x921 = x557[x919];
float x923 = x884[x922];
float x925 = x903[x924];
float x926 = x920 + x925;
x558[x919] = x926;
float x928 = x886[x922];
float x929 = x557[x919];
float x930 = x884[x922];
float x931 = x903[x924];
float x932 = x928 + x931;
x886[x922] = x932;
// += tensor of dim 0
float x936 = x886[0];
for(int x937=0; x937 < 20; x937++) {
float x938 = x870[x937];
float x939 = x938 + x936;
x870[x937] = x939;

}
for(int x944=0; x944 < 20; x944++) {
int32_t x945 = x943;
int32_t x946 = x765[x944];
int32_t x947 = x945 + x946;
float x948 = x853[x947];
float x949 = x870[x944];
float x950 = -1.0f * x949;
float x951 = x948 + x950;
x853[x947] = x951;
x943 += 26;

}
for(int x957=0; x957 < 20; x957++) {
x956[x957] = 0.0f;

}
for(int x961=0; x961 < 20; x961++) {
int32_t x962 = x961;
int32_t x963 = x961 * 26;
int32_t x964 = x963;
for(int x965=0; x965 < 26; x965++) {
int32_t x966 = x962;
float x967 = x956[x966];
int32_t x968 = x964;
float x969 = x853[x968];
float x970 = x967 + x969;
x956[x966] = x970;
x964 += 1;

}

}
for(int x978=0; x978 < 20; x978++) {
for(int x979=0; x979 < 26; x979++) {
int32_t x980 = x977;
float x981 = x760[x980];
float x982 = x853[x980];
float x983 = x793[x980];
float x987 = x956[x978];
double x984 = (double)x983;
double x985 = exp(x984);
float x986 = (float)x985;
float x988 = x986 * x987;
float x989 = x982 - x988;
float x990 = x981 + x989;
x760[x980] = x990;
x977 += 1;

}

}
for(int x1000=0; x1000 < 20; x1000++) {
int32_t x1001 = x997;
int32_t x1002 = x998;
int32_t x1003 = x999;
int32_t x1004 = x1001;
int32_t x1005 = x1002;
int32_t x1006 = x1003;
for(int x1007=0; x1007 < 26; x1007++) {
int32_t x1008 = x1004;
float x1009 = x727[x1008];
float x1010 = x704[x1008];
int32_t x1011 = x1005;
float x1012 = x66[x1011];
int32_t x1013 = x1006;
float x1014 = x760[x1013];
float x1015 = x1009 + x1014;
x727[x1008] = x1015;
float x1017 = x72[x1011];
float x1018 = x704[x1008];
float x1019 = x66[x1011];
float x1020 = x760[x1013];
float x1021 = x1017 + x1020;
x72[x1011] = x1021;
x1006 += 1;
x1004 += 1;
x1005 += 1;

}
x999 += 26;
x997 += 26;

}
for(int x1032=0; x1032 < 20; x1032++) {
int32_t x1035 = x1032 * 50;
int32_t x1041 = x1032 * 26;
for(int x1033=0; x1033 < 26; x1033++) {
int32_t x1042 = x1041 + x1033;
for(int x1034=0; x1034 < 50; x1034++) {
int32_t x1036 = x1035 + x1034;
float x1037 = x698[x1036];
int32_t x1038 = x1034 * 26;
int32_t x1039 = x1038 + x1033;
float x1040 = x53[x1039];
float x1043 = x727[x1042];
float x1044 = x1040 * x1043;
float x1045 = x1037 + x1044;
x698[x1036] = x1045;
float x1047 = x61[x1039];
float x1048 = x689[x1036];
float x1049 = x727[x1042];
float x1050 = x1048 * x1049;
float x1051 = x1047 + x1050;
x61[x1039] = x1051;

}

}

}
for(int x1059=0; x1059 < 1000; x1059++) {
float x1060 = x684[x1059];
float x1061 = x689[x1059];
float x1064 = x698[x1059];
float x1062 = x1061 * x1061;
float x1063 = 1.0f - x1062;
float x1065 = x1063 * x1064;
float x1066 = x1060 + x1065;
x684[x1059] = x1066;

}
for(int x1073=0; x1073 < 20; x1073++) {
int32_t x1074 = x1070;
int32_t x1075 = x1071;
int32_t x1076 = x1072;
int32_t x1077 = x1074;
int32_t x1078 = x1075;
int32_t x1079 = x1076;
for(int x1080=0; x1080 < 50; x1080++) {
int32_t x1081 = x1077;
float x1082 = x651[x1081];
float x1083 = x622[x1081];
int32_t x1084 = x1078;
float x1085 = x42[x1084];
int32_t x1086 = x1079;
float x1087 = x684[x1086];
float x1088 = x1082 + x1087;
x651[x1081] = x1088;
float x1090 = x48[x1084];
float x1091 = x622[x1081];
float x1092 = x42[x1084];
float x1093 = x684[x1086];
float x1094 = x1090 + x1093;
x48[x1084] = x1094;
x1079 += 1;
x1077 += 1;
x1078 += 1;

}
x1072 += 50;
x1070 += 50;

}
for(int x1108=0; x1108 < 20; x1108++) {
int32_t x1109 = x1105;
int32_t x1110 = x1106;
int32_t x1111 = x1107;
int32_t x1112 = x1109;
int32_t x1113 = x1110;
int32_t x1114 = x1111;
for(int x1115=0; x1115 < 50; x1115++) {
int32_t x1116 = x1112;
float x1117 = x589[x1116];
float x1118 = x566[x1116];
int32_t x1119 = x1113;
float x1120 = x595[x1119];
int32_t x1121 = x1114;
float x1122 = x651[x1121];
float x1123 = x1117 + x1122;
x589[x1116] = x1123;
float x1125 = x617[x1119];
float x1126 = x566[x1116];
float x1127 = x595[x1119];
float x1128 = x651[x1121];
float x1129 = x1125 + x1128;
x617[x1119] = x1129;
x1114 += 1;
x1112 += 1;
x1113 += 1;

}
x1107 += 50;
x1105 += 50;
x1106 += 50;

}
for(int x1141=0; x1141 < 20; x1141++) {
int32_t x1144 = x1141 * 50;
for(int x1142=0; x1142 < 50; x1142++) {
int32_t x1150 = x1144 + x1142;
for(int x1143=0; x1143 < 50; x1143++) {
int32_t x1145 = x1144 + x1143;
float x1146 = x560[x1145];
int32_t x1147 = x1143 * 50;
int32_t x1148 = x1147 + x1142;
float x1149 = x28[x1148];
float x1151 = x617[x1150];
float x1152 = x1149 * x1151;
float x1153 = x1146 + x1152;
x560[x1145] = x1153;
float x1155 = x37[x1148];
float x1156 = x559[x1145];
float x1157 = x617[x1150];
float x1158 = x1156 * x1157;
float x1159 = x1155 + x1158;
x37[x1148] = x1159;

}

}

}
for(int x1167=0; x1167 < 20; x1167++) {
int32_t x1170 = x1167 * 26;
int32_t x1176 = x1167 * 50;
for(int x1168=0; x1168 < 50; x1168++) {
int32_t x1177 = x1176 + x1168;
for(int x1169=0; x1169 < 26; x1169++) {
int32_t x1171 = x1170 + x1169;
float x1172 = x564[x1171];
int32_t x1173 = x1169 * 50;
int32_t x1174 = x1173 + x1168;
float x1175 = x14[x1174];
float x1178 = x589[x1177];
float x1179 = x1175 * x1178;
float x1180 = x1172 + x1179;
x564[x1171] = x1180;
float x1182 = x23[x1174];
float x1183 = x563[x1171];
float x1184 = x589[x1177];
float x1185 = x1183 * x1184;
float x1186 = x1182 + x1185;
x23[x1174] = x1186;

}

}

}
} else {
float x1195 = 0.0f;
float x1196 = x1195;
float x1197 = x557[0];
float x1198 = x1196 + x1197;
x1195 = x1198;
float x1200 = x1195;
float* x1201 = (float*)myMalloc(1 * sizeof(float));;
x1201[0] = x1200;
float* x1203 = (float*)myMalloc(1 * sizeof(float));;
for(int x1204=0; x1204 < 1; x1204++) {
x1203[x1204] = 0.0f;

}
float x1208 = x1203[0];
x1203[0] = 1.0f;
float x1210 = x1201[0];
x141[0] = x1210;
// += tensor of dim 0
float x1213 = x1203[0];
float x1214 = x558[0];
float x1215 = x1214 + x1213;
x558[0] = x1215;
}
};
x105 += 400;
int32_t x110 = x105;
int32_t x111 = x110 + 400;
int32_t x112 = x111 + 1;
bool x113 = x112 >= x3;
if (x113) {
x105 = 0;
} else {
}
int* x117 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
for(int x120=0; x120 < 400; x120++) {
int32_t x121 = x105;
int32_t x122 = x121 + x120;
int32_t x123 = x5[x122];
x117[x120] = x123;
int32_t x125 = x122 + 1;
int32_t x126 = x5[x125];
x118[x120] = x126;

}
float* x130 = (float*)myMalloc(1 * sizeof(float));;
for(int x132=0; x132 < 1; x132++) {
x130[x132] = 0.0f;

}
float* x136 = (float*)myMalloc(1 * sizeof(float));;
for(int x137=0; x137 < 1; x137++) {
x136[x137] = 0.0f;

}
for(int x142=0; x142 < 1; x142++) {
x141[x142] = 0.0f;

}
for(int x148=0; x148 < 10400; x148++) {
x146[x148] = 0.0f;

}
for(int x153=0; x153 < 20; x153++) {
int32_t x155 = x153 * 26;
int32_t x156 = x155 * 20;
for(int x154=0; x154 < 20; x154++) {
int32_t x159 = x154 * 20;
int32_t x160 = x159 + x153;
int32_t x161 = x117[x160];
int32_t x157 = x154 * 26;
int32_t x158 = x156 + x157;
int32_t x162 = x158 + x161;
x146[x162] = 1.0f;

}

}
for(int x169=0; x169 < 10400; x169++) {
x168[x169] = 0.0f;

}
float* x173 = (float*)myMalloc(1 * sizeof(float));;
for(int x174=0; x174 < 1; x174++) {
x173[x174] = 0.0f;

}
float* x178 = (float*)myMalloc(1 * sizeof(float));;
for(int x179=0; x179 < 1; x179++) {
x178[x179] = 0.0f;

}
float* x183 = (float*)myMalloc(1000 * sizeof(float));;
for(int x185=0; x185 < 1000; x185++) {
x183[x185] = 0.0f;

}
float* x189 = (float*)myMalloc(1000 * sizeof(float));;
for(int x190=0; x190 < 1000; x190++) {
x189[x190] = 0.0f;

}
float** x1531 = (float**)myMalloc(4 * sizeof(float*));;
x1531[0] = x173;
x1531[1] = x178;
x1531[2] = x183;
x1531[3] = x189;
function<void(int32_t,float**)> x194 = [&](int32_t x195,float** x196) {
float** x198 = x196;
float* x199 = x198[0];
float* x200 = x198[1];
float* x201 = x198[2];
float* x202 = x198[3];
int32_t x197 = x195;
bool x203 = x197 < 20;
if (x203) {
int32_t x204 = x197 * 520;
float* x205 = x146+x204;
float* x206 = x168+x204;
// dot: WrappedArray(20, 26), WrappedArray(26, 50)
float* x208 = (float*)myMalloc(1000 * sizeof(float));;
for(int x209=0; x209 < 20; x209++) {
int32_t x213 = x209 * 26;
int32_t x223 = x209 * 50;
for(int x210=0; x210 < 50; x210++) {
float x211 = 0.0f;
for(int x212=0; x212 < 26; x212++) {
int32_t x214 = x213 + x212;
float x215 = x205[x214];
int32_t x216 = x212 * 50;
int32_t x217 = x216 + x210;
float x218 = x14[x217];
float x219 = x215 * x218;
x211 += x219;

}
float x225 = x211;
int32_t x224 = x223 + x210;
x208[x224] = x225;

}

}
float* x231 = (float*)myMalloc(1000 * sizeof(float));;
for(int x232=0; x232 < 1000; x232++) {
x231[x232] = 0.0f;

}
// dot: WrappedArray(20, 50), WrappedArray(50, 50)
float* x237 = (float*)myMalloc(1000 * sizeof(float));;
for(int x238=0; x238 < 20; x238++) {
int32_t x242 = x238 * 50;
for(int x239=0; x239 < 50; x239++) {
float x240 = 0.0f;
for(int x241=0; x241 < 50; x241++) {
int32_t x243 = x242 + x241;
float x244 = x201[x243];
int32_t x245 = x241 * 50;
int32_t x246 = x245 + x239;
float x247 = x28[x246];
float x248 = x244 * x247;
x240 += x248;

}
float x253 = x240;
int32_t x252 = x242 + x239;
x237[x252] = x253;

}

}
float* x259 = (float*)myMalloc(1000 * sizeof(float));;
for(int x260=0; x260 < 1000; x260++) {
x259[x260] = 0.0f;

}
float* x264 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x265 = 0;
int32_t x266 = 0;
int32_t x267 = 0;
for(int x268=0; x268 < 20; x268++) {
int32_t x269 = x266;
int32_t x270 = x267;
int32_t x271 = x265;
int32_t x272 = x271;
int32_t x273 = x269;
int32_t x274 = x270;
for(int x275=0; x275 < 50; x275++) {
int32_t x276 = x272;
int32_t x277 = x273;
float x278 = x208[x277];
int32_t x279 = x274;
float x280 = x237[x279];
float x281 = x278 + x280;
x264[x276] = x281;
x272 += 1;
x273 += 1;
x274 += 1;

}
x265 += 50;
x266 += 50;
x267 += 50;

}
float* x293 = (float*)myMalloc(1000 * sizeof(float));;
for(int x294=0; x294 < 1000; x294++) {
x293[x294] = 0.0f;

}
float* x298 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x299 = 0;
int32_t x300 = 0;
int32_t x301 = 0;
for(int x302=0; x302 < 20; x302++) {
int32_t x303 = x300;
int32_t x304 = x301;
int32_t x305 = x299;
int32_t x306 = x305;
int32_t x307 = x303;
int32_t x308 = x304;
for(int x309=0; x309 < 50; x309++) {
int32_t x310 = x306;
int32_t x311 = x307;
float x312 = x264[x311];
int32_t x313 = x308;
float x314 = x42[x313];
float x315 = x312 + x314;
x298[x310] = x315;
x306 += 1;
x307 += 1;
x308 += 1;

}
x299 += 50;
x300 += 50;

}
float* x326 = (float*)myMalloc(1000 * sizeof(float));;
for(int x327=0; x327 < 1000; x327++) {
x326[x327] = 0.0f;

}
float* x331 = (float*)myMalloc(1000 * sizeof(float));;
for(int x332=0; x332 < 1000; x332++) {
float x333 = x298[x332];
double x334 = (double)x333;
double x335 = tanh(x334);
float x336 = (float)x335;
x331[x332] = x336;

}
float* x340 = (float*)myMalloc(1000 * sizeof(float));;
for(int x341=0; x341 < 1000; x341++) {
x340[x341] = 0.0f;

}
// dot: List(20, 50), WrappedArray(50, 26)
float* x346 = (float*)myMalloc(520 * sizeof(float));;
for(int x347=0; x347 < 20; x347++) {
int32_t x351 = x347 * 50;
int32_t x361 = x347 * 26;
for(int x348=0; x348 < 26; x348++) {
float x349 = 0.0f;
for(int x350=0; x350 < 50; x350++) {
int32_t x352 = x351 + x350;
float x353 = x331[x352];
int32_t x354 = x350 * 26;
int32_t x355 = x354 + x348;
float x356 = x53[x355];
float x357 = x353 * x356;
x349 += x357;

}
float x363 = x349;
int32_t x362 = x361 + x348;
x346[x362] = x363;

}

}
float* x369 = (float*)myMalloc(520 * sizeof(float));;
for(int x371=0; x371 < 520; x371++) {
x369[x371] = 0.0f;

}
float* x375 = (float*)myMalloc(520 * sizeof(float));;
int32_t x376 = 0;
int32_t x377 = 0;
int32_t x378 = 0;
for(int x379=0; x379 < 20; x379++) {
int32_t x380 = x377;
int32_t x381 = x378;
int32_t x382 = x376;
int32_t x383 = x382;
int32_t x384 = x380;
int32_t x385 = x381;
for(int x386=0; x386 < 26; x386++) {
int32_t x387 = x383;
int32_t x388 = x384;
float x389 = x346[x388];
int32_t x390 = x385;
float x391 = x66[x390];
float x392 = x389 + x391;
x375[x387] = x392;
x383 += 1;
x384 += 1;
x385 += 1;

}
x376 += 26;
x377 += 26;

}
float* x403 = (float*)myMalloc(520 * sizeof(float));;
for(int x404=0; x404 < 520; x404++) {
x403[x404] = 0.0f;

}
int* x408 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x409=0; x409 < 20; x409++) {
int32_t x410 = x409 * 20;
int32_t x411 = x197 + x410;
int32_t x412 = x118[x411];
x408[x409] = x412;

}
float* x416 = (float*)myMalloc(20 * sizeof(float));;
int32_t x417 = 0;
for(int x418=0; x418 < 20; x418++) {
float x419 = -3.4028235E38f;
for(int x420=0; x420 < 26; x420++) {
int32_t x421 = x417;
float x422 = x375[x421];
float x423 = x419;
bool x424 = x422 > x423;
if (x424) {
float x425 = x375[x421];
x419 = x425;
} else {
}
x417 += 1;

}
float x432 = x419;
x416[x418] = x432;

}
float* x436 = (float*)myMalloc(520 * sizeof(float));;
for(int x437=0; x437 < 520; x437++) {
x436[x437] = 0.0f;

}
int32_t x441 = 0;
for(int x442=0; x442 < 20; x442++) {
for(int x443=0; x443 < 26; x443++) {
int32_t x444 = x441;
float x445 = x375[x444];
float x446 = x416[x442];
float x447 = x445 - x446;
double x448 = (double)x447;
double x449 = exp(x448);
float x450 = (float)x449;
x436[x444] = x450;
x441 += 1;

}

}
float* x457 = (float*)myMalloc(20 * sizeof(float));;
for(int x458=0; x458 < 20; x458++) {
x457[x458] = 0.0f;

}
for(int x462=0; x462 < 20; x462++) {
int32_t x463 = x462;
int32_t x464 = x462 * 26;
int32_t x465 = x464;
for(int x466=0; x466 < 26; x466++) {
int32_t x467 = x463;
float x468 = x457[x467];
int32_t x469 = x465;
float x470 = x436[x469];
float x471 = x468 + x470;
x457[x467] = x471;
x465 += 1;

}

}
x441 = 0;
for(int x479=0; x479 < 20; x479++) {
float x480 = x416[x479];
float x481 = x457[x479];
double x482 = (double)x481;
double x483 = log(x482);
float x484 = (float)x483;
float x485 = x480 + x484;
for(int x486=0; x486 < 26; x486++) {
int32_t x487 = x441;
float x488 = x375[x487];
float x489 = x488 - x485;
x436[x487] = x489;
x441 += 1;

}

}
float* x496 = (float*)myMalloc(520 * sizeof(float));;
for(int x497=0; x497 < 520; x497++) {
x496[x497] = 0.0f;

}
float* x501 = (float*)myMalloc(20 * sizeof(float));;
int32_t x502 = 0;
for(int x503=0; x503 < 20; x503++) {
int32_t x504 = x502;
int32_t x505 = x408[x503];
int32_t x506 = x504 + x505;
float x507 = x436[x506];
float x508 = -1.0f * x507;
x501[x503] = x508;
x502 += 26;

}
float* x513 = (float*)myMalloc(20 * sizeof(float));;
for(int x514=0; x514 < 20; x514++) {
x513[x514] = 0.0f;

}
float x518 = 0.0f;
for(int x519=0; x519 < 20; x519++) {
float x520 = x518;
float x521 = x501[x519];
float x522 = x520 + x521;
x518 = x522;

}
float x526 = x518;
float* x527 = (float*)myMalloc(1 * sizeof(float));;
x527[0] = x526;
float* x529 = (float*)myMalloc(1 * sizeof(float));;
for(int x530=0; x530 < 1; x530++) {
x529[x530] = 0.0f;

}
float* x534 = (float*)myMalloc(1 * sizeof(float));;
int32_t x535 = 0;
int32_t x536 = 0;
int32_t x537 = 0;
int32_t x538 = x535;
int32_t x539 = x536;
float x540 = x199[x539];
int32_t x541 = x537;
float x542 = x527[x541];
float x543 = x540 + x542;
x534[x538] = x543;
x535 += 1;
float* x546 = (float*)myMalloc(1 * sizeof(float));;
for(int x547=0; x547 < 1; x547++) {
x546[x547] = 0.0f;

}
float** x1220 = (float**)myMalloc(4 * sizeof(float*));;
x1220[0] = x534;
x1220[1] = x546;
x1220[2] = x331;
x1220[3] = x340;
int32_t x551 = x197 + 1;
x552(x551,x1220);
int32_t x1227 = 0;
int32_t x1228 = 0;
int32_t x1229 = 0;
int32_t x1230 = x1227;
float x1231 = x200[x1230];
float x1232 = x199[x1230];
int32_t x1233 = x1228;
float x1234 = x527[x1233];
int32_t x1235 = x1229;
float x1236 = x546[x1235];
float x1237 = x1231 + x1236;
x200[x1230] = x1237;
float x1239 = x529[x1233];
float x1240 = x199[x1230];
float x1241 = x527[x1233];
float x1242 = x546[x1235];
float x1243 = x1239 + x1242;
x529[x1233] = x1243;
x1229 += 1;
// += tensor of dim 0
float x1247 = x529[0];
for(int x1248=0; x1248 < 20; x1248++) {
float x1249 = x513[x1248];
float x1250 = x1249 + x1247;
x513[x1248] = x1250;

}
int32_t x1254 = 0;
for(int x1255=0; x1255 < 20; x1255++) {
int32_t x1256 = x1254;
int32_t x1257 = x408[x1255];
int32_t x1258 = x1256 + x1257;
float x1259 = x496[x1258];
float x1260 = x513[x1255];
float x1261 = -1.0f * x1260;
float x1262 = x1259 + x1261;
x496[x1258] = x1262;
x1254 += 26;

}
float* x1267 = (float*)myMalloc(20 * sizeof(float));;
for(int x1268=0; x1268 < 20; x1268++) {
x1267[x1268] = 0.0f;

}
for(int x1272=0; x1272 < 20; x1272++) {
int32_t x1273 = x1272;
int32_t x1274 = x1272 * 26;
int32_t x1275 = x1274;
for(int x1276=0; x1276 < 26; x1276++) {
int32_t x1277 = x1273;
float x1278 = x1267[x1277];
int32_t x1279 = x1275;
float x1280 = x496[x1279];
float x1281 = x1278 + x1280;
x1267[x1277] = x1281;
x1275 += 1;

}

}
int32_t x1288 = 0;
for(int x1289=0; x1289 < 20; x1289++) {
for(int x1290=0; x1290 < 26; x1290++) {
int32_t x1291 = x1288;
float x1292 = x403[x1291];
float x1293 = x496[x1291];
float x1294 = x436[x1291];
float x1298 = x1267[x1289];
double x1295 = (double)x1294;
double x1296 = exp(x1295);
float x1297 = (float)x1296;
float x1299 = x1297 * x1298;
float x1300 = x1293 - x1299;
float x1301 = x1292 + x1300;
x403[x1291] = x1301;
x1288 += 1;

}

}
int32_t x1308 = 0;
int32_t x1309 = 0;
int32_t x1310 = 0;
for(int x1311=0; x1311 < 20; x1311++) {
int32_t x1312 = x1308;
int32_t x1313 = x1309;
int32_t x1314 = x1310;
int32_t x1315 = x1312;
int32_t x1316 = x1313;
int32_t x1317 = x1314;
for(int x1318=0; x1318 < 26; x1318++) {
int32_t x1319 = x1315;
float x1320 = x369[x1319];
float x1321 = x346[x1319];
int32_t x1322 = x1316;
float x1323 = x66[x1322];
int32_t x1324 = x1317;
float x1325 = x403[x1324];
float x1326 = x1320 + x1325;
x369[x1319] = x1326;
float x1328 = x72[x1322];
float x1329 = x346[x1319];
float x1330 = x66[x1322];
float x1331 = x403[x1324];
float x1332 = x1328 + x1331;
x72[x1322] = x1332;
x1317 += 1;
x1315 += 1;
x1316 += 1;

}
x1310 += 26;
x1308 += 26;

}
for(int x1343=0; x1343 < 20; x1343++) {
int32_t x1346 = x1343 * 50;
int32_t x1352 = x1343 * 26;
for(int x1344=0; x1344 < 26; x1344++) {
int32_t x1353 = x1352 + x1344;
for(int x1345=0; x1345 < 50; x1345++) {
int32_t x1347 = x1346 + x1345;
float x1348 = x340[x1347];
int32_t x1349 = x1345 * 26;
int32_t x1350 = x1349 + x1344;
float x1351 = x53[x1350];
float x1354 = x369[x1353];
float x1355 = x1351 * x1354;
float x1356 = x1348 + x1355;
x340[x1347] = x1356;
float x1358 = x61[x1350];
float x1359 = x331[x1347];
float x1360 = x369[x1353];
float x1361 = x1359 * x1360;
float x1362 = x1358 + x1361;
x61[x1350] = x1362;

}

}

}
for(int x1370=0; x1370 < 1000; x1370++) {
float x1371 = x326[x1370];
float x1372 = x331[x1370];
float x1375 = x340[x1370];
float x1373 = x1372 * x1372;
float x1374 = 1.0f - x1373;
float x1376 = x1374 * x1375;
float x1377 = x1371 + x1376;
x326[x1370] = x1377;

}
int32_t x1381 = 0;
int32_t x1382 = 0;
int32_t x1383 = 0;
for(int x1384=0; x1384 < 20; x1384++) {
int32_t x1385 = x1381;
int32_t x1386 = x1382;
int32_t x1387 = x1383;
int32_t x1388 = x1385;
int32_t x1389 = x1386;
int32_t x1390 = x1387;
for(int x1391=0; x1391 < 50; x1391++) {
int32_t x1392 = x1388;
float x1393 = x293[x1392];
float x1394 = x264[x1392];
int32_t x1395 = x1389;
float x1396 = x42[x1395];
int32_t x1397 = x1390;
float x1398 = x326[x1397];
float x1399 = x1393 + x1398;
x293[x1392] = x1399;
float x1401 = x48[x1395];
float x1402 = x264[x1392];
float x1403 = x42[x1395];
float x1404 = x326[x1397];
float x1405 = x1401 + x1404;
x48[x1395] = x1405;
x1390 += 1;
x1388 += 1;
x1389 += 1;

}
x1383 += 50;
x1381 += 50;

}
int32_t x1416 = 0;
int32_t x1417 = 0;
int32_t x1418 = 0;
for(int x1419=0; x1419 < 20; x1419++) {
int32_t x1420 = x1416;
int32_t x1421 = x1417;
int32_t x1422 = x1418;
int32_t x1423 = x1420;
int32_t x1424 = x1421;
int32_t x1425 = x1422;
for(int x1426=0; x1426 < 50; x1426++) {
int32_t x1427 = x1423;
float x1428 = x231[x1427];
float x1429 = x208[x1427];
int32_t x1430 = x1424;
float x1431 = x237[x1430];
int32_t x1432 = x1425;
float x1433 = x293[x1432];
float x1434 = x1428 + x1433;
x231[x1427] = x1434;
float x1436 = x259[x1430];
float x1437 = x208[x1427];
float x1438 = x237[x1430];
float x1439 = x293[x1432];
float x1440 = x1436 + x1439;
x259[x1430] = x1440;
x1425 += 1;
x1423 += 1;
x1424 += 1;

}
x1418 += 50;
x1416 += 50;
x1417 += 50;

}
for(int x1452=0; x1452 < 20; x1452++) {
int32_t x1455 = x1452 * 50;
for(int x1453=0; x1453 < 50; x1453++) {
int32_t x1461 = x1455 + x1453;
for(int x1454=0; x1454 < 50; x1454++) {
int32_t x1456 = x1455 + x1454;
float x1457 = x202[x1456];
int32_t x1458 = x1454 * 50;
int32_t x1459 = x1458 + x1453;
float x1460 = x28[x1459];
float x1462 = x259[x1461];
float x1463 = x1460 * x1462;
float x1464 = x1457 + x1463;
x202[x1456] = x1464;
float x1466 = x37[x1459];
float x1467 = x201[x1456];
float x1468 = x259[x1461];
float x1469 = x1467 * x1468;
float x1470 = x1466 + x1469;
x37[x1459] = x1470;

}

}

}
for(int x1478=0; x1478 < 20; x1478++) {
int32_t x1481 = x1478 * 26;
int32_t x1487 = x1478 * 50;
for(int x1479=0; x1479 < 50; x1479++) {
int32_t x1488 = x1487 + x1479;
for(int x1480=0; x1480 < 26; x1480++) {
int32_t x1482 = x1481 + x1480;
float x1483 = x206[x1482];
int32_t x1484 = x1480 * 50;
int32_t x1485 = x1484 + x1479;
float x1486 = x14[x1485];
float x1489 = x231[x1488];
float x1490 = x1486 * x1489;
float x1491 = x1483 + x1490;
x206[x1482] = x1491;
float x1493 = x23[x1485];
float x1494 = x205[x1482];
float x1495 = x231[x1488];
float x1496 = x1494 * x1495;
float x1497 = x1493 + x1496;
x23[x1485] = x1497;

}

}

}
} else {
float x1506 = 0.0f;
float x1507 = x1506;
float x1508 = x199[0];
float x1509 = x1507 + x1508;
x1506 = x1509;
float x1511 = x1506;
float* x1512 = (float*)myMalloc(1 * sizeof(float));;
x1512[0] = x1511;
float* x1514 = (float*)myMalloc(1 * sizeof(float));;
for(int x1515=0; x1515 < 1; x1515++) {
x1514[x1515] = 0.0f;

}
float x1519 = x1514[0];
x1514[0] = 1.0f;
float x1521 = x1512[0];
x141[0] = x1521;
// += tensor of dim 0
float x1524 = x1514[0];
float x1525 = x200[0];
float x1526 = x1525 + x1524;
x200[0] = x1526;
}
};
x194(0,x1531);
float x1538 = x141[0];
int32_t x1539 = x108 % 100;
bool x1540 = x1539 == 0;
if (x1540) {
printf("iter %d, loss %f\n",x108,x1538);
int32_t x1542 = x108 / 100;
double x1543 = (double)x1538;
x102[x1542] = x1543;
} else {
}
for(int x1547=0; x1547 < 26; x1547++) {
float x1548 = x72[x1547];
bool x1549 = x1548 > 5.0f;
if (x1549) {
x72[x1547] = 5.0f;
} else {
}
float x1553 = x72[x1547];
bool x1554 = x1553 < -5.0f;
if (x1554) {
x72[x1547] = -5.0f;
} else {
}

}
float* x1560 = (float*)myMalloc(26 * sizeof(float));;
int32_t x1561 = 0;
int32_t x1562 = 0;
int32_t x1563 = 0;
for(int x1564=0; x1564 < 26; x1564++) {
int32_t x1565 = x1561;
int32_t x1566 = x1562;
float x1567 = x72[x1566];
int32_t x1568 = x1563;
float x1569 = x72[x1568];
float x1570 = x1567 * x1569;
x1560[x1565] = x1570;
x1561 += 1;
x1562 += 1;
x1563 += 1;

}
for(int x1577=0; x1577 < 26; x1577++) {
float x1578 = x77[x1577];
float x1579 = x1560[x1577];
float x1580 = x1578 + x1579;
x77[x1577] = x1580;

}
float* x1584 = (float*)myMalloc(26 * sizeof(float));;
for(int x1585=0; x1585 < 26; x1585++) {
float x1586 = x72[x1585];
float x1587 = x1586 * 0.1f;
x1584[x1585] = x1587;

}
float* x1591 = (float*)myMalloc(26 * sizeof(float));;
for(int x1592=0; x1592 < 26; x1592++) {
float x1593 = x77[x1592];
float x1594 = x1593 + 1.0E-8f;
x1591[x1592] = x1594;

}
float* x1598 = (float*)myMalloc(26 * sizeof(float));;
for(int x1599=0; x1599 < 26; x1599++) {
float x1600 = x1591[x1599];
double x1601 = (double)x1600;
double x1602 = sqrt(x1601);
float x1603 = (float)x1602;
x1598[x1599] = x1603;

}
float* x1607 = (float*)myMalloc(26 * sizeof(float));;
int32_t x1608 = 0;
int32_t x1609 = 0;
int32_t x1610 = 0;
for(int x1611=0; x1611 < 26; x1611++) {
int32_t x1612 = x1608;
int32_t x1613 = x1609;
float x1614 = x1584[x1613];
int32_t x1615 = x1610;
float x1616 = x1598[x1615];
float x1617 = x1614 / x1616;
x1607[x1612] = x1617;
x1608 += 1;
x1609 += 1;
x1610 += 1;

}
for(int x1624=0; x1624 < 26; x1624++) {
float x1625 = x66[x1624];
float x1626 = x1607[x1624];
float x1627 = x1625 - x1626;
x66[x1624] = x1627;

}
for(int x1631=0; x1631 < 26; x1631++) {
float x1632 = x72[x1631];
x72[x1631] = 0.0f;

}
for(int x1636=0; x1636 < 1300; x1636++) {
float x1637 = x61[x1636];
bool x1638 = x1637 > 5.0f;
if (x1638) {
x61[x1636] = 5.0f;
} else {
}
float x1642 = x61[x1636];
bool x1643 = x1642 < -5.0f;
if (x1643) {
x61[x1636] = -5.0f;
} else {
}

}
float* x1649 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x1650 = 0;
int32_t x1651 = 0;
int32_t x1652 = 0;
for(int x1653=0; x1653 < 50; x1653++) {
int32_t x1654 = x1651;
int32_t x1655 = x1652;
int32_t x1656 = x1650;
int32_t x1657 = x1656;
int32_t x1658 = x1654;
int32_t x1659 = x1655;
for(int x1660=0; x1660 < 26; x1660++) {
int32_t x1661 = x1657;
int32_t x1662 = x1658;
float x1663 = x61[x1662];
int32_t x1664 = x1659;
float x1665 = x61[x1664];
float x1666 = x1663 * x1665;
x1649[x1661] = x1666;
x1657 += 1;
x1658 += 1;
x1659 += 1;

}
x1650 += 26;
x1651 += 26;
x1652 += 26;

}
for(int x1678=0; x1678 < 1300; x1678++) {
float x1679 = x82[x1678];
float x1680 = x1649[x1678];
float x1681 = x1679 + x1680;
x82[x1678] = x1681;

}
float* x1685 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1686=0; x1686 < 1300; x1686++) {
float x1687 = x61[x1686];
float x1688 = x1687 * 0.1f;
x1685[x1686] = x1688;

}
float* x1692 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1693=0; x1693 < 1300; x1693++) {
float x1694 = x82[x1693];
float x1695 = x1694 + 1.0E-8f;
x1692[x1693] = x1695;

}
float* x1699 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1700=0; x1700 < 1300; x1700++) {
float x1701 = x1692[x1700];
double x1702 = (double)x1701;
double x1703 = sqrt(x1702);
float x1704 = (float)x1703;
x1699[x1700] = x1704;

}
float* x1708 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x1709 = 0;
int32_t x1710 = 0;
int32_t x1711 = 0;
for(int x1712=0; x1712 < 50; x1712++) {
int32_t x1713 = x1710;
int32_t x1714 = x1711;
int32_t x1715 = x1709;
int32_t x1716 = x1715;
int32_t x1717 = x1713;
int32_t x1718 = x1714;
for(int x1719=0; x1719 < 26; x1719++) {
int32_t x1720 = x1716;
int32_t x1721 = x1717;
float x1722 = x1685[x1721];
int32_t x1723 = x1718;
float x1724 = x1699[x1723];
float x1725 = x1722 / x1724;
x1708[x1720] = x1725;
x1716 += 1;
x1717 += 1;
x1718 += 1;

}
x1709 += 26;
x1710 += 26;
x1711 += 26;

}
for(int x1737=0; x1737 < 1300; x1737++) {
float x1738 = x53[x1737];
float x1739 = x1708[x1737];
float x1740 = x1738 - x1739;
x53[x1737] = x1740;

}
for(int x1744=0; x1744 < 1300; x1744++) {
float x1745 = x61[x1744];
x61[x1744] = 0.0f;

}
for(int x1749=0; x1749 < 2500; x1749++) {
float x1750 = x37[x1749];
bool x1751 = x1750 > 5.0f;
if (x1751) {
x37[x1749] = 5.0f;
} else {
}
float x1755 = x37[x1749];
bool x1756 = x1755 < -5.0f;
if (x1756) {
x37[x1749] = -5.0f;
} else {
}

}
float* x1762 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x1763 = 0;
int32_t x1764 = 0;
int32_t x1765 = 0;
for(int x1766=0; x1766 < 50; x1766++) {
int32_t x1767 = x1764;
int32_t x1768 = x1765;
int32_t x1769 = x1763;
int32_t x1770 = x1769;
int32_t x1771 = x1767;
int32_t x1772 = x1768;
for(int x1773=0; x1773 < 50; x1773++) {
int32_t x1774 = x1770;
int32_t x1775 = x1771;
float x1776 = x37[x1775];
int32_t x1777 = x1772;
float x1778 = x37[x1777];
float x1779 = x1776 * x1778;
x1762[x1774] = x1779;
x1770 += 1;
x1771 += 1;
x1772 += 1;

}
x1763 += 50;
x1764 += 50;
x1765 += 50;

}
for(int x1791=0; x1791 < 2500; x1791++) {
float x1792 = x87[x1791];
float x1793 = x1762[x1791];
float x1794 = x1792 + x1793;
x87[x1791] = x1794;

}
float* x1798 = (float*)myMalloc(2500 * sizeof(float));;
for(int x1799=0; x1799 < 2500; x1799++) {
float x1800 = x37[x1799];
float x1801 = x1800 * 0.1f;
x1798[x1799] = x1801;

}
float* x1805 = (float*)myMalloc(2500 * sizeof(float));;
for(int x1806=0; x1806 < 2500; x1806++) {
float x1807 = x87[x1806];
float x1808 = x1807 + 1.0E-8f;
x1805[x1806] = x1808;

}
float* x1812 = (float*)myMalloc(2500 * sizeof(float));;
for(int x1813=0; x1813 < 2500; x1813++) {
float x1814 = x1805[x1813];
double x1815 = (double)x1814;
double x1816 = sqrt(x1815);
float x1817 = (float)x1816;
x1812[x1813] = x1817;

}
float* x1821 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x1822 = 0;
int32_t x1823 = 0;
int32_t x1824 = 0;
for(int x1825=0; x1825 < 50; x1825++) {
int32_t x1826 = x1823;
int32_t x1827 = x1824;
int32_t x1828 = x1822;
int32_t x1829 = x1828;
int32_t x1830 = x1826;
int32_t x1831 = x1827;
for(int x1832=0; x1832 < 50; x1832++) {
int32_t x1833 = x1829;
int32_t x1834 = x1830;
float x1835 = x1798[x1834];
int32_t x1836 = x1831;
float x1837 = x1812[x1836];
float x1838 = x1835 / x1837;
x1821[x1833] = x1838;
x1829 += 1;
x1830 += 1;
x1831 += 1;

}
x1822 += 50;
x1823 += 50;
x1824 += 50;

}
for(int x1850=0; x1850 < 2500; x1850++) {
float x1851 = x28[x1850];
float x1852 = x1821[x1850];
float x1853 = x1851 - x1852;
x28[x1850] = x1853;

}
for(int x1857=0; x1857 < 2500; x1857++) {
float x1858 = x37[x1857];
x37[x1857] = 0.0f;

}
for(int x1862=0; x1862 < 1300; x1862++) {
float x1863 = x23[x1862];
bool x1864 = x1863 > 5.0f;
if (x1864) {
x23[x1862] = 5.0f;
} else {
}
float x1868 = x23[x1862];
bool x1869 = x1868 < -5.0f;
if (x1869) {
x23[x1862] = -5.0f;
} else {
}

}
float* x1875 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x1876 = 0;
int32_t x1877 = 0;
int32_t x1878 = 0;
for(int x1879=0; x1879 < 26; x1879++) {
int32_t x1880 = x1877;
int32_t x1881 = x1878;
int32_t x1882 = x1876;
int32_t x1883 = x1882;
int32_t x1884 = x1880;
int32_t x1885 = x1881;
for(int x1886=0; x1886 < 50; x1886++) {
int32_t x1887 = x1883;
int32_t x1888 = x1884;
float x1889 = x23[x1888];
int32_t x1890 = x1885;
float x1891 = x23[x1890];
float x1892 = x1889 * x1891;
x1875[x1887] = x1892;
x1883 += 1;
x1884 += 1;
x1885 += 1;

}
x1876 += 50;
x1877 += 50;
x1878 += 50;

}
for(int x1904=0; x1904 < 1300; x1904++) {
float x1905 = x92[x1904];
float x1906 = x1875[x1904];
float x1907 = x1905 + x1906;
x92[x1904] = x1907;

}
float* x1911 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1912=0; x1912 < 1300; x1912++) {
float x1913 = x23[x1912];
float x1914 = x1913 * 0.1f;
x1911[x1912] = x1914;

}
float* x1918 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1919=0; x1919 < 1300; x1919++) {
float x1920 = x92[x1919];
float x1921 = x1920 + 1.0E-8f;
x1918[x1919] = x1921;

}
float* x1925 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1926=0; x1926 < 1300; x1926++) {
float x1927 = x1918[x1926];
double x1928 = (double)x1927;
double x1929 = sqrt(x1928);
float x1930 = (float)x1929;
x1925[x1926] = x1930;

}
float* x1934 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x1935 = 0;
int32_t x1936 = 0;
int32_t x1937 = 0;
for(int x1938=0; x1938 < 26; x1938++) {
int32_t x1939 = x1936;
int32_t x1940 = x1937;
int32_t x1941 = x1935;
int32_t x1942 = x1941;
int32_t x1943 = x1939;
int32_t x1944 = x1940;
for(int x1945=0; x1945 < 50; x1945++) {
int32_t x1946 = x1942;
int32_t x1947 = x1943;
float x1948 = x1911[x1947];
int32_t x1949 = x1944;
float x1950 = x1925[x1949];
float x1951 = x1948 / x1950;
x1934[x1946] = x1951;
x1942 += 1;
x1943 += 1;
x1944 += 1;

}
x1935 += 50;
x1936 += 50;
x1937 += 50;

}
for(int x1963=0; x1963 < 1300; x1963++) {
float x1964 = x14[x1963];
float x1965 = x1934[x1963];
float x1966 = x1964 - x1965;
x14[x1963] = x1966;

}
for(int x1970=0; x1970 < 1300; x1970++) {
float x1971 = x23[x1970];
x23[x1970] = 0.0f;

}
for(int x1975=0; x1975 < 50; x1975++) {
float x1976 = x48[x1975];
bool x1977 = x1976 > 5.0f;
if (x1977) {
x48[x1975] = 5.0f;
} else {
}
float x1981 = x48[x1975];
bool x1982 = x1981 < -5.0f;
if (x1982) {
x48[x1975] = -5.0f;
} else {
}

}
float* x1988 = (float*)myMalloc(50 * sizeof(float));;
int32_t x1989 = 0;
int32_t x1990 = 0;
int32_t x1991 = 0;
for(int x1992=0; x1992 < 50; x1992++) {
int32_t x1993 = x1989;
int32_t x1994 = x1990;
float x1995 = x48[x1994];
int32_t x1996 = x1991;
float x1997 = x48[x1996];
float x1998 = x1995 * x1997;
x1988[x1993] = x1998;
x1989 += 1;
x1990 += 1;
x1991 += 1;

}
for(int x2005=0; x2005 < 50; x2005++) {
float x2006 = x97[x2005];
float x2007 = x1988[x2005];
float x2008 = x2006 + x2007;
x97[x2005] = x2008;

}
float* x2012 = (float*)myMalloc(50 * sizeof(float));;
for(int x2013=0; x2013 < 50; x2013++) {
float x2014 = x48[x2013];
float x2015 = x2014 * 0.1f;
x2012[x2013] = x2015;

}
float* x2019 = (float*)myMalloc(50 * sizeof(float));;
for(int x2020=0; x2020 < 50; x2020++) {
float x2021 = x97[x2020];
float x2022 = x2021 + 1.0E-8f;
x2019[x2020] = x2022;

}
float* x2026 = (float*)myMalloc(50 * sizeof(float));;
for(int x2027=0; x2027 < 50; x2027++) {
float x2028 = x2019[x2027];
double x2029 = (double)x2028;
double x2030 = sqrt(x2029);
float x2031 = (float)x2030;
x2026[x2027] = x2031;

}
float* x2035 = (float*)myMalloc(50 * sizeof(float));;
int32_t x2036 = 0;
int32_t x2037 = 0;
int32_t x2038 = 0;
for(int x2039=0; x2039 < 50; x2039++) {
int32_t x2040 = x2036;
int32_t x2041 = x2037;
float x2042 = x2012[x2041];
int32_t x2043 = x2038;
float x2044 = x2026[x2043];
float x2045 = x2042 / x2044;
x2035[x2040] = x2045;
x2036 += 1;
x2037 += 1;
x2038 += 1;

}
for(int x2052=0; x2052 < 50; x2052++) {
float x2053 = x42[x2052];
float x2054 = x2035[x2052];
float x2055 = x2053 - x2054;
x42[x2052] = x2055;

}
for(int x2059=0; x2059 < 50; x2059++) {
float x2060 = x48[x2059];
x48[x2059] = 0.0f;

}
mallocAddr = (void*)x104;

}
double x2067 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x2070 = (long)fopen(x0, "w");
fprintf((FILE *)x2070, "unit: %s\n", "100 iteration");
for(int x2073=0; x2073 < 51; x2073++) {
double x2074 = x102[x2073];
fprintf((FILE *)x2070, "%lf\n", x2074);

}
double x2068 = x103 - x1;
double x2069 = x2067 - x103;
fprintf((FILE *)x2070, "run time: %lf %lf\n", x2068, x2069);
fclose((FILE*)x2070);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

