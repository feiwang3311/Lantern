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
float* x92 = (float*)myMalloc(50 * sizeof(float));;
for(int x93=0; x93 < 50; x93++) {
x92[x93] = 0.0f;

}
float* x97 = (float*)myMalloc(1300 * sizeof(float));;
for(int x98=0; x98 < 1300; x98++) {
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
function<void(int32_t,float**)> x549 = [&](int32_t x550,float** x551) {
float** x553 = x551;
float* x554 = x553[0];
float* x555 = x553[1];
float* x556 = x553[2];
float* x557 = x553[3];
int32_t x552 = x550;
bool x558 = x552 < 20;
if (x558) {
int32_t x559 = x552 * 520;
float* x560 = x146+x559;
float* x561 = x168+x559;
float* x562 = (float*)myMalloc(1000 * sizeof(float));;
for(int x563=0; x563 < 20; x563++) {
int32_t x567 = x563 * 26;
int32_t x577 = x563 * 50;
for(int x564=0; x564 < 50; x564++) {
float x565 = 0.0f;
int32_t x570 = x564 * 26;
for(int x566=0; x566 < 26; x566++) {
int32_t x568 = x567 + x566;
float x569 = x560[x568];
int32_t x571 = x570 + x566;
float x572 = x14[x571];
float x573 = x569 * x572;
x565 += x573;

}
float x579 = x565;
int32_t x578 = x577 + x564;
x562[x578] = x579;

}

}
float* x585 = (float*)myMalloc(1000 * sizeof(float));;
for(int x586=0; x586 < 1000; x586++) {
x585[x586] = 0.0f;

}
float* x590 = (float*)myMalloc(1000 * sizeof(float));;
for(int x591=0; x591 < 20; x591++) {
int32_t x595 = x591 * 50;
for(int x592=0; x592 < 50; x592++) {
float x593 = 0.0f;
int32_t x598 = x592 * 50;
for(int x594=0; x594 < 50; x594++) {
int32_t x596 = x595 + x594;
float x597 = x556[x596];
int32_t x599 = x598 + x594;
float x600 = x28[x599];
float x601 = x597 * x600;
x593 += x601;

}
float x606 = x593;
int32_t x605 = x595 + x592;
x590[x605] = x606;

}

}
float* x612 = (float*)myMalloc(1000 * sizeof(float));;
for(int x613=0; x613 < 1000; x613++) {
x612[x613] = 0.0f;

}
float* x617 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x618 = 0;
int32_t x619 = 0;
int32_t x620 = 0;
for(int x621=0; x621 < 20; x621++) {
int32_t x622 = x619;
int32_t x623 = x620;
int32_t x624 = x618;
int32_t x625 = x624;
int32_t x626 = x622;
int32_t x627 = x623;
for(int x628=0; x628 < 50; x628++) {
int32_t x629 = x625;
int32_t x630 = x626;
float x631 = x562[x630];
int32_t x632 = x627;
float x633 = x590[x632];
float x634 = x631 + x633;
x617[x629] = x634;
x625 += 1;
x626 += 1;
x627 += 1;

}
x618 += 50;
x619 += 50;
x620 += 50;

}
float* x646 = (float*)myMalloc(1000 * sizeof(float));;
for(int x647=0; x647 < 1000; x647++) {
x646[x647] = 0.0f;

}
float* x651 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x652 = 0;
int32_t x653 = 0;
int32_t x654 = 0;
for(int x655=0; x655 < 20; x655++) {
int32_t x656 = x653;
int32_t x657 = x654;
int32_t x658 = x652;
int32_t x659 = x658;
int32_t x660 = x656;
int32_t x661 = x657;
for(int x662=0; x662 < 50; x662++) {
int32_t x663 = x659;
int32_t x664 = x660;
float x665 = x617[x664];
int32_t x666 = x661;
float x667 = x42[x666];
float x668 = x665 + x667;
x651[x663] = x668;
x659 += 1;
x660 += 1;
x661 += 1;

}
x652 += 50;
x653 += 50;

}
float* x679 = (float*)myMalloc(1000 * sizeof(float));;
for(int x680=0; x680 < 1000; x680++) {
x679[x680] = 0.0f;

}
float* x684 = (float*)myMalloc(1000 * sizeof(float));;
for(int x685=0; x685 < 1000; x685++) {
float x686 = x651[x685];
double x687 = (double)x686;
double x688 = tanh(x687);
float x689 = (float)x688;
x684[x685] = x689;

}
float* x693 = (float*)myMalloc(1000 * sizeof(float));;
for(int x694=0; x694 < 1000; x694++) {
x693[x694] = 0.0f;

}
float* x698 = (float*)myMalloc(520 * sizeof(float));;
for(int x699=0; x699 < 20; x699++) {
int32_t x703 = x699 * 50;
int32_t x713 = x699 * 26;
for(int x700=0; x700 < 26; x700++) {
float x701 = 0.0f;
int32_t x706 = x700 * 50;
for(int x702=0; x702 < 50; x702++) {
int32_t x704 = x703 + x702;
float x705 = x684[x704];
int32_t x707 = x706 + x702;
float x708 = x53[x707];
float x709 = x705 * x708;
x701 += x709;

}
float x715 = x701;
int32_t x714 = x713 + x700;
x698[x714] = x715;

}

}
float* x721 = (float*)myMalloc(520 * sizeof(float));;
for(int x722=0; x722 < 520; x722++) {
x721[x722] = 0.0f;

}
float* x726 = (float*)myMalloc(520 * sizeof(float));;
int32_t x727 = 0;
int32_t x728 = 0;
int32_t x729 = 0;
for(int x730=0; x730 < 20; x730++) {
int32_t x731 = x728;
int32_t x732 = x729;
int32_t x733 = x727;
int32_t x734 = x733;
int32_t x735 = x731;
int32_t x736 = x732;
for(int x737=0; x737 < 26; x737++) {
int32_t x738 = x734;
int32_t x739 = x735;
float x740 = x698[x739];
int32_t x741 = x736;
float x742 = x66[x741];
float x743 = x740 + x742;
x726[x738] = x743;
x734 += 1;
x735 += 1;
x736 += 1;

}
x727 += 26;
x728 += 26;

}
float* x754 = (float*)myMalloc(520 * sizeof(float));;
for(int x755=0; x755 < 520; x755++) {
x754[x755] = 0.0f;

}
int* x759 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x760=0; x760 < 20; x760++) {
int32_t x761 = x760 * 20;
int32_t x762 = x552 + x761;
int32_t x763 = x118[x762];
x759[x760] = x763;

}
float* x767 = (float*)myMalloc(20 * sizeof(float));;
int32_t x768 = 0;
for(int x769=0; x769 < 20; x769++) {
float x770 = -3.4028235E38f;
for(int x771=0; x771 < 26; x771++) {
int32_t x772 = x768;
float x773 = x726[x772];
float x774 = x770;
bool x775 = x773 > x774;
if (x775) {
float x776 = x726[x772];
x770 = x776;
} else {
}
x768 += 1;

}
float x783 = x770;
x767[x769] = x783;

}
float* x787 = (float*)myMalloc(520 * sizeof(float));;
for(int x788=0; x788 < 520; x788++) {
x787[x788] = 0.0f;

}
int32_t x792 = 0;
for(int x793=0; x793 < 20; x793++) {
for(int x794=0; x794 < 26; x794++) {
int32_t x795 = x792;
float x796 = x726[x795];
float x797 = x767[x793];
float x798 = x796 - x797;
double x799 = (double)x798;
double x800 = exp(x799);
float x801 = (float)x800;
x787[x795] = x801;
x792 += 1;

}

}
float* x808 = (float*)myMalloc(20 * sizeof(float));;
for(int x809=0; x809 < 20; x809++) {
x808[x809] = 0.0f;

}
for(int x813=0; x813 < 20; x813++) {
int32_t x814 = x813;
int32_t x815 = x813 * 26;
int32_t x816 = x815;
for(int x817=0; x817 < 26; x817++) {
int32_t x818 = x814;
float x819 = x808[x818];
int32_t x820 = x816;
float x821 = x787[x820];
float x822 = x819 + x821;
x808[x818] = x822;
x816 += 1;

}

}
x792 = 0;
for(int x830=0; x830 < 20; x830++) {
float x831 = x767[x830];
float x832 = x808[x830];
double x833 = (double)x832;
double x834 = log(x833);
float x835 = (float)x834;
float x836 = x831 + x835;
for(int x837=0; x837 < 26; x837++) {
int32_t x838 = x792;
float x839 = x726[x838];
float x840 = x839 - x836;
x787[x838] = x840;
x792 += 1;

}

}
float* x847 = (float*)myMalloc(520 * sizeof(float));;
for(int x848=0; x848 < 520; x848++) {
x847[x848] = 0.0f;

}
float* x852 = (float*)myMalloc(20 * sizeof(float));;
int32_t x853 = 0;
for(int x854=0; x854 < 20; x854++) {
int32_t x855 = x853;
int32_t x856 = x759[x854];
int32_t x857 = x855 + x856;
float x858 = x787[x857];
float x859 = -1.0f * x858;
x852[x854] = x859;
x853 += 26;

}
float* x864 = (float*)myMalloc(20 * sizeof(float));;
for(int x865=0; x865 < 20; x865++) {
x864[x865] = 0.0f;

}
float x869 = 0.0f;
for(int x870=0; x870 < 20; x870++) {
float x871 = x869;
float x872 = x852[x870];
float x873 = x871 + x872;
x869 = x873;

}
float x877 = x869;
float* x878 = (float*)myMalloc(1 * sizeof(float));;
x878[0] = x877;
float* x880 = (float*)myMalloc(1 * sizeof(float));;
for(int x881=0; x881 < 1; x881++) {
x880[x881] = 0.0f;

}
float* x885 = (float*)myMalloc(1 * sizeof(float));;
int32_t x886 = 0;
int32_t x887 = 0;
int32_t x888 = 0;
int32_t x889 = x886;
int32_t x890 = x887;
float x891 = x554[x890];
int32_t x892 = x888;
float x893 = x878[x892];
float x894 = x891 + x893;
x885[x889] = x894;
x886 += 1;
float* x897 = (float*)myMalloc(1 * sizeof(float));;
for(int x898=0; x898 < 1; x898++) {
x897[x898] = 0.0f;

}
float** x903 = (float**)myMalloc(4 * sizeof(float*));;
x903[0] = x885;
x903[1] = x897;
x903[2] = x684;
x903[3] = x693;
int32_t x910 = 0;
int32_t x911 = 0;
int32_t x912 = 0;
int32_t x913 = x910;
int32_t x916 = x911;
int32_t x918 = x912;
x912 += 1;
int32_t x937 = 0;
float* x950 = (float*)myMalloc(20 * sizeof(float));;
int32_t x971 = 0;
int32_t x991 = 0;
int32_t x992 = 0;
int32_t x993 = 0;
int32_t x1063 = 0;
int32_t x1064 = 0;
int32_t x1065 = 0;
int32_t x1098 = 0;
int32_t x1099 = 0;
int32_t x1100 = 0;
int32_t x902 = x552 + 1;
x549(x902,x903);
float x914 = x555[x913];
float x915 = x554[x913];
float x917 = x878[x916];
float x919 = x897[x918];
float x920 = x914 + x919;
x555[x913] = x920;
float x922 = x880[x916];
float x923 = x554[x913];
float x924 = x878[x916];
float x925 = x897[x918];
float x926 = x922 + x925;
x880[x916] = x926;
// += tensor of dim 0
float x930 = x880[0];
for(int x931=0; x931 < 20; x931++) {
float x932 = x864[x931];
float x933 = x932 + x930;
x864[x931] = x933;

}
for(int x938=0; x938 < 20; x938++) {
int32_t x939 = x937;
int32_t x940 = x759[x938];
int32_t x941 = x939 + x940;
float x942 = x847[x941];
float x943 = x864[x938];
float x944 = -1.0f * x943;
float x945 = x942 + x944;
x847[x941] = x945;
x937 += 26;

}
for(int x951=0; x951 < 20; x951++) {
x950[x951] = 0.0f;

}
for(int x955=0; x955 < 20; x955++) {
int32_t x956 = x955;
int32_t x957 = x955 * 26;
int32_t x958 = x957;
for(int x959=0; x959 < 26; x959++) {
int32_t x960 = x956;
float x961 = x950[x960];
int32_t x962 = x958;
float x963 = x847[x962];
float x964 = x961 + x963;
x950[x960] = x964;
x958 += 1;

}

}
for(int x972=0; x972 < 20; x972++) {
for(int x973=0; x973 < 26; x973++) {
int32_t x974 = x971;
float x975 = x754[x974];
float x976 = x847[x974];
float x977 = x787[x974];
float x981 = x950[x972];
double x978 = (double)x977;
double x979 = exp(x978);
float x980 = (float)x979;
float x982 = x980 * x981;
float x983 = x976 - x982;
float x984 = x975 + x983;
x754[x974] = x984;
x971 += 1;

}

}
for(int x994=0; x994 < 20; x994++) {
int32_t x995 = x991;
int32_t x996 = x992;
int32_t x997 = x993;
int32_t x998 = x995;
int32_t x999 = x996;
int32_t x1000 = x997;
for(int x1001=0; x1001 < 26; x1001++) {
int32_t x1002 = x998;
float x1003 = x721[x1002];
float x1004 = x698[x1002];
int32_t x1005 = x999;
float x1006 = x66[x1005];
int32_t x1007 = x1000;
float x1008 = x754[x1007];
float x1009 = x1003 + x1008;
x721[x1002] = x1009;
float x1011 = x72[x1005];
float x1012 = x698[x1002];
float x1013 = x66[x1005];
float x1014 = x754[x1007];
float x1015 = x1011 + x1014;
x72[x1005] = x1015;
x1000 += 1;
x998 += 1;
x999 += 1;

}
x993 += 26;
x991 += 26;

}
for(int x1026=0; x1026 < 20; x1026++) {
int32_t x1028 = x1026 * 26;
int32_t x1032 = x1026 * 50;
for(int x1027=0; x1027 < 26; x1027++) {
int32_t x1029 = x1028 + x1027;
float x1030 = x721[x1029];
int32_t x1035 = x1027 * 50;
for(int x1031=0; x1031 < 50; x1031++) {
int32_t x1033 = x1032 + x1031;
float x1034 = x693[x1033];
int32_t x1036 = x1035 + x1031;
float x1037 = x53[x1036];
float x1038 = x1037 * x1030;
float x1039 = x1034 + x1038;
x693[x1033] = x1039;
float x1041 = x61[x1036];
float x1042 = x684[x1033];
float x1043 = x1042 * x1030;
float x1044 = x1041 + x1043;
x61[x1036] = x1044;

}

}

}
for(int x1052=0; x1052 < 1000; x1052++) {
float x1053 = x679[x1052];
float x1054 = x684[x1052];
float x1057 = x693[x1052];
float x1055 = x1054 * x1054;
float x1056 = 1.0f - x1055;
float x1058 = x1056 * x1057;
float x1059 = x1053 + x1058;
x679[x1052] = x1059;

}
for(int x1066=0; x1066 < 20; x1066++) {
int32_t x1067 = x1063;
int32_t x1068 = x1064;
int32_t x1069 = x1065;
int32_t x1070 = x1067;
int32_t x1071 = x1068;
int32_t x1072 = x1069;
for(int x1073=0; x1073 < 50; x1073++) {
int32_t x1074 = x1070;
float x1075 = x646[x1074];
float x1076 = x617[x1074];
int32_t x1077 = x1071;
float x1078 = x42[x1077];
int32_t x1079 = x1072;
float x1080 = x679[x1079];
float x1081 = x1075 + x1080;
x646[x1074] = x1081;
float x1083 = x48[x1077];
float x1084 = x617[x1074];
float x1085 = x42[x1077];
float x1086 = x679[x1079];
float x1087 = x1083 + x1086;
x48[x1077] = x1087;
x1072 += 1;
x1070 += 1;
x1071 += 1;

}
x1065 += 50;
x1063 += 50;

}
for(int x1101=0; x1101 < 20; x1101++) {
int32_t x1102 = x1098;
int32_t x1103 = x1099;
int32_t x1104 = x1100;
int32_t x1105 = x1102;
int32_t x1106 = x1103;
int32_t x1107 = x1104;
for(int x1108=0; x1108 < 50; x1108++) {
int32_t x1109 = x1105;
float x1110 = x585[x1109];
float x1111 = x562[x1109];
int32_t x1112 = x1106;
float x1113 = x590[x1112];
int32_t x1114 = x1107;
float x1115 = x646[x1114];
float x1116 = x1110 + x1115;
x585[x1109] = x1116;
float x1118 = x612[x1112];
float x1119 = x562[x1109];
float x1120 = x590[x1112];
float x1121 = x646[x1114];
float x1122 = x1118 + x1121;
x612[x1112] = x1122;
x1107 += 1;
x1105 += 1;
x1106 += 1;

}
x1100 += 50;
x1098 += 50;
x1099 += 50;

}
for(int x1134=0; x1134 < 20; x1134++) {
int32_t x1136 = x1134 * 50;
for(int x1135=0; x1135 < 50; x1135++) {
int32_t x1137 = x1136 + x1135;
float x1138 = x612[x1137];
int32_t x1142 = x1135 * 50;
for(int x1139=0; x1139 < 50; x1139++) {
int32_t x1140 = x1136 + x1139;
float x1141 = x557[x1140];
int32_t x1143 = x1142 + x1139;
float x1144 = x28[x1143];
float x1145 = x1144 * x1138;
float x1146 = x1141 + x1145;
x557[x1140] = x1146;
float x1148 = x37[x1143];
float x1149 = x556[x1140];
float x1150 = x1149 * x1138;
float x1151 = x1148 + x1150;
x37[x1143] = x1151;

}

}

}
for(int x1159=0; x1159 < 20; x1159++) {
int32_t x1161 = x1159 * 50;
int32_t x1165 = x1159 * 26;
for(int x1160=0; x1160 < 50; x1160++) {
int32_t x1162 = x1161 + x1160;
float x1163 = x585[x1162];
int32_t x1168 = x1160 * 26;
for(int x1164=0; x1164 < 26; x1164++) {
int32_t x1166 = x1165 + x1164;
float x1167 = x561[x1166];
int32_t x1169 = x1168 + x1164;
float x1170 = x14[x1169];
float x1171 = x1170 * x1163;
float x1172 = x1167 + x1171;
x561[x1166] = x1172;
float x1174 = x23[x1169];
float x1175 = x560[x1166];
float x1176 = x1175 * x1163;
float x1177 = x1174 + x1176;
x23[x1169] = x1177;

}

}

}
} else {
float x1186 = 0.0f;
float x1187 = x1186;
float x1188 = x554[0];
float x1189 = x1187 + x1188;
x1186 = x1189;
float x1191 = x1186;
float* x1192 = (float*)myMalloc(1 * sizeof(float));;
x1192[0] = x1191;
float* x1194 = (float*)myMalloc(1 * sizeof(float));;
for(int x1195=0; x1195 < 1; x1195++) {
x1194[x1195] = 0.0f;

}
float x1199 = x1194[0];
x1194[0] = 1.0f;
float x1201 = x1192[0];
x141[0] = x1201;
// += tensor of dim 0
float x1204 = x1194[0];
float x1205 = x555[0];
float x1206 = x1205 + x1204;
x555[0] = x1206;
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
float** x1519 = (float**)myMalloc(4 * sizeof(float*));;
x1519[0] = x173;
x1519[1] = x178;
x1519[2] = x183;
x1519[3] = x189;
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
float* x207 = (float*)myMalloc(1000 * sizeof(float));;
for(int x208=0; x208 < 20; x208++) {
int32_t x212 = x208 * 26;
int32_t x222 = x208 * 50;
for(int x209=0; x209 < 50; x209++) {
float x210 = 0.0f;
int32_t x215 = x209 * 26;
for(int x211=0; x211 < 26; x211++) {
int32_t x213 = x212 + x211;
float x214 = x205[x213];
int32_t x216 = x215 + x211;
float x217 = x14[x216];
float x218 = x214 * x217;
x210 += x218;

}
float x224 = x210;
int32_t x223 = x222 + x209;
x207[x223] = x224;

}

}
float* x230 = (float*)myMalloc(1000 * sizeof(float));;
for(int x231=0; x231 < 1000; x231++) {
x230[x231] = 0.0f;

}
float* x235 = (float*)myMalloc(1000 * sizeof(float));;
for(int x236=0; x236 < 20; x236++) {
int32_t x240 = x236 * 50;
for(int x237=0; x237 < 50; x237++) {
float x238 = 0.0f;
int32_t x243 = x237 * 50;
for(int x239=0; x239 < 50; x239++) {
int32_t x241 = x240 + x239;
float x242 = x201[x241];
int32_t x244 = x243 + x239;
float x245 = x28[x244];
float x246 = x242 * x245;
x238 += x246;

}
float x251 = x238;
int32_t x250 = x240 + x237;
x235[x250] = x251;

}

}
float* x257 = (float*)myMalloc(1000 * sizeof(float));;
for(int x258=0; x258 < 1000; x258++) {
x257[x258] = 0.0f;

}
float* x262 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x263 = 0;
int32_t x264 = 0;
int32_t x265 = 0;
for(int x266=0; x266 < 20; x266++) {
int32_t x267 = x264;
int32_t x268 = x265;
int32_t x269 = x263;
int32_t x270 = x269;
int32_t x271 = x267;
int32_t x272 = x268;
for(int x273=0; x273 < 50; x273++) {
int32_t x274 = x270;
int32_t x275 = x271;
float x276 = x207[x275];
int32_t x277 = x272;
float x278 = x235[x277];
float x279 = x276 + x278;
x262[x274] = x279;
x270 += 1;
x271 += 1;
x272 += 1;

}
x263 += 50;
x264 += 50;
x265 += 50;

}
float* x291 = (float*)myMalloc(1000 * sizeof(float));;
for(int x292=0; x292 < 1000; x292++) {
x291[x292] = 0.0f;

}
float* x296 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x297 = 0;
int32_t x298 = 0;
int32_t x299 = 0;
for(int x300=0; x300 < 20; x300++) {
int32_t x301 = x298;
int32_t x302 = x299;
int32_t x303 = x297;
int32_t x304 = x303;
int32_t x305 = x301;
int32_t x306 = x302;
for(int x307=0; x307 < 50; x307++) {
int32_t x308 = x304;
int32_t x309 = x305;
float x310 = x262[x309];
int32_t x311 = x306;
float x312 = x42[x311];
float x313 = x310 + x312;
x296[x308] = x313;
x304 += 1;
x305 += 1;
x306 += 1;

}
x297 += 50;
x298 += 50;

}
float* x324 = (float*)myMalloc(1000 * sizeof(float));;
for(int x325=0; x325 < 1000; x325++) {
x324[x325] = 0.0f;

}
float* x329 = (float*)myMalloc(1000 * sizeof(float));;
for(int x330=0; x330 < 1000; x330++) {
float x331 = x296[x330];
double x332 = (double)x331;
double x333 = tanh(x332);
float x334 = (float)x333;
x329[x330] = x334;

}
float* x338 = (float*)myMalloc(1000 * sizeof(float));;
for(int x339=0; x339 < 1000; x339++) {
x338[x339] = 0.0f;

}
float* x343 = (float*)myMalloc(520 * sizeof(float));;
for(int x344=0; x344 < 20; x344++) {
int32_t x348 = x344 * 50;
int32_t x358 = x344 * 26;
for(int x345=0; x345 < 26; x345++) {
float x346 = 0.0f;
int32_t x351 = x345 * 50;
for(int x347=0; x347 < 50; x347++) {
int32_t x349 = x348 + x347;
float x350 = x329[x349];
int32_t x352 = x351 + x347;
float x353 = x53[x352];
float x354 = x350 * x353;
x346 += x354;

}
float x360 = x346;
int32_t x359 = x358 + x345;
x343[x359] = x360;

}

}
float* x366 = (float*)myMalloc(520 * sizeof(float));;
for(int x368=0; x368 < 520; x368++) {
x366[x368] = 0.0f;

}
float* x372 = (float*)myMalloc(520 * sizeof(float));;
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
for(int x383=0; x383 < 26; x383++) {
int32_t x384 = x380;
int32_t x385 = x381;
float x386 = x343[x385];
int32_t x387 = x382;
float x388 = x66[x387];
float x389 = x386 + x388;
x372[x384] = x389;
x380 += 1;
x381 += 1;
x382 += 1;

}
x373 += 26;
x374 += 26;

}
float* x400 = (float*)myMalloc(520 * sizeof(float));;
for(int x401=0; x401 < 520; x401++) {
x400[x401] = 0.0f;

}
int* x405 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x406=0; x406 < 20; x406++) {
int32_t x407 = x406 * 20;
int32_t x408 = x197 + x407;
int32_t x409 = x118[x408];
x405[x406] = x409;

}
float* x413 = (float*)myMalloc(20 * sizeof(float));;
int32_t x414 = 0;
for(int x415=0; x415 < 20; x415++) {
float x416 = -3.4028235E38f;
for(int x417=0; x417 < 26; x417++) {
int32_t x418 = x414;
float x419 = x372[x418];
float x420 = x416;
bool x421 = x419 > x420;
if (x421) {
float x422 = x372[x418];
x416 = x422;
} else {
}
x414 += 1;

}
float x429 = x416;
x413[x415] = x429;

}
float* x433 = (float*)myMalloc(520 * sizeof(float));;
for(int x434=0; x434 < 520; x434++) {
x433[x434] = 0.0f;

}
int32_t x438 = 0;
for(int x439=0; x439 < 20; x439++) {
for(int x440=0; x440 < 26; x440++) {
int32_t x441 = x438;
float x442 = x372[x441];
float x443 = x413[x439];
float x444 = x442 - x443;
double x445 = (double)x444;
double x446 = exp(x445);
float x447 = (float)x446;
x433[x441] = x447;
x438 += 1;

}

}
float* x454 = (float*)myMalloc(20 * sizeof(float));;
for(int x455=0; x455 < 20; x455++) {
x454[x455] = 0.0f;

}
for(int x459=0; x459 < 20; x459++) {
int32_t x460 = x459;
int32_t x461 = x459 * 26;
int32_t x462 = x461;
for(int x463=0; x463 < 26; x463++) {
int32_t x464 = x460;
float x465 = x454[x464];
int32_t x466 = x462;
float x467 = x433[x466];
float x468 = x465 + x467;
x454[x464] = x468;
x462 += 1;

}

}
x438 = 0;
for(int x476=0; x476 < 20; x476++) {
float x477 = x413[x476];
float x478 = x454[x476];
double x479 = (double)x478;
double x480 = log(x479);
float x481 = (float)x480;
float x482 = x477 + x481;
for(int x483=0; x483 < 26; x483++) {
int32_t x484 = x438;
float x485 = x372[x484];
float x486 = x485 - x482;
x433[x484] = x486;
x438 += 1;

}

}
float* x493 = (float*)myMalloc(520 * sizeof(float));;
for(int x494=0; x494 < 520; x494++) {
x493[x494] = 0.0f;

}
float* x498 = (float*)myMalloc(20 * sizeof(float));;
int32_t x499 = 0;
for(int x500=0; x500 < 20; x500++) {
int32_t x501 = x499;
int32_t x502 = x405[x500];
int32_t x503 = x501 + x502;
float x504 = x433[x503];
float x505 = -1.0f * x504;
x498[x500] = x505;
x499 += 26;

}
float* x510 = (float*)myMalloc(20 * sizeof(float));;
for(int x511=0; x511 < 20; x511++) {
x510[x511] = 0.0f;

}
float x515 = 0.0f;
for(int x516=0; x516 < 20; x516++) {
float x517 = x515;
float x518 = x498[x516];
float x519 = x517 + x518;
x515 = x519;

}
float x523 = x515;
float* x524 = (float*)myMalloc(1 * sizeof(float));;
x524[0] = x523;
float* x526 = (float*)myMalloc(1 * sizeof(float));;
for(int x527=0; x527 < 1; x527++) {
x526[x527] = 0.0f;

}
float* x531 = (float*)myMalloc(1 * sizeof(float));;
int32_t x532 = 0;
int32_t x533 = 0;
int32_t x534 = 0;
int32_t x535 = x532;
int32_t x536 = x533;
float x537 = x199[x536];
int32_t x538 = x534;
float x539 = x524[x538];
float x540 = x537 + x539;
x531[x535] = x540;
x532 += 1;
float* x543 = (float*)myMalloc(1 * sizeof(float));;
for(int x544=0; x544 < 1; x544++) {
x543[x544] = 0.0f;

}
float** x1211 = (float**)myMalloc(4 * sizeof(float*));;
x1211[0] = x531;
x1211[1] = x543;
x1211[2] = x329;
x1211[3] = x338;
int32_t x548 = x197 + 1;
x549(x548,x1211);
int32_t x1218 = 0;
int32_t x1219 = 0;
int32_t x1220 = 0;
int32_t x1221 = x1218;
float x1222 = x200[x1221];
float x1223 = x199[x1221];
int32_t x1224 = x1219;
float x1225 = x524[x1224];
int32_t x1226 = x1220;
float x1227 = x543[x1226];
float x1228 = x1222 + x1227;
x200[x1221] = x1228;
float x1230 = x526[x1224];
float x1231 = x199[x1221];
float x1232 = x524[x1224];
float x1233 = x543[x1226];
float x1234 = x1230 + x1233;
x526[x1224] = x1234;
x1220 += 1;
// += tensor of dim 0
float x1238 = x526[0];
for(int x1239=0; x1239 < 20; x1239++) {
float x1240 = x510[x1239];
float x1241 = x1240 + x1238;
x510[x1239] = x1241;

}
int32_t x1245 = 0;
for(int x1246=0; x1246 < 20; x1246++) {
int32_t x1247 = x1245;
int32_t x1248 = x405[x1246];
int32_t x1249 = x1247 + x1248;
float x1250 = x493[x1249];
float x1251 = x510[x1246];
float x1252 = -1.0f * x1251;
float x1253 = x1250 + x1252;
x493[x1249] = x1253;
x1245 += 26;

}
float* x1258 = (float*)myMalloc(20 * sizeof(float));;
for(int x1259=0; x1259 < 20; x1259++) {
x1258[x1259] = 0.0f;

}
for(int x1263=0; x1263 < 20; x1263++) {
int32_t x1264 = x1263;
int32_t x1265 = x1263 * 26;
int32_t x1266 = x1265;
for(int x1267=0; x1267 < 26; x1267++) {
int32_t x1268 = x1264;
float x1269 = x1258[x1268];
int32_t x1270 = x1266;
float x1271 = x493[x1270];
float x1272 = x1269 + x1271;
x1258[x1268] = x1272;
x1266 += 1;

}

}
int32_t x1279 = 0;
for(int x1280=0; x1280 < 20; x1280++) {
for(int x1281=0; x1281 < 26; x1281++) {
int32_t x1282 = x1279;
float x1283 = x400[x1282];
float x1284 = x493[x1282];
float x1285 = x433[x1282];
float x1289 = x1258[x1280];
double x1286 = (double)x1285;
double x1287 = exp(x1286);
float x1288 = (float)x1287;
float x1290 = x1288 * x1289;
float x1291 = x1284 - x1290;
float x1292 = x1283 + x1291;
x400[x1282] = x1292;
x1279 += 1;

}

}
int32_t x1299 = 0;
int32_t x1300 = 0;
int32_t x1301 = 0;
for(int x1302=0; x1302 < 20; x1302++) {
int32_t x1303 = x1299;
int32_t x1304 = x1300;
int32_t x1305 = x1301;
int32_t x1306 = x1303;
int32_t x1307 = x1304;
int32_t x1308 = x1305;
for(int x1309=0; x1309 < 26; x1309++) {
int32_t x1310 = x1306;
float x1311 = x366[x1310];
float x1312 = x343[x1310];
int32_t x1313 = x1307;
float x1314 = x66[x1313];
int32_t x1315 = x1308;
float x1316 = x400[x1315];
float x1317 = x1311 + x1316;
x366[x1310] = x1317;
float x1319 = x72[x1313];
float x1320 = x343[x1310];
float x1321 = x66[x1313];
float x1322 = x400[x1315];
float x1323 = x1319 + x1322;
x72[x1313] = x1323;
x1308 += 1;
x1306 += 1;
x1307 += 1;

}
x1301 += 26;
x1299 += 26;

}
for(int x1334=0; x1334 < 20; x1334++) {
int32_t x1336 = x1334 * 26;
int32_t x1340 = x1334 * 50;
for(int x1335=0; x1335 < 26; x1335++) {
int32_t x1337 = x1336 + x1335;
float x1338 = x366[x1337];
int32_t x1343 = x1335 * 50;
for(int x1339=0; x1339 < 50; x1339++) {
int32_t x1341 = x1340 + x1339;
float x1342 = x338[x1341];
int32_t x1344 = x1343 + x1339;
float x1345 = x53[x1344];
float x1346 = x1345 * x1338;
float x1347 = x1342 + x1346;
x338[x1341] = x1347;
float x1349 = x61[x1344];
float x1350 = x329[x1341];
float x1351 = x1350 * x1338;
float x1352 = x1349 + x1351;
x61[x1344] = x1352;

}

}

}
for(int x1360=0; x1360 < 1000; x1360++) {
float x1361 = x324[x1360];
float x1362 = x329[x1360];
float x1365 = x338[x1360];
float x1363 = x1362 * x1362;
float x1364 = 1.0f - x1363;
float x1366 = x1364 * x1365;
float x1367 = x1361 + x1366;
x324[x1360] = x1367;

}
int32_t x1371 = 0;
int32_t x1372 = 0;
int32_t x1373 = 0;
for(int x1374=0; x1374 < 20; x1374++) {
int32_t x1375 = x1371;
int32_t x1376 = x1372;
int32_t x1377 = x1373;
int32_t x1378 = x1375;
int32_t x1379 = x1376;
int32_t x1380 = x1377;
for(int x1381=0; x1381 < 50; x1381++) {
int32_t x1382 = x1378;
float x1383 = x291[x1382];
float x1384 = x262[x1382];
int32_t x1385 = x1379;
float x1386 = x42[x1385];
int32_t x1387 = x1380;
float x1388 = x324[x1387];
float x1389 = x1383 + x1388;
x291[x1382] = x1389;
float x1391 = x48[x1385];
float x1392 = x262[x1382];
float x1393 = x42[x1385];
float x1394 = x324[x1387];
float x1395 = x1391 + x1394;
x48[x1385] = x1395;
x1380 += 1;
x1378 += 1;
x1379 += 1;

}
x1373 += 50;
x1371 += 50;

}
int32_t x1406 = 0;
int32_t x1407 = 0;
int32_t x1408 = 0;
for(int x1409=0; x1409 < 20; x1409++) {
int32_t x1410 = x1406;
int32_t x1411 = x1407;
int32_t x1412 = x1408;
int32_t x1413 = x1410;
int32_t x1414 = x1411;
int32_t x1415 = x1412;
for(int x1416=0; x1416 < 50; x1416++) {
int32_t x1417 = x1413;
float x1418 = x230[x1417];
float x1419 = x207[x1417];
int32_t x1420 = x1414;
float x1421 = x235[x1420];
int32_t x1422 = x1415;
float x1423 = x291[x1422];
float x1424 = x1418 + x1423;
x230[x1417] = x1424;
float x1426 = x257[x1420];
float x1427 = x207[x1417];
float x1428 = x235[x1420];
float x1429 = x291[x1422];
float x1430 = x1426 + x1429;
x257[x1420] = x1430;
x1415 += 1;
x1413 += 1;
x1414 += 1;

}
x1408 += 50;
x1406 += 50;
x1407 += 50;

}
for(int x1442=0; x1442 < 20; x1442++) {
int32_t x1444 = x1442 * 50;
for(int x1443=0; x1443 < 50; x1443++) {
int32_t x1445 = x1444 + x1443;
float x1446 = x257[x1445];
int32_t x1450 = x1443 * 50;
for(int x1447=0; x1447 < 50; x1447++) {
int32_t x1448 = x1444 + x1447;
float x1449 = x202[x1448];
int32_t x1451 = x1450 + x1447;
float x1452 = x28[x1451];
float x1453 = x1452 * x1446;
float x1454 = x1449 + x1453;
x202[x1448] = x1454;
float x1456 = x37[x1451];
float x1457 = x201[x1448];
float x1458 = x1457 * x1446;
float x1459 = x1456 + x1458;
x37[x1451] = x1459;

}

}

}
for(int x1467=0; x1467 < 20; x1467++) {
int32_t x1469 = x1467 * 50;
int32_t x1473 = x1467 * 26;
for(int x1468=0; x1468 < 50; x1468++) {
int32_t x1470 = x1469 + x1468;
float x1471 = x230[x1470];
int32_t x1476 = x1468 * 26;
for(int x1472=0; x1472 < 26; x1472++) {
int32_t x1474 = x1473 + x1472;
float x1475 = x206[x1474];
int32_t x1477 = x1476 + x1472;
float x1478 = x14[x1477];
float x1479 = x1478 * x1471;
float x1480 = x1475 + x1479;
x206[x1474] = x1480;
float x1482 = x23[x1477];
float x1483 = x205[x1474];
float x1484 = x1483 * x1471;
float x1485 = x1482 + x1484;
x23[x1477] = x1485;

}

}

}
} else {
float x1494 = 0.0f;
float x1495 = x1494;
float x1496 = x199[0];
float x1497 = x1495 + x1496;
x1494 = x1497;
float x1499 = x1494;
float* x1500 = (float*)myMalloc(1 * sizeof(float));;
x1500[0] = x1499;
float* x1502 = (float*)myMalloc(1 * sizeof(float));;
for(int x1503=0; x1503 < 1; x1503++) {
x1502[x1503] = 0.0f;

}
float x1507 = x1502[0];
x1502[0] = 1.0f;
float x1509 = x1500[0];
x141[0] = x1509;
// += tensor of dim 0
float x1512 = x1502[0];
float x1513 = x200[0];
float x1514 = x1513 + x1512;
x200[0] = x1514;
}
};
x194(0,x1519);
float x1526 = x141[0];
int32_t x1527 = x108 % 100;
bool x1528 = x1527 == 0;
if (x1528) {
printf("iter %d, loss %f\n",x108,x1526);
int32_t x1530 = x108 / 100;
double x1531 = (double)x1526;
x102[x1530] = x1531;
} else {
}
for(int x1535=0; x1535 < 26; x1535++) {
float x1536 = x72[x1535];
bool x1537 = x1536 > 5.0f;
if (x1537) {
x72[x1535] = 5.0f;
} else {
}
float x1541 = x72[x1535];
bool x1542 = x1541 < -5.0f;
if (x1542) {
x72[x1535] = -5.0f;
} else {
}

}
float* x1548 = (float*)myMalloc(26 * sizeof(float));;
int32_t x1549 = 0;
int32_t x1550 = 0;
int32_t x1551 = 0;
for(int x1552=0; x1552 < 26; x1552++) {
int32_t x1553 = x1549;
int32_t x1554 = x1550;
float x1555 = x72[x1554];
int32_t x1556 = x1551;
float x1557 = x72[x1556];
float x1558 = x1555 * x1557;
x1548[x1553] = x1558;
x1549 += 1;
x1550 += 1;
x1551 += 1;

}
for(int x1565=0; x1565 < 26; x1565++) {
float x1566 = x77[x1565];
float x1567 = x1548[x1565];
float x1568 = x1566 + x1567;
x77[x1565] = x1568;

}
float* x1572 = (float*)myMalloc(26 * sizeof(float));;
for(int x1573=0; x1573 < 26; x1573++) {
float x1574 = x72[x1573];
float x1575 = x1574 * 0.1f;
x1572[x1573] = x1575;

}
float* x1579 = (float*)myMalloc(26 * sizeof(float));;
for(int x1580=0; x1580 < 26; x1580++) {
float x1581 = x77[x1580];
float x1582 = x1581 + 1.0E-8f;
x1579[x1580] = x1582;

}
float* x1586 = (float*)myMalloc(26 * sizeof(float));;
for(int x1587=0; x1587 < 26; x1587++) {
float x1588 = x1579[x1587];
double x1589 = (double)x1588;
double x1590 = sqrt(x1589);
float x1591 = (float)x1590;
x1586[x1587] = x1591;

}
float* x1595 = (float*)myMalloc(26 * sizeof(float));;
int32_t x1596 = 0;
int32_t x1597 = 0;
int32_t x1598 = 0;
for(int x1599=0; x1599 < 26; x1599++) {
int32_t x1600 = x1596;
int32_t x1601 = x1597;
float x1602 = x1572[x1601];
int32_t x1603 = x1598;
float x1604 = x1586[x1603];
float x1605 = x1602 / x1604;
x1595[x1600] = x1605;
x1596 += 1;
x1597 += 1;
x1598 += 1;

}
for(int x1612=0; x1612 < 26; x1612++) {
float x1613 = x66[x1612];
float x1614 = x1595[x1612];
float x1615 = x1613 - x1614;
x66[x1612] = x1615;

}
for(int x1619=0; x1619 < 26; x1619++) {
float x1620 = x72[x1619];
x72[x1619] = 0.0f;

}
for(int x1624=0; x1624 < 1300; x1624++) {
float x1625 = x61[x1624];
bool x1626 = x1625 > 5.0f;
if (x1626) {
x61[x1624] = 5.0f;
} else {
}
float x1630 = x61[x1624];
bool x1631 = x1630 < -5.0f;
if (x1631) {
x61[x1624] = -5.0f;
} else {
}

}
float* x1637 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x1638 = 0;
int32_t x1639 = 0;
int32_t x1640 = 0;
for(int x1641=0; x1641 < 26; x1641++) {
int32_t x1642 = x1639;
int32_t x1643 = x1640;
int32_t x1644 = x1638;
int32_t x1645 = x1644;
int32_t x1646 = x1642;
int32_t x1647 = x1643;
for(int x1648=0; x1648 < 50; x1648++) {
int32_t x1649 = x1645;
int32_t x1650 = x1646;
float x1651 = x61[x1650];
int32_t x1652 = x1647;
float x1653 = x61[x1652];
float x1654 = x1651 * x1653;
x1637[x1649] = x1654;
x1645 += 1;
x1646 += 1;
x1647 += 1;

}
x1638 += 50;
x1639 += 50;
x1640 += 50;

}
for(int x1666=0; x1666 < 1300; x1666++) {
float x1667 = x82[x1666];
float x1668 = x1637[x1666];
float x1669 = x1667 + x1668;
x82[x1666] = x1669;

}
float* x1673 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1674=0; x1674 < 1300; x1674++) {
float x1675 = x61[x1674];
float x1676 = x1675 * 0.1f;
x1673[x1674] = x1676;

}
float* x1680 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1681=0; x1681 < 1300; x1681++) {
float x1682 = x82[x1681];
float x1683 = x1682 + 1.0E-8f;
x1680[x1681] = x1683;

}
float* x1687 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1688=0; x1688 < 1300; x1688++) {
float x1689 = x1680[x1688];
double x1690 = (double)x1689;
double x1691 = sqrt(x1690);
float x1692 = (float)x1691;
x1687[x1688] = x1692;

}
float* x1696 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x1697 = 0;
int32_t x1698 = 0;
int32_t x1699 = 0;
for(int x1700=0; x1700 < 26; x1700++) {
int32_t x1701 = x1698;
int32_t x1702 = x1699;
int32_t x1703 = x1697;
int32_t x1704 = x1703;
int32_t x1705 = x1701;
int32_t x1706 = x1702;
for(int x1707=0; x1707 < 50; x1707++) {
int32_t x1708 = x1704;
int32_t x1709 = x1705;
float x1710 = x1673[x1709];
int32_t x1711 = x1706;
float x1712 = x1687[x1711];
float x1713 = x1710 / x1712;
x1696[x1708] = x1713;
x1704 += 1;
x1705 += 1;
x1706 += 1;

}
x1697 += 50;
x1698 += 50;
x1699 += 50;

}
for(int x1725=0; x1725 < 1300; x1725++) {
float x1726 = x53[x1725];
float x1727 = x1696[x1725];
float x1728 = x1726 - x1727;
x53[x1725] = x1728;

}
for(int x1732=0; x1732 < 1300; x1732++) {
float x1733 = x61[x1732];
x61[x1732] = 0.0f;

}
for(int x1737=0; x1737 < 2500; x1737++) {
float x1738 = x37[x1737];
bool x1739 = x1738 > 5.0f;
if (x1739) {
x37[x1737] = 5.0f;
} else {
}
float x1743 = x37[x1737];
bool x1744 = x1743 < -5.0f;
if (x1744) {
x37[x1737] = -5.0f;
} else {
}

}
float* x1750 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x1751 = 0;
int32_t x1752 = 0;
int32_t x1753 = 0;
for(int x1754=0; x1754 < 50; x1754++) {
int32_t x1755 = x1752;
int32_t x1756 = x1753;
int32_t x1757 = x1751;
int32_t x1758 = x1757;
int32_t x1759 = x1755;
int32_t x1760 = x1756;
for(int x1761=0; x1761 < 50; x1761++) {
int32_t x1762 = x1758;
int32_t x1763 = x1759;
float x1764 = x37[x1763];
int32_t x1765 = x1760;
float x1766 = x37[x1765];
float x1767 = x1764 * x1766;
x1750[x1762] = x1767;
x1758 += 1;
x1759 += 1;
x1760 += 1;

}
x1751 += 50;
x1752 += 50;
x1753 += 50;

}
for(int x1779=0; x1779 < 2500; x1779++) {
float x1780 = x87[x1779];
float x1781 = x1750[x1779];
float x1782 = x1780 + x1781;
x87[x1779] = x1782;

}
float* x1786 = (float*)myMalloc(2500 * sizeof(float));;
for(int x1787=0; x1787 < 2500; x1787++) {
float x1788 = x37[x1787];
float x1789 = x1788 * 0.1f;
x1786[x1787] = x1789;

}
float* x1793 = (float*)myMalloc(2500 * sizeof(float));;
for(int x1794=0; x1794 < 2500; x1794++) {
float x1795 = x87[x1794];
float x1796 = x1795 + 1.0E-8f;
x1793[x1794] = x1796;

}
float* x1800 = (float*)myMalloc(2500 * sizeof(float));;
for(int x1801=0; x1801 < 2500; x1801++) {
float x1802 = x1793[x1801];
double x1803 = (double)x1802;
double x1804 = sqrt(x1803);
float x1805 = (float)x1804;
x1800[x1801] = x1805;

}
float* x1809 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x1810 = 0;
int32_t x1811 = 0;
int32_t x1812 = 0;
for(int x1813=0; x1813 < 50; x1813++) {
int32_t x1814 = x1811;
int32_t x1815 = x1812;
int32_t x1816 = x1810;
int32_t x1817 = x1816;
int32_t x1818 = x1814;
int32_t x1819 = x1815;
for(int x1820=0; x1820 < 50; x1820++) {
int32_t x1821 = x1817;
int32_t x1822 = x1818;
float x1823 = x1786[x1822];
int32_t x1824 = x1819;
float x1825 = x1800[x1824];
float x1826 = x1823 / x1825;
x1809[x1821] = x1826;
x1817 += 1;
x1818 += 1;
x1819 += 1;

}
x1810 += 50;
x1811 += 50;
x1812 += 50;

}
for(int x1838=0; x1838 < 2500; x1838++) {
float x1839 = x28[x1838];
float x1840 = x1809[x1838];
float x1841 = x1839 - x1840;
x28[x1838] = x1841;

}
for(int x1845=0; x1845 < 2500; x1845++) {
float x1846 = x37[x1845];
x37[x1845] = 0.0f;

}
for(int x1850=0; x1850 < 50; x1850++) {
float x1851 = x48[x1850];
bool x1852 = x1851 > 5.0f;
if (x1852) {
x48[x1850] = 5.0f;
} else {
}
float x1856 = x48[x1850];
bool x1857 = x1856 < -5.0f;
if (x1857) {
x48[x1850] = -5.0f;
} else {
}

}
float* x1863 = (float*)myMalloc(50 * sizeof(float));;
int32_t x1864 = 0;
int32_t x1865 = 0;
int32_t x1866 = 0;
for(int x1867=0; x1867 < 50; x1867++) {
int32_t x1868 = x1864;
int32_t x1869 = x1865;
float x1870 = x48[x1869];
int32_t x1871 = x1866;
float x1872 = x48[x1871];
float x1873 = x1870 * x1872;
x1863[x1868] = x1873;
x1864 += 1;
x1865 += 1;
x1866 += 1;

}
for(int x1880=0; x1880 < 50; x1880++) {
float x1881 = x92[x1880];
float x1882 = x1863[x1880];
float x1883 = x1881 + x1882;
x92[x1880] = x1883;

}
float* x1887 = (float*)myMalloc(50 * sizeof(float));;
for(int x1888=0; x1888 < 50; x1888++) {
float x1889 = x48[x1888];
float x1890 = x1889 * 0.1f;
x1887[x1888] = x1890;

}
float* x1894 = (float*)myMalloc(50 * sizeof(float));;
for(int x1895=0; x1895 < 50; x1895++) {
float x1896 = x92[x1895];
float x1897 = x1896 + 1.0E-8f;
x1894[x1895] = x1897;

}
float* x1901 = (float*)myMalloc(50 * sizeof(float));;
for(int x1902=0; x1902 < 50; x1902++) {
float x1903 = x1894[x1902];
double x1904 = (double)x1903;
double x1905 = sqrt(x1904);
float x1906 = (float)x1905;
x1901[x1902] = x1906;

}
float* x1910 = (float*)myMalloc(50 * sizeof(float));;
int32_t x1911 = 0;
int32_t x1912 = 0;
int32_t x1913 = 0;
for(int x1914=0; x1914 < 50; x1914++) {
int32_t x1915 = x1911;
int32_t x1916 = x1912;
float x1917 = x1887[x1916];
int32_t x1918 = x1913;
float x1919 = x1901[x1918];
float x1920 = x1917 / x1919;
x1910[x1915] = x1920;
x1911 += 1;
x1912 += 1;
x1913 += 1;

}
for(int x1927=0; x1927 < 50; x1927++) {
float x1928 = x42[x1927];
float x1929 = x1910[x1927];
float x1930 = x1928 - x1929;
x42[x1927] = x1930;

}
for(int x1934=0; x1934 < 50; x1934++) {
float x1935 = x48[x1934];
x48[x1934] = 0.0f;

}
for(int x1939=0; x1939 < 1300; x1939++) {
float x1940 = x23[x1939];
bool x1941 = x1940 > 5.0f;
if (x1941) {
x23[x1939] = 5.0f;
} else {
}
float x1945 = x23[x1939];
bool x1946 = x1945 < -5.0f;
if (x1946) {
x23[x1939] = -5.0f;
} else {
}

}
float* x1952 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x1953 = 0;
int32_t x1954 = 0;
int32_t x1955 = 0;
for(int x1956=0; x1956 < 50; x1956++) {
int32_t x1957 = x1954;
int32_t x1958 = x1955;
int32_t x1959 = x1953;
int32_t x1960 = x1959;
int32_t x1961 = x1957;
int32_t x1962 = x1958;
for(int x1963=0; x1963 < 26; x1963++) {
int32_t x1964 = x1960;
int32_t x1965 = x1961;
float x1966 = x23[x1965];
int32_t x1967 = x1962;
float x1968 = x23[x1967];
float x1969 = x1966 * x1968;
x1952[x1964] = x1969;
x1960 += 1;
x1961 += 1;
x1962 += 1;

}
x1953 += 26;
x1954 += 26;
x1955 += 26;

}
for(int x1981=0; x1981 < 1300; x1981++) {
float x1982 = x97[x1981];
float x1983 = x1952[x1981];
float x1984 = x1982 + x1983;
x97[x1981] = x1984;

}
float* x1988 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1989=0; x1989 < 1300; x1989++) {
float x1990 = x23[x1989];
float x1991 = x1990 * 0.1f;
x1988[x1989] = x1991;

}
float* x1995 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1996=0; x1996 < 1300; x1996++) {
float x1997 = x97[x1996];
float x1998 = x1997 + 1.0E-8f;
x1995[x1996] = x1998;

}
float* x2002 = (float*)myMalloc(1300 * sizeof(float));;
for(int x2003=0; x2003 < 1300; x2003++) {
float x2004 = x1995[x2003];
double x2005 = (double)x2004;
double x2006 = sqrt(x2005);
float x2007 = (float)x2006;
x2002[x2003] = x2007;

}
float* x2011 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x2012 = 0;
int32_t x2013 = 0;
int32_t x2014 = 0;
for(int x2015=0; x2015 < 50; x2015++) {
int32_t x2016 = x2013;
int32_t x2017 = x2014;
int32_t x2018 = x2012;
int32_t x2019 = x2018;
int32_t x2020 = x2016;
int32_t x2021 = x2017;
for(int x2022=0; x2022 < 26; x2022++) {
int32_t x2023 = x2019;
int32_t x2024 = x2020;
float x2025 = x1988[x2024];
int32_t x2026 = x2021;
float x2027 = x2002[x2026];
float x2028 = x2025 / x2027;
x2011[x2023] = x2028;
x2019 += 1;
x2020 += 1;
x2021 += 1;

}
x2012 += 26;
x2013 += 26;
x2014 += 26;

}
for(int x2040=0; x2040 < 1300; x2040++) {
float x2041 = x14[x2040];
float x2042 = x2011[x2040];
float x2043 = x2041 - x2042;
x14[x2040] = x2043;

}
for(int x2047=0; x2047 < 1300; x2047++) {
float x2048 = x23[x2047];
x23[x2047] = 0.0f;

}
mallocAddr = (void*)x104;

}
double x2055 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x2058 = (long)fopen(x0, "w");
fprintf((FILE *)x2058, "unit: %s\n", "100 iteration");
for(int x2061=0; x2061 < 51; x2061++) {
double x2062 = x102[x2061];
fprintf((FILE *)x2058, "%lf\n", x2062);

}
double x2056 = x103 - x1;
double x2057 = x2055 - x103;
fprintf((FILE *)x2058, "run time: %lf %lf\n", x2056, x2057);
fclose((FILE*)x2058);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

