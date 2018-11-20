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
bool x785 = true || true;
bool x786 = x785 || true;
bool x1473 = true || false;
for(int x133=0; x133 < 5001; x133++) {
float* x158 = (float*)myMalloc(1 * sizeof(float));;
float* x159 = (float*)myMalloc(10400 * sizeof(float));;
float* x176 = (float*)myMalloc(10400 * sizeof(float));;
int* x143 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
function<void(int32_t,float**)> x809 = [&](int32_t x810,float** x811) {
float** x813 = x811;
float* x814 = x813[0];
float* x815 = x813[1];
float* x816 = x813[2];
float* x817 = x813[3];
float* x818 = x813[4];
float* x819 = x813[5];
int32_t x812 = x810;
bool x820 = x812 < 20;
if (x820) {
int32_t x821 = x812 * 520;
float* x822 = x159+x821;
float* x823 = x176+x821;
float* x824 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x822,26,x18,50,0,x824,50);
float* x826 = (float*)myMalloc(1000 * sizeof(float));;
float* x827 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x816,50,x29,50,0,x827,50);
float* x829 = (float*)myMalloc(1000 * sizeof(float));;
float* x830 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x831 = 0;
int32_t x832 = 0;
int32_t x833 = 0;
for(int x834=0; x834 < 20; x834++) {
int32_t x835 = x832;
int32_t x836 = x833;
int32_t x837 = x831;
int32_t x838 = x837;
int32_t x839 = x835;
int32_t x840 = x836;
for(int x841=0; x841 < 50; x841++) {
int32_t x842 = x838;
int32_t x843 = x839;
float x844 = x824[x843];
int32_t x845 = x840;
float x846 = x827[x845];
float x847 = x844 + x846;
x830[x842] = x847;
x838 += 1;
x839 += 1;
x840 += 1;

}
x831 += 50;
x832 += 50;
x833 += 50;

}
float* x859 = (float*)myMalloc(1000 * sizeof(float));;
float* x860 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x861 = 0;
int32_t x862 = 0;
int32_t x863 = 0;
for(int x864=0; x864 < 20; x864++) {
int32_t x865 = x862;
int32_t x866 = x863;
int32_t x867 = x861;
int32_t x868 = x867;
int32_t x869 = x865;
int32_t x870 = x866;
for(int x871=0; x871 < 50; x871++) {
int32_t x872 = x868;
int32_t x873 = x869;
float x874 = x830[x873];
int32_t x875 = x870;
float x876 = x39[x875];
float x877 = x874 + x876;
x860[x872] = x877;
x868 += 1;
x869 += 1;
x870 += 1;

}
x861 += 50;
x862 += 50;

}
float* x888 = (float*)myMalloc(1000 * sizeof(float));;
float* x889 = (float*)myMalloc(1000 * sizeof(float));;
for(int x890=0; x890 < 1000; x890++) {
float x891 = x860[x890];
float x892 = -1.0f * x891;
double x893 = (double)x892;
double x894 = exp(x893);
float x895 = (float)x894;
float x896 = x895 + 1.0f;
float x897 = 1.0f / x896;
x889[x890] = x897;

}
float* x901 = (float*)myMalloc(1000 * sizeof(float));;
float* x902 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x822,26,x41,50,0,x902,50);
float* x904 = (float*)myMalloc(1000 * sizeof(float));;
float* x905 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x816,50,x50,50,0,x905,50);
float* x907 = (float*)myMalloc(1000 * sizeof(float));;
float* x908 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x909 = 0;
int32_t x910 = 0;
int32_t x911 = 0;
for(int x912=0; x912 < 20; x912++) {
int32_t x913 = x910;
int32_t x914 = x911;
int32_t x915 = x909;
int32_t x916 = x915;
int32_t x917 = x913;
int32_t x918 = x914;
for(int x919=0; x919 < 50; x919++) {
int32_t x920 = x916;
int32_t x921 = x917;
float x922 = x902[x921];
int32_t x923 = x918;
float x924 = x905[x923];
float x925 = x922 + x924;
x908[x920] = x925;
x916 += 1;
x917 += 1;
x918 += 1;

}
x909 += 50;
x910 += 50;
x911 += 50;

}
float* x937 = (float*)myMalloc(1000 * sizeof(float));;
float* x938 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x939 = 0;
int32_t x940 = 0;
int32_t x941 = 0;
for(int x942=0; x942 < 20; x942++) {
int32_t x943 = x940;
int32_t x944 = x941;
int32_t x945 = x939;
int32_t x946 = x945;
int32_t x947 = x943;
int32_t x948 = x944;
for(int x949=0; x949 < 50; x949++) {
int32_t x950 = x946;
int32_t x951 = x947;
float x952 = x908[x951];
int32_t x953 = x948;
float x954 = x59[x953];
float x955 = x952 + x954;
x938[x950] = x955;
x946 += 1;
x947 += 1;
x948 += 1;

}
x939 += 50;
x940 += 50;

}
float* x966 = (float*)myMalloc(1000 * sizeof(float));;
float* x967 = (float*)myMalloc(1000 * sizeof(float));;
for(int x968=0; x968 < 1000; x968++) {
float x969 = x938[x968];
float x970 = -1.0f * x969;
double x971 = (double)x970;
double x972 = exp(x971);
float x973 = (float)x972;
float x974 = x973 + 1.0f;
float x975 = 1.0f / x974;
x967[x968] = x975;

}
float* x979 = (float*)myMalloc(1000 * sizeof(float));;
float* x980 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x822,26,x81,50,0,x980,50);
float* x982 = (float*)myMalloc(1000 * sizeof(float));;
float* x983 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x816,50,x90,50,0,x983,50);
float* x985 = (float*)myMalloc(1000 * sizeof(float));;
float* x986 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x987 = 0;
int32_t x988 = 0;
int32_t x989 = 0;
for(int x990=0; x990 < 20; x990++) {
int32_t x991 = x988;
int32_t x992 = x989;
int32_t x993 = x987;
int32_t x994 = x993;
int32_t x995 = x991;
int32_t x996 = x992;
for(int x997=0; x997 < 50; x997++) {
int32_t x998 = x994;
int32_t x999 = x995;
float x1000 = x980[x999];
int32_t x1001 = x996;
float x1002 = x983[x1001];
float x1003 = x1000 + x1002;
x986[x998] = x1003;
x994 += 1;
x995 += 1;
x996 += 1;

}
x987 += 50;
x988 += 50;
x989 += 50;

}
float* x1015 = (float*)myMalloc(1000 * sizeof(float));;
float* x1016 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1017 = 0;
int32_t x1018 = 0;
int32_t x1019 = 0;
for(int x1020=0; x1020 < 20; x1020++) {
int32_t x1021 = x1018;
int32_t x1022 = x1019;
int32_t x1023 = x1017;
int32_t x1024 = x1023;
int32_t x1025 = x1021;
int32_t x1026 = x1022;
for(int x1027=0; x1027 < 50; x1027++) {
int32_t x1028 = x1024;
int32_t x1029 = x1025;
float x1030 = x986[x1029];
int32_t x1031 = x1026;
float x1032 = x99[x1031];
float x1033 = x1030 + x1032;
x1016[x1028] = x1033;
x1024 += 1;
x1025 += 1;
x1026 += 1;

}
x1017 += 50;
x1018 += 50;

}
float* x1044 = (float*)myMalloc(1000 * sizeof(float));;
float* x1045 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1046=0; x1046 < 1000; x1046++) {
float x1047 = x1016[x1046];
float x1048 = -1.0f * x1047;
double x1049 = (double)x1048;
double x1050 = exp(x1049);
float x1051 = (float)x1050;
float x1052 = x1051 + 1.0f;
float x1053 = 1.0f / x1052;
x1045[x1046] = x1053;

}
float* x1057 = (float*)myMalloc(1000 * sizeof(float));;
float* x1058 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x822,26,x61,50,0,x1058,50);
float* x1060 = (float*)myMalloc(1000 * sizeof(float));;
float* x1061 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x816,50,x70,50,0,x1061,50);
float* x1063 = (float*)myMalloc(1000 * sizeof(float));;
float* x1064 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1065 = 0;
int32_t x1066 = 0;
int32_t x1067 = 0;
for(int x1068=0; x1068 < 20; x1068++) {
int32_t x1069 = x1066;
int32_t x1070 = x1067;
int32_t x1071 = x1065;
int32_t x1072 = x1071;
int32_t x1073 = x1069;
int32_t x1074 = x1070;
for(int x1075=0; x1075 < 50; x1075++) {
int32_t x1076 = x1072;
int32_t x1077 = x1073;
float x1078 = x1058[x1077];
int32_t x1079 = x1074;
float x1080 = x1061[x1079];
float x1081 = x1078 + x1080;
x1064[x1076] = x1081;
x1072 += 1;
x1073 += 1;
x1074 += 1;

}
x1065 += 50;
x1066 += 50;
x1067 += 50;

}
float* x1093 = (float*)myMalloc(1000 * sizeof(float));;
float* x1094 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1095 = 0;
int32_t x1096 = 0;
int32_t x1097 = 0;
for(int x1098=0; x1098 < 20; x1098++) {
int32_t x1099 = x1096;
int32_t x1100 = x1097;
int32_t x1101 = x1095;
int32_t x1102 = x1101;
int32_t x1103 = x1099;
int32_t x1104 = x1100;
for(int x1105=0; x1105 < 50; x1105++) {
int32_t x1106 = x1102;
int32_t x1107 = x1103;
float x1108 = x1064[x1107];
int32_t x1109 = x1104;
float x1110 = x79[x1109];
float x1111 = x1108 + x1110;
x1094[x1106] = x1111;
x1102 += 1;
x1103 += 1;
x1104 += 1;

}
x1095 += 50;
x1096 += 50;

}
float* x1122 = (float*)myMalloc(1000 * sizeof(float));;
float* x1123 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1124=0; x1124 < 1000; x1124++) {
float x1125 = x1094[x1124];
double x1126 = (double)x1125;
double x1127 = tanh(x1126);
float x1128 = (float)x1127;
x1123[x1124] = x1128;

}
float* x1132 = (float*)myMalloc(1000 * sizeof(float));;
float* x1133 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1134 = 0;
int32_t x1135 = 0;
int32_t x1136 = 0;
for(int x1137=0; x1137 < 20; x1137++) {
int32_t x1138 = x1135;
int32_t x1139 = x1136;
int32_t x1140 = x1134;
int32_t x1141 = x1140;
int32_t x1142 = x1138;
int32_t x1143 = x1139;
for(int x1144=0; x1144 < 50; x1144++) {
int32_t x1145 = x1141;
int32_t x1146 = x1142;
float x1147 = x889[x1146];
int32_t x1148 = x1143;
float x1149 = x818[x1148];
float x1150 = x1147 * x1149;
x1133[x1145] = x1150;
x1141 += 1;
x1142 += 1;
x1143 += 1;

}
x1134 += 50;
x1135 += 50;
x1136 += 50;

}
float* x1162 = (float*)myMalloc(1000 * sizeof(float));;
float* x1163 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1164 = 0;
int32_t x1165 = 0;
int32_t x1166 = 0;
for(int x1167=0; x1167 < 20; x1167++) {
int32_t x1168 = x1165;
int32_t x1169 = x1166;
int32_t x1170 = x1164;
int32_t x1171 = x1170;
int32_t x1172 = x1168;
int32_t x1173 = x1169;
for(int x1174=0; x1174 < 50; x1174++) {
int32_t x1175 = x1171;
int32_t x1176 = x1172;
float x1177 = x967[x1176];
int32_t x1178 = x1173;
float x1179 = x1123[x1178];
float x1180 = x1177 * x1179;
x1163[x1175] = x1180;
x1171 += 1;
x1172 += 1;
x1173 += 1;

}
x1164 += 50;
x1165 += 50;
x1166 += 50;

}
float* x1192 = (float*)myMalloc(1000 * sizeof(float));;
float* x1193 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1194 = 0;
int32_t x1195 = 0;
int32_t x1196 = 0;
for(int x1197=0; x1197 < 20; x1197++) {
int32_t x1198 = x1195;
int32_t x1199 = x1196;
int32_t x1200 = x1194;
int32_t x1201 = x1200;
int32_t x1202 = x1198;
int32_t x1203 = x1199;
for(int x1204=0; x1204 < 50; x1204++) {
int32_t x1205 = x1201;
int32_t x1206 = x1202;
float x1207 = x1133[x1206];
int32_t x1208 = x1203;
float x1209 = x1163[x1208];
float x1210 = x1207 + x1209;
x1193[x1205] = x1210;
x1201 += 1;
x1202 += 1;
x1203 += 1;

}
x1194 += 50;
x1195 += 50;
x1196 += 50;

}
float* x1222 = (float*)myMalloc(1000 * sizeof(float));;
float* x1223 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1224=0; x1224 < 1000; x1224++) {
float x1225 = x1193[x1224];
double x1226 = (double)x1225;
double x1227 = tanh(x1226);
float x1228 = (float)x1227;
x1223[x1224] = x1228;

}
float* x1232 = (float*)myMalloc(1000 * sizeof(float));;
float* x1233 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1234 = 0;
int32_t x1235 = 0;
int32_t x1236 = 0;
for(int x1237=0; x1237 < 20; x1237++) {
int32_t x1238 = x1235;
int32_t x1239 = x1236;
int32_t x1240 = x1234;
int32_t x1241 = x1240;
int32_t x1242 = x1238;
int32_t x1243 = x1239;
for(int x1244=0; x1244 < 50; x1244++) {
int32_t x1245 = x1241;
int32_t x1246 = x1242;
float x1247 = x1045[x1246];
int32_t x1248 = x1243;
float x1249 = x1223[x1248];
float x1250 = x1247 * x1249;
x1233[x1245] = x1250;
x1241 += 1;
x1242 += 1;
x1243 += 1;

}
x1234 += 50;
x1235 += 50;
x1236 += 50;

}
float* x1262 = (float*)myMalloc(1000 * sizeof(float));;
float* x1263 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x1233,50,x101,26,0,x1263,26);
float* x1265 = (float*)myMalloc(520 * sizeof(float));;
int32_t x1266 = 0;
int32_t x1267 = 0;
int32_t x1268 = 0;
for(int x1269=0; x1269 < 20; x1269++) {
int32_t x1270 = x1267;
int32_t x1271 = x1268;
int32_t x1272 = x1266;
int32_t x1273 = x1272;
int32_t x1274 = x1270;
int32_t x1275 = x1271;
for(int x1276=0; x1276 < 26; x1276++) {
int32_t x1277 = x1274;
float x1278 = x1263[x1277];
int32_t x1279 = x1275;
float x1280 = x110[x1279];
float x1281 = x1278 + x1280;
x1263[x1277] = x1281;
x1273 += 1;
x1274 += 1;
x1275 += 1;

}
x1266 += 26;
x1267 += 26;

}
int* x1292 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x1293=0; x1293 < 20; x1293++) {
int32_t x1294 = x1293 * 20;
int32_t x1295 = x812 + x1294;
int32_t x1296 = x143[x1295];
x1292[x1293] = x1296;

}
float* x1300 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1301 = 0;
for(int x1302=0; x1302 < 20; x1302++) {
float x1303 = -3.4028235E38f;
for(int x1304=0; x1304 < 26; x1304++) {
int32_t x1305 = x1301;
float x1306 = x1263[x1305];
float x1307 = x1303;
bool x1308 = x1306 > x1307;
if (x1308) {
float x1309 = x1263[x1305];
x1303 = x1309;
} else {
}
x1301 += 1;

}
float x1316 = x1303;
x1300[x1302] = x1316;

}
float* x1320 = (float*)myMalloc(520 * sizeof(float));;
int32_t x1321 = 0;
for(int x1322=0; x1322 < 20; x1322++) {
for(int x1323=0; x1323 < 26; x1323++) {
int32_t x1324 = x1321;
float x1325 = x1263[x1324];
float x1326 = x1300[x1322];
float x1327 = x1325 - x1326;
double x1328 = (double)x1327;
double x1329 = exp(x1328);
float x1330 = (float)x1329;
x1320[x1324] = x1330;
x1321 += 1;

}

}
float* x1337 = (float*)myMalloc(20 * sizeof(float));;
for(int x1338=0; x1338 < 20; x1338++) {
int32_t x1339 = x1338;
int32_t x1340 = x1338 * 26;
int32_t x1341 = x1340;
for(int x1342=0; x1342 < 26; x1342++) {
for(int x1343=0; x1343 < 1; x1343++) {
int32_t x1344 = x1339;
int32_t x1345 = x1344 + x1343;
float x1346 = x1337[x1345];
int32_t x1347 = x1341;
int32_t x1348 = x1347 + x1343;
float x1349 = x1320[x1348];
float x1350 = x1346 + x1349;
x1337[x1345] = x1350;

}
x1341 += 1;

}

}
x1321 = 0;
for(int x1360=0; x1360 < 20; x1360++) {
float x1361 = x1300[x1360];
float x1362 = x1337[x1360];
double x1363 = (double)x1362;
double x1364 = log(x1363);
float x1365 = (float)x1364;
float x1366 = x1361 + x1365;
for(int x1367=0; x1367 < 26; x1367++) {
int32_t x1368 = x1321;
float x1369 = x1263[x1368];
float x1370 = x1369 - x1366;
x1320[x1368] = x1370;
x1321 += 1;

}

}
float* x1377 = (float*)myMalloc(520 * sizeof(float));;
// nllLoss forward in CPU
float* x1379 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1380 = 0;
for(int x1381=0; x1381 < 20; x1381++) {
int32_t x1382 = x1380;
int32_t x1383 = x1292[x1381];
int32_t x1384 = x1382 + x1383;
float x1385 = x1320[x1384];
float x1386 = -1.0f * x1385;
x1379[x1381] = x1386;
x1380 += 26;

}
float* x1391 = (float*)myMalloc(20 * sizeof(float));;
float x1392 = 0.0f;
for(int x1393=0; x1393 < 20; x1393++) {
float x1394 = x1392;
float x1395 = x1379[x1393];
float x1396 = x1394 + x1395;
x1392 = x1396;

}
float x1400 = x1392;
float* x1401 = (float*)myMalloc(1 * sizeof(float));;
for(int x1402=0; x1402 < 1; x1402++) {
x1401[x1402] = x1400;

}
float* x1406 = (float*)myMalloc(1 * sizeof(float));;
if (x786) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x1411 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1412 = 0;
int32_t x1413 = 0;
int32_t x1414 = 0;
for(int x1415=0; x1415 < 1; x1415++) {
int32_t x1416 = x1412;
int32_t x1417 = x1413;
float x1418 = x814[x1417];
int32_t x1419 = x1414;
float x1420 = x1401[x1419];
float x1421 = x1418 + x1420;
x1411[x1416] = x1421;
x1412 += 1;

}
float* x1426 = (float*)myMalloc(1 * sizeof(float));;
float** x1428 = (float**)myMalloc(6 * sizeof(float*));;
x1428[0] = x1411;
x1428[1] = x1426;
x1428[2] = x1233;
x1428[3] = x1262;
x1428[4] = x1193;
x1428[5] = x1222;
int32_t x1442 = 0;
int32_t x1443 = 0;
int32_t x1444 = 0;
int32_t x1459 = 0;
int32_t x1460 = 0;
int32_t x1461 = 0;
int32_t x1479 = 0;
int32_t x1480 = 0;
int32_t x1481 = 0;
int32_t x1495 = 0;
float* x1508 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1530 = 0;
int32_t x1550 = 0;
int32_t x1551 = 0;
int32_t x1552 = 0;
int32_t x1580 = 0;
int32_t x1581 = 0;
int32_t x1582 = 0;
int32_t x1630 = 0;
int32_t x1631 = 0;
int32_t x1632 = 0;
int32_t x1657 = 0;
int32_t x1658 = 0;
int32_t x1659 = 0;
int32_t x1685 = 0;
int32_t x1686 = 0;
int32_t x1687 = 0;
int32_t x1724 = 0;
int32_t x1725 = 0;
int32_t x1726 = 0;
int32_t x1774 = 0;
int32_t x1775 = 0;
int32_t x1776 = 0;
int32_t x1801 = 0;
int32_t x1802 = 0;
int32_t x1803 = 0;
int32_t x1828 = 0;
int32_t x1829 = 0;
int32_t x1830 = 0;
int32_t x1855 = 0;
int32_t x1856 = 0;
int32_t x1857 = 0;
int32_t x1900 = 0;
int32_t x1901 = 0;
int32_t x1902 = 0;
int32_t x1927 = 0;
int32_t x1928 = 0;
int32_t x1929 = 0;
int32_t x1954 = 0;
int32_t x1955 = 0;
int32_t x1956 = 0;
int32_t x1981 = 0;
int32_t x1982 = 0;
int32_t x1983 = 0;
int32_t x2026 = 0;
int32_t x2027 = 0;
int32_t x2028 = 0;
int32_t x2053 = 0;
int32_t x2054 = 0;
int32_t x2055 = 0;
int32_t x2080 = 0;
int32_t x2081 = 0;
int32_t x2082 = 0;
int32_t x2107 = 0;
int32_t x2108 = 0;
int32_t x2109 = 0;
int32_t x2152 = 0;
int32_t x2153 = 0;
int32_t x2154 = 0;
int32_t x2179 = 0;
int32_t x2180 = 0;
int32_t x2181 = 0;
int32_t x2206 = 0;
int32_t x2207 = 0;
int32_t x2208 = 0;
int32_t x2233 = 0;
int32_t x2234 = 0;
int32_t x2235 = 0;
int32_t x1427 = x812 + 1;
x809(x1427,x1428);
// back prop for + op
if (x786) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1445=0; x1445 < 1; x1445++) {
int32_t x1446 = x1443;
float x1447 = x815[x1446];
int32_t x1448 = x1444;
float x1449 = x1426[x1448];
float x1450 = x1447 + x1449;
x815[x1446] = x1450;
x1442 += 1;

}
if (x786) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
for(int x1462=0; x1462 < 1; x1462++) {
int32_t x1463 = x1460;
float x1464 = x1406[x1463];
int32_t x1465 = x1461;
float x1466 = x1426[x1465];
float x1467 = x1464 + x1466;
x1406[x1463] = x1467;
x1459 += 1;

}
// 'sum' gradient.
if (x1473) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",20,1);
assert(false && "");
}
for(int x1482=0; x1482 < 20; x1482++) {
int32_t x1483 = x1480;
float x1484 = x1391[x1483];
int32_t x1485 = x1481;
float x1486 = x1406[x1485];
float x1487 = x1484 + x1486;
x1391[x1483] = x1487;
x1479 += 1;
x1480 += 1;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
for(int x1496=0; x1496 < 20; x1496++) {
int32_t x1497 = x1495;
int32_t x1498 = x1292[x1496];
int32_t x1499 = x1497 + x1498;
float x1500 = x1377[x1499];
float x1501 = x1391[x1496];
float x1502 = -1.0f * x1501;
float x1503 = x1500 + x1502;
x1377[x1499] = x1503;
x1495 += 26;

}
for(int x1509=0; x1509 < 20; x1509++) {
int32_t x1510 = x1509;
int32_t x1511 = x1509 * 26;
int32_t x1512 = x1511;
for(int x1513=0; x1513 < 26; x1513++) {
for(int x1514=0; x1514 < 1; x1514++) {
int32_t x1515 = x1510;
int32_t x1516 = x1515 + x1514;
float x1517 = x1508[x1516];
int32_t x1518 = x1512;
int32_t x1519 = x1518 + x1514;
float x1520 = x1377[x1519];
float x1521 = x1517 + x1520;
x1508[x1516] = x1521;

}
x1512 += 1;

}

}
for(int x1531=0; x1531 < 20; x1531++) {
for(int x1532=0; x1532 < 26; x1532++) {
int32_t x1533 = x1530;
float x1534 = x1265[x1533];
float x1535 = x1377[x1533];
float x1536 = x1320[x1533];
float x1540 = x1508[x1531];
double x1537 = (double)x1536;
double x1538 = exp(x1537);
float x1539 = (float)x1538;
float x1541 = x1539 * x1540;
float x1542 = x1535 - x1541;
float x1543 = x1534 + x1542;
x1265[x1533] = x1543;
x1530 += 1;

}

}
for(int x1553=0; x1553 < 20; x1553++) {
int32_t x1554 = x1551;
int32_t x1555 = x1552;
int32_t x1556 = x1550;
int32_t x1557 = x1556;
int32_t x1558 = x1554;
int32_t x1559 = x1555;
for(int x1560=0; x1560 < 26; x1560++) {
int32_t x1561 = x1558;
float x1562 = x111[x1561];
int32_t x1563 = x1559;
float x1564 = x1265[x1563];
float x1565 = x1562 + x1564;
x111[x1561] = x1565;
x1557 += 1;
x1558 += 1;
x1559 += 1;

}
x1550 += 26;
x1552 += 26;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x1265,26,x101,26,1,x1262,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x1233,50,x1265,26,1,x109,26);
// backprop for * op
for(int x1583=0; x1583 < 20; x1583++) {
int32_t x1584 = x1580;
int32_t x1585 = x1581;
int32_t x1586 = x1582;
int32_t x1587 = x1584;
int32_t x1588 = x1585;
int32_t x1589 = x1586;
for(int x1590=0; x1590 < 50; x1590++) {
int32_t x1591 = x1587;
float x1592 = x1057[x1591];
float x1593 = x1045[x1591];
int32_t x1594 = x1588;
float x1595 = x1223[x1594];
int32_t x1596 = x1589;
float x1597 = x1262[x1596];
float x1598 = x1597 * x1595;
float x1599 = x1592 + x1598;
x1057[x1591] = x1599;
float x1601 = x1232[x1594];
float x1602 = x1045[x1591];
float x1603 = x1223[x1594];
float x1604 = x1262[x1596];
float x1605 = x1604 * x1602;
float x1606 = x1601 + x1605;
x1232[x1594] = x1606;
x1589 += 1;
x1587 += 1;
x1588 += 1;

}
x1582 += 50;
x1580 += 50;
x1581 += 50;

}
for(int x1618=0; x1618 < 1000; x1618++) {
float x1619 = x1222[x1618];
float x1620 = x1223[x1618];
float x1623 = x1232[x1618];
float x1621 = x1620 * x1620;
float x1622 = 1.0f - x1621;
float x1624 = x1622 * x1623;
float x1625 = x1619 + x1624;
x1222[x1618] = x1625;

}
// back prop for + op
for(int x1633=0; x1633 < 20; x1633++) {
int32_t x1634 = x1631;
int32_t x1635 = x1632;
int32_t x1636 = x1630;
int32_t x1637 = x1636;
int32_t x1638 = x1634;
int32_t x1639 = x1635;
for(int x1640=0; x1640 < 50; x1640++) {
int32_t x1641 = x1638;
float x1642 = x1162[x1641];
int32_t x1643 = x1639;
float x1644 = x1222[x1643];
float x1645 = x1642 + x1644;
x1162[x1641] = x1645;
x1637 += 1;
x1638 += 1;
x1639 += 1;

}
x1630 += 50;
x1631 += 50;
x1632 += 50;

}
for(int x1660=0; x1660 < 20; x1660++) {
int32_t x1661 = x1658;
int32_t x1662 = x1659;
int32_t x1663 = x1657;
int32_t x1664 = x1663;
int32_t x1665 = x1661;
int32_t x1666 = x1662;
for(int x1667=0; x1667 < 50; x1667++) {
int32_t x1668 = x1665;
float x1669 = x1192[x1668];
int32_t x1670 = x1666;
float x1671 = x1222[x1670];
float x1672 = x1669 + x1671;
x1192[x1668] = x1672;
x1664 += 1;
x1665 += 1;
x1666 += 1;

}
x1657 += 50;
x1658 += 50;
x1659 += 50;

}
// backprop for * op
for(int x1688=0; x1688 < 20; x1688++) {
int32_t x1689 = x1685;
int32_t x1690 = x1686;
int32_t x1691 = x1687;
int32_t x1692 = x1689;
int32_t x1693 = x1690;
int32_t x1694 = x1691;
for(int x1695=0; x1695 < 50; x1695++) {
int32_t x1696 = x1692;
float x1697 = x979[x1696];
float x1698 = x967[x1696];
int32_t x1699 = x1693;
float x1700 = x1123[x1699];
int32_t x1701 = x1694;
float x1702 = x1192[x1701];
float x1703 = x1702 * x1700;
float x1704 = x1697 + x1703;
x979[x1696] = x1704;
float x1706 = x1132[x1699];
float x1707 = x967[x1696];
float x1708 = x1123[x1699];
float x1709 = x1192[x1701];
float x1710 = x1709 * x1707;
float x1711 = x1706 + x1710;
x1132[x1699] = x1711;
x1694 += 1;
x1692 += 1;
x1693 += 1;

}
x1687 += 50;
x1685 += 50;
x1686 += 50;

}
// backprop for * op
for(int x1727=0; x1727 < 20; x1727++) {
int32_t x1728 = x1724;
int32_t x1729 = x1725;
int32_t x1730 = x1726;
int32_t x1731 = x1728;
int32_t x1732 = x1729;
int32_t x1733 = x1730;
for(int x1734=0; x1734 < 50; x1734++) {
int32_t x1735 = x1731;
float x1736 = x901[x1735];
float x1737 = x889[x1735];
int32_t x1738 = x1732;
float x1739 = x818[x1738];
int32_t x1740 = x1733;
float x1741 = x1162[x1740];
float x1742 = x1741 * x1739;
float x1743 = x1736 + x1742;
x901[x1735] = x1743;
float x1745 = x819[x1738];
float x1746 = x889[x1735];
float x1747 = x818[x1738];
float x1748 = x1162[x1740];
float x1749 = x1748 * x1746;
float x1750 = x1745 + x1749;
x819[x1738] = x1750;
x1733 += 1;
x1731 += 1;
x1732 += 1;

}
x1726 += 50;
x1724 += 50;
x1725 += 50;

}
for(int x1762=0; x1762 < 1000; x1762++) {
float x1763 = x1122[x1762];
float x1764 = x1123[x1762];
float x1767 = x1132[x1762];
float x1765 = x1764 * x1764;
float x1766 = 1.0f - x1765;
float x1768 = x1766 * x1767;
float x1769 = x1763 + x1768;
x1122[x1762] = x1769;

}
// back prop for + op
for(int x1777=0; x1777 < 20; x1777++) {
int32_t x1778 = x1775;
int32_t x1779 = x1776;
int32_t x1780 = x1774;
int32_t x1781 = x1780;
int32_t x1782 = x1778;
int32_t x1783 = x1779;
for(int x1784=0; x1784 < 50; x1784++) {
int32_t x1785 = x1782;
float x1786 = x1093[x1785];
int32_t x1787 = x1783;
float x1788 = x1122[x1787];
float x1789 = x1786 + x1788;
x1093[x1785] = x1789;
x1781 += 1;
x1782 += 1;
x1783 += 1;

}
x1774 += 50;
x1775 += 50;
x1776 += 50;

}
for(int x1804=0; x1804 < 20; x1804++) {
int32_t x1805 = x1802;
int32_t x1806 = x1803;
int32_t x1807 = x1801;
int32_t x1808 = x1807;
int32_t x1809 = x1805;
int32_t x1810 = x1806;
for(int x1811=0; x1811 < 50; x1811++) {
int32_t x1812 = x1809;
float x1813 = x80[x1812];
int32_t x1814 = x1810;
float x1815 = x1122[x1814];
float x1816 = x1813 + x1815;
x80[x1812] = x1816;
x1808 += 1;
x1809 += 1;
x1810 += 1;

}
x1801 += 50;
x1803 += 50;

}
// back prop for + op
for(int x1831=0; x1831 < 20; x1831++) {
int32_t x1832 = x1829;
int32_t x1833 = x1830;
int32_t x1834 = x1828;
int32_t x1835 = x1834;
int32_t x1836 = x1832;
int32_t x1837 = x1833;
for(int x1838=0; x1838 < 50; x1838++) {
int32_t x1839 = x1836;
float x1840 = x1060[x1839];
int32_t x1841 = x1837;
float x1842 = x1093[x1841];
float x1843 = x1840 + x1842;
x1060[x1839] = x1843;
x1835 += 1;
x1836 += 1;
x1837 += 1;

}
x1828 += 50;
x1829 += 50;
x1830 += 50;

}
for(int x1858=0; x1858 < 20; x1858++) {
int32_t x1859 = x1856;
int32_t x1860 = x1857;
int32_t x1861 = x1855;
int32_t x1862 = x1861;
int32_t x1863 = x1859;
int32_t x1864 = x1860;
for(int x1865=0; x1865 < 50; x1865++) {
int32_t x1866 = x1863;
float x1867 = x1063[x1866];
int32_t x1868 = x1864;
float x1869 = x1093[x1868];
float x1870 = x1867 + x1869;
x1063[x1866] = x1870;
x1862 += 1;
x1863 += 1;
x1864 += 1;

}
x1855 += 50;
x1856 += 50;
x1857 += 50;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x1063,50,x70,50,1,x817,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x816,50,x1063,50,1,x78,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x1060,50,x61,50,1,x823,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x822,26,x1060,50,1,x69,50);
for(int x1888=0; x1888 < 1000; x1888++) {
float x1889 = x1044[x1888];
float x1890 = x1045[x1888];
float x1893 = x1057[x1888];
float x1891 = 1.0f - x1890;
float x1892 = x1891 * x1890;
float x1894 = x1892 * x1893;
float x1895 = x1889 + x1894;
x1044[x1888] = x1895;

}
// back prop for + op
for(int x1903=0; x1903 < 20; x1903++) {
int32_t x1904 = x1901;
int32_t x1905 = x1902;
int32_t x1906 = x1900;
int32_t x1907 = x1906;
int32_t x1908 = x1904;
int32_t x1909 = x1905;
for(int x1910=0; x1910 < 50; x1910++) {
int32_t x1911 = x1908;
float x1912 = x1015[x1911];
int32_t x1913 = x1909;
float x1914 = x1044[x1913];
float x1915 = x1912 + x1914;
x1015[x1911] = x1915;
x1907 += 1;
x1908 += 1;
x1909 += 1;

}
x1900 += 50;
x1901 += 50;
x1902 += 50;

}
for(int x1930=0; x1930 < 20; x1930++) {
int32_t x1931 = x1928;
int32_t x1932 = x1929;
int32_t x1933 = x1927;
int32_t x1934 = x1933;
int32_t x1935 = x1931;
int32_t x1936 = x1932;
for(int x1937=0; x1937 < 50; x1937++) {
int32_t x1938 = x1935;
float x1939 = x100[x1938];
int32_t x1940 = x1936;
float x1941 = x1044[x1940];
float x1942 = x1939 + x1941;
x100[x1938] = x1942;
x1934 += 1;
x1935 += 1;
x1936 += 1;

}
x1927 += 50;
x1929 += 50;

}
// back prop for + op
for(int x1957=0; x1957 < 20; x1957++) {
int32_t x1958 = x1955;
int32_t x1959 = x1956;
int32_t x1960 = x1954;
int32_t x1961 = x1960;
int32_t x1962 = x1958;
int32_t x1963 = x1959;
for(int x1964=0; x1964 < 50; x1964++) {
int32_t x1965 = x1962;
float x1966 = x982[x1965];
int32_t x1967 = x1963;
float x1968 = x1015[x1967];
float x1969 = x1966 + x1968;
x982[x1965] = x1969;
x1961 += 1;
x1962 += 1;
x1963 += 1;

}
x1954 += 50;
x1955 += 50;
x1956 += 50;

}
for(int x1984=0; x1984 < 20; x1984++) {
int32_t x1985 = x1982;
int32_t x1986 = x1983;
int32_t x1987 = x1981;
int32_t x1988 = x1987;
int32_t x1989 = x1985;
int32_t x1990 = x1986;
for(int x1991=0; x1991 < 50; x1991++) {
int32_t x1992 = x1989;
float x1993 = x985[x1992];
int32_t x1994 = x1990;
float x1995 = x1015[x1994];
float x1996 = x1993 + x1995;
x985[x1992] = x1996;
x1988 += 1;
x1989 += 1;
x1990 += 1;

}
x1981 += 50;
x1982 += 50;
x1983 += 50;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x985,50,x90,50,1,x817,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x816,50,x985,50,1,x98,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x982,50,x81,50,1,x823,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x822,26,x982,50,1,x89,50);
for(int x2014=0; x2014 < 1000; x2014++) {
float x2015 = x966[x2014];
float x2016 = x967[x2014];
float x2019 = x979[x2014];
float x2017 = 1.0f - x2016;
float x2018 = x2017 * x2016;
float x2020 = x2018 * x2019;
float x2021 = x2015 + x2020;
x966[x2014] = x2021;

}
// back prop for + op
for(int x2029=0; x2029 < 20; x2029++) {
int32_t x2030 = x2027;
int32_t x2031 = x2028;
int32_t x2032 = x2026;
int32_t x2033 = x2032;
int32_t x2034 = x2030;
int32_t x2035 = x2031;
for(int x2036=0; x2036 < 50; x2036++) {
int32_t x2037 = x2034;
float x2038 = x937[x2037];
int32_t x2039 = x2035;
float x2040 = x966[x2039];
float x2041 = x2038 + x2040;
x937[x2037] = x2041;
x2033 += 1;
x2034 += 1;
x2035 += 1;

}
x2026 += 50;
x2027 += 50;
x2028 += 50;

}
for(int x2056=0; x2056 < 20; x2056++) {
int32_t x2057 = x2054;
int32_t x2058 = x2055;
int32_t x2059 = x2053;
int32_t x2060 = x2059;
int32_t x2061 = x2057;
int32_t x2062 = x2058;
for(int x2063=0; x2063 < 50; x2063++) {
int32_t x2064 = x2061;
float x2065 = x60[x2064];
int32_t x2066 = x2062;
float x2067 = x966[x2066];
float x2068 = x2065 + x2067;
x60[x2064] = x2068;
x2060 += 1;
x2061 += 1;
x2062 += 1;

}
x2053 += 50;
x2055 += 50;

}
// back prop for + op
for(int x2083=0; x2083 < 20; x2083++) {
int32_t x2084 = x2081;
int32_t x2085 = x2082;
int32_t x2086 = x2080;
int32_t x2087 = x2086;
int32_t x2088 = x2084;
int32_t x2089 = x2085;
for(int x2090=0; x2090 < 50; x2090++) {
int32_t x2091 = x2088;
float x2092 = x904[x2091];
int32_t x2093 = x2089;
float x2094 = x937[x2093];
float x2095 = x2092 + x2094;
x904[x2091] = x2095;
x2087 += 1;
x2088 += 1;
x2089 += 1;

}
x2080 += 50;
x2081 += 50;
x2082 += 50;

}
for(int x2110=0; x2110 < 20; x2110++) {
int32_t x2111 = x2108;
int32_t x2112 = x2109;
int32_t x2113 = x2107;
int32_t x2114 = x2113;
int32_t x2115 = x2111;
int32_t x2116 = x2112;
for(int x2117=0; x2117 < 50; x2117++) {
int32_t x2118 = x2115;
float x2119 = x907[x2118];
int32_t x2120 = x2116;
float x2121 = x937[x2120];
float x2122 = x2119 + x2121;
x907[x2118] = x2122;
x2114 += 1;
x2115 += 1;
x2116 += 1;

}
x2107 += 50;
x2108 += 50;
x2109 += 50;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x907,50,x50,50,1,x817,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x816,50,x907,50,1,x58,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x904,50,x41,50,1,x823,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x822,26,x904,50,1,x49,50);
for(int x2140=0; x2140 < 1000; x2140++) {
float x2141 = x888[x2140];
float x2142 = x889[x2140];
float x2145 = x901[x2140];
float x2143 = 1.0f - x2142;
float x2144 = x2143 * x2142;
float x2146 = x2144 * x2145;
float x2147 = x2141 + x2146;
x888[x2140] = x2147;

}
// back prop for + op
for(int x2155=0; x2155 < 20; x2155++) {
int32_t x2156 = x2153;
int32_t x2157 = x2154;
int32_t x2158 = x2152;
int32_t x2159 = x2158;
int32_t x2160 = x2156;
int32_t x2161 = x2157;
for(int x2162=0; x2162 < 50; x2162++) {
int32_t x2163 = x2160;
float x2164 = x859[x2163];
int32_t x2165 = x2161;
float x2166 = x888[x2165];
float x2167 = x2164 + x2166;
x859[x2163] = x2167;
x2159 += 1;
x2160 += 1;
x2161 += 1;

}
x2152 += 50;
x2153 += 50;
x2154 += 50;

}
for(int x2182=0; x2182 < 20; x2182++) {
int32_t x2183 = x2180;
int32_t x2184 = x2181;
int32_t x2185 = x2179;
int32_t x2186 = x2185;
int32_t x2187 = x2183;
int32_t x2188 = x2184;
for(int x2189=0; x2189 < 50; x2189++) {
int32_t x2190 = x2187;
float x2191 = x40[x2190];
int32_t x2192 = x2188;
float x2193 = x888[x2192];
float x2194 = x2191 + x2193;
x40[x2190] = x2194;
x2186 += 1;
x2187 += 1;
x2188 += 1;

}
x2179 += 50;
x2181 += 50;

}
// back prop for + op
for(int x2209=0; x2209 < 20; x2209++) {
int32_t x2210 = x2207;
int32_t x2211 = x2208;
int32_t x2212 = x2206;
int32_t x2213 = x2212;
int32_t x2214 = x2210;
int32_t x2215 = x2211;
for(int x2216=0; x2216 < 50; x2216++) {
int32_t x2217 = x2214;
float x2218 = x826[x2217];
int32_t x2219 = x2215;
float x2220 = x859[x2219];
float x2221 = x2218 + x2220;
x826[x2217] = x2221;
x2213 += 1;
x2214 += 1;
x2215 += 1;

}
x2206 += 50;
x2207 += 50;
x2208 += 50;

}
for(int x2236=0; x2236 < 20; x2236++) {
int32_t x2237 = x2234;
int32_t x2238 = x2235;
int32_t x2239 = x2233;
int32_t x2240 = x2239;
int32_t x2241 = x2237;
int32_t x2242 = x2238;
for(int x2243=0; x2243 < 50; x2243++) {
int32_t x2244 = x2241;
float x2245 = x829[x2244];
int32_t x2246 = x2242;
float x2247 = x859[x2246];
float x2248 = x2245 + x2247;
x829[x2244] = x2248;
x2240 += 1;
x2241 += 1;
x2242 += 1;

}
x2233 += 50;
x2234 += 50;
x2235 += 50;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x829,50,x29,50,1,x817,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x816,50,x829,50,1,x38,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x826,50,x18,50,1,x823,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x822,26,x826,50,1,x28,50);
} else {
float x2267 = 0.0f;
for(int x2268=0; x2268 < 1; x2268++) {
float x2269 = x2267;
float x2270 = x814[x2268];
float x2271 = x2269 + x2270;
x2267 = x2271;

}
float x2275 = x2267;
float* x2276 = (float*)myMalloc(1 * sizeof(float));;
for(int x2277=0; x2277 < 1; x2277++) {
x2276[x2277] = x2275;

}
float* x2281 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x2283=0; x2283 < 1; x2283++) {
x2281[x2283] = 1.0f;

}
// backend is lantern.TensorDsl$BackendCPU@4d7884c9
for(int x2288=0; x2288 < 1; x2288++) {
float x2289 = x2276[x2288];
x158[x2288] = x2289;

}
// 'sum' gradient.
if (x786) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
int32_t x2298 = 0;
int32_t x2299 = 0;
int32_t x2300 = 0;
for(int x2301=0; x2301 < 1; x2301++) {
int32_t x2302 = x2299;
float x2303 = x815[x2302];
int32_t x2304 = x2300;
float x2305 = x2281[x2304];
float x2306 = x2303 + x2305;
x815[x2302] = x2306;
x2298 += 1;

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
float** x3198 = (float**)myMalloc(6 * sizeof(float*));;
x3198[0] = x177;
x3198[1] = x178;
x3198[2] = x179;
x3198[3] = x180;
x3198[4] = x181;
x3198[5] = x182;
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
int32_t x205 = 0;
int32_t x206 = 0;
int32_t x207 = 0;
for(int x208=0; x208 < 20; x208++) {
int32_t x209 = x206;
int32_t x210 = x207;
int32_t x211 = x205;
int32_t x212 = x211;
int32_t x213 = x209;
int32_t x214 = x210;
for(int x216=0; x216 < 50; x216++) {
int32_t x217 = x212;
int32_t x218 = x213;
float x219 = x198[x218];
int32_t x220 = x214;
float x221 = x201[x220];
float x222 = x219 + x221;
x204[x217] = x222;
x212 += 1;
x213 += 1;
x214 += 1;

}
x205 += 50;
x206 += 50;
x207 += 50;

}
float* x234 = (float*)myMalloc(1000 * sizeof(float));;
float* x235 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x236 = 0;
int32_t x237 = 0;
int32_t x238 = 0;
for(int x239=0; x239 < 20; x239++) {
int32_t x240 = x237;
int32_t x241 = x238;
int32_t x242 = x236;
int32_t x243 = x242;
int32_t x244 = x240;
int32_t x245 = x241;
for(int x246=0; x246 < 50; x246++) {
int32_t x247 = x243;
int32_t x248 = x244;
float x249 = x204[x248];
int32_t x250 = x245;
float x251 = x39[x250];
float x252 = x249 + x251;
x235[x247] = x252;
x243 += 1;
x244 += 1;
x245 += 1;

}
x236 += 50;
x237 += 50;

}
float* x263 = (float*)myMalloc(1000 * sizeof(float));;
float* x264 = (float*)myMalloc(1000 * sizeof(float));;
for(int x266=0; x266 < 1000; x266++) {
float x267 = x235[x266];
float x268 = -1.0f * x267;
double x269 = (double)x268;
double x270 = exp(x269);
float x271 = (float)x270;
float x272 = x271 + 1.0f;
float x273 = 1.0f / x272;
x264[x266] = x273;

}
float* x277 = (float*)myMalloc(1000 * sizeof(float));;
float* x278 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x196,26,x41,50,0,x278,50);
float* x280 = (float*)myMalloc(1000 * sizeof(float));;
float* x281 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x190,50,x50,50,0,x281,50);
float* x283 = (float*)myMalloc(1000 * sizeof(float));;
float* x284 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x285 = 0;
int32_t x286 = 0;
int32_t x287 = 0;
for(int x288=0; x288 < 20; x288++) {
int32_t x289 = x286;
int32_t x290 = x287;
int32_t x291 = x285;
int32_t x292 = x291;
int32_t x293 = x289;
int32_t x294 = x290;
for(int x295=0; x295 < 50; x295++) {
int32_t x296 = x292;
int32_t x297 = x293;
float x298 = x278[x297];
int32_t x299 = x294;
float x300 = x281[x299];
float x301 = x298 + x300;
x284[x296] = x301;
x292 += 1;
x293 += 1;
x294 += 1;

}
x285 += 50;
x286 += 50;
x287 += 50;

}
float* x313 = (float*)myMalloc(1000 * sizeof(float));;
float* x314 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x315 = 0;
int32_t x316 = 0;
int32_t x317 = 0;
for(int x318=0; x318 < 20; x318++) {
int32_t x319 = x316;
int32_t x320 = x317;
int32_t x321 = x315;
int32_t x322 = x321;
int32_t x323 = x319;
int32_t x324 = x320;
for(int x325=0; x325 < 50; x325++) {
int32_t x326 = x322;
int32_t x327 = x323;
float x328 = x284[x327];
int32_t x329 = x324;
float x330 = x59[x329];
float x331 = x328 + x330;
x314[x326] = x331;
x322 += 1;
x323 += 1;
x324 += 1;

}
x315 += 50;
x316 += 50;

}
float* x342 = (float*)myMalloc(1000 * sizeof(float));;
float* x343 = (float*)myMalloc(1000 * sizeof(float));;
for(int x344=0; x344 < 1000; x344++) {
float x345 = x314[x344];
float x346 = -1.0f * x345;
double x347 = (double)x346;
double x348 = exp(x347);
float x349 = (float)x348;
float x350 = x349 + 1.0f;
float x351 = 1.0f / x350;
x343[x344] = x351;

}
float* x355 = (float*)myMalloc(1000 * sizeof(float));;
float* x356 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x196,26,x81,50,0,x356,50);
float* x358 = (float*)myMalloc(1000 * sizeof(float));;
float* x359 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x190,50,x90,50,0,x359,50);
float* x361 = (float*)myMalloc(1000 * sizeof(float));;
float* x362 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x363 = 0;
int32_t x364 = 0;
int32_t x365 = 0;
for(int x366=0; x366 < 20; x366++) {
int32_t x367 = x364;
int32_t x368 = x365;
int32_t x369 = x363;
int32_t x370 = x369;
int32_t x371 = x367;
int32_t x372 = x368;
for(int x373=0; x373 < 50; x373++) {
int32_t x374 = x370;
int32_t x375 = x371;
float x376 = x356[x375];
int32_t x377 = x372;
float x378 = x359[x377];
float x379 = x376 + x378;
x362[x374] = x379;
x370 += 1;
x371 += 1;
x372 += 1;

}
x363 += 50;
x364 += 50;
x365 += 50;

}
float* x391 = (float*)myMalloc(1000 * sizeof(float));;
float* x392 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x393 = 0;
int32_t x394 = 0;
int32_t x395 = 0;
for(int x396=0; x396 < 20; x396++) {
int32_t x397 = x394;
int32_t x398 = x395;
int32_t x399 = x393;
int32_t x400 = x399;
int32_t x401 = x397;
int32_t x402 = x398;
for(int x403=0; x403 < 50; x403++) {
int32_t x404 = x400;
int32_t x405 = x401;
float x406 = x362[x405];
int32_t x407 = x402;
float x408 = x99[x407];
float x409 = x406 + x408;
x392[x404] = x409;
x400 += 1;
x401 += 1;
x402 += 1;

}
x393 += 50;
x394 += 50;

}
float* x420 = (float*)myMalloc(1000 * sizeof(float));;
float* x421 = (float*)myMalloc(1000 * sizeof(float));;
for(int x422=0; x422 < 1000; x422++) {
float x423 = x392[x422];
float x424 = -1.0f * x423;
double x425 = (double)x424;
double x426 = exp(x425);
float x427 = (float)x426;
float x428 = x427 + 1.0f;
float x429 = 1.0f / x428;
x421[x422] = x429;

}
float* x433 = (float*)myMalloc(1000 * sizeof(float));;
float* x434 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x196,26,x61,50,0,x434,50);
float* x436 = (float*)myMalloc(1000 * sizeof(float));;
float* x437 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x190,50,x70,50,0,x437,50);
float* x439 = (float*)myMalloc(1000 * sizeof(float));;
float* x440 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x441 = 0;
int32_t x442 = 0;
int32_t x443 = 0;
for(int x444=0; x444 < 20; x444++) {
int32_t x445 = x442;
int32_t x446 = x443;
int32_t x447 = x441;
int32_t x448 = x447;
int32_t x449 = x445;
int32_t x450 = x446;
for(int x451=0; x451 < 50; x451++) {
int32_t x452 = x448;
int32_t x453 = x449;
float x454 = x434[x453];
int32_t x455 = x450;
float x456 = x437[x455];
float x457 = x454 + x456;
x440[x452] = x457;
x448 += 1;
x449 += 1;
x450 += 1;

}
x441 += 50;
x442 += 50;
x443 += 50;

}
float* x469 = (float*)myMalloc(1000 * sizeof(float));;
float* x470 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x471 = 0;
int32_t x472 = 0;
int32_t x473 = 0;
for(int x474=0; x474 < 20; x474++) {
int32_t x475 = x472;
int32_t x476 = x473;
int32_t x477 = x471;
int32_t x478 = x477;
int32_t x479 = x475;
int32_t x480 = x476;
for(int x481=0; x481 < 50; x481++) {
int32_t x482 = x478;
int32_t x483 = x479;
float x484 = x440[x483];
int32_t x485 = x480;
float x486 = x79[x485];
float x487 = x484 + x486;
x470[x482] = x487;
x478 += 1;
x479 += 1;
x480 += 1;

}
x471 += 50;
x472 += 50;

}
float* x498 = (float*)myMalloc(1000 * sizeof(float));;
float* x499 = (float*)myMalloc(1000 * sizeof(float));;
for(int x500=0; x500 < 1000; x500++) {
float x501 = x470[x500];
double x502 = (double)x501;
double x503 = tanh(x502);
float x504 = (float)x503;
x499[x500] = x504;

}
float* x508 = (float*)myMalloc(1000 * sizeof(float));;
float* x509 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x510 = 0;
int32_t x511 = 0;
int32_t x512 = 0;
for(int x513=0; x513 < 20; x513++) {
int32_t x514 = x511;
int32_t x515 = x512;
int32_t x516 = x510;
int32_t x517 = x516;
int32_t x518 = x514;
int32_t x519 = x515;
for(int x520=0; x520 < 50; x520++) {
int32_t x521 = x517;
int32_t x522 = x518;
float x523 = x264[x522];
int32_t x524 = x519;
float x525 = x192[x524];
float x526 = x523 * x525;
x509[x521] = x526;
x517 += 1;
x518 += 1;
x519 += 1;

}
x510 += 50;
x511 += 50;
x512 += 50;

}
float* x538 = (float*)myMalloc(1000 * sizeof(float));;
float* x539 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x540 = 0;
int32_t x541 = 0;
int32_t x542 = 0;
for(int x543=0; x543 < 20; x543++) {
int32_t x544 = x541;
int32_t x545 = x542;
int32_t x546 = x540;
int32_t x547 = x546;
int32_t x548 = x544;
int32_t x549 = x545;
for(int x550=0; x550 < 50; x550++) {
int32_t x551 = x547;
int32_t x552 = x548;
float x553 = x343[x552];
int32_t x554 = x549;
float x555 = x499[x554];
float x556 = x553 * x555;
x539[x551] = x556;
x547 += 1;
x548 += 1;
x549 += 1;

}
x540 += 50;
x541 += 50;
x542 += 50;

}
float* x568 = (float*)myMalloc(1000 * sizeof(float));;
float* x569 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x570 = 0;
int32_t x571 = 0;
int32_t x572 = 0;
for(int x573=0; x573 < 20; x573++) {
int32_t x574 = x571;
int32_t x575 = x572;
int32_t x576 = x570;
int32_t x577 = x576;
int32_t x578 = x574;
int32_t x579 = x575;
for(int x580=0; x580 < 50; x580++) {
int32_t x581 = x577;
int32_t x582 = x578;
float x583 = x509[x582];
int32_t x584 = x579;
float x585 = x539[x584];
float x586 = x583 + x585;
x569[x581] = x586;
x577 += 1;
x578 += 1;
x579 += 1;

}
x570 += 50;
x571 += 50;
x572 += 50;

}
float* x598 = (float*)myMalloc(1000 * sizeof(float));;
float* x599 = (float*)myMalloc(1000 * sizeof(float));;
for(int x600=0; x600 < 1000; x600++) {
float x601 = x569[x600];
double x602 = (double)x601;
double x603 = tanh(x602);
float x604 = (float)x603;
x599[x600] = x604;

}
float* x608 = (float*)myMalloc(1000 * sizeof(float));;
float* x609 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x610 = 0;
int32_t x611 = 0;
int32_t x612 = 0;
for(int x613=0; x613 < 20; x613++) {
int32_t x614 = x611;
int32_t x615 = x612;
int32_t x616 = x610;
int32_t x617 = x616;
int32_t x618 = x614;
int32_t x619 = x615;
for(int x620=0; x620 < 50; x620++) {
int32_t x621 = x617;
int32_t x622 = x618;
float x623 = x421[x622];
int32_t x624 = x619;
float x625 = x599[x624];
float x626 = x623 * x625;
x609[x621] = x626;
x617 += 1;
x618 += 1;
x619 += 1;

}
x610 += 50;
x611 += 50;
x612 += 50;

}
float* x638 = (float*)myMalloc(1000 * sizeof(float));;
float* x639 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x609,50,x101,26,0,x639,26);
float* x641 = (float*)myMalloc(520 * sizeof(float));;
int32_t x642 = 0;
int32_t x643 = 0;
int32_t x644 = 0;
for(int x645=0; x645 < 20; x645++) {
int32_t x646 = x643;
int32_t x647 = x644;
int32_t x648 = x642;
int32_t x649 = x648;
int32_t x650 = x646;
int32_t x651 = x647;
for(int x653=0; x653 < 26; x653++) {
int32_t x654 = x650;
float x655 = x639[x654];
int32_t x656 = x651;
float x657 = x110[x656];
float x658 = x655 + x657;
x639[x654] = x658;
x649 += 1;
x650 += 1;
x651 += 1;

}
x642 += 26;
x643 += 26;

}
int* x669 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x670=0; x670 < 20; x670++) {
int32_t x671 = x670 * 20;
int32_t x672 = x186 + x671;
int32_t x673 = x143[x672];
x669[x670] = x673;

}
float* x677 = (float*)myMalloc(20 * sizeof(float));;
int32_t x678 = 0;
for(int x679=0; x679 < 20; x679++) {
float x680 = -3.4028235E38f;
for(int x681=0; x681 < 26; x681++) {
int32_t x682 = x678;
float x683 = x639[x682];
float x684 = x680;
bool x685 = x683 > x684;
if (x685) {
float x686 = x639[x682];
x680 = x686;
} else {
}
x678 += 1;

}
float x693 = x680;
x677[x679] = x693;

}
float* x697 = (float*)myMalloc(520 * sizeof(float));;
int32_t x698 = 0;
for(int x699=0; x699 < 20; x699++) {
for(int x700=0; x700 < 26; x700++) {
int32_t x701 = x698;
float x702 = x639[x701];
float x703 = x677[x699];
float x704 = x702 - x703;
double x705 = (double)x704;
double x706 = exp(x705);
float x707 = (float)x706;
x697[x701] = x707;
x698 += 1;

}

}
float* x714 = (float*)myMalloc(20 * sizeof(float));;
for(int x715=0; x715 < 20; x715++) {
int32_t x716 = x715;
int32_t x717 = x715 * 26;
int32_t x718 = x717;
for(int x719=0; x719 < 26; x719++) {
for(int x721=0; x721 < 1; x721++) {
int32_t x722 = x716;
int32_t x723 = x722 + x721;
float x724 = x714[x723];
int32_t x725 = x718;
int32_t x726 = x725 + x721;
float x727 = x697[x726];
float x728 = x724 + x727;
x714[x723] = x728;

}
x718 += 1;

}

}
x698 = 0;
for(int x738=0; x738 < 20; x738++) {
float x739 = x677[x738];
float x740 = x714[x738];
double x741 = (double)x740;
double x742 = log(x741);
float x743 = (float)x742;
float x744 = x739 + x743;
for(int x745=0; x745 < 26; x745++) {
int32_t x746 = x698;
float x747 = x639[x746];
float x748 = x747 - x744;
x697[x746] = x748;
x698 += 1;

}

}
float* x755 = (float*)myMalloc(520 * sizeof(float));;
// nllLoss forward in CPU
float* x757 = (float*)myMalloc(20 * sizeof(float));;
int32_t x758 = 0;
for(int x759=0; x759 < 20; x759++) {
int32_t x760 = x758;
int32_t x761 = x669[x759];
int32_t x762 = x760 + x761;
float x763 = x697[x762];
float x764 = -1.0f * x763;
x757[x759] = x764;
x758 += 26;

}
float* x769 = (float*)myMalloc(20 * sizeof(float));;
float x770 = 0.0f;
for(int x771=0; x771 < 20; x771++) {
float x772 = x770;
float x773 = x757[x771];
float x774 = x772 + x773;
x770 = x774;

}
float x778 = x770;
float* x779 = (float*)myMalloc(1 * sizeof(float));;
for(int x780=0; x780 < 1; x780++) {
x779[x780] = x778;

}
float* x784 = (float*)myMalloc(1 * sizeof(float));;
if (x786) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
float* x792 = (float*)myMalloc(1 * sizeof(float));;
int32_t x793 = 0;
int32_t x794 = 0;
int32_t x795 = 0;
for(int x796=0; x796 < 1; x796++) {
int32_t x797 = x793;
int32_t x798 = x794;
float x799 = x188[x798];
int32_t x800 = x795;
float x801 = x779[x800];
float x802 = x799 + x801;
x792[x797] = x802;
x793 += 1;

}
float* x807 = (float*)myMalloc(1 * sizeof(float));;
float** x2314 = (float**)myMalloc(6 * sizeof(float*));;
x2314[0] = x792;
x2314[1] = x807;
x2314[2] = x609;
x2314[3] = x638;
x2314[4] = x569;
x2314[5] = x598;
int32_t x808 = x186 + 1;
x809(x808,x2314);
// back prop for + op
if (x786) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
int32_t x2328 = 0;
int32_t x2329 = 0;
int32_t x2330 = 0;
for(int x2331=0; x2331 < 1; x2331++) {
int32_t x2332 = x2329;
float x2333 = x189[x2332];
int32_t x2334 = x2330;
float x2335 = x807[x2334];
float x2336 = x2333 + x2335;
x189[x2332] = x2336;
x2328 += 1;

}
if (x786) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
int32_t x2345 = 0;
int32_t x2346 = 0;
int32_t x2347 = 0;
for(int x2348=0; x2348 < 1; x2348++) {
int32_t x2349 = x2346;
float x2350 = x784[x2349];
int32_t x2351 = x2347;
float x2352 = x807[x2351];
float x2353 = x2350 + x2352;
x784[x2349] = x2353;
x2345 += 1;

}
// 'sum' gradient.
if (x1473) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",20,1);
assert(false && "");
}
int32_t x2363 = 0;
int32_t x2364 = 0;
int32_t x2365 = 0;
for(int x2366=0; x2366 < 20; x2366++) {
int32_t x2367 = x2364;
float x2368 = x769[x2367];
int32_t x2369 = x2365;
float x2370 = x784[x2369];
float x2371 = x2368 + x2370;
x769[x2367] = x2371;
x2363 += 1;
x2364 += 1;

}
// 'nllLossB' gradient.
// nllLoss_grad implementation in CPU
int32_t x2379 = 0;
for(int x2380=0; x2380 < 20; x2380++) {
int32_t x2381 = x2379;
int32_t x2382 = x669[x2380];
int32_t x2383 = x2381 + x2382;
float x2384 = x755[x2383];
float x2385 = x769[x2380];
float x2386 = -1.0f * x2385;
float x2387 = x2384 + x2386;
x755[x2383] = x2387;
x2379 += 26;

}
float* x2392 = (float*)myMalloc(20 * sizeof(float));;
for(int x2393=0; x2393 < 20; x2393++) {
int32_t x2394 = x2393;
int32_t x2395 = x2393 * 26;
int32_t x2396 = x2395;
for(int x2397=0; x2397 < 26; x2397++) {
for(int x2398=0; x2398 < 1; x2398++) {
int32_t x2399 = x2394;
int32_t x2400 = x2399 + x2398;
float x2401 = x2392[x2400];
int32_t x2402 = x2396;
int32_t x2403 = x2402 + x2398;
float x2404 = x755[x2403];
float x2405 = x2401 + x2404;
x2392[x2400] = x2405;

}
x2396 += 1;

}

}
int32_t x2414 = 0;
for(int x2415=0; x2415 < 20; x2415++) {
for(int x2416=0; x2416 < 26; x2416++) {
int32_t x2417 = x2414;
float x2418 = x641[x2417];
float x2419 = x755[x2417];
float x2420 = x697[x2417];
float x2424 = x2392[x2415];
double x2421 = (double)x2420;
double x2422 = exp(x2421);
float x2423 = (float)x2422;
float x2425 = x2423 * x2424;
float x2426 = x2419 - x2425;
float x2427 = x2418 + x2426;
x641[x2417] = x2427;
x2414 += 1;

}

}
int32_t x2434 = 0;
int32_t x2435 = 0;
int32_t x2436 = 0;
for(int x2437=0; x2437 < 20; x2437++) {
int32_t x2438 = x2435;
int32_t x2439 = x2436;
int32_t x2440 = x2434;
int32_t x2441 = x2440;
int32_t x2442 = x2438;
int32_t x2443 = x2439;
for(int x2444=0; x2444 < 26; x2444++) {
int32_t x2445 = x2442;
float x2446 = x111[x2445];
int32_t x2447 = x2443;
float x2448 = x641[x2447];
float x2449 = x2446 + x2448;
x111[x2445] = x2449;
x2441 += 1;
x2442 += 1;
x2443 += 1;

}
x2434 += 26;
x2436 += 26;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x641,26,x101,26,1,x638,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x609,50,x641,26,1,x109,26);
// backprop for * op
int32_t x2464 = 0;
int32_t x2465 = 0;
int32_t x2466 = 0;
for(int x2467=0; x2467 < 20; x2467++) {
int32_t x2468 = x2464;
int32_t x2469 = x2465;
int32_t x2470 = x2466;
int32_t x2471 = x2468;
int32_t x2472 = x2469;
int32_t x2473 = x2470;
for(int x2474=0; x2474 < 50; x2474++) {
int32_t x2475 = x2471;
float x2476 = x433[x2475];
float x2477 = x421[x2475];
int32_t x2478 = x2472;
float x2479 = x599[x2478];
int32_t x2480 = x2473;
float x2481 = x638[x2480];
float x2482 = x2481 * x2479;
float x2483 = x2476 + x2482;
x433[x2475] = x2483;
float x2485 = x608[x2478];
float x2486 = x421[x2475];
float x2487 = x599[x2478];
float x2488 = x638[x2480];
float x2489 = x2488 * x2486;
float x2490 = x2485 + x2489;
x608[x2478] = x2490;
x2473 += 1;
x2471 += 1;
x2472 += 1;

}
x2466 += 50;
x2464 += 50;
x2465 += 50;

}
for(int x2502=0; x2502 < 1000; x2502++) {
float x2503 = x598[x2502];
float x2504 = x599[x2502];
float x2507 = x608[x2502];
float x2505 = x2504 * x2504;
float x2506 = 1.0f - x2505;
float x2508 = x2506 * x2507;
float x2509 = x2503 + x2508;
x598[x2502] = x2509;

}
// back prop for + op
int32_t x2514 = 0;
int32_t x2515 = 0;
int32_t x2516 = 0;
for(int x2517=0; x2517 < 20; x2517++) {
int32_t x2518 = x2515;
int32_t x2519 = x2516;
int32_t x2520 = x2514;
int32_t x2521 = x2520;
int32_t x2522 = x2518;
int32_t x2523 = x2519;
for(int x2524=0; x2524 < 50; x2524++) {
int32_t x2525 = x2522;
float x2526 = x538[x2525];
int32_t x2527 = x2523;
float x2528 = x598[x2527];
float x2529 = x2526 + x2528;
x538[x2525] = x2529;
x2521 += 1;
x2522 += 1;
x2523 += 1;

}
x2514 += 50;
x2515 += 50;
x2516 += 50;

}
int32_t x2541 = 0;
int32_t x2542 = 0;
int32_t x2543 = 0;
for(int x2544=0; x2544 < 20; x2544++) {
int32_t x2545 = x2542;
int32_t x2546 = x2543;
int32_t x2547 = x2541;
int32_t x2548 = x2547;
int32_t x2549 = x2545;
int32_t x2550 = x2546;
for(int x2551=0; x2551 < 50; x2551++) {
int32_t x2552 = x2549;
float x2553 = x568[x2552];
int32_t x2554 = x2550;
float x2555 = x598[x2554];
float x2556 = x2553 + x2555;
x568[x2552] = x2556;
x2548 += 1;
x2549 += 1;
x2550 += 1;

}
x2541 += 50;
x2542 += 50;
x2543 += 50;

}
// backprop for * op
int32_t x2569 = 0;
int32_t x2570 = 0;
int32_t x2571 = 0;
for(int x2572=0; x2572 < 20; x2572++) {
int32_t x2573 = x2569;
int32_t x2574 = x2570;
int32_t x2575 = x2571;
int32_t x2576 = x2573;
int32_t x2577 = x2574;
int32_t x2578 = x2575;
for(int x2579=0; x2579 < 50; x2579++) {
int32_t x2580 = x2576;
float x2581 = x355[x2580];
float x2582 = x343[x2580];
int32_t x2583 = x2577;
float x2584 = x499[x2583];
int32_t x2585 = x2578;
float x2586 = x568[x2585];
float x2587 = x2586 * x2584;
float x2588 = x2581 + x2587;
x355[x2580] = x2588;
float x2590 = x508[x2583];
float x2591 = x343[x2580];
float x2592 = x499[x2583];
float x2593 = x568[x2585];
float x2594 = x2593 * x2591;
float x2595 = x2590 + x2594;
x508[x2583] = x2595;
x2578 += 1;
x2576 += 1;
x2577 += 1;

}
x2571 += 50;
x2569 += 50;
x2570 += 50;

}
// backprop for * op
int32_t x2608 = 0;
int32_t x2609 = 0;
int32_t x2610 = 0;
for(int x2611=0; x2611 < 20; x2611++) {
int32_t x2612 = x2608;
int32_t x2613 = x2609;
int32_t x2614 = x2610;
int32_t x2615 = x2612;
int32_t x2616 = x2613;
int32_t x2617 = x2614;
for(int x2618=0; x2618 < 50; x2618++) {
int32_t x2619 = x2615;
float x2620 = x277[x2619];
float x2621 = x264[x2619];
int32_t x2622 = x2616;
float x2623 = x192[x2622];
int32_t x2624 = x2617;
float x2625 = x538[x2624];
float x2626 = x2625 * x2623;
float x2627 = x2620 + x2626;
x277[x2619] = x2627;
float x2629 = x193[x2622];
float x2630 = x264[x2619];
float x2631 = x192[x2622];
float x2632 = x538[x2624];
float x2633 = x2632 * x2630;
float x2634 = x2629 + x2633;
x193[x2622] = x2634;
x2617 += 1;
x2615 += 1;
x2616 += 1;

}
x2610 += 50;
x2608 += 50;
x2609 += 50;

}
for(int x2646=0; x2646 < 1000; x2646++) {
float x2647 = x498[x2646];
float x2648 = x499[x2646];
float x2651 = x508[x2646];
float x2649 = x2648 * x2648;
float x2650 = 1.0f - x2649;
float x2652 = x2650 * x2651;
float x2653 = x2647 + x2652;
x498[x2646] = x2653;

}
// back prop for + op
int32_t x2658 = 0;
int32_t x2659 = 0;
int32_t x2660 = 0;
for(int x2661=0; x2661 < 20; x2661++) {
int32_t x2662 = x2659;
int32_t x2663 = x2660;
int32_t x2664 = x2658;
int32_t x2665 = x2664;
int32_t x2666 = x2662;
int32_t x2667 = x2663;
for(int x2668=0; x2668 < 50; x2668++) {
int32_t x2669 = x2666;
float x2670 = x469[x2669];
int32_t x2671 = x2667;
float x2672 = x498[x2671];
float x2673 = x2670 + x2672;
x469[x2669] = x2673;
x2665 += 1;
x2666 += 1;
x2667 += 1;

}
x2658 += 50;
x2659 += 50;
x2660 += 50;

}
int32_t x2685 = 0;
int32_t x2686 = 0;
int32_t x2687 = 0;
for(int x2688=0; x2688 < 20; x2688++) {
int32_t x2689 = x2686;
int32_t x2690 = x2687;
int32_t x2691 = x2685;
int32_t x2692 = x2691;
int32_t x2693 = x2689;
int32_t x2694 = x2690;
for(int x2695=0; x2695 < 50; x2695++) {
int32_t x2696 = x2693;
float x2697 = x80[x2696];
int32_t x2698 = x2694;
float x2699 = x498[x2698];
float x2700 = x2697 + x2699;
x80[x2696] = x2700;
x2692 += 1;
x2693 += 1;
x2694 += 1;

}
x2685 += 50;
x2687 += 50;

}
// back prop for + op
int32_t x2712 = 0;
int32_t x2713 = 0;
int32_t x2714 = 0;
for(int x2715=0; x2715 < 20; x2715++) {
int32_t x2716 = x2713;
int32_t x2717 = x2714;
int32_t x2718 = x2712;
int32_t x2719 = x2718;
int32_t x2720 = x2716;
int32_t x2721 = x2717;
for(int x2722=0; x2722 < 50; x2722++) {
int32_t x2723 = x2720;
float x2724 = x436[x2723];
int32_t x2725 = x2721;
float x2726 = x469[x2725];
float x2727 = x2724 + x2726;
x436[x2723] = x2727;
x2719 += 1;
x2720 += 1;
x2721 += 1;

}
x2712 += 50;
x2713 += 50;
x2714 += 50;

}
int32_t x2739 = 0;
int32_t x2740 = 0;
int32_t x2741 = 0;
for(int x2742=0; x2742 < 20; x2742++) {
int32_t x2743 = x2740;
int32_t x2744 = x2741;
int32_t x2745 = x2739;
int32_t x2746 = x2745;
int32_t x2747 = x2743;
int32_t x2748 = x2744;
for(int x2749=0; x2749 < 50; x2749++) {
int32_t x2750 = x2747;
float x2751 = x439[x2750];
int32_t x2752 = x2748;
float x2753 = x469[x2752];
float x2754 = x2751 + x2753;
x439[x2750] = x2754;
x2746 += 1;
x2747 += 1;
x2748 += 1;

}
x2739 += 50;
x2740 += 50;
x2741 += 50;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x439,50,x70,50,1,x191,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x190,50,x439,50,1,x78,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x436,50,x61,50,1,x197,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x196,26,x436,50,1,x69,50);
for(int x2772=0; x2772 < 1000; x2772++) {
float x2773 = x420[x2772];
float x2774 = x421[x2772];
float x2777 = x433[x2772];
float x2775 = 1.0f - x2774;
float x2776 = x2775 * x2774;
float x2778 = x2776 * x2777;
float x2779 = x2773 + x2778;
x420[x2772] = x2779;

}
// back prop for + op
int32_t x2784 = 0;
int32_t x2785 = 0;
int32_t x2786 = 0;
for(int x2787=0; x2787 < 20; x2787++) {
int32_t x2788 = x2785;
int32_t x2789 = x2786;
int32_t x2790 = x2784;
int32_t x2791 = x2790;
int32_t x2792 = x2788;
int32_t x2793 = x2789;
for(int x2794=0; x2794 < 50; x2794++) {
int32_t x2795 = x2792;
float x2796 = x391[x2795];
int32_t x2797 = x2793;
float x2798 = x420[x2797];
float x2799 = x2796 + x2798;
x391[x2795] = x2799;
x2791 += 1;
x2792 += 1;
x2793 += 1;

}
x2784 += 50;
x2785 += 50;
x2786 += 50;

}
int32_t x2811 = 0;
int32_t x2812 = 0;
int32_t x2813 = 0;
for(int x2814=0; x2814 < 20; x2814++) {
int32_t x2815 = x2812;
int32_t x2816 = x2813;
int32_t x2817 = x2811;
int32_t x2818 = x2817;
int32_t x2819 = x2815;
int32_t x2820 = x2816;
for(int x2821=0; x2821 < 50; x2821++) {
int32_t x2822 = x2819;
float x2823 = x100[x2822];
int32_t x2824 = x2820;
float x2825 = x420[x2824];
float x2826 = x2823 + x2825;
x100[x2822] = x2826;
x2818 += 1;
x2819 += 1;
x2820 += 1;

}
x2811 += 50;
x2813 += 50;

}
// back prop for + op
int32_t x2838 = 0;
int32_t x2839 = 0;
int32_t x2840 = 0;
for(int x2841=0; x2841 < 20; x2841++) {
int32_t x2842 = x2839;
int32_t x2843 = x2840;
int32_t x2844 = x2838;
int32_t x2845 = x2844;
int32_t x2846 = x2842;
int32_t x2847 = x2843;
for(int x2848=0; x2848 < 50; x2848++) {
int32_t x2849 = x2846;
float x2850 = x358[x2849];
int32_t x2851 = x2847;
float x2852 = x391[x2851];
float x2853 = x2850 + x2852;
x358[x2849] = x2853;
x2845 += 1;
x2846 += 1;
x2847 += 1;

}
x2838 += 50;
x2839 += 50;
x2840 += 50;

}
int32_t x2865 = 0;
int32_t x2866 = 0;
int32_t x2867 = 0;
for(int x2868=0; x2868 < 20; x2868++) {
int32_t x2869 = x2866;
int32_t x2870 = x2867;
int32_t x2871 = x2865;
int32_t x2872 = x2871;
int32_t x2873 = x2869;
int32_t x2874 = x2870;
for(int x2875=0; x2875 < 50; x2875++) {
int32_t x2876 = x2873;
float x2877 = x361[x2876];
int32_t x2878 = x2874;
float x2879 = x391[x2878];
float x2880 = x2877 + x2879;
x361[x2876] = x2880;
x2872 += 1;
x2873 += 1;
x2874 += 1;

}
x2865 += 50;
x2866 += 50;
x2867 += 50;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x361,50,x90,50,1,x191,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x190,50,x361,50,1,x98,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x358,50,x81,50,1,x197,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x196,26,x358,50,1,x89,50);
for(int x2898=0; x2898 < 1000; x2898++) {
float x2899 = x342[x2898];
float x2900 = x343[x2898];
float x2903 = x355[x2898];
float x2901 = 1.0f - x2900;
float x2902 = x2901 * x2900;
float x2904 = x2902 * x2903;
float x2905 = x2899 + x2904;
x342[x2898] = x2905;

}
// back prop for + op
int32_t x2910 = 0;
int32_t x2911 = 0;
int32_t x2912 = 0;
for(int x2913=0; x2913 < 20; x2913++) {
int32_t x2914 = x2911;
int32_t x2915 = x2912;
int32_t x2916 = x2910;
int32_t x2917 = x2916;
int32_t x2918 = x2914;
int32_t x2919 = x2915;
for(int x2920=0; x2920 < 50; x2920++) {
int32_t x2921 = x2918;
float x2922 = x313[x2921];
int32_t x2923 = x2919;
float x2924 = x342[x2923];
float x2925 = x2922 + x2924;
x313[x2921] = x2925;
x2917 += 1;
x2918 += 1;
x2919 += 1;

}
x2910 += 50;
x2911 += 50;
x2912 += 50;

}
int32_t x2937 = 0;
int32_t x2938 = 0;
int32_t x2939 = 0;
for(int x2940=0; x2940 < 20; x2940++) {
int32_t x2941 = x2938;
int32_t x2942 = x2939;
int32_t x2943 = x2937;
int32_t x2944 = x2943;
int32_t x2945 = x2941;
int32_t x2946 = x2942;
for(int x2947=0; x2947 < 50; x2947++) {
int32_t x2948 = x2945;
float x2949 = x60[x2948];
int32_t x2950 = x2946;
float x2951 = x342[x2950];
float x2952 = x2949 + x2951;
x60[x2948] = x2952;
x2944 += 1;
x2945 += 1;
x2946 += 1;

}
x2937 += 50;
x2939 += 50;

}
// back prop for + op
int32_t x2964 = 0;
int32_t x2965 = 0;
int32_t x2966 = 0;
for(int x2967=0; x2967 < 20; x2967++) {
int32_t x2968 = x2965;
int32_t x2969 = x2966;
int32_t x2970 = x2964;
int32_t x2971 = x2970;
int32_t x2972 = x2968;
int32_t x2973 = x2969;
for(int x2974=0; x2974 < 50; x2974++) {
int32_t x2975 = x2972;
float x2976 = x280[x2975];
int32_t x2977 = x2973;
float x2978 = x313[x2977];
float x2979 = x2976 + x2978;
x280[x2975] = x2979;
x2971 += 1;
x2972 += 1;
x2973 += 1;

}
x2964 += 50;
x2965 += 50;
x2966 += 50;

}
int32_t x2991 = 0;
int32_t x2992 = 0;
int32_t x2993 = 0;
for(int x2994=0; x2994 < 20; x2994++) {
int32_t x2995 = x2992;
int32_t x2996 = x2993;
int32_t x2997 = x2991;
int32_t x2998 = x2997;
int32_t x2999 = x2995;
int32_t x3000 = x2996;
for(int x3001=0; x3001 < 50; x3001++) {
int32_t x3002 = x2999;
float x3003 = x283[x3002];
int32_t x3004 = x3000;
float x3005 = x313[x3004];
float x3006 = x3003 + x3005;
x283[x3002] = x3006;
x2998 += 1;
x2999 += 1;
x3000 += 1;

}
x2991 += 50;
x2992 += 50;
x2993 += 50;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x283,50,x50,50,1,x191,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x190,50,x283,50,1,x58,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x280,50,x41,50,1,x197,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x196,26,x280,50,1,x49,50);
for(int x3024=0; x3024 < 1000; x3024++) {
float x3025 = x263[x3024];
float x3026 = x264[x3024];
float x3029 = x277[x3024];
float x3027 = 1.0f - x3026;
float x3028 = x3027 * x3026;
float x3030 = x3028 * x3029;
float x3031 = x3025 + x3030;
x263[x3024] = x3031;

}
// back prop for + op
int32_t x3036 = 0;
int32_t x3037 = 0;
int32_t x3038 = 0;
for(int x3039=0; x3039 < 20; x3039++) {
int32_t x3040 = x3037;
int32_t x3041 = x3038;
int32_t x3042 = x3036;
int32_t x3043 = x3042;
int32_t x3044 = x3040;
int32_t x3045 = x3041;
for(int x3046=0; x3046 < 50; x3046++) {
int32_t x3047 = x3044;
float x3048 = x234[x3047];
int32_t x3049 = x3045;
float x3050 = x263[x3049];
float x3051 = x3048 + x3050;
x234[x3047] = x3051;
x3043 += 1;
x3044 += 1;
x3045 += 1;

}
x3036 += 50;
x3037 += 50;
x3038 += 50;

}
int32_t x3063 = 0;
int32_t x3064 = 0;
int32_t x3065 = 0;
for(int x3066=0; x3066 < 20; x3066++) {
int32_t x3067 = x3064;
int32_t x3068 = x3065;
int32_t x3069 = x3063;
int32_t x3070 = x3069;
int32_t x3071 = x3067;
int32_t x3072 = x3068;
for(int x3073=0; x3073 < 50; x3073++) {
int32_t x3074 = x3071;
float x3075 = x40[x3074];
int32_t x3076 = x3072;
float x3077 = x263[x3076];
float x3078 = x3075 + x3077;
x40[x3074] = x3078;
x3070 += 1;
x3071 += 1;
x3072 += 1;

}
x3063 += 50;
x3065 += 50;

}
// back prop for + op
int32_t x3090 = 0;
int32_t x3091 = 0;
int32_t x3092 = 0;
for(int x3093=0; x3093 < 20; x3093++) {
int32_t x3094 = x3091;
int32_t x3095 = x3092;
int32_t x3096 = x3090;
int32_t x3097 = x3096;
int32_t x3098 = x3094;
int32_t x3099 = x3095;
for(int x3100=0; x3100 < 50; x3100++) {
int32_t x3101 = x3098;
float x3102 = x200[x3101];
int32_t x3103 = x3099;
float x3104 = x234[x3103];
float x3105 = x3102 + x3104;
x200[x3101] = x3105;
x3097 += 1;
x3098 += 1;
x3099 += 1;

}
x3090 += 50;
x3091 += 50;
x3092 += 50;

}
int32_t x3117 = 0;
int32_t x3118 = 0;
int32_t x3119 = 0;
for(int x3120=0; x3120 < 20; x3120++) {
int32_t x3121 = x3118;
int32_t x3122 = x3119;
int32_t x3123 = x3117;
int32_t x3124 = x3123;
int32_t x3125 = x3121;
int32_t x3126 = x3122;
for(int x3127=0; x3127 < 50; x3127++) {
int32_t x3128 = x3125;
float x3129 = x203[x3128];
int32_t x3130 = x3126;
float x3131 = x234[x3130];
float x3132 = x3129 + x3131;
x203[x3128] = x3132;
x3124 += 1;
x3125 += 1;
x3126 += 1;

}
x3117 += 50;
x3118 += 50;
x3119 += 50;

}
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x203,50,x29,50,1,x191,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x190,50,x203,50,1,x38,50);
// backprop of matrix-matrix-dot
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x200,50,x18,50,1,x197,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x196,26,x200,50,1,x28,50);
} else {
float x3151 = 0.0f;
for(int x3152=0; x3152 < 1; x3152++) {
float x3153 = x3151;
float x3154 = x188[x3152];
float x3155 = x3153 + x3154;
x3151 = x3155;

}
float x3159 = x3151;
float* x3160 = (float*)myMalloc(1 * sizeof(float));;
for(int x3161=0; x3161 < 1; x3161++) {
x3160[x3161] = x3159;

}
float* x3165 = (float*)myMalloc(1 * sizeof(float));;
// make sure the size of loss is 1
for(int x3167=0; x3167 < 1; x3167++) {
x3165[x3167] = 1.0f;

}
// backend is lantern.TensorDsl$BackendCPU@4d7884c9
for(int x3172=0; x3172 < 1; x3172++) {
float x3173 = x3160[x3172];
x158[x3172] = x3173;

}
// 'sum' gradient.
if (x786) {
} else {
printf("dimensions not compatible for broadcasting %d, with %d,\n",1,1);
assert(false && "");
}
int32_t x3182 = 0;
int32_t x3183 = 0;
int32_t x3184 = 0;
for(int x3185=0; x3185 < 1; x3185++) {
int32_t x3186 = x3183;
float x3187 = x189[x3186];
int32_t x3188 = x3184;
float x3189 = x3165[x3188];
float x3190 = x3187 + x3189;
x189[x3186] = x3190;
x3182 += 1;

}
}
};
x183(0,x3198);
float x3207 = x158[0];
int32_t x3208 = x133 % 100;
bool x3209 = x3208 == 0;
if (x3209) {
printf("iter %d, loss %f\n",x133,x3207);
int32_t x3211 = x133 / 100;
double x3212 = (double)x3207;
x127[x3211] = x3212;
} else {
}
for(int x3216=0; x3216 < 1300; x3216++) {
float x3217 = x49[x3216];
float x3218 = x3217;
float x3219 = x3218;
bool x3220 = x3219 > 5.0f;
if (x3220) {
x3218 = 5.0f;
} else {
}
float x3224 = x3218;
bool x3225 = x3224 < -5.0f;
if (x3225) {
x3218 = -5.0f;
} else {
}
float x3229 = x112[x3216];
float x3230 = x3218;
float x3231 = x3230 * x3230;
float x3232 = x3229 + x3231;
x112[x3216] = x3232;
float x3234 = x41[x3216];
float x3236 = x112[x3216];
float x3235 = 0.1f * x3230;
double x3237 = (double)x3236;
double x3238 = x3237 + 9.99999993922529E-9;
double x3239 = sqrt(x3238);
float x3240 = (float)x3239;
float x3241 = x3235 / x3240;
float x3242 = x3234 - x3241;
x41[x3216] = x3242;
x49[x3216] = 0.0f;

}
for(int x3247=0; x3247 < 50; x3247++) {
float x3248 = x60[x3247];
float x3249 = x3248;
float x3250 = x3249;
bool x3251 = x3250 > 5.0f;
if (x3251) {
x3249 = 5.0f;
} else {
}
float x3255 = x3249;
bool x3256 = x3255 < -5.0f;
if (x3256) {
x3249 = -5.0f;
} else {
}
float x3260 = x113[x3247];
float x3261 = x3249;
float x3262 = x3261 * x3261;
float x3263 = x3260 + x3262;
x113[x3247] = x3263;
float x3265 = x59[x3247];
float x3267 = x113[x3247];
float x3266 = 0.1f * x3261;
double x3268 = (double)x3267;
double x3269 = x3268 + 9.99999993922529E-9;
double x3270 = sqrt(x3269);
float x3271 = (float)x3270;
float x3272 = x3266 / x3271;
float x3273 = x3265 - x3272;
x59[x3247] = x3273;
x60[x3247] = 0.0f;

}
for(int x3278=0; x3278 < 2500; x3278++) {
float x3279 = x58[x3278];
float x3280 = x3279;
float x3281 = x3280;
bool x3282 = x3281 > 5.0f;
if (x3282) {
x3280 = 5.0f;
} else {
}
float x3286 = x3280;
bool x3287 = x3286 < -5.0f;
if (x3287) {
x3280 = -5.0f;
} else {
}
float x3291 = x114[x3278];
float x3292 = x3280;
float x3293 = x3292 * x3292;
float x3294 = x3291 + x3293;
x114[x3278] = x3294;
float x3296 = x50[x3278];
float x3298 = x114[x3278];
float x3297 = 0.1f * x3292;
double x3299 = (double)x3298;
double x3300 = x3299 + 9.99999993922529E-9;
double x3301 = sqrt(x3300);
float x3302 = (float)x3301;
float x3303 = x3297 / x3302;
float x3304 = x3296 - x3303;
x50[x3278] = x3304;
x58[x3278] = 0.0f;

}
for(int x3309=0; x3309 < 50; x3309++) {
float x3310 = x40[x3309];
float x3311 = x3310;
float x3312 = x3311;
bool x3313 = x3312 > 5.0f;
if (x3313) {
x3311 = 5.0f;
} else {
}
float x3317 = x3311;
bool x3318 = x3317 < -5.0f;
if (x3318) {
x3311 = -5.0f;
} else {
}
float x3322 = x115[x3309];
float x3323 = x3311;
float x3324 = x3323 * x3323;
float x3325 = x3322 + x3324;
x115[x3309] = x3325;
float x3327 = x39[x3309];
float x3329 = x115[x3309];
float x3328 = 0.1f * x3323;
double x3330 = (double)x3329;
double x3331 = x3330 + 9.99999993922529E-9;
double x3332 = sqrt(x3331);
float x3333 = (float)x3332;
float x3334 = x3328 / x3333;
float x3335 = x3327 - x3334;
x39[x3309] = x3335;
x40[x3309] = 0.0f;

}
for(int x3340=0; x3340 < 2500; x3340++) {
float x3341 = x38[x3340];
float x3342 = x3341;
float x3343 = x3342;
bool x3344 = x3343 > 5.0f;
if (x3344) {
x3342 = 5.0f;
} else {
}
float x3348 = x3342;
bool x3349 = x3348 < -5.0f;
if (x3349) {
x3342 = -5.0f;
} else {
}
float x3353 = x116[x3340];
float x3354 = x3342;
float x3355 = x3354 * x3354;
float x3356 = x3353 + x3355;
x116[x3340] = x3356;
float x3358 = x29[x3340];
float x3360 = x116[x3340];
float x3359 = 0.1f * x3354;
double x3361 = (double)x3360;
double x3362 = x3361 + 9.99999993922529E-9;
double x3363 = sqrt(x3362);
float x3364 = (float)x3363;
float x3365 = x3359 / x3364;
float x3366 = x3358 - x3365;
x29[x3340] = x3366;
x38[x3340] = 0.0f;

}
for(int x3371=0; x3371 < 1300; x3371++) {
float x3372 = x28[x3371];
float x3373 = x3372;
float x3374 = x3373;
bool x3375 = x3374 > 5.0f;
if (x3375) {
x3373 = 5.0f;
} else {
}
float x3379 = x3373;
bool x3380 = x3379 < -5.0f;
if (x3380) {
x3373 = -5.0f;
} else {
}
float x3384 = x117[x3371];
float x3385 = x3373;
float x3386 = x3385 * x3385;
float x3387 = x3384 + x3386;
x117[x3371] = x3387;
float x3389 = x18[x3371];
float x3391 = x117[x3371];
float x3390 = 0.1f * x3385;
double x3392 = (double)x3391;
double x3393 = x3392 + 9.99999993922529E-9;
double x3394 = sqrt(x3393);
float x3395 = (float)x3394;
float x3396 = x3390 / x3395;
float x3397 = x3389 - x3396;
x18[x3371] = x3397;
x28[x3371] = 0.0f;

}
for(int x3402=0; x3402 < 1300; x3402++) {
float x3403 = x69[x3402];
float x3404 = x3403;
float x3405 = x3404;
bool x3406 = x3405 > 5.0f;
if (x3406) {
x3404 = 5.0f;
} else {
}
float x3410 = x3404;
bool x3411 = x3410 < -5.0f;
if (x3411) {
x3404 = -5.0f;
} else {
}
float x3415 = x118[x3402];
float x3416 = x3404;
float x3417 = x3416 * x3416;
float x3418 = x3415 + x3417;
x118[x3402] = x3418;
float x3420 = x61[x3402];
float x3422 = x118[x3402];
float x3421 = 0.1f * x3416;
double x3423 = (double)x3422;
double x3424 = x3423 + 9.99999993922529E-9;
double x3425 = sqrt(x3424);
float x3426 = (float)x3425;
float x3427 = x3421 / x3426;
float x3428 = x3420 - x3427;
x61[x3402] = x3428;
x69[x3402] = 0.0f;

}
for(int x3433=0; x3433 < 50; x3433++) {
float x3434 = x80[x3433];
float x3435 = x3434;
float x3436 = x3435;
bool x3437 = x3436 > 5.0f;
if (x3437) {
x3435 = 5.0f;
} else {
}
float x3441 = x3435;
bool x3442 = x3441 < -5.0f;
if (x3442) {
x3435 = -5.0f;
} else {
}
float x3446 = x119[x3433];
float x3447 = x3435;
float x3448 = x3447 * x3447;
float x3449 = x3446 + x3448;
x119[x3433] = x3449;
float x3451 = x79[x3433];
float x3453 = x119[x3433];
float x3452 = 0.1f * x3447;
double x3454 = (double)x3453;
double x3455 = x3454 + 9.99999993922529E-9;
double x3456 = sqrt(x3455);
float x3457 = (float)x3456;
float x3458 = x3452 / x3457;
float x3459 = x3451 - x3458;
x79[x3433] = x3459;
x80[x3433] = 0.0f;

}
for(int x3464=0; x3464 < 2500; x3464++) {
float x3465 = x78[x3464];
float x3466 = x3465;
float x3467 = x3466;
bool x3468 = x3467 > 5.0f;
if (x3468) {
x3466 = 5.0f;
} else {
}
float x3472 = x3466;
bool x3473 = x3472 < -5.0f;
if (x3473) {
x3466 = -5.0f;
} else {
}
float x3477 = x120[x3464];
float x3478 = x3466;
float x3479 = x3478 * x3478;
float x3480 = x3477 + x3479;
x120[x3464] = x3480;
float x3482 = x70[x3464];
float x3484 = x120[x3464];
float x3483 = 0.1f * x3478;
double x3485 = (double)x3484;
double x3486 = x3485 + 9.99999993922529E-9;
double x3487 = sqrt(x3486);
float x3488 = (float)x3487;
float x3489 = x3483 / x3488;
float x3490 = x3482 - x3489;
x70[x3464] = x3490;
x78[x3464] = 0.0f;

}
for(int x3495=0; x3495 < 26; x3495++) {
float x3496 = x111[x3495];
float x3497 = x3496;
float x3498 = x3497;
bool x3499 = x3498 > 5.0f;
if (x3499) {
x3497 = 5.0f;
} else {
}
float x3503 = x3497;
bool x3504 = x3503 < -5.0f;
if (x3504) {
x3497 = -5.0f;
} else {
}
float x3508 = x121[x3495];
float x3509 = x3497;
float x3510 = x3509 * x3509;
float x3511 = x3508 + x3510;
x121[x3495] = x3511;
float x3513 = x110[x3495];
float x3515 = x121[x3495];
float x3514 = 0.1f * x3509;
double x3516 = (double)x3515;
double x3517 = x3516 + 9.99999993922529E-9;
double x3518 = sqrt(x3517);
float x3519 = (float)x3518;
float x3520 = x3514 / x3519;
float x3521 = x3513 - x3520;
x110[x3495] = x3521;
x111[x3495] = 0.0f;

}
for(int x3526=0; x3526 < 1300; x3526++) {
float x3527 = x109[x3526];
float x3528 = x3527;
float x3529 = x3528;
bool x3530 = x3529 > 5.0f;
if (x3530) {
x3528 = 5.0f;
} else {
}
float x3534 = x3528;
bool x3535 = x3534 < -5.0f;
if (x3535) {
x3528 = -5.0f;
} else {
}
float x3539 = x122[x3526];
float x3540 = x3528;
float x3541 = x3540 * x3540;
float x3542 = x3539 + x3541;
x122[x3526] = x3542;
float x3544 = x101[x3526];
float x3546 = x122[x3526];
float x3545 = 0.1f * x3540;
double x3547 = (double)x3546;
double x3548 = x3547 + 9.99999993922529E-9;
double x3549 = sqrt(x3548);
float x3550 = (float)x3549;
float x3551 = x3545 / x3550;
float x3552 = x3544 - x3551;
x101[x3526] = x3552;
x109[x3526] = 0.0f;

}
for(int x3557=0; x3557 < 2500; x3557++) {
float x3558 = x98[x3557];
float x3559 = x3558;
float x3560 = x3559;
bool x3561 = x3560 > 5.0f;
if (x3561) {
x3559 = 5.0f;
} else {
}
float x3565 = x3559;
bool x3566 = x3565 < -5.0f;
if (x3566) {
x3559 = -5.0f;
} else {
}
float x3570 = x123[x3557];
float x3571 = x3559;
float x3572 = x3571 * x3571;
float x3573 = x3570 + x3572;
x123[x3557] = x3573;
float x3575 = x90[x3557];
float x3577 = x123[x3557];
float x3576 = 0.1f * x3571;
double x3578 = (double)x3577;
double x3579 = x3578 + 9.99999993922529E-9;
double x3580 = sqrt(x3579);
float x3581 = (float)x3580;
float x3582 = x3576 / x3581;
float x3583 = x3575 - x3582;
x90[x3557] = x3583;
x98[x3557] = 0.0f;

}
for(int x3588=0; x3588 < 1300; x3588++) {
float x3589 = x89[x3588];
float x3590 = x3589;
float x3591 = x3590;
bool x3592 = x3591 > 5.0f;
if (x3592) {
x3590 = 5.0f;
} else {
}
float x3596 = x3590;
bool x3597 = x3596 < -5.0f;
if (x3597) {
x3590 = -5.0f;
} else {
}
float x3601 = x124[x3588];
float x3602 = x3590;
float x3603 = x3602 * x3602;
float x3604 = x3601 + x3603;
x124[x3588] = x3604;
float x3606 = x81[x3588];
float x3608 = x124[x3588];
float x3607 = 0.1f * x3602;
double x3609 = (double)x3608;
double x3610 = x3609 + 9.99999993922529E-9;
double x3611 = sqrt(x3610);
float x3612 = (float)x3611;
float x3613 = x3607 / x3612;
float x3614 = x3606 - x3613;
x81[x3588] = x3614;
x89[x3588] = 0.0f;

}
for(int x3619=0; x3619 < 50; x3619++) {
float x3620 = x100[x3619];
float x3621 = x3620;
float x3622 = x3621;
bool x3623 = x3622 > 5.0f;
if (x3623) {
x3621 = 5.0f;
} else {
}
float x3627 = x3621;
bool x3628 = x3627 < -5.0f;
if (x3628) {
x3621 = -5.0f;
} else {
}
float x3632 = x125[x3619];
float x3633 = x3621;
float x3634 = x3633 * x3633;
float x3635 = x3632 + x3634;
x125[x3619] = x3635;
float x3637 = x99[x3619];
float x3639 = x125[x3619];
float x3638 = 0.1f * x3633;
double x3640 = (double)x3639;
double x3641 = x3640 + 9.99999993922529E-9;
double x3642 = sqrt(x3641);
float x3643 = (float)x3642;
float x3644 = x3638 / x3643;
float x3645 = x3637 - x3644;
x99[x3619] = x3645;
x100[x3619] = 0.0f;

}
int64_t x3650 = (long)mallocAddr;
int64_t x3651 = x3650 - x128;
memset((void*)x128, 0, x3651);
mallocAddr = (void*)x128;

}
double x3656 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x3659 = (long)fopen(x0, "w");
fprintf((FILE *)x3659, "unit: %s\n", "100 iteration");
for(int x3662=0; x3662 < 51; x3662++) {
double x3663 = x127[x3662];
fprintf((FILE *)x3659, "%lf\n", x3663);

}
double x3657 = x126 - x2;
double x3658 = x3656 - x126;
fprintf((FILE *)x3659, "run time: %lf %lf\n", x3657, x3658);
fclose((FILE*)x3659);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

