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
// Backend setup.
double x2 = ((double)clock() / CLOCKS_PER_SEC);
int32_t x3 = open("graham.txt",0);
int32_t x4 = fsize(x3);
printf("LSTM Test: >> data has %d chars\n",x4);
int* x7 = (int32_t*)myMalloc(x4 * sizeof(int32_t));;
char* x5 = (char*)mmap(0, x4, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x3, 0);
for(int x9=0; x9 < x4; x9++) {
char x10 = x5[x9];
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
float* x25 = (float*)myMalloc(1300 * sizeof(float));;
float* x26 = (float*)myMalloc(2500 * sizeof(float));;
for(int x28=0; x28 < 2500; x28++) {
float x29 = (float)rand()/RAND_MAX;
float x30 = x29 - 0.5f;
float x31 = x30 * 0.14142136f;
x26[x28] = x31;

}
float* x35 = (float*)myMalloc(2500 * sizeof(float));;
float* x36 = (float*)myMalloc(50 * sizeof(float));;
float* x37 = (float*)myMalloc(50 * sizeof(float));;
float* x38 = (float*)myMalloc(1300 * sizeof(float));;
for(int x39=0; x39 < 1300; x39++) {
float x40 = (float)rand()/RAND_MAX;
float x41 = x40 - 0.5f;
float x42 = x41 * 0.19611613f;
x38[x39] = x42;

}
float* x46 = (float*)myMalloc(1300 * sizeof(float));;
float* x47 = (float*)myMalloc(2500 * sizeof(float));;
for(int x48=0; x48 < 2500; x48++) {
float x49 = (float)rand()/RAND_MAX;
float x50 = x49 - 0.5f;
float x51 = x50 * 0.14142136f;
x47[x48] = x51;

}
float* x55 = (float*)myMalloc(2500 * sizeof(float));;
float* x56 = (float*)myMalloc(50 * sizeof(float));;
float* x57 = (float*)myMalloc(50 * sizeof(float));;
float* x58 = (float*)myMalloc(1300 * sizeof(float));;
for(int x59=0; x59 < 1300; x59++) {
float x60 = (float)rand()/RAND_MAX;
float x61 = x60 - 0.5f;
float x62 = x61 * 0.19611613f;
x58[x59] = x62;

}
float* x66 = (float*)myMalloc(1300 * sizeof(float));;
float* x67 = (float*)myMalloc(2500 * sizeof(float));;
for(int x68=0; x68 < 2500; x68++) {
float x69 = (float)rand()/RAND_MAX;
float x70 = x69 - 0.5f;
float x71 = x70 * 0.14142136f;
x67[x68] = x71;

}
float* x75 = (float*)myMalloc(2500 * sizeof(float));;
float* x76 = (float*)myMalloc(50 * sizeof(float));;
float* x77 = (float*)myMalloc(50 * sizeof(float));;
float* x78 = (float*)myMalloc(1300 * sizeof(float));;
for(int x79=0; x79 < 1300; x79++) {
float x80 = (float)rand()/RAND_MAX;
float x81 = x80 - 0.5f;
float x82 = x81 * 0.19611613f;
x78[x79] = x82;

}
float* x86 = (float*)myMalloc(1300 * sizeof(float));;
float* x87 = (float*)myMalloc(2500 * sizeof(float));;
for(int x88=0; x88 < 2500; x88++) {
float x89 = (float)rand()/RAND_MAX;
float x90 = x89 - 0.5f;
float x91 = x90 * 0.14142136f;
x87[x88] = x91;

}
float* x95 = (float*)myMalloc(2500 * sizeof(float));;
float* x96 = (float*)myMalloc(50 * sizeof(float));;
float* x97 = (float*)myMalloc(50 * sizeof(float));;
float* x98 = (float*)myMalloc(1300 * sizeof(float));;
for(int x99=0; x99 < 1300; x99++) {
float x100 = (float)rand()/RAND_MAX;
float x101 = x100 - 0.5f;
float x102 = x101 * 0.14142136f;
x98[x99] = x102;

}
float* x106 = (float*)myMalloc(1300 * sizeof(float));;
float* x107 = (float*)myMalloc(26 * sizeof(float));;
float* x108 = (float*)myMalloc(26 * sizeof(float));;
float* x109 = (float*)myMalloc(1300 * sizeof(float));;
float* x110 = (float*)myMalloc(50 * sizeof(float));;
float* x111 = (float*)myMalloc(2500 * sizeof(float));;
float* x112 = (float*)myMalloc(50 * sizeof(float));;
float* x113 = (float*)myMalloc(2500 * sizeof(float));;
float* x114 = (float*)myMalloc(1300 * sizeof(float));;
float* x115 = (float*)myMalloc(1300 * sizeof(float));;
float* x116 = (float*)myMalloc(50 * sizeof(float));;
float* x117 = (float*)myMalloc(2500 * sizeof(float));;
float* x118 = (float*)myMalloc(26 * sizeof(float));;
float* x119 = (float*)myMalloc(1300 * sizeof(float));;
float* x120 = (float*)myMalloc(2500 * sizeof(float));;
float* x121 = (float*)myMalloc(1300 * sizeof(float));;
float* x122 = (float*)myMalloc(50 * sizeof(float));;
double x123 = ((double)clock() / CLOCKS_PER_SEC);
double* x124 = (double*)myMalloc(51 * sizeof(double));;
int64_t x125 = (long)mallocAddr;
int32_t x126 = 0;
x126 -= 400;
double x128 = 70.0;
for(int x130=0; x130 < 5001; x130++) {
float* x154 = (float*)myMalloc(1 * sizeof(float));;
float* x155 = (float*)myMalloc(10400 * sizeof(float));;
float* x172 = (float*)myMalloc(10400 * sizeof(float));;
int* x140 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
function<void(int32_t,float**)> x797 = [&](int32_t x798,float** x799) {
float** x801 = x799;
float* x802 = x801[0];
float* x803 = x801[1];
float* x804 = x801[2];
float* x805 = x801[3];
float* x806 = x801[4];
float* x807 = x801[5];
int32_t x800 = x798;
bool x808 = x800 < 20;
if (x808) {
int32_t x809 = x800 * 520;
float* x810 = x155+x809;
float* x811 = x172+x809;
// dot: WrappedArray(20, 26), List(26, 50)
float* x813 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x810,26,x16,50,0,x813,50);
float* x815 = (float*)myMalloc(1000 * sizeof(float));;
// dot: List(20, 50), List(50, 50)
float* x817 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x804,50,x26,50,0,x817,50);
float* x819 = (float*)myMalloc(1000 * sizeof(float));;
float* x820 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x821 = 0;
int32_t x822 = 0;
int32_t x823 = 0;
for(int x824=0; x824 < 20; x824++) {
int32_t x825 = x822;
int32_t x826 = x823;
int32_t x827 = x821;
int32_t x828 = x827;
int32_t x829 = x825;
int32_t x830 = x826;
for(int x831=0; x831 < 50; x831++) {
int32_t x832 = x828;
int32_t x833 = x829;
float x834 = x813[x833];
int32_t x835 = x830;
float x836 = x817[x835];
float x837 = x834 + x836;
x820[x832] = x837;
x828 += 1;
x829 += 1;
x830 += 1;

}
x821 += 50;
x822 += 50;
x823 += 50;

}
float* x849 = (float*)myMalloc(1000 * sizeof(float));;
float* x850 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x851 = 0;
int32_t x852 = 0;
int32_t x853 = 0;
for(int x854=0; x854 < 20; x854++) {
int32_t x855 = x852;
int32_t x856 = x853;
int32_t x857 = x851;
int32_t x858 = x857;
int32_t x859 = x855;
int32_t x860 = x856;
for(int x861=0; x861 < 50; x861++) {
int32_t x862 = x858;
int32_t x863 = x859;
float x864 = x820[x863];
int32_t x865 = x860;
float x866 = x36[x865];
float x867 = x864 + x866;
x850[x862] = x867;
x858 += 1;
x859 += 1;
x860 += 1;

}
x851 += 50;
x852 += 50;

}
float* x878 = (float*)myMalloc(1000 * sizeof(float));;
float* x879 = (float*)myMalloc(1000 * sizeof(float));;
for(int x880=0; x880 < 1000; x880++) {
float x881 = x850[x880];
float x882 = -1.0f * x881;
double x883 = (double)x882;
double x884 = exp(x883);
float x885 = (float)x884;
float x886 = x885 + 1.0f;
float x887 = 1.0f / x886;
x879[x880] = x887;

}
float* x891 = (float*)myMalloc(1000 * sizeof(float));;
// dot: WrappedArray(20, 26), List(26, 50)
float* x893 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x810,26,x38,50,0,x893,50);
float* x895 = (float*)myMalloc(1000 * sizeof(float));;
// dot: List(20, 50), List(50, 50)
float* x897 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x804,50,x47,50,0,x897,50);
float* x899 = (float*)myMalloc(1000 * sizeof(float));;
float* x900 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x901 = 0;
int32_t x902 = 0;
int32_t x903 = 0;
for(int x904=0; x904 < 20; x904++) {
int32_t x905 = x902;
int32_t x906 = x903;
int32_t x907 = x901;
int32_t x908 = x907;
int32_t x909 = x905;
int32_t x910 = x906;
for(int x911=0; x911 < 50; x911++) {
int32_t x912 = x908;
int32_t x913 = x909;
float x914 = x893[x913];
int32_t x915 = x910;
float x916 = x897[x915];
float x917 = x914 + x916;
x900[x912] = x917;
x908 += 1;
x909 += 1;
x910 += 1;

}
x901 += 50;
x902 += 50;
x903 += 50;

}
float* x929 = (float*)myMalloc(1000 * sizeof(float));;
float* x930 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x931 = 0;
int32_t x932 = 0;
int32_t x933 = 0;
for(int x934=0; x934 < 20; x934++) {
int32_t x935 = x932;
int32_t x936 = x933;
int32_t x937 = x931;
int32_t x938 = x937;
int32_t x939 = x935;
int32_t x940 = x936;
for(int x941=0; x941 < 50; x941++) {
int32_t x942 = x938;
int32_t x943 = x939;
float x944 = x900[x943];
int32_t x945 = x940;
float x946 = x56[x945];
float x947 = x944 + x946;
x930[x942] = x947;
x938 += 1;
x939 += 1;
x940 += 1;

}
x931 += 50;
x932 += 50;

}
float* x958 = (float*)myMalloc(1000 * sizeof(float));;
float* x959 = (float*)myMalloc(1000 * sizeof(float));;
for(int x960=0; x960 < 1000; x960++) {
float x961 = x930[x960];
float x962 = -1.0f * x961;
double x963 = (double)x962;
double x964 = exp(x963);
float x965 = (float)x964;
float x966 = x965 + 1.0f;
float x967 = 1.0f / x966;
x959[x960] = x967;

}
float* x971 = (float*)myMalloc(1000 * sizeof(float));;
// dot: WrappedArray(20, 26), List(26, 50)
float* x973 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x810,26,x78,50,0,x973,50);
float* x975 = (float*)myMalloc(1000 * sizeof(float));;
// dot: List(20, 50), List(50, 50)
float* x977 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x804,50,x87,50,0,x977,50);
float* x979 = (float*)myMalloc(1000 * sizeof(float));;
float* x980 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x981 = 0;
int32_t x982 = 0;
int32_t x983 = 0;
for(int x984=0; x984 < 20; x984++) {
int32_t x985 = x982;
int32_t x986 = x983;
int32_t x987 = x981;
int32_t x988 = x987;
int32_t x989 = x985;
int32_t x990 = x986;
for(int x991=0; x991 < 50; x991++) {
int32_t x992 = x988;
int32_t x993 = x989;
float x994 = x973[x993];
int32_t x995 = x990;
float x996 = x977[x995];
float x997 = x994 + x996;
x980[x992] = x997;
x988 += 1;
x989 += 1;
x990 += 1;

}
x981 += 50;
x982 += 50;
x983 += 50;

}
float* x1009 = (float*)myMalloc(1000 * sizeof(float));;
float* x1010 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1011 = 0;
int32_t x1012 = 0;
int32_t x1013 = 0;
for(int x1014=0; x1014 < 20; x1014++) {
int32_t x1015 = x1012;
int32_t x1016 = x1013;
int32_t x1017 = x1011;
int32_t x1018 = x1017;
int32_t x1019 = x1015;
int32_t x1020 = x1016;
for(int x1021=0; x1021 < 50; x1021++) {
int32_t x1022 = x1018;
int32_t x1023 = x1019;
float x1024 = x980[x1023];
int32_t x1025 = x1020;
float x1026 = x96[x1025];
float x1027 = x1024 + x1026;
x1010[x1022] = x1027;
x1018 += 1;
x1019 += 1;
x1020 += 1;

}
x1011 += 50;
x1012 += 50;

}
float* x1038 = (float*)myMalloc(1000 * sizeof(float));;
float* x1039 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1040=0; x1040 < 1000; x1040++) {
float x1041 = x1010[x1040];
float x1042 = -1.0f * x1041;
double x1043 = (double)x1042;
double x1044 = exp(x1043);
float x1045 = (float)x1044;
float x1046 = x1045 + 1.0f;
float x1047 = 1.0f / x1046;
x1039[x1040] = x1047;

}
float* x1051 = (float*)myMalloc(1000 * sizeof(float));;
// dot: WrappedArray(20, 26), List(26, 50)
float* x1053 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x810,26,x58,50,0,x1053,50);
float* x1055 = (float*)myMalloc(1000 * sizeof(float));;
// dot: List(20, 50), List(50, 50)
float* x1057 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x804,50,x67,50,0,x1057,50);
float* x1059 = (float*)myMalloc(1000 * sizeof(float));;
float* x1060 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1061 = 0;
int32_t x1062 = 0;
int32_t x1063 = 0;
for(int x1064=0; x1064 < 20; x1064++) {
int32_t x1065 = x1062;
int32_t x1066 = x1063;
int32_t x1067 = x1061;
int32_t x1068 = x1067;
int32_t x1069 = x1065;
int32_t x1070 = x1066;
for(int x1071=0; x1071 < 50; x1071++) {
int32_t x1072 = x1068;
int32_t x1073 = x1069;
float x1074 = x1053[x1073];
int32_t x1075 = x1070;
float x1076 = x1057[x1075];
float x1077 = x1074 + x1076;
x1060[x1072] = x1077;
x1068 += 1;
x1069 += 1;
x1070 += 1;

}
x1061 += 50;
x1062 += 50;
x1063 += 50;

}
float* x1089 = (float*)myMalloc(1000 * sizeof(float));;
float* x1090 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1091 = 0;
int32_t x1092 = 0;
int32_t x1093 = 0;
for(int x1094=0; x1094 < 20; x1094++) {
int32_t x1095 = x1092;
int32_t x1096 = x1093;
int32_t x1097 = x1091;
int32_t x1098 = x1097;
int32_t x1099 = x1095;
int32_t x1100 = x1096;
for(int x1101=0; x1101 < 50; x1101++) {
int32_t x1102 = x1098;
int32_t x1103 = x1099;
float x1104 = x1060[x1103];
int32_t x1105 = x1100;
float x1106 = x76[x1105];
float x1107 = x1104 + x1106;
x1090[x1102] = x1107;
x1098 += 1;
x1099 += 1;
x1100 += 1;

}
x1091 += 50;
x1092 += 50;

}
float* x1118 = (float*)myMalloc(1000 * sizeof(float));;
float* x1119 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1120=0; x1120 < 1000; x1120++) {
float x1121 = x1090[x1120];
double x1122 = (double)x1121;
double x1123 = tanh(x1122);
float x1124 = (float)x1123;
x1119[x1120] = x1124;

}
float* x1128 = (float*)myMalloc(1000 * sizeof(float));;
float* x1129 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1130 = 0;
int32_t x1131 = 0;
int32_t x1132 = 0;
for(int x1133=0; x1133 < 20; x1133++) {
int32_t x1134 = x1131;
int32_t x1135 = x1132;
int32_t x1136 = x1130;
int32_t x1137 = x1136;
int32_t x1138 = x1134;
int32_t x1139 = x1135;
for(int x1140=0; x1140 < 50; x1140++) {
int32_t x1141 = x1137;
int32_t x1142 = x1138;
float x1143 = x879[x1142];
int32_t x1144 = x1139;
float x1145 = x806[x1144];
float x1146 = x1143 * x1145;
x1129[x1141] = x1146;
x1137 += 1;
x1138 += 1;
x1139 += 1;

}
x1130 += 50;
x1131 += 50;
x1132 += 50;

}
float* x1158 = (float*)myMalloc(1000 * sizeof(float));;
float* x1159 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1160 = 0;
int32_t x1161 = 0;
int32_t x1162 = 0;
for(int x1163=0; x1163 < 20; x1163++) {
int32_t x1164 = x1161;
int32_t x1165 = x1162;
int32_t x1166 = x1160;
int32_t x1167 = x1166;
int32_t x1168 = x1164;
int32_t x1169 = x1165;
for(int x1170=0; x1170 < 50; x1170++) {
int32_t x1171 = x1167;
int32_t x1172 = x1168;
float x1173 = x959[x1172];
int32_t x1174 = x1169;
float x1175 = x1119[x1174];
float x1176 = x1173 * x1175;
x1159[x1171] = x1176;
x1167 += 1;
x1168 += 1;
x1169 += 1;

}
x1160 += 50;
x1161 += 50;
x1162 += 50;

}
float* x1188 = (float*)myMalloc(1000 * sizeof(float));;
float* x1189 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1190 = 0;
int32_t x1191 = 0;
int32_t x1192 = 0;
for(int x1193=0; x1193 < 20; x1193++) {
int32_t x1194 = x1191;
int32_t x1195 = x1192;
int32_t x1196 = x1190;
int32_t x1197 = x1196;
int32_t x1198 = x1194;
int32_t x1199 = x1195;
for(int x1200=0; x1200 < 50; x1200++) {
int32_t x1201 = x1197;
int32_t x1202 = x1198;
float x1203 = x1129[x1202];
int32_t x1204 = x1199;
float x1205 = x1159[x1204];
float x1206 = x1203 + x1205;
x1189[x1201] = x1206;
x1197 += 1;
x1198 += 1;
x1199 += 1;

}
x1190 += 50;
x1191 += 50;
x1192 += 50;

}
float* x1218 = (float*)myMalloc(1000 * sizeof(float));;
float* x1219 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1220=0; x1220 < 1000; x1220++) {
float x1221 = x1189[x1220];
double x1222 = (double)x1221;
double x1223 = tanh(x1222);
float x1224 = (float)x1223;
x1219[x1220] = x1224;

}
float* x1228 = (float*)myMalloc(1000 * sizeof(float));;
float* x1229 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1230 = 0;
int32_t x1231 = 0;
int32_t x1232 = 0;
for(int x1233=0; x1233 < 20; x1233++) {
int32_t x1234 = x1231;
int32_t x1235 = x1232;
int32_t x1236 = x1230;
int32_t x1237 = x1236;
int32_t x1238 = x1234;
int32_t x1239 = x1235;
for(int x1240=0; x1240 < 50; x1240++) {
int32_t x1241 = x1237;
int32_t x1242 = x1238;
float x1243 = x1039[x1242];
int32_t x1244 = x1239;
float x1245 = x1219[x1244];
float x1246 = x1243 * x1245;
x1229[x1241] = x1246;
x1237 += 1;
x1238 += 1;
x1239 += 1;

}
x1230 += 50;
x1231 += 50;
x1232 += 50;

}
float* x1258 = (float*)myMalloc(1000 * sizeof(float));;
// dot: List(20, 50), List(50, 26)
float* x1260 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x1229,50,x98,26,0,x1260,26);
float* x1262 = (float*)myMalloc(520 * sizeof(float));;
float* x1263 = (float*)myMalloc(520 * sizeof(float));;
int32_t x1264 = 0;
int32_t x1265 = 0;
int32_t x1266 = 0;
for(int x1267=0; x1267 < 20; x1267++) {
int32_t x1268 = x1265;
int32_t x1269 = x1266;
int32_t x1270 = x1264;
int32_t x1271 = x1270;
int32_t x1272 = x1268;
int32_t x1273 = x1269;
for(int x1274=0; x1274 < 26; x1274++) {
int32_t x1275 = x1271;
int32_t x1276 = x1272;
float x1277 = x1260[x1276];
int32_t x1278 = x1273;
float x1279 = x107[x1278];
float x1280 = x1277 + x1279;
x1263[x1275] = x1280;
x1271 += 1;
x1272 += 1;
x1273 += 1;

}
x1264 += 26;
x1265 += 26;

}
float* x1291 = (float*)myMalloc(520 * sizeof(float));;
int* x1292 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x1293=0; x1293 < 20; x1293++) {
int32_t x1294 = x1293 * 20;
int32_t x1295 = x800 + x1294;
int32_t x1296 = x140[x1295];
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
int32_t x1343 = x1339;
float x1344 = x1337[x1343];
int32_t x1345 = x1341;
float x1346 = x1320[x1345];
float x1347 = x1344 + x1346;
x1337[x1343] = x1347;
x1341 += 1;

}

}
x1321 = 0;
for(int x1355=0; x1355 < 20; x1355++) {
float x1356 = x1300[x1355];
float x1357 = x1337[x1355];
double x1358 = (double)x1357;
double x1359 = log(x1358);
float x1360 = (float)x1359;
float x1361 = x1356 + x1360;
for(int x1362=0; x1362 < 26; x1362++) {
int32_t x1363 = x1321;
float x1364 = x1263[x1363];
float x1365 = x1364 - x1361;
x1320[x1363] = x1365;
x1321 += 1;

}

}
float* x1372 = (float*)myMalloc(520 * sizeof(float));;
float* x1373 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1374 = 0;
for(int x1375=0; x1375 < 20; x1375++) {
int32_t x1376 = x1374;
int32_t x1377 = x1292[x1375];
int32_t x1378 = x1376 + x1377;
float x1379 = x1320[x1378];
float x1380 = -1.0f * x1379;
x1373[x1375] = x1380;
x1374 += 26;

}
float* x1385 = (float*)myMalloc(20 * sizeof(float));;
float x1386 = 0.0f;
for(int x1387=0; x1387 < 20; x1387++) {
float x1388 = x1386;
float x1389 = x1373[x1387];
float x1390 = x1388 + x1389;
x1386 = x1390;

}
float x1394 = x1386;
float* x1395 = (float*)myMalloc(1 * sizeof(float));;
x1395[0] = x1394;
float* x1397 = (float*)myMalloc(1 * sizeof(float));;
float* x1398 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1399 = 0;
int32_t x1400 = 0;
int32_t x1401 = 0;
int32_t x1402 = x1399;
int32_t x1403 = x1400;
float x1404 = x802[x1403];
int32_t x1405 = x1401;
float x1406 = x1395[x1405];
float x1407 = x1404 + x1406;
x1398[x1402] = x1407;
x1399 += 1;
float* x1410 = (float*)myMalloc(1 * sizeof(float));;
float** x1412 = (float**)myMalloc(6 * sizeof(float*));;
x1412[0] = x1398;
x1412[1] = x1410;
x1412[2] = x1229;
x1412[3] = x1258;
x1412[4] = x1189;
x1412[5] = x1218;
int32_t x1421 = 0;
int32_t x1422 = 0;
int32_t x1423 = 0;
int32_t x1424 = x1421;
int32_t x1427 = x1422;
int32_t x1429 = x1423;
x1423 += 1;
int32_t x1448 = 0;
float* x1461 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1478 = 0;
int32_t x1498 = 0;
int32_t x1499 = 0;
int32_t x1500 = 0;
int32_t x1535 = 0;
int32_t x1536 = 0;
int32_t x1537 = 0;
int32_t x1584 = 0;
int32_t x1585 = 0;
int32_t x1586 = 0;
int32_t x1620 = 0;
int32_t x1621 = 0;
int32_t x1622 = 0;
int32_t x1658 = 0;
int32_t x1659 = 0;
int32_t x1660 = 0;
int32_t x1707 = 0;
int32_t x1708 = 0;
int32_t x1709 = 0;
int32_t x1742 = 0;
int32_t x1743 = 0;
int32_t x1744 = 0;
int32_t x1793 = 0;
int32_t x1794 = 0;
int32_t x1795 = 0;
int32_t x1828 = 0;
int32_t x1829 = 0;
int32_t x1830 = 0;
int32_t x1879 = 0;
int32_t x1880 = 0;
int32_t x1881 = 0;
int32_t x1914 = 0;
int32_t x1915 = 0;
int32_t x1916 = 0;
int32_t x1965 = 0;
int32_t x1966 = 0;
int32_t x1967 = 0;
int32_t x2000 = 0;
int32_t x2001 = 0;
int32_t x2002 = 0;
int32_t x1411 = x800 + 1;
x797(x1411,x1412);
float x1425 = x803[x1424];
float x1426 = x802[x1424];
float x1428 = x1395[x1427];
float x1430 = x1410[x1429];
float x1431 = x1425 + x1430;
x803[x1424] = x1431;
float x1433 = x1397[x1427];
float x1434 = x802[x1424];
float x1435 = x1395[x1427];
float x1436 = x1410[x1429];
float x1437 = x1433 + x1436;
x1397[x1427] = x1437;
// += tensor of dim 0
float x1441 = x1397[0];
for(int x1442=0; x1442 < 20; x1442++) {
float x1443 = x1385[x1442];
float x1444 = x1443 + x1441;
x1385[x1442] = x1444;

}
for(int x1449=0; x1449 < 20; x1449++) {
int32_t x1450 = x1448;
int32_t x1451 = x1292[x1449];
int32_t x1452 = x1450 + x1451;
float x1453 = x1372[x1452];
float x1454 = x1385[x1449];
float x1455 = -1.0f * x1454;
float x1456 = x1453 + x1455;
x1372[x1452] = x1456;
x1448 += 26;

}
for(int x1462=0; x1462 < 20; x1462++) {
int32_t x1463 = x1462;
int32_t x1464 = x1462 * 26;
int32_t x1465 = x1464;
for(int x1466=0; x1466 < 26; x1466++) {
int32_t x1467 = x1463;
float x1468 = x1461[x1467];
int32_t x1469 = x1465;
float x1470 = x1372[x1469];
float x1471 = x1468 + x1470;
x1461[x1467] = x1471;
x1465 += 1;

}

}
for(int x1479=0; x1479 < 20; x1479++) {
for(int x1480=0; x1480 < 26; x1480++) {
int32_t x1481 = x1478;
float x1482 = x1291[x1481];
float x1483 = x1372[x1481];
float x1484 = x1320[x1481];
float x1488 = x1461[x1479];
double x1485 = (double)x1484;
double x1486 = exp(x1485);
float x1487 = (float)x1486;
float x1489 = x1487 * x1488;
float x1490 = x1483 - x1489;
float x1491 = x1482 + x1490;
x1291[x1481] = x1491;
x1478 += 1;

}

}
for(int x1501=0; x1501 < 20; x1501++) {
int32_t x1502 = x1498;
int32_t x1503 = x1499;
int32_t x1504 = x1500;
int32_t x1505 = x1502;
int32_t x1506 = x1503;
int32_t x1507 = x1504;
for(int x1508=0; x1508 < 26; x1508++) {
int32_t x1509 = x1505;
float x1510 = x1262[x1509];
float x1511 = x1260[x1509];
int32_t x1512 = x1506;
float x1513 = x107[x1512];
int32_t x1514 = x1507;
float x1515 = x1291[x1514];
float x1516 = x1510 + x1515;
x1262[x1509] = x1516;
float x1518 = x108[x1512];
float x1519 = x1260[x1509];
float x1520 = x107[x1512];
float x1521 = x1291[x1514];
float x1522 = x1518 + x1521;
x108[x1512] = x1522;
x1507 += 1;
x1505 += 1;
x1506 += 1;

}
x1500 += 26;
x1498 += 26;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x1262,26,x98,26,1,x1258,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x1229,50,x1262,26,1,x106,26);
for(int x1538=0; x1538 < 20; x1538++) {
int32_t x1539 = x1535;
int32_t x1540 = x1536;
int32_t x1541 = x1537;
int32_t x1542 = x1539;
int32_t x1543 = x1540;
int32_t x1544 = x1541;
for(int x1545=0; x1545 < 50; x1545++) {
int32_t x1546 = x1542;
float x1547 = x1051[x1546];
float x1548 = x1039[x1546];
int32_t x1549 = x1543;
float x1550 = x1219[x1549];
int32_t x1551 = x1544;
float x1552 = x1258[x1551];
float x1553 = x1552 * x1550;
float x1554 = x1547 + x1553;
x1051[x1546] = x1554;
float x1556 = x1228[x1549];
float x1557 = x1039[x1546];
float x1558 = x1219[x1549];
float x1559 = x1258[x1551];
float x1560 = x1559 * x1557;
float x1561 = x1556 + x1560;
x1228[x1549] = x1561;
x1544 += 1;
x1542 += 1;
x1543 += 1;

}
x1537 += 50;
x1535 += 50;
x1536 += 50;

}
for(int x1573=0; x1573 < 1000; x1573++) {
float x1574 = x1218[x1573];
float x1575 = x1219[x1573];
float x1578 = x1228[x1573];
float x1576 = x1575 * x1575;
float x1577 = 1.0f - x1576;
float x1579 = x1577 * x1578;
float x1580 = x1574 + x1579;
x1218[x1573] = x1580;

}
for(int x1587=0; x1587 < 20; x1587++) {
int32_t x1588 = x1584;
int32_t x1589 = x1585;
int32_t x1590 = x1586;
int32_t x1591 = x1588;
int32_t x1592 = x1589;
int32_t x1593 = x1590;
for(int x1594=0; x1594 < 50; x1594++) {
int32_t x1595 = x1591;
float x1596 = x1158[x1595];
float x1597 = x1129[x1595];
int32_t x1598 = x1592;
float x1599 = x1159[x1598];
int32_t x1600 = x1593;
float x1601 = x1218[x1600];
float x1602 = x1596 + x1601;
x1158[x1595] = x1602;
float x1604 = x1188[x1598];
float x1605 = x1129[x1595];
float x1606 = x1159[x1598];
float x1607 = x1218[x1600];
float x1608 = x1604 + x1607;
x1188[x1598] = x1608;
x1593 += 1;
x1591 += 1;
x1592 += 1;

}
x1586 += 50;
x1584 += 50;
x1585 += 50;

}
for(int x1623=0; x1623 < 20; x1623++) {
int32_t x1624 = x1620;
int32_t x1625 = x1621;
int32_t x1626 = x1622;
int32_t x1627 = x1624;
int32_t x1628 = x1625;
int32_t x1629 = x1626;
for(int x1630=0; x1630 < 50; x1630++) {
int32_t x1631 = x1627;
float x1632 = x971[x1631];
float x1633 = x959[x1631];
int32_t x1634 = x1628;
float x1635 = x1119[x1634];
int32_t x1636 = x1629;
float x1637 = x1188[x1636];
float x1638 = x1637 * x1635;
float x1639 = x1632 + x1638;
x971[x1631] = x1639;
float x1641 = x1128[x1634];
float x1642 = x959[x1631];
float x1643 = x1119[x1634];
float x1644 = x1188[x1636];
float x1645 = x1644 * x1642;
float x1646 = x1641 + x1645;
x1128[x1634] = x1646;
x1629 += 1;
x1627 += 1;
x1628 += 1;

}
x1622 += 50;
x1620 += 50;
x1621 += 50;

}
for(int x1661=0; x1661 < 20; x1661++) {
int32_t x1662 = x1658;
int32_t x1663 = x1659;
int32_t x1664 = x1660;
int32_t x1665 = x1662;
int32_t x1666 = x1663;
int32_t x1667 = x1664;
for(int x1668=0; x1668 < 50; x1668++) {
int32_t x1669 = x1665;
float x1670 = x891[x1669];
float x1671 = x879[x1669];
int32_t x1672 = x1666;
float x1673 = x806[x1672];
int32_t x1674 = x1667;
float x1675 = x1158[x1674];
float x1676 = x1675 * x1673;
float x1677 = x1670 + x1676;
x891[x1669] = x1677;
float x1679 = x807[x1672];
float x1680 = x879[x1669];
float x1681 = x806[x1672];
float x1682 = x1158[x1674];
float x1683 = x1682 * x1680;
float x1684 = x1679 + x1683;
x807[x1672] = x1684;
x1667 += 1;
x1665 += 1;
x1666 += 1;

}
x1660 += 50;
x1658 += 50;
x1659 += 50;

}
for(int x1696=0; x1696 < 1000; x1696++) {
float x1697 = x1118[x1696];
float x1698 = x1119[x1696];
float x1701 = x1128[x1696];
float x1699 = x1698 * x1698;
float x1700 = 1.0f - x1699;
float x1702 = x1700 * x1701;
float x1703 = x1697 + x1702;
x1118[x1696] = x1703;

}
for(int x1710=0; x1710 < 20; x1710++) {
int32_t x1711 = x1707;
int32_t x1712 = x1708;
int32_t x1713 = x1709;
int32_t x1714 = x1711;
int32_t x1715 = x1712;
int32_t x1716 = x1713;
for(int x1717=0; x1717 < 50; x1717++) {
int32_t x1718 = x1714;
float x1719 = x1089[x1718];
float x1720 = x1060[x1718];
int32_t x1721 = x1715;
float x1722 = x76[x1721];
int32_t x1723 = x1716;
float x1724 = x1118[x1723];
float x1725 = x1719 + x1724;
x1089[x1718] = x1725;
float x1727 = x77[x1721];
float x1728 = x1060[x1718];
float x1729 = x76[x1721];
float x1730 = x1118[x1723];
float x1731 = x1727 + x1730;
x77[x1721] = x1731;
x1716 += 1;
x1714 += 1;
x1715 += 1;

}
x1709 += 50;
x1707 += 50;

}
for(int x1745=0; x1745 < 20; x1745++) {
int32_t x1746 = x1742;
int32_t x1747 = x1743;
int32_t x1748 = x1744;
int32_t x1749 = x1746;
int32_t x1750 = x1747;
int32_t x1751 = x1748;
for(int x1752=0; x1752 < 50; x1752++) {
int32_t x1753 = x1749;
float x1754 = x1055[x1753];
float x1755 = x1053[x1753];
int32_t x1756 = x1750;
float x1757 = x1057[x1756];
int32_t x1758 = x1751;
float x1759 = x1089[x1758];
float x1760 = x1754 + x1759;
x1055[x1753] = x1760;
float x1762 = x1059[x1756];
float x1763 = x1053[x1753];
float x1764 = x1057[x1756];
float x1765 = x1089[x1758];
float x1766 = x1762 + x1765;
x1059[x1756] = x1766;
x1751 += 1;
x1749 += 1;
x1750 += 1;

}
x1744 += 50;
x1742 += 50;
x1743 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x1059,50,x67,50,1,x805,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x804,50,x1059,50,1,x75,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x1055,50,x58,50,1,x811,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x810,26,x1055,50,1,x66,50);
for(int x1782=0; x1782 < 1000; x1782++) {
float x1783 = x1038[x1782];
float x1784 = x1039[x1782];
float x1787 = x1051[x1782];
float x1785 = 1.0f - x1784;
float x1786 = x1785 * x1784;
float x1788 = x1786 * x1787;
float x1789 = x1783 + x1788;
x1038[x1782] = x1789;

}
for(int x1796=0; x1796 < 20; x1796++) {
int32_t x1797 = x1793;
int32_t x1798 = x1794;
int32_t x1799 = x1795;
int32_t x1800 = x1797;
int32_t x1801 = x1798;
int32_t x1802 = x1799;
for(int x1803=0; x1803 < 50; x1803++) {
int32_t x1804 = x1800;
float x1805 = x1009[x1804];
float x1806 = x980[x1804];
int32_t x1807 = x1801;
float x1808 = x96[x1807];
int32_t x1809 = x1802;
float x1810 = x1038[x1809];
float x1811 = x1805 + x1810;
x1009[x1804] = x1811;
float x1813 = x97[x1807];
float x1814 = x980[x1804];
float x1815 = x96[x1807];
float x1816 = x1038[x1809];
float x1817 = x1813 + x1816;
x97[x1807] = x1817;
x1802 += 1;
x1800 += 1;
x1801 += 1;

}
x1795 += 50;
x1793 += 50;

}
for(int x1831=0; x1831 < 20; x1831++) {
int32_t x1832 = x1828;
int32_t x1833 = x1829;
int32_t x1834 = x1830;
int32_t x1835 = x1832;
int32_t x1836 = x1833;
int32_t x1837 = x1834;
for(int x1838=0; x1838 < 50; x1838++) {
int32_t x1839 = x1835;
float x1840 = x975[x1839];
float x1841 = x973[x1839];
int32_t x1842 = x1836;
float x1843 = x977[x1842];
int32_t x1844 = x1837;
float x1845 = x1009[x1844];
float x1846 = x1840 + x1845;
x975[x1839] = x1846;
float x1848 = x979[x1842];
float x1849 = x973[x1839];
float x1850 = x977[x1842];
float x1851 = x1009[x1844];
float x1852 = x1848 + x1851;
x979[x1842] = x1852;
x1837 += 1;
x1835 += 1;
x1836 += 1;

}
x1830 += 50;
x1828 += 50;
x1829 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x979,50,x87,50,1,x805,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x804,50,x979,50,1,x95,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x975,50,x78,50,1,x811,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x810,26,x975,50,1,x86,50);
for(int x1868=0; x1868 < 1000; x1868++) {
float x1869 = x958[x1868];
float x1870 = x959[x1868];
float x1873 = x971[x1868];
float x1871 = 1.0f - x1870;
float x1872 = x1871 * x1870;
float x1874 = x1872 * x1873;
float x1875 = x1869 + x1874;
x958[x1868] = x1875;

}
for(int x1882=0; x1882 < 20; x1882++) {
int32_t x1883 = x1879;
int32_t x1884 = x1880;
int32_t x1885 = x1881;
int32_t x1886 = x1883;
int32_t x1887 = x1884;
int32_t x1888 = x1885;
for(int x1889=0; x1889 < 50; x1889++) {
int32_t x1890 = x1886;
float x1891 = x929[x1890];
float x1892 = x900[x1890];
int32_t x1893 = x1887;
float x1894 = x56[x1893];
int32_t x1895 = x1888;
float x1896 = x958[x1895];
float x1897 = x1891 + x1896;
x929[x1890] = x1897;
float x1899 = x57[x1893];
float x1900 = x900[x1890];
float x1901 = x56[x1893];
float x1902 = x958[x1895];
float x1903 = x1899 + x1902;
x57[x1893] = x1903;
x1888 += 1;
x1886 += 1;
x1887 += 1;

}
x1881 += 50;
x1879 += 50;

}
for(int x1917=0; x1917 < 20; x1917++) {
int32_t x1918 = x1914;
int32_t x1919 = x1915;
int32_t x1920 = x1916;
int32_t x1921 = x1918;
int32_t x1922 = x1919;
int32_t x1923 = x1920;
for(int x1924=0; x1924 < 50; x1924++) {
int32_t x1925 = x1921;
float x1926 = x895[x1925];
float x1927 = x893[x1925];
int32_t x1928 = x1922;
float x1929 = x897[x1928];
int32_t x1930 = x1923;
float x1931 = x929[x1930];
float x1932 = x1926 + x1931;
x895[x1925] = x1932;
float x1934 = x899[x1928];
float x1935 = x893[x1925];
float x1936 = x897[x1928];
float x1937 = x929[x1930];
float x1938 = x1934 + x1937;
x899[x1928] = x1938;
x1923 += 1;
x1921 += 1;
x1922 += 1;

}
x1916 += 50;
x1914 += 50;
x1915 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x899,50,x47,50,1,x805,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x804,50,x899,50,1,x55,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x895,50,x38,50,1,x811,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x810,26,x895,50,1,x46,50);
for(int x1954=0; x1954 < 1000; x1954++) {
float x1955 = x878[x1954];
float x1956 = x879[x1954];
float x1959 = x891[x1954];
float x1957 = 1.0f - x1956;
float x1958 = x1957 * x1956;
float x1960 = x1958 * x1959;
float x1961 = x1955 + x1960;
x878[x1954] = x1961;

}
for(int x1968=0; x1968 < 20; x1968++) {
int32_t x1969 = x1965;
int32_t x1970 = x1966;
int32_t x1971 = x1967;
int32_t x1972 = x1969;
int32_t x1973 = x1970;
int32_t x1974 = x1971;
for(int x1975=0; x1975 < 50; x1975++) {
int32_t x1976 = x1972;
float x1977 = x849[x1976];
float x1978 = x820[x1976];
int32_t x1979 = x1973;
float x1980 = x36[x1979];
int32_t x1981 = x1974;
float x1982 = x878[x1981];
float x1983 = x1977 + x1982;
x849[x1976] = x1983;
float x1985 = x37[x1979];
float x1986 = x820[x1976];
float x1987 = x36[x1979];
float x1988 = x878[x1981];
float x1989 = x1985 + x1988;
x37[x1979] = x1989;
x1974 += 1;
x1972 += 1;
x1973 += 1;

}
x1967 += 50;
x1965 += 50;

}
for(int x2003=0; x2003 < 20; x2003++) {
int32_t x2004 = x2000;
int32_t x2005 = x2001;
int32_t x2006 = x2002;
int32_t x2007 = x2004;
int32_t x2008 = x2005;
int32_t x2009 = x2006;
for(int x2010=0; x2010 < 50; x2010++) {
int32_t x2011 = x2007;
float x2012 = x815[x2011];
float x2013 = x813[x2011];
int32_t x2014 = x2008;
float x2015 = x817[x2014];
int32_t x2016 = x2009;
float x2017 = x849[x2016];
float x2018 = x2012 + x2017;
x815[x2011] = x2018;
float x2020 = x819[x2014];
float x2021 = x813[x2011];
float x2022 = x817[x2014];
float x2023 = x849[x2016];
float x2024 = x2020 + x2023;
x819[x2014] = x2024;
x2009 += 1;
x2007 += 1;
x2008 += 1;

}
x2002 += 50;
x2000 += 50;
x2001 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x819,50,x26,50,1,x805,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x804,50,x819,50,1,x35,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x815,50,x16,50,1,x811,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x810,26,x815,50,1,x25,50);
} else {
float x2041 = 0.0f;
float x2042 = x2041;
float x2043 = x802[0];
float x2044 = x2042 + x2043;
x2041 = x2044;
float x2046 = x2041;
float* x2047 = (float*)myMalloc(1 * sizeof(float));;
x2047[0] = x2046;
float* x2049 = (float*)myMalloc(1 * sizeof(float));;
float x2050 = x2049[0];
x2049[0] = 1.0f;
float x2052 = x2047[0];
x154[0] = x2052;
// += tensor of dim 0
float x2055 = x2049[0];
float x2056 = x803[0];
float x2057 = x2056 + x2055;
x803[0] = x2057;
}
};
x126 += 400;
int32_t x132 = x126;
int32_t x133 = x132 + 400;
int32_t x134 = x133 + 1;
bool x135 = x134 >= x4;
if (x135) {
x126 = 0;
} else {
}
int* x139 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
for(int x142=0; x142 < 400; x142++) {
int32_t x143 = x126;
int32_t x144 = x143 + x142;
int32_t x145 = x7[x144];
x139[x142] = x145;
int32_t x147 = x144 + 1;
int32_t x148 = x7[x147];
x140[x142] = x148;

}
float* x152 = (float*)myMalloc(1 * sizeof(float));;
float* x153 = (float*)myMalloc(1 * sizeof(float));;
for(int x157=0; x157 < 20; x157++) {
int32_t x159 = x157 * 26;
int32_t x160 = x159 * 20;
for(int x158=0; x158 < 20; x158++) {
int32_t x163 = x158 * 20;
int32_t x164 = x163 + x157;
int32_t x165 = x139[x164];
int32_t x161 = x158 * 26;
int32_t x162 = x160 + x161;
int32_t x166 = x162 + x165;
x155[x166] = 1.0f;

}

}
float* x173 = (float*)myMalloc(1 * sizeof(float));;
float* x174 = (float*)myMalloc(1 * sizeof(float));;
float* x175 = (float*)myMalloc(1000 * sizeof(float));;
float* x176 = (float*)myMalloc(1000 * sizeof(float));;
float* x177 = (float*)myMalloc(1000 * sizeof(float));;
float* x178 = (float*)myMalloc(1000 * sizeof(float));;
float** x2712 = (float**)myMalloc(6 * sizeof(float*));;
x2712[0] = x173;
x2712[1] = x174;
x2712[2] = x175;
x2712[3] = x176;
x2712[4] = x177;
x2712[5] = x178;
function<void(int32_t,float**)> x179 = [&](int32_t x180,float** x181) {
float** x183 = x181;
float* x184 = x183[0];
float* x185 = x183[1];
float* x186 = x183[2];
float* x187 = x183[3];
float* x188 = x183[4];
float* x189 = x183[5];
int32_t x182 = x180;
bool x190 = x182 < 20;
if (x190) {
int32_t x191 = x182 * 520;
float* x192 = x155+x191;
float* x193 = x172+x191;
// dot: WrappedArray(20, 26), List(26, 50)
float* x195 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x192,26,x16,50,0,x195,50);
float* x197 = (float*)myMalloc(1000 * sizeof(float));;
// dot: WrappedArray(20, 50), List(50, 50)
float* x199 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x186,50,x26,50,0,x199,50);
float* x201 = (float*)myMalloc(1000 * sizeof(float));;
float* x202 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x203 = 0;
int32_t x204 = 0;
int32_t x205 = 0;
for(int x206=0; x206 < 20; x206++) {
int32_t x207 = x204;
int32_t x208 = x205;
int32_t x209 = x203;
int32_t x210 = x209;
int32_t x211 = x207;
int32_t x212 = x208;
for(int x214=0; x214 < 50; x214++) {
int32_t x215 = x210;
int32_t x216 = x211;
float x217 = x195[x216];
int32_t x218 = x212;
float x219 = x199[x218];
float x220 = x217 + x219;
x202[x215] = x220;
x210 += 1;
x211 += 1;
x212 += 1;

}
x203 += 50;
x204 += 50;
x205 += 50;

}
float* x232 = (float*)myMalloc(1000 * sizeof(float));;
float* x233 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x234 = 0;
int32_t x235 = 0;
int32_t x236 = 0;
for(int x237=0; x237 < 20; x237++) {
int32_t x238 = x235;
int32_t x239 = x236;
int32_t x240 = x234;
int32_t x241 = x240;
int32_t x242 = x238;
int32_t x243 = x239;
for(int x244=0; x244 < 50; x244++) {
int32_t x245 = x241;
int32_t x246 = x242;
float x247 = x202[x246];
int32_t x248 = x243;
float x249 = x36[x248];
float x250 = x247 + x249;
x233[x245] = x250;
x241 += 1;
x242 += 1;
x243 += 1;

}
x234 += 50;
x235 += 50;

}
float* x261 = (float*)myMalloc(1000 * sizeof(float));;
float* x262 = (float*)myMalloc(1000 * sizeof(float));;
for(int x264=0; x264 < 1000; x264++) {
float x265 = x233[x264];
float x266 = -1.0f * x265;
double x267 = (double)x266;
double x268 = exp(x267);
float x269 = (float)x268;
float x270 = x269 + 1.0f;
float x271 = 1.0f / x270;
x262[x264] = x271;

}
float* x275 = (float*)myMalloc(1000 * sizeof(float));;
// dot: WrappedArray(20, 26), List(26, 50)
float* x277 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x192,26,x38,50,0,x277,50);
float* x279 = (float*)myMalloc(1000 * sizeof(float));;
// dot: WrappedArray(20, 50), List(50, 50)
float* x281 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x186,50,x47,50,0,x281,50);
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
float x298 = x277[x297];
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
float x330 = x56[x329];
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
// dot: WrappedArray(20, 26), List(26, 50)
float* x357 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x192,26,x78,50,0,x357,50);
float* x359 = (float*)myMalloc(1000 * sizeof(float));;
// dot: WrappedArray(20, 50), List(50, 50)
float* x361 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x186,50,x87,50,0,x361,50);
float* x363 = (float*)myMalloc(1000 * sizeof(float));;
float* x364 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x365 = 0;
int32_t x366 = 0;
int32_t x367 = 0;
for(int x368=0; x368 < 20; x368++) {
int32_t x369 = x366;
int32_t x370 = x367;
int32_t x371 = x365;
int32_t x372 = x371;
int32_t x373 = x369;
int32_t x374 = x370;
for(int x375=0; x375 < 50; x375++) {
int32_t x376 = x372;
int32_t x377 = x373;
float x378 = x357[x377];
int32_t x379 = x374;
float x380 = x361[x379];
float x381 = x378 + x380;
x364[x376] = x381;
x372 += 1;
x373 += 1;
x374 += 1;

}
x365 += 50;
x366 += 50;
x367 += 50;

}
float* x393 = (float*)myMalloc(1000 * sizeof(float));;
float* x394 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x395 = 0;
int32_t x396 = 0;
int32_t x397 = 0;
for(int x398=0; x398 < 20; x398++) {
int32_t x399 = x396;
int32_t x400 = x397;
int32_t x401 = x395;
int32_t x402 = x401;
int32_t x403 = x399;
int32_t x404 = x400;
for(int x405=0; x405 < 50; x405++) {
int32_t x406 = x402;
int32_t x407 = x403;
float x408 = x364[x407];
int32_t x409 = x404;
float x410 = x96[x409];
float x411 = x408 + x410;
x394[x406] = x411;
x402 += 1;
x403 += 1;
x404 += 1;

}
x395 += 50;
x396 += 50;

}
float* x422 = (float*)myMalloc(1000 * sizeof(float));;
float* x423 = (float*)myMalloc(1000 * sizeof(float));;
for(int x424=0; x424 < 1000; x424++) {
float x425 = x394[x424];
float x426 = -1.0f * x425;
double x427 = (double)x426;
double x428 = exp(x427);
float x429 = (float)x428;
float x430 = x429 + 1.0f;
float x431 = 1.0f / x430;
x423[x424] = x431;

}
float* x435 = (float*)myMalloc(1000 * sizeof(float));;
// dot: WrappedArray(20, 26), List(26, 50)
float* x437 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x192,26,x58,50,0,x437,50);
float* x439 = (float*)myMalloc(1000 * sizeof(float));;
// dot: WrappedArray(20, 50), List(50, 50)
float* x441 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x186,50,x67,50,0,x441,50);
float* x443 = (float*)myMalloc(1000 * sizeof(float));;
float* x444 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x445 = 0;
int32_t x446 = 0;
int32_t x447 = 0;
for(int x448=0; x448 < 20; x448++) {
int32_t x449 = x446;
int32_t x450 = x447;
int32_t x451 = x445;
int32_t x452 = x451;
int32_t x453 = x449;
int32_t x454 = x450;
for(int x455=0; x455 < 50; x455++) {
int32_t x456 = x452;
int32_t x457 = x453;
float x458 = x437[x457];
int32_t x459 = x454;
float x460 = x441[x459];
float x461 = x458 + x460;
x444[x456] = x461;
x452 += 1;
x453 += 1;
x454 += 1;

}
x445 += 50;
x446 += 50;
x447 += 50;

}
float* x473 = (float*)myMalloc(1000 * sizeof(float));;
float* x474 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x475 = 0;
int32_t x476 = 0;
int32_t x477 = 0;
for(int x478=0; x478 < 20; x478++) {
int32_t x479 = x476;
int32_t x480 = x477;
int32_t x481 = x475;
int32_t x482 = x481;
int32_t x483 = x479;
int32_t x484 = x480;
for(int x485=0; x485 < 50; x485++) {
int32_t x486 = x482;
int32_t x487 = x483;
float x488 = x444[x487];
int32_t x489 = x484;
float x490 = x76[x489];
float x491 = x488 + x490;
x474[x486] = x491;
x482 += 1;
x483 += 1;
x484 += 1;

}
x475 += 50;
x476 += 50;

}
float* x502 = (float*)myMalloc(1000 * sizeof(float));;
float* x503 = (float*)myMalloc(1000 * sizeof(float));;
for(int x504=0; x504 < 1000; x504++) {
float x505 = x474[x504];
double x506 = (double)x505;
double x507 = tanh(x506);
float x508 = (float)x507;
x503[x504] = x508;

}
float* x512 = (float*)myMalloc(1000 * sizeof(float));;
float* x513 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x514 = 0;
int32_t x515 = 0;
int32_t x516 = 0;
for(int x517=0; x517 < 20; x517++) {
int32_t x518 = x515;
int32_t x519 = x516;
int32_t x520 = x514;
int32_t x521 = x520;
int32_t x522 = x518;
int32_t x523 = x519;
for(int x524=0; x524 < 50; x524++) {
int32_t x525 = x521;
int32_t x526 = x522;
float x527 = x262[x526];
int32_t x528 = x523;
float x529 = x188[x528];
float x530 = x527 * x529;
x513[x525] = x530;
x521 += 1;
x522 += 1;
x523 += 1;

}
x514 += 50;
x515 += 50;
x516 += 50;

}
float* x542 = (float*)myMalloc(1000 * sizeof(float));;
float* x543 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x544 = 0;
int32_t x545 = 0;
int32_t x546 = 0;
for(int x547=0; x547 < 20; x547++) {
int32_t x548 = x545;
int32_t x549 = x546;
int32_t x550 = x544;
int32_t x551 = x550;
int32_t x552 = x548;
int32_t x553 = x549;
for(int x554=0; x554 < 50; x554++) {
int32_t x555 = x551;
int32_t x556 = x552;
float x557 = x343[x556];
int32_t x558 = x553;
float x559 = x503[x558];
float x560 = x557 * x559;
x543[x555] = x560;
x551 += 1;
x552 += 1;
x553 += 1;

}
x544 += 50;
x545 += 50;
x546 += 50;

}
float* x572 = (float*)myMalloc(1000 * sizeof(float));;
float* x573 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x574 = 0;
int32_t x575 = 0;
int32_t x576 = 0;
for(int x577=0; x577 < 20; x577++) {
int32_t x578 = x575;
int32_t x579 = x576;
int32_t x580 = x574;
int32_t x581 = x580;
int32_t x582 = x578;
int32_t x583 = x579;
for(int x584=0; x584 < 50; x584++) {
int32_t x585 = x581;
int32_t x586 = x582;
float x587 = x513[x586];
int32_t x588 = x583;
float x589 = x543[x588];
float x590 = x587 + x589;
x573[x585] = x590;
x581 += 1;
x582 += 1;
x583 += 1;

}
x574 += 50;
x575 += 50;
x576 += 50;

}
float* x602 = (float*)myMalloc(1000 * sizeof(float));;
float* x603 = (float*)myMalloc(1000 * sizeof(float));;
for(int x604=0; x604 < 1000; x604++) {
float x605 = x573[x604];
double x606 = (double)x605;
double x607 = tanh(x606);
float x608 = (float)x607;
x603[x604] = x608;

}
float* x612 = (float*)myMalloc(1000 * sizeof(float));;
float* x613 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x614 = 0;
int32_t x615 = 0;
int32_t x616 = 0;
for(int x617=0; x617 < 20; x617++) {
int32_t x618 = x615;
int32_t x619 = x616;
int32_t x620 = x614;
int32_t x621 = x620;
int32_t x622 = x618;
int32_t x623 = x619;
for(int x624=0; x624 < 50; x624++) {
int32_t x625 = x621;
int32_t x626 = x622;
float x627 = x423[x626];
int32_t x628 = x623;
float x629 = x603[x628];
float x630 = x627 * x629;
x613[x625] = x630;
x621 += 1;
x622 += 1;
x623 += 1;

}
x614 += 50;
x615 += 50;
x616 += 50;

}
float* x642 = (float*)myMalloc(1000 * sizeof(float));;
// dot: List(20, 50), List(50, 26)
float* x644 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x613,50,x98,26,0,x644,26);
float* x646 = (float*)myMalloc(520 * sizeof(float));;
float* x647 = (float*)myMalloc(520 * sizeof(float));;
int32_t x648 = 0;
int32_t x649 = 0;
int32_t x650 = 0;
for(int x651=0; x651 < 20; x651++) {
int32_t x652 = x649;
int32_t x653 = x650;
int32_t x654 = x648;
int32_t x655 = x654;
int32_t x656 = x652;
int32_t x657 = x653;
for(int x659=0; x659 < 26; x659++) {
int32_t x660 = x655;
int32_t x661 = x656;
float x662 = x644[x661];
int32_t x663 = x657;
float x664 = x107[x663];
float x665 = x662 + x664;
x647[x660] = x665;
x655 += 1;
x656 += 1;
x657 += 1;

}
x648 += 26;
x649 += 26;

}
float* x676 = (float*)myMalloc(520 * sizeof(float));;
int* x677 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x678=0; x678 < 20; x678++) {
int32_t x679 = x678 * 20;
int32_t x680 = x182 + x679;
int32_t x681 = x140[x680];
x677[x678] = x681;

}
float* x685 = (float*)myMalloc(20 * sizeof(float));;
int32_t x686 = 0;
for(int x687=0; x687 < 20; x687++) {
float x688 = -3.4028235E38f;
for(int x689=0; x689 < 26; x689++) {
int32_t x690 = x686;
float x691 = x647[x690];
float x692 = x688;
bool x693 = x691 > x692;
if (x693) {
float x694 = x647[x690];
x688 = x694;
} else {
}
x686 += 1;

}
float x701 = x688;
x685[x687] = x701;

}
float* x705 = (float*)myMalloc(520 * sizeof(float));;
int32_t x706 = 0;
for(int x707=0; x707 < 20; x707++) {
for(int x708=0; x708 < 26; x708++) {
int32_t x709 = x706;
float x710 = x647[x709];
float x711 = x685[x707];
float x712 = x710 - x711;
double x713 = (double)x712;
double x714 = exp(x713);
float x715 = (float)x714;
x705[x709] = x715;
x706 += 1;

}

}
float* x722 = (float*)myMalloc(20 * sizeof(float));;
for(int x723=0; x723 < 20; x723++) {
int32_t x724 = x723;
int32_t x725 = x723 * 26;
int32_t x726 = x725;
for(int x727=0; x727 < 26; x727++) {
int32_t x728 = x724;
float x729 = x722[x728];
int32_t x730 = x726;
float x731 = x705[x730];
float x732 = x729 + x731;
x722[x728] = x732;
x726 += 1;

}

}
x706 = 0;
for(int x740=0; x740 < 20; x740++) {
float x741 = x685[x740];
float x742 = x722[x740];
double x743 = (double)x742;
double x744 = log(x743);
float x745 = (float)x744;
float x746 = x741 + x745;
for(int x747=0; x747 < 26; x747++) {
int32_t x748 = x706;
float x749 = x647[x748];
float x750 = x749 - x746;
x705[x748] = x750;
x706 += 1;

}

}
float* x757 = (float*)myMalloc(520 * sizeof(float));;
float* x758 = (float*)myMalloc(20 * sizeof(float));;
int32_t x759 = 0;
for(int x760=0; x760 < 20; x760++) {
int32_t x761 = x759;
int32_t x762 = x677[x760];
int32_t x763 = x761 + x762;
float x764 = x705[x763];
float x765 = -1.0f * x764;
x758[x760] = x765;
x759 += 26;

}
float* x770 = (float*)myMalloc(20 * sizeof(float));;
float x771 = 0.0f;
for(int x772=0; x772 < 20; x772++) {
float x773 = x771;
float x774 = x758[x772];
float x775 = x773 + x774;
x771 = x775;

}
float x779 = x771;
float* x780 = (float*)myMalloc(1 * sizeof(float));;
x780[0] = x779;
float* x782 = (float*)myMalloc(1 * sizeof(float));;
float* x783 = (float*)myMalloc(1 * sizeof(float));;
int32_t x784 = 0;
int32_t x785 = 0;
int32_t x786 = 0;
int32_t x787 = x784;
int32_t x788 = x785;
float x789 = x184[x788];
int32_t x790 = x786;
float x791 = x780[x790];
float x792 = x789 + x791;
x783[x787] = x792;
x784 += 1;
float* x795 = (float*)myMalloc(1 * sizeof(float));;
float** x2062 = (float**)myMalloc(6 * sizeof(float*));;
x2062[0] = x783;
x2062[1] = x795;
x2062[2] = x613;
x2062[3] = x642;
x2062[4] = x573;
x2062[5] = x602;
int32_t x796 = x182 + 1;
x797(x796,x2062);
int32_t x2071 = 0;
int32_t x2072 = 0;
int32_t x2073 = 0;
int32_t x2074 = x2071;
float x2075 = x185[x2074];
float x2076 = x184[x2074];
int32_t x2077 = x2072;
float x2078 = x780[x2077];
int32_t x2079 = x2073;
float x2080 = x795[x2079];
float x2081 = x2075 + x2080;
x185[x2074] = x2081;
float x2083 = x782[x2077];
float x2084 = x184[x2074];
float x2085 = x780[x2077];
float x2086 = x795[x2079];
float x2087 = x2083 + x2086;
x782[x2077] = x2087;
x2073 += 1;
// += tensor of dim 0
float x2091 = x782[0];
for(int x2092=0; x2092 < 20; x2092++) {
float x2093 = x770[x2092];
float x2094 = x2093 + x2091;
x770[x2092] = x2094;

}
int32_t x2098 = 0;
for(int x2099=0; x2099 < 20; x2099++) {
int32_t x2100 = x2098;
int32_t x2101 = x677[x2099];
int32_t x2102 = x2100 + x2101;
float x2103 = x757[x2102];
float x2104 = x770[x2099];
float x2105 = -1.0f * x2104;
float x2106 = x2103 + x2105;
x757[x2102] = x2106;
x2098 += 26;

}
float* x2111 = (float*)myMalloc(20 * sizeof(float));;
for(int x2112=0; x2112 < 20; x2112++) {
int32_t x2113 = x2112;
int32_t x2114 = x2112 * 26;
int32_t x2115 = x2114;
for(int x2116=0; x2116 < 26; x2116++) {
int32_t x2117 = x2113;
float x2118 = x2111[x2117];
int32_t x2119 = x2115;
float x2120 = x757[x2119];
float x2121 = x2118 + x2120;
x2111[x2117] = x2121;
x2115 += 1;

}

}
int32_t x2128 = 0;
for(int x2129=0; x2129 < 20; x2129++) {
for(int x2130=0; x2130 < 26; x2130++) {
int32_t x2131 = x2128;
float x2132 = x676[x2131];
float x2133 = x757[x2131];
float x2134 = x705[x2131];
float x2138 = x2111[x2129];
double x2135 = (double)x2134;
double x2136 = exp(x2135);
float x2137 = (float)x2136;
float x2139 = x2137 * x2138;
float x2140 = x2133 - x2139;
float x2141 = x2132 + x2140;
x676[x2131] = x2141;
x2128 += 1;

}

}
int32_t x2148 = 0;
int32_t x2149 = 0;
int32_t x2150 = 0;
for(int x2151=0; x2151 < 20; x2151++) {
int32_t x2152 = x2148;
int32_t x2153 = x2149;
int32_t x2154 = x2150;
int32_t x2155 = x2152;
int32_t x2156 = x2153;
int32_t x2157 = x2154;
for(int x2158=0; x2158 < 26; x2158++) {
int32_t x2159 = x2155;
float x2160 = x646[x2159];
float x2161 = x644[x2159];
int32_t x2162 = x2156;
float x2163 = x107[x2162];
int32_t x2164 = x2157;
float x2165 = x676[x2164];
float x2166 = x2160 + x2165;
x646[x2159] = x2166;
float x2168 = x108[x2162];
float x2169 = x644[x2159];
float x2170 = x107[x2162];
float x2171 = x676[x2164];
float x2172 = x2168 + x2171;
x108[x2162] = x2172;
x2157 += 1;
x2155 += 1;
x2156 += 1;

}
x2150 += 26;
x2148 += 26;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x646,26,x98,26,1,x642,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x613,50,x646,26,1,x106,26);
int32_t x2185 = 0;
int32_t x2186 = 0;
int32_t x2187 = 0;
for(int x2188=0; x2188 < 20; x2188++) {
int32_t x2189 = x2185;
int32_t x2190 = x2186;
int32_t x2191 = x2187;
int32_t x2192 = x2189;
int32_t x2193 = x2190;
int32_t x2194 = x2191;
for(int x2195=0; x2195 < 50; x2195++) {
int32_t x2196 = x2192;
float x2197 = x435[x2196];
float x2198 = x423[x2196];
int32_t x2199 = x2193;
float x2200 = x603[x2199];
int32_t x2201 = x2194;
float x2202 = x642[x2201];
float x2203 = x2202 * x2200;
float x2204 = x2197 + x2203;
x435[x2196] = x2204;
float x2206 = x612[x2199];
float x2207 = x423[x2196];
float x2208 = x603[x2199];
float x2209 = x642[x2201];
float x2210 = x2209 * x2207;
float x2211 = x2206 + x2210;
x612[x2199] = x2211;
x2194 += 1;
x2192 += 1;
x2193 += 1;

}
x2187 += 50;
x2185 += 50;
x2186 += 50;

}
for(int x2223=0; x2223 < 1000; x2223++) {
float x2224 = x602[x2223];
float x2225 = x603[x2223];
float x2228 = x612[x2223];
float x2226 = x2225 * x2225;
float x2227 = 1.0f - x2226;
float x2229 = x2227 * x2228;
float x2230 = x2224 + x2229;
x602[x2223] = x2230;

}
int32_t x2234 = 0;
int32_t x2235 = 0;
int32_t x2236 = 0;
for(int x2237=0; x2237 < 20; x2237++) {
int32_t x2238 = x2234;
int32_t x2239 = x2235;
int32_t x2240 = x2236;
int32_t x2241 = x2238;
int32_t x2242 = x2239;
int32_t x2243 = x2240;
for(int x2244=0; x2244 < 50; x2244++) {
int32_t x2245 = x2241;
float x2246 = x542[x2245];
float x2247 = x513[x2245];
int32_t x2248 = x2242;
float x2249 = x543[x2248];
int32_t x2250 = x2243;
float x2251 = x602[x2250];
float x2252 = x2246 + x2251;
x542[x2245] = x2252;
float x2254 = x572[x2248];
float x2255 = x513[x2245];
float x2256 = x543[x2248];
float x2257 = x602[x2250];
float x2258 = x2254 + x2257;
x572[x2248] = x2258;
x2243 += 1;
x2241 += 1;
x2242 += 1;

}
x2236 += 50;
x2234 += 50;
x2235 += 50;

}
int32_t x2270 = 0;
int32_t x2271 = 0;
int32_t x2272 = 0;
for(int x2273=0; x2273 < 20; x2273++) {
int32_t x2274 = x2270;
int32_t x2275 = x2271;
int32_t x2276 = x2272;
int32_t x2277 = x2274;
int32_t x2278 = x2275;
int32_t x2279 = x2276;
for(int x2280=0; x2280 < 50; x2280++) {
int32_t x2281 = x2277;
float x2282 = x355[x2281];
float x2283 = x343[x2281];
int32_t x2284 = x2278;
float x2285 = x503[x2284];
int32_t x2286 = x2279;
float x2287 = x572[x2286];
float x2288 = x2287 * x2285;
float x2289 = x2282 + x2288;
x355[x2281] = x2289;
float x2291 = x512[x2284];
float x2292 = x343[x2281];
float x2293 = x503[x2284];
float x2294 = x572[x2286];
float x2295 = x2294 * x2292;
float x2296 = x2291 + x2295;
x512[x2284] = x2296;
x2279 += 1;
x2277 += 1;
x2278 += 1;

}
x2272 += 50;
x2270 += 50;
x2271 += 50;

}
int32_t x2308 = 0;
int32_t x2309 = 0;
int32_t x2310 = 0;
for(int x2311=0; x2311 < 20; x2311++) {
int32_t x2312 = x2308;
int32_t x2313 = x2309;
int32_t x2314 = x2310;
int32_t x2315 = x2312;
int32_t x2316 = x2313;
int32_t x2317 = x2314;
for(int x2318=0; x2318 < 50; x2318++) {
int32_t x2319 = x2315;
float x2320 = x275[x2319];
float x2321 = x262[x2319];
int32_t x2322 = x2316;
float x2323 = x188[x2322];
int32_t x2324 = x2317;
float x2325 = x542[x2324];
float x2326 = x2325 * x2323;
float x2327 = x2320 + x2326;
x275[x2319] = x2327;
float x2329 = x189[x2322];
float x2330 = x262[x2319];
float x2331 = x188[x2322];
float x2332 = x542[x2324];
float x2333 = x2332 * x2330;
float x2334 = x2329 + x2333;
x189[x2322] = x2334;
x2317 += 1;
x2315 += 1;
x2316 += 1;

}
x2310 += 50;
x2308 += 50;
x2309 += 50;

}
for(int x2346=0; x2346 < 1000; x2346++) {
float x2347 = x502[x2346];
float x2348 = x503[x2346];
float x2351 = x512[x2346];
float x2349 = x2348 * x2348;
float x2350 = 1.0f - x2349;
float x2352 = x2350 * x2351;
float x2353 = x2347 + x2352;
x502[x2346] = x2353;

}
int32_t x2357 = 0;
int32_t x2358 = 0;
int32_t x2359 = 0;
for(int x2360=0; x2360 < 20; x2360++) {
int32_t x2361 = x2357;
int32_t x2362 = x2358;
int32_t x2363 = x2359;
int32_t x2364 = x2361;
int32_t x2365 = x2362;
int32_t x2366 = x2363;
for(int x2367=0; x2367 < 50; x2367++) {
int32_t x2368 = x2364;
float x2369 = x473[x2368];
float x2370 = x444[x2368];
int32_t x2371 = x2365;
float x2372 = x76[x2371];
int32_t x2373 = x2366;
float x2374 = x502[x2373];
float x2375 = x2369 + x2374;
x473[x2368] = x2375;
float x2377 = x77[x2371];
float x2378 = x444[x2368];
float x2379 = x76[x2371];
float x2380 = x502[x2373];
float x2381 = x2377 + x2380;
x77[x2371] = x2381;
x2366 += 1;
x2364 += 1;
x2365 += 1;

}
x2359 += 50;
x2357 += 50;

}
int32_t x2392 = 0;
int32_t x2393 = 0;
int32_t x2394 = 0;
for(int x2395=0; x2395 < 20; x2395++) {
int32_t x2396 = x2392;
int32_t x2397 = x2393;
int32_t x2398 = x2394;
int32_t x2399 = x2396;
int32_t x2400 = x2397;
int32_t x2401 = x2398;
for(int x2402=0; x2402 < 50; x2402++) {
int32_t x2403 = x2399;
float x2404 = x439[x2403];
float x2405 = x437[x2403];
int32_t x2406 = x2400;
float x2407 = x441[x2406];
int32_t x2408 = x2401;
float x2409 = x473[x2408];
float x2410 = x2404 + x2409;
x439[x2403] = x2410;
float x2412 = x443[x2406];
float x2413 = x437[x2403];
float x2414 = x441[x2406];
float x2415 = x473[x2408];
float x2416 = x2412 + x2415;
x443[x2406] = x2416;
x2401 += 1;
x2399 += 1;
x2400 += 1;

}
x2394 += 50;
x2392 += 50;
x2393 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x443,50,x67,50,1,x187,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x186,50,x443,50,1,x75,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x439,50,x58,50,1,x193,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x192,26,x439,50,1,x66,50);
for(int x2432=0; x2432 < 1000; x2432++) {
float x2433 = x422[x2432];
float x2434 = x423[x2432];
float x2437 = x435[x2432];
float x2435 = 1.0f - x2434;
float x2436 = x2435 * x2434;
float x2438 = x2436 * x2437;
float x2439 = x2433 + x2438;
x422[x2432] = x2439;

}
int32_t x2443 = 0;
int32_t x2444 = 0;
int32_t x2445 = 0;
for(int x2446=0; x2446 < 20; x2446++) {
int32_t x2447 = x2443;
int32_t x2448 = x2444;
int32_t x2449 = x2445;
int32_t x2450 = x2447;
int32_t x2451 = x2448;
int32_t x2452 = x2449;
for(int x2453=0; x2453 < 50; x2453++) {
int32_t x2454 = x2450;
float x2455 = x393[x2454];
float x2456 = x364[x2454];
int32_t x2457 = x2451;
float x2458 = x96[x2457];
int32_t x2459 = x2452;
float x2460 = x422[x2459];
float x2461 = x2455 + x2460;
x393[x2454] = x2461;
float x2463 = x97[x2457];
float x2464 = x364[x2454];
float x2465 = x96[x2457];
float x2466 = x422[x2459];
float x2467 = x2463 + x2466;
x97[x2457] = x2467;
x2452 += 1;
x2450 += 1;
x2451 += 1;

}
x2445 += 50;
x2443 += 50;

}
int32_t x2478 = 0;
int32_t x2479 = 0;
int32_t x2480 = 0;
for(int x2481=0; x2481 < 20; x2481++) {
int32_t x2482 = x2478;
int32_t x2483 = x2479;
int32_t x2484 = x2480;
int32_t x2485 = x2482;
int32_t x2486 = x2483;
int32_t x2487 = x2484;
for(int x2488=0; x2488 < 50; x2488++) {
int32_t x2489 = x2485;
float x2490 = x359[x2489];
float x2491 = x357[x2489];
int32_t x2492 = x2486;
float x2493 = x361[x2492];
int32_t x2494 = x2487;
float x2495 = x393[x2494];
float x2496 = x2490 + x2495;
x359[x2489] = x2496;
float x2498 = x363[x2492];
float x2499 = x357[x2489];
float x2500 = x361[x2492];
float x2501 = x393[x2494];
float x2502 = x2498 + x2501;
x363[x2492] = x2502;
x2487 += 1;
x2485 += 1;
x2486 += 1;

}
x2480 += 50;
x2478 += 50;
x2479 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x363,50,x87,50,1,x187,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x186,50,x363,50,1,x95,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x359,50,x78,50,1,x193,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x192,26,x359,50,1,x86,50);
for(int x2518=0; x2518 < 1000; x2518++) {
float x2519 = x342[x2518];
float x2520 = x343[x2518];
float x2523 = x355[x2518];
float x2521 = 1.0f - x2520;
float x2522 = x2521 * x2520;
float x2524 = x2522 * x2523;
float x2525 = x2519 + x2524;
x342[x2518] = x2525;

}
int32_t x2529 = 0;
int32_t x2530 = 0;
int32_t x2531 = 0;
for(int x2532=0; x2532 < 20; x2532++) {
int32_t x2533 = x2529;
int32_t x2534 = x2530;
int32_t x2535 = x2531;
int32_t x2536 = x2533;
int32_t x2537 = x2534;
int32_t x2538 = x2535;
for(int x2539=0; x2539 < 50; x2539++) {
int32_t x2540 = x2536;
float x2541 = x313[x2540];
float x2542 = x284[x2540];
int32_t x2543 = x2537;
float x2544 = x56[x2543];
int32_t x2545 = x2538;
float x2546 = x342[x2545];
float x2547 = x2541 + x2546;
x313[x2540] = x2547;
float x2549 = x57[x2543];
float x2550 = x284[x2540];
float x2551 = x56[x2543];
float x2552 = x342[x2545];
float x2553 = x2549 + x2552;
x57[x2543] = x2553;
x2538 += 1;
x2536 += 1;
x2537 += 1;

}
x2531 += 50;
x2529 += 50;

}
int32_t x2564 = 0;
int32_t x2565 = 0;
int32_t x2566 = 0;
for(int x2567=0; x2567 < 20; x2567++) {
int32_t x2568 = x2564;
int32_t x2569 = x2565;
int32_t x2570 = x2566;
int32_t x2571 = x2568;
int32_t x2572 = x2569;
int32_t x2573 = x2570;
for(int x2574=0; x2574 < 50; x2574++) {
int32_t x2575 = x2571;
float x2576 = x279[x2575];
float x2577 = x277[x2575];
int32_t x2578 = x2572;
float x2579 = x281[x2578];
int32_t x2580 = x2573;
float x2581 = x313[x2580];
float x2582 = x2576 + x2581;
x279[x2575] = x2582;
float x2584 = x283[x2578];
float x2585 = x277[x2575];
float x2586 = x281[x2578];
float x2587 = x313[x2580];
float x2588 = x2584 + x2587;
x283[x2578] = x2588;
x2573 += 1;
x2571 += 1;
x2572 += 1;

}
x2566 += 50;
x2564 += 50;
x2565 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x283,50,x47,50,1,x187,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x186,50,x283,50,1,x55,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x279,50,x38,50,1,x193,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x192,26,x279,50,1,x46,50);
for(int x2604=0; x2604 < 1000; x2604++) {
float x2605 = x261[x2604];
float x2606 = x262[x2604];
float x2609 = x275[x2604];
float x2607 = 1.0f - x2606;
float x2608 = x2607 * x2606;
float x2610 = x2608 * x2609;
float x2611 = x2605 + x2610;
x261[x2604] = x2611;

}
int32_t x2615 = 0;
int32_t x2616 = 0;
int32_t x2617 = 0;
for(int x2618=0; x2618 < 20; x2618++) {
int32_t x2619 = x2615;
int32_t x2620 = x2616;
int32_t x2621 = x2617;
int32_t x2622 = x2619;
int32_t x2623 = x2620;
int32_t x2624 = x2621;
for(int x2625=0; x2625 < 50; x2625++) {
int32_t x2626 = x2622;
float x2627 = x232[x2626];
float x2628 = x202[x2626];
int32_t x2629 = x2623;
float x2630 = x36[x2629];
int32_t x2631 = x2624;
float x2632 = x261[x2631];
float x2633 = x2627 + x2632;
x232[x2626] = x2633;
float x2635 = x37[x2629];
float x2636 = x202[x2626];
float x2637 = x36[x2629];
float x2638 = x261[x2631];
float x2639 = x2635 + x2638;
x37[x2629] = x2639;
x2624 += 1;
x2622 += 1;
x2623 += 1;

}
x2617 += 50;
x2615 += 50;

}
int32_t x2650 = 0;
int32_t x2651 = 0;
int32_t x2652 = 0;
for(int x2653=0; x2653 < 20; x2653++) {
int32_t x2654 = x2650;
int32_t x2655 = x2651;
int32_t x2656 = x2652;
int32_t x2657 = x2654;
int32_t x2658 = x2655;
int32_t x2659 = x2656;
for(int x2660=0; x2660 < 50; x2660++) {
int32_t x2661 = x2657;
float x2662 = x197[x2661];
float x2663 = x195[x2661];
int32_t x2664 = x2658;
float x2665 = x199[x2664];
int32_t x2666 = x2659;
float x2667 = x232[x2666];
float x2668 = x2662 + x2667;
x197[x2661] = x2668;
float x2670 = x201[x2664];
float x2671 = x195[x2661];
float x2672 = x199[x2664];
float x2673 = x232[x2666];
float x2674 = x2670 + x2673;
x201[x2664] = x2674;
x2659 += 1;
x2657 += 1;
x2658 += 1;

}
x2652 += 50;
x2650 += 50;
x2651 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x201,50,x26,50,1,x187,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x186,50,x201,50,1,x35,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x197,50,x16,50,1,x193,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x192,26,x197,50,1,x25,50);
} else {
float x2691 = 0.0f;
float x2692 = x2691;
float x2693 = x184[0];
float x2694 = x2692 + x2693;
x2691 = x2694;
float x2696 = x2691;
float* x2697 = (float*)myMalloc(1 * sizeof(float));;
x2697[0] = x2696;
float* x2699 = (float*)myMalloc(1 * sizeof(float));;
float x2700 = x2699[0];
x2699[0] = 1.0f;
float x2702 = x2697[0];
x154[0] = x2702;
// += tensor of dim 0
float x2705 = x2699[0];
float x2706 = x185[0];
float x2707 = x2706 + x2705;
x185[0] = x2707;
}
};
x179(0,x2712);
float x2721 = x154[0];
int32_t x2722 = x130 % 100;
bool x2723 = x2722 == 0;
if (x2723) {
printf("iter %d, loss %f\n",x130,x2721);
int32_t x2725 = x130 / 100;
double x2726 = (double)x2721;
x124[x2725] = x2726;
} else {
}
for(int x2730=0; x2730 < 1300; x2730++) {
float x2731 = x46[x2730];
float x2732 = x2731;
float x2733 = x2732;
bool x2734 = x2733 > 5.0f;
if (x2734) {
x2732 = 5.0f;
} else {
}
float x2738 = x2732;
bool x2739 = x2738 < -5.0f;
if (x2739) {
x2732 = -5.0f;
} else {
}
float x2743 = x109[x2730];
float x2744 = x2732;
float x2745 = x2744 * x2744;
float x2746 = x2743 + x2745;
x109[x2730] = x2746;
float x2748 = x38[x2730];
float x2750 = x109[x2730];
float x2749 = 0.1f * x2744;
double x2751 = (double)x2750;
double x2752 = x2751 + 9.99999993922529E-9;
double x2753 = sqrt(x2752);
float x2754 = (float)x2753;
float x2755 = x2749 / x2754;
float x2756 = x2748 - x2755;
x38[x2730] = x2756;
x46[x2730] = 0.0f;

}
for(int x2761=0; x2761 < 50; x2761++) {
float x2762 = x57[x2761];
float x2763 = x2762;
float x2764 = x2763;
bool x2765 = x2764 > 5.0f;
if (x2765) {
x2763 = 5.0f;
} else {
}
float x2769 = x2763;
bool x2770 = x2769 < -5.0f;
if (x2770) {
x2763 = -5.0f;
} else {
}
float x2774 = x110[x2761];
float x2775 = x2763;
float x2776 = x2775 * x2775;
float x2777 = x2774 + x2776;
x110[x2761] = x2777;
float x2779 = x56[x2761];
float x2781 = x110[x2761];
float x2780 = 0.1f * x2775;
double x2782 = (double)x2781;
double x2783 = x2782 + 9.99999993922529E-9;
double x2784 = sqrt(x2783);
float x2785 = (float)x2784;
float x2786 = x2780 / x2785;
float x2787 = x2779 - x2786;
x56[x2761] = x2787;
x57[x2761] = 0.0f;

}
for(int x2792=0; x2792 < 2500; x2792++) {
float x2793 = x55[x2792];
float x2794 = x2793;
float x2795 = x2794;
bool x2796 = x2795 > 5.0f;
if (x2796) {
x2794 = 5.0f;
} else {
}
float x2800 = x2794;
bool x2801 = x2800 < -5.0f;
if (x2801) {
x2794 = -5.0f;
} else {
}
float x2805 = x111[x2792];
float x2806 = x2794;
float x2807 = x2806 * x2806;
float x2808 = x2805 + x2807;
x111[x2792] = x2808;
float x2810 = x47[x2792];
float x2812 = x111[x2792];
float x2811 = 0.1f * x2806;
double x2813 = (double)x2812;
double x2814 = x2813 + 9.99999993922529E-9;
double x2815 = sqrt(x2814);
float x2816 = (float)x2815;
float x2817 = x2811 / x2816;
float x2818 = x2810 - x2817;
x47[x2792] = x2818;
x55[x2792] = 0.0f;

}
for(int x2823=0; x2823 < 50; x2823++) {
float x2824 = x37[x2823];
float x2825 = x2824;
float x2826 = x2825;
bool x2827 = x2826 > 5.0f;
if (x2827) {
x2825 = 5.0f;
} else {
}
float x2831 = x2825;
bool x2832 = x2831 < -5.0f;
if (x2832) {
x2825 = -5.0f;
} else {
}
float x2836 = x112[x2823];
float x2837 = x2825;
float x2838 = x2837 * x2837;
float x2839 = x2836 + x2838;
x112[x2823] = x2839;
float x2841 = x36[x2823];
float x2843 = x112[x2823];
float x2842 = 0.1f * x2837;
double x2844 = (double)x2843;
double x2845 = x2844 + 9.99999993922529E-9;
double x2846 = sqrt(x2845);
float x2847 = (float)x2846;
float x2848 = x2842 / x2847;
float x2849 = x2841 - x2848;
x36[x2823] = x2849;
x37[x2823] = 0.0f;

}
for(int x2854=0; x2854 < 2500; x2854++) {
float x2855 = x35[x2854];
float x2856 = x2855;
float x2857 = x2856;
bool x2858 = x2857 > 5.0f;
if (x2858) {
x2856 = 5.0f;
} else {
}
float x2862 = x2856;
bool x2863 = x2862 < -5.0f;
if (x2863) {
x2856 = -5.0f;
} else {
}
float x2867 = x113[x2854];
float x2868 = x2856;
float x2869 = x2868 * x2868;
float x2870 = x2867 + x2869;
x113[x2854] = x2870;
float x2872 = x26[x2854];
float x2874 = x113[x2854];
float x2873 = 0.1f * x2868;
double x2875 = (double)x2874;
double x2876 = x2875 + 9.99999993922529E-9;
double x2877 = sqrt(x2876);
float x2878 = (float)x2877;
float x2879 = x2873 / x2878;
float x2880 = x2872 - x2879;
x26[x2854] = x2880;
x35[x2854] = 0.0f;

}
for(int x2885=0; x2885 < 1300; x2885++) {
float x2886 = x25[x2885];
float x2887 = x2886;
float x2888 = x2887;
bool x2889 = x2888 > 5.0f;
if (x2889) {
x2887 = 5.0f;
} else {
}
float x2893 = x2887;
bool x2894 = x2893 < -5.0f;
if (x2894) {
x2887 = -5.0f;
} else {
}
float x2898 = x114[x2885];
float x2899 = x2887;
float x2900 = x2899 * x2899;
float x2901 = x2898 + x2900;
x114[x2885] = x2901;
float x2903 = x16[x2885];
float x2905 = x114[x2885];
float x2904 = 0.1f * x2899;
double x2906 = (double)x2905;
double x2907 = x2906 + 9.99999993922529E-9;
double x2908 = sqrt(x2907);
float x2909 = (float)x2908;
float x2910 = x2904 / x2909;
float x2911 = x2903 - x2910;
x16[x2885] = x2911;
x25[x2885] = 0.0f;

}
for(int x2916=0; x2916 < 1300; x2916++) {
float x2917 = x66[x2916];
float x2918 = x2917;
float x2919 = x2918;
bool x2920 = x2919 > 5.0f;
if (x2920) {
x2918 = 5.0f;
} else {
}
float x2924 = x2918;
bool x2925 = x2924 < -5.0f;
if (x2925) {
x2918 = -5.0f;
} else {
}
float x2929 = x115[x2916];
float x2930 = x2918;
float x2931 = x2930 * x2930;
float x2932 = x2929 + x2931;
x115[x2916] = x2932;
float x2934 = x58[x2916];
float x2936 = x115[x2916];
float x2935 = 0.1f * x2930;
double x2937 = (double)x2936;
double x2938 = x2937 + 9.99999993922529E-9;
double x2939 = sqrt(x2938);
float x2940 = (float)x2939;
float x2941 = x2935 / x2940;
float x2942 = x2934 - x2941;
x58[x2916] = x2942;
x66[x2916] = 0.0f;

}
for(int x2947=0; x2947 < 50; x2947++) {
float x2948 = x77[x2947];
float x2949 = x2948;
float x2950 = x2949;
bool x2951 = x2950 > 5.0f;
if (x2951) {
x2949 = 5.0f;
} else {
}
float x2955 = x2949;
bool x2956 = x2955 < -5.0f;
if (x2956) {
x2949 = -5.0f;
} else {
}
float x2960 = x116[x2947];
float x2961 = x2949;
float x2962 = x2961 * x2961;
float x2963 = x2960 + x2962;
x116[x2947] = x2963;
float x2965 = x76[x2947];
float x2967 = x116[x2947];
float x2966 = 0.1f * x2961;
double x2968 = (double)x2967;
double x2969 = x2968 + 9.99999993922529E-9;
double x2970 = sqrt(x2969);
float x2971 = (float)x2970;
float x2972 = x2966 / x2971;
float x2973 = x2965 - x2972;
x76[x2947] = x2973;
x77[x2947] = 0.0f;

}
for(int x2978=0; x2978 < 2500; x2978++) {
float x2979 = x75[x2978];
float x2980 = x2979;
float x2981 = x2980;
bool x2982 = x2981 > 5.0f;
if (x2982) {
x2980 = 5.0f;
} else {
}
float x2986 = x2980;
bool x2987 = x2986 < -5.0f;
if (x2987) {
x2980 = -5.0f;
} else {
}
float x2991 = x117[x2978];
float x2992 = x2980;
float x2993 = x2992 * x2992;
float x2994 = x2991 + x2993;
x117[x2978] = x2994;
float x2996 = x67[x2978];
float x2998 = x117[x2978];
float x2997 = 0.1f * x2992;
double x2999 = (double)x2998;
double x3000 = x2999 + 9.99999993922529E-9;
double x3001 = sqrt(x3000);
float x3002 = (float)x3001;
float x3003 = x2997 / x3002;
float x3004 = x2996 - x3003;
x67[x2978] = x3004;
x75[x2978] = 0.0f;

}
for(int x3009=0; x3009 < 26; x3009++) {
float x3010 = x108[x3009];
float x3011 = x3010;
float x3012 = x3011;
bool x3013 = x3012 > 5.0f;
if (x3013) {
x3011 = 5.0f;
} else {
}
float x3017 = x3011;
bool x3018 = x3017 < -5.0f;
if (x3018) {
x3011 = -5.0f;
} else {
}
float x3022 = x118[x3009];
float x3023 = x3011;
float x3024 = x3023 * x3023;
float x3025 = x3022 + x3024;
x118[x3009] = x3025;
float x3027 = x107[x3009];
float x3029 = x118[x3009];
float x3028 = 0.1f * x3023;
double x3030 = (double)x3029;
double x3031 = x3030 + 9.99999993922529E-9;
double x3032 = sqrt(x3031);
float x3033 = (float)x3032;
float x3034 = x3028 / x3033;
float x3035 = x3027 - x3034;
x107[x3009] = x3035;
x108[x3009] = 0.0f;

}
for(int x3040=0; x3040 < 1300; x3040++) {
float x3041 = x106[x3040];
float x3042 = x3041;
float x3043 = x3042;
bool x3044 = x3043 > 5.0f;
if (x3044) {
x3042 = 5.0f;
} else {
}
float x3048 = x3042;
bool x3049 = x3048 < -5.0f;
if (x3049) {
x3042 = -5.0f;
} else {
}
float x3053 = x119[x3040];
float x3054 = x3042;
float x3055 = x3054 * x3054;
float x3056 = x3053 + x3055;
x119[x3040] = x3056;
float x3058 = x98[x3040];
float x3060 = x119[x3040];
float x3059 = 0.1f * x3054;
double x3061 = (double)x3060;
double x3062 = x3061 + 9.99999993922529E-9;
double x3063 = sqrt(x3062);
float x3064 = (float)x3063;
float x3065 = x3059 / x3064;
float x3066 = x3058 - x3065;
x98[x3040] = x3066;
x106[x3040] = 0.0f;

}
for(int x3071=0; x3071 < 2500; x3071++) {
float x3072 = x95[x3071];
float x3073 = x3072;
float x3074 = x3073;
bool x3075 = x3074 > 5.0f;
if (x3075) {
x3073 = 5.0f;
} else {
}
float x3079 = x3073;
bool x3080 = x3079 < -5.0f;
if (x3080) {
x3073 = -5.0f;
} else {
}
float x3084 = x120[x3071];
float x3085 = x3073;
float x3086 = x3085 * x3085;
float x3087 = x3084 + x3086;
x120[x3071] = x3087;
float x3089 = x87[x3071];
float x3091 = x120[x3071];
float x3090 = 0.1f * x3085;
double x3092 = (double)x3091;
double x3093 = x3092 + 9.99999993922529E-9;
double x3094 = sqrt(x3093);
float x3095 = (float)x3094;
float x3096 = x3090 / x3095;
float x3097 = x3089 - x3096;
x87[x3071] = x3097;
x95[x3071] = 0.0f;

}
for(int x3102=0; x3102 < 1300; x3102++) {
float x3103 = x86[x3102];
float x3104 = x3103;
float x3105 = x3104;
bool x3106 = x3105 > 5.0f;
if (x3106) {
x3104 = 5.0f;
} else {
}
float x3110 = x3104;
bool x3111 = x3110 < -5.0f;
if (x3111) {
x3104 = -5.0f;
} else {
}
float x3115 = x121[x3102];
float x3116 = x3104;
float x3117 = x3116 * x3116;
float x3118 = x3115 + x3117;
x121[x3102] = x3118;
float x3120 = x78[x3102];
float x3122 = x121[x3102];
float x3121 = 0.1f * x3116;
double x3123 = (double)x3122;
double x3124 = x3123 + 9.99999993922529E-9;
double x3125 = sqrt(x3124);
float x3126 = (float)x3125;
float x3127 = x3121 / x3126;
float x3128 = x3120 - x3127;
x78[x3102] = x3128;
x86[x3102] = 0.0f;

}
for(int x3133=0; x3133 < 50; x3133++) {
float x3134 = x97[x3133];
float x3135 = x3134;
float x3136 = x3135;
bool x3137 = x3136 > 5.0f;
if (x3137) {
x3135 = 5.0f;
} else {
}
float x3141 = x3135;
bool x3142 = x3141 < -5.0f;
if (x3142) {
x3135 = -5.0f;
} else {
}
float x3146 = x122[x3133];
float x3147 = x3135;
float x3148 = x3147 * x3147;
float x3149 = x3146 + x3148;
x122[x3133] = x3149;
float x3151 = x96[x3133];
float x3153 = x122[x3133];
float x3152 = 0.1f * x3147;
double x3154 = (double)x3153;
double x3155 = x3154 + 9.99999993922529E-9;
double x3156 = sqrt(x3155);
float x3157 = (float)x3156;
float x3158 = x3152 / x3157;
float x3159 = x3151 - x3158;
x96[x3133] = x3159;
x97[x3133] = 0.0f;

}
int64_t x3164 = (long)mallocAddr;
int64_t x3165 = x3164 - x125;
memset((void*)x125, 0, x3165);
mallocAddr = (void*)x125;

}
double x3170 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x3173 = (long)fopen(x0, "w");
fprintf((FILE *)x3173, "unit: %s\n", "100 iteration");
for(int x3176=0; x3176 < 51; x3176++) {
double x3177 = x124[x3176];
fprintf((FILE *)x3173, "%lf\n", x3177);

}
double x3171 = x123 - x2;
double x3172 = x3170 - x123;
fprintf((FILE *)x3173, "run time: %lf %lf\n", x3171, x3172);
fclose((FILE*)x3173);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

