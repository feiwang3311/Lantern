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
for(int x26=0; x26 < 1300; x26++) {
x25[x26] = 0.0f;

}
float* x30 = (float*)myMalloc(2500 * sizeof(float));;
for(int x32=0; x32 < 2500; x32++) {
float x33 = (float)rand()/RAND_MAX;
float x34 = x33 - 0.5f;
float x35 = x34 * 0.14142136f;
x30[x32] = x35;

}
float* x39 = (float*)myMalloc(2500 * sizeof(float));;
for(int x40=0; x40 < 2500; x40++) {
x39[x40] = 0.0f;

}
float* x44 = (float*)myMalloc(50 * sizeof(float));;
for(int x46=0; x46 < 50; x46++) {
x44[x46] = 0.0f;

}
float* x50 = (float*)myMalloc(50 * sizeof(float));;
for(int x51=0; x51 < 50; x51++) {
x50[x51] = 0.0f;

}
float* x55 = (float*)myMalloc(1300 * sizeof(float));;
for(int x56=0; x56 < 1300; x56++) {
float x57 = (float)rand()/RAND_MAX;
float x58 = x57 - 0.5f;
float x59 = x58 * 0.19611613f;
x55[x56] = x59;

}
float* x63 = (float*)myMalloc(1300 * sizeof(float));;
for(int x64=0; x64 < 1300; x64++) {
x63[x64] = 0.0f;

}
float* x68 = (float*)myMalloc(2500 * sizeof(float));;
for(int x69=0; x69 < 2500; x69++) {
float x70 = (float)rand()/RAND_MAX;
float x71 = x70 - 0.5f;
float x72 = x71 * 0.14142136f;
x68[x69] = x72;

}
float* x76 = (float*)myMalloc(2500 * sizeof(float));;
for(int x77=0; x77 < 2500; x77++) {
x76[x77] = 0.0f;

}
float* x81 = (float*)myMalloc(50 * sizeof(float));;
for(int x82=0; x82 < 50; x82++) {
x81[x82] = 0.0f;

}
float* x86 = (float*)myMalloc(50 * sizeof(float));;
for(int x87=0; x87 < 50; x87++) {
x86[x87] = 0.0f;

}
float* x91 = (float*)myMalloc(1300 * sizeof(float));;
for(int x92=0; x92 < 1300; x92++) {
float x93 = (float)rand()/RAND_MAX;
float x94 = x93 - 0.5f;
float x95 = x94 * 0.19611613f;
x91[x92] = x95;

}
float* x99 = (float*)myMalloc(1300 * sizeof(float));;
for(int x100=0; x100 < 1300; x100++) {
x99[x100] = 0.0f;

}
float* x104 = (float*)myMalloc(2500 * sizeof(float));;
for(int x105=0; x105 < 2500; x105++) {
float x106 = (float)rand()/RAND_MAX;
float x107 = x106 - 0.5f;
float x108 = x107 * 0.14142136f;
x104[x105] = x108;

}
float* x112 = (float*)myMalloc(2500 * sizeof(float));;
for(int x113=0; x113 < 2500; x113++) {
x112[x113] = 0.0f;

}
float* x117 = (float*)myMalloc(50 * sizeof(float));;
for(int x118=0; x118 < 50; x118++) {
x117[x118] = 0.0f;

}
float* x122 = (float*)myMalloc(50 * sizeof(float));;
for(int x123=0; x123 < 50; x123++) {
x122[x123] = 0.0f;

}
float* x127 = (float*)myMalloc(1300 * sizeof(float));;
for(int x128=0; x128 < 1300; x128++) {
float x129 = (float)rand()/RAND_MAX;
float x130 = x129 - 0.5f;
float x131 = x130 * 0.19611613f;
x127[x128] = x131;

}
float* x135 = (float*)myMalloc(1300 * sizeof(float));;
for(int x136=0; x136 < 1300; x136++) {
x135[x136] = 0.0f;

}
float* x140 = (float*)myMalloc(2500 * sizeof(float));;
for(int x141=0; x141 < 2500; x141++) {
float x142 = (float)rand()/RAND_MAX;
float x143 = x142 - 0.5f;
float x144 = x143 * 0.14142136f;
x140[x141] = x144;

}
float* x148 = (float*)myMalloc(2500 * sizeof(float));;
for(int x149=0; x149 < 2500; x149++) {
x148[x149] = 0.0f;

}
float* x153 = (float*)myMalloc(50 * sizeof(float));;
for(int x154=0; x154 < 50; x154++) {
x153[x154] = 0.0f;

}
float* x158 = (float*)myMalloc(50 * sizeof(float));;
for(int x159=0; x159 < 50; x159++) {
x158[x159] = 0.0f;

}
float* x163 = (float*)myMalloc(1300 * sizeof(float));;
for(int x164=0; x164 < 1300; x164++) {
float x165 = (float)rand()/RAND_MAX;
float x166 = x165 - 0.5f;
float x167 = x166 * 0.14142136f;
x163[x164] = x167;

}
float* x171 = (float*)myMalloc(1300 * sizeof(float));;
for(int x172=0; x172 < 1300; x172++) {
x171[x172] = 0.0f;

}
float* x176 = (float*)myMalloc(26 * sizeof(float));;
for(int x178=0; x178 < 26; x178++) {
x176[x178] = 0.0f;

}
float* x182 = (float*)myMalloc(26 * sizeof(float));;
for(int x183=0; x183 < 26; x183++) {
x182[x183] = 0.0f;

}
float* x187 = (float*)myMalloc(1300 * sizeof(float));;
for(int x188=0; x188 < 1300; x188++) {
x187[x188] = 0.0f;

}
float* x192 = (float*)myMalloc(50 * sizeof(float));;
for(int x193=0; x193 < 50; x193++) {
x192[x193] = 0.0f;

}
float* x197 = (float*)myMalloc(2500 * sizeof(float));;
for(int x198=0; x198 < 2500; x198++) {
x197[x198] = 0.0f;

}
float* x202 = (float*)myMalloc(50 * sizeof(float));;
for(int x203=0; x203 < 50; x203++) {
x202[x203] = 0.0f;

}
float* x207 = (float*)myMalloc(2500 * sizeof(float));;
for(int x208=0; x208 < 2500; x208++) {
x207[x208] = 0.0f;

}
float* x212 = (float*)myMalloc(1300 * sizeof(float));;
for(int x213=0; x213 < 1300; x213++) {
x212[x213] = 0.0f;

}
float* x217 = (float*)myMalloc(1300 * sizeof(float));;
for(int x218=0; x218 < 1300; x218++) {
x217[x218] = 0.0f;

}
float* x222 = (float*)myMalloc(50 * sizeof(float));;
for(int x223=0; x223 < 50; x223++) {
x222[x223] = 0.0f;

}
float* x227 = (float*)myMalloc(2500 * sizeof(float));;
for(int x228=0; x228 < 2500; x228++) {
x227[x228] = 0.0f;

}
float* x232 = (float*)myMalloc(26 * sizeof(float));;
for(int x233=0; x233 < 26; x233++) {
x232[x233] = 0.0f;

}
float* x237 = (float*)myMalloc(1300 * sizeof(float));;
for(int x238=0; x238 < 1300; x238++) {
x237[x238] = 0.0f;

}
float* x242 = (float*)myMalloc(2500 * sizeof(float));;
for(int x243=0; x243 < 2500; x243++) {
x242[x243] = 0.0f;

}
float* x247 = (float*)myMalloc(1300 * sizeof(float));;
for(int x248=0; x248 < 1300; x248++) {
x247[x248] = 0.0f;

}
float* x252 = (float*)myMalloc(50 * sizeof(float));;
for(int x253=0; x253 < 50; x253++) {
x252[x253] = 0.0f;

}
double x257 = ((double)clock() / CLOCKS_PER_SEC);
double* x258 = (double*)myMalloc(51 * sizeof(double));;
int64_t x259 = (long)mallocAddr;
int32_t x260 = 0;
x260 -= 400;
double x262 = 70.0;
for(int x264=0; x264 < 5001; x264++) {
float* x297 = (float*)myMalloc(1 * sizeof(float));;
float* x302 = (float*)myMalloc(10400 * sizeof(float));;
float* x324 = (float*)myMalloc(10400 * sizeof(float));;
int* x274 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
function<void(int32_t,float**)> x1108 = [&](int32_t x1109,float** x1110) {
float** x1112 = x1110;
float* x1113 = x1112[0];
float* x1114 = x1112[1];
float* x1115 = x1112[2];
float* x1116 = x1112[3];
float* x1117 = x1112[4];
float* x1118 = x1112[5];
int32_t x1111 = x1109;
bool x1119 = x1111 < 20;
if (x1119) {
int32_t x1120 = x1111 * 520;
float* x1121 = x302+x1120;
float* x1122 = x324+x1120;
// dot: WrappedArray(20, 26), List(26, 50)
float* x1124 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x1121,26,x16,50,0,x1124,50);
float* x1126 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1127=0; x1127 < 1000; x1127++) {
x1126[x1127] = 0.0f;

}
// dot: List(20, 50), List(50, 50)
float* x1132 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x1115,50,x30,50,0,x1132,50);
float* x1134 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1135=0; x1135 < 1000; x1135++) {
x1134[x1135] = 0.0f;

}
float* x1139 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1140 = 0;
int32_t x1141 = 0;
int32_t x1142 = 0;
for(int x1143=0; x1143 < 20; x1143++) {
int32_t x1144 = x1141;
int32_t x1145 = x1142;
int32_t x1146 = x1140;
int32_t x1147 = x1146;
int32_t x1148 = x1144;
int32_t x1149 = x1145;
for(int x1150=0; x1150 < 50; x1150++) {
int32_t x1151 = x1147;
int32_t x1152 = x1148;
float x1153 = x1124[x1152];
int32_t x1154 = x1149;
float x1155 = x1132[x1154];
float x1156 = x1153 + x1155;
x1139[x1151] = x1156;
x1147 += 1;
x1148 += 1;
x1149 += 1;

}
x1140 += 50;
x1141 += 50;
x1142 += 50;

}
float* x1168 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1169=0; x1169 < 1000; x1169++) {
x1168[x1169] = 0.0f;

}
float* x1173 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1174 = 0;
int32_t x1175 = 0;
int32_t x1176 = 0;
for(int x1177=0; x1177 < 20; x1177++) {
int32_t x1178 = x1175;
int32_t x1179 = x1176;
int32_t x1180 = x1174;
int32_t x1181 = x1180;
int32_t x1182 = x1178;
int32_t x1183 = x1179;
for(int x1184=0; x1184 < 50; x1184++) {
int32_t x1185 = x1181;
int32_t x1186 = x1182;
float x1187 = x1139[x1186];
int32_t x1188 = x1183;
float x1189 = x44[x1188];
float x1190 = x1187 + x1189;
x1173[x1185] = x1190;
x1181 += 1;
x1182 += 1;
x1183 += 1;

}
x1174 += 50;
x1175 += 50;

}
float* x1201 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1202=0; x1202 < 1000; x1202++) {
x1201[x1202] = 0.0f;

}
float* x1206 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1207=0; x1207 < 1000; x1207++) {
float x1208 = x1173[x1207];
float x1209 = -1.0f * x1208;
double x1210 = (double)x1209;
double x1211 = exp(x1210);
float x1212 = (float)x1211;
float x1213 = x1212 + 1.0f;
float x1214 = 1.0f / x1213;
x1206[x1207] = x1214;

}
float* x1218 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1219=0; x1219 < 1000; x1219++) {
x1218[x1219] = 0.0f;

}
// dot: WrappedArray(20, 26), List(26, 50)
float* x1224 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x1121,26,x55,50,0,x1224,50);
float* x1226 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1227=0; x1227 < 1000; x1227++) {
x1226[x1227] = 0.0f;

}
// dot: List(20, 50), List(50, 50)
float* x1232 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x1115,50,x68,50,0,x1232,50);
float* x1234 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1235=0; x1235 < 1000; x1235++) {
x1234[x1235] = 0.0f;

}
float* x1239 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1240 = 0;
int32_t x1241 = 0;
int32_t x1242 = 0;
for(int x1243=0; x1243 < 20; x1243++) {
int32_t x1244 = x1241;
int32_t x1245 = x1242;
int32_t x1246 = x1240;
int32_t x1247 = x1246;
int32_t x1248 = x1244;
int32_t x1249 = x1245;
for(int x1250=0; x1250 < 50; x1250++) {
int32_t x1251 = x1247;
int32_t x1252 = x1248;
float x1253 = x1224[x1252];
int32_t x1254 = x1249;
float x1255 = x1232[x1254];
float x1256 = x1253 + x1255;
x1239[x1251] = x1256;
x1247 += 1;
x1248 += 1;
x1249 += 1;

}
x1240 += 50;
x1241 += 50;
x1242 += 50;

}
float* x1268 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1269=0; x1269 < 1000; x1269++) {
x1268[x1269] = 0.0f;

}
float* x1273 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1274 = 0;
int32_t x1275 = 0;
int32_t x1276 = 0;
for(int x1277=0; x1277 < 20; x1277++) {
int32_t x1278 = x1275;
int32_t x1279 = x1276;
int32_t x1280 = x1274;
int32_t x1281 = x1280;
int32_t x1282 = x1278;
int32_t x1283 = x1279;
for(int x1284=0; x1284 < 50; x1284++) {
int32_t x1285 = x1281;
int32_t x1286 = x1282;
float x1287 = x1239[x1286];
int32_t x1288 = x1283;
float x1289 = x81[x1288];
float x1290 = x1287 + x1289;
x1273[x1285] = x1290;
x1281 += 1;
x1282 += 1;
x1283 += 1;

}
x1274 += 50;
x1275 += 50;

}
float* x1301 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1302=0; x1302 < 1000; x1302++) {
x1301[x1302] = 0.0f;

}
float* x1306 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1307=0; x1307 < 1000; x1307++) {
float x1308 = x1273[x1307];
float x1309 = -1.0f * x1308;
double x1310 = (double)x1309;
double x1311 = exp(x1310);
float x1312 = (float)x1311;
float x1313 = x1312 + 1.0f;
float x1314 = 1.0f / x1313;
x1306[x1307] = x1314;

}
float* x1318 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1319=0; x1319 < 1000; x1319++) {
x1318[x1319] = 0.0f;

}
// dot: WrappedArray(20, 26), List(26, 50)
float* x1324 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x1121,26,x127,50,0,x1324,50);
float* x1326 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1327=0; x1327 < 1000; x1327++) {
x1326[x1327] = 0.0f;

}
// dot: List(20, 50), List(50, 50)
float* x1332 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x1115,50,x140,50,0,x1332,50);
float* x1334 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1335=0; x1335 < 1000; x1335++) {
x1334[x1335] = 0.0f;

}
float* x1339 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1340 = 0;
int32_t x1341 = 0;
int32_t x1342 = 0;
for(int x1343=0; x1343 < 20; x1343++) {
int32_t x1344 = x1341;
int32_t x1345 = x1342;
int32_t x1346 = x1340;
int32_t x1347 = x1346;
int32_t x1348 = x1344;
int32_t x1349 = x1345;
for(int x1350=0; x1350 < 50; x1350++) {
int32_t x1351 = x1347;
int32_t x1352 = x1348;
float x1353 = x1324[x1352];
int32_t x1354 = x1349;
float x1355 = x1332[x1354];
float x1356 = x1353 + x1355;
x1339[x1351] = x1356;
x1347 += 1;
x1348 += 1;
x1349 += 1;

}
x1340 += 50;
x1341 += 50;
x1342 += 50;

}
float* x1368 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1369=0; x1369 < 1000; x1369++) {
x1368[x1369] = 0.0f;

}
float* x1373 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1374 = 0;
int32_t x1375 = 0;
int32_t x1376 = 0;
for(int x1377=0; x1377 < 20; x1377++) {
int32_t x1378 = x1375;
int32_t x1379 = x1376;
int32_t x1380 = x1374;
int32_t x1381 = x1380;
int32_t x1382 = x1378;
int32_t x1383 = x1379;
for(int x1384=0; x1384 < 50; x1384++) {
int32_t x1385 = x1381;
int32_t x1386 = x1382;
float x1387 = x1339[x1386];
int32_t x1388 = x1383;
float x1389 = x153[x1388];
float x1390 = x1387 + x1389;
x1373[x1385] = x1390;
x1381 += 1;
x1382 += 1;
x1383 += 1;

}
x1374 += 50;
x1375 += 50;

}
float* x1401 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1402=0; x1402 < 1000; x1402++) {
x1401[x1402] = 0.0f;

}
float* x1406 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1407=0; x1407 < 1000; x1407++) {
float x1408 = x1373[x1407];
float x1409 = -1.0f * x1408;
double x1410 = (double)x1409;
double x1411 = exp(x1410);
float x1412 = (float)x1411;
float x1413 = x1412 + 1.0f;
float x1414 = 1.0f / x1413;
x1406[x1407] = x1414;

}
float* x1418 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1419=0; x1419 < 1000; x1419++) {
x1418[x1419] = 0.0f;

}
// dot: WrappedArray(20, 26), List(26, 50)
float* x1424 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x1121,26,x91,50,0,x1424,50);
float* x1426 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1427=0; x1427 < 1000; x1427++) {
x1426[x1427] = 0.0f;

}
// dot: List(20, 50), List(50, 50)
float* x1432 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x1115,50,x104,50,0,x1432,50);
float* x1434 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1435=0; x1435 < 1000; x1435++) {
x1434[x1435] = 0.0f;

}
float* x1439 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1440 = 0;
int32_t x1441 = 0;
int32_t x1442 = 0;
for(int x1443=0; x1443 < 20; x1443++) {
int32_t x1444 = x1441;
int32_t x1445 = x1442;
int32_t x1446 = x1440;
int32_t x1447 = x1446;
int32_t x1448 = x1444;
int32_t x1449 = x1445;
for(int x1450=0; x1450 < 50; x1450++) {
int32_t x1451 = x1447;
int32_t x1452 = x1448;
float x1453 = x1424[x1452];
int32_t x1454 = x1449;
float x1455 = x1432[x1454];
float x1456 = x1453 + x1455;
x1439[x1451] = x1456;
x1447 += 1;
x1448 += 1;
x1449 += 1;

}
x1440 += 50;
x1441 += 50;
x1442 += 50;

}
float* x1468 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1469=0; x1469 < 1000; x1469++) {
x1468[x1469] = 0.0f;

}
float* x1473 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1474 = 0;
int32_t x1475 = 0;
int32_t x1476 = 0;
for(int x1477=0; x1477 < 20; x1477++) {
int32_t x1478 = x1475;
int32_t x1479 = x1476;
int32_t x1480 = x1474;
int32_t x1481 = x1480;
int32_t x1482 = x1478;
int32_t x1483 = x1479;
for(int x1484=0; x1484 < 50; x1484++) {
int32_t x1485 = x1481;
int32_t x1486 = x1482;
float x1487 = x1439[x1486];
int32_t x1488 = x1483;
float x1489 = x117[x1488];
float x1490 = x1487 + x1489;
x1473[x1485] = x1490;
x1481 += 1;
x1482 += 1;
x1483 += 1;

}
x1474 += 50;
x1475 += 50;

}
float* x1501 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1502=0; x1502 < 1000; x1502++) {
x1501[x1502] = 0.0f;

}
float* x1506 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1507=0; x1507 < 1000; x1507++) {
float x1508 = x1473[x1507];
double x1509 = (double)x1508;
double x1510 = tanh(x1509);
float x1511 = (float)x1510;
x1506[x1507] = x1511;

}
float* x1515 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1516=0; x1516 < 1000; x1516++) {
x1515[x1516] = 0.0f;

}
float* x1520 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1521 = 0;
int32_t x1522 = 0;
int32_t x1523 = 0;
for(int x1524=0; x1524 < 20; x1524++) {
int32_t x1525 = x1522;
int32_t x1526 = x1523;
int32_t x1527 = x1521;
int32_t x1528 = x1527;
int32_t x1529 = x1525;
int32_t x1530 = x1526;
for(int x1531=0; x1531 < 50; x1531++) {
int32_t x1532 = x1528;
int32_t x1533 = x1529;
float x1534 = x1206[x1533];
int32_t x1535 = x1530;
float x1536 = x1117[x1535];
float x1537 = x1534 * x1536;
x1520[x1532] = x1537;
x1528 += 1;
x1529 += 1;
x1530 += 1;

}
x1521 += 50;
x1522 += 50;
x1523 += 50;

}
float* x1549 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1550=0; x1550 < 1000; x1550++) {
x1549[x1550] = 0.0f;

}
float* x1554 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1555 = 0;
int32_t x1556 = 0;
int32_t x1557 = 0;
for(int x1558=0; x1558 < 20; x1558++) {
int32_t x1559 = x1556;
int32_t x1560 = x1557;
int32_t x1561 = x1555;
int32_t x1562 = x1561;
int32_t x1563 = x1559;
int32_t x1564 = x1560;
for(int x1565=0; x1565 < 50; x1565++) {
int32_t x1566 = x1562;
int32_t x1567 = x1563;
float x1568 = x1306[x1567];
int32_t x1569 = x1564;
float x1570 = x1506[x1569];
float x1571 = x1568 * x1570;
x1554[x1566] = x1571;
x1562 += 1;
x1563 += 1;
x1564 += 1;

}
x1555 += 50;
x1556 += 50;
x1557 += 50;

}
float* x1583 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1584=0; x1584 < 1000; x1584++) {
x1583[x1584] = 0.0f;

}
float* x1588 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1589 = 0;
int32_t x1590 = 0;
int32_t x1591 = 0;
for(int x1592=0; x1592 < 20; x1592++) {
int32_t x1593 = x1590;
int32_t x1594 = x1591;
int32_t x1595 = x1589;
int32_t x1596 = x1595;
int32_t x1597 = x1593;
int32_t x1598 = x1594;
for(int x1599=0; x1599 < 50; x1599++) {
int32_t x1600 = x1596;
int32_t x1601 = x1597;
float x1602 = x1520[x1601];
int32_t x1603 = x1598;
float x1604 = x1554[x1603];
float x1605 = x1602 + x1604;
x1588[x1600] = x1605;
x1596 += 1;
x1597 += 1;
x1598 += 1;

}
x1589 += 50;
x1590 += 50;
x1591 += 50;

}
float* x1617 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1618=0; x1618 < 1000; x1618++) {
x1617[x1618] = 0.0f;

}
float* x1622 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1623=0; x1623 < 1000; x1623++) {
float x1624 = x1588[x1623];
double x1625 = (double)x1624;
double x1626 = tanh(x1625);
float x1627 = (float)x1626;
x1622[x1623] = x1627;

}
float* x1631 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1632=0; x1632 < 1000; x1632++) {
x1631[x1632] = 0.0f;

}
float* x1636 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1637 = 0;
int32_t x1638 = 0;
int32_t x1639 = 0;
for(int x1640=0; x1640 < 20; x1640++) {
int32_t x1641 = x1638;
int32_t x1642 = x1639;
int32_t x1643 = x1637;
int32_t x1644 = x1643;
int32_t x1645 = x1641;
int32_t x1646 = x1642;
for(int x1647=0; x1647 < 50; x1647++) {
int32_t x1648 = x1644;
int32_t x1649 = x1645;
float x1650 = x1406[x1649];
int32_t x1651 = x1646;
float x1652 = x1622[x1651];
float x1653 = x1650 * x1652;
x1636[x1648] = x1653;
x1644 += 1;
x1645 += 1;
x1646 += 1;

}
x1637 += 50;
x1638 += 50;
x1639 += 50;

}
float* x1665 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1666=0; x1666 < 1000; x1666++) {
x1665[x1666] = 0.0f;

}
// dot: List(20, 50), List(50, 26)
float* x1671 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x1636,50,x163,26,0,x1671,26);
float* x1673 = (float*)myMalloc(520 * sizeof(float));;
for(int x1674=0; x1674 < 520; x1674++) {
x1673[x1674] = 0.0f;

}
float* x1678 = (float*)myMalloc(520 * sizeof(float));;
int32_t x1679 = 0;
int32_t x1680 = 0;
int32_t x1681 = 0;
for(int x1682=0; x1682 < 20; x1682++) {
int32_t x1683 = x1680;
int32_t x1684 = x1681;
int32_t x1685 = x1679;
int32_t x1686 = x1685;
int32_t x1687 = x1683;
int32_t x1688 = x1684;
for(int x1689=0; x1689 < 26; x1689++) {
int32_t x1690 = x1686;
int32_t x1691 = x1687;
float x1692 = x1671[x1691];
int32_t x1693 = x1688;
float x1694 = x176[x1693];
float x1695 = x1692 + x1694;
x1678[x1690] = x1695;
x1686 += 1;
x1687 += 1;
x1688 += 1;

}
x1679 += 26;
x1680 += 26;

}
float* x1706 = (float*)myMalloc(520 * sizeof(float));;
for(int x1707=0; x1707 < 520; x1707++) {
x1706[x1707] = 0.0f;

}
int* x1711 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x1712=0; x1712 < 20; x1712++) {
int32_t x1713 = x1712 * 20;
int32_t x1714 = x1111 + x1713;
int32_t x1715 = x274[x1714];
x1711[x1712] = x1715;

}
float* x1719 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1720 = 0;
for(int x1721=0; x1721 < 20; x1721++) {
float x1722 = -3.4028235E38f;
for(int x1723=0; x1723 < 26; x1723++) {
int32_t x1724 = x1720;
float x1725 = x1678[x1724];
float x1726 = x1722;
bool x1727 = x1725 > x1726;
if (x1727) {
float x1728 = x1678[x1724];
x1722 = x1728;
} else {
}
x1720 += 1;

}
float x1735 = x1722;
x1719[x1721] = x1735;

}
float* x1739 = (float*)myMalloc(520 * sizeof(float));;
for(int x1740=0; x1740 < 520; x1740++) {
x1739[x1740] = 0.0f;

}
int32_t x1744 = 0;
for(int x1745=0; x1745 < 20; x1745++) {
for(int x1746=0; x1746 < 26; x1746++) {
int32_t x1747 = x1744;
float x1748 = x1678[x1747];
float x1749 = x1719[x1745];
float x1750 = x1748 - x1749;
double x1751 = (double)x1750;
double x1752 = exp(x1751);
float x1753 = (float)x1752;
x1739[x1747] = x1753;
x1744 += 1;

}

}
float* x1760 = (float*)myMalloc(20 * sizeof(float));;
for(int x1761=0; x1761 < 20; x1761++) {
x1760[x1761] = 0.0f;

}
for(int x1765=0; x1765 < 20; x1765++) {
int32_t x1766 = x1765;
int32_t x1767 = x1765 * 26;
int32_t x1768 = x1767;
for(int x1769=0; x1769 < 26; x1769++) {
int32_t x1770 = x1766;
float x1771 = x1760[x1770];
int32_t x1772 = x1768;
float x1773 = x1739[x1772];
float x1774 = x1771 + x1773;
x1760[x1770] = x1774;
x1768 += 1;

}

}
x1744 = 0;
for(int x1782=0; x1782 < 20; x1782++) {
float x1783 = x1719[x1782];
float x1784 = x1760[x1782];
double x1785 = (double)x1784;
double x1786 = log(x1785);
float x1787 = (float)x1786;
float x1788 = x1783 + x1787;
for(int x1789=0; x1789 < 26; x1789++) {
int32_t x1790 = x1744;
float x1791 = x1678[x1790];
float x1792 = x1791 - x1788;
x1739[x1790] = x1792;
x1744 += 1;

}

}
float* x1799 = (float*)myMalloc(520 * sizeof(float));;
for(int x1800=0; x1800 < 520; x1800++) {
x1799[x1800] = 0.0f;

}
float* x1804 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1805 = 0;
for(int x1806=0; x1806 < 20; x1806++) {
int32_t x1807 = x1805;
int32_t x1808 = x1711[x1806];
int32_t x1809 = x1807 + x1808;
float x1810 = x1739[x1809];
float x1811 = -1.0f * x1810;
x1804[x1806] = x1811;
x1805 += 26;

}
float* x1816 = (float*)myMalloc(20 * sizeof(float));;
for(int x1817=0; x1817 < 20; x1817++) {
x1816[x1817] = 0.0f;

}
float x1821 = 0.0f;
for(int x1822=0; x1822 < 20; x1822++) {
float x1823 = x1821;
float x1824 = x1804[x1822];
float x1825 = x1823 + x1824;
x1821 = x1825;

}
float x1829 = x1821;
float* x1830 = (float*)myMalloc(1 * sizeof(float));;
x1830[0] = x1829;
float* x1832 = (float*)myMalloc(1 * sizeof(float));;
for(int x1833=0; x1833 < 1; x1833++) {
x1832[x1833] = 0.0f;

}
float* x1837 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1838 = 0;
int32_t x1839 = 0;
int32_t x1840 = 0;
int32_t x1841 = x1838;
int32_t x1842 = x1839;
float x1843 = x1113[x1842];
int32_t x1844 = x1840;
float x1845 = x1830[x1844];
float x1846 = x1843 + x1845;
x1837[x1841] = x1846;
x1838 += 1;
float* x1849 = (float*)myMalloc(1 * sizeof(float));;
for(int x1850=0; x1850 < 1; x1850++) {
x1849[x1850] = 0.0f;

}
float** x1855 = (float**)myMalloc(6 * sizeof(float*));;
x1855[0] = x1837;
x1855[1] = x1849;
x1855[2] = x1636;
x1855[3] = x1665;
x1855[4] = x1588;
x1855[5] = x1617;
int32_t x1864 = 0;
int32_t x1865 = 0;
int32_t x1866 = 0;
int32_t x1867 = x1864;
int32_t x1870 = x1865;
int32_t x1872 = x1866;
x1866 += 1;
int32_t x1891 = 0;
float* x1904 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1925 = 0;
int32_t x1945 = 0;
int32_t x1946 = 0;
int32_t x1947 = 0;
int32_t x1982 = 0;
int32_t x1983 = 0;
int32_t x1984 = 0;
int32_t x2031 = 0;
int32_t x2032 = 0;
int32_t x2033 = 0;
int32_t x2067 = 0;
int32_t x2068 = 0;
int32_t x2069 = 0;
int32_t x2105 = 0;
int32_t x2106 = 0;
int32_t x2107 = 0;
int32_t x2154 = 0;
int32_t x2155 = 0;
int32_t x2156 = 0;
int32_t x2189 = 0;
int32_t x2190 = 0;
int32_t x2191 = 0;
int32_t x2240 = 0;
int32_t x2241 = 0;
int32_t x2242 = 0;
int32_t x2275 = 0;
int32_t x2276 = 0;
int32_t x2277 = 0;
int32_t x2326 = 0;
int32_t x2327 = 0;
int32_t x2328 = 0;
int32_t x2361 = 0;
int32_t x2362 = 0;
int32_t x2363 = 0;
int32_t x2412 = 0;
int32_t x2413 = 0;
int32_t x2414 = 0;
int32_t x2447 = 0;
int32_t x2448 = 0;
int32_t x2449 = 0;
int32_t x1854 = x1111 + 1;
x1108(x1854,x1855);
float x1868 = x1114[x1867];
float x1869 = x1113[x1867];
float x1871 = x1830[x1870];
float x1873 = x1849[x1872];
float x1874 = x1868 + x1873;
x1114[x1867] = x1874;
float x1876 = x1832[x1870];
float x1877 = x1113[x1867];
float x1878 = x1830[x1870];
float x1879 = x1849[x1872];
float x1880 = x1876 + x1879;
x1832[x1870] = x1880;
// += tensor of dim 0
float x1884 = x1832[0];
for(int x1885=0; x1885 < 20; x1885++) {
float x1886 = x1816[x1885];
float x1887 = x1886 + x1884;
x1816[x1885] = x1887;

}
for(int x1892=0; x1892 < 20; x1892++) {
int32_t x1893 = x1891;
int32_t x1894 = x1711[x1892];
int32_t x1895 = x1893 + x1894;
float x1896 = x1799[x1895];
float x1897 = x1816[x1892];
float x1898 = -1.0f * x1897;
float x1899 = x1896 + x1898;
x1799[x1895] = x1899;
x1891 += 26;

}
for(int x1905=0; x1905 < 20; x1905++) {
x1904[x1905] = 0.0f;

}
for(int x1909=0; x1909 < 20; x1909++) {
int32_t x1910 = x1909;
int32_t x1911 = x1909 * 26;
int32_t x1912 = x1911;
for(int x1913=0; x1913 < 26; x1913++) {
int32_t x1914 = x1910;
float x1915 = x1904[x1914];
int32_t x1916 = x1912;
float x1917 = x1799[x1916];
float x1918 = x1915 + x1917;
x1904[x1914] = x1918;
x1912 += 1;

}

}
for(int x1926=0; x1926 < 20; x1926++) {
for(int x1927=0; x1927 < 26; x1927++) {
int32_t x1928 = x1925;
float x1929 = x1706[x1928];
float x1930 = x1799[x1928];
float x1931 = x1739[x1928];
float x1935 = x1904[x1926];
double x1932 = (double)x1931;
double x1933 = exp(x1932);
float x1934 = (float)x1933;
float x1936 = x1934 * x1935;
float x1937 = x1930 - x1936;
float x1938 = x1929 + x1937;
x1706[x1928] = x1938;
x1925 += 1;

}

}
for(int x1948=0; x1948 < 20; x1948++) {
int32_t x1949 = x1945;
int32_t x1950 = x1946;
int32_t x1951 = x1947;
int32_t x1952 = x1949;
int32_t x1953 = x1950;
int32_t x1954 = x1951;
for(int x1955=0; x1955 < 26; x1955++) {
int32_t x1956 = x1952;
float x1957 = x1673[x1956];
float x1958 = x1671[x1956];
int32_t x1959 = x1953;
float x1960 = x176[x1959];
int32_t x1961 = x1954;
float x1962 = x1706[x1961];
float x1963 = x1957 + x1962;
x1673[x1956] = x1963;
float x1965 = x182[x1959];
float x1966 = x1671[x1956];
float x1967 = x176[x1959];
float x1968 = x1706[x1961];
float x1969 = x1965 + x1968;
x182[x1959] = x1969;
x1954 += 1;
x1952 += 1;
x1953 += 1;

}
x1947 += 26;
x1945 += 26;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x1673,26,x163,26,1,x1665,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x1636,50,x1673,26,1,x171,26);
for(int x1985=0; x1985 < 20; x1985++) {
int32_t x1986 = x1982;
int32_t x1987 = x1983;
int32_t x1988 = x1984;
int32_t x1989 = x1986;
int32_t x1990 = x1987;
int32_t x1991 = x1988;
for(int x1992=0; x1992 < 50; x1992++) {
int32_t x1993 = x1989;
float x1994 = x1418[x1993];
float x1995 = x1406[x1993];
int32_t x1996 = x1990;
float x1997 = x1622[x1996];
int32_t x1998 = x1991;
float x1999 = x1665[x1998];
float x2000 = x1999 * x1997;
float x2001 = x1994 + x2000;
x1418[x1993] = x2001;
float x2003 = x1631[x1996];
float x2004 = x1406[x1993];
float x2005 = x1622[x1996];
float x2006 = x1665[x1998];
float x2007 = x2006 * x2004;
float x2008 = x2003 + x2007;
x1631[x1996] = x2008;
x1991 += 1;
x1989 += 1;
x1990 += 1;

}
x1984 += 50;
x1982 += 50;
x1983 += 50;

}
for(int x2020=0; x2020 < 1000; x2020++) {
float x2021 = x1617[x2020];
float x2022 = x1622[x2020];
float x2025 = x1631[x2020];
float x2023 = x2022 * x2022;
float x2024 = 1.0f - x2023;
float x2026 = x2024 * x2025;
float x2027 = x2021 + x2026;
x1617[x2020] = x2027;

}
for(int x2034=0; x2034 < 20; x2034++) {
int32_t x2035 = x2031;
int32_t x2036 = x2032;
int32_t x2037 = x2033;
int32_t x2038 = x2035;
int32_t x2039 = x2036;
int32_t x2040 = x2037;
for(int x2041=0; x2041 < 50; x2041++) {
int32_t x2042 = x2038;
float x2043 = x1549[x2042];
float x2044 = x1520[x2042];
int32_t x2045 = x2039;
float x2046 = x1554[x2045];
int32_t x2047 = x2040;
float x2048 = x1617[x2047];
float x2049 = x2043 + x2048;
x1549[x2042] = x2049;
float x2051 = x1583[x2045];
float x2052 = x1520[x2042];
float x2053 = x1554[x2045];
float x2054 = x1617[x2047];
float x2055 = x2051 + x2054;
x1583[x2045] = x2055;
x2040 += 1;
x2038 += 1;
x2039 += 1;

}
x2033 += 50;
x2031 += 50;
x2032 += 50;

}
for(int x2070=0; x2070 < 20; x2070++) {
int32_t x2071 = x2067;
int32_t x2072 = x2068;
int32_t x2073 = x2069;
int32_t x2074 = x2071;
int32_t x2075 = x2072;
int32_t x2076 = x2073;
for(int x2077=0; x2077 < 50; x2077++) {
int32_t x2078 = x2074;
float x2079 = x1318[x2078];
float x2080 = x1306[x2078];
int32_t x2081 = x2075;
float x2082 = x1506[x2081];
int32_t x2083 = x2076;
float x2084 = x1583[x2083];
float x2085 = x2084 * x2082;
float x2086 = x2079 + x2085;
x1318[x2078] = x2086;
float x2088 = x1515[x2081];
float x2089 = x1306[x2078];
float x2090 = x1506[x2081];
float x2091 = x1583[x2083];
float x2092 = x2091 * x2089;
float x2093 = x2088 + x2092;
x1515[x2081] = x2093;
x2076 += 1;
x2074 += 1;
x2075 += 1;

}
x2069 += 50;
x2067 += 50;
x2068 += 50;

}
for(int x2108=0; x2108 < 20; x2108++) {
int32_t x2109 = x2105;
int32_t x2110 = x2106;
int32_t x2111 = x2107;
int32_t x2112 = x2109;
int32_t x2113 = x2110;
int32_t x2114 = x2111;
for(int x2115=0; x2115 < 50; x2115++) {
int32_t x2116 = x2112;
float x2117 = x1218[x2116];
float x2118 = x1206[x2116];
int32_t x2119 = x2113;
float x2120 = x1117[x2119];
int32_t x2121 = x2114;
float x2122 = x1549[x2121];
float x2123 = x2122 * x2120;
float x2124 = x2117 + x2123;
x1218[x2116] = x2124;
float x2126 = x1118[x2119];
float x2127 = x1206[x2116];
float x2128 = x1117[x2119];
float x2129 = x1549[x2121];
float x2130 = x2129 * x2127;
float x2131 = x2126 + x2130;
x1118[x2119] = x2131;
x2114 += 1;
x2112 += 1;
x2113 += 1;

}
x2107 += 50;
x2105 += 50;
x2106 += 50;

}
for(int x2143=0; x2143 < 1000; x2143++) {
float x2144 = x1501[x2143];
float x2145 = x1506[x2143];
float x2148 = x1515[x2143];
float x2146 = x2145 * x2145;
float x2147 = 1.0f - x2146;
float x2149 = x2147 * x2148;
float x2150 = x2144 + x2149;
x1501[x2143] = x2150;

}
for(int x2157=0; x2157 < 20; x2157++) {
int32_t x2158 = x2154;
int32_t x2159 = x2155;
int32_t x2160 = x2156;
int32_t x2161 = x2158;
int32_t x2162 = x2159;
int32_t x2163 = x2160;
for(int x2164=0; x2164 < 50; x2164++) {
int32_t x2165 = x2161;
float x2166 = x1468[x2165];
float x2167 = x1439[x2165];
int32_t x2168 = x2162;
float x2169 = x117[x2168];
int32_t x2170 = x2163;
float x2171 = x1501[x2170];
float x2172 = x2166 + x2171;
x1468[x2165] = x2172;
float x2174 = x122[x2168];
float x2175 = x1439[x2165];
float x2176 = x117[x2168];
float x2177 = x1501[x2170];
float x2178 = x2174 + x2177;
x122[x2168] = x2178;
x2163 += 1;
x2161 += 1;
x2162 += 1;

}
x2156 += 50;
x2154 += 50;

}
for(int x2192=0; x2192 < 20; x2192++) {
int32_t x2193 = x2189;
int32_t x2194 = x2190;
int32_t x2195 = x2191;
int32_t x2196 = x2193;
int32_t x2197 = x2194;
int32_t x2198 = x2195;
for(int x2199=0; x2199 < 50; x2199++) {
int32_t x2200 = x2196;
float x2201 = x1426[x2200];
float x2202 = x1424[x2200];
int32_t x2203 = x2197;
float x2204 = x1432[x2203];
int32_t x2205 = x2198;
float x2206 = x1468[x2205];
float x2207 = x2201 + x2206;
x1426[x2200] = x2207;
float x2209 = x1434[x2203];
float x2210 = x1424[x2200];
float x2211 = x1432[x2203];
float x2212 = x1468[x2205];
float x2213 = x2209 + x2212;
x1434[x2203] = x2213;
x2198 += 1;
x2196 += 1;
x2197 += 1;

}
x2191 += 50;
x2189 += 50;
x2190 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x1434,50,x104,50,1,x1116,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x1115,50,x1434,50,1,x112,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x1426,50,x91,50,1,x1122,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x1121,26,x1426,50,1,x99,50);
for(int x2229=0; x2229 < 1000; x2229++) {
float x2230 = x1401[x2229];
float x2231 = x1406[x2229];
float x2234 = x1418[x2229];
float x2232 = 1.0f - x2231;
float x2233 = x2232 * x2231;
float x2235 = x2233 * x2234;
float x2236 = x2230 + x2235;
x1401[x2229] = x2236;

}
for(int x2243=0; x2243 < 20; x2243++) {
int32_t x2244 = x2240;
int32_t x2245 = x2241;
int32_t x2246 = x2242;
int32_t x2247 = x2244;
int32_t x2248 = x2245;
int32_t x2249 = x2246;
for(int x2250=0; x2250 < 50; x2250++) {
int32_t x2251 = x2247;
float x2252 = x1368[x2251];
float x2253 = x1339[x2251];
int32_t x2254 = x2248;
float x2255 = x153[x2254];
int32_t x2256 = x2249;
float x2257 = x1401[x2256];
float x2258 = x2252 + x2257;
x1368[x2251] = x2258;
float x2260 = x158[x2254];
float x2261 = x1339[x2251];
float x2262 = x153[x2254];
float x2263 = x1401[x2256];
float x2264 = x2260 + x2263;
x158[x2254] = x2264;
x2249 += 1;
x2247 += 1;
x2248 += 1;

}
x2242 += 50;
x2240 += 50;

}
for(int x2278=0; x2278 < 20; x2278++) {
int32_t x2279 = x2275;
int32_t x2280 = x2276;
int32_t x2281 = x2277;
int32_t x2282 = x2279;
int32_t x2283 = x2280;
int32_t x2284 = x2281;
for(int x2285=0; x2285 < 50; x2285++) {
int32_t x2286 = x2282;
float x2287 = x1326[x2286];
float x2288 = x1324[x2286];
int32_t x2289 = x2283;
float x2290 = x1332[x2289];
int32_t x2291 = x2284;
float x2292 = x1368[x2291];
float x2293 = x2287 + x2292;
x1326[x2286] = x2293;
float x2295 = x1334[x2289];
float x2296 = x1324[x2286];
float x2297 = x1332[x2289];
float x2298 = x1368[x2291];
float x2299 = x2295 + x2298;
x1334[x2289] = x2299;
x2284 += 1;
x2282 += 1;
x2283 += 1;

}
x2277 += 50;
x2275 += 50;
x2276 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x1334,50,x140,50,1,x1116,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x1115,50,x1334,50,1,x148,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x1326,50,x127,50,1,x1122,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x1121,26,x1326,50,1,x135,50);
for(int x2315=0; x2315 < 1000; x2315++) {
float x2316 = x1301[x2315];
float x2317 = x1306[x2315];
float x2320 = x1318[x2315];
float x2318 = 1.0f - x2317;
float x2319 = x2318 * x2317;
float x2321 = x2319 * x2320;
float x2322 = x2316 + x2321;
x1301[x2315] = x2322;

}
for(int x2329=0; x2329 < 20; x2329++) {
int32_t x2330 = x2326;
int32_t x2331 = x2327;
int32_t x2332 = x2328;
int32_t x2333 = x2330;
int32_t x2334 = x2331;
int32_t x2335 = x2332;
for(int x2336=0; x2336 < 50; x2336++) {
int32_t x2337 = x2333;
float x2338 = x1268[x2337];
float x2339 = x1239[x2337];
int32_t x2340 = x2334;
float x2341 = x81[x2340];
int32_t x2342 = x2335;
float x2343 = x1301[x2342];
float x2344 = x2338 + x2343;
x1268[x2337] = x2344;
float x2346 = x86[x2340];
float x2347 = x1239[x2337];
float x2348 = x81[x2340];
float x2349 = x1301[x2342];
float x2350 = x2346 + x2349;
x86[x2340] = x2350;
x2335 += 1;
x2333 += 1;
x2334 += 1;

}
x2328 += 50;
x2326 += 50;

}
for(int x2364=0; x2364 < 20; x2364++) {
int32_t x2365 = x2361;
int32_t x2366 = x2362;
int32_t x2367 = x2363;
int32_t x2368 = x2365;
int32_t x2369 = x2366;
int32_t x2370 = x2367;
for(int x2371=0; x2371 < 50; x2371++) {
int32_t x2372 = x2368;
float x2373 = x1226[x2372];
float x2374 = x1224[x2372];
int32_t x2375 = x2369;
float x2376 = x1232[x2375];
int32_t x2377 = x2370;
float x2378 = x1268[x2377];
float x2379 = x2373 + x2378;
x1226[x2372] = x2379;
float x2381 = x1234[x2375];
float x2382 = x1224[x2372];
float x2383 = x1232[x2375];
float x2384 = x1268[x2377];
float x2385 = x2381 + x2384;
x1234[x2375] = x2385;
x2370 += 1;
x2368 += 1;
x2369 += 1;

}
x2363 += 50;
x2361 += 50;
x2362 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x1234,50,x68,50,1,x1116,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x1115,50,x1234,50,1,x76,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x1226,50,x55,50,1,x1122,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x1121,26,x1226,50,1,x63,50);
for(int x2401=0; x2401 < 1000; x2401++) {
float x2402 = x1201[x2401];
float x2403 = x1206[x2401];
float x2406 = x1218[x2401];
float x2404 = 1.0f - x2403;
float x2405 = x2404 * x2403;
float x2407 = x2405 * x2406;
float x2408 = x2402 + x2407;
x1201[x2401] = x2408;

}
for(int x2415=0; x2415 < 20; x2415++) {
int32_t x2416 = x2412;
int32_t x2417 = x2413;
int32_t x2418 = x2414;
int32_t x2419 = x2416;
int32_t x2420 = x2417;
int32_t x2421 = x2418;
for(int x2422=0; x2422 < 50; x2422++) {
int32_t x2423 = x2419;
float x2424 = x1168[x2423];
float x2425 = x1139[x2423];
int32_t x2426 = x2420;
float x2427 = x44[x2426];
int32_t x2428 = x2421;
float x2429 = x1201[x2428];
float x2430 = x2424 + x2429;
x1168[x2423] = x2430;
float x2432 = x50[x2426];
float x2433 = x1139[x2423];
float x2434 = x44[x2426];
float x2435 = x1201[x2428];
float x2436 = x2432 + x2435;
x50[x2426] = x2436;
x2421 += 1;
x2419 += 1;
x2420 += 1;

}
x2414 += 50;
x2412 += 50;

}
for(int x2450=0; x2450 < 20; x2450++) {
int32_t x2451 = x2447;
int32_t x2452 = x2448;
int32_t x2453 = x2449;
int32_t x2454 = x2451;
int32_t x2455 = x2452;
int32_t x2456 = x2453;
for(int x2457=0; x2457 < 50; x2457++) {
int32_t x2458 = x2454;
float x2459 = x1126[x2458];
float x2460 = x1124[x2458];
int32_t x2461 = x2455;
float x2462 = x1132[x2461];
int32_t x2463 = x2456;
float x2464 = x1168[x2463];
float x2465 = x2459 + x2464;
x1126[x2458] = x2465;
float x2467 = x1134[x2461];
float x2468 = x1124[x2458];
float x2469 = x1132[x2461];
float x2470 = x1168[x2463];
float x2471 = x2467 + x2470;
x1134[x2461] = x2471;
x2456 += 1;
x2454 += 1;
x2455 += 1;

}
x2449 += 50;
x2447 += 50;
x2448 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x1134,50,x30,50,1,x1116,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x1115,50,x1134,50,1,x39,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x1126,50,x16,50,1,x1122,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x1121,26,x1126,50,1,x25,50);
} else {
float x2488 = 0.0f;
float x2489 = x2488;
float x2490 = x1113[0];
float x2491 = x2489 + x2490;
x2488 = x2491;
float x2493 = x2488;
float* x2494 = (float*)myMalloc(1 * sizeof(float));;
x2494[0] = x2493;
float* x2496 = (float*)myMalloc(1 * sizeof(float));;
for(int x2497=0; x2497 < 1; x2497++) {
x2496[x2497] = 0.0f;

}
float x2501 = x2496[0];
x2496[0] = 1.0f;
float x2503 = x2494[0];
x297[0] = x2503;
// += tensor of dim 0
float x2506 = x2496[0];
float x2507 = x1114[0];
float x2508 = x2507 + x2506;
x1114[0] = x2508;
}
};
x260 += 400;
int32_t x266 = x260;
int32_t x267 = x266 + 400;
int32_t x268 = x267 + 1;
bool x269 = x268 >= x4;
if (x269) {
x260 = 0;
} else {
}
int* x273 = (int32_t*)myMalloc(400 * sizeof(int32_t));;
for(int x276=0; x276 < 400; x276++) {
int32_t x277 = x260;
int32_t x278 = x277 + x276;
int32_t x279 = x7[x278];
x273[x276] = x279;
int32_t x281 = x278 + 1;
int32_t x282 = x7[x281];
x274[x276] = x282;

}
float* x286 = (float*)myMalloc(1 * sizeof(float));;
for(int x288=0; x288 < 1; x288++) {
x286[x288] = 0.0f;

}
float* x292 = (float*)myMalloc(1 * sizeof(float));;
for(int x293=0; x293 < 1; x293++) {
x292[x293] = 0.0f;

}
for(int x298=0; x298 < 1; x298++) {
x297[x298] = 0.0f;

}
for(int x304=0; x304 < 10400; x304++) {
x302[x304] = 0.0f;

}
for(int x309=0; x309 < 20; x309++) {
int32_t x311 = x309 * 26;
int32_t x312 = x311 * 20;
for(int x310=0; x310 < 20; x310++) {
int32_t x315 = x310 * 20;
int32_t x316 = x315 + x309;
int32_t x317 = x273[x316];
int32_t x313 = x310 * 26;
int32_t x314 = x312 + x313;
int32_t x318 = x314 + x317;
x302[x318] = 1.0f;

}

}
for(int x325=0; x325 < 10400; x325++) {
x324[x325] = 0.0f;

}
float* x329 = (float*)myMalloc(1 * sizeof(float));;
for(int x330=0; x330 < 1; x330++) {
x329[x330] = 0.0f;

}
float* x334 = (float*)myMalloc(1 * sizeof(float));;
for(int x335=0; x335 < 1; x335++) {
x334[x335] = 0.0f;

}
float* x339 = (float*)myMalloc(1000 * sizeof(float));;
for(int x341=0; x341 < 1000; x341++) {
x339[x341] = 0.0f;

}
float* x345 = (float*)myMalloc(1000 * sizeof(float));;
for(int x346=0; x346 < 1000; x346++) {
x345[x346] = 0.0f;

}
float* x350 = (float*)myMalloc(1000 * sizeof(float));;
for(int x351=0; x351 < 1000; x351++) {
x350[x351] = 0.0f;

}
float* x355 = (float*)myMalloc(1000 * sizeof(float));;
for(int x356=0; x356 < 1000; x356++) {
x355[x356] = 0.0f;

}
float** x3171 = (float**)myMalloc(6 * sizeof(float*));;
x3171[0] = x329;
x3171[1] = x334;
x3171[2] = x339;
x3171[3] = x345;
x3171[4] = x350;
x3171[5] = x355;
function<void(int32_t,float**)> x360 = [&](int32_t x361,float** x362) {
float** x364 = x362;
float* x365 = x364[0];
float* x366 = x364[1];
float* x367 = x364[2];
float* x368 = x364[3];
float* x369 = x364[4];
float* x370 = x364[5];
int32_t x363 = x361;
bool x371 = x363 < 20;
if (x371) {
int32_t x372 = x363 * 520;
float* x373 = x302+x372;
float* x374 = x324+x372;
// dot: WrappedArray(20, 26), List(26, 50)
float* x376 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x373,26,x16,50,0,x376,50);
float* x378 = (float*)myMalloc(1000 * sizeof(float));;
for(int x379=0; x379 < 1000; x379++) {
x378[x379] = 0.0f;

}
// dot: WrappedArray(20, 50), List(50, 50)
float* x384 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x367,50,x30,50,0,x384,50);
float* x386 = (float*)myMalloc(1000 * sizeof(float));;
for(int x387=0; x387 < 1000; x387++) {
x386[x387] = 0.0f;

}
float* x391 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x392 = 0;
int32_t x393 = 0;
int32_t x394 = 0;
for(int x395=0; x395 < 20; x395++) {
int32_t x396 = x393;
int32_t x397 = x394;
int32_t x398 = x392;
int32_t x399 = x398;
int32_t x400 = x396;
int32_t x401 = x397;
for(int x402=0; x402 < 50; x402++) {
int32_t x403 = x399;
int32_t x404 = x400;
float x405 = x376[x404];
int32_t x406 = x401;
float x407 = x384[x406];
float x408 = x405 + x407;
x391[x403] = x408;
x399 += 1;
x400 += 1;
x401 += 1;

}
x392 += 50;
x393 += 50;
x394 += 50;

}
float* x420 = (float*)myMalloc(1000 * sizeof(float));;
for(int x421=0; x421 < 1000; x421++) {
x420[x421] = 0.0f;

}
float* x425 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x426 = 0;
int32_t x427 = 0;
int32_t x428 = 0;
for(int x429=0; x429 < 20; x429++) {
int32_t x430 = x427;
int32_t x431 = x428;
int32_t x432 = x426;
int32_t x433 = x432;
int32_t x434 = x430;
int32_t x435 = x431;
for(int x436=0; x436 < 50; x436++) {
int32_t x437 = x433;
int32_t x438 = x434;
float x439 = x391[x438];
int32_t x440 = x435;
float x441 = x44[x440];
float x442 = x439 + x441;
x425[x437] = x442;
x433 += 1;
x434 += 1;
x435 += 1;

}
x426 += 50;
x427 += 50;

}
float* x453 = (float*)myMalloc(1000 * sizeof(float));;
for(int x454=0; x454 < 1000; x454++) {
x453[x454] = 0.0f;

}
float* x458 = (float*)myMalloc(1000 * sizeof(float));;
for(int x459=0; x459 < 1000; x459++) {
float x460 = x425[x459];
float x461 = -1.0f * x460;
double x462 = (double)x461;
double x463 = exp(x462);
float x464 = (float)x463;
float x465 = x464 + 1.0f;
float x466 = 1.0f / x465;
x458[x459] = x466;

}
float* x470 = (float*)myMalloc(1000 * sizeof(float));;
for(int x471=0; x471 < 1000; x471++) {
x470[x471] = 0.0f;

}
// dot: WrappedArray(20, 26), List(26, 50)
float* x476 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x373,26,x55,50,0,x476,50);
float* x478 = (float*)myMalloc(1000 * sizeof(float));;
for(int x479=0; x479 < 1000; x479++) {
x478[x479] = 0.0f;

}
// dot: WrappedArray(20, 50), List(50, 50)
float* x484 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x367,50,x68,50,0,x484,50);
float* x486 = (float*)myMalloc(1000 * sizeof(float));;
for(int x487=0; x487 < 1000; x487++) {
x486[x487] = 0.0f;

}
float* x491 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x492 = 0;
int32_t x493 = 0;
int32_t x494 = 0;
for(int x495=0; x495 < 20; x495++) {
int32_t x496 = x493;
int32_t x497 = x494;
int32_t x498 = x492;
int32_t x499 = x498;
int32_t x500 = x496;
int32_t x501 = x497;
for(int x502=0; x502 < 50; x502++) {
int32_t x503 = x499;
int32_t x504 = x500;
float x505 = x476[x504];
int32_t x506 = x501;
float x507 = x484[x506];
float x508 = x505 + x507;
x491[x503] = x508;
x499 += 1;
x500 += 1;
x501 += 1;

}
x492 += 50;
x493 += 50;
x494 += 50;

}
float* x520 = (float*)myMalloc(1000 * sizeof(float));;
for(int x521=0; x521 < 1000; x521++) {
x520[x521] = 0.0f;

}
float* x525 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x526 = 0;
int32_t x527 = 0;
int32_t x528 = 0;
for(int x529=0; x529 < 20; x529++) {
int32_t x530 = x527;
int32_t x531 = x528;
int32_t x532 = x526;
int32_t x533 = x532;
int32_t x534 = x530;
int32_t x535 = x531;
for(int x536=0; x536 < 50; x536++) {
int32_t x537 = x533;
int32_t x538 = x534;
float x539 = x491[x538];
int32_t x540 = x535;
float x541 = x81[x540];
float x542 = x539 + x541;
x525[x537] = x542;
x533 += 1;
x534 += 1;
x535 += 1;

}
x526 += 50;
x527 += 50;

}
float* x553 = (float*)myMalloc(1000 * sizeof(float));;
for(int x554=0; x554 < 1000; x554++) {
x553[x554] = 0.0f;

}
float* x558 = (float*)myMalloc(1000 * sizeof(float));;
for(int x559=0; x559 < 1000; x559++) {
float x560 = x525[x559];
float x561 = -1.0f * x560;
double x562 = (double)x561;
double x563 = exp(x562);
float x564 = (float)x563;
float x565 = x564 + 1.0f;
float x566 = 1.0f / x565;
x558[x559] = x566;

}
float* x570 = (float*)myMalloc(1000 * sizeof(float));;
for(int x571=0; x571 < 1000; x571++) {
x570[x571] = 0.0f;

}
// dot: WrappedArray(20, 26), List(26, 50)
float* x576 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x373,26,x127,50,0,x576,50);
float* x578 = (float*)myMalloc(1000 * sizeof(float));;
for(int x579=0; x579 < 1000; x579++) {
x578[x579] = 0.0f;

}
// dot: WrappedArray(20, 50), List(50, 50)
float* x584 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x367,50,x140,50,0,x584,50);
float* x586 = (float*)myMalloc(1000 * sizeof(float));;
for(int x587=0; x587 < 1000; x587++) {
x586[x587] = 0.0f;

}
float* x591 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x592 = 0;
int32_t x593 = 0;
int32_t x594 = 0;
for(int x595=0; x595 < 20; x595++) {
int32_t x596 = x593;
int32_t x597 = x594;
int32_t x598 = x592;
int32_t x599 = x598;
int32_t x600 = x596;
int32_t x601 = x597;
for(int x602=0; x602 < 50; x602++) {
int32_t x603 = x599;
int32_t x604 = x600;
float x605 = x576[x604];
int32_t x606 = x601;
float x607 = x584[x606];
float x608 = x605 + x607;
x591[x603] = x608;
x599 += 1;
x600 += 1;
x601 += 1;

}
x592 += 50;
x593 += 50;
x594 += 50;

}
float* x620 = (float*)myMalloc(1000 * sizeof(float));;
for(int x621=0; x621 < 1000; x621++) {
x620[x621] = 0.0f;

}
float* x625 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x626 = 0;
int32_t x627 = 0;
int32_t x628 = 0;
for(int x629=0; x629 < 20; x629++) {
int32_t x630 = x627;
int32_t x631 = x628;
int32_t x632 = x626;
int32_t x633 = x632;
int32_t x634 = x630;
int32_t x635 = x631;
for(int x636=0; x636 < 50; x636++) {
int32_t x637 = x633;
int32_t x638 = x634;
float x639 = x591[x638];
int32_t x640 = x635;
float x641 = x153[x640];
float x642 = x639 + x641;
x625[x637] = x642;
x633 += 1;
x634 += 1;
x635 += 1;

}
x626 += 50;
x627 += 50;

}
float* x653 = (float*)myMalloc(1000 * sizeof(float));;
for(int x654=0; x654 < 1000; x654++) {
x653[x654] = 0.0f;

}
float* x658 = (float*)myMalloc(1000 * sizeof(float));;
for(int x659=0; x659 < 1000; x659++) {
float x660 = x625[x659];
float x661 = -1.0f * x660;
double x662 = (double)x661;
double x663 = exp(x662);
float x664 = (float)x663;
float x665 = x664 + 1.0f;
float x666 = 1.0f / x665;
x658[x659] = x666;

}
float* x670 = (float*)myMalloc(1000 * sizeof(float));;
for(int x671=0; x671 < 1000; x671++) {
x670[x671] = 0.0f;

}
// dot: WrappedArray(20, 26), List(26, 50)
float* x676 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x373,26,x91,50,0,x676,50);
float* x678 = (float*)myMalloc(1000 * sizeof(float));;
for(int x679=0; x679 < 1000; x679++) {
x678[x679] = 0.0f;

}
// dot: WrappedArray(20, 50), List(50, 50)
float* x684 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x367,50,x104,50,0,x684,50);
float* x686 = (float*)myMalloc(1000 * sizeof(float));;
for(int x687=0; x687 < 1000; x687++) {
x686[x687] = 0.0f;

}
float* x691 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x692 = 0;
int32_t x693 = 0;
int32_t x694 = 0;
for(int x695=0; x695 < 20; x695++) {
int32_t x696 = x693;
int32_t x697 = x694;
int32_t x698 = x692;
int32_t x699 = x698;
int32_t x700 = x696;
int32_t x701 = x697;
for(int x702=0; x702 < 50; x702++) {
int32_t x703 = x699;
int32_t x704 = x700;
float x705 = x676[x704];
int32_t x706 = x701;
float x707 = x684[x706];
float x708 = x705 + x707;
x691[x703] = x708;
x699 += 1;
x700 += 1;
x701 += 1;

}
x692 += 50;
x693 += 50;
x694 += 50;

}
float* x720 = (float*)myMalloc(1000 * sizeof(float));;
for(int x721=0; x721 < 1000; x721++) {
x720[x721] = 0.0f;

}
float* x725 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x726 = 0;
int32_t x727 = 0;
int32_t x728 = 0;
for(int x729=0; x729 < 20; x729++) {
int32_t x730 = x727;
int32_t x731 = x728;
int32_t x732 = x726;
int32_t x733 = x732;
int32_t x734 = x730;
int32_t x735 = x731;
for(int x736=0; x736 < 50; x736++) {
int32_t x737 = x733;
int32_t x738 = x734;
float x739 = x691[x738];
int32_t x740 = x735;
float x741 = x117[x740];
float x742 = x739 + x741;
x725[x737] = x742;
x733 += 1;
x734 += 1;
x735 += 1;

}
x726 += 50;
x727 += 50;

}
float* x753 = (float*)myMalloc(1000 * sizeof(float));;
for(int x754=0; x754 < 1000; x754++) {
x753[x754] = 0.0f;

}
float* x758 = (float*)myMalloc(1000 * sizeof(float));;
for(int x759=0; x759 < 1000; x759++) {
float x760 = x725[x759];
double x761 = (double)x760;
double x762 = tanh(x761);
float x763 = (float)x762;
x758[x759] = x763;

}
float* x767 = (float*)myMalloc(1000 * sizeof(float));;
for(int x768=0; x768 < 1000; x768++) {
x767[x768] = 0.0f;

}
float* x772 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x773 = 0;
int32_t x774 = 0;
int32_t x775 = 0;
for(int x776=0; x776 < 20; x776++) {
int32_t x777 = x774;
int32_t x778 = x775;
int32_t x779 = x773;
int32_t x780 = x779;
int32_t x781 = x777;
int32_t x782 = x778;
for(int x783=0; x783 < 50; x783++) {
int32_t x784 = x780;
int32_t x785 = x781;
float x786 = x458[x785];
int32_t x787 = x782;
float x788 = x369[x787];
float x789 = x786 * x788;
x772[x784] = x789;
x780 += 1;
x781 += 1;
x782 += 1;

}
x773 += 50;
x774 += 50;
x775 += 50;

}
float* x801 = (float*)myMalloc(1000 * sizeof(float));;
for(int x802=0; x802 < 1000; x802++) {
x801[x802] = 0.0f;

}
float* x806 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x807 = 0;
int32_t x808 = 0;
int32_t x809 = 0;
for(int x810=0; x810 < 20; x810++) {
int32_t x811 = x808;
int32_t x812 = x809;
int32_t x813 = x807;
int32_t x814 = x813;
int32_t x815 = x811;
int32_t x816 = x812;
for(int x817=0; x817 < 50; x817++) {
int32_t x818 = x814;
int32_t x819 = x815;
float x820 = x558[x819];
int32_t x821 = x816;
float x822 = x758[x821];
float x823 = x820 * x822;
x806[x818] = x823;
x814 += 1;
x815 += 1;
x816 += 1;

}
x807 += 50;
x808 += 50;
x809 += 50;

}
float* x835 = (float*)myMalloc(1000 * sizeof(float));;
for(int x836=0; x836 < 1000; x836++) {
x835[x836] = 0.0f;

}
float* x840 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x841 = 0;
int32_t x842 = 0;
int32_t x843 = 0;
for(int x844=0; x844 < 20; x844++) {
int32_t x845 = x842;
int32_t x846 = x843;
int32_t x847 = x841;
int32_t x848 = x847;
int32_t x849 = x845;
int32_t x850 = x846;
for(int x851=0; x851 < 50; x851++) {
int32_t x852 = x848;
int32_t x853 = x849;
float x854 = x772[x853];
int32_t x855 = x850;
float x856 = x806[x855];
float x857 = x854 + x856;
x840[x852] = x857;
x848 += 1;
x849 += 1;
x850 += 1;

}
x841 += 50;
x842 += 50;
x843 += 50;

}
float* x869 = (float*)myMalloc(1000 * sizeof(float));;
for(int x870=0; x870 < 1000; x870++) {
x869[x870] = 0.0f;

}
float* x874 = (float*)myMalloc(1000 * sizeof(float));;
for(int x875=0; x875 < 1000; x875++) {
float x876 = x840[x875];
double x877 = (double)x876;
double x878 = tanh(x877);
float x879 = (float)x878;
x874[x875] = x879;

}
float* x883 = (float*)myMalloc(1000 * sizeof(float));;
for(int x884=0; x884 < 1000; x884++) {
x883[x884] = 0.0f;

}
float* x888 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x889 = 0;
int32_t x890 = 0;
int32_t x891 = 0;
for(int x892=0; x892 < 20; x892++) {
int32_t x893 = x890;
int32_t x894 = x891;
int32_t x895 = x889;
int32_t x896 = x895;
int32_t x897 = x893;
int32_t x898 = x894;
for(int x899=0; x899 < 50; x899++) {
int32_t x900 = x896;
int32_t x901 = x897;
float x902 = x658[x901];
int32_t x903 = x898;
float x904 = x874[x903];
float x905 = x902 * x904;
x888[x900] = x905;
x896 += 1;
x897 += 1;
x898 += 1;

}
x889 += 50;
x890 += 50;
x891 += 50;

}
float* x917 = (float*)myMalloc(1000 * sizeof(float));;
for(int x918=0; x918 < 1000; x918++) {
x917[x918] = 0.0f;

}
// dot: List(20, 50), List(50, 26)
float* x923 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x888,50,x163,26,0,x923,26);
float* x925 = (float*)myMalloc(520 * sizeof(float));;
for(int x927=0; x927 < 520; x927++) {
x925[x927] = 0.0f;

}
float* x931 = (float*)myMalloc(520 * sizeof(float));;
int32_t x932 = 0;
int32_t x933 = 0;
int32_t x934 = 0;
for(int x935=0; x935 < 20; x935++) {
int32_t x936 = x933;
int32_t x937 = x934;
int32_t x938 = x932;
int32_t x939 = x938;
int32_t x940 = x936;
int32_t x941 = x937;
for(int x942=0; x942 < 26; x942++) {
int32_t x943 = x939;
int32_t x944 = x940;
float x945 = x923[x944];
int32_t x946 = x941;
float x947 = x176[x946];
float x948 = x945 + x947;
x931[x943] = x948;
x939 += 1;
x940 += 1;
x941 += 1;

}
x932 += 26;
x933 += 26;

}
float* x959 = (float*)myMalloc(520 * sizeof(float));;
for(int x960=0; x960 < 520; x960++) {
x959[x960] = 0.0f;

}
int* x964 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x965=0; x965 < 20; x965++) {
int32_t x966 = x965 * 20;
int32_t x967 = x363 + x966;
int32_t x968 = x274[x967];
x964[x965] = x968;

}
float* x972 = (float*)myMalloc(20 * sizeof(float));;
int32_t x973 = 0;
for(int x974=0; x974 < 20; x974++) {
float x975 = -3.4028235E38f;
for(int x976=0; x976 < 26; x976++) {
int32_t x977 = x973;
float x978 = x931[x977];
float x979 = x975;
bool x980 = x978 > x979;
if (x980) {
float x981 = x931[x977];
x975 = x981;
} else {
}
x973 += 1;

}
float x988 = x975;
x972[x974] = x988;

}
float* x992 = (float*)myMalloc(520 * sizeof(float));;
for(int x993=0; x993 < 520; x993++) {
x992[x993] = 0.0f;

}
int32_t x997 = 0;
for(int x998=0; x998 < 20; x998++) {
for(int x999=0; x999 < 26; x999++) {
int32_t x1000 = x997;
float x1001 = x931[x1000];
float x1002 = x972[x998];
float x1003 = x1001 - x1002;
double x1004 = (double)x1003;
double x1005 = exp(x1004);
float x1006 = (float)x1005;
x992[x1000] = x1006;
x997 += 1;

}

}
float* x1013 = (float*)myMalloc(20 * sizeof(float));;
for(int x1014=0; x1014 < 20; x1014++) {
x1013[x1014] = 0.0f;

}
for(int x1018=0; x1018 < 20; x1018++) {
int32_t x1019 = x1018;
int32_t x1020 = x1018 * 26;
int32_t x1021 = x1020;
for(int x1022=0; x1022 < 26; x1022++) {
int32_t x1023 = x1019;
float x1024 = x1013[x1023];
int32_t x1025 = x1021;
float x1026 = x992[x1025];
float x1027 = x1024 + x1026;
x1013[x1023] = x1027;
x1021 += 1;

}

}
x997 = 0;
for(int x1035=0; x1035 < 20; x1035++) {
float x1036 = x972[x1035];
float x1037 = x1013[x1035];
double x1038 = (double)x1037;
double x1039 = log(x1038);
float x1040 = (float)x1039;
float x1041 = x1036 + x1040;
for(int x1042=0; x1042 < 26; x1042++) {
int32_t x1043 = x997;
float x1044 = x931[x1043];
float x1045 = x1044 - x1041;
x992[x1043] = x1045;
x997 += 1;

}

}
float* x1052 = (float*)myMalloc(520 * sizeof(float));;
for(int x1053=0; x1053 < 520; x1053++) {
x1052[x1053] = 0.0f;

}
float* x1057 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1058 = 0;
for(int x1059=0; x1059 < 20; x1059++) {
int32_t x1060 = x1058;
int32_t x1061 = x964[x1059];
int32_t x1062 = x1060 + x1061;
float x1063 = x992[x1062];
float x1064 = -1.0f * x1063;
x1057[x1059] = x1064;
x1058 += 26;

}
float* x1069 = (float*)myMalloc(20 * sizeof(float));;
for(int x1070=0; x1070 < 20; x1070++) {
x1069[x1070] = 0.0f;

}
float x1074 = 0.0f;
for(int x1075=0; x1075 < 20; x1075++) {
float x1076 = x1074;
float x1077 = x1057[x1075];
float x1078 = x1076 + x1077;
x1074 = x1078;

}
float x1082 = x1074;
float* x1083 = (float*)myMalloc(1 * sizeof(float));;
x1083[0] = x1082;
float* x1085 = (float*)myMalloc(1 * sizeof(float));;
for(int x1086=0; x1086 < 1; x1086++) {
x1085[x1086] = 0.0f;

}
float* x1090 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1091 = 0;
int32_t x1092 = 0;
int32_t x1093 = 0;
int32_t x1094 = x1091;
int32_t x1095 = x1092;
float x1096 = x365[x1095];
int32_t x1097 = x1093;
float x1098 = x1083[x1097];
float x1099 = x1096 + x1098;
x1090[x1094] = x1099;
x1091 += 1;
float* x1102 = (float*)myMalloc(1 * sizeof(float));;
for(int x1103=0; x1103 < 1; x1103++) {
x1102[x1103] = 0.0f;

}
float** x2513 = (float**)myMalloc(6 * sizeof(float*));;
x2513[0] = x1090;
x2513[1] = x1102;
x2513[2] = x888;
x2513[3] = x917;
x2513[4] = x840;
x2513[5] = x869;
int32_t x1107 = x363 + 1;
x1108(x1107,x2513);
int32_t x2522 = 0;
int32_t x2523 = 0;
int32_t x2524 = 0;
int32_t x2525 = x2522;
float x2526 = x366[x2525];
float x2527 = x365[x2525];
int32_t x2528 = x2523;
float x2529 = x1083[x2528];
int32_t x2530 = x2524;
float x2531 = x1102[x2530];
float x2532 = x2526 + x2531;
x366[x2525] = x2532;
float x2534 = x1085[x2528];
float x2535 = x365[x2525];
float x2536 = x1083[x2528];
float x2537 = x1102[x2530];
float x2538 = x2534 + x2537;
x1085[x2528] = x2538;
x2524 += 1;
// += tensor of dim 0
float x2542 = x1085[0];
for(int x2543=0; x2543 < 20; x2543++) {
float x2544 = x1069[x2543];
float x2545 = x2544 + x2542;
x1069[x2543] = x2545;

}
int32_t x2549 = 0;
for(int x2550=0; x2550 < 20; x2550++) {
int32_t x2551 = x2549;
int32_t x2552 = x964[x2550];
int32_t x2553 = x2551 + x2552;
float x2554 = x1052[x2553];
float x2555 = x1069[x2550];
float x2556 = -1.0f * x2555;
float x2557 = x2554 + x2556;
x1052[x2553] = x2557;
x2549 += 26;

}
float* x2562 = (float*)myMalloc(20 * sizeof(float));;
for(int x2563=0; x2563 < 20; x2563++) {
x2562[x2563] = 0.0f;

}
for(int x2567=0; x2567 < 20; x2567++) {
int32_t x2568 = x2567;
int32_t x2569 = x2567 * 26;
int32_t x2570 = x2569;
for(int x2571=0; x2571 < 26; x2571++) {
int32_t x2572 = x2568;
float x2573 = x2562[x2572];
int32_t x2574 = x2570;
float x2575 = x1052[x2574];
float x2576 = x2573 + x2575;
x2562[x2572] = x2576;
x2570 += 1;

}

}
int32_t x2583 = 0;
for(int x2584=0; x2584 < 20; x2584++) {
for(int x2585=0; x2585 < 26; x2585++) {
int32_t x2586 = x2583;
float x2587 = x959[x2586];
float x2588 = x1052[x2586];
float x2589 = x992[x2586];
float x2593 = x2562[x2584];
double x2590 = (double)x2589;
double x2591 = exp(x2590);
float x2592 = (float)x2591;
float x2594 = x2592 * x2593;
float x2595 = x2588 - x2594;
float x2596 = x2587 + x2595;
x959[x2586] = x2596;
x2583 += 1;

}

}
int32_t x2603 = 0;
int32_t x2604 = 0;
int32_t x2605 = 0;
for(int x2606=0; x2606 < 20; x2606++) {
int32_t x2607 = x2603;
int32_t x2608 = x2604;
int32_t x2609 = x2605;
int32_t x2610 = x2607;
int32_t x2611 = x2608;
int32_t x2612 = x2609;
for(int x2613=0; x2613 < 26; x2613++) {
int32_t x2614 = x2610;
float x2615 = x925[x2614];
float x2616 = x923[x2614];
int32_t x2617 = x2611;
float x2618 = x176[x2617];
int32_t x2619 = x2612;
float x2620 = x959[x2619];
float x2621 = x2615 + x2620;
x925[x2614] = x2621;
float x2623 = x182[x2617];
float x2624 = x923[x2614];
float x2625 = x176[x2617];
float x2626 = x959[x2619];
float x2627 = x2623 + x2626;
x182[x2617] = x2627;
x2612 += 1;
x2610 += 1;
x2611 += 1;

}
x2605 += 26;
x2603 += 26;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x925,26,x163,26,1,x917,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x888,50,x925,26,1,x171,26);
int32_t x2640 = 0;
int32_t x2641 = 0;
int32_t x2642 = 0;
for(int x2643=0; x2643 < 20; x2643++) {
int32_t x2644 = x2640;
int32_t x2645 = x2641;
int32_t x2646 = x2642;
int32_t x2647 = x2644;
int32_t x2648 = x2645;
int32_t x2649 = x2646;
for(int x2650=0; x2650 < 50; x2650++) {
int32_t x2651 = x2647;
float x2652 = x670[x2651];
float x2653 = x658[x2651];
int32_t x2654 = x2648;
float x2655 = x874[x2654];
int32_t x2656 = x2649;
float x2657 = x917[x2656];
float x2658 = x2657 * x2655;
float x2659 = x2652 + x2658;
x670[x2651] = x2659;
float x2661 = x883[x2654];
float x2662 = x658[x2651];
float x2663 = x874[x2654];
float x2664 = x917[x2656];
float x2665 = x2664 * x2662;
float x2666 = x2661 + x2665;
x883[x2654] = x2666;
x2649 += 1;
x2647 += 1;
x2648 += 1;

}
x2642 += 50;
x2640 += 50;
x2641 += 50;

}
for(int x2678=0; x2678 < 1000; x2678++) {
float x2679 = x869[x2678];
float x2680 = x874[x2678];
float x2683 = x883[x2678];
float x2681 = x2680 * x2680;
float x2682 = 1.0f - x2681;
float x2684 = x2682 * x2683;
float x2685 = x2679 + x2684;
x869[x2678] = x2685;

}
int32_t x2689 = 0;
int32_t x2690 = 0;
int32_t x2691 = 0;
for(int x2692=0; x2692 < 20; x2692++) {
int32_t x2693 = x2689;
int32_t x2694 = x2690;
int32_t x2695 = x2691;
int32_t x2696 = x2693;
int32_t x2697 = x2694;
int32_t x2698 = x2695;
for(int x2699=0; x2699 < 50; x2699++) {
int32_t x2700 = x2696;
float x2701 = x801[x2700];
float x2702 = x772[x2700];
int32_t x2703 = x2697;
float x2704 = x806[x2703];
int32_t x2705 = x2698;
float x2706 = x869[x2705];
float x2707 = x2701 + x2706;
x801[x2700] = x2707;
float x2709 = x835[x2703];
float x2710 = x772[x2700];
float x2711 = x806[x2703];
float x2712 = x869[x2705];
float x2713 = x2709 + x2712;
x835[x2703] = x2713;
x2698 += 1;
x2696 += 1;
x2697 += 1;

}
x2691 += 50;
x2689 += 50;
x2690 += 50;

}
int32_t x2725 = 0;
int32_t x2726 = 0;
int32_t x2727 = 0;
for(int x2728=0; x2728 < 20; x2728++) {
int32_t x2729 = x2725;
int32_t x2730 = x2726;
int32_t x2731 = x2727;
int32_t x2732 = x2729;
int32_t x2733 = x2730;
int32_t x2734 = x2731;
for(int x2735=0; x2735 < 50; x2735++) {
int32_t x2736 = x2732;
float x2737 = x570[x2736];
float x2738 = x558[x2736];
int32_t x2739 = x2733;
float x2740 = x758[x2739];
int32_t x2741 = x2734;
float x2742 = x835[x2741];
float x2743 = x2742 * x2740;
float x2744 = x2737 + x2743;
x570[x2736] = x2744;
float x2746 = x767[x2739];
float x2747 = x558[x2736];
float x2748 = x758[x2739];
float x2749 = x835[x2741];
float x2750 = x2749 * x2747;
float x2751 = x2746 + x2750;
x767[x2739] = x2751;
x2734 += 1;
x2732 += 1;
x2733 += 1;

}
x2727 += 50;
x2725 += 50;
x2726 += 50;

}
int32_t x2763 = 0;
int32_t x2764 = 0;
int32_t x2765 = 0;
for(int x2766=0; x2766 < 20; x2766++) {
int32_t x2767 = x2763;
int32_t x2768 = x2764;
int32_t x2769 = x2765;
int32_t x2770 = x2767;
int32_t x2771 = x2768;
int32_t x2772 = x2769;
for(int x2773=0; x2773 < 50; x2773++) {
int32_t x2774 = x2770;
float x2775 = x470[x2774];
float x2776 = x458[x2774];
int32_t x2777 = x2771;
float x2778 = x369[x2777];
int32_t x2779 = x2772;
float x2780 = x801[x2779];
float x2781 = x2780 * x2778;
float x2782 = x2775 + x2781;
x470[x2774] = x2782;
float x2784 = x370[x2777];
float x2785 = x458[x2774];
float x2786 = x369[x2777];
float x2787 = x801[x2779];
float x2788 = x2787 * x2785;
float x2789 = x2784 + x2788;
x370[x2777] = x2789;
x2772 += 1;
x2770 += 1;
x2771 += 1;

}
x2765 += 50;
x2763 += 50;
x2764 += 50;

}
for(int x2801=0; x2801 < 1000; x2801++) {
float x2802 = x753[x2801];
float x2803 = x758[x2801];
float x2806 = x767[x2801];
float x2804 = x2803 * x2803;
float x2805 = 1.0f - x2804;
float x2807 = x2805 * x2806;
float x2808 = x2802 + x2807;
x753[x2801] = x2808;

}
int32_t x2812 = 0;
int32_t x2813 = 0;
int32_t x2814 = 0;
for(int x2815=0; x2815 < 20; x2815++) {
int32_t x2816 = x2812;
int32_t x2817 = x2813;
int32_t x2818 = x2814;
int32_t x2819 = x2816;
int32_t x2820 = x2817;
int32_t x2821 = x2818;
for(int x2822=0; x2822 < 50; x2822++) {
int32_t x2823 = x2819;
float x2824 = x720[x2823];
float x2825 = x691[x2823];
int32_t x2826 = x2820;
float x2827 = x117[x2826];
int32_t x2828 = x2821;
float x2829 = x753[x2828];
float x2830 = x2824 + x2829;
x720[x2823] = x2830;
float x2832 = x122[x2826];
float x2833 = x691[x2823];
float x2834 = x117[x2826];
float x2835 = x753[x2828];
float x2836 = x2832 + x2835;
x122[x2826] = x2836;
x2821 += 1;
x2819 += 1;
x2820 += 1;

}
x2814 += 50;
x2812 += 50;

}
int32_t x2847 = 0;
int32_t x2848 = 0;
int32_t x2849 = 0;
for(int x2850=0; x2850 < 20; x2850++) {
int32_t x2851 = x2847;
int32_t x2852 = x2848;
int32_t x2853 = x2849;
int32_t x2854 = x2851;
int32_t x2855 = x2852;
int32_t x2856 = x2853;
for(int x2857=0; x2857 < 50; x2857++) {
int32_t x2858 = x2854;
float x2859 = x678[x2858];
float x2860 = x676[x2858];
int32_t x2861 = x2855;
float x2862 = x684[x2861];
int32_t x2863 = x2856;
float x2864 = x720[x2863];
float x2865 = x2859 + x2864;
x678[x2858] = x2865;
float x2867 = x686[x2861];
float x2868 = x676[x2858];
float x2869 = x684[x2861];
float x2870 = x720[x2863];
float x2871 = x2867 + x2870;
x686[x2861] = x2871;
x2856 += 1;
x2854 += 1;
x2855 += 1;

}
x2849 += 50;
x2847 += 50;
x2848 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x686,50,x104,50,1,x368,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x367,50,x686,50,1,x112,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x678,50,x91,50,1,x374,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x373,26,x678,50,1,x99,50);
for(int x2887=0; x2887 < 1000; x2887++) {
float x2888 = x653[x2887];
float x2889 = x658[x2887];
float x2892 = x670[x2887];
float x2890 = 1.0f - x2889;
float x2891 = x2890 * x2889;
float x2893 = x2891 * x2892;
float x2894 = x2888 + x2893;
x653[x2887] = x2894;

}
int32_t x2898 = 0;
int32_t x2899 = 0;
int32_t x2900 = 0;
for(int x2901=0; x2901 < 20; x2901++) {
int32_t x2902 = x2898;
int32_t x2903 = x2899;
int32_t x2904 = x2900;
int32_t x2905 = x2902;
int32_t x2906 = x2903;
int32_t x2907 = x2904;
for(int x2908=0; x2908 < 50; x2908++) {
int32_t x2909 = x2905;
float x2910 = x620[x2909];
float x2911 = x591[x2909];
int32_t x2912 = x2906;
float x2913 = x153[x2912];
int32_t x2914 = x2907;
float x2915 = x653[x2914];
float x2916 = x2910 + x2915;
x620[x2909] = x2916;
float x2918 = x158[x2912];
float x2919 = x591[x2909];
float x2920 = x153[x2912];
float x2921 = x653[x2914];
float x2922 = x2918 + x2921;
x158[x2912] = x2922;
x2907 += 1;
x2905 += 1;
x2906 += 1;

}
x2900 += 50;
x2898 += 50;

}
int32_t x2933 = 0;
int32_t x2934 = 0;
int32_t x2935 = 0;
for(int x2936=0; x2936 < 20; x2936++) {
int32_t x2937 = x2933;
int32_t x2938 = x2934;
int32_t x2939 = x2935;
int32_t x2940 = x2937;
int32_t x2941 = x2938;
int32_t x2942 = x2939;
for(int x2943=0; x2943 < 50; x2943++) {
int32_t x2944 = x2940;
float x2945 = x578[x2944];
float x2946 = x576[x2944];
int32_t x2947 = x2941;
float x2948 = x584[x2947];
int32_t x2949 = x2942;
float x2950 = x620[x2949];
float x2951 = x2945 + x2950;
x578[x2944] = x2951;
float x2953 = x586[x2947];
float x2954 = x576[x2944];
float x2955 = x584[x2947];
float x2956 = x620[x2949];
float x2957 = x2953 + x2956;
x586[x2947] = x2957;
x2942 += 1;
x2940 += 1;
x2941 += 1;

}
x2935 += 50;
x2933 += 50;
x2934 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x586,50,x140,50,1,x368,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x367,50,x586,50,1,x148,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x578,50,x127,50,1,x374,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x373,26,x578,50,1,x135,50);
for(int x2973=0; x2973 < 1000; x2973++) {
float x2974 = x553[x2973];
float x2975 = x558[x2973];
float x2978 = x570[x2973];
float x2976 = 1.0f - x2975;
float x2977 = x2976 * x2975;
float x2979 = x2977 * x2978;
float x2980 = x2974 + x2979;
x553[x2973] = x2980;

}
int32_t x2984 = 0;
int32_t x2985 = 0;
int32_t x2986 = 0;
for(int x2987=0; x2987 < 20; x2987++) {
int32_t x2988 = x2984;
int32_t x2989 = x2985;
int32_t x2990 = x2986;
int32_t x2991 = x2988;
int32_t x2992 = x2989;
int32_t x2993 = x2990;
for(int x2994=0; x2994 < 50; x2994++) {
int32_t x2995 = x2991;
float x2996 = x520[x2995];
float x2997 = x491[x2995];
int32_t x2998 = x2992;
float x2999 = x81[x2998];
int32_t x3000 = x2993;
float x3001 = x553[x3000];
float x3002 = x2996 + x3001;
x520[x2995] = x3002;
float x3004 = x86[x2998];
float x3005 = x491[x2995];
float x3006 = x81[x2998];
float x3007 = x553[x3000];
float x3008 = x3004 + x3007;
x86[x2998] = x3008;
x2993 += 1;
x2991 += 1;
x2992 += 1;

}
x2986 += 50;
x2984 += 50;

}
int32_t x3019 = 0;
int32_t x3020 = 0;
int32_t x3021 = 0;
for(int x3022=0; x3022 < 20; x3022++) {
int32_t x3023 = x3019;
int32_t x3024 = x3020;
int32_t x3025 = x3021;
int32_t x3026 = x3023;
int32_t x3027 = x3024;
int32_t x3028 = x3025;
for(int x3029=0; x3029 < 50; x3029++) {
int32_t x3030 = x3026;
float x3031 = x478[x3030];
float x3032 = x476[x3030];
int32_t x3033 = x3027;
float x3034 = x484[x3033];
int32_t x3035 = x3028;
float x3036 = x520[x3035];
float x3037 = x3031 + x3036;
x478[x3030] = x3037;
float x3039 = x486[x3033];
float x3040 = x476[x3030];
float x3041 = x484[x3033];
float x3042 = x520[x3035];
float x3043 = x3039 + x3042;
x486[x3033] = x3043;
x3028 += 1;
x3026 += 1;
x3027 += 1;

}
x3021 += 50;
x3019 += 50;
x3020 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x486,50,x68,50,1,x368,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x367,50,x486,50,1,x76,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x478,50,x55,50,1,x374,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x373,26,x478,50,1,x63,50);
for(int x3059=0; x3059 < 1000; x3059++) {
float x3060 = x453[x3059];
float x3061 = x458[x3059];
float x3064 = x470[x3059];
float x3062 = 1.0f - x3061;
float x3063 = x3062 * x3061;
float x3065 = x3063 * x3064;
float x3066 = x3060 + x3065;
x453[x3059] = x3066;

}
int32_t x3070 = 0;
int32_t x3071 = 0;
int32_t x3072 = 0;
for(int x3073=0; x3073 < 20; x3073++) {
int32_t x3074 = x3070;
int32_t x3075 = x3071;
int32_t x3076 = x3072;
int32_t x3077 = x3074;
int32_t x3078 = x3075;
int32_t x3079 = x3076;
for(int x3080=0; x3080 < 50; x3080++) {
int32_t x3081 = x3077;
float x3082 = x420[x3081];
float x3083 = x391[x3081];
int32_t x3084 = x3078;
float x3085 = x44[x3084];
int32_t x3086 = x3079;
float x3087 = x453[x3086];
float x3088 = x3082 + x3087;
x420[x3081] = x3088;
float x3090 = x50[x3084];
float x3091 = x391[x3081];
float x3092 = x44[x3084];
float x3093 = x453[x3086];
float x3094 = x3090 + x3093;
x50[x3084] = x3094;
x3079 += 1;
x3077 += 1;
x3078 += 1;

}
x3072 += 50;
x3070 += 50;

}
int32_t x3105 = 0;
int32_t x3106 = 0;
int32_t x3107 = 0;
for(int x3108=0; x3108 < 20; x3108++) {
int32_t x3109 = x3105;
int32_t x3110 = x3106;
int32_t x3111 = x3107;
int32_t x3112 = x3109;
int32_t x3113 = x3110;
int32_t x3114 = x3111;
for(int x3115=0; x3115 < 50; x3115++) {
int32_t x3116 = x3112;
float x3117 = x378[x3116];
float x3118 = x376[x3116];
int32_t x3119 = x3113;
float x3120 = x384[x3119];
int32_t x3121 = x3114;
float x3122 = x420[x3121];
float x3123 = x3117 + x3122;
x378[x3116] = x3123;
float x3125 = x386[x3119];
float x3126 = x376[x3116];
float x3127 = x384[x3119];
float x3128 = x420[x3121];
float x3129 = x3125 + x3128;
x386[x3119] = x3129;
x3114 += 1;
x3112 += 1;
x3113 += 1;

}
x3107 += 50;
x3105 += 50;
x3106 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x386,50,x30,50,1,x368,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x367,50,x386,50,1,x39,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x378,50,x16,50,1,x374,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x373,26,x378,50,1,x25,50);
} else {
float x3146 = 0.0f;
float x3147 = x3146;
float x3148 = x365[0];
float x3149 = x3147 + x3148;
x3146 = x3149;
float x3151 = x3146;
float* x3152 = (float*)myMalloc(1 * sizeof(float));;
x3152[0] = x3151;
float* x3154 = (float*)myMalloc(1 * sizeof(float));;
for(int x3155=0; x3155 < 1; x3155++) {
x3154[x3155] = 0.0f;

}
float x3159 = x3154[0];
x3154[0] = 1.0f;
float x3161 = x3152[0];
x297[0] = x3161;
// += tensor of dim 0
float x3164 = x3154[0];
float x3165 = x366[0];
float x3166 = x3165 + x3164;
x366[0] = x3166;
}
};
x360(0,x3171);
float x3180 = x297[0];
int32_t x3181 = x264 % 100;
bool x3182 = x3181 == 0;
if (x3182) {
printf("iter %d, loss %f\n",x264,x3180);
int32_t x3184 = x264 / 100;
double x3185 = (double)x3180;
x258[x3184] = x3185;
} else {
}
for(int x3189=0; x3189 < 1300; x3189++) {
float x3190 = x63[x3189];
bool x3191 = x3190 > 5.0f;
if (x3191) {
x63[x3189] = 5.0f;
} else {
}
float x3195 = x63[x3189];
bool x3196 = x3195 < -5.0f;
if (x3196) {
x63[x3189] = -5.0f;
} else {
}

}
float* x3202 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x3203 = 0;
int32_t x3204 = 0;
int32_t x3205 = 0;
for(int x3206=0; x3206 < 26; x3206++) {
int32_t x3207 = x3204;
int32_t x3208 = x3205;
int32_t x3209 = x3203;
int32_t x3210 = x3209;
int32_t x3211 = x3207;
int32_t x3212 = x3208;
for(int x3213=0; x3213 < 50; x3213++) {
int32_t x3214 = x3210;
int32_t x3215 = x3211;
float x3216 = x63[x3215];
int32_t x3217 = x3212;
float x3218 = x63[x3217];
float x3219 = x3216 * x3218;
x3202[x3214] = x3219;
x3210 += 1;
x3211 += 1;
x3212 += 1;

}
x3203 += 50;
x3204 += 50;
x3205 += 50;

}
for(int x3231=0; x3231 < 1300; x3231++) {
float x3232 = x187[x3231];
float x3233 = x3202[x3231];
float x3234 = x3232 + x3233;
x187[x3231] = x3234;

}
float* x3238 = (float*)myMalloc(1300 * sizeof(float));;
for(int x3239=0; x3239 < 1300; x3239++) {
float x3240 = x63[x3239];
float x3241 = x3240 * 0.1f;
x3238[x3239] = x3241;

}
float* x3245 = (float*)myMalloc(1300 * sizeof(float));;
for(int x3246=0; x3246 < 1300; x3246++) {
float x3247 = x187[x3246];
float x3248 = x3247 + 1.0E-8f;
x3245[x3246] = x3248;

}
float* x3252 = (float*)myMalloc(1300 * sizeof(float));;
for(int x3253=0; x3253 < 1300; x3253++) {
float x3254 = x3245[x3253];
double x3255 = (double)x3254;
double x3256 = sqrt(x3255);
float x3257 = (float)x3256;
x3252[x3253] = x3257;

}
float* x3261 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x3262 = 0;
int32_t x3263 = 0;
int32_t x3264 = 0;
for(int x3265=0; x3265 < 26; x3265++) {
int32_t x3266 = x3263;
int32_t x3267 = x3264;
int32_t x3268 = x3262;
int32_t x3269 = x3268;
int32_t x3270 = x3266;
int32_t x3271 = x3267;
for(int x3272=0; x3272 < 50; x3272++) {
int32_t x3273 = x3269;
int32_t x3274 = x3270;
float x3275 = x3238[x3274];
int32_t x3276 = x3271;
float x3277 = x3252[x3276];
float x3278 = x3275 / x3277;
x3261[x3273] = x3278;
x3269 += 1;
x3270 += 1;
x3271 += 1;

}
x3262 += 50;
x3263 += 50;
x3264 += 50;

}
for(int x3290=0; x3290 < 1300; x3290++) {
float x3291 = x55[x3290];
float x3292 = x3261[x3290];
float x3293 = x3291 - x3292;
x55[x3290] = x3293;

}
for(int x3297=0; x3297 < 1300; x3297++) {
float x3298 = x63[x3297];
x63[x3297] = 0.0f;

}
for(int x3302=0; x3302 < 50; x3302++) {
float x3303 = x86[x3302];
bool x3304 = x3303 > 5.0f;
if (x3304) {
x86[x3302] = 5.0f;
} else {
}
float x3308 = x86[x3302];
bool x3309 = x3308 < -5.0f;
if (x3309) {
x86[x3302] = -5.0f;
} else {
}

}
float* x3315 = (float*)myMalloc(50 * sizeof(float));;
int32_t x3316 = 0;
int32_t x3317 = 0;
int32_t x3318 = 0;
for(int x3319=0; x3319 < 50; x3319++) {
int32_t x3320 = x3316;
int32_t x3321 = x3317;
float x3322 = x86[x3321];
int32_t x3323 = x3318;
float x3324 = x86[x3323];
float x3325 = x3322 * x3324;
x3315[x3320] = x3325;
x3316 += 1;
x3317 += 1;
x3318 += 1;

}
for(int x3332=0; x3332 < 50; x3332++) {
float x3333 = x192[x3332];
float x3334 = x3315[x3332];
float x3335 = x3333 + x3334;
x192[x3332] = x3335;

}
float* x3339 = (float*)myMalloc(50 * sizeof(float));;
for(int x3340=0; x3340 < 50; x3340++) {
float x3341 = x86[x3340];
float x3342 = x3341 * 0.1f;
x3339[x3340] = x3342;

}
float* x3346 = (float*)myMalloc(50 * sizeof(float));;
for(int x3347=0; x3347 < 50; x3347++) {
float x3348 = x192[x3347];
float x3349 = x3348 + 1.0E-8f;
x3346[x3347] = x3349;

}
float* x3353 = (float*)myMalloc(50 * sizeof(float));;
for(int x3354=0; x3354 < 50; x3354++) {
float x3355 = x3346[x3354];
double x3356 = (double)x3355;
double x3357 = sqrt(x3356);
float x3358 = (float)x3357;
x3353[x3354] = x3358;

}
float* x3362 = (float*)myMalloc(50 * sizeof(float));;
int32_t x3363 = 0;
int32_t x3364 = 0;
int32_t x3365 = 0;
for(int x3366=0; x3366 < 50; x3366++) {
int32_t x3367 = x3363;
int32_t x3368 = x3364;
float x3369 = x3339[x3368];
int32_t x3370 = x3365;
float x3371 = x3353[x3370];
float x3372 = x3369 / x3371;
x3362[x3367] = x3372;
x3363 += 1;
x3364 += 1;
x3365 += 1;

}
for(int x3379=0; x3379 < 50; x3379++) {
float x3380 = x81[x3379];
float x3381 = x3362[x3379];
float x3382 = x3380 - x3381;
x81[x3379] = x3382;

}
for(int x3386=0; x3386 < 50; x3386++) {
float x3387 = x86[x3386];
x86[x3386] = 0.0f;

}
for(int x3391=0; x3391 < 2500; x3391++) {
float x3392 = x76[x3391];
bool x3393 = x3392 > 5.0f;
if (x3393) {
x76[x3391] = 5.0f;
} else {
}
float x3397 = x76[x3391];
bool x3398 = x3397 < -5.0f;
if (x3398) {
x76[x3391] = -5.0f;
} else {
}

}
float* x3404 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x3405 = 0;
int32_t x3406 = 0;
int32_t x3407 = 0;
for(int x3408=0; x3408 < 50; x3408++) {
int32_t x3409 = x3406;
int32_t x3410 = x3407;
int32_t x3411 = x3405;
int32_t x3412 = x3411;
int32_t x3413 = x3409;
int32_t x3414 = x3410;
for(int x3415=0; x3415 < 50; x3415++) {
int32_t x3416 = x3412;
int32_t x3417 = x3413;
float x3418 = x76[x3417];
int32_t x3419 = x3414;
float x3420 = x76[x3419];
float x3421 = x3418 * x3420;
x3404[x3416] = x3421;
x3412 += 1;
x3413 += 1;
x3414 += 1;

}
x3405 += 50;
x3406 += 50;
x3407 += 50;

}
for(int x3433=0; x3433 < 2500; x3433++) {
float x3434 = x197[x3433];
float x3435 = x3404[x3433];
float x3436 = x3434 + x3435;
x197[x3433] = x3436;

}
float* x3440 = (float*)myMalloc(2500 * sizeof(float));;
for(int x3441=0; x3441 < 2500; x3441++) {
float x3442 = x76[x3441];
float x3443 = x3442 * 0.1f;
x3440[x3441] = x3443;

}
float* x3447 = (float*)myMalloc(2500 * sizeof(float));;
for(int x3448=0; x3448 < 2500; x3448++) {
float x3449 = x197[x3448];
float x3450 = x3449 + 1.0E-8f;
x3447[x3448] = x3450;

}
float* x3454 = (float*)myMalloc(2500 * sizeof(float));;
for(int x3455=0; x3455 < 2500; x3455++) {
float x3456 = x3447[x3455];
double x3457 = (double)x3456;
double x3458 = sqrt(x3457);
float x3459 = (float)x3458;
x3454[x3455] = x3459;

}
float* x3463 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x3464 = 0;
int32_t x3465 = 0;
int32_t x3466 = 0;
for(int x3467=0; x3467 < 50; x3467++) {
int32_t x3468 = x3465;
int32_t x3469 = x3466;
int32_t x3470 = x3464;
int32_t x3471 = x3470;
int32_t x3472 = x3468;
int32_t x3473 = x3469;
for(int x3474=0; x3474 < 50; x3474++) {
int32_t x3475 = x3471;
int32_t x3476 = x3472;
float x3477 = x3440[x3476];
int32_t x3478 = x3473;
float x3479 = x3454[x3478];
float x3480 = x3477 / x3479;
x3463[x3475] = x3480;
x3471 += 1;
x3472 += 1;
x3473 += 1;

}
x3464 += 50;
x3465 += 50;
x3466 += 50;

}
for(int x3492=0; x3492 < 2500; x3492++) {
float x3493 = x68[x3492];
float x3494 = x3463[x3492];
float x3495 = x3493 - x3494;
x68[x3492] = x3495;

}
for(int x3499=0; x3499 < 2500; x3499++) {
float x3500 = x76[x3499];
x76[x3499] = 0.0f;

}
for(int x3504=0; x3504 < 50; x3504++) {
float x3505 = x50[x3504];
bool x3506 = x3505 > 5.0f;
if (x3506) {
x50[x3504] = 5.0f;
} else {
}
float x3510 = x50[x3504];
bool x3511 = x3510 < -5.0f;
if (x3511) {
x50[x3504] = -5.0f;
} else {
}

}
float* x3517 = (float*)myMalloc(50 * sizeof(float));;
int32_t x3518 = 0;
int32_t x3519 = 0;
int32_t x3520 = 0;
for(int x3521=0; x3521 < 50; x3521++) {
int32_t x3522 = x3518;
int32_t x3523 = x3519;
float x3524 = x50[x3523];
int32_t x3525 = x3520;
float x3526 = x50[x3525];
float x3527 = x3524 * x3526;
x3517[x3522] = x3527;
x3518 += 1;
x3519 += 1;
x3520 += 1;

}
for(int x3534=0; x3534 < 50; x3534++) {
float x3535 = x202[x3534];
float x3536 = x3517[x3534];
float x3537 = x3535 + x3536;
x202[x3534] = x3537;

}
float* x3541 = (float*)myMalloc(50 * sizeof(float));;
for(int x3542=0; x3542 < 50; x3542++) {
float x3543 = x50[x3542];
float x3544 = x3543 * 0.1f;
x3541[x3542] = x3544;

}
float* x3548 = (float*)myMalloc(50 * sizeof(float));;
for(int x3549=0; x3549 < 50; x3549++) {
float x3550 = x202[x3549];
float x3551 = x3550 + 1.0E-8f;
x3548[x3549] = x3551;

}
float* x3555 = (float*)myMalloc(50 * sizeof(float));;
for(int x3556=0; x3556 < 50; x3556++) {
float x3557 = x3548[x3556];
double x3558 = (double)x3557;
double x3559 = sqrt(x3558);
float x3560 = (float)x3559;
x3555[x3556] = x3560;

}
float* x3564 = (float*)myMalloc(50 * sizeof(float));;
int32_t x3565 = 0;
int32_t x3566 = 0;
int32_t x3567 = 0;
for(int x3568=0; x3568 < 50; x3568++) {
int32_t x3569 = x3565;
int32_t x3570 = x3566;
float x3571 = x3541[x3570];
int32_t x3572 = x3567;
float x3573 = x3555[x3572];
float x3574 = x3571 / x3573;
x3564[x3569] = x3574;
x3565 += 1;
x3566 += 1;
x3567 += 1;

}
for(int x3581=0; x3581 < 50; x3581++) {
float x3582 = x44[x3581];
float x3583 = x3564[x3581];
float x3584 = x3582 - x3583;
x44[x3581] = x3584;

}
for(int x3588=0; x3588 < 50; x3588++) {
float x3589 = x50[x3588];
x50[x3588] = 0.0f;

}
for(int x3593=0; x3593 < 2500; x3593++) {
float x3594 = x39[x3593];
bool x3595 = x3594 > 5.0f;
if (x3595) {
x39[x3593] = 5.0f;
} else {
}
float x3599 = x39[x3593];
bool x3600 = x3599 < -5.0f;
if (x3600) {
x39[x3593] = -5.0f;
} else {
}

}
float* x3606 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x3607 = 0;
int32_t x3608 = 0;
int32_t x3609 = 0;
for(int x3610=0; x3610 < 50; x3610++) {
int32_t x3611 = x3608;
int32_t x3612 = x3609;
int32_t x3613 = x3607;
int32_t x3614 = x3613;
int32_t x3615 = x3611;
int32_t x3616 = x3612;
for(int x3617=0; x3617 < 50; x3617++) {
int32_t x3618 = x3614;
int32_t x3619 = x3615;
float x3620 = x39[x3619];
int32_t x3621 = x3616;
float x3622 = x39[x3621];
float x3623 = x3620 * x3622;
x3606[x3618] = x3623;
x3614 += 1;
x3615 += 1;
x3616 += 1;

}
x3607 += 50;
x3608 += 50;
x3609 += 50;

}
for(int x3635=0; x3635 < 2500; x3635++) {
float x3636 = x207[x3635];
float x3637 = x3606[x3635];
float x3638 = x3636 + x3637;
x207[x3635] = x3638;

}
float* x3642 = (float*)myMalloc(2500 * sizeof(float));;
for(int x3643=0; x3643 < 2500; x3643++) {
float x3644 = x39[x3643];
float x3645 = x3644 * 0.1f;
x3642[x3643] = x3645;

}
float* x3649 = (float*)myMalloc(2500 * sizeof(float));;
for(int x3650=0; x3650 < 2500; x3650++) {
float x3651 = x207[x3650];
float x3652 = x3651 + 1.0E-8f;
x3649[x3650] = x3652;

}
float* x3656 = (float*)myMalloc(2500 * sizeof(float));;
for(int x3657=0; x3657 < 2500; x3657++) {
float x3658 = x3649[x3657];
double x3659 = (double)x3658;
double x3660 = sqrt(x3659);
float x3661 = (float)x3660;
x3656[x3657] = x3661;

}
float* x3665 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x3666 = 0;
int32_t x3667 = 0;
int32_t x3668 = 0;
for(int x3669=0; x3669 < 50; x3669++) {
int32_t x3670 = x3667;
int32_t x3671 = x3668;
int32_t x3672 = x3666;
int32_t x3673 = x3672;
int32_t x3674 = x3670;
int32_t x3675 = x3671;
for(int x3676=0; x3676 < 50; x3676++) {
int32_t x3677 = x3673;
int32_t x3678 = x3674;
float x3679 = x3642[x3678];
int32_t x3680 = x3675;
float x3681 = x3656[x3680];
float x3682 = x3679 / x3681;
x3665[x3677] = x3682;
x3673 += 1;
x3674 += 1;
x3675 += 1;

}
x3666 += 50;
x3667 += 50;
x3668 += 50;

}
for(int x3694=0; x3694 < 2500; x3694++) {
float x3695 = x30[x3694];
float x3696 = x3665[x3694];
float x3697 = x3695 - x3696;
x30[x3694] = x3697;

}
for(int x3701=0; x3701 < 2500; x3701++) {
float x3702 = x39[x3701];
x39[x3701] = 0.0f;

}
for(int x3706=0; x3706 < 1300; x3706++) {
float x3707 = x25[x3706];
bool x3708 = x3707 > 5.0f;
if (x3708) {
x25[x3706] = 5.0f;
} else {
}
float x3712 = x25[x3706];
bool x3713 = x3712 < -5.0f;
if (x3713) {
x25[x3706] = -5.0f;
} else {
}

}
float* x3719 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x3720 = 0;
int32_t x3721 = 0;
int32_t x3722 = 0;
for(int x3723=0; x3723 < 26; x3723++) {
int32_t x3724 = x3721;
int32_t x3725 = x3722;
int32_t x3726 = x3720;
int32_t x3727 = x3726;
int32_t x3728 = x3724;
int32_t x3729 = x3725;
for(int x3730=0; x3730 < 50; x3730++) {
int32_t x3731 = x3727;
int32_t x3732 = x3728;
float x3733 = x25[x3732];
int32_t x3734 = x3729;
float x3735 = x25[x3734];
float x3736 = x3733 * x3735;
x3719[x3731] = x3736;
x3727 += 1;
x3728 += 1;
x3729 += 1;

}
x3720 += 50;
x3721 += 50;
x3722 += 50;

}
for(int x3748=0; x3748 < 1300; x3748++) {
float x3749 = x212[x3748];
float x3750 = x3719[x3748];
float x3751 = x3749 + x3750;
x212[x3748] = x3751;

}
float* x3755 = (float*)myMalloc(1300 * sizeof(float));;
for(int x3756=0; x3756 < 1300; x3756++) {
float x3757 = x25[x3756];
float x3758 = x3757 * 0.1f;
x3755[x3756] = x3758;

}
float* x3762 = (float*)myMalloc(1300 * sizeof(float));;
for(int x3763=0; x3763 < 1300; x3763++) {
float x3764 = x212[x3763];
float x3765 = x3764 + 1.0E-8f;
x3762[x3763] = x3765;

}
float* x3769 = (float*)myMalloc(1300 * sizeof(float));;
for(int x3770=0; x3770 < 1300; x3770++) {
float x3771 = x3762[x3770];
double x3772 = (double)x3771;
double x3773 = sqrt(x3772);
float x3774 = (float)x3773;
x3769[x3770] = x3774;

}
float* x3778 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x3779 = 0;
int32_t x3780 = 0;
int32_t x3781 = 0;
for(int x3782=0; x3782 < 26; x3782++) {
int32_t x3783 = x3780;
int32_t x3784 = x3781;
int32_t x3785 = x3779;
int32_t x3786 = x3785;
int32_t x3787 = x3783;
int32_t x3788 = x3784;
for(int x3789=0; x3789 < 50; x3789++) {
int32_t x3790 = x3786;
int32_t x3791 = x3787;
float x3792 = x3755[x3791];
int32_t x3793 = x3788;
float x3794 = x3769[x3793];
float x3795 = x3792 / x3794;
x3778[x3790] = x3795;
x3786 += 1;
x3787 += 1;
x3788 += 1;

}
x3779 += 50;
x3780 += 50;
x3781 += 50;

}
for(int x3807=0; x3807 < 1300; x3807++) {
float x3808 = x16[x3807];
float x3809 = x3778[x3807];
float x3810 = x3808 - x3809;
x16[x3807] = x3810;

}
for(int x3814=0; x3814 < 1300; x3814++) {
float x3815 = x25[x3814];
x25[x3814] = 0.0f;

}
for(int x3819=0; x3819 < 1300; x3819++) {
float x3820 = x99[x3819];
bool x3821 = x3820 > 5.0f;
if (x3821) {
x99[x3819] = 5.0f;
} else {
}
float x3825 = x99[x3819];
bool x3826 = x3825 < -5.0f;
if (x3826) {
x99[x3819] = -5.0f;
} else {
}

}
float* x3832 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x3833 = 0;
int32_t x3834 = 0;
int32_t x3835 = 0;
for(int x3836=0; x3836 < 26; x3836++) {
int32_t x3837 = x3834;
int32_t x3838 = x3835;
int32_t x3839 = x3833;
int32_t x3840 = x3839;
int32_t x3841 = x3837;
int32_t x3842 = x3838;
for(int x3843=0; x3843 < 50; x3843++) {
int32_t x3844 = x3840;
int32_t x3845 = x3841;
float x3846 = x99[x3845];
int32_t x3847 = x3842;
float x3848 = x99[x3847];
float x3849 = x3846 * x3848;
x3832[x3844] = x3849;
x3840 += 1;
x3841 += 1;
x3842 += 1;

}
x3833 += 50;
x3834 += 50;
x3835 += 50;

}
for(int x3861=0; x3861 < 1300; x3861++) {
float x3862 = x217[x3861];
float x3863 = x3832[x3861];
float x3864 = x3862 + x3863;
x217[x3861] = x3864;

}
float* x3868 = (float*)myMalloc(1300 * sizeof(float));;
for(int x3869=0; x3869 < 1300; x3869++) {
float x3870 = x99[x3869];
float x3871 = x3870 * 0.1f;
x3868[x3869] = x3871;

}
float* x3875 = (float*)myMalloc(1300 * sizeof(float));;
for(int x3876=0; x3876 < 1300; x3876++) {
float x3877 = x217[x3876];
float x3878 = x3877 + 1.0E-8f;
x3875[x3876] = x3878;

}
float* x3882 = (float*)myMalloc(1300 * sizeof(float));;
for(int x3883=0; x3883 < 1300; x3883++) {
float x3884 = x3875[x3883];
double x3885 = (double)x3884;
double x3886 = sqrt(x3885);
float x3887 = (float)x3886;
x3882[x3883] = x3887;

}
float* x3891 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x3892 = 0;
int32_t x3893 = 0;
int32_t x3894 = 0;
for(int x3895=0; x3895 < 26; x3895++) {
int32_t x3896 = x3893;
int32_t x3897 = x3894;
int32_t x3898 = x3892;
int32_t x3899 = x3898;
int32_t x3900 = x3896;
int32_t x3901 = x3897;
for(int x3902=0; x3902 < 50; x3902++) {
int32_t x3903 = x3899;
int32_t x3904 = x3900;
float x3905 = x3868[x3904];
int32_t x3906 = x3901;
float x3907 = x3882[x3906];
float x3908 = x3905 / x3907;
x3891[x3903] = x3908;
x3899 += 1;
x3900 += 1;
x3901 += 1;

}
x3892 += 50;
x3893 += 50;
x3894 += 50;

}
for(int x3920=0; x3920 < 1300; x3920++) {
float x3921 = x91[x3920];
float x3922 = x3891[x3920];
float x3923 = x3921 - x3922;
x91[x3920] = x3923;

}
for(int x3927=0; x3927 < 1300; x3927++) {
float x3928 = x99[x3927];
x99[x3927] = 0.0f;

}
for(int x3932=0; x3932 < 50; x3932++) {
float x3933 = x122[x3932];
bool x3934 = x3933 > 5.0f;
if (x3934) {
x122[x3932] = 5.0f;
} else {
}
float x3938 = x122[x3932];
bool x3939 = x3938 < -5.0f;
if (x3939) {
x122[x3932] = -5.0f;
} else {
}

}
float* x3945 = (float*)myMalloc(50 * sizeof(float));;
int32_t x3946 = 0;
int32_t x3947 = 0;
int32_t x3948 = 0;
for(int x3949=0; x3949 < 50; x3949++) {
int32_t x3950 = x3946;
int32_t x3951 = x3947;
float x3952 = x122[x3951];
int32_t x3953 = x3948;
float x3954 = x122[x3953];
float x3955 = x3952 * x3954;
x3945[x3950] = x3955;
x3946 += 1;
x3947 += 1;
x3948 += 1;

}
for(int x3962=0; x3962 < 50; x3962++) {
float x3963 = x222[x3962];
float x3964 = x3945[x3962];
float x3965 = x3963 + x3964;
x222[x3962] = x3965;

}
float* x3969 = (float*)myMalloc(50 * sizeof(float));;
for(int x3970=0; x3970 < 50; x3970++) {
float x3971 = x122[x3970];
float x3972 = x3971 * 0.1f;
x3969[x3970] = x3972;

}
float* x3976 = (float*)myMalloc(50 * sizeof(float));;
for(int x3977=0; x3977 < 50; x3977++) {
float x3978 = x222[x3977];
float x3979 = x3978 + 1.0E-8f;
x3976[x3977] = x3979;

}
float* x3983 = (float*)myMalloc(50 * sizeof(float));;
for(int x3984=0; x3984 < 50; x3984++) {
float x3985 = x3976[x3984];
double x3986 = (double)x3985;
double x3987 = sqrt(x3986);
float x3988 = (float)x3987;
x3983[x3984] = x3988;

}
float* x3992 = (float*)myMalloc(50 * sizeof(float));;
int32_t x3993 = 0;
int32_t x3994 = 0;
int32_t x3995 = 0;
for(int x3996=0; x3996 < 50; x3996++) {
int32_t x3997 = x3993;
int32_t x3998 = x3994;
float x3999 = x3969[x3998];
int32_t x4000 = x3995;
float x4001 = x3983[x4000];
float x4002 = x3999 / x4001;
x3992[x3997] = x4002;
x3993 += 1;
x3994 += 1;
x3995 += 1;

}
for(int x4009=0; x4009 < 50; x4009++) {
float x4010 = x117[x4009];
float x4011 = x3992[x4009];
float x4012 = x4010 - x4011;
x117[x4009] = x4012;

}
for(int x4016=0; x4016 < 50; x4016++) {
float x4017 = x122[x4016];
x122[x4016] = 0.0f;

}
for(int x4021=0; x4021 < 2500; x4021++) {
float x4022 = x112[x4021];
bool x4023 = x4022 > 5.0f;
if (x4023) {
x112[x4021] = 5.0f;
} else {
}
float x4027 = x112[x4021];
bool x4028 = x4027 < -5.0f;
if (x4028) {
x112[x4021] = -5.0f;
} else {
}

}
float* x4034 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x4035 = 0;
int32_t x4036 = 0;
int32_t x4037 = 0;
for(int x4038=0; x4038 < 50; x4038++) {
int32_t x4039 = x4036;
int32_t x4040 = x4037;
int32_t x4041 = x4035;
int32_t x4042 = x4041;
int32_t x4043 = x4039;
int32_t x4044 = x4040;
for(int x4045=0; x4045 < 50; x4045++) {
int32_t x4046 = x4042;
int32_t x4047 = x4043;
float x4048 = x112[x4047];
int32_t x4049 = x4044;
float x4050 = x112[x4049];
float x4051 = x4048 * x4050;
x4034[x4046] = x4051;
x4042 += 1;
x4043 += 1;
x4044 += 1;

}
x4035 += 50;
x4036 += 50;
x4037 += 50;

}
for(int x4063=0; x4063 < 2500; x4063++) {
float x4064 = x227[x4063];
float x4065 = x4034[x4063];
float x4066 = x4064 + x4065;
x227[x4063] = x4066;

}
float* x4070 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4071=0; x4071 < 2500; x4071++) {
float x4072 = x112[x4071];
float x4073 = x4072 * 0.1f;
x4070[x4071] = x4073;

}
float* x4077 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4078=0; x4078 < 2500; x4078++) {
float x4079 = x227[x4078];
float x4080 = x4079 + 1.0E-8f;
x4077[x4078] = x4080;

}
float* x4084 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4085=0; x4085 < 2500; x4085++) {
float x4086 = x4077[x4085];
double x4087 = (double)x4086;
double x4088 = sqrt(x4087);
float x4089 = (float)x4088;
x4084[x4085] = x4089;

}
float* x4093 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x4094 = 0;
int32_t x4095 = 0;
int32_t x4096 = 0;
for(int x4097=0; x4097 < 50; x4097++) {
int32_t x4098 = x4095;
int32_t x4099 = x4096;
int32_t x4100 = x4094;
int32_t x4101 = x4100;
int32_t x4102 = x4098;
int32_t x4103 = x4099;
for(int x4104=0; x4104 < 50; x4104++) {
int32_t x4105 = x4101;
int32_t x4106 = x4102;
float x4107 = x4070[x4106];
int32_t x4108 = x4103;
float x4109 = x4084[x4108];
float x4110 = x4107 / x4109;
x4093[x4105] = x4110;
x4101 += 1;
x4102 += 1;
x4103 += 1;

}
x4094 += 50;
x4095 += 50;
x4096 += 50;

}
for(int x4122=0; x4122 < 2500; x4122++) {
float x4123 = x104[x4122];
float x4124 = x4093[x4122];
float x4125 = x4123 - x4124;
x104[x4122] = x4125;

}
for(int x4129=0; x4129 < 2500; x4129++) {
float x4130 = x112[x4129];
x112[x4129] = 0.0f;

}
for(int x4134=0; x4134 < 26; x4134++) {
float x4135 = x182[x4134];
bool x4136 = x4135 > 5.0f;
if (x4136) {
x182[x4134] = 5.0f;
} else {
}
float x4140 = x182[x4134];
bool x4141 = x4140 < -5.0f;
if (x4141) {
x182[x4134] = -5.0f;
} else {
}

}
float* x4147 = (float*)myMalloc(26 * sizeof(float));;
int32_t x4148 = 0;
int32_t x4149 = 0;
int32_t x4150 = 0;
for(int x4151=0; x4151 < 26; x4151++) {
int32_t x4152 = x4148;
int32_t x4153 = x4149;
float x4154 = x182[x4153];
int32_t x4155 = x4150;
float x4156 = x182[x4155];
float x4157 = x4154 * x4156;
x4147[x4152] = x4157;
x4148 += 1;
x4149 += 1;
x4150 += 1;

}
for(int x4164=0; x4164 < 26; x4164++) {
float x4165 = x232[x4164];
float x4166 = x4147[x4164];
float x4167 = x4165 + x4166;
x232[x4164] = x4167;

}
float* x4171 = (float*)myMalloc(26 * sizeof(float));;
for(int x4172=0; x4172 < 26; x4172++) {
float x4173 = x182[x4172];
float x4174 = x4173 * 0.1f;
x4171[x4172] = x4174;

}
float* x4178 = (float*)myMalloc(26 * sizeof(float));;
for(int x4179=0; x4179 < 26; x4179++) {
float x4180 = x232[x4179];
float x4181 = x4180 + 1.0E-8f;
x4178[x4179] = x4181;

}
float* x4185 = (float*)myMalloc(26 * sizeof(float));;
for(int x4186=0; x4186 < 26; x4186++) {
float x4187 = x4178[x4186];
double x4188 = (double)x4187;
double x4189 = sqrt(x4188);
float x4190 = (float)x4189;
x4185[x4186] = x4190;

}
float* x4194 = (float*)myMalloc(26 * sizeof(float));;
int32_t x4195 = 0;
int32_t x4196 = 0;
int32_t x4197 = 0;
for(int x4198=0; x4198 < 26; x4198++) {
int32_t x4199 = x4195;
int32_t x4200 = x4196;
float x4201 = x4171[x4200];
int32_t x4202 = x4197;
float x4203 = x4185[x4202];
float x4204 = x4201 / x4203;
x4194[x4199] = x4204;
x4195 += 1;
x4196 += 1;
x4197 += 1;

}
for(int x4211=0; x4211 < 26; x4211++) {
float x4212 = x176[x4211];
float x4213 = x4194[x4211];
float x4214 = x4212 - x4213;
x176[x4211] = x4214;

}
for(int x4218=0; x4218 < 26; x4218++) {
float x4219 = x182[x4218];
x182[x4218] = 0.0f;

}
for(int x4223=0; x4223 < 1300; x4223++) {
float x4224 = x171[x4223];
bool x4225 = x4224 > 5.0f;
if (x4225) {
x171[x4223] = 5.0f;
} else {
}
float x4229 = x171[x4223];
bool x4230 = x4229 < -5.0f;
if (x4230) {
x171[x4223] = -5.0f;
} else {
}

}
float* x4236 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4237 = 0;
int32_t x4238 = 0;
int32_t x4239 = 0;
for(int x4240=0; x4240 < 50; x4240++) {
int32_t x4241 = x4238;
int32_t x4242 = x4239;
int32_t x4243 = x4237;
int32_t x4244 = x4243;
int32_t x4245 = x4241;
int32_t x4246 = x4242;
for(int x4247=0; x4247 < 26; x4247++) {
int32_t x4248 = x4244;
int32_t x4249 = x4245;
float x4250 = x171[x4249];
int32_t x4251 = x4246;
float x4252 = x171[x4251];
float x4253 = x4250 * x4252;
x4236[x4248] = x4253;
x4244 += 1;
x4245 += 1;
x4246 += 1;

}
x4237 += 26;
x4238 += 26;
x4239 += 26;

}
for(int x4265=0; x4265 < 1300; x4265++) {
float x4266 = x237[x4265];
float x4267 = x4236[x4265];
float x4268 = x4266 + x4267;
x237[x4265] = x4268;

}
float* x4272 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4273=0; x4273 < 1300; x4273++) {
float x4274 = x171[x4273];
float x4275 = x4274 * 0.1f;
x4272[x4273] = x4275;

}
float* x4279 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4280=0; x4280 < 1300; x4280++) {
float x4281 = x237[x4280];
float x4282 = x4281 + 1.0E-8f;
x4279[x4280] = x4282;

}
float* x4286 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4287=0; x4287 < 1300; x4287++) {
float x4288 = x4279[x4287];
double x4289 = (double)x4288;
double x4290 = sqrt(x4289);
float x4291 = (float)x4290;
x4286[x4287] = x4291;

}
float* x4295 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4296 = 0;
int32_t x4297 = 0;
int32_t x4298 = 0;
for(int x4299=0; x4299 < 50; x4299++) {
int32_t x4300 = x4297;
int32_t x4301 = x4298;
int32_t x4302 = x4296;
int32_t x4303 = x4302;
int32_t x4304 = x4300;
int32_t x4305 = x4301;
for(int x4306=0; x4306 < 26; x4306++) {
int32_t x4307 = x4303;
int32_t x4308 = x4304;
float x4309 = x4272[x4308];
int32_t x4310 = x4305;
float x4311 = x4286[x4310];
float x4312 = x4309 / x4311;
x4295[x4307] = x4312;
x4303 += 1;
x4304 += 1;
x4305 += 1;

}
x4296 += 26;
x4297 += 26;
x4298 += 26;

}
for(int x4324=0; x4324 < 1300; x4324++) {
float x4325 = x163[x4324];
float x4326 = x4295[x4324];
float x4327 = x4325 - x4326;
x163[x4324] = x4327;

}
for(int x4331=0; x4331 < 1300; x4331++) {
float x4332 = x171[x4331];
x171[x4331] = 0.0f;

}
for(int x4336=0; x4336 < 2500; x4336++) {
float x4337 = x148[x4336];
bool x4338 = x4337 > 5.0f;
if (x4338) {
x148[x4336] = 5.0f;
} else {
}
float x4342 = x148[x4336];
bool x4343 = x4342 < -5.0f;
if (x4343) {
x148[x4336] = -5.0f;
} else {
}

}
float* x4349 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x4350 = 0;
int32_t x4351 = 0;
int32_t x4352 = 0;
for(int x4353=0; x4353 < 50; x4353++) {
int32_t x4354 = x4351;
int32_t x4355 = x4352;
int32_t x4356 = x4350;
int32_t x4357 = x4356;
int32_t x4358 = x4354;
int32_t x4359 = x4355;
for(int x4360=0; x4360 < 50; x4360++) {
int32_t x4361 = x4357;
int32_t x4362 = x4358;
float x4363 = x148[x4362];
int32_t x4364 = x4359;
float x4365 = x148[x4364];
float x4366 = x4363 * x4365;
x4349[x4361] = x4366;
x4357 += 1;
x4358 += 1;
x4359 += 1;

}
x4350 += 50;
x4351 += 50;
x4352 += 50;

}
for(int x4378=0; x4378 < 2500; x4378++) {
float x4379 = x242[x4378];
float x4380 = x4349[x4378];
float x4381 = x4379 + x4380;
x242[x4378] = x4381;

}
float* x4385 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4386=0; x4386 < 2500; x4386++) {
float x4387 = x148[x4386];
float x4388 = x4387 * 0.1f;
x4385[x4386] = x4388;

}
float* x4392 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4393=0; x4393 < 2500; x4393++) {
float x4394 = x242[x4393];
float x4395 = x4394 + 1.0E-8f;
x4392[x4393] = x4395;

}
float* x4399 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4400=0; x4400 < 2500; x4400++) {
float x4401 = x4392[x4400];
double x4402 = (double)x4401;
double x4403 = sqrt(x4402);
float x4404 = (float)x4403;
x4399[x4400] = x4404;

}
float* x4408 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x4409 = 0;
int32_t x4410 = 0;
int32_t x4411 = 0;
for(int x4412=0; x4412 < 50; x4412++) {
int32_t x4413 = x4410;
int32_t x4414 = x4411;
int32_t x4415 = x4409;
int32_t x4416 = x4415;
int32_t x4417 = x4413;
int32_t x4418 = x4414;
for(int x4419=0; x4419 < 50; x4419++) {
int32_t x4420 = x4416;
int32_t x4421 = x4417;
float x4422 = x4385[x4421];
int32_t x4423 = x4418;
float x4424 = x4399[x4423];
float x4425 = x4422 / x4424;
x4408[x4420] = x4425;
x4416 += 1;
x4417 += 1;
x4418 += 1;

}
x4409 += 50;
x4410 += 50;
x4411 += 50;

}
for(int x4437=0; x4437 < 2500; x4437++) {
float x4438 = x140[x4437];
float x4439 = x4408[x4437];
float x4440 = x4438 - x4439;
x140[x4437] = x4440;

}
for(int x4444=0; x4444 < 2500; x4444++) {
float x4445 = x148[x4444];
x148[x4444] = 0.0f;

}
for(int x4449=0; x4449 < 1300; x4449++) {
float x4450 = x135[x4449];
bool x4451 = x4450 > 5.0f;
if (x4451) {
x135[x4449] = 5.0f;
} else {
}
float x4455 = x135[x4449];
bool x4456 = x4455 < -5.0f;
if (x4456) {
x135[x4449] = -5.0f;
} else {
}

}
float* x4462 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4463 = 0;
int32_t x4464 = 0;
int32_t x4465 = 0;
for(int x4466=0; x4466 < 26; x4466++) {
int32_t x4467 = x4464;
int32_t x4468 = x4465;
int32_t x4469 = x4463;
int32_t x4470 = x4469;
int32_t x4471 = x4467;
int32_t x4472 = x4468;
for(int x4473=0; x4473 < 50; x4473++) {
int32_t x4474 = x4470;
int32_t x4475 = x4471;
float x4476 = x135[x4475];
int32_t x4477 = x4472;
float x4478 = x135[x4477];
float x4479 = x4476 * x4478;
x4462[x4474] = x4479;
x4470 += 1;
x4471 += 1;
x4472 += 1;

}
x4463 += 50;
x4464 += 50;
x4465 += 50;

}
for(int x4491=0; x4491 < 1300; x4491++) {
float x4492 = x247[x4491];
float x4493 = x4462[x4491];
float x4494 = x4492 + x4493;
x247[x4491] = x4494;

}
float* x4498 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4499=0; x4499 < 1300; x4499++) {
float x4500 = x135[x4499];
float x4501 = x4500 * 0.1f;
x4498[x4499] = x4501;

}
float* x4505 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4506=0; x4506 < 1300; x4506++) {
float x4507 = x247[x4506];
float x4508 = x4507 + 1.0E-8f;
x4505[x4506] = x4508;

}
float* x4512 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4513=0; x4513 < 1300; x4513++) {
float x4514 = x4505[x4513];
double x4515 = (double)x4514;
double x4516 = sqrt(x4515);
float x4517 = (float)x4516;
x4512[x4513] = x4517;

}
float* x4521 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4522 = 0;
int32_t x4523 = 0;
int32_t x4524 = 0;
for(int x4525=0; x4525 < 26; x4525++) {
int32_t x4526 = x4523;
int32_t x4527 = x4524;
int32_t x4528 = x4522;
int32_t x4529 = x4528;
int32_t x4530 = x4526;
int32_t x4531 = x4527;
for(int x4532=0; x4532 < 50; x4532++) {
int32_t x4533 = x4529;
int32_t x4534 = x4530;
float x4535 = x4498[x4534];
int32_t x4536 = x4531;
float x4537 = x4512[x4536];
float x4538 = x4535 / x4537;
x4521[x4533] = x4538;
x4529 += 1;
x4530 += 1;
x4531 += 1;

}
x4522 += 50;
x4523 += 50;
x4524 += 50;

}
for(int x4550=0; x4550 < 1300; x4550++) {
float x4551 = x127[x4550];
float x4552 = x4521[x4550];
float x4553 = x4551 - x4552;
x127[x4550] = x4553;

}
for(int x4557=0; x4557 < 1300; x4557++) {
float x4558 = x135[x4557];
x135[x4557] = 0.0f;

}
for(int x4562=0; x4562 < 50; x4562++) {
float x4563 = x158[x4562];
bool x4564 = x4563 > 5.0f;
if (x4564) {
x158[x4562] = 5.0f;
} else {
}
float x4568 = x158[x4562];
bool x4569 = x4568 < -5.0f;
if (x4569) {
x158[x4562] = -5.0f;
} else {
}

}
float* x4575 = (float*)myMalloc(50 * sizeof(float));;
int32_t x4576 = 0;
int32_t x4577 = 0;
int32_t x4578 = 0;
for(int x4579=0; x4579 < 50; x4579++) {
int32_t x4580 = x4576;
int32_t x4581 = x4577;
float x4582 = x158[x4581];
int32_t x4583 = x4578;
float x4584 = x158[x4583];
float x4585 = x4582 * x4584;
x4575[x4580] = x4585;
x4576 += 1;
x4577 += 1;
x4578 += 1;

}
for(int x4592=0; x4592 < 50; x4592++) {
float x4593 = x252[x4592];
float x4594 = x4575[x4592];
float x4595 = x4593 + x4594;
x252[x4592] = x4595;

}
float* x4599 = (float*)myMalloc(50 * sizeof(float));;
for(int x4600=0; x4600 < 50; x4600++) {
float x4601 = x158[x4600];
float x4602 = x4601 * 0.1f;
x4599[x4600] = x4602;

}
float* x4606 = (float*)myMalloc(50 * sizeof(float));;
for(int x4607=0; x4607 < 50; x4607++) {
float x4608 = x252[x4607];
float x4609 = x4608 + 1.0E-8f;
x4606[x4607] = x4609;

}
float* x4613 = (float*)myMalloc(50 * sizeof(float));;
for(int x4614=0; x4614 < 50; x4614++) {
float x4615 = x4606[x4614];
double x4616 = (double)x4615;
double x4617 = sqrt(x4616);
float x4618 = (float)x4617;
x4613[x4614] = x4618;

}
float* x4622 = (float*)myMalloc(50 * sizeof(float));;
int32_t x4623 = 0;
int32_t x4624 = 0;
int32_t x4625 = 0;
for(int x4626=0; x4626 < 50; x4626++) {
int32_t x4627 = x4623;
int32_t x4628 = x4624;
float x4629 = x4599[x4628];
int32_t x4630 = x4625;
float x4631 = x4613[x4630];
float x4632 = x4629 / x4631;
x4622[x4627] = x4632;
x4623 += 1;
x4624 += 1;
x4625 += 1;

}
for(int x4639=0; x4639 < 50; x4639++) {
float x4640 = x153[x4639];
float x4641 = x4622[x4639];
float x4642 = x4640 - x4641;
x153[x4639] = x4642;

}
for(int x4646=0; x4646 < 50; x4646++) {
float x4647 = x158[x4646];
x158[x4646] = 0.0f;

}
mallocAddr = (void*)x259;

}
double x4654 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x4657 = (long)fopen(x0, "w");
fprintf((FILE *)x4657, "unit: %s\n", "100 iteration");
for(int x4660=0; x4660 < 51; x4660++) {
double x4661 = x258[x4660];
fprintf((FILE *)x4657, "%lf\n", x4661);

}
double x4655 = x257 - x2;
double x4656 = x4654 - x257;
fprintf((FILE *)x4657, "run time: %lf %lf\n", x4655, x4656);
fclose((FILE*)x4657);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

