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
float* x202 = (float*)myMalloc(1300 * sizeof(float));;
for(int x203=0; x203 < 1300; x203++) {
x202[x203] = 0.0f;

}
float* x207 = (float*)myMalloc(50 * sizeof(float));;
for(int x208=0; x208 < 50; x208++) {
x207[x208] = 0.0f;

}
float* x212 = (float*)myMalloc(2500 * sizeof(float));;
for(int x213=0; x213 < 2500; x213++) {
x212[x213] = 0.0f;

}
float* x217 = (float*)myMalloc(26 * sizeof(float));;
for(int x218=0; x218 < 26; x218++) {
x217[x218] = 0.0f;

}
float* x222 = (float*)myMalloc(1300 * sizeof(float));;
for(int x223=0; x223 < 1300; x223++) {
x222[x223] = 0.0f;

}
float* x227 = (float*)myMalloc(1300 * sizeof(float));;
for(int x228=0; x228 < 1300; x228++) {
x227[x228] = 0.0f;

}
float* x232 = (float*)myMalloc(50 * sizeof(float));;
for(int x233=0; x233 < 50; x233++) {
x232[x233] = 0.0f;

}
float* x237 = (float*)myMalloc(2500 * sizeof(float));;
for(int x238=0; x238 < 2500; x238++) {
x237[x238] = 0.0f;

}
float* x242 = (float*)myMalloc(1300 * sizeof(float));;
for(int x243=0; x243 < 1300; x243++) {
x242[x243] = 0.0f;

}
float* x247 = (float*)myMalloc(50 * sizeof(float));;
for(int x248=0; x248 < 50; x248++) {
x247[x248] = 0.0f;

}
float* x252 = (float*)myMalloc(2500 * sizeof(float));;
for(int x253=0; x253 < 2500; x253++) {
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
function<void(int32_t,float**)> x1293 = [&](int32_t x1294,float** x1295) {
float** x1297 = x1295;
float* x1298 = x1297[0];
float* x1299 = x1297[1];
float* x1300 = x1297[2];
float* x1301 = x1297[3];
float* x1302 = x1297[4];
float* x1303 = x1297[5];
int32_t x1296 = x1294;
bool x1304 = x1296 < 20;
if (x1304) {
int32_t x1305 = x1296 * 520;
float* x1306 = x302+x1305;
float* x1307 = x324+x1305;
// dot: WrappedArray(20, 26), WrappedArray(26, 50)
float* x1309 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1310=0; x1310 < 20; x1310++) {
int32_t x1314 = x1310 * 26;
int32_t x1324 = x1310 * 50;
for(int x1311=0; x1311 < 50; x1311++) {
float x1312 = 0.0f;
for(int x1313=0; x1313 < 26; x1313++) {
int32_t x1315 = x1314 + x1313;
float x1316 = x1306[x1315];
int32_t x1317 = x1313 * 50;
int32_t x1318 = x1317 + x1311;
float x1319 = x16[x1318];
float x1320 = x1316 * x1319;
x1312 += x1320;

}
float x1326 = x1312;
int32_t x1325 = x1324 + x1311;
x1309[x1325] = x1326;

}

}
float* x1332 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1333=0; x1333 < 1000; x1333++) {
x1332[x1333] = 0.0f;

}
// dot: List(20, 50), WrappedArray(50, 50)
float* x1338 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1339=0; x1339 < 20; x1339++) {
int32_t x1343 = x1339 * 50;
for(int x1340=0; x1340 < 50; x1340++) {
float x1341 = 0.0f;
for(int x1342=0; x1342 < 50; x1342++) {
int32_t x1344 = x1343 + x1342;
float x1345 = x1300[x1344];
int32_t x1346 = x1342 * 50;
int32_t x1347 = x1346 + x1340;
float x1348 = x30[x1347];
float x1349 = x1345 * x1348;
x1341 += x1349;

}
float x1354 = x1341;
int32_t x1353 = x1343 + x1340;
x1338[x1353] = x1354;

}

}
float* x1360 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1361=0; x1361 < 1000; x1361++) {
x1360[x1361] = 0.0f;

}
float* x1365 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1366 = 0;
int32_t x1367 = 0;
int32_t x1368 = 0;
for(int x1369=0; x1369 < 20; x1369++) {
int32_t x1370 = x1367;
int32_t x1371 = x1368;
int32_t x1372 = x1366;
int32_t x1373 = x1372;
int32_t x1374 = x1370;
int32_t x1375 = x1371;
for(int x1376=0; x1376 < 50; x1376++) {
int32_t x1377 = x1373;
int32_t x1378 = x1374;
float x1379 = x1309[x1378];
int32_t x1380 = x1375;
float x1381 = x1338[x1380];
float x1382 = x1379 + x1381;
x1365[x1377] = x1382;
x1373 += 1;
x1374 += 1;
x1375 += 1;

}
x1366 += 50;
x1367 += 50;
x1368 += 50;

}
float* x1394 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1395=0; x1395 < 1000; x1395++) {
x1394[x1395] = 0.0f;

}
float* x1399 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1400 = 0;
int32_t x1401 = 0;
int32_t x1402 = 0;
for(int x1403=0; x1403 < 20; x1403++) {
int32_t x1404 = x1401;
int32_t x1405 = x1402;
int32_t x1406 = x1400;
int32_t x1407 = x1406;
int32_t x1408 = x1404;
int32_t x1409 = x1405;
for(int x1410=0; x1410 < 50; x1410++) {
int32_t x1411 = x1407;
int32_t x1412 = x1408;
float x1413 = x1365[x1412];
int32_t x1414 = x1409;
float x1415 = x44[x1414];
float x1416 = x1413 + x1415;
x1399[x1411] = x1416;
x1407 += 1;
x1408 += 1;
x1409 += 1;

}
x1400 += 50;
x1401 += 50;

}
float* x1427 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1428=0; x1428 < 1000; x1428++) {
x1427[x1428] = 0.0f;

}
float* x1432 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1433=0; x1433 < 1000; x1433++) {
float x1434 = x1399[x1433];
float x1435 = -1.0f * x1434;
double x1436 = (double)x1435;
double x1437 = exp(x1436);
float x1438 = (float)x1437;
float x1439 = x1438 + 1.0f;
float x1440 = 1.0f / x1439;
x1432[x1433] = x1440;

}
float* x1444 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1445=0; x1445 < 1000; x1445++) {
x1444[x1445] = 0.0f;

}
// dot: WrappedArray(20, 26), WrappedArray(26, 50)
float* x1450 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1451=0; x1451 < 20; x1451++) {
int32_t x1455 = x1451 * 26;
int32_t x1465 = x1451 * 50;
for(int x1452=0; x1452 < 50; x1452++) {
float x1453 = 0.0f;
for(int x1454=0; x1454 < 26; x1454++) {
int32_t x1456 = x1455 + x1454;
float x1457 = x1306[x1456];
int32_t x1458 = x1454 * 50;
int32_t x1459 = x1458 + x1452;
float x1460 = x55[x1459];
float x1461 = x1457 * x1460;
x1453 += x1461;

}
float x1467 = x1453;
int32_t x1466 = x1465 + x1452;
x1450[x1466] = x1467;

}

}
float* x1473 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1474=0; x1474 < 1000; x1474++) {
x1473[x1474] = 0.0f;

}
// dot: List(20, 50), WrappedArray(50, 50)
float* x1479 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1480=0; x1480 < 20; x1480++) {
int32_t x1484 = x1480 * 50;
for(int x1481=0; x1481 < 50; x1481++) {
float x1482 = 0.0f;
for(int x1483=0; x1483 < 50; x1483++) {
int32_t x1485 = x1484 + x1483;
float x1486 = x1300[x1485];
int32_t x1487 = x1483 * 50;
int32_t x1488 = x1487 + x1481;
float x1489 = x68[x1488];
float x1490 = x1486 * x1489;
x1482 += x1490;

}
float x1495 = x1482;
int32_t x1494 = x1484 + x1481;
x1479[x1494] = x1495;

}

}
float* x1501 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1502=0; x1502 < 1000; x1502++) {
x1501[x1502] = 0.0f;

}
float* x1506 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1507 = 0;
int32_t x1508 = 0;
int32_t x1509 = 0;
for(int x1510=0; x1510 < 20; x1510++) {
int32_t x1511 = x1508;
int32_t x1512 = x1509;
int32_t x1513 = x1507;
int32_t x1514 = x1513;
int32_t x1515 = x1511;
int32_t x1516 = x1512;
for(int x1517=0; x1517 < 50; x1517++) {
int32_t x1518 = x1514;
int32_t x1519 = x1515;
float x1520 = x1450[x1519];
int32_t x1521 = x1516;
float x1522 = x1479[x1521];
float x1523 = x1520 + x1522;
x1506[x1518] = x1523;
x1514 += 1;
x1515 += 1;
x1516 += 1;

}
x1507 += 50;
x1508 += 50;
x1509 += 50;

}
float* x1535 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1536=0; x1536 < 1000; x1536++) {
x1535[x1536] = 0.0f;

}
float* x1540 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1541 = 0;
int32_t x1542 = 0;
int32_t x1543 = 0;
for(int x1544=0; x1544 < 20; x1544++) {
int32_t x1545 = x1542;
int32_t x1546 = x1543;
int32_t x1547 = x1541;
int32_t x1548 = x1547;
int32_t x1549 = x1545;
int32_t x1550 = x1546;
for(int x1551=0; x1551 < 50; x1551++) {
int32_t x1552 = x1548;
int32_t x1553 = x1549;
float x1554 = x1506[x1553];
int32_t x1555 = x1550;
float x1556 = x81[x1555];
float x1557 = x1554 + x1556;
x1540[x1552] = x1557;
x1548 += 1;
x1549 += 1;
x1550 += 1;

}
x1541 += 50;
x1542 += 50;

}
float* x1568 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1569=0; x1569 < 1000; x1569++) {
x1568[x1569] = 0.0f;

}
float* x1573 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1574=0; x1574 < 1000; x1574++) {
float x1575 = x1540[x1574];
float x1576 = -1.0f * x1575;
double x1577 = (double)x1576;
double x1578 = exp(x1577);
float x1579 = (float)x1578;
float x1580 = x1579 + 1.0f;
float x1581 = 1.0f / x1580;
x1573[x1574] = x1581;

}
float* x1585 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1586=0; x1586 < 1000; x1586++) {
x1585[x1586] = 0.0f;

}
// dot: WrappedArray(20, 26), WrappedArray(26, 50)
float* x1591 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1592=0; x1592 < 20; x1592++) {
int32_t x1596 = x1592 * 26;
int32_t x1606 = x1592 * 50;
for(int x1593=0; x1593 < 50; x1593++) {
float x1594 = 0.0f;
for(int x1595=0; x1595 < 26; x1595++) {
int32_t x1597 = x1596 + x1595;
float x1598 = x1306[x1597];
int32_t x1599 = x1595 * 50;
int32_t x1600 = x1599 + x1593;
float x1601 = x127[x1600];
float x1602 = x1598 * x1601;
x1594 += x1602;

}
float x1608 = x1594;
int32_t x1607 = x1606 + x1593;
x1591[x1607] = x1608;

}

}
float* x1614 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1615=0; x1615 < 1000; x1615++) {
x1614[x1615] = 0.0f;

}
// dot: List(20, 50), WrappedArray(50, 50)
float* x1620 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1621=0; x1621 < 20; x1621++) {
int32_t x1625 = x1621 * 50;
for(int x1622=0; x1622 < 50; x1622++) {
float x1623 = 0.0f;
for(int x1624=0; x1624 < 50; x1624++) {
int32_t x1626 = x1625 + x1624;
float x1627 = x1300[x1626];
int32_t x1628 = x1624 * 50;
int32_t x1629 = x1628 + x1622;
float x1630 = x140[x1629];
float x1631 = x1627 * x1630;
x1623 += x1631;

}
float x1636 = x1623;
int32_t x1635 = x1625 + x1622;
x1620[x1635] = x1636;

}

}
float* x1642 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1643=0; x1643 < 1000; x1643++) {
x1642[x1643] = 0.0f;

}
float* x1647 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1648 = 0;
int32_t x1649 = 0;
int32_t x1650 = 0;
for(int x1651=0; x1651 < 20; x1651++) {
int32_t x1652 = x1649;
int32_t x1653 = x1650;
int32_t x1654 = x1648;
int32_t x1655 = x1654;
int32_t x1656 = x1652;
int32_t x1657 = x1653;
for(int x1658=0; x1658 < 50; x1658++) {
int32_t x1659 = x1655;
int32_t x1660 = x1656;
float x1661 = x1591[x1660];
int32_t x1662 = x1657;
float x1663 = x1620[x1662];
float x1664 = x1661 + x1663;
x1647[x1659] = x1664;
x1655 += 1;
x1656 += 1;
x1657 += 1;

}
x1648 += 50;
x1649 += 50;
x1650 += 50;

}
float* x1676 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1677=0; x1677 < 1000; x1677++) {
x1676[x1677] = 0.0f;

}
float* x1681 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1682 = 0;
int32_t x1683 = 0;
int32_t x1684 = 0;
for(int x1685=0; x1685 < 20; x1685++) {
int32_t x1686 = x1683;
int32_t x1687 = x1684;
int32_t x1688 = x1682;
int32_t x1689 = x1688;
int32_t x1690 = x1686;
int32_t x1691 = x1687;
for(int x1692=0; x1692 < 50; x1692++) {
int32_t x1693 = x1689;
int32_t x1694 = x1690;
float x1695 = x1647[x1694];
int32_t x1696 = x1691;
float x1697 = x153[x1696];
float x1698 = x1695 + x1697;
x1681[x1693] = x1698;
x1689 += 1;
x1690 += 1;
x1691 += 1;

}
x1682 += 50;
x1683 += 50;

}
float* x1709 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1710=0; x1710 < 1000; x1710++) {
x1709[x1710] = 0.0f;

}
float* x1714 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1715=0; x1715 < 1000; x1715++) {
float x1716 = x1681[x1715];
float x1717 = -1.0f * x1716;
double x1718 = (double)x1717;
double x1719 = exp(x1718);
float x1720 = (float)x1719;
float x1721 = x1720 + 1.0f;
float x1722 = 1.0f / x1721;
x1714[x1715] = x1722;

}
float* x1726 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1727=0; x1727 < 1000; x1727++) {
x1726[x1727] = 0.0f;

}
// dot: WrappedArray(20, 26), WrappedArray(26, 50)
float* x1732 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1733=0; x1733 < 20; x1733++) {
int32_t x1737 = x1733 * 26;
int32_t x1747 = x1733 * 50;
for(int x1734=0; x1734 < 50; x1734++) {
float x1735 = 0.0f;
for(int x1736=0; x1736 < 26; x1736++) {
int32_t x1738 = x1737 + x1736;
float x1739 = x1306[x1738];
int32_t x1740 = x1736 * 50;
int32_t x1741 = x1740 + x1734;
float x1742 = x91[x1741];
float x1743 = x1739 * x1742;
x1735 += x1743;

}
float x1749 = x1735;
int32_t x1748 = x1747 + x1734;
x1732[x1748] = x1749;

}

}
float* x1755 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1756=0; x1756 < 1000; x1756++) {
x1755[x1756] = 0.0f;

}
// dot: List(20, 50), WrappedArray(50, 50)
float* x1761 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1762=0; x1762 < 20; x1762++) {
int32_t x1766 = x1762 * 50;
for(int x1763=0; x1763 < 50; x1763++) {
float x1764 = 0.0f;
for(int x1765=0; x1765 < 50; x1765++) {
int32_t x1767 = x1766 + x1765;
float x1768 = x1300[x1767];
int32_t x1769 = x1765 * 50;
int32_t x1770 = x1769 + x1763;
float x1771 = x104[x1770];
float x1772 = x1768 * x1771;
x1764 += x1772;

}
float x1777 = x1764;
int32_t x1776 = x1766 + x1763;
x1761[x1776] = x1777;

}

}
float* x1783 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1784=0; x1784 < 1000; x1784++) {
x1783[x1784] = 0.0f;

}
float* x1788 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1789 = 0;
int32_t x1790 = 0;
int32_t x1791 = 0;
for(int x1792=0; x1792 < 20; x1792++) {
int32_t x1793 = x1790;
int32_t x1794 = x1791;
int32_t x1795 = x1789;
int32_t x1796 = x1795;
int32_t x1797 = x1793;
int32_t x1798 = x1794;
for(int x1799=0; x1799 < 50; x1799++) {
int32_t x1800 = x1796;
int32_t x1801 = x1797;
float x1802 = x1732[x1801];
int32_t x1803 = x1798;
float x1804 = x1761[x1803];
float x1805 = x1802 + x1804;
x1788[x1800] = x1805;
x1796 += 1;
x1797 += 1;
x1798 += 1;

}
x1789 += 50;
x1790 += 50;
x1791 += 50;

}
float* x1817 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1818=0; x1818 < 1000; x1818++) {
x1817[x1818] = 0.0f;

}
float* x1822 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1823 = 0;
int32_t x1824 = 0;
int32_t x1825 = 0;
for(int x1826=0; x1826 < 20; x1826++) {
int32_t x1827 = x1824;
int32_t x1828 = x1825;
int32_t x1829 = x1823;
int32_t x1830 = x1829;
int32_t x1831 = x1827;
int32_t x1832 = x1828;
for(int x1833=0; x1833 < 50; x1833++) {
int32_t x1834 = x1830;
int32_t x1835 = x1831;
float x1836 = x1788[x1835];
int32_t x1837 = x1832;
float x1838 = x117[x1837];
float x1839 = x1836 + x1838;
x1822[x1834] = x1839;
x1830 += 1;
x1831 += 1;
x1832 += 1;

}
x1823 += 50;
x1824 += 50;

}
float* x1850 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1851=0; x1851 < 1000; x1851++) {
x1850[x1851] = 0.0f;

}
float* x1855 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1856=0; x1856 < 1000; x1856++) {
float x1857 = x1822[x1856];
double x1858 = (double)x1857;
double x1859 = tanh(x1858);
float x1860 = (float)x1859;
x1855[x1856] = x1860;

}
float* x1864 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1865=0; x1865 < 1000; x1865++) {
x1864[x1865] = 0.0f;

}
float* x1869 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1870 = 0;
int32_t x1871 = 0;
int32_t x1872 = 0;
for(int x1873=0; x1873 < 20; x1873++) {
int32_t x1874 = x1871;
int32_t x1875 = x1872;
int32_t x1876 = x1870;
int32_t x1877 = x1876;
int32_t x1878 = x1874;
int32_t x1879 = x1875;
for(int x1880=0; x1880 < 50; x1880++) {
int32_t x1881 = x1877;
int32_t x1882 = x1878;
float x1883 = x1432[x1882];
int32_t x1884 = x1879;
float x1885 = x1302[x1884];
float x1886 = x1883 * x1885;
x1869[x1881] = x1886;
x1877 += 1;
x1878 += 1;
x1879 += 1;

}
x1870 += 50;
x1871 += 50;
x1872 += 50;

}
float* x1898 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1899=0; x1899 < 1000; x1899++) {
x1898[x1899] = 0.0f;

}
float* x1903 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1904 = 0;
int32_t x1905 = 0;
int32_t x1906 = 0;
for(int x1907=0; x1907 < 20; x1907++) {
int32_t x1908 = x1905;
int32_t x1909 = x1906;
int32_t x1910 = x1904;
int32_t x1911 = x1910;
int32_t x1912 = x1908;
int32_t x1913 = x1909;
for(int x1914=0; x1914 < 50; x1914++) {
int32_t x1915 = x1911;
int32_t x1916 = x1912;
float x1917 = x1573[x1916];
int32_t x1918 = x1913;
float x1919 = x1855[x1918];
float x1920 = x1917 * x1919;
x1903[x1915] = x1920;
x1911 += 1;
x1912 += 1;
x1913 += 1;

}
x1904 += 50;
x1905 += 50;
x1906 += 50;

}
float* x1932 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1933=0; x1933 < 1000; x1933++) {
x1932[x1933] = 0.0f;

}
float* x1937 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1938 = 0;
int32_t x1939 = 0;
int32_t x1940 = 0;
for(int x1941=0; x1941 < 20; x1941++) {
int32_t x1942 = x1939;
int32_t x1943 = x1940;
int32_t x1944 = x1938;
int32_t x1945 = x1944;
int32_t x1946 = x1942;
int32_t x1947 = x1943;
for(int x1948=0; x1948 < 50; x1948++) {
int32_t x1949 = x1945;
int32_t x1950 = x1946;
float x1951 = x1869[x1950];
int32_t x1952 = x1947;
float x1953 = x1903[x1952];
float x1954 = x1951 + x1953;
x1937[x1949] = x1954;
x1945 += 1;
x1946 += 1;
x1947 += 1;

}
x1938 += 50;
x1939 += 50;
x1940 += 50;

}
float* x1966 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1967=0; x1967 < 1000; x1967++) {
x1966[x1967] = 0.0f;

}
float* x1971 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1972=0; x1972 < 1000; x1972++) {
float x1973 = x1937[x1972];
double x1974 = (double)x1973;
double x1975 = tanh(x1974);
float x1976 = (float)x1975;
x1971[x1972] = x1976;

}
float* x1980 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1981=0; x1981 < 1000; x1981++) {
x1980[x1981] = 0.0f;

}
float* x1985 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1986 = 0;
int32_t x1987 = 0;
int32_t x1988 = 0;
for(int x1989=0; x1989 < 20; x1989++) {
int32_t x1990 = x1987;
int32_t x1991 = x1988;
int32_t x1992 = x1986;
int32_t x1993 = x1992;
int32_t x1994 = x1990;
int32_t x1995 = x1991;
for(int x1996=0; x1996 < 50; x1996++) {
int32_t x1997 = x1993;
int32_t x1998 = x1994;
float x1999 = x1714[x1998];
int32_t x2000 = x1995;
float x2001 = x1971[x2000];
float x2002 = x1999 * x2001;
x1985[x1997] = x2002;
x1993 += 1;
x1994 += 1;
x1995 += 1;

}
x1986 += 50;
x1987 += 50;
x1988 += 50;

}
float* x2014 = (float*)myMalloc(1000 * sizeof(float));;
for(int x2015=0; x2015 < 1000; x2015++) {
x2014[x2015] = 0.0f;

}
// dot: List(20, 50), WrappedArray(50, 26)
float* x2020 = (float*)myMalloc(520 * sizeof(float));;
for(int x2021=0; x2021 < 20; x2021++) {
int32_t x2025 = x2021 * 50;
int32_t x2035 = x2021 * 26;
for(int x2022=0; x2022 < 26; x2022++) {
float x2023 = 0.0f;
for(int x2024=0; x2024 < 50; x2024++) {
int32_t x2026 = x2025 + x2024;
float x2027 = x1985[x2026];
int32_t x2028 = x2024 * 26;
int32_t x2029 = x2028 + x2022;
float x2030 = x163[x2029];
float x2031 = x2027 * x2030;
x2023 += x2031;

}
float x2037 = x2023;
int32_t x2036 = x2035 + x2022;
x2020[x2036] = x2037;

}

}
float* x2043 = (float*)myMalloc(520 * sizeof(float));;
for(int x2044=0; x2044 < 520; x2044++) {
x2043[x2044] = 0.0f;

}
float* x2048 = (float*)myMalloc(520 * sizeof(float));;
int32_t x2049 = 0;
int32_t x2050 = 0;
int32_t x2051 = 0;
for(int x2052=0; x2052 < 20; x2052++) {
int32_t x2053 = x2050;
int32_t x2054 = x2051;
int32_t x2055 = x2049;
int32_t x2056 = x2055;
int32_t x2057 = x2053;
int32_t x2058 = x2054;
for(int x2059=0; x2059 < 26; x2059++) {
int32_t x2060 = x2056;
int32_t x2061 = x2057;
float x2062 = x2020[x2061];
int32_t x2063 = x2058;
float x2064 = x176[x2063];
float x2065 = x2062 + x2064;
x2048[x2060] = x2065;
x2056 += 1;
x2057 += 1;
x2058 += 1;

}
x2049 += 26;
x2050 += 26;

}
float* x2076 = (float*)myMalloc(520 * sizeof(float));;
for(int x2077=0; x2077 < 520; x2077++) {
x2076[x2077] = 0.0f;

}
int* x2081 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x2082=0; x2082 < 20; x2082++) {
int32_t x2083 = x2082 * 20;
int32_t x2084 = x1296 + x2083;
int32_t x2085 = x274[x2084];
x2081[x2082] = x2085;

}
float* x2089 = (float*)myMalloc(20 * sizeof(float));;
int32_t x2090 = 0;
for(int x2091=0; x2091 < 20; x2091++) {
float x2092 = -3.4028235E38f;
for(int x2093=0; x2093 < 26; x2093++) {
int32_t x2094 = x2090;
float x2095 = x2048[x2094];
float x2096 = x2092;
bool x2097 = x2095 > x2096;
if (x2097) {
float x2098 = x2048[x2094];
x2092 = x2098;
} else {
}
x2090 += 1;

}
float x2105 = x2092;
x2089[x2091] = x2105;

}
float* x2109 = (float*)myMalloc(520 * sizeof(float));;
for(int x2110=0; x2110 < 520; x2110++) {
x2109[x2110] = 0.0f;

}
int32_t x2114 = 0;
for(int x2115=0; x2115 < 20; x2115++) {
for(int x2116=0; x2116 < 26; x2116++) {
int32_t x2117 = x2114;
float x2118 = x2048[x2117];
float x2119 = x2089[x2115];
float x2120 = x2118 - x2119;
double x2121 = (double)x2120;
double x2122 = exp(x2121);
float x2123 = (float)x2122;
x2109[x2117] = x2123;
x2114 += 1;

}

}
float* x2130 = (float*)myMalloc(20 * sizeof(float));;
for(int x2131=0; x2131 < 20; x2131++) {
x2130[x2131] = 0.0f;

}
for(int x2135=0; x2135 < 20; x2135++) {
int32_t x2136 = x2135;
int32_t x2137 = x2135 * 26;
int32_t x2138 = x2137;
for(int x2139=0; x2139 < 26; x2139++) {
int32_t x2140 = x2136;
float x2141 = x2130[x2140];
int32_t x2142 = x2138;
float x2143 = x2109[x2142];
float x2144 = x2141 + x2143;
x2130[x2140] = x2144;
x2138 += 1;

}

}
x2114 = 0;
for(int x2152=0; x2152 < 20; x2152++) {
float x2153 = x2089[x2152];
float x2154 = x2130[x2152];
double x2155 = (double)x2154;
double x2156 = log(x2155);
float x2157 = (float)x2156;
float x2158 = x2153 + x2157;
for(int x2159=0; x2159 < 26; x2159++) {
int32_t x2160 = x2114;
float x2161 = x2048[x2160];
float x2162 = x2161 - x2158;
x2109[x2160] = x2162;
x2114 += 1;

}

}
float* x2169 = (float*)myMalloc(520 * sizeof(float));;
for(int x2170=0; x2170 < 520; x2170++) {
x2169[x2170] = 0.0f;

}
float* x2174 = (float*)myMalloc(20 * sizeof(float));;
int32_t x2175 = 0;
for(int x2176=0; x2176 < 20; x2176++) {
int32_t x2177 = x2175;
int32_t x2178 = x2081[x2176];
int32_t x2179 = x2177 + x2178;
float x2180 = x2109[x2179];
float x2181 = -1.0f * x2180;
x2174[x2176] = x2181;
x2175 += 26;

}
float* x2186 = (float*)myMalloc(20 * sizeof(float));;
for(int x2187=0; x2187 < 20; x2187++) {
x2186[x2187] = 0.0f;

}
float x2191 = 0.0f;
for(int x2192=0; x2192 < 20; x2192++) {
float x2193 = x2191;
float x2194 = x2174[x2192];
float x2195 = x2193 + x2194;
x2191 = x2195;

}
float x2199 = x2191;
float* x2200 = (float*)myMalloc(1 * sizeof(float));;
x2200[0] = x2199;
float* x2202 = (float*)myMalloc(1 * sizeof(float));;
for(int x2203=0; x2203 < 1; x2203++) {
x2202[x2203] = 0.0f;

}
float* x2207 = (float*)myMalloc(1 * sizeof(float));;
int32_t x2208 = 0;
int32_t x2209 = 0;
int32_t x2210 = 0;
int32_t x2211 = x2208;
int32_t x2212 = x2209;
float x2213 = x1298[x2212];
int32_t x2214 = x2210;
float x2215 = x2200[x2214];
float x2216 = x2213 + x2215;
x2207[x2211] = x2216;
x2208 += 1;
float* x2219 = (float*)myMalloc(1 * sizeof(float));;
for(int x2220=0; x2220 < 1; x2220++) {
x2219[x2220] = 0.0f;

}
float** x2225 = (float**)myMalloc(6 * sizeof(float*));;
x2225[0] = x2207;
x2225[1] = x2219;
x2225[2] = x1985;
x2225[3] = x2014;
x2225[4] = x1937;
x2225[5] = x1966;
int32_t x2234 = 0;
int32_t x2235 = 0;
int32_t x2236 = 0;
int32_t x2237 = x2234;
int32_t x2240 = x2235;
int32_t x2242 = x2236;
x2236 += 1;
int32_t x2261 = 0;
float* x2274 = (float*)myMalloc(20 * sizeof(float));;
int32_t x2295 = 0;
int32_t x2315 = 0;
int32_t x2316 = 0;
int32_t x2317 = 0;
int32_t x2377 = 0;
int32_t x2378 = 0;
int32_t x2379 = 0;
int32_t x2426 = 0;
int32_t x2427 = 0;
int32_t x2428 = 0;
int32_t x2462 = 0;
int32_t x2463 = 0;
int32_t x2464 = 0;
int32_t x2500 = 0;
int32_t x2501 = 0;
int32_t x2502 = 0;
int32_t x2549 = 0;
int32_t x2550 = 0;
int32_t x2551 = 0;
int32_t x2584 = 0;
int32_t x2585 = 0;
int32_t x2586 = 0;
int32_t x2684 = 0;
int32_t x2685 = 0;
int32_t x2686 = 0;
int32_t x2719 = 0;
int32_t x2720 = 0;
int32_t x2721 = 0;
int32_t x2819 = 0;
int32_t x2820 = 0;
int32_t x2821 = 0;
int32_t x2854 = 0;
int32_t x2855 = 0;
int32_t x2856 = 0;
int32_t x2954 = 0;
int32_t x2955 = 0;
int32_t x2956 = 0;
int32_t x2989 = 0;
int32_t x2990 = 0;
int32_t x2991 = 0;
int32_t x2224 = x1296 + 1;
x1293(x2224,x2225);
float x2238 = x1299[x2237];
float x2239 = x1298[x2237];
float x2241 = x2200[x2240];
float x2243 = x2219[x2242];
float x2244 = x2238 + x2243;
x1299[x2237] = x2244;
float x2246 = x2202[x2240];
float x2247 = x1298[x2237];
float x2248 = x2200[x2240];
float x2249 = x2219[x2242];
float x2250 = x2246 + x2249;
x2202[x2240] = x2250;
// += tensor of dim 0
float x2254 = x2202[0];
for(int x2255=0; x2255 < 20; x2255++) {
float x2256 = x2186[x2255];
float x2257 = x2256 + x2254;
x2186[x2255] = x2257;

}
for(int x2262=0; x2262 < 20; x2262++) {
int32_t x2263 = x2261;
int32_t x2264 = x2081[x2262];
int32_t x2265 = x2263 + x2264;
float x2266 = x2169[x2265];
float x2267 = x2186[x2262];
float x2268 = -1.0f * x2267;
float x2269 = x2266 + x2268;
x2169[x2265] = x2269;
x2261 += 26;

}
for(int x2275=0; x2275 < 20; x2275++) {
x2274[x2275] = 0.0f;

}
for(int x2279=0; x2279 < 20; x2279++) {
int32_t x2280 = x2279;
int32_t x2281 = x2279 * 26;
int32_t x2282 = x2281;
for(int x2283=0; x2283 < 26; x2283++) {
int32_t x2284 = x2280;
float x2285 = x2274[x2284];
int32_t x2286 = x2282;
float x2287 = x2169[x2286];
float x2288 = x2285 + x2287;
x2274[x2284] = x2288;
x2282 += 1;

}

}
for(int x2296=0; x2296 < 20; x2296++) {
for(int x2297=0; x2297 < 26; x2297++) {
int32_t x2298 = x2295;
float x2299 = x2076[x2298];
float x2300 = x2169[x2298];
float x2301 = x2109[x2298];
float x2305 = x2274[x2296];
double x2302 = (double)x2301;
double x2303 = exp(x2302);
float x2304 = (float)x2303;
float x2306 = x2304 * x2305;
float x2307 = x2300 - x2306;
float x2308 = x2299 + x2307;
x2076[x2298] = x2308;
x2295 += 1;

}

}
for(int x2318=0; x2318 < 20; x2318++) {
int32_t x2319 = x2315;
int32_t x2320 = x2316;
int32_t x2321 = x2317;
int32_t x2322 = x2319;
int32_t x2323 = x2320;
int32_t x2324 = x2321;
for(int x2325=0; x2325 < 26; x2325++) {
int32_t x2326 = x2322;
float x2327 = x2043[x2326];
float x2328 = x2020[x2326];
int32_t x2329 = x2323;
float x2330 = x176[x2329];
int32_t x2331 = x2324;
float x2332 = x2076[x2331];
float x2333 = x2327 + x2332;
x2043[x2326] = x2333;
float x2335 = x182[x2329];
float x2336 = x2020[x2326];
float x2337 = x176[x2329];
float x2338 = x2076[x2331];
float x2339 = x2335 + x2338;
x182[x2329] = x2339;
x2324 += 1;
x2322 += 1;
x2323 += 1;

}
x2317 += 26;
x2315 += 26;

}
for(int x2350=0; x2350 < 20; x2350++) {
int32_t x2353 = x2350 * 50;
int32_t x2359 = x2350 * 26;
for(int x2351=0; x2351 < 26; x2351++) {
int32_t x2360 = x2359 + x2351;
for(int x2352=0; x2352 < 50; x2352++) {
int32_t x2354 = x2353 + x2352;
float x2355 = x2014[x2354];
int32_t x2356 = x2352 * 26;
int32_t x2357 = x2356 + x2351;
float x2358 = x163[x2357];
float x2361 = x2043[x2360];
float x2362 = x2358 * x2361;
float x2363 = x2355 + x2362;
x2014[x2354] = x2363;
float x2365 = x171[x2357];
float x2366 = x1985[x2354];
float x2367 = x2043[x2360];
float x2368 = x2366 * x2367;
float x2369 = x2365 + x2368;
x171[x2357] = x2369;

}

}

}
for(int x2380=0; x2380 < 20; x2380++) {
int32_t x2381 = x2377;
int32_t x2382 = x2378;
int32_t x2383 = x2379;
int32_t x2384 = x2381;
int32_t x2385 = x2382;
int32_t x2386 = x2383;
for(int x2387=0; x2387 < 50; x2387++) {
int32_t x2388 = x2384;
float x2389 = x1726[x2388];
float x2390 = x1714[x2388];
int32_t x2391 = x2385;
float x2392 = x1971[x2391];
int32_t x2393 = x2386;
float x2394 = x2014[x2393];
float x2395 = x2394 * x2392;
float x2396 = x2389 + x2395;
x1726[x2388] = x2396;
float x2398 = x1980[x2391];
float x2399 = x1714[x2388];
float x2400 = x1971[x2391];
float x2401 = x2014[x2393];
float x2402 = x2401 * x2399;
float x2403 = x2398 + x2402;
x1980[x2391] = x2403;
x2386 += 1;
x2384 += 1;
x2385 += 1;

}
x2379 += 50;
x2377 += 50;
x2378 += 50;

}
for(int x2415=0; x2415 < 1000; x2415++) {
float x2416 = x1966[x2415];
float x2417 = x1971[x2415];
float x2420 = x1980[x2415];
float x2418 = x2417 * x2417;
float x2419 = 1.0f - x2418;
float x2421 = x2419 * x2420;
float x2422 = x2416 + x2421;
x1966[x2415] = x2422;

}
for(int x2429=0; x2429 < 20; x2429++) {
int32_t x2430 = x2426;
int32_t x2431 = x2427;
int32_t x2432 = x2428;
int32_t x2433 = x2430;
int32_t x2434 = x2431;
int32_t x2435 = x2432;
for(int x2436=0; x2436 < 50; x2436++) {
int32_t x2437 = x2433;
float x2438 = x1898[x2437];
float x2439 = x1869[x2437];
int32_t x2440 = x2434;
float x2441 = x1903[x2440];
int32_t x2442 = x2435;
float x2443 = x1966[x2442];
float x2444 = x2438 + x2443;
x1898[x2437] = x2444;
float x2446 = x1932[x2440];
float x2447 = x1869[x2437];
float x2448 = x1903[x2440];
float x2449 = x1966[x2442];
float x2450 = x2446 + x2449;
x1932[x2440] = x2450;
x2435 += 1;
x2433 += 1;
x2434 += 1;

}
x2428 += 50;
x2426 += 50;
x2427 += 50;

}
for(int x2465=0; x2465 < 20; x2465++) {
int32_t x2466 = x2462;
int32_t x2467 = x2463;
int32_t x2468 = x2464;
int32_t x2469 = x2466;
int32_t x2470 = x2467;
int32_t x2471 = x2468;
for(int x2472=0; x2472 < 50; x2472++) {
int32_t x2473 = x2469;
float x2474 = x1585[x2473];
float x2475 = x1573[x2473];
int32_t x2476 = x2470;
float x2477 = x1855[x2476];
int32_t x2478 = x2471;
float x2479 = x1932[x2478];
float x2480 = x2479 * x2477;
float x2481 = x2474 + x2480;
x1585[x2473] = x2481;
float x2483 = x1864[x2476];
float x2484 = x1573[x2473];
float x2485 = x1855[x2476];
float x2486 = x1932[x2478];
float x2487 = x2486 * x2484;
float x2488 = x2483 + x2487;
x1864[x2476] = x2488;
x2471 += 1;
x2469 += 1;
x2470 += 1;

}
x2464 += 50;
x2462 += 50;
x2463 += 50;

}
for(int x2503=0; x2503 < 20; x2503++) {
int32_t x2504 = x2500;
int32_t x2505 = x2501;
int32_t x2506 = x2502;
int32_t x2507 = x2504;
int32_t x2508 = x2505;
int32_t x2509 = x2506;
for(int x2510=0; x2510 < 50; x2510++) {
int32_t x2511 = x2507;
float x2512 = x1444[x2511];
float x2513 = x1432[x2511];
int32_t x2514 = x2508;
float x2515 = x1302[x2514];
int32_t x2516 = x2509;
float x2517 = x1898[x2516];
float x2518 = x2517 * x2515;
float x2519 = x2512 + x2518;
x1444[x2511] = x2519;
float x2521 = x1303[x2514];
float x2522 = x1432[x2511];
float x2523 = x1302[x2514];
float x2524 = x1898[x2516];
float x2525 = x2524 * x2522;
float x2526 = x2521 + x2525;
x1303[x2514] = x2526;
x2509 += 1;
x2507 += 1;
x2508 += 1;

}
x2502 += 50;
x2500 += 50;
x2501 += 50;

}
for(int x2538=0; x2538 < 1000; x2538++) {
float x2539 = x1850[x2538];
float x2540 = x1855[x2538];
float x2543 = x1864[x2538];
float x2541 = x2540 * x2540;
float x2542 = 1.0f - x2541;
float x2544 = x2542 * x2543;
float x2545 = x2539 + x2544;
x1850[x2538] = x2545;

}
for(int x2552=0; x2552 < 20; x2552++) {
int32_t x2553 = x2549;
int32_t x2554 = x2550;
int32_t x2555 = x2551;
int32_t x2556 = x2553;
int32_t x2557 = x2554;
int32_t x2558 = x2555;
for(int x2559=0; x2559 < 50; x2559++) {
int32_t x2560 = x2556;
float x2561 = x1817[x2560];
float x2562 = x1788[x2560];
int32_t x2563 = x2557;
float x2564 = x117[x2563];
int32_t x2565 = x2558;
float x2566 = x1850[x2565];
float x2567 = x2561 + x2566;
x1817[x2560] = x2567;
float x2569 = x122[x2563];
float x2570 = x1788[x2560];
float x2571 = x117[x2563];
float x2572 = x1850[x2565];
float x2573 = x2569 + x2572;
x122[x2563] = x2573;
x2558 += 1;
x2556 += 1;
x2557 += 1;

}
x2551 += 50;
x2549 += 50;

}
for(int x2587=0; x2587 < 20; x2587++) {
int32_t x2588 = x2584;
int32_t x2589 = x2585;
int32_t x2590 = x2586;
int32_t x2591 = x2588;
int32_t x2592 = x2589;
int32_t x2593 = x2590;
for(int x2594=0; x2594 < 50; x2594++) {
int32_t x2595 = x2591;
float x2596 = x1755[x2595];
float x2597 = x1732[x2595];
int32_t x2598 = x2592;
float x2599 = x1761[x2598];
int32_t x2600 = x2593;
float x2601 = x1817[x2600];
float x2602 = x2596 + x2601;
x1755[x2595] = x2602;
float x2604 = x1783[x2598];
float x2605 = x1732[x2595];
float x2606 = x1761[x2598];
float x2607 = x1817[x2600];
float x2608 = x2604 + x2607;
x1783[x2598] = x2608;
x2593 += 1;
x2591 += 1;
x2592 += 1;

}
x2586 += 50;
x2584 += 50;
x2585 += 50;

}
for(int x2620=0; x2620 < 20; x2620++) {
int32_t x2623 = x2620 * 50;
for(int x2621=0; x2621 < 50; x2621++) {
int32_t x2629 = x2623 + x2621;
for(int x2622=0; x2622 < 50; x2622++) {
int32_t x2624 = x2623 + x2622;
float x2625 = x1301[x2624];
int32_t x2626 = x2622 * 50;
int32_t x2627 = x2626 + x2621;
float x2628 = x104[x2627];
float x2630 = x1783[x2629];
float x2631 = x2628 * x2630;
float x2632 = x2625 + x2631;
x1301[x2624] = x2632;
float x2634 = x112[x2627];
float x2635 = x1300[x2624];
float x2636 = x1783[x2629];
float x2637 = x2635 * x2636;
float x2638 = x2634 + x2637;
x112[x2627] = x2638;

}

}

}
for(int x2646=0; x2646 < 20; x2646++) {
int32_t x2649 = x2646 * 26;
int32_t x2655 = x2646 * 50;
for(int x2647=0; x2647 < 50; x2647++) {
int32_t x2656 = x2655 + x2647;
for(int x2648=0; x2648 < 26; x2648++) {
int32_t x2650 = x2649 + x2648;
float x2651 = x1307[x2650];
int32_t x2652 = x2648 * 50;
int32_t x2653 = x2652 + x2647;
float x2654 = x91[x2653];
float x2657 = x1755[x2656];
float x2658 = x2654 * x2657;
float x2659 = x2651 + x2658;
x1307[x2650] = x2659;
float x2661 = x99[x2653];
float x2662 = x1306[x2650];
float x2663 = x1755[x2656];
float x2664 = x2662 * x2663;
float x2665 = x2661 + x2664;
x99[x2653] = x2665;

}

}

}
for(int x2673=0; x2673 < 1000; x2673++) {
float x2674 = x1709[x2673];
float x2675 = x1714[x2673];
float x2678 = x1726[x2673];
float x2676 = 1.0f - x2675;
float x2677 = x2676 * x2675;
float x2679 = x2677 * x2678;
float x2680 = x2674 + x2679;
x1709[x2673] = x2680;

}
for(int x2687=0; x2687 < 20; x2687++) {
int32_t x2688 = x2684;
int32_t x2689 = x2685;
int32_t x2690 = x2686;
int32_t x2691 = x2688;
int32_t x2692 = x2689;
int32_t x2693 = x2690;
for(int x2694=0; x2694 < 50; x2694++) {
int32_t x2695 = x2691;
float x2696 = x1676[x2695];
float x2697 = x1647[x2695];
int32_t x2698 = x2692;
float x2699 = x153[x2698];
int32_t x2700 = x2693;
float x2701 = x1709[x2700];
float x2702 = x2696 + x2701;
x1676[x2695] = x2702;
float x2704 = x158[x2698];
float x2705 = x1647[x2695];
float x2706 = x153[x2698];
float x2707 = x1709[x2700];
float x2708 = x2704 + x2707;
x158[x2698] = x2708;
x2693 += 1;
x2691 += 1;
x2692 += 1;

}
x2686 += 50;
x2684 += 50;

}
for(int x2722=0; x2722 < 20; x2722++) {
int32_t x2723 = x2719;
int32_t x2724 = x2720;
int32_t x2725 = x2721;
int32_t x2726 = x2723;
int32_t x2727 = x2724;
int32_t x2728 = x2725;
for(int x2729=0; x2729 < 50; x2729++) {
int32_t x2730 = x2726;
float x2731 = x1614[x2730];
float x2732 = x1591[x2730];
int32_t x2733 = x2727;
float x2734 = x1620[x2733];
int32_t x2735 = x2728;
float x2736 = x1676[x2735];
float x2737 = x2731 + x2736;
x1614[x2730] = x2737;
float x2739 = x1642[x2733];
float x2740 = x1591[x2730];
float x2741 = x1620[x2733];
float x2742 = x1676[x2735];
float x2743 = x2739 + x2742;
x1642[x2733] = x2743;
x2728 += 1;
x2726 += 1;
x2727 += 1;

}
x2721 += 50;
x2719 += 50;
x2720 += 50;

}
for(int x2755=0; x2755 < 20; x2755++) {
int32_t x2758 = x2755 * 50;
for(int x2756=0; x2756 < 50; x2756++) {
int32_t x2764 = x2758 + x2756;
for(int x2757=0; x2757 < 50; x2757++) {
int32_t x2759 = x2758 + x2757;
float x2760 = x1301[x2759];
int32_t x2761 = x2757 * 50;
int32_t x2762 = x2761 + x2756;
float x2763 = x140[x2762];
float x2765 = x1642[x2764];
float x2766 = x2763 * x2765;
float x2767 = x2760 + x2766;
x1301[x2759] = x2767;
float x2769 = x148[x2762];
float x2770 = x1300[x2759];
float x2771 = x1642[x2764];
float x2772 = x2770 * x2771;
float x2773 = x2769 + x2772;
x148[x2762] = x2773;

}

}

}
for(int x2781=0; x2781 < 20; x2781++) {
int32_t x2784 = x2781 * 26;
int32_t x2790 = x2781 * 50;
for(int x2782=0; x2782 < 50; x2782++) {
int32_t x2791 = x2790 + x2782;
for(int x2783=0; x2783 < 26; x2783++) {
int32_t x2785 = x2784 + x2783;
float x2786 = x1307[x2785];
int32_t x2787 = x2783 * 50;
int32_t x2788 = x2787 + x2782;
float x2789 = x127[x2788];
float x2792 = x1614[x2791];
float x2793 = x2789 * x2792;
float x2794 = x2786 + x2793;
x1307[x2785] = x2794;
float x2796 = x135[x2788];
float x2797 = x1306[x2785];
float x2798 = x1614[x2791];
float x2799 = x2797 * x2798;
float x2800 = x2796 + x2799;
x135[x2788] = x2800;

}

}

}
for(int x2808=0; x2808 < 1000; x2808++) {
float x2809 = x1568[x2808];
float x2810 = x1573[x2808];
float x2813 = x1585[x2808];
float x2811 = 1.0f - x2810;
float x2812 = x2811 * x2810;
float x2814 = x2812 * x2813;
float x2815 = x2809 + x2814;
x1568[x2808] = x2815;

}
for(int x2822=0; x2822 < 20; x2822++) {
int32_t x2823 = x2819;
int32_t x2824 = x2820;
int32_t x2825 = x2821;
int32_t x2826 = x2823;
int32_t x2827 = x2824;
int32_t x2828 = x2825;
for(int x2829=0; x2829 < 50; x2829++) {
int32_t x2830 = x2826;
float x2831 = x1535[x2830];
float x2832 = x1506[x2830];
int32_t x2833 = x2827;
float x2834 = x81[x2833];
int32_t x2835 = x2828;
float x2836 = x1568[x2835];
float x2837 = x2831 + x2836;
x1535[x2830] = x2837;
float x2839 = x86[x2833];
float x2840 = x1506[x2830];
float x2841 = x81[x2833];
float x2842 = x1568[x2835];
float x2843 = x2839 + x2842;
x86[x2833] = x2843;
x2828 += 1;
x2826 += 1;
x2827 += 1;

}
x2821 += 50;
x2819 += 50;

}
for(int x2857=0; x2857 < 20; x2857++) {
int32_t x2858 = x2854;
int32_t x2859 = x2855;
int32_t x2860 = x2856;
int32_t x2861 = x2858;
int32_t x2862 = x2859;
int32_t x2863 = x2860;
for(int x2864=0; x2864 < 50; x2864++) {
int32_t x2865 = x2861;
float x2866 = x1473[x2865];
float x2867 = x1450[x2865];
int32_t x2868 = x2862;
float x2869 = x1479[x2868];
int32_t x2870 = x2863;
float x2871 = x1535[x2870];
float x2872 = x2866 + x2871;
x1473[x2865] = x2872;
float x2874 = x1501[x2868];
float x2875 = x1450[x2865];
float x2876 = x1479[x2868];
float x2877 = x1535[x2870];
float x2878 = x2874 + x2877;
x1501[x2868] = x2878;
x2863 += 1;
x2861 += 1;
x2862 += 1;

}
x2856 += 50;
x2854 += 50;
x2855 += 50;

}
for(int x2890=0; x2890 < 20; x2890++) {
int32_t x2893 = x2890 * 50;
for(int x2891=0; x2891 < 50; x2891++) {
int32_t x2899 = x2893 + x2891;
for(int x2892=0; x2892 < 50; x2892++) {
int32_t x2894 = x2893 + x2892;
float x2895 = x1301[x2894];
int32_t x2896 = x2892 * 50;
int32_t x2897 = x2896 + x2891;
float x2898 = x68[x2897];
float x2900 = x1501[x2899];
float x2901 = x2898 * x2900;
float x2902 = x2895 + x2901;
x1301[x2894] = x2902;
float x2904 = x76[x2897];
float x2905 = x1300[x2894];
float x2906 = x1501[x2899];
float x2907 = x2905 * x2906;
float x2908 = x2904 + x2907;
x76[x2897] = x2908;

}

}

}
for(int x2916=0; x2916 < 20; x2916++) {
int32_t x2919 = x2916 * 26;
int32_t x2925 = x2916 * 50;
for(int x2917=0; x2917 < 50; x2917++) {
int32_t x2926 = x2925 + x2917;
for(int x2918=0; x2918 < 26; x2918++) {
int32_t x2920 = x2919 + x2918;
float x2921 = x1307[x2920];
int32_t x2922 = x2918 * 50;
int32_t x2923 = x2922 + x2917;
float x2924 = x55[x2923];
float x2927 = x1473[x2926];
float x2928 = x2924 * x2927;
float x2929 = x2921 + x2928;
x1307[x2920] = x2929;
float x2931 = x63[x2923];
float x2932 = x1306[x2920];
float x2933 = x1473[x2926];
float x2934 = x2932 * x2933;
float x2935 = x2931 + x2934;
x63[x2923] = x2935;

}

}

}
for(int x2943=0; x2943 < 1000; x2943++) {
float x2944 = x1427[x2943];
float x2945 = x1432[x2943];
float x2948 = x1444[x2943];
float x2946 = 1.0f - x2945;
float x2947 = x2946 * x2945;
float x2949 = x2947 * x2948;
float x2950 = x2944 + x2949;
x1427[x2943] = x2950;

}
for(int x2957=0; x2957 < 20; x2957++) {
int32_t x2958 = x2954;
int32_t x2959 = x2955;
int32_t x2960 = x2956;
int32_t x2961 = x2958;
int32_t x2962 = x2959;
int32_t x2963 = x2960;
for(int x2964=0; x2964 < 50; x2964++) {
int32_t x2965 = x2961;
float x2966 = x1394[x2965];
float x2967 = x1365[x2965];
int32_t x2968 = x2962;
float x2969 = x44[x2968];
int32_t x2970 = x2963;
float x2971 = x1427[x2970];
float x2972 = x2966 + x2971;
x1394[x2965] = x2972;
float x2974 = x50[x2968];
float x2975 = x1365[x2965];
float x2976 = x44[x2968];
float x2977 = x1427[x2970];
float x2978 = x2974 + x2977;
x50[x2968] = x2978;
x2963 += 1;
x2961 += 1;
x2962 += 1;

}
x2956 += 50;
x2954 += 50;

}
for(int x2992=0; x2992 < 20; x2992++) {
int32_t x2993 = x2989;
int32_t x2994 = x2990;
int32_t x2995 = x2991;
int32_t x2996 = x2993;
int32_t x2997 = x2994;
int32_t x2998 = x2995;
for(int x2999=0; x2999 < 50; x2999++) {
int32_t x3000 = x2996;
float x3001 = x1332[x3000];
float x3002 = x1309[x3000];
int32_t x3003 = x2997;
float x3004 = x1338[x3003];
int32_t x3005 = x2998;
float x3006 = x1394[x3005];
float x3007 = x3001 + x3006;
x1332[x3000] = x3007;
float x3009 = x1360[x3003];
float x3010 = x1309[x3000];
float x3011 = x1338[x3003];
float x3012 = x1394[x3005];
float x3013 = x3009 + x3012;
x1360[x3003] = x3013;
x2998 += 1;
x2996 += 1;
x2997 += 1;

}
x2991 += 50;
x2989 += 50;
x2990 += 50;

}
for(int x3025=0; x3025 < 20; x3025++) {
int32_t x3028 = x3025 * 50;
for(int x3026=0; x3026 < 50; x3026++) {
int32_t x3034 = x3028 + x3026;
for(int x3027=0; x3027 < 50; x3027++) {
int32_t x3029 = x3028 + x3027;
float x3030 = x1301[x3029];
int32_t x3031 = x3027 * 50;
int32_t x3032 = x3031 + x3026;
float x3033 = x30[x3032];
float x3035 = x1360[x3034];
float x3036 = x3033 * x3035;
float x3037 = x3030 + x3036;
x1301[x3029] = x3037;
float x3039 = x39[x3032];
float x3040 = x1300[x3029];
float x3041 = x1360[x3034];
float x3042 = x3040 * x3041;
float x3043 = x3039 + x3042;
x39[x3032] = x3043;

}

}

}
for(int x3051=0; x3051 < 20; x3051++) {
int32_t x3054 = x3051 * 26;
int32_t x3060 = x3051 * 50;
for(int x3052=0; x3052 < 50; x3052++) {
int32_t x3061 = x3060 + x3052;
for(int x3053=0; x3053 < 26; x3053++) {
int32_t x3055 = x3054 + x3053;
float x3056 = x1307[x3055];
int32_t x3057 = x3053 * 50;
int32_t x3058 = x3057 + x3052;
float x3059 = x16[x3058];
float x3062 = x1332[x3061];
float x3063 = x3059 * x3062;
float x3064 = x3056 + x3063;
x1307[x3055] = x3064;
float x3066 = x25[x3058];
float x3067 = x1306[x3055];
float x3068 = x1332[x3061];
float x3069 = x3067 * x3068;
float x3070 = x3066 + x3069;
x25[x3058] = x3070;

}

}

}
} else {
float x3079 = 0.0f;
float x3080 = x3079;
float x3081 = x1298[0];
float x3082 = x3080 + x3081;
x3079 = x3082;
float x3084 = x3079;
float* x3085 = (float*)myMalloc(1 * sizeof(float));;
x3085[0] = x3084;
float* x3087 = (float*)myMalloc(1 * sizeof(float));;
for(int x3088=0; x3088 < 1; x3088++) {
x3087[x3088] = 0.0f;

}
float x3092 = x3087[0];
x3087[0] = 1.0f;
float x3094 = x3085[0];
x297[0] = x3094;
// += tensor of dim 0
float x3097 = x3087[0];
float x3098 = x1299[0];
float x3099 = x3098 + x3097;
x1299[0] = x3099;
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
float** x3983 = (float**)myMalloc(6 * sizeof(float*));;
x3983[0] = x329;
x3983[1] = x334;
x3983[2] = x339;
x3983[3] = x345;
x3983[4] = x350;
x3983[5] = x355;
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
// dot: WrappedArray(20, 26), WrappedArray(26, 50)
float* x376 = (float*)myMalloc(1000 * sizeof(float));;
for(int x377=0; x377 < 20; x377++) {
int32_t x381 = x377 * 26;
int32_t x391 = x377 * 50;
for(int x378=0; x378 < 50; x378++) {
float x379 = 0.0f;
for(int x380=0; x380 < 26; x380++) {
int32_t x382 = x381 + x380;
float x383 = x373[x382];
int32_t x384 = x380 * 50;
int32_t x385 = x384 + x378;
float x386 = x16[x385];
float x387 = x383 * x386;
x379 += x387;

}
float x393 = x379;
int32_t x392 = x391 + x378;
x376[x392] = x393;

}

}
float* x399 = (float*)myMalloc(1000 * sizeof(float));;
for(int x400=0; x400 < 1000; x400++) {
x399[x400] = 0.0f;

}
// dot: WrappedArray(20, 50), WrappedArray(50, 50)
float* x405 = (float*)myMalloc(1000 * sizeof(float));;
for(int x406=0; x406 < 20; x406++) {
int32_t x410 = x406 * 50;
for(int x407=0; x407 < 50; x407++) {
float x408 = 0.0f;
for(int x409=0; x409 < 50; x409++) {
int32_t x411 = x410 + x409;
float x412 = x367[x411];
int32_t x413 = x409 * 50;
int32_t x414 = x413 + x407;
float x415 = x30[x414];
float x416 = x412 * x415;
x408 += x416;

}
float x421 = x408;
int32_t x420 = x410 + x407;
x405[x420] = x421;

}

}
float* x427 = (float*)myMalloc(1000 * sizeof(float));;
for(int x428=0; x428 < 1000; x428++) {
x427[x428] = 0.0f;

}
float* x432 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x433 = 0;
int32_t x434 = 0;
int32_t x435 = 0;
for(int x436=0; x436 < 20; x436++) {
int32_t x437 = x434;
int32_t x438 = x435;
int32_t x439 = x433;
int32_t x440 = x439;
int32_t x441 = x437;
int32_t x442 = x438;
for(int x443=0; x443 < 50; x443++) {
int32_t x444 = x440;
int32_t x445 = x441;
float x446 = x376[x445];
int32_t x447 = x442;
float x448 = x405[x447];
float x449 = x446 + x448;
x432[x444] = x449;
x440 += 1;
x441 += 1;
x442 += 1;

}
x433 += 50;
x434 += 50;
x435 += 50;

}
float* x461 = (float*)myMalloc(1000 * sizeof(float));;
for(int x462=0; x462 < 1000; x462++) {
x461[x462] = 0.0f;

}
float* x466 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x467 = 0;
int32_t x468 = 0;
int32_t x469 = 0;
for(int x470=0; x470 < 20; x470++) {
int32_t x471 = x468;
int32_t x472 = x469;
int32_t x473 = x467;
int32_t x474 = x473;
int32_t x475 = x471;
int32_t x476 = x472;
for(int x477=0; x477 < 50; x477++) {
int32_t x478 = x474;
int32_t x479 = x475;
float x480 = x432[x479];
int32_t x481 = x476;
float x482 = x44[x481];
float x483 = x480 + x482;
x466[x478] = x483;
x474 += 1;
x475 += 1;
x476 += 1;

}
x467 += 50;
x468 += 50;

}
float* x494 = (float*)myMalloc(1000 * sizeof(float));;
for(int x495=0; x495 < 1000; x495++) {
x494[x495] = 0.0f;

}
float* x499 = (float*)myMalloc(1000 * sizeof(float));;
for(int x500=0; x500 < 1000; x500++) {
float x501 = x466[x500];
float x502 = -1.0f * x501;
double x503 = (double)x502;
double x504 = exp(x503);
float x505 = (float)x504;
float x506 = x505 + 1.0f;
float x507 = 1.0f / x506;
x499[x500] = x507;

}
float* x511 = (float*)myMalloc(1000 * sizeof(float));;
for(int x512=0; x512 < 1000; x512++) {
x511[x512] = 0.0f;

}
// dot: WrappedArray(20, 26), WrappedArray(26, 50)
float* x517 = (float*)myMalloc(1000 * sizeof(float));;
for(int x518=0; x518 < 20; x518++) {
int32_t x522 = x518 * 26;
int32_t x532 = x518 * 50;
for(int x519=0; x519 < 50; x519++) {
float x520 = 0.0f;
for(int x521=0; x521 < 26; x521++) {
int32_t x523 = x522 + x521;
float x524 = x373[x523];
int32_t x525 = x521 * 50;
int32_t x526 = x525 + x519;
float x527 = x55[x526];
float x528 = x524 * x527;
x520 += x528;

}
float x534 = x520;
int32_t x533 = x532 + x519;
x517[x533] = x534;

}

}
float* x540 = (float*)myMalloc(1000 * sizeof(float));;
for(int x541=0; x541 < 1000; x541++) {
x540[x541] = 0.0f;

}
// dot: WrappedArray(20, 50), WrappedArray(50, 50)
float* x546 = (float*)myMalloc(1000 * sizeof(float));;
for(int x547=0; x547 < 20; x547++) {
int32_t x551 = x547 * 50;
for(int x548=0; x548 < 50; x548++) {
float x549 = 0.0f;
for(int x550=0; x550 < 50; x550++) {
int32_t x552 = x551 + x550;
float x553 = x367[x552];
int32_t x554 = x550 * 50;
int32_t x555 = x554 + x548;
float x556 = x68[x555];
float x557 = x553 * x556;
x549 += x557;

}
float x562 = x549;
int32_t x561 = x551 + x548;
x546[x561] = x562;

}

}
float* x568 = (float*)myMalloc(1000 * sizeof(float));;
for(int x569=0; x569 < 1000; x569++) {
x568[x569] = 0.0f;

}
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
float x587 = x517[x586];
int32_t x588 = x583;
float x589 = x546[x588];
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
for(int x603=0; x603 < 1000; x603++) {
x602[x603] = 0.0f;

}
float* x607 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x608 = 0;
int32_t x609 = 0;
int32_t x610 = 0;
for(int x611=0; x611 < 20; x611++) {
int32_t x612 = x609;
int32_t x613 = x610;
int32_t x614 = x608;
int32_t x615 = x614;
int32_t x616 = x612;
int32_t x617 = x613;
for(int x618=0; x618 < 50; x618++) {
int32_t x619 = x615;
int32_t x620 = x616;
float x621 = x573[x620];
int32_t x622 = x617;
float x623 = x81[x622];
float x624 = x621 + x623;
x607[x619] = x624;
x615 += 1;
x616 += 1;
x617 += 1;

}
x608 += 50;
x609 += 50;

}
float* x635 = (float*)myMalloc(1000 * sizeof(float));;
for(int x636=0; x636 < 1000; x636++) {
x635[x636] = 0.0f;

}
float* x640 = (float*)myMalloc(1000 * sizeof(float));;
for(int x641=0; x641 < 1000; x641++) {
float x642 = x607[x641];
float x643 = -1.0f * x642;
double x644 = (double)x643;
double x645 = exp(x644);
float x646 = (float)x645;
float x647 = x646 + 1.0f;
float x648 = 1.0f / x647;
x640[x641] = x648;

}
float* x652 = (float*)myMalloc(1000 * sizeof(float));;
for(int x653=0; x653 < 1000; x653++) {
x652[x653] = 0.0f;

}
// dot: WrappedArray(20, 26), WrappedArray(26, 50)
float* x658 = (float*)myMalloc(1000 * sizeof(float));;
for(int x659=0; x659 < 20; x659++) {
int32_t x663 = x659 * 26;
int32_t x673 = x659 * 50;
for(int x660=0; x660 < 50; x660++) {
float x661 = 0.0f;
for(int x662=0; x662 < 26; x662++) {
int32_t x664 = x663 + x662;
float x665 = x373[x664];
int32_t x666 = x662 * 50;
int32_t x667 = x666 + x660;
float x668 = x127[x667];
float x669 = x665 * x668;
x661 += x669;

}
float x675 = x661;
int32_t x674 = x673 + x660;
x658[x674] = x675;

}

}
float* x681 = (float*)myMalloc(1000 * sizeof(float));;
for(int x682=0; x682 < 1000; x682++) {
x681[x682] = 0.0f;

}
// dot: WrappedArray(20, 50), WrappedArray(50, 50)
float* x687 = (float*)myMalloc(1000 * sizeof(float));;
for(int x688=0; x688 < 20; x688++) {
int32_t x692 = x688 * 50;
for(int x689=0; x689 < 50; x689++) {
float x690 = 0.0f;
for(int x691=0; x691 < 50; x691++) {
int32_t x693 = x692 + x691;
float x694 = x367[x693];
int32_t x695 = x691 * 50;
int32_t x696 = x695 + x689;
float x697 = x140[x696];
float x698 = x694 * x697;
x690 += x698;

}
float x703 = x690;
int32_t x702 = x692 + x689;
x687[x702] = x703;

}

}
float* x709 = (float*)myMalloc(1000 * sizeof(float));;
for(int x710=0; x710 < 1000; x710++) {
x709[x710] = 0.0f;

}
float* x714 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x715 = 0;
int32_t x716 = 0;
int32_t x717 = 0;
for(int x718=0; x718 < 20; x718++) {
int32_t x719 = x716;
int32_t x720 = x717;
int32_t x721 = x715;
int32_t x722 = x721;
int32_t x723 = x719;
int32_t x724 = x720;
for(int x725=0; x725 < 50; x725++) {
int32_t x726 = x722;
int32_t x727 = x723;
float x728 = x658[x727];
int32_t x729 = x724;
float x730 = x687[x729];
float x731 = x728 + x730;
x714[x726] = x731;
x722 += 1;
x723 += 1;
x724 += 1;

}
x715 += 50;
x716 += 50;
x717 += 50;

}
float* x743 = (float*)myMalloc(1000 * sizeof(float));;
for(int x744=0; x744 < 1000; x744++) {
x743[x744] = 0.0f;

}
float* x748 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x749 = 0;
int32_t x750 = 0;
int32_t x751 = 0;
for(int x752=0; x752 < 20; x752++) {
int32_t x753 = x750;
int32_t x754 = x751;
int32_t x755 = x749;
int32_t x756 = x755;
int32_t x757 = x753;
int32_t x758 = x754;
for(int x759=0; x759 < 50; x759++) {
int32_t x760 = x756;
int32_t x761 = x757;
float x762 = x714[x761];
int32_t x763 = x758;
float x764 = x153[x763];
float x765 = x762 + x764;
x748[x760] = x765;
x756 += 1;
x757 += 1;
x758 += 1;

}
x749 += 50;
x750 += 50;

}
float* x776 = (float*)myMalloc(1000 * sizeof(float));;
for(int x777=0; x777 < 1000; x777++) {
x776[x777] = 0.0f;

}
float* x781 = (float*)myMalloc(1000 * sizeof(float));;
for(int x782=0; x782 < 1000; x782++) {
float x783 = x748[x782];
float x784 = -1.0f * x783;
double x785 = (double)x784;
double x786 = exp(x785);
float x787 = (float)x786;
float x788 = x787 + 1.0f;
float x789 = 1.0f / x788;
x781[x782] = x789;

}
float* x793 = (float*)myMalloc(1000 * sizeof(float));;
for(int x794=0; x794 < 1000; x794++) {
x793[x794] = 0.0f;

}
// dot: WrappedArray(20, 26), WrappedArray(26, 50)
float* x799 = (float*)myMalloc(1000 * sizeof(float));;
for(int x800=0; x800 < 20; x800++) {
int32_t x804 = x800 * 26;
int32_t x814 = x800 * 50;
for(int x801=0; x801 < 50; x801++) {
float x802 = 0.0f;
for(int x803=0; x803 < 26; x803++) {
int32_t x805 = x804 + x803;
float x806 = x373[x805];
int32_t x807 = x803 * 50;
int32_t x808 = x807 + x801;
float x809 = x91[x808];
float x810 = x806 * x809;
x802 += x810;

}
float x816 = x802;
int32_t x815 = x814 + x801;
x799[x815] = x816;

}

}
float* x822 = (float*)myMalloc(1000 * sizeof(float));;
for(int x823=0; x823 < 1000; x823++) {
x822[x823] = 0.0f;

}
// dot: WrappedArray(20, 50), WrappedArray(50, 50)
float* x828 = (float*)myMalloc(1000 * sizeof(float));;
for(int x829=0; x829 < 20; x829++) {
int32_t x833 = x829 * 50;
for(int x830=0; x830 < 50; x830++) {
float x831 = 0.0f;
for(int x832=0; x832 < 50; x832++) {
int32_t x834 = x833 + x832;
float x835 = x367[x834];
int32_t x836 = x832 * 50;
int32_t x837 = x836 + x830;
float x838 = x104[x837];
float x839 = x835 * x838;
x831 += x839;

}
float x844 = x831;
int32_t x843 = x833 + x830;
x828[x843] = x844;

}

}
float* x850 = (float*)myMalloc(1000 * sizeof(float));;
for(int x851=0; x851 < 1000; x851++) {
x850[x851] = 0.0f;

}
float* x855 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x856 = 0;
int32_t x857 = 0;
int32_t x858 = 0;
for(int x859=0; x859 < 20; x859++) {
int32_t x860 = x857;
int32_t x861 = x858;
int32_t x862 = x856;
int32_t x863 = x862;
int32_t x864 = x860;
int32_t x865 = x861;
for(int x866=0; x866 < 50; x866++) {
int32_t x867 = x863;
int32_t x868 = x864;
float x869 = x799[x868];
int32_t x870 = x865;
float x871 = x828[x870];
float x872 = x869 + x871;
x855[x867] = x872;
x863 += 1;
x864 += 1;
x865 += 1;

}
x856 += 50;
x857 += 50;
x858 += 50;

}
float* x884 = (float*)myMalloc(1000 * sizeof(float));;
for(int x885=0; x885 < 1000; x885++) {
x884[x885] = 0.0f;

}
float* x889 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x890 = 0;
int32_t x891 = 0;
int32_t x892 = 0;
for(int x893=0; x893 < 20; x893++) {
int32_t x894 = x891;
int32_t x895 = x892;
int32_t x896 = x890;
int32_t x897 = x896;
int32_t x898 = x894;
int32_t x899 = x895;
for(int x900=0; x900 < 50; x900++) {
int32_t x901 = x897;
int32_t x902 = x898;
float x903 = x855[x902];
int32_t x904 = x899;
float x905 = x117[x904];
float x906 = x903 + x905;
x889[x901] = x906;
x897 += 1;
x898 += 1;
x899 += 1;

}
x890 += 50;
x891 += 50;

}
float* x917 = (float*)myMalloc(1000 * sizeof(float));;
for(int x918=0; x918 < 1000; x918++) {
x917[x918] = 0.0f;

}
float* x922 = (float*)myMalloc(1000 * sizeof(float));;
for(int x923=0; x923 < 1000; x923++) {
float x924 = x889[x923];
double x925 = (double)x924;
double x926 = tanh(x925);
float x927 = (float)x926;
x922[x923] = x927;

}
float* x931 = (float*)myMalloc(1000 * sizeof(float));;
for(int x932=0; x932 < 1000; x932++) {
x931[x932] = 0.0f;

}
float* x936 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x937 = 0;
int32_t x938 = 0;
int32_t x939 = 0;
for(int x940=0; x940 < 20; x940++) {
int32_t x941 = x938;
int32_t x942 = x939;
int32_t x943 = x937;
int32_t x944 = x943;
int32_t x945 = x941;
int32_t x946 = x942;
for(int x947=0; x947 < 50; x947++) {
int32_t x948 = x944;
int32_t x949 = x945;
float x950 = x499[x949];
int32_t x951 = x946;
float x952 = x369[x951];
float x953 = x950 * x952;
x936[x948] = x953;
x944 += 1;
x945 += 1;
x946 += 1;

}
x937 += 50;
x938 += 50;
x939 += 50;

}
float* x965 = (float*)myMalloc(1000 * sizeof(float));;
for(int x966=0; x966 < 1000; x966++) {
x965[x966] = 0.0f;

}
float* x970 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x971 = 0;
int32_t x972 = 0;
int32_t x973 = 0;
for(int x974=0; x974 < 20; x974++) {
int32_t x975 = x972;
int32_t x976 = x973;
int32_t x977 = x971;
int32_t x978 = x977;
int32_t x979 = x975;
int32_t x980 = x976;
for(int x981=0; x981 < 50; x981++) {
int32_t x982 = x978;
int32_t x983 = x979;
float x984 = x640[x983];
int32_t x985 = x980;
float x986 = x922[x985];
float x987 = x984 * x986;
x970[x982] = x987;
x978 += 1;
x979 += 1;
x980 += 1;

}
x971 += 50;
x972 += 50;
x973 += 50;

}
float* x999 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1000=0; x1000 < 1000; x1000++) {
x999[x1000] = 0.0f;

}
float* x1004 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1005 = 0;
int32_t x1006 = 0;
int32_t x1007 = 0;
for(int x1008=0; x1008 < 20; x1008++) {
int32_t x1009 = x1006;
int32_t x1010 = x1007;
int32_t x1011 = x1005;
int32_t x1012 = x1011;
int32_t x1013 = x1009;
int32_t x1014 = x1010;
for(int x1015=0; x1015 < 50; x1015++) {
int32_t x1016 = x1012;
int32_t x1017 = x1013;
float x1018 = x936[x1017];
int32_t x1019 = x1014;
float x1020 = x970[x1019];
float x1021 = x1018 + x1020;
x1004[x1016] = x1021;
x1012 += 1;
x1013 += 1;
x1014 += 1;

}
x1005 += 50;
x1006 += 50;
x1007 += 50;

}
float* x1033 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1034=0; x1034 < 1000; x1034++) {
x1033[x1034] = 0.0f;

}
float* x1038 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1039=0; x1039 < 1000; x1039++) {
float x1040 = x1004[x1039];
double x1041 = (double)x1040;
double x1042 = tanh(x1041);
float x1043 = (float)x1042;
x1038[x1039] = x1043;

}
float* x1047 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1048=0; x1048 < 1000; x1048++) {
x1047[x1048] = 0.0f;

}
float* x1052 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1053 = 0;
int32_t x1054 = 0;
int32_t x1055 = 0;
for(int x1056=0; x1056 < 20; x1056++) {
int32_t x1057 = x1054;
int32_t x1058 = x1055;
int32_t x1059 = x1053;
int32_t x1060 = x1059;
int32_t x1061 = x1057;
int32_t x1062 = x1058;
for(int x1063=0; x1063 < 50; x1063++) {
int32_t x1064 = x1060;
int32_t x1065 = x1061;
float x1066 = x781[x1065];
int32_t x1067 = x1062;
float x1068 = x1038[x1067];
float x1069 = x1066 * x1068;
x1052[x1064] = x1069;
x1060 += 1;
x1061 += 1;
x1062 += 1;

}
x1053 += 50;
x1054 += 50;
x1055 += 50;

}
float* x1081 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1082=0; x1082 < 1000; x1082++) {
x1081[x1082] = 0.0f;

}
// dot: List(20, 50), WrappedArray(50, 26)
float* x1087 = (float*)myMalloc(520 * sizeof(float));;
for(int x1088=0; x1088 < 20; x1088++) {
int32_t x1092 = x1088 * 50;
int32_t x1102 = x1088 * 26;
for(int x1089=0; x1089 < 26; x1089++) {
float x1090 = 0.0f;
for(int x1091=0; x1091 < 50; x1091++) {
int32_t x1093 = x1092 + x1091;
float x1094 = x1052[x1093];
int32_t x1095 = x1091 * 26;
int32_t x1096 = x1095 + x1089;
float x1097 = x163[x1096];
float x1098 = x1094 * x1097;
x1090 += x1098;

}
float x1104 = x1090;
int32_t x1103 = x1102 + x1089;
x1087[x1103] = x1104;

}

}
float* x1110 = (float*)myMalloc(520 * sizeof(float));;
for(int x1112=0; x1112 < 520; x1112++) {
x1110[x1112] = 0.0f;

}
float* x1116 = (float*)myMalloc(520 * sizeof(float));;
int32_t x1117 = 0;
int32_t x1118 = 0;
int32_t x1119 = 0;
for(int x1120=0; x1120 < 20; x1120++) {
int32_t x1121 = x1118;
int32_t x1122 = x1119;
int32_t x1123 = x1117;
int32_t x1124 = x1123;
int32_t x1125 = x1121;
int32_t x1126 = x1122;
for(int x1127=0; x1127 < 26; x1127++) {
int32_t x1128 = x1124;
int32_t x1129 = x1125;
float x1130 = x1087[x1129];
int32_t x1131 = x1126;
float x1132 = x176[x1131];
float x1133 = x1130 + x1132;
x1116[x1128] = x1133;
x1124 += 1;
x1125 += 1;
x1126 += 1;

}
x1117 += 26;
x1118 += 26;

}
float* x1144 = (float*)myMalloc(520 * sizeof(float));;
for(int x1145=0; x1145 < 520; x1145++) {
x1144[x1145] = 0.0f;

}
int* x1149 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x1150=0; x1150 < 20; x1150++) {
int32_t x1151 = x1150 * 20;
int32_t x1152 = x363 + x1151;
int32_t x1153 = x274[x1152];
x1149[x1150] = x1153;

}
float* x1157 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1158 = 0;
for(int x1159=0; x1159 < 20; x1159++) {
float x1160 = -3.4028235E38f;
for(int x1161=0; x1161 < 26; x1161++) {
int32_t x1162 = x1158;
float x1163 = x1116[x1162];
float x1164 = x1160;
bool x1165 = x1163 > x1164;
if (x1165) {
float x1166 = x1116[x1162];
x1160 = x1166;
} else {
}
x1158 += 1;

}
float x1173 = x1160;
x1157[x1159] = x1173;

}
float* x1177 = (float*)myMalloc(520 * sizeof(float));;
for(int x1178=0; x1178 < 520; x1178++) {
x1177[x1178] = 0.0f;

}
int32_t x1182 = 0;
for(int x1183=0; x1183 < 20; x1183++) {
for(int x1184=0; x1184 < 26; x1184++) {
int32_t x1185 = x1182;
float x1186 = x1116[x1185];
float x1187 = x1157[x1183];
float x1188 = x1186 - x1187;
double x1189 = (double)x1188;
double x1190 = exp(x1189);
float x1191 = (float)x1190;
x1177[x1185] = x1191;
x1182 += 1;

}

}
float* x1198 = (float*)myMalloc(20 * sizeof(float));;
for(int x1199=0; x1199 < 20; x1199++) {
x1198[x1199] = 0.0f;

}
for(int x1203=0; x1203 < 20; x1203++) {
int32_t x1204 = x1203;
int32_t x1205 = x1203 * 26;
int32_t x1206 = x1205;
for(int x1207=0; x1207 < 26; x1207++) {
int32_t x1208 = x1204;
float x1209 = x1198[x1208];
int32_t x1210 = x1206;
float x1211 = x1177[x1210];
float x1212 = x1209 + x1211;
x1198[x1208] = x1212;
x1206 += 1;

}

}
x1182 = 0;
for(int x1220=0; x1220 < 20; x1220++) {
float x1221 = x1157[x1220];
float x1222 = x1198[x1220];
double x1223 = (double)x1222;
double x1224 = log(x1223);
float x1225 = (float)x1224;
float x1226 = x1221 + x1225;
for(int x1227=0; x1227 < 26; x1227++) {
int32_t x1228 = x1182;
float x1229 = x1116[x1228];
float x1230 = x1229 - x1226;
x1177[x1228] = x1230;
x1182 += 1;

}

}
float* x1237 = (float*)myMalloc(520 * sizeof(float));;
for(int x1238=0; x1238 < 520; x1238++) {
x1237[x1238] = 0.0f;

}
float* x1242 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1243 = 0;
for(int x1244=0; x1244 < 20; x1244++) {
int32_t x1245 = x1243;
int32_t x1246 = x1149[x1244];
int32_t x1247 = x1245 + x1246;
float x1248 = x1177[x1247];
float x1249 = -1.0f * x1248;
x1242[x1244] = x1249;
x1243 += 26;

}
float* x1254 = (float*)myMalloc(20 * sizeof(float));;
for(int x1255=0; x1255 < 20; x1255++) {
x1254[x1255] = 0.0f;

}
float x1259 = 0.0f;
for(int x1260=0; x1260 < 20; x1260++) {
float x1261 = x1259;
float x1262 = x1242[x1260];
float x1263 = x1261 + x1262;
x1259 = x1263;

}
float x1267 = x1259;
float* x1268 = (float*)myMalloc(1 * sizeof(float));;
x1268[0] = x1267;
float* x1270 = (float*)myMalloc(1 * sizeof(float));;
for(int x1271=0; x1271 < 1; x1271++) {
x1270[x1271] = 0.0f;

}
float* x1275 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1276 = 0;
int32_t x1277 = 0;
int32_t x1278 = 0;
int32_t x1279 = x1276;
int32_t x1280 = x1277;
float x1281 = x365[x1280];
int32_t x1282 = x1278;
float x1283 = x1268[x1282];
float x1284 = x1281 + x1283;
x1275[x1279] = x1284;
x1276 += 1;
float* x1287 = (float*)myMalloc(1 * sizeof(float));;
for(int x1288=0; x1288 < 1; x1288++) {
x1287[x1288] = 0.0f;

}
float** x3104 = (float**)myMalloc(6 * sizeof(float*));;
x3104[0] = x1275;
x3104[1] = x1287;
x3104[2] = x1052;
x3104[3] = x1081;
x3104[4] = x1004;
x3104[5] = x1033;
int32_t x1292 = x363 + 1;
x1293(x1292,x3104);
int32_t x3113 = 0;
int32_t x3114 = 0;
int32_t x3115 = 0;
int32_t x3116 = x3113;
float x3117 = x366[x3116];
float x3118 = x365[x3116];
int32_t x3119 = x3114;
float x3120 = x1268[x3119];
int32_t x3121 = x3115;
float x3122 = x1287[x3121];
float x3123 = x3117 + x3122;
x366[x3116] = x3123;
float x3125 = x1270[x3119];
float x3126 = x365[x3116];
float x3127 = x1268[x3119];
float x3128 = x1287[x3121];
float x3129 = x3125 + x3128;
x1270[x3119] = x3129;
x3115 += 1;
// += tensor of dim 0
float x3133 = x1270[0];
for(int x3134=0; x3134 < 20; x3134++) {
float x3135 = x1254[x3134];
float x3136 = x3135 + x3133;
x1254[x3134] = x3136;

}
int32_t x3140 = 0;
for(int x3141=0; x3141 < 20; x3141++) {
int32_t x3142 = x3140;
int32_t x3143 = x1149[x3141];
int32_t x3144 = x3142 + x3143;
float x3145 = x1237[x3144];
float x3146 = x1254[x3141];
float x3147 = -1.0f * x3146;
float x3148 = x3145 + x3147;
x1237[x3144] = x3148;
x3140 += 26;

}
float* x3153 = (float*)myMalloc(20 * sizeof(float));;
for(int x3154=0; x3154 < 20; x3154++) {
x3153[x3154] = 0.0f;

}
for(int x3158=0; x3158 < 20; x3158++) {
int32_t x3159 = x3158;
int32_t x3160 = x3158 * 26;
int32_t x3161 = x3160;
for(int x3162=0; x3162 < 26; x3162++) {
int32_t x3163 = x3159;
float x3164 = x3153[x3163];
int32_t x3165 = x3161;
float x3166 = x1237[x3165];
float x3167 = x3164 + x3166;
x3153[x3163] = x3167;
x3161 += 1;

}

}
int32_t x3174 = 0;
for(int x3175=0; x3175 < 20; x3175++) {
for(int x3176=0; x3176 < 26; x3176++) {
int32_t x3177 = x3174;
float x3178 = x1144[x3177];
float x3179 = x1237[x3177];
float x3180 = x1177[x3177];
float x3184 = x3153[x3175];
double x3181 = (double)x3180;
double x3182 = exp(x3181);
float x3183 = (float)x3182;
float x3185 = x3183 * x3184;
float x3186 = x3179 - x3185;
float x3187 = x3178 + x3186;
x1144[x3177] = x3187;
x3174 += 1;

}

}
int32_t x3194 = 0;
int32_t x3195 = 0;
int32_t x3196 = 0;
for(int x3197=0; x3197 < 20; x3197++) {
int32_t x3198 = x3194;
int32_t x3199 = x3195;
int32_t x3200 = x3196;
int32_t x3201 = x3198;
int32_t x3202 = x3199;
int32_t x3203 = x3200;
for(int x3204=0; x3204 < 26; x3204++) {
int32_t x3205 = x3201;
float x3206 = x1110[x3205];
float x3207 = x1087[x3205];
int32_t x3208 = x3202;
float x3209 = x176[x3208];
int32_t x3210 = x3203;
float x3211 = x1144[x3210];
float x3212 = x3206 + x3211;
x1110[x3205] = x3212;
float x3214 = x182[x3208];
float x3215 = x1087[x3205];
float x3216 = x176[x3208];
float x3217 = x1144[x3210];
float x3218 = x3214 + x3217;
x182[x3208] = x3218;
x3203 += 1;
x3201 += 1;
x3202 += 1;

}
x3196 += 26;
x3194 += 26;

}
for(int x3229=0; x3229 < 20; x3229++) {
int32_t x3232 = x3229 * 50;
int32_t x3238 = x3229 * 26;
for(int x3230=0; x3230 < 26; x3230++) {
int32_t x3239 = x3238 + x3230;
for(int x3231=0; x3231 < 50; x3231++) {
int32_t x3233 = x3232 + x3231;
float x3234 = x1081[x3233];
int32_t x3235 = x3231 * 26;
int32_t x3236 = x3235 + x3230;
float x3237 = x163[x3236];
float x3240 = x1110[x3239];
float x3241 = x3237 * x3240;
float x3242 = x3234 + x3241;
x1081[x3233] = x3242;
float x3244 = x171[x3236];
float x3245 = x1052[x3233];
float x3246 = x1110[x3239];
float x3247 = x3245 * x3246;
float x3248 = x3244 + x3247;
x171[x3236] = x3248;

}

}

}
int32_t x3256 = 0;
int32_t x3257 = 0;
int32_t x3258 = 0;
for(int x3259=0; x3259 < 20; x3259++) {
int32_t x3260 = x3256;
int32_t x3261 = x3257;
int32_t x3262 = x3258;
int32_t x3263 = x3260;
int32_t x3264 = x3261;
int32_t x3265 = x3262;
for(int x3266=0; x3266 < 50; x3266++) {
int32_t x3267 = x3263;
float x3268 = x793[x3267];
float x3269 = x781[x3267];
int32_t x3270 = x3264;
float x3271 = x1038[x3270];
int32_t x3272 = x3265;
float x3273 = x1081[x3272];
float x3274 = x3273 * x3271;
float x3275 = x3268 + x3274;
x793[x3267] = x3275;
float x3277 = x1047[x3270];
float x3278 = x781[x3267];
float x3279 = x1038[x3270];
float x3280 = x1081[x3272];
float x3281 = x3280 * x3278;
float x3282 = x3277 + x3281;
x1047[x3270] = x3282;
x3265 += 1;
x3263 += 1;
x3264 += 1;

}
x3258 += 50;
x3256 += 50;
x3257 += 50;

}
for(int x3294=0; x3294 < 1000; x3294++) {
float x3295 = x1033[x3294];
float x3296 = x1038[x3294];
float x3299 = x1047[x3294];
float x3297 = x3296 * x3296;
float x3298 = 1.0f - x3297;
float x3300 = x3298 * x3299;
float x3301 = x3295 + x3300;
x1033[x3294] = x3301;

}
int32_t x3305 = 0;
int32_t x3306 = 0;
int32_t x3307 = 0;
for(int x3308=0; x3308 < 20; x3308++) {
int32_t x3309 = x3305;
int32_t x3310 = x3306;
int32_t x3311 = x3307;
int32_t x3312 = x3309;
int32_t x3313 = x3310;
int32_t x3314 = x3311;
for(int x3315=0; x3315 < 50; x3315++) {
int32_t x3316 = x3312;
float x3317 = x965[x3316];
float x3318 = x936[x3316];
int32_t x3319 = x3313;
float x3320 = x970[x3319];
int32_t x3321 = x3314;
float x3322 = x1033[x3321];
float x3323 = x3317 + x3322;
x965[x3316] = x3323;
float x3325 = x999[x3319];
float x3326 = x936[x3316];
float x3327 = x970[x3319];
float x3328 = x1033[x3321];
float x3329 = x3325 + x3328;
x999[x3319] = x3329;
x3314 += 1;
x3312 += 1;
x3313 += 1;

}
x3307 += 50;
x3305 += 50;
x3306 += 50;

}
int32_t x3341 = 0;
int32_t x3342 = 0;
int32_t x3343 = 0;
for(int x3344=0; x3344 < 20; x3344++) {
int32_t x3345 = x3341;
int32_t x3346 = x3342;
int32_t x3347 = x3343;
int32_t x3348 = x3345;
int32_t x3349 = x3346;
int32_t x3350 = x3347;
for(int x3351=0; x3351 < 50; x3351++) {
int32_t x3352 = x3348;
float x3353 = x652[x3352];
float x3354 = x640[x3352];
int32_t x3355 = x3349;
float x3356 = x922[x3355];
int32_t x3357 = x3350;
float x3358 = x999[x3357];
float x3359 = x3358 * x3356;
float x3360 = x3353 + x3359;
x652[x3352] = x3360;
float x3362 = x931[x3355];
float x3363 = x640[x3352];
float x3364 = x922[x3355];
float x3365 = x999[x3357];
float x3366 = x3365 * x3363;
float x3367 = x3362 + x3366;
x931[x3355] = x3367;
x3350 += 1;
x3348 += 1;
x3349 += 1;

}
x3343 += 50;
x3341 += 50;
x3342 += 50;

}
int32_t x3379 = 0;
int32_t x3380 = 0;
int32_t x3381 = 0;
for(int x3382=0; x3382 < 20; x3382++) {
int32_t x3383 = x3379;
int32_t x3384 = x3380;
int32_t x3385 = x3381;
int32_t x3386 = x3383;
int32_t x3387 = x3384;
int32_t x3388 = x3385;
for(int x3389=0; x3389 < 50; x3389++) {
int32_t x3390 = x3386;
float x3391 = x511[x3390];
float x3392 = x499[x3390];
int32_t x3393 = x3387;
float x3394 = x369[x3393];
int32_t x3395 = x3388;
float x3396 = x965[x3395];
float x3397 = x3396 * x3394;
float x3398 = x3391 + x3397;
x511[x3390] = x3398;
float x3400 = x370[x3393];
float x3401 = x499[x3390];
float x3402 = x369[x3393];
float x3403 = x965[x3395];
float x3404 = x3403 * x3401;
float x3405 = x3400 + x3404;
x370[x3393] = x3405;
x3388 += 1;
x3386 += 1;
x3387 += 1;

}
x3381 += 50;
x3379 += 50;
x3380 += 50;

}
for(int x3417=0; x3417 < 1000; x3417++) {
float x3418 = x917[x3417];
float x3419 = x922[x3417];
float x3422 = x931[x3417];
float x3420 = x3419 * x3419;
float x3421 = 1.0f - x3420;
float x3423 = x3421 * x3422;
float x3424 = x3418 + x3423;
x917[x3417] = x3424;

}
int32_t x3428 = 0;
int32_t x3429 = 0;
int32_t x3430 = 0;
for(int x3431=0; x3431 < 20; x3431++) {
int32_t x3432 = x3428;
int32_t x3433 = x3429;
int32_t x3434 = x3430;
int32_t x3435 = x3432;
int32_t x3436 = x3433;
int32_t x3437 = x3434;
for(int x3438=0; x3438 < 50; x3438++) {
int32_t x3439 = x3435;
float x3440 = x884[x3439];
float x3441 = x855[x3439];
int32_t x3442 = x3436;
float x3443 = x117[x3442];
int32_t x3444 = x3437;
float x3445 = x917[x3444];
float x3446 = x3440 + x3445;
x884[x3439] = x3446;
float x3448 = x122[x3442];
float x3449 = x855[x3439];
float x3450 = x117[x3442];
float x3451 = x917[x3444];
float x3452 = x3448 + x3451;
x122[x3442] = x3452;
x3437 += 1;
x3435 += 1;
x3436 += 1;

}
x3430 += 50;
x3428 += 50;

}
int32_t x3463 = 0;
int32_t x3464 = 0;
int32_t x3465 = 0;
for(int x3466=0; x3466 < 20; x3466++) {
int32_t x3467 = x3463;
int32_t x3468 = x3464;
int32_t x3469 = x3465;
int32_t x3470 = x3467;
int32_t x3471 = x3468;
int32_t x3472 = x3469;
for(int x3473=0; x3473 < 50; x3473++) {
int32_t x3474 = x3470;
float x3475 = x822[x3474];
float x3476 = x799[x3474];
int32_t x3477 = x3471;
float x3478 = x828[x3477];
int32_t x3479 = x3472;
float x3480 = x884[x3479];
float x3481 = x3475 + x3480;
x822[x3474] = x3481;
float x3483 = x850[x3477];
float x3484 = x799[x3474];
float x3485 = x828[x3477];
float x3486 = x884[x3479];
float x3487 = x3483 + x3486;
x850[x3477] = x3487;
x3472 += 1;
x3470 += 1;
x3471 += 1;

}
x3465 += 50;
x3463 += 50;
x3464 += 50;

}
for(int x3499=0; x3499 < 20; x3499++) {
int32_t x3502 = x3499 * 50;
for(int x3500=0; x3500 < 50; x3500++) {
int32_t x3508 = x3502 + x3500;
for(int x3501=0; x3501 < 50; x3501++) {
int32_t x3503 = x3502 + x3501;
float x3504 = x368[x3503];
int32_t x3505 = x3501 * 50;
int32_t x3506 = x3505 + x3500;
float x3507 = x104[x3506];
float x3509 = x850[x3508];
float x3510 = x3507 * x3509;
float x3511 = x3504 + x3510;
x368[x3503] = x3511;
float x3513 = x112[x3506];
float x3514 = x367[x3503];
float x3515 = x850[x3508];
float x3516 = x3514 * x3515;
float x3517 = x3513 + x3516;
x112[x3506] = x3517;

}

}

}
for(int x3525=0; x3525 < 20; x3525++) {
int32_t x3528 = x3525 * 26;
int32_t x3534 = x3525 * 50;
for(int x3526=0; x3526 < 50; x3526++) {
int32_t x3535 = x3534 + x3526;
for(int x3527=0; x3527 < 26; x3527++) {
int32_t x3529 = x3528 + x3527;
float x3530 = x374[x3529];
int32_t x3531 = x3527 * 50;
int32_t x3532 = x3531 + x3526;
float x3533 = x91[x3532];
float x3536 = x822[x3535];
float x3537 = x3533 * x3536;
float x3538 = x3530 + x3537;
x374[x3529] = x3538;
float x3540 = x99[x3532];
float x3541 = x373[x3529];
float x3542 = x822[x3535];
float x3543 = x3541 * x3542;
float x3544 = x3540 + x3543;
x99[x3532] = x3544;

}

}

}
for(int x3552=0; x3552 < 1000; x3552++) {
float x3553 = x776[x3552];
float x3554 = x781[x3552];
float x3557 = x793[x3552];
float x3555 = 1.0f - x3554;
float x3556 = x3555 * x3554;
float x3558 = x3556 * x3557;
float x3559 = x3553 + x3558;
x776[x3552] = x3559;

}
int32_t x3563 = 0;
int32_t x3564 = 0;
int32_t x3565 = 0;
for(int x3566=0; x3566 < 20; x3566++) {
int32_t x3567 = x3563;
int32_t x3568 = x3564;
int32_t x3569 = x3565;
int32_t x3570 = x3567;
int32_t x3571 = x3568;
int32_t x3572 = x3569;
for(int x3573=0; x3573 < 50; x3573++) {
int32_t x3574 = x3570;
float x3575 = x743[x3574];
float x3576 = x714[x3574];
int32_t x3577 = x3571;
float x3578 = x153[x3577];
int32_t x3579 = x3572;
float x3580 = x776[x3579];
float x3581 = x3575 + x3580;
x743[x3574] = x3581;
float x3583 = x158[x3577];
float x3584 = x714[x3574];
float x3585 = x153[x3577];
float x3586 = x776[x3579];
float x3587 = x3583 + x3586;
x158[x3577] = x3587;
x3572 += 1;
x3570 += 1;
x3571 += 1;

}
x3565 += 50;
x3563 += 50;

}
int32_t x3598 = 0;
int32_t x3599 = 0;
int32_t x3600 = 0;
for(int x3601=0; x3601 < 20; x3601++) {
int32_t x3602 = x3598;
int32_t x3603 = x3599;
int32_t x3604 = x3600;
int32_t x3605 = x3602;
int32_t x3606 = x3603;
int32_t x3607 = x3604;
for(int x3608=0; x3608 < 50; x3608++) {
int32_t x3609 = x3605;
float x3610 = x681[x3609];
float x3611 = x658[x3609];
int32_t x3612 = x3606;
float x3613 = x687[x3612];
int32_t x3614 = x3607;
float x3615 = x743[x3614];
float x3616 = x3610 + x3615;
x681[x3609] = x3616;
float x3618 = x709[x3612];
float x3619 = x658[x3609];
float x3620 = x687[x3612];
float x3621 = x743[x3614];
float x3622 = x3618 + x3621;
x709[x3612] = x3622;
x3607 += 1;
x3605 += 1;
x3606 += 1;

}
x3600 += 50;
x3598 += 50;
x3599 += 50;

}
for(int x3634=0; x3634 < 20; x3634++) {
int32_t x3637 = x3634 * 50;
for(int x3635=0; x3635 < 50; x3635++) {
int32_t x3643 = x3637 + x3635;
for(int x3636=0; x3636 < 50; x3636++) {
int32_t x3638 = x3637 + x3636;
float x3639 = x368[x3638];
int32_t x3640 = x3636 * 50;
int32_t x3641 = x3640 + x3635;
float x3642 = x140[x3641];
float x3644 = x709[x3643];
float x3645 = x3642 * x3644;
float x3646 = x3639 + x3645;
x368[x3638] = x3646;
float x3648 = x148[x3641];
float x3649 = x367[x3638];
float x3650 = x709[x3643];
float x3651 = x3649 * x3650;
float x3652 = x3648 + x3651;
x148[x3641] = x3652;

}

}

}
for(int x3660=0; x3660 < 20; x3660++) {
int32_t x3663 = x3660 * 26;
int32_t x3669 = x3660 * 50;
for(int x3661=0; x3661 < 50; x3661++) {
int32_t x3670 = x3669 + x3661;
for(int x3662=0; x3662 < 26; x3662++) {
int32_t x3664 = x3663 + x3662;
float x3665 = x374[x3664];
int32_t x3666 = x3662 * 50;
int32_t x3667 = x3666 + x3661;
float x3668 = x127[x3667];
float x3671 = x681[x3670];
float x3672 = x3668 * x3671;
float x3673 = x3665 + x3672;
x374[x3664] = x3673;
float x3675 = x135[x3667];
float x3676 = x373[x3664];
float x3677 = x681[x3670];
float x3678 = x3676 * x3677;
float x3679 = x3675 + x3678;
x135[x3667] = x3679;

}

}

}
for(int x3687=0; x3687 < 1000; x3687++) {
float x3688 = x635[x3687];
float x3689 = x640[x3687];
float x3692 = x652[x3687];
float x3690 = 1.0f - x3689;
float x3691 = x3690 * x3689;
float x3693 = x3691 * x3692;
float x3694 = x3688 + x3693;
x635[x3687] = x3694;

}
int32_t x3698 = 0;
int32_t x3699 = 0;
int32_t x3700 = 0;
for(int x3701=0; x3701 < 20; x3701++) {
int32_t x3702 = x3698;
int32_t x3703 = x3699;
int32_t x3704 = x3700;
int32_t x3705 = x3702;
int32_t x3706 = x3703;
int32_t x3707 = x3704;
for(int x3708=0; x3708 < 50; x3708++) {
int32_t x3709 = x3705;
float x3710 = x602[x3709];
float x3711 = x573[x3709];
int32_t x3712 = x3706;
float x3713 = x81[x3712];
int32_t x3714 = x3707;
float x3715 = x635[x3714];
float x3716 = x3710 + x3715;
x602[x3709] = x3716;
float x3718 = x86[x3712];
float x3719 = x573[x3709];
float x3720 = x81[x3712];
float x3721 = x635[x3714];
float x3722 = x3718 + x3721;
x86[x3712] = x3722;
x3707 += 1;
x3705 += 1;
x3706 += 1;

}
x3700 += 50;
x3698 += 50;

}
int32_t x3733 = 0;
int32_t x3734 = 0;
int32_t x3735 = 0;
for(int x3736=0; x3736 < 20; x3736++) {
int32_t x3737 = x3733;
int32_t x3738 = x3734;
int32_t x3739 = x3735;
int32_t x3740 = x3737;
int32_t x3741 = x3738;
int32_t x3742 = x3739;
for(int x3743=0; x3743 < 50; x3743++) {
int32_t x3744 = x3740;
float x3745 = x540[x3744];
float x3746 = x517[x3744];
int32_t x3747 = x3741;
float x3748 = x546[x3747];
int32_t x3749 = x3742;
float x3750 = x602[x3749];
float x3751 = x3745 + x3750;
x540[x3744] = x3751;
float x3753 = x568[x3747];
float x3754 = x517[x3744];
float x3755 = x546[x3747];
float x3756 = x602[x3749];
float x3757 = x3753 + x3756;
x568[x3747] = x3757;
x3742 += 1;
x3740 += 1;
x3741 += 1;

}
x3735 += 50;
x3733 += 50;
x3734 += 50;

}
for(int x3769=0; x3769 < 20; x3769++) {
int32_t x3772 = x3769 * 50;
for(int x3770=0; x3770 < 50; x3770++) {
int32_t x3778 = x3772 + x3770;
for(int x3771=0; x3771 < 50; x3771++) {
int32_t x3773 = x3772 + x3771;
float x3774 = x368[x3773];
int32_t x3775 = x3771 * 50;
int32_t x3776 = x3775 + x3770;
float x3777 = x68[x3776];
float x3779 = x568[x3778];
float x3780 = x3777 * x3779;
float x3781 = x3774 + x3780;
x368[x3773] = x3781;
float x3783 = x76[x3776];
float x3784 = x367[x3773];
float x3785 = x568[x3778];
float x3786 = x3784 * x3785;
float x3787 = x3783 + x3786;
x76[x3776] = x3787;

}

}

}
for(int x3795=0; x3795 < 20; x3795++) {
int32_t x3798 = x3795 * 26;
int32_t x3804 = x3795 * 50;
for(int x3796=0; x3796 < 50; x3796++) {
int32_t x3805 = x3804 + x3796;
for(int x3797=0; x3797 < 26; x3797++) {
int32_t x3799 = x3798 + x3797;
float x3800 = x374[x3799];
int32_t x3801 = x3797 * 50;
int32_t x3802 = x3801 + x3796;
float x3803 = x55[x3802];
float x3806 = x540[x3805];
float x3807 = x3803 * x3806;
float x3808 = x3800 + x3807;
x374[x3799] = x3808;
float x3810 = x63[x3802];
float x3811 = x373[x3799];
float x3812 = x540[x3805];
float x3813 = x3811 * x3812;
float x3814 = x3810 + x3813;
x63[x3802] = x3814;

}

}

}
for(int x3822=0; x3822 < 1000; x3822++) {
float x3823 = x494[x3822];
float x3824 = x499[x3822];
float x3827 = x511[x3822];
float x3825 = 1.0f - x3824;
float x3826 = x3825 * x3824;
float x3828 = x3826 * x3827;
float x3829 = x3823 + x3828;
x494[x3822] = x3829;

}
int32_t x3833 = 0;
int32_t x3834 = 0;
int32_t x3835 = 0;
for(int x3836=0; x3836 < 20; x3836++) {
int32_t x3837 = x3833;
int32_t x3838 = x3834;
int32_t x3839 = x3835;
int32_t x3840 = x3837;
int32_t x3841 = x3838;
int32_t x3842 = x3839;
for(int x3843=0; x3843 < 50; x3843++) {
int32_t x3844 = x3840;
float x3845 = x461[x3844];
float x3846 = x432[x3844];
int32_t x3847 = x3841;
float x3848 = x44[x3847];
int32_t x3849 = x3842;
float x3850 = x494[x3849];
float x3851 = x3845 + x3850;
x461[x3844] = x3851;
float x3853 = x50[x3847];
float x3854 = x432[x3844];
float x3855 = x44[x3847];
float x3856 = x494[x3849];
float x3857 = x3853 + x3856;
x50[x3847] = x3857;
x3842 += 1;
x3840 += 1;
x3841 += 1;

}
x3835 += 50;
x3833 += 50;

}
int32_t x3868 = 0;
int32_t x3869 = 0;
int32_t x3870 = 0;
for(int x3871=0; x3871 < 20; x3871++) {
int32_t x3872 = x3868;
int32_t x3873 = x3869;
int32_t x3874 = x3870;
int32_t x3875 = x3872;
int32_t x3876 = x3873;
int32_t x3877 = x3874;
for(int x3878=0; x3878 < 50; x3878++) {
int32_t x3879 = x3875;
float x3880 = x399[x3879];
float x3881 = x376[x3879];
int32_t x3882 = x3876;
float x3883 = x405[x3882];
int32_t x3884 = x3877;
float x3885 = x461[x3884];
float x3886 = x3880 + x3885;
x399[x3879] = x3886;
float x3888 = x427[x3882];
float x3889 = x376[x3879];
float x3890 = x405[x3882];
float x3891 = x461[x3884];
float x3892 = x3888 + x3891;
x427[x3882] = x3892;
x3877 += 1;
x3875 += 1;
x3876 += 1;

}
x3870 += 50;
x3868 += 50;
x3869 += 50;

}
for(int x3904=0; x3904 < 20; x3904++) {
int32_t x3907 = x3904 * 50;
for(int x3905=0; x3905 < 50; x3905++) {
int32_t x3913 = x3907 + x3905;
for(int x3906=0; x3906 < 50; x3906++) {
int32_t x3908 = x3907 + x3906;
float x3909 = x368[x3908];
int32_t x3910 = x3906 * 50;
int32_t x3911 = x3910 + x3905;
float x3912 = x30[x3911];
float x3914 = x427[x3913];
float x3915 = x3912 * x3914;
float x3916 = x3909 + x3915;
x368[x3908] = x3916;
float x3918 = x39[x3911];
float x3919 = x367[x3908];
float x3920 = x427[x3913];
float x3921 = x3919 * x3920;
float x3922 = x3918 + x3921;
x39[x3911] = x3922;

}

}

}
for(int x3930=0; x3930 < 20; x3930++) {
int32_t x3933 = x3930 * 26;
int32_t x3939 = x3930 * 50;
for(int x3931=0; x3931 < 50; x3931++) {
int32_t x3940 = x3939 + x3931;
for(int x3932=0; x3932 < 26; x3932++) {
int32_t x3934 = x3933 + x3932;
float x3935 = x374[x3934];
int32_t x3936 = x3932 * 50;
int32_t x3937 = x3936 + x3931;
float x3938 = x16[x3937];
float x3941 = x399[x3940];
float x3942 = x3938 * x3941;
float x3943 = x3935 + x3942;
x374[x3934] = x3943;
float x3945 = x25[x3937];
float x3946 = x373[x3934];
float x3947 = x399[x3940];
float x3948 = x3946 * x3947;
float x3949 = x3945 + x3948;
x25[x3937] = x3949;

}

}

}
} else {
float x3958 = 0.0f;
float x3959 = x3958;
float x3960 = x365[0];
float x3961 = x3959 + x3960;
x3958 = x3961;
float x3963 = x3958;
float* x3964 = (float*)myMalloc(1 * sizeof(float));;
x3964[0] = x3963;
float* x3966 = (float*)myMalloc(1 * sizeof(float));;
for(int x3967=0; x3967 < 1; x3967++) {
x3966[x3967] = 0.0f;

}
float x3971 = x3966[0];
x3966[0] = 1.0f;
float x3973 = x3964[0];
x297[0] = x3973;
// += tensor of dim 0
float x3976 = x3966[0];
float x3977 = x366[0];
float x3978 = x3977 + x3976;
x366[0] = x3978;
}
};
x360(0,x3983);
float x3992 = x297[0];
int32_t x3993 = x264 % 100;
bool x3994 = x3993 == 0;
if (x3994) {
printf("iter %d, loss %f\n",x264,x3992);
int32_t x3996 = x264 / 100;
double x3997 = (double)x3992;
x258[x3996] = x3997;
} else {
}
for(int x4001=0; x4001 < 1300; x4001++) {
float x4002 = x135[x4001];
bool x4003 = x4002 > 5.0f;
if (x4003) {
x135[x4001] = 5.0f;
} else {
}
float x4007 = x135[x4001];
bool x4008 = x4007 < -5.0f;
if (x4008) {
x135[x4001] = -5.0f;
} else {
}

}
float* x4014 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4015 = 0;
int32_t x4016 = 0;
int32_t x4017 = 0;
for(int x4018=0; x4018 < 26; x4018++) {
int32_t x4019 = x4016;
int32_t x4020 = x4017;
int32_t x4021 = x4015;
int32_t x4022 = x4021;
int32_t x4023 = x4019;
int32_t x4024 = x4020;
for(int x4025=0; x4025 < 50; x4025++) {
int32_t x4026 = x4022;
int32_t x4027 = x4023;
float x4028 = x135[x4027];
int32_t x4029 = x4024;
float x4030 = x135[x4029];
float x4031 = x4028 * x4030;
x4014[x4026] = x4031;
x4022 += 1;
x4023 += 1;
x4024 += 1;

}
x4015 += 50;
x4016 += 50;
x4017 += 50;

}
for(int x4043=0; x4043 < 1300; x4043++) {
float x4044 = x187[x4043];
float x4045 = x4014[x4043];
float x4046 = x4044 + x4045;
x187[x4043] = x4046;

}
float* x4050 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4051=0; x4051 < 1300; x4051++) {
float x4052 = x135[x4051];
float x4053 = x4052 * 0.1f;
x4050[x4051] = x4053;

}
float* x4057 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4058=0; x4058 < 1300; x4058++) {
float x4059 = x187[x4058];
float x4060 = x4059 + 1.0E-8f;
x4057[x4058] = x4060;

}
float* x4064 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4065=0; x4065 < 1300; x4065++) {
float x4066 = x4057[x4065];
double x4067 = (double)x4066;
double x4068 = sqrt(x4067);
float x4069 = (float)x4068;
x4064[x4065] = x4069;

}
float* x4073 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4074 = 0;
int32_t x4075 = 0;
int32_t x4076 = 0;
for(int x4077=0; x4077 < 26; x4077++) {
int32_t x4078 = x4075;
int32_t x4079 = x4076;
int32_t x4080 = x4074;
int32_t x4081 = x4080;
int32_t x4082 = x4078;
int32_t x4083 = x4079;
for(int x4084=0; x4084 < 50; x4084++) {
int32_t x4085 = x4081;
int32_t x4086 = x4082;
float x4087 = x4050[x4086];
int32_t x4088 = x4083;
float x4089 = x4064[x4088];
float x4090 = x4087 / x4089;
x4073[x4085] = x4090;
x4081 += 1;
x4082 += 1;
x4083 += 1;

}
x4074 += 50;
x4075 += 50;
x4076 += 50;

}
for(int x4102=0; x4102 < 1300; x4102++) {
float x4103 = x127[x4102];
float x4104 = x4073[x4102];
float x4105 = x4103 - x4104;
x127[x4102] = x4105;

}
for(int x4109=0; x4109 < 1300; x4109++) {
float x4110 = x135[x4109];
x135[x4109] = 0.0f;

}
for(int x4114=0; x4114 < 50; x4114++) {
float x4115 = x158[x4114];
bool x4116 = x4115 > 5.0f;
if (x4116) {
x158[x4114] = 5.0f;
} else {
}
float x4120 = x158[x4114];
bool x4121 = x4120 < -5.0f;
if (x4121) {
x158[x4114] = -5.0f;
} else {
}

}
float* x4127 = (float*)myMalloc(50 * sizeof(float));;
int32_t x4128 = 0;
int32_t x4129 = 0;
int32_t x4130 = 0;
for(int x4131=0; x4131 < 50; x4131++) {
int32_t x4132 = x4128;
int32_t x4133 = x4129;
float x4134 = x158[x4133];
int32_t x4135 = x4130;
float x4136 = x158[x4135];
float x4137 = x4134 * x4136;
x4127[x4132] = x4137;
x4128 += 1;
x4129 += 1;
x4130 += 1;

}
for(int x4144=0; x4144 < 50; x4144++) {
float x4145 = x192[x4144];
float x4146 = x4127[x4144];
float x4147 = x4145 + x4146;
x192[x4144] = x4147;

}
float* x4151 = (float*)myMalloc(50 * sizeof(float));;
for(int x4152=0; x4152 < 50; x4152++) {
float x4153 = x158[x4152];
float x4154 = x4153 * 0.1f;
x4151[x4152] = x4154;

}
float* x4158 = (float*)myMalloc(50 * sizeof(float));;
for(int x4159=0; x4159 < 50; x4159++) {
float x4160 = x192[x4159];
float x4161 = x4160 + 1.0E-8f;
x4158[x4159] = x4161;

}
float* x4165 = (float*)myMalloc(50 * sizeof(float));;
for(int x4166=0; x4166 < 50; x4166++) {
float x4167 = x4158[x4166];
double x4168 = (double)x4167;
double x4169 = sqrt(x4168);
float x4170 = (float)x4169;
x4165[x4166] = x4170;

}
float* x4174 = (float*)myMalloc(50 * sizeof(float));;
int32_t x4175 = 0;
int32_t x4176 = 0;
int32_t x4177 = 0;
for(int x4178=0; x4178 < 50; x4178++) {
int32_t x4179 = x4175;
int32_t x4180 = x4176;
float x4181 = x4151[x4180];
int32_t x4182 = x4177;
float x4183 = x4165[x4182];
float x4184 = x4181 / x4183;
x4174[x4179] = x4184;
x4175 += 1;
x4176 += 1;
x4177 += 1;

}
for(int x4191=0; x4191 < 50; x4191++) {
float x4192 = x153[x4191];
float x4193 = x4174[x4191];
float x4194 = x4192 - x4193;
x153[x4191] = x4194;

}
for(int x4198=0; x4198 < 50; x4198++) {
float x4199 = x158[x4198];
x158[x4198] = 0.0f;

}
for(int x4203=0; x4203 < 2500; x4203++) {
float x4204 = x148[x4203];
bool x4205 = x4204 > 5.0f;
if (x4205) {
x148[x4203] = 5.0f;
} else {
}
float x4209 = x148[x4203];
bool x4210 = x4209 < -5.0f;
if (x4210) {
x148[x4203] = -5.0f;
} else {
}

}
float* x4216 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x4217 = 0;
int32_t x4218 = 0;
int32_t x4219 = 0;
for(int x4220=0; x4220 < 50; x4220++) {
int32_t x4221 = x4218;
int32_t x4222 = x4219;
int32_t x4223 = x4217;
int32_t x4224 = x4223;
int32_t x4225 = x4221;
int32_t x4226 = x4222;
for(int x4227=0; x4227 < 50; x4227++) {
int32_t x4228 = x4224;
int32_t x4229 = x4225;
float x4230 = x148[x4229];
int32_t x4231 = x4226;
float x4232 = x148[x4231];
float x4233 = x4230 * x4232;
x4216[x4228] = x4233;
x4224 += 1;
x4225 += 1;
x4226 += 1;

}
x4217 += 50;
x4218 += 50;
x4219 += 50;

}
for(int x4245=0; x4245 < 2500; x4245++) {
float x4246 = x197[x4245];
float x4247 = x4216[x4245];
float x4248 = x4246 + x4247;
x197[x4245] = x4248;

}
float* x4252 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4253=0; x4253 < 2500; x4253++) {
float x4254 = x148[x4253];
float x4255 = x4254 * 0.1f;
x4252[x4253] = x4255;

}
float* x4259 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4260=0; x4260 < 2500; x4260++) {
float x4261 = x197[x4260];
float x4262 = x4261 + 1.0E-8f;
x4259[x4260] = x4262;

}
float* x4266 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4267=0; x4267 < 2500; x4267++) {
float x4268 = x4259[x4267];
double x4269 = (double)x4268;
double x4270 = sqrt(x4269);
float x4271 = (float)x4270;
x4266[x4267] = x4271;

}
float* x4275 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x4276 = 0;
int32_t x4277 = 0;
int32_t x4278 = 0;
for(int x4279=0; x4279 < 50; x4279++) {
int32_t x4280 = x4277;
int32_t x4281 = x4278;
int32_t x4282 = x4276;
int32_t x4283 = x4282;
int32_t x4284 = x4280;
int32_t x4285 = x4281;
for(int x4286=0; x4286 < 50; x4286++) {
int32_t x4287 = x4283;
int32_t x4288 = x4284;
float x4289 = x4252[x4288];
int32_t x4290 = x4285;
float x4291 = x4266[x4290];
float x4292 = x4289 / x4291;
x4275[x4287] = x4292;
x4283 += 1;
x4284 += 1;
x4285 += 1;

}
x4276 += 50;
x4277 += 50;
x4278 += 50;

}
for(int x4304=0; x4304 < 2500; x4304++) {
float x4305 = x140[x4304];
float x4306 = x4275[x4304];
float x4307 = x4305 - x4306;
x140[x4304] = x4307;

}
for(int x4311=0; x4311 < 2500; x4311++) {
float x4312 = x148[x4311];
x148[x4311] = 0.0f;

}
for(int x4316=0; x4316 < 1300; x4316++) {
float x4317 = x63[x4316];
bool x4318 = x4317 > 5.0f;
if (x4318) {
x63[x4316] = 5.0f;
} else {
}
float x4322 = x63[x4316];
bool x4323 = x4322 < -5.0f;
if (x4323) {
x63[x4316] = -5.0f;
} else {
}

}
float* x4329 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4330 = 0;
int32_t x4331 = 0;
int32_t x4332 = 0;
for(int x4333=0; x4333 < 26; x4333++) {
int32_t x4334 = x4331;
int32_t x4335 = x4332;
int32_t x4336 = x4330;
int32_t x4337 = x4336;
int32_t x4338 = x4334;
int32_t x4339 = x4335;
for(int x4340=0; x4340 < 50; x4340++) {
int32_t x4341 = x4337;
int32_t x4342 = x4338;
float x4343 = x63[x4342];
int32_t x4344 = x4339;
float x4345 = x63[x4344];
float x4346 = x4343 * x4345;
x4329[x4341] = x4346;
x4337 += 1;
x4338 += 1;
x4339 += 1;

}
x4330 += 50;
x4331 += 50;
x4332 += 50;

}
for(int x4358=0; x4358 < 1300; x4358++) {
float x4359 = x202[x4358];
float x4360 = x4329[x4358];
float x4361 = x4359 + x4360;
x202[x4358] = x4361;

}
float* x4365 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4366=0; x4366 < 1300; x4366++) {
float x4367 = x63[x4366];
float x4368 = x4367 * 0.1f;
x4365[x4366] = x4368;

}
float* x4372 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4373=0; x4373 < 1300; x4373++) {
float x4374 = x202[x4373];
float x4375 = x4374 + 1.0E-8f;
x4372[x4373] = x4375;

}
float* x4379 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4380=0; x4380 < 1300; x4380++) {
float x4381 = x4372[x4380];
double x4382 = (double)x4381;
double x4383 = sqrt(x4382);
float x4384 = (float)x4383;
x4379[x4380] = x4384;

}
float* x4388 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4389 = 0;
int32_t x4390 = 0;
int32_t x4391 = 0;
for(int x4392=0; x4392 < 26; x4392++) {
int32_t x4393 = x4390;
int32_t x4394 = x4391;
int32_t x4395 = x4389;
int32_t x4396 = x4395;
int32_t x4397 = x4393;
int32_t x4398 = x4394;
for(int x4399=0; x4399 < 50; x4399++) {
int32_t x4400 = x4396;
int32_t x4401 = x4397;
float x4402 = x4365[x4401];
int32_t x4403 = x4398;
float x4404 = x4379[x4403];
float x4405 = x4402 / x4404;
x4388[x4400] = x4405;
x4396 += 1;
x4397 += 1;
x4398 += 1;

}
x4389 += 50;
x4390 += 50;
x4391 += 50;

}
for(int x4417=0; x4417 < 1300; x4417++) {
float x4418 = x55[x4417];
float x4419 = x4388[x4417];
float x4420 = x4418 - x4419;
x55[x4417] = x4420;

}
for(int x4424=0; x4424 < 1300; x4424++) {
float x4425 = x63[x4424];
x63[x4424] = 0.0f;

}
for(int x4429=0; x4429 < 50; x4429++) {
float x4430 = x86[x4429];
bool x4431 = x4430 > 5.0f;
if (x4431) {
x86[x4429] = 5.0f;
} else {
}
float x4435 = x86[x4429];
bool x4436 = x4435 < -5.0f;
if (x4436) {
x86[x4429] = -5.0f;
} else {
}

}
float* x4442 = (float*)myMalloc(50 * sizeof(float));;
int32_t x4443 = 0;
int32_t x4444 = 0;
int32_t x4445 = 0;
for(int x4446=0; x4446 < 50; x4446++) {
int32_t x4447 = x4443;
int32_t x4448 = x4444;
float x4449 = x86[x4448];
int32_t x4450 = x4445;
float x4451 = x86[x4450];
float x4452 = x4449 * x4451;
x4442[x4447] = x4452;
x4443 += 1;
x4444 += 1;
x4445 += 1;

}
for(int x4459=0; x4459 < 50; x4459++) {
float x4460 = x207[x4459];
float x4461 = x4442[x4459];
float x4462 = x4460 + x4461;
x207[x4459] = x4462;

}
float* x4466 = (float*)myMalloc(50 * sizeof(float));;
for(int x4467=0; x4467 < 50; x4467++) {
float x4468 = x86[x4467];
float x4469 = x4468 * 0.1f;
x4466[x4467] = x4469;

}
float* x4473 = (float*)myMalloc(50 * sizeof(float));;
for(int x4474=0; x4474 < 50; x4474++) {
float x4475 = x207[x4474];
float x4476 = x4475 + 1.0E-8f;
x4473[x4474] = x4476;

}
float* x4480 = (float*)myMalloc(50 * sizeof(float));;
for(int x4481=0; x4481 < 50; x4481++) {
float x4482 = x4473[x4481];
double x4483 = (double)x4482;
double x4484 = sqrt(x4483);
float x4485 = (float)x4484;
x4480[x4481] = x4485;

}
float* x4489 = (float*)myMalloc(50 * sizeof(float));;
int32_t x4490 = 0;
int32_t x4491 = 0;
int32_t x4492 = 0;
for(int x4493=0; x4493 < 50; x4493++) {
int32_t x4494 = x4490;
int32_t x4495 = x4491;
float x4496 = x4466[x4495];
int32_t x4497 = x4492;
float x4498 = x4480[x4497];
float x4499 = x4496 / x4498;
x4489[x4494] = x4499;
x4490 += 1;
x4491 += 1;
x4492 += 1;

}
for(int x4506=0; x4506 < 50; x4506++) {
float x4507 = x81[x4506];
float x4508 = x4489[x4506];
float x4509 = x4507 - x4508;
x81[x4506] = x4509;

}
for(int x4513=0; x4513 < 50; x4513++) {
float x4514 = x86[x4513];
x86[x4513] = 0.0f;

}
for(int x4518=0; x4518 < 2500; x4518++) {
float x4519 = x76[x4518];
bool x4520 = x4519 > 5.0f;
if (x4520) {
x76[x4518] = 5.0f;
} else {
}
float x4524 = x76[x4518];
bool x4525 = x4524 < -5.0f;
if (x4525) {
x76[x4518] = -5.0f;
} else {
}

}
float* x4531 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x4532 = 0;
int32_t x4533 = 0;
int32_t x4534 = 0;
for(int x4535=0; x4535 < 50; x4535++) {
int32_t x4536 = x4533;
int32_t x4537 = x4534;
int32_t x4538 = x4532;
int32_t x4539 = x4538;
int32_t x4540 = x4536;
int32_t x4541 = x4537;
for(int x4542=0; x4542 < 50; x4542++) {
int32_t x4543 = x4539;
int32_t x4544 = x4540;
float x4545 = x76[x4544];
int32_t x4546 = x4541;
float x4547 = x76[x4546];
float x4548 = x4545 * x4547;
x4531[x4543] = x4548;
x4539 += 1;
x4540 += 1;
x4541 += 1;

}
x4532 += 50;
x4533 += 50;
x4534 += 50;

}
for(int x4560=0; x4560 < 2500; x4560++) {
float x4561 = x212[x4560];
float x4562 = x4531[x4560];
float x4563 = x4561 + x4562;
x212[x4560] = x4563;

}
float* x4567 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4568=0; x4568 < 2500; x4568++) {
float x4569 = x76[x4568];
float x4570 = x4569 * 0.1f;
x4567[x4568] = x4570;

}
float* x4574 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4575=0; x4575 < 2500; x4575++) {
float x4576 = x212[x4575];
float x4577 = x4576 + 1.0E-8f;
x4574[x4575] = x4577;

}
float* x4581 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4582=0; x4582 < 2500; x4582++) {
float x4583 = x4574[x4582];
double x4584 = (double)x4583;
double x4585 = sqrt(x4584);
float x4586 = (float)x4585;
x4581[x4582] = x4586;

}
float* x4590 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x4591 = 0;
int32_t x4592 = 0;
int32_t x4593 = 0;
for(int x4594=0; x4594 < 50; x4594++) {
int32_t x4595 = x4592;
int32_t x4596 = x4593;
int32_t x4597 = x4591;
int32_t x4598 = x4597;
int32_t x4599 = x4595;
int32_t x4600 = x4596;
for(int x4601=0; x4601 < 50; x4601++) {
int32_t x4602 = x4598;
int32_t x4603 = x4599;
float x4604 = x4567[x4603];
int32_t x4605 = x4600;
float x4606 = x4581[x4605];
float x4607 = x4604 / x4606;
x4590[x4602] = x4607;
x4598 += 1;
x4599 += 1;
x4600 += 1;

}
x4591 += 50;
x4592 += 50;
x4593 += 50;

}
for(int x4619=0; x4619 < 2500; x4619++) {
float x4620 = x68[x4619];
float x4621 = x4590[x4619];
float x4622 = x4620 - x4621;
x68[x4619] = x4622;

}
for(int x4626=0; x4626 < 2500; x4626++) {
float x4627 = x76[x4626];
x76[x4626] = 0.0f;

}
for(int x4631=0; x4631 < 26; x4631++) {
float x4632 = x182[x4631];
bool x4633 = x4632 > 5.0f;
if (x4633) {
x182[x4631] = 5.0f;
} else {
}
float x4637 = x182[x4631];
bool x4638 = x4637 < -5.0f;
if (x4638) {
x182[x4631] = -5.0f;
} else {
}

}
float* x4644 = (float*)myMalloc(26 * sizeof(float));;
int32_t x4645 = 0;
int32_t x4646 = 0;
int32_t x4647 = 0;
for(int x4648=0; x4648 < 26; x4648++) {
int32_t x4649 = x4645;
int32_t x4650 = x4646;
float x4651 = x182[x4650];
int32_t x4652 = x4647;
float x4653 = x182[x4652];
float x4654 = x4651 * x4653;
x4644[x4649] = x4654;
x4645 += 1;
x4646 += 1;
x4647 += 1;

}
for(int x4661=0; x4661 < 26; x4661++) {
float x4662 = x217[x4661];
float x4663 = x4644[x4661];
float x4664 = x4662 + x4663;
x217[x4661] = x4664;

}
float* x4668 = (float*)myMalloc(26 * sizeof(float));;
for(int x4669=0; x4669 < 26; x4669++) {
float x4670 = x182[x4669];
float x4671 = x4670 * 0.1f;
x4668[x4669] = x4671;

}
float* x4675 = (float*)myMalloc(26 * sizeof(float));;
for(int x4676=0; x4676 < 26; x4676++) {
float x4677 = x217[x4676];
float x4678 = x4677 + 1.0E-8f;
x4675[x4676] = x4678;

}
float* x4682 = (float*)myMalloc(26 * sizeof(float));;
for(int x4683=0; x4683 < 26; x4683++) {
float x4684 = x4675[x4683];
double x4685 = (double)x4684;
double x4686 = sqrt(x4685);
float x4687 = (float)x4686;
x4682[x4683] = x4687;

}
float* x4691 = (float*)myMalloc(26 * sizeof(float));;
int32_t x4692 = 0;
int32_t x4693 = 0;
int32_t x4694 = 0;
for(int x4695=0; x4695 < 26; x4695++) {
int32_t x4696 = x4692;
int32_t x4697 = x4693;
float x4698 = x4668[x4697];
int32_t x4699 = x4694;
float x4700 = x4682[x4699];
float x4701 = x4698 / x4700;
x4691[x4696] = x4701;
x4692 += 1;
x4693 += 1;
x4694 += 1;

}
for(int x4708=0; x4708 < 26; x4708++) {
float x4709 = x176[x4708];
float x4710 = x4691[x4708];
float x4711 = x4709 - x4710;
x176[x4708] = x4711;

}
for(int x4715=0; x4715 < 26; x4715++) {
float x4716 = x182[x4715];
x182[x4715] = 0.0f;

}
for(int x4720=0; x4720 < 1300; x4720++) {
float x4721 = x171[x4720];
bool x4722 = x4721 > 5.0f;
if (x4722) {
x171[x4720] = 5.0f;
} else {
}
float x4726 = x171[x4720];
bool x4727 = x4726 < -5.0f;
if (x4727) {
x171[x4720] = -5.0f;
} else {
}

}
float* x4733 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4734 = 0;
int32_t x4735 = 0;
int32_t x4736 = 0;
for(int x4737=0; x4737 < 50; x4737++) {
int32_t x4738 = x4735;
int32_t x4739 = x4736;
int32_t x4740 = x4734;
int32_t x4741 = x4740;
int32_t x4742 = x4738;
int32_t x4743 = x4739;
for(int x4744=0; x4744 < 26; x4744++) {
int32_t x4745 = x4741;
int32_t x4746 = x4742;
float x4747 = x171[x4746];
int32_t x4748 = x4743;
float x4749 = x171[x4748];
float x4750 = x4747 * x4749;
x4733[x4745] = x4750;
x4741 += 1;
x4742 += 1;
x4743 += 1;

}
x4734 += 26;
x4735 += 26;
x4736 += 26;

}
for(int x4762=0; x4762 < 1300; x4762++) {
float x4763 = x222[x4762];
float x4764 = x4733[x4762];
float x4765 = x4763 + x4764;
x222[x4762] = x4765;

}
float* x4769 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4770=0; x4770 < 1300; x4770++) {
float x4771 = x171[x4770];
float x4772 = x4771 * 0.1f;
x4769[x4770] = x4772;

}
float* x4776 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4777=0; x4777 < 1300; x4777++) {
float x4778 = x222[x4777];
float x4779 = x4778 + 1.0E-8f;
x4776[x4777] = x4779;

}
float* x4783 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4784=0; x4784 < 1300; x4784++) {
float x4785 = x4776[x4784];
double x4786 = (double)x4785;
double x4787 = sqrt(x4786);
float x4788 = (float)x4787;
x4783[x4784] = x4788;

}
float* x4792 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4793 = 0;
int32_t x4794 = 0;
int32_t x4795 = 0;
for(int x4796=0; x4796 < 50; x4796++) {
int32_t x4797 = x4794;
int32_t x4798 = x4795;
int32_t x4799 = x4793;
int32_t x4800 = x4799;
int32_t x4801 = x4797;
int32_t x4802 = x4798;
for(int x4803=0; x4803 < 26; x4803++) {
int32_t x4804 = x4800;
int32_t x4805 = x4801;
float x4806 = x4769[x4805];
int32_t x4807 = x4802;
float x4808 = x4783[x4807];
float x4809 = x4806 / x4808;
x4792[x4804] = x4809;
x4800 += 1;
x4801 += 1;
x4802 += 1;

}
x4793 += 26;
x4794 += 26;
x4795 += 26;

}
for(int x4821=0; x4821 < 1300; x4821++) {
float x4822 = x163[x4821];
float x4823 = x4792[x4821];
float x4824 = x4822 - x4823;
x163[x4821] = x4824;

}
for(int x4828=0; x4828 < 1300; x4828++) {
float x4829 = x171[x4828];
x171[x4828] = 0.0f;

}
for(int x4833=0; x4833 < 1300; x4833++) {
float x4834 = x25[x4833];
bool x4835 = x4834 > 5.0f;
if (x4835) {
x25[x4833] = 5.0f;
} else {
}
float x4839 = x25[x4833];
bool x4840 = x4839 < -5.0f;
if (x4840) {
x25[x4833] = -5.0f;
} else {
}

}
float* x4846 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4847 = 0;
int32_t x4848 = 0;
int32_t x4849 = 0;
for(int x4850=0; x4850 < 26; x4850++) {
int32_t x4851 = x4848;
int32_t x4852 = x4849;
int32_t x4853 = x4847;
int32_t x4854 = x4853;
int32_t x4855 = x4851;
int32_t x4856 = x4852;
for(int x4857=0; x4857 < 50; x4857++) {
int32_t x4858 = x4854;
int32_t x4859 = x4855;
float x4860 = x25[x4859];
int32_t x4861 = x4856;
float x4862 = x25[x4861];
float x4863 = x4860 * x4862;
x4846[x4858] = x4863;
x4854 += 1;
x4855 += 1;
x4856 += 1;

}
x4847 += 50;
x4848 += 50;
x4849 += 50;

}
for(int x4875=0; x4875 < 1300; x4875++) {
float x4876 = x227[x4875];
float x4877 = x4846[x4875];
float x4878 = x4876 + x4877;
x227[x4875] = x4878;

}
float* x4882 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4883=0; x4883 < 1300; x4883++) {
float x4884 = x25[x4883];
float x4885 = x4884 * 0.1f;
x4882[x4883] = x4885;

}
float* x4889 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4890=0; x4890 < 1300; x4890++) {
float x4891 = x227[x4890];
float x4892 = x4891 + 1.0E-8f;
x4889[x4890] = x4892;

}
float* x4896 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4897=0; x4897 < 1300; x4897++) {
float x4898 = x4889[x4897];
double x4899 = (double)x4898;
double x4900 = sqrt(x4899);
float x4901 = (float)x4900;
x4896[x4897] = x4901;

}
float* x4905 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4906 = 0;
int32_t x4907 = 0;
int32_t x4908 = 0;
for(int x4909=0; x4909 < 26; x4909++) {
int32_t x4910 = x4907;
int32_t x4911 = x4908;
int32_t x4912 = x4906;
int32_t x4913 = x4912;
int32_t x4914 = x4910;
int32_t x4915 = x4911;
for(int x4916=0; x4916 < 50; x4916++) {
int32_t x4917 = x4913;
int32_t x4918 = x4914;
float x4919 = x4882[x4918];
int32_t x4920 = x4915;
float x4921 = x4896[x4920];
float x4922 = x4919 / x4921;
x4905[x4917] = x4922;
x4913 += 1;
x4914 += 1;
x4915 += 1;

}
x4906 += 50;
x4907 += 50;
x4908 += 50;

}
for(int x4934=0; x4934 < 1300; x4934++) {
float x4935 = x16[x4934];
float x4936 = x4905[x4934];
float x4937 = x4935 - x4936;
x16[x4934] = x4937;

}
for(int x4941=0; x4941 < 1300; x4941++) {
float x4942 = x25[x4941];
x25[x4941] = 0.0f;

}
for(int x4946=0; x4946 < 50; x4946++) {
float x4947 = x50[x4946];
bool x4948 = x4947 > 5.0f;
if (x4948) {
x50[x4946] = 5.0f;
} else {
}
float x4952 = x50[x4946];
bool x4953 = x4952 < -5.0f;
if (x4953) {
x50[x4946] = -5.0f;
} else {
}

}
float* x4959 = (float*)myMalloc(50 * sizeof(float));;
int32_t x4960 = 0;
int32_t x4961 = 0;
int32_t x4962 = 0;
for(int x4963=0; x4963 < 50; x4963++) {
int32_t x4964 = x4960;
int32_t x4965 = x4961;
float x4966 = x50[x4965];
int32_t x4967 = x4962;
float x4968 = x50[x4967];
float x4969 = x4966 * x4968;
x4959[x4964] = x4969;
x4960 += 1;
x4961 += 1;
x4962 += 1;

}
for(int x4976=0; x4976 < 50; x4976++) {
float x4977 = x232[x4976];
float x4978 = x4959[x4976];
float x4979 = x4977 + x4978;
x232[x4976] = x4979;

}
float* x4983 = (float*)myMalloc(50 * sizeof(float));;
for(int x4984=0; x4984 < 50; x4984++) {
float x4985 = x50[x4984];
float x4986 = x4985 * 0.1f;
x4983[x4984] = x4986;

}
float* x4990 = (float*)myMalloc(50 * sizeof(float));;
for(int x4991=0; x4991 < 50; x4991++) {
float x4992 = x232[x4991];
float x4993 = x4992 + 1.0E-8f;
x4990[x4991] = x4993;

}
float* x4997 = (float*)myMalloc(50 * sizeof(float));;
for(int x4998=0; x4998 < 50; x4998++) {
float x4999 = x4990[x4998];
double x5000 = (double)x4999;
double x5001 = sqrt(x5000);
float x5002 = (float)x5001;
x4997[x4998] = x5002;

}
float* x5006 = (float*)myMalloc(50 * sizeof(float));;
int32_t x5007 = 0;
int32_t x5008 = 0;
int32_t x5009 = 0;
for(int x5010=0; x5010 < 50; x5010++) {
int32_t x5011 = x5007;
int32_t x5012 = x5008;
float x5013 = x4983[x5012];
int32_t x5014 = x5009;
float x5015 = x4997[x5014];
float x5016 = x5013 / x5015;
x5006[x5011] = x5016;
x5007 += 1;
x5008 += 1;
x5009 += 1;

}
for(int x5023=0; x5023 < 50; x5023++) {
float x5024 = x44[x5023];
float x5025 = x5006[x5023];
float x5026 = x5024 - x5025;
x44[x5023] = x5026;

}
for(int x5030=0; x5030 < 50; x5030++) {
float x5031 = x50[x5030];
x50[x5030] = 0.0f;

}
for(int x5035=0; x5035 < 2500; x5035++) {
float x5036 = x39[x5035];
bool x5037 = x5036 > 5.0f;
if (x5037) {
x39[x5035] = 5.0f;
} else {
}
float x5041 = x39[x5035];
bool x5042 = x5041 < -5.0f;
if (x5042) {
x39[x5035] = -5.0f;
} else {
}

}
float* x5048 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x5049 = 0;
int32_t x5050 = 0;
int32_t x5051 = 0;
for(int x5052=0; x5052 < 50; x5052++) {
int32_t x5053 = x5050;
int32_t x5054 = x5051;
int32_t x5055 = x5049;
int32_t x5056 = x5055;
int32_t x5057 = x5053;
int32_t x5058 = x5054;
for(int x5059=0; x5059 < 50; x5059++) {
int32_t x5060 = x5056;
int32_t x5061 = x5057;
float x5062 = x39[x5061];
int32_t x5063 = x5058;
float x5064 = x39[x5063];
float x5065 = x5062 * x5064;
x5048[x5060] = x5065;
x5056 += 1;
x5057 += 1;
x5058 += 1;

}
x5049 += 50;
x5050 += 50;
x5051 += 50;

}
for(int x5077=0; x5077 < 2500; x5077++) {
float x5078 = x237[x5077];
float x5079 = x5048[x5077];
float x5080 = x5078 + x5079;
x237[x5077] = x5080;

}
float* x5084 = (float*)myMalloc(2500 * sizeof(float));;
for(int x5085=0; x5085 < 2500; x5085++) {
float x5086 = x39[x5085];
float x5087 = x5086 * 0.1f;
x5084[x5085] = x5087;

}
float* x5091 = (float*)myMalloc(2500 * sizeof(float));;
for(int x5092=0; x5092 < 2500; x5092++) {
float x5093 = x237[x5092];
float x5094 = x5093 + 1.0E-8f;
x5091[x5092] = x5094;

}
float* x5098 = (float*)myMalloc(2500 * sizeof(float));;
for(int x5099=0; x5099 < 2500; x5099++) {
float x5100 = x5091[x5099];
double x5101 = (double)x5100;
double x5102 = sqrt(x5101);
float x5103 = (float)x5102;
x5098[x5099] = x5103;

}
float* x5107 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x5108 = 0;
int32_t x5109 = 0;
int32_t x5110 = 0;
for(int x5111=0; x5111 < 50; x5111++) {
int32_t x5112 = x5109;
int32_t x5113 = x5110;
int32_t x5114 = x5108;
int32_t x5115 = x5114;
int32_t x5116 = x5112;
int32_t x5117 = x5113;
for(int x5118=0; x5118 < 50; x5118++) {
int32_t x5119 = x5115;
int32_t x5120 = x5116;
float x5121 = x5084[x5120];
int32_t x5122 = x5117;
float x5123 = x5098[x5122];
float x5124 = x5121 / x5123;
x5107[x5119] = x5124;
x5115 += 1;
x5116 += 1;
x5117 += 1;

}
x5108 += 50;
x5109 += 50;
x5110 += 50;

}
for(int x5136=0; x5136 < 2500; x5136++) {
float x5137 = x30[x5136];
float x5138 = x5107[x5136];
float x5139 = x5137 - x5138;
x30[x5136] = x5139;

}
for(int x5143=0; x5143 < 2500; x5143++) {
float x5144 = x39[x5143];
x39[x5143] = 0.0f;

}
for(int x5148=0; x5148 < 1300; x5148++) {
float x5149 = x99[x5148];
bool x5150 = x5149 > 5.0f;
if (x5150) {
x99[x5148] = 5.0f;
} else {
}
float x5154 = x99[x5148];
bool x5155 = x5154 < -5.0f;
if (x5155) {
x99[x5148] = -5.0f;
} else {
}

}
float* x5161 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x5162 = 0;
int32_t x5163 = 0;
int32_t x5164 = 0;
for(int x5165=0; x5165 < 26; x5165++) {
int32_t x5166 = x5163;
int32_t x5167 = x5164;
int32_t x5168 = x5162;
int32_t x5169 = x5168;
int32_t x5170 = x5166;
int32_t x5171 = x5167;
for(int x5172=0; x5172 < 50; x5172++) {
int32_t x5173 = x5169;
int32_t x5174 = x5170;
float x5175 = x99[x5174];
int32_t x5176 = x5171;
float x5177 = x99[x5176];
float x5178 = x5175 * x5177;
x5161[x5173] = x5178;
x5169 += 1;
x5170 += 1;
x5171 += 1;

}
x5162 += 50;
x5163 += 50;
x5164 += 50;

}
for(int x5190=0; x5190 < 1300; x5190++) {
float x5191 = x242[x5190];
float x5192 = x5161[x5190];
float x5193 = x5191 + x5192;
x242[x5190] = x5193;

}
float* x5197 = (float*)myMalloc(1300 * sizeof(float));;
for(int x5198=0; x5198 < 1300; x5198++) {
float x5199 = x99[x5198];
float x5200 = x5199 * 0.1f;
x5197[x5198] = x5200;

}
float* x5204 = (float*)myMalloc(1300 * sizeof(float));;
for(int x5205=0; x5205 < 1300; x5205++) {
float x5206 = x242[x5205];
float x5207 = x5206 + 1.0E-8f;
x5204[x5205] = x5207;

}
float* x5211 = (float*)myMalloc(1300 * sizeof(float));;
for(int x5212=0; x5212 < 1300; x5212++) {
float x5213 = x5204[x5212];
double x5214 = (double)x5213;
double x5215 = sqrt(x5214);
float x5216 = (float)x5215;
x5211[x5212] = x5216;

}
float* x5220 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x5221 = 0;
int32_t x5222 = 0;
int32_t x5223 = 0;
for(int x5224=0; x5224 < 26; x5224++) {
int32_t x5225 = x5222;
int32_t x5226 = x5223;
int32_t x5227 = x5221;
int32_t x5228 = x5227;
int32_t x5229 = x5225;
int32_t x5230 = x5226;
for(int x5231=0; x5231 < 50; x5231++) {
int32_t x5232 = x5228;
int32_t x5233 = x5229;
float x5234 = x5197[x5233];
int32_t x5235 = x5230;
float x5236 = x5211[x5235];
float x5237 = x5234 / x5236;
x5220[x5232] = x5237;
x5228 += 1;
x5229 += 1;
x5230 += 1;

}
x5221 += 50;
x5222 += 50;
x5223 += 50;

}
for(int x5249=0; x5249 < 1300; x5249++) {
float x5250 = x91[x5249];
float x5251 = x5220[x5249];
float x5252 = x5250 - x5251;
x91[x5249] = x5252;

}
for(int x5256=0; x5256 < 1300; x5256++) {
float x5257 = x99[x5256];
x99[x5256] = 0.0f;

}
for(int x5261=0; x5261 < 50; x5261++) {
float x5262 = x122[x5261];
bool x5263 = x5262 > 5.0f;
if (x5263) {
x122[x5261] = 5.0f;
} else {
}
float x5267 = x122[x5261];
bool x5268 = x5267 < -5.0f;
if (x5268) {
x122[x5261] = -5.0f;
} else {
}

}
float* x5274 = (float*)myMalloc(50 * sizeof(float));;
int32_t x5275 = 0;
int32_t x5276 = 0;
int32_t x5277 = 0;
for(int x5278=0; x5278 < 50; x5278++) {
int32_t x5279 = x5275;
int32_t x5280 = x5276;
float x5281 = x122[x5280];
int32_t x5282 = x5277;
float x5283 = x122[x5282];
float x5284 = x5281 * x5283;
x5274[x5279] = x5284;
x5275 += 1;
x5276 += 1;
x5277 += 1;

}
for(int x5291=0; x5291 < 50; x5291++) {
float x5292 = x247[x5291];
float x5293 = x5274[x5291];
float x5294 = x5292 + x5293;
x247[x5291] = x5294;

}
float* x5298 = (float*)myMalloc(50 * sizeof(float));;
for(int x5299=0; x5299 < 50; x5299++) {
float x5300 = x122[x5299];
float x5301 = x5300 * 0.1f;
x5298[x5299] = x5301;

}
float* x5305 = (float*)myMalloc(50 * sizeof(float));;
for(int x5306=0; x5306 < 50; x5306++) {
float x5307 = x247[x5306];
float x5308 = x5307 + 1.0E-8f;
x5305[x5306] = x5308;

}
float* x5312 = (float*)myMalloc(50 * sizeof(float));;
for(int x5313=0; x5313 < 50; x5313++) {
float x5314 = x5305[x5313];
double x5315 = (double)x5314;
double x5316 = sqrt(x5315);
float x5317 = (float)x5316;
x5312[x5313] = x5317;

}
float* x5321 = (float*)myMalloc(50 * sizeof(float));;
int32_t x5322 = 0;
int32_t x5323 = 0;
int32_t x5324 = 0;
for(int x5325=0; x5325 < 50; x5325++) {
int32_t x5326 = x5322;
int32_t x5327 = x5323;
float x5328 = x5298[x5327];
int32_t x5329 = x5324;
float x5330 = x5312[x5329];
float x5331 = x5328 / x5330;
x5321[x5326] = x5331;
x5322 += 1;
x5323 += 1;
x5324 += 1;

}
for(int x5338=0; x5338 < 50; x5338++) {
float x5339 = x117[x5338];
float x5340 = x5321[x5338];
float x5341 = x5339 - x5340;
x117[x5338] = x5341;

}
for(int x5345=0; x5345 < 50; x5345++) {
float x5346 = x122[x5345];
x122[x5345] = 0.0f;

}
for(int x5350=0; x5350 < 2500; x5350++) {
float x5351 = x112[x5350];
bool x5352 = x5351 > 5.0f;
if (x5352) {
x112[x5350] = 5.0f;
} else {
}
float x5356 = x112[x5350];
bool x5357 = x5356 < -5.0f;
if (x5357) {
x112[x5350] = -5.0f;
} else {
}

}
float* x5363 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x5364 = 0;
int32_t x5365 = 0;
int32_t x5366 = 0;
for(int x5367=0; x5367 < 50; x5367++) {
int32_t x5368 = x5365;
int32_t x5369 = x5366;
int32_t x5370 = x5364;
int32_t x5371 = x5370;
int32_t x5372 = x5368;
int32_t x5373 = x5369;
for(int x5374=0; x5374 < 50; x5374++) {
int32_t x5375 = x5371;
int32_t x5376 = x5372;
float x5377 = x112[x5376];
int32_t x5378 = x5373;
float x5379 = x112[x5378];
float x5380 = x5377 * x5379;
x5363[x5375] = x5380;
x5371 += 1;
x5372 += 1;
x5373 += 1;

}
x5364 += 50;
x5365 += 50;
x5366 += 50;

}
for(int x5392=0; x5392 < 2500; x5392++) {
float x5393 = x252[x5392];
float x5394 = x5363[x5392];
float x5395 = x5393 + x5394;
x252[x5392] = x5395;

}
float* x5399 = (float*)myMalloc(2500 * sizeof(float));;
for(int x5400=0; x5400 < 2500; x5400++) {
float x5401 = x112[x5400];
float x5402 = x5401 * 0.1f;
x5399[x5400] = x5402;

}
float* x5406 = (float*)myMalloc(2500 * sizeof(float));;
for(int x5407=0; x5407 < 2500; x5407++) {
float x5408 = x252[x5407];
float x5409 = x5408 + 1.0E-8f;
x5406[x5407] = x5409;

}
float* x5413 = (float*)myMalloc(2500 * sizeof(float));;
for(int x5414=0; x5414 < 2500; x5414++) {
float x5415 = x5406[x5414];
double x5416 = (double)x5415;
double x5417 = sqrt(x5416);
float x5418 = (float)x5417;
x5413[x5414] = x5418;

}
float* x5422 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x5423 = 0;
int32_t x5424 = 0;
int32_t x5425 = 0;
for(int x5426=0; x5426 < 50; x5426++) {
int32_t x5427 = x5424;
int32_t x5428 = x5425;
int32_t x5429 = x5423;
int32_t x5430 = x5429;
int32_t x5431 = x5427;
int32_t x5432 = x5428;
for(int x5433=0; x5433 < 50; x5433++) {
int32_t x5434 = x5430;
int32_t x5435 = x5431;
float x5436 = x5399[x5435];
int32_t x5437 = x5432;
float x5438 = x5413[x5437];
float x5439 = x5436 / x5438;
x5422[x5434] = x5439;
x5430 += 1;
x5431 += 1;
x5432 += 1;

}
x5423 += 50;
x5424 += 50;
x5425 += 50;

}
for(int x5451=0; x5451 < 2500; x5451++) {
float x5452 = x104[x5451];
float x5453 = x5422[x5451];
float x5454 = x5452 - x5453;
x104[x5451] = x5454;

}
for(int x5458=0; x5458 < 2500; x5458++) {
float x5459 = x112[x5458];
x112[x5458] = 0.0f;

}
mallocAddr = (void*)x259;

}
double x5466 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x5469 = (long)fopen(x0, "w");
fprintf((FILE *)x5469, "unit: %s\n", "100 iteration");
for(int x5472=0; x5472 < 51; x5472++) {
double x5473 = x258[x5472];
fprintf((FILE *)x5469, "%lf\n", x5473);

}
double x5467 = x257 - x2;
double x5468 = x5466 - x257;
fprintf((FILE *)x5469, "run time: %lf %lf\n", x5467, x5468);
fclose((FILE*)x5469);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

