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
function<void(int32_t,float**)> x1284 = [&](int32_t x1285,float** x1286) {
float** x1288 = x1286;
float* x1289 = x1288[0];
float* x1290 = x1288[1];
float* x1291 = x1288[2];
float* x1292 = x1288[3];
float* x1293 = x1288[4];
float* x1294 = x1288[5];
int32_t x1287 = x1285;
bool x1295 = x1287 < 20;
if (x1295) {
int32_t x1296 = x1287 * 520;
float* x1297 = x302+x1296;
float* x1298 = x324+x1296;
float* x1299 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1300=0; x1300 < 20; x1300++) {
int32_t x1304 = x1300 * 26;
int32_t x1314 = x1300 * 50;
for(int x1301=0; x1301 < 50; x1301++) {
float x1302 = 0.0f;
int32_t x1307 = x1301 * 26;
for(int x1303=0; x1303 < 26; x1303++) {
int32_t x1305 = x1304 + x1303;
float x1306 = x1297[x1305];
int32_t x1308 = x1307 + x1303;
float x1309 = x16[x1308];
float x1310 = x1306 * x1309;
x1302 += x1310;

}
float x1316 = x1302;
int32_t x1315 = x1314 + x1301;
x1299[x1315] = x1316;

}

}
float* x1322 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1323=0; x1323 < 1000; x1323++) {
x1322[x1323] = 0.0f;

}
float* x1327 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1328=0; x1328 < 20; x1328++) {
int32_t x1332 = x1328 * 50;
for(int x1329=0; x1329 < 50; x1329++) {
float x1330 = 0.0f;
int32_t x1335 = x1329 * 50;
for(int x1331=0; x1331 < 50; x1331++) {
int32_t x1333 = x1332 + x1331;
float x1334 = x1291[x1333];
int32_t x1336 = x1335 + x1331;
float x1337 = x30[x1336];
float x1338 = x1334 * x1337;
x1330 += x1338;

}
float x1343 = x1330;
int32_t x1342 = x1332 + x1329;
x1327[x1342] = x1343;

}

}
float* x1349 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1350=0; x1350 < 1000; x1350++) {
x1349[x1350] = 0.0f;

}
float* x1354 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1355 = 0;
int32_t x1356 = 0;
int32_t x1357 = 0;
for(int x1358=0; x1358 < 20; x1358++) {
int32_t x1359 = x1356;
int32_t x1360 = x1357;
int32_t x1361 = x1355;
int32_t x1362 = x1361;
int32_t x1363 = x1359;
int32_t x1364 = x1360;
for(int x1365=0; x1365 < 50; x1365++) {
int32_t x1366 = x1362;
int32_t x1367 = x1363;
float x1368 = x1299[x1367];
int32_t x1369 = x1364;
float x1370 = x1327[x1369];
float x1371 = x1368 + x1370;
x1354[x1366] = x1371;
x1362 += 1;
x1363 += 1;
x1364 += 1;

}
x1355 += 50;
x1356 += 50;
x1357 += 50;

}
float* x1383 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1384=0; x1384 < 1000; x1384++) {
x1383[x1384] = 0.0f;

}
float* x1388 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1389 = 0;
int32_t x1390 = 0;
int32_t x1391 = 0;
for(int x1392=0; x1392 < 20; x1392++) {
int32_t x1393 = x1390;
int32_t x1394 = x1391;
int32_t x1395 = x1389;
int32_t x1396 = x1395;
int32_t x1397 = x1393;
int32_t x1398 = x1394;
for(int x1399=0; x1399 < 50; x1399++) {
int32_t x1400 = x1396;
int32_t x1401 = x1397;
float x1402 = x1354[x1401];
int32_t x1403 = x1398;
float x1404 = x44[x1403];
float x1405 = x1402 + x1404;
x1388[x1400] = x1405;
x1396 += 1;
x1397 += 1;
x1398 += 1;

}
x1389 += 50;
x1390 += 50;

}
float* x1416 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1417=0; x1417 < 1000; x1417++) {
x1416[x1417] = 0.0f;

}
float* x1421 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1422=0; x1422 < 1000; x1422++) {
float x1423 = x1388[x1422];
float x1424 = -1.0f * x1423;
double x1425 = (double)x1424;
double x1426 = exp(x1425);
float x1427 = (float)x1426;
float x1428 = x1427 + 1.0f;
float x1429 = 1.0f / x1428;
x1421[x1422] = x1429;

}
float* x1433 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1434=0; x1434 < 1000; x1434++) {
x1433[x1434] = 0.0f;

}
float* x1438 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1439=0; x1439 < 20; x1439++) {
int32_t x1443 = x1439 * 26;
int32_t x1453 = x1439 * 50;
for(int x1440=0; x1440 < 50; x1440++) {
float x1441 = 0.0f;
int32_t x1446 = x1440 * 26;
for(int x1442=0; x1442 < 26; x1442++) {
int32_t x1444 = x1443 + x1442;
float x1445 = x1297[x1444];
int32_t x1447 = x1446 + x1442;
float x1448 = x55[x1447];
float x1449 = x1445 * x1448;
x1441 += x1449;

}
float x1455 = x1441;
int32_t x1454 = x1453 + x1440;
x1438[x1454] = x1455;

}

}
float* x1461 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1462=0; x1462 < 1000; x1462++) {
x1461[x1462] = 0.0f;

}
float* x1466 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1467=0; x1467 < 20; x1467++) {
int32_t x1471 = x1467 * 50;
for(int x1468=0; x1468 < 50; x1468++) {
float x1469 = 0.0f;
int32_t x1474 = x1468 * 50;
for(int x1470=0; x1470 < 50; x1470++) {
int32_t x1472 = x1471 + x1470;
float x1473 = x1291[x1472];
int32_t x1475 = x1474 + x1470;
float x1476 = x68[x1475];
float x1477 = x1473 * x1476;
x1469 += x1477;

}
float x1482 = x1469;
int32_t x1481 = x1471 + x1468;
x1466[x1481] = x1482;

}

}
float* x1488 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1489=0; x1489 < 1000; x1489++) {
x1488[x1489] = 0.0f;

}
float* x1493 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1494 = 0;
int32_t x1495 = 0;
int32_t x1496 = 0;
for(int x1497=0; x1497 < 20; x1497++) {
int32_t x1498 = x1495;
int32_t x1499 = x1496;
int32_t x1500 = x1494;
int32_t x1501 = x1500;
int32_t x1502 = x1498;
int32_t x1503 = x1499;
for(int x1504=0; x1504 < 50; x1504++) {
int32_t x1505 = x1501;
int32_t x1506 = x1502;
float x1507 = x1438[x1506];
int32_t x1508 = x1503;
float x1509 = x1466[x1508];
float x1510 = x1507 + x1509;
x1493[x1505] = x1510;
x1501 += 1;
x1502 += 1;
x1503 += 1;

}
x1494 += 50;
x1495 += 50;
x1496 += 50;

}
float* x1522 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1523=0; x1523 < 1000; x1523++) {
x1522[x1523] = 0.0f;

}
float* x1527 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1528 = 0;
int32_t x1529 = 0;
int32_t x1530 = 0;
for(int x1531=0; x1531 < 20; x1531++) {
int32_t x1532 = x1529;
int32_t x1533 = x1530;
int32_t x1534 = x1528;
int32_t x1535 = x1534;
int32_t x1536 = x1532;
int32_t x1537 = x1533;
for(int x1538=0; x1538 < 50; x1538++) {
int32_t x1539 = x1535;
int32_t x1540 = x1536;
float x1541 = x1493[x1540];
int32_t x1542 = x1537;
float x1543 = x81[x1542];
float x1544 = x1541 + x1543;
x1527[x1539] = x1544;
x1535 += 1;
x1536 += 1;
x1537 += 1;

}
x1528 += 50;
x1529 += 50;

}
float* x1555 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1556=0; x1556 < 1000; x1556++) {
x1555[x1556] = 0.0f;

}
float* x1560 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1561=0; x1561 < 1000; x1561++) {
float x1562 = x1527[x1561];
float x1563 = -1.0f * x1562;
double x1564 = (double)x1563;
double x1565 = exp(x1564);
float x1566 = (float)x1565;
float x1567 = x1566 + 1.0f;
float x1568 = 1.0f / x1567;
x1560[x1561] = x1568;

}
float* x1572 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1573=0; x1573 < 1000; x1573++) {
x1572[x1573] = 0.0f;

}
float* x1577 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1578=0; x1578 < 20; x1578++) {
int32_t x1582 = x1578 * 26;
int32_t x1592 = x1578 * 50;
for(int x1579=0; x1579 < 50; x1579++) {
float x1580 = 0.0f;
int32_t x1585 = x1579 * 26;
for(int x1581=0; x1581 < 26; x1581++) {
int32_t x1583 = x1582 + x1581;
float x1584 = x1297[x1583];
int32_t x1586 = x1585 + x1581;
float x1587 = x127[x1586];
float x1588 = x1584 * x1587;
x1580 += x1588;

}
float x1594 = x1580;
int32_t x1593 = x1592 + x1579;
x1577[x1593] = x1594;

}

}
float* x1600 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1601=0; x1601 < 1000; x1601++) {
x1600[x1601] = 0.0f;

}
float* x1605 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1606=0; x1606 < 20; x1606++) {
int32_t x1610 = x1606 * 50;
for(int x1607=0; x1607 < 50; x1607++) {
float x1608 = 0.0f;
int32_t x1613 = x1607 * 50;
for(int x1609=0; x1609 < 50; x1609++) {
int32_t x1611 = x1610 + x1609;
float x1612 = x1291[x1611];
int32_t x1614 = x1613 + x1609;
float x1615 = x140[x1614];
float x1616 = x1612 * x1615;
x1608 += x1616;

}
float x1621 = x1608;
int32_t x1620 = x1610 + x1607;
x1605[x1620] = x1621;

}

}
float* x1627 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1628=0; x1628 < 1000; x1628++) {
x1627[x1628] = 0.0f;

}
float* x1632 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1633 = 0;
int32_t x1634 = 0;
int32_t x1635 = 0;
for(int x1636=0; x1636 < 20; x1636++) {
int32_t x1637 = x1634;
int32_t x1638 = x1635;
int32_t x1639 = x1633;
int32_t x1640 = x1639;
int32_t x1641 = x1637;
int32_t x1642 = x1638;
for(int x1643=0; x1643 < 50; x1643++) {
int32_t x1644 = x1640;
int32_t x1645 = x1641;
float x1646 = x1577[x1645];
int32_t x1647 = x1642;
float x1648 = x1605[x1647];
float x1649 = x1646 + x1648;
x1632[x1644] = x1649;
x1640 += 1;
x1641 += 1;
x1642 += 1;

}
x1633 += 50;
x1634 += 50;
x1635 += 50;

}
float* x1661 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1662=0; x1662 < 1000; x1662++) {
x1661[x1662] = 0.0f;

}
float* x1666 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1667 = 0;
int32_t x1668 = 0;
int32_t x1669 = 0;
for(int x1670=0; x1670 < 20; x1670++) {
int32_t x1671 = x1668;
int32_t x1672 = x1669;
int32_t x1673 = x1667;
int32_t x1674 = x1673;
int32_t x1675 = x1671;
int32_t x1676 = x1672;
for(int x1677=0; x1677 < 50; x1677++) {
int32_t x1678 = x1674;
int32_t x1679 = x1675;
float x1680 = x1632[x1679];
int32_t x1681 = x1676;
float x1682 = x153[x1681];
float x1683 = x1680 + x1682;
x1666[x1678] = x1683;
x1674 += 1;
x1675 += 1;
x1676 += 1;

}
x1667 += 50;
x1668 += 50;

}
float* x1694 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1695=0; x1695 < 1000; x1695++) {
x1694[x1695] = 0.0f;

}
float* x1699 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1700=0; x1700 < 1000; x1700++) {
float x1701 = x1666[x1700];
float x1702 = -1.0f * x1701;
double x1703 = (double)x1702;
double x1704 = exp(x1703);
float x1705 = (float)x1704;
float x1706 = x1705 + 1.0f;
float x1707 = 1.0f / x1706;
x1699[x1700] = x1707;

}
float* x1711 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1712=0; x1712 < 1000; x1712++) {
x1711[x1712] = 0.0f;

}
float* x1716 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1717=0; x1717 < 20; x1717++) {
int32_t x1721 = x1717 * 26;
int32_t x1731 = x1717 * 50;
for(int x1718=0; x1718 < 50; x1718++) {
float x1719 = 0.0f;
int32_t x1724 = x1718 * 26;
for(int x1720=0; x1720 < 26; x1720++) {
int32_t x1722 = x1721 + x1720;
float x1723 = x1297[x1722];
int32_t x1725 = x1724 + x1720;
float x1726 = x91[x1725];
float x1727 = x1723 * x1726;
x1719 += x1727;

}
float x1733 = x1719;
int32_t x1732 = x1731 + x1718;
x1716[x1732] = x1733;

}

}
float* x1739 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1740=0; x1740 < 1000; x1740++) {
x1739[x1740] = 0.0f;

}
float* x1744 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1745=0; x1745 < 20; x1745++) {
int32_t x1749 = x1745 * 50;
for(int x1746=0; x1746 < 50; x1746++) {
float x1747 = 0.0f;
int32_t x1752 = x1746 * 50;
for(int x1748=0; x1748 < 50; x1748++) {
int32_t x1750 = x1749 + x1748;
float x1751 = x1291[x1750];
int32_t x1753 = x1752 + x1748;
float x1754 = x104[x1753];
float x1755 = x1751 * x1754;
x1747 += x1755;

}
float x1760 = x1747;
int32_t x1759 = x1749 + x1746;
x1744[x1759] = x1760;

}

}
float* x1766 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1767=0; x1767 < 1000; x1767++) {
x1766[x1767] = 0.0f;

}
float* x1771 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1772 = 0;
int32_t x1773 = 0;
int32_t x1774 = 0;
for(int x1775=0; x1775 < 20; x1775++) {
int32_t x1776 = x1773;
int32_t x1777 = x1774;
int32_t x1778 = x1772;
int32_t x1779 = x1778;
int32_t x1780 = x1776;
int32_t x1781 = x1777;
for(int x1782=0; x1782 < 50; x1782++) {
int32_t x1783 = x1779;
int32_t x1784 = x1780;
float x1785 = x1716[x1784];
int32_t x1786 = x1781;
float x1787 = x1744[x1786];
float x1788 = x1785 + x1787;
x1771[x1783] = x1788;
x1779 += 1;
x1780 += 1;
x1781 += 1;

}
x1772 += 50;
x1773 += 50;
x1774 += 50;

}
float* x1800 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1801=0; x1801 < 1000; x1801++) {
x1800[x1801] = 0.0f;

}
float* x1805 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1806 = 0;
int32_t x1807 = 0;
int32_t x1808 = 0;
for(int x1809=0; x1809 < 20; x1809++) {
int32_t x1810 = x1807;
int32_t x1811 = x1808;
int32_t x1812 = x1806;
int32_t x1813 = x1812;
int32_t x1814 = x1810;
int32_t x1815 = x1811;
for(int x1816=0; x1816 < 50; x1816++) {
int32_t x1817 = x1813;
int32_t x1818 = x1814;
float x1819 = x1771[x1818];
int32_t x1820 = x1815;
float x1821 = x117[x1820];
float x1822 = x1819 + x1821;
x1805[x1817] = x1822;
x1813 += 1;
x1814 += 1;
x1815 += 1;

}
x1806 += 50;
x1807 += 50;

}
float* x1833 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1834=0; x1834 < 1000; x1834++) {
x1833[x1834] = 0.0f;

}
float* x1838 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1839=0; x1839 < 1000; x1839++) {
float x1840 = x1805[x1839];
double x1841 = (double)x1840;
double x1842 = tanh(x1841);
float x1843 = (float)x1842;
x1838[x1839] = x1843;

}
float* x1847 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1848=0; x1848 < 1000; x1848++) {
x1847[x1848] = 0.0f;

}
float* x1852 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1853 = 0;
int32_t x1854 = 0;
int32_t x1855 = 0;
for(int x1856=0; x1856 < 20; x1856++) {
int32_t x1857 = x1854;
int32_t x1858 = x1855;
int32_t x1859 = x1853;
int32_t x1860 = x1859;
int32_t x1861 = x1857;
int32_t x1862 = x1858;
for(int x1863=0; x1863 < 50; x1863++) {
int32_t x1864 = x1860;
int32_t x1865 = x1861;
float x1866 = x1421[x1865];
int32_t x1867 = x1862;
float x1868 = x1293[x1867];
float x1869 = x1866 * x1868;
x1852[x1864] = x1869;
x1860 += 1;
x1861 += 1;
x1862 += 1;

}
x1853 += 50;
x1854 += 50;
x1855 += 50;

}
float* x1881 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1882=0; x1882 < 1000; x1882++) {
x1881[x1882] = 0.0f;

}
float* x1886 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1887 = 0;
int32_t x1888 = 0;
int32_t x1889 = 0;
for(int x1890=0; x1890 < 20; x1890++) {
int32_t x1891 = x1888;
int32_t x1892 = x1889;
int32_t x1893 = x1887;
int32_t x1894 = x1893;
int32_t x1895 = x1891;
int32_t x1896 = x1892;
for(int x1897=0; x1897 < 50; x1897++) {
int32_t x1898 = x1894;
int32_t x1899 = x1895;
float x1900 = x1560[x1899];
int32_t x1901 = x1896;
float x1902 = x1838[x1901];
float x1903 = x1900 * x1902;
x1886[x1898] = x1903;
x1894 += 1;
x1895 += 1;
x1896 += 1;

}
x1887 += 50;
x1888 += 50;
x1889 += 50;

}
float* x1915 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1916=0; x1916 < 1000; x1916++) {
x1915[x1916] = 0.0f;

}
float* x1920 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1921 = 0;
int32_t x1922 = 0;
int32_t x1923 = 0;
for(int x1924=0; x1924 < 20; x1924++) {
int32_t x1925 = x1922;
int32_t x1926 = x1923;
int32_t x1927 = x1921;
int32_t x1928 = x1927;
int32_t x1929 = x1925;
int32_t x1930 = x1926;
for(int x1931=0; x1931 < 50; x1931++) {
int32_t x1932 = x1928;
int32_t x1933 = x1929;
float x1934 = x1852[x1933];
int32_t x1935 = x1930;
float x1936 = x1886[x1935];
float x1937 = x1934 + x1936;
x1920[x1932] = x1937;
x1928 += 1;
x1929 += 1;
x1930 += 1;

}
x1921 += 50;
x1922 += 50;
x1923 += 50;

}
float* x1949 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1950=0; x1950 < 1000; x1950++) {
x1949[x1950] = 0.0f;

}
float* x1954 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1955=0; x1955 < 1000; x1955++) {
float x1956 = x1920[x1955];
double x1957 = (double)x1956;
double x1958 = tanh(x1957);
float x1959 = (float)x1958;
x1954[x1955] = x1959;

}
float* x1963 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1964=0; x1964 < 1000; x1964++) {
x1963[x1964] = 0.0f;

}
float* x1968 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1969 = 0;
int32_t x1970 = 0;
int32_t x1971 = 0;
for(int x1972=0; x1972 < 20; x1972++) {
int32_t x1973 = x1970;
int32_t x1974 = x1971;
int32_t x1975 = x1969;
int32_t x1976 = x1975;
int32_t x1977 = x1973;
int32_t x1978 = x1974;
for(int x1979=0; x1979 < 50; x1979++) {
int32_t x1980 = x1976;
int32_t x1981 = x1977;
float x1982 = x1699[x1981];
int32_t x1983 = x1978;
float x1984 = x1954[x1983];
float x1985 = x1982 * x1984;
x1968[x1980] = x1985;
x1976 += 1;
x1977 += 1;
x1978 += 1;

}
x1969 += 50;
x1970 += 50;
x1971 += 50;

}
float* x1997 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1998=0; x1998 < 1000; x1998++) {
x1997[x1998] = 0.0f;

}
float* x2002 = (float*)myMalloc(520 * sizeof(float));;
for(int x2003=0; x2003 < 20; x2003++) {
int32_t x2007 = x2003 * 50;
int32_t x2017 = x2003 * 26;
for(int x2004=0; x2004 < 26; x2004++) {
float x2005 = 0.0f;
int32_t x2010 = x2004 * 50;
for(int x2006=0; x2006 < 50; x2006++) {
int32_t x2008 = x2007 + x2006;
float x2009 = x1968[x2008];
int32_t x2011 = x2010 + x2006;
float x2012 = x163[x2011];
float x2013 = x2009 * x2012;
x2005 += x2013;

}
float x2019 = x2005;
int32_t x2018 = x2017 + x2004;
x2002[x2018] = x2019;

}

}
float* x2025 = (float*)myMalloc(520 * sizeof(float));;
for(int x2026=0; x2026 < 520; x2026++) {
x2025[x2026] = 0.0f;

}
float* x2030 = (float*)myMalloc(520 * sizeof(float));;
int32_t x2031 = 0;
int32_t x2032 = 0;
int32_t x2033 = 0;
for(int x2034=0; x2034 < 20; x2034++) {
int32_t x2035 = x2032;
int32_t x2036 = x2033;
int32_t x2037 = x2031;
int32_t x2038 = x2037;
int32_t x2039 = x2035;
int32_t x2040 = x2036;
for(int x2041=0; x2041 < 26; x2041++) {
int32_t x2042 = x2038;
int32_t x2043 = x2039;
float x2044 = x2002[x2043];
int32_t x2045 = x2040;
float x2046 = x176[x2045];
float x2047 = x2044 + x2046;
x2030[x2042] = x2047;
x2038 += 1;
x2039 += 1;
x2040 += 1;

}
x2031 += 26;
x2032 += 26;

}
float* x2058 = (float*)myMalloc(520 * sizeof(float));;
for(int x2059=0; x2059 < 520; x2059++) {
x2058[x2059] = 0.0f;

}
int* x2063 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x2064=0; x2064 < 20; x2064++) {
int32_t x2065 = x2064 * 20;
int32_t x2066 = x1287 + x2065;
int32_t x2067 = x274[x2066];
x2063[x2064] = x2067;

}
float* x2071 = (float*)myMalloc(20 * sizeof(float));;
int32_t x2072 = 0;
for(int x2073=0; x2073 < 20; x2073++) {
float x2074 = -3.4028235E38f;
for(int x2075=0; x2075 < 26; x2075++) {
int32_t x2076 = x2072;
float x2077 = x2030[x2076];
float x2078 = x2074;
bool x2079 = x2077 > x2078;
if (x2079) {
float x2080 = x2030[x2076];
x2074 = x2080;
} else {
}
x2072 += 1;

}
float x2087 = x2074;
x2071[x2073] = x2087;

}
float* x2091 = (float*)myMalloc(520 * sizeof(float));;
for(int x2092=0; x2092 < 520; x2092++) {
x2091[x2092] = 0.0f;

}
int32_t x2096 = 0;
for(int x2097=0; x2097 < 20; x2097++) {
for(int x2098=0; x2098 < 26; x2098++) {
int32_t x2099 = x2096;
float x2100 = x2030[x2099];
float x2101 = x2071[x2097];
float x2102 = x2100 - x2101;
double x2103 = (double)x2102;
double x2104 = exp(x2103);
float x2105 = (float)x2104;
x2091[x2099] = x2105;
x2096 += 1;

}

}
float* x2112 = (float*)myMalloc(20 * sizeof(float));;
for(int x2113=0; x2113 < 20; x2113++) {
x2112[x2113] = 0.0f;

}
for(int x2117=0; x2117 < 20; x2117++) {
int32_t x2118 = x2117;
int32_t x2119 = x2117 * 26;
int32_t x2120 = x2119;
for(int x2121=0; x2121 < 26; x2121++) {
int32_t x2122 = x2118;
float x2123 = x2112[x2122];
int32_t x2124 = x2120;
float x2125 = x2091[x2124];
float x2126 = x2123 + x2125;
x2112[x2122] = x2126;
x2120 += 1;

}

}
x2096 = 0;
for(int x2134=0; x2134 < 20; x2134++) {
float x2135 = x2071[x2134];
float x2136 = x2112[x2134];
double x2137 = (double)x2136;
double x2138 = log(x2137);
float x2139 = (float)x2138;
float x2140 = x2135 + x2139;
for(int x2141=0; x2141 < 26; x2141++) {
int32_t x2142 = x2096;
float x2143 = x2030[x2142];
float x2144 = x2143 - x2140;
x2091[x2142] = x2144;
x2096 += 1;

}

}
float* x2151 = (float*)myMalloc(520 * sizeof(float));;
for(int x2152=0; x2152 < 520; x2152++) {
x2151[x2152] = 0.0f;

}
float* x2156 = (float*)myMalloc(20 * sizeof(float));;
int32_t x2157 = 0;
for(int x2158=0; x2158 < 20; x2158++) {
int32_t x2159 = x2157;
int32_t x2160 = x2063[x2158];
int32_t x2161 = x2159 + x2160;
float x2162 = x2091[x2161];
float x2163 = -1.0f * x2162;
x2156[x2158] = x2163;
x2157 += 26;

}
float* x2168 = (float*)myMalloc(20 * sizeof(float));;
for(int x2169=0; x2169 < 20; x2169++) {
x2168[x2169] = 0.0f;

}
float x2173 = 0.0f;
for(int x2174=0; x2174 < 20; x2174++) {
float x2175 = x2173;
float x2176 = x2156[x2174];
float x2177 = x2175 + x2176;
x2173 = x2177;

}
float x2181 = x2173;
float* x2182 = (float*)myMalloc(1 * sizeof(float));;
x2182[0] = x2181;
float* x2184 = (float*)myMalloc(1 * sizeof(float));;
for(int x2185=0; x2185 < 1; x2185++) {
x2184[x2185] = 0.0f;

}
float* x2189 = (float*)myMalloc(1 * sizeof(float));;
int32_t x2190 = 0;
int32_t x2191 = 0;
int32_t x2192 = 0;
int32_t x2193 = x2190;
int32_t x2194 = x2191;
float x2195 = x1289[x2194];
int32_t x2196 = x2192;
float x2197 = x2182[x2196];
float x2198 = x2195 + x2197;
x2189[x2193] = x2198;
x2190 += 1;
float* x2201 = (float*)myMalloc(1 * sizeof(float));;
for(int x2202=0; x2202 < 1; x2202++) {
x2201[x2202] = 0.0f;

}
float** x2207 = (float**)myMalloc(6 * sizeof(float*));;
x2207[0] = x2189;
x2207[1] = x2201;
x2207[2] = x1968;
x2207[3] = x1997;
x2207[4] = x1920;
x2207[5] = x1949;
int32_t x2216 = 0;
int32_t x2217 = 0;
int32_t x2218 = 0;
int32_t x2219 = x2216;
int32_t x2222 = x2217;
int32_t x2224 = x2218;
x2218 += 1;
int32_t x2243 = 0;
float* x2256 = (float*)myMalloc(20 * sizeof(float));;
int32_t x2277 = 0;
int32_t x2297 = 0;
int32_t x2298 = 0;
int32_t x2299 = 0;
int32_t x2358 = 0;
int32_t x2359 = 0;
int32_t x2360 = 0;
int32_t x2407 = 0;
int32_t x2408 = 0;
int32_t x2409 = 0;
int32_t x2443 = 0;
int32_t x2444 = 0;
int32_t x2445 = 0;
int32_t x2481 = 0;
int32_t x2482 = 0;
int32_t x2483 = 0;
int32_t x2530 = 0;
int32_t x2531 = 0;
int32_t x2532 = 0;
int32_t x2565 = 0;
int32_t x2566 = 0;
int32_t x2567 = 0;
int32_t x2663 = 0;
int32_t x2664 = 0;
int32_t x2665 = 0;
int32_t x2698 = 0;
int32_t x2699 = 0;
int32_t x2700 = 0;
int32_t x2796 = 0;
int32_t x2797 = 0;
int32_t x2798 = 0;
int32_t x2831 = 0;
int32_t x2832 = 0;
int32_t x2833 = 0;
int32_t x2929 = 0;
int32_t x2930 = 0;
int32_t x2931 = 0;
int32_t x2964 = 0;
int32_t x2965 = 0;
int32_t x2966 = 0;
int32_t x2206 = x1287 + 1;
x1284(x2206,x2207);
float x2220 = x1290[x2219];
float x2221 = x1289[x2219];
float x2223 = x2182[x2222];
float x2225 = x2201[x2224];
float x2226 = x2220 + x2225;
x1290[x2219] = x2226;
float x2228 = x2184[x2222];
float x2229 = x1289[x2219];
float x2230 = x2182[x2222];
float x2231 = x2201[x2224];
float x2232 = x2228 + x2231;
x2184[x2222] = x2232;
// += tensor of dim 0
float x2236 = x2184[0];
for(int x2237=0; x2237 < 20; x2237++) {
float x2238 = x2168[x2237];
float x2239 = x2238 + x2236;
x2168[x2237] = x2239;

}
for(int x2244=0; x2244 < 20; x2244++) {
int32_t x2245 = x2243;
int32_t x2246 = x2063[x2244];
int32_t x2247 = x2245 + x2246;
float x2248 = x2151[x2247];
float x2249 = x2168[x2244];
float x2250 = -1.0f * x2249;
float x2251 = x2248 + x2250;
x2151[x2247] = x2251;
x2243 += 26;

}
for(int x2257=0; x2257 < 20; x2257++) {
x2256[x2257] = 0.0f;

}
for(int x2261=0; x2261 < 20; x2261++) {
int32_t x2262 = x2261;
int32_t x2263 = x2261 * 26;
int32_t x2264 = x2263;
for(int x2265=0; x2265 < 26; x2265++) {
int32_t x2266 = x2262;
float x2267 = x2256[x2266];
int32_t x2268 = x2264;
float x2269 = x2151[x2268];
float x2270 = x2267 + x2269;
x2256[x2266] = x2270;
x2264 += 1;

}

}
for(int x2278=0; x2278 < 20; x2278++) {
for(int x2279=0; x2279 < 26; x2279++) {
int32_t x2280 = x2277;
float x2281 = x2058[x2280];
float x2282 = x2151[x2280];
float x2283 = x2091[x2280];
float x2287 = x2256[x2278];
double x2284 = (double)x2283;
double x2285 = exp(x2284);
float x2286 = (float)x2285;
float x2288 = x2286 * x2287;
float x2289 = x2282 - x2288;
float x2290 = x2281 + x2289;
x2058[x2280] = x2290;
x2277 += 1;

}

}
for(int x2300=0; x2300 < 20; x2300++) {
int32_t x2301 = x2297;
int32_t x2302 = x2298;
int32_t x2303 = x2299;
int32_t x2304 = x2301;
int32_t x2305 = x2302;
int32_t x2306 = x2303;
for(int x2307=0; x2307 < 26; x2307++) {
int32_t x2308 = x2304;
float x2309 = x2025[x2308];
float x2310 = x2002[x2308];
int32_t x2311 = x2305;
float x2312 = x176[x2311];
int32_t x2313 = x2306;
float x2314 = x2058[x2313];
float x2315 = x2309 + x2314;
x2025[x2308] = x2315;
float x2317 = x182[x2311];
float x2318 = x2002[x2308];
float x2319 = x176[x2311];
float x2320 = x2058[x2313];
float x2321 = x2317 + x2320;
x182[x2311] = x2321;
x2306 += 1;
x2304 += 1;
x2305 += 1;

}
x2299 += 26;
x2297 += 26;

}
for(int x2332=0; x2332 < 20; x2332++) {
int32_t x2334 = x2332 * 26;
int32_t x2338 = x2332 * 50;
for(int x2333=0; x2333 < 26; x2333++) {
int32_t x2335 = x2334 + x2333;
float x2336 = x2025[x2335];
int32_t x2341 = x2333 * 50;
for(int x2337=0; x2337 < 50; x2337++) {
int32_t x2339 = x2338 + x2337;
float x2340 = x1997[x2339];
int32_t x2342 = x2341 + x2337;
float x2343 = x163[x2342];
float x2344 = x2343 * x2336;
float x2345 = x2340 + x2344;
x1997[x2339] = x2345;
float x2347 = x171[x2342];
float x2348 = x1968[x2339];
float x2349 = x2348 * x2336;
float x2350 = x2347 + x2349;
x171[x2342] = x2350;

}

}

}
for(int x2361=0; x2361 < 20; x2361++) {
int32_t x2362 = x2358;
int32_t x2363 = x2359;
int32_t x2364 = x2360;
int32_t x2365 = x2362;
int32_t x2366 = x2363;
int32_t x2367 = x2364;
for(int x2368=0; x2368 < 50; x2368++) {
int32_t x2369 = x2365;
float x2370 = x1711[x2369];
float x2371 = x1699[x2369];
int32_t x2372 = x2366;
float x2373 = x1954[x2372];
int32_t x2374 = x2367;
float x2375 = x1997[x2374];
float x2376 = x2375 * x2373;
float x2377 = x2370 + x2376;
x1711[x2369] = x2377;
float x2379 = x1963[x2372];
float x2380 = x1699[x2369];
float x2381 = x1954[x2372];
float x2382 = x1997[x2374];
float x2383 = x2382 * x2380;
float x2384 = x2379 + x2383;
x1963[x2372] = x2384;
x2367 += 1;
x2365 += 1;
x2366 += 1;

}
x2360 += 50;
x2358 += 50;
x2359 += 50;

}
for(int x2396=0; x2396 < 1000; x2396++) {
float x2397 = x1949[x2396];
float x2398 = x1954[x2396];
float x2401 = x1963[x2396];
float x2399 = x2398 * x2398;
float x2400 = 1.0f - x2399;
float x2402 = x2400 * x2401;
float x2403 = x2397 + x2402;
x1949[x2396] = x2403;

}
for(int x2410=0; x2410 < 20; x2410++) {
int32_t x2411 = x2407;
int32_t x2412 = x2408;
int32_t x2413 = x2409;
int32_t x2414 = x2411;
int32_t x2415 = x2412;
int32_t x2416 = x2413;
for(int x2417=0; x2417 < 50; x2417++) {
int32_t x2418 = x2414;
float x2419 = x1881[x2418];
float x2420 = x1852[x2418];
int32_t x2421 = x2415;
float x2422 = x1886[x2421];
int32_t x2423 = x2416;
float x2424 = x1949[x2423];
float x2425 = x2419 + x2424;
x1881[x2418] = x2425;
float x2427 = x1915[x2421];
float x2428 = x1852[x2418];
float x2429 = x1886[x2421];
float x2430 = x1949[x2423];
float x2431 = x2427 + x2430;
x1915[x2421] = x2431;
x2416 += 1;
x2414 += 1;
x2415 += 1;

}
x2409 += 50;
x2407 += 50;
x2408 += 50;

}
for(int x2446=0; x2446 < 20; x2446++) {
int32_t x2447 = x2443;
int32_t x2448 = x2444;
int32_t x2449 = x2445;
int32_t x2450 = x2447;
int32_t x2451 = x2448;
int32_t x2452 = x2449;
for(int x2453=0; x2453 < 50; x2453++) {
int32_t x2454 = x2450;
float x2455 = x1572[x2454];
float x2456 = x1560[x2454];
int32_t x2457 = x2451;
float x2458 = x1838[x2457];
int32_t x2459 = x2452;
float x2460 = x1915[x2459];
float x2461 = x2460 * x2458;
float x2462 = x2455 + x2461;
x1572[x2454] = x2462;
float x2464 = x1847[x2457];
float x2465 = x1560[x2454];
float x2466 = x1838[x2457];
float x2467 = x1915[x2459];
float x2468 = x2467 * x2465;
float x2469 = x2464 + x2468;
x1847[x2457] = x2469;
x2452 += 1;
x2450 += 1;
x2451 += 1;

}
x2445 += 50;
x2443 += 50;
x2444 += 50;

}
for(int x2484=0; x2484 < 20; x2484++) {
int32_t x2485 = x2481;
int32_t x2486 = x2482;
int32_t x2487 = x2483;
int32_t x2488 = x2485;
int32_t x2489 = x2486;
int32_t x2490 = x2487;
for(int x2491=0; x2491 < 50; x2491++) {
int32_t x2492 = x2488;
float x2493 = x1433[x2492];
float x2494 = x1421[x2492];
int32_t x2495 = x2489;
float x2496 = x1293[x2495];
int32_t x2497 = x2490;
float x2498 = x1881[x2497];
float x2499 = x2498 * x2496;
float x2500 = x2493 + x2499;
x1433[x2492] = x2500;
float x2502 = x1294[x2495];
float x2503 = x1421[x2492];
float x2504 = x1293[x2495];
float x2505 = x1881[x2497];
float x2506 = x2505 * x2503;
float x2507 = x2502 + x2506;
x1294[x2495] = x2507;
x2490 += 1;
x2488 += 1;
x2489 += 1;

}
x2483 += 50;
x2481 += 50;
x2482 += 50;

}
for(int x2519=0; x2519 < 1000; x2519++) {
float x2520 = x1833[x2519];
float x2521 = x1838[x2519];
float x2524 = x1847[x2519];
float x2522 = x2521 * x2521;
float x2523 = 1.0f - x2522;
float x2525 = x2523 * x2524;
float x2526 = x2520 + x2525;
x1833[x2519] = x2526;

}
for(int x2533=0; x2533 < 20; x2533++) {
int32_t x2534 = x2530;
int32_t x2535 = x2531;
int32_t x2536 = x2532;
int32_t x2537 = x2534;
int32_t x2538 = x2535;
int32_t x2539 = x2536;
for(int x2540=0; x2540 < 50; x2540++) {
int32_t x2541 = x2537;
float x2542 = x1800[x2541];
float x2543 = x1771[x2541];
int32_t x2544 = x2538;
float x2545 = x117[x2544];
int32_t x2546 = x2539;
float x2547 = x1833[x2546];
float x2548 = x2542 + x2547;
x1800[x2541] = x2548;
float x2550 = x122[x2544];
float x2551 = x1771[x2541];
float x2552 = x117[x2544];
float x2553 = x1833[x2546];
float x2554 = x2550 + x2553;
x122[x2544] = x2554;
x2539 += 1;
x2537 += 1;
x2538 += 1;

}
x2532 += 50;
x2530 += 50;

}
for(int x2568=0; x2568 < 20; x2568++) {
int32_t x2569 = x2565;
int32_t x2570 = x2566;
int32_t x2571 = x2567;
int32_t x2572 = x2569;
int32_t x2573 = x2570;
int32_t x2574 = x2571;
for(int x2575=0; x2575 < 50; x2575++) {
int32_t x2576 = x2572;
float x2577 = x1739[x2576];
float x2578 = x1716[x2576];
int32_t x2579 = x2573;
float x2580 = x1744[x2579];
int32_t x2581 = x2574;
float x2582 = x1800[x2581];
float x2583 = x2577 + x2582;
x1739[x2576] = x2583;
float x2585 = x1766[x2579];
float x2586 = x1716[x2576];
float x2587 = x1744[x2579];
float x2588 = x1800[x2581];
float x2589 = x2585 + x2588;
x1766[x2579] = x2589;
x2574 += 1;
x2572 += 1;
x2573 += 1;

}
x2567 += 50;
x2565 += 50;
x2566 += 50;

}
for(int x2601=0; x2601 < 20; x2601++) {
int32_t x2603 = x2601 * 50;
for(int x2602=0; x2602 < 50; x2602++) {
int32_t x2604 = x2603 + x2602;
float x2605 = x1766[x2604];
int32_t x2609 = x2602 * 50;
for(int x2606=0; x2606 < 50; x2606++) {
int32_t x2607 = x2603 + x2606;
float x2608 = x1292[x2607];
int32_t x2610 = x2609 + x2606;
float x2611 = x104[x2610];
float x2612 = x2611 * x2605;
float x2613 = x2608 + x2612;
x1292[x2607] = x2613;
float x2615 = x112[x2610];
float x2616 = x1291[x2607];
float x2617 = x2616 * x2605;
float x2618 = x2615 + x2617;
x112[x2610] = x2618;

}

}

}
for(int x2626=0; x2626 < 20; x2626++) {
int32_t x2628 = x2626 * 50;
int32_t x2632 = x2626 * 26;
for(int x2627=0; x2627 < 50; x2627++) {
int32_t x2629 = x2628 + x2627;
float x2630 = x1739[x2629];
int32_t x2635 = x2627 * 26;
for(int x2631=0; x2631 < 26; x2631++) {
int32_t x2633 = x2632 + x2631;
float x2634 = x1298[x2633];
int32_t x2636 = x2635 + x2631;
float x2637 = x91[x2636];
float x2638 = x2637 * x2630;
float x2639 = x2634 + x2638;
x1298[x2633] = x2639;
float x2641 = x99[x2636];
float x2642 = x1297[x2633];
float x2643 = x2642 * x2630;
float x2644 = x2641 + x2643;
x99[x2636] = x2644;

}

}

}
for(int x2652=0; x2652 < 1000; x2652++) {
float x2653 = x1694[x2652];
float x2654 = x1699[x2652];
float x2657 = x1711[x2652];
float x2655 = 1.0f - x2654;
float x2656 = x2655 * x2654;
float x2658 = x2656 * x2657;
float x2659 = x2653 + x2658;
x1694[x2652] = x2659;

}
for(int x2666=0; x2666 < 20; x2666++) {
int32_t x2667 = x2663;
int32_t x2668 = x2664;
int32_t x2669 = x2665;
int32_t x2670 = x2667;
int32_t x2671 = x2668;
int32_t x2672 = x2669;
for(int x2673=0; x2673 < 50; x2673++) {
int32_t x2674 = x2670;
float x2675 = x1661[x2674];
float x2676 = x1632[x2674];
int32_t x2677 = x2671;
float x2678 = x153[x2677];
int32_t x2679 = x2672;
float x2680 = x1694[x2679];
float x2681 = x2675 + x2680;
x1661[x2674] = x2681;
float x2683 = x158[x2677];
float x2684 = x1632[x2674];
float x2685 = x153[x2677];
float x2686 = x1694[x2679];
float x2687 = x2683 + x2686;
x158[x2677] = x2687;
x2672 += 1;
x2670 += 1;
x2671 += 1;

}
x2665 += 50;
x2663 += 50;

}
for(int x2701=0; x2701 < 20; x2701++) {
int32_t x2702 = x2698;
int32_t x2703 = x2699;
int32_t x2704 = x2700;
int32_t x2705 = x2702;
int32_t x2706 = x2703;
int32_t x2707 = x2704;
for(int x2708=0; x2708 < 50; x2708++) {
int32_t x2709 = x2705;
float x2710 = x1600[x2709];
float x2711 = x1577[x2709];
int32_t x2712 = x2706;
float x2713 = x1605[x2712];
int32_t x2714 = x2707;
float x2715 = x1661[x2714];
float x2716 = x2710 + x2715;
x1600[x2709] = x2716;
float x2718 = x1627[x2712];
float x2719 = x1577[x2709];
float x2720 = x1605[x2712];
float x2721 = x1661[x2714];
float x2722 = x2718 + x2721;
x1627[x2712] = x2722;
x2707 += 1;
x2705 += 1;
x2706 += 1;

}
x2700 += 50;
x2698 += 50;
x2699 += 50;

}
for(int x2734=0; x2734 < 20; x2734++) {
int32_t x2736 = x2734 * 50;
for(int x2735=0; x2735 < 50; x2735++) {
int32_t x2737 = x2736 + x2735;
float x2738 = x1627[x2737];
int32_t x2742 = x2735 * 50;
for(int x2739=0; x2739 < 50; x2739++) {
int32_t x2740 = x2736 + x2739;
float x2741 = x1292[x2740];
int32_t x2743 = x2742 + x2739;
float x2744 = x140[x2743];
float x2745 = x2744 * x2738;
float x2746 = x2741 + x2745;
x1292[x2740] = x2746;
float x2748 = x148[x2743];
float x2749 = x1291[x2740];
float x2750 = x2749 * x2738;
float x2751 = x2748 + x2750;
x148[x2743] = x2751;

}

}

}
for(int x2759=0; x2759 < 20; x2759++) {
int32_t x2761 = x2759 * 50;
int32_t x2765 = x2759 * 26;
for(int x2760=0; x2760 < 50; x2760++) {
int32_t x2762 = x2761 + x2760;
float x2763 = x1600[x2762];
int32_t x2768 = x2760 * 26;
for(int x2764=0; x2764 < 26; x2764++) {
int32_t x2766 = x2765 + x2764;
float x2767 = x1298[x2766];
int32_t x2769 = x2768 + x2764;
float x2770 = x127[x2769];
float x2771 = x2770 * x2763;
float x2772 = x2767 + x2771;
x1298[x2766] = x2772;
float x2774 = x135[x2769];
float x2775 = x1297[x2766];
float x2776 = x2775 * x2763;
float x2777 = x2774 + x2776;
x135[x2769] = x2777;

}

}

}
for(int x2785=0; x2785 < 1000; x2785++) {
float x2786 = x1555[x2785];
float x2787 = x1560[x2785];
float x2790 = x1572[x2785];
float x2788 = 1.0f - x2787;
float x2789 = x2788 * x2787;
float x2791 = x2789 * x2790;
float x2792 = x2786 + x2791;
x1555[x2785] = x2792;

}
for(int x2799=0; x2799 < 20; x2799++) {
int32_t x2800 = x2796;
int32_t x2801 = x2797;
int32_t x2802 = x2798;
int32_t x2803 = x2800;
int32_t x2804 = x2801;
int32_t x2805 = x2802;
for(int x2806=0; x2806 < 50; x2806++) {
int32_t x2807 = x2803;
float x2808 = x1522[x2807];
float x2809 = x1493[x2807];
int32_t x2810 = x2804;
float x2811 = x81[x2810];
int32_t x2812 = x2805;
float x2813 = x1555[x2812];
float x2814 = x2808 + x2813;
x1522[x2807] = x2814;
float x2816 = x86[x2810];
float x2817 = x1493[x2807];
float x2818 = x81[x2810];
float x2819 = x1555[x2812];
float x2820 = x2816 + x2819;
x86[x2810] = x2820;
x2805 += 1;
x2803 += 1;
x2804 += 1;

}
x2798 += 50;
x2796 += 50;

}
for(int x2834=0; x2834 < 20; x2834++) {
int32_t x2835 = x2831;
int32_t x2836 = x2832;
int32_t x2837 = x2833;
int32_t x2838 = x2835;
int32_t x2839 = x2836;
int32_t x2840 = x2837;
for(int x2841=0; x2841 < 50; x2841++) {
int32_t x2842 = x2838;
float x2843 = x1461[x2842];
float x2844 = x1438[x2842];
int32_t x2845 = x2839;
float x2846 = x1466[x2845];
int32_t x2847 = x2840;
float x2848 = x1522[x2847];
float x2849 = x2843 + x2848;
x1461[x2842] = x2849;
float x2851 = x1488[x2845];
float x2852 = x1438[x2842];
float x2853 = x1466[x2845];
float x2854 = x1522[x2847];
float x2855 = x2851 + x2854;
x1488[x2845] = x2855;
x2840 += 1;
x2838 += 1;
x2839 += 1;

}
x2833 += 50;
x2831 += 50;
x2832 += 50;

}
for(int x2867=0; x2867 < 20; x2867++) {
int32_t x2869 = x2867 * 50;
for(int x2868=0; x2868 < 50; x2868++) {
int32_t x2870 = x2869 + x2868;
float x2871 = x1488[x2870];
int32_t x2875 = x2868 * 50;
for(int x2872=0; x2872 < 50; x2872++) {
int32_t x2873 = x2869 + x2872;
float x2874 = x1292[x2873];
int32_t x2876 = x2875 + x2872;
float x2877 = x68[x2876];
float x2878 = x2877 * x2871;
float x2879 = x2874 + x2878;
x1292[x2873] = x2879;
float x2881 = x76[x2876];
float x2882 = x1291[x2873];
float x2883 = x2882 * x2871;
float x2884 = x2881 + x2883;
x76[x2876] = x2884;

}

}

}
for(int x2892=0; x2892 < 20; x2892++) {
int32_t x2894 = x2892 * 50;
int32_t x2898 = x2892 * 26;
for(int x2893=0; x2893 < 50; x2893++) {
int32_t x2895 = x2894 + x2893;
float x2896 = x1461[x2895];
int32_t x2901 = x2893 * 26;
for(int x2897=0; x2897 < 26; x2897++) {
int32_t x2899 = x2898 + x2897;
float x2900 = x1298[x2899];
int32_t x2902 = x2901 + x2897;
float x2903 = x55[x2902];
float x2904 = x2903 * x2896;
float x2905 = x2900 + x2904;
x1298[x2899] = x2905;
float x2907 = x63[x2902];
float x2908 = x1297[x2899];
float x2909 = x2908 * x2896;
float x2910 = x2907 + x2909;
x63[x2902] = x2910;

}

}

}
for(int x2918=0; x2918 < 1000; x2918++) {
float x2919 = x1416[x2918];
float x2920 = x1421[x2918];
float x2923 = x1433[x2918];
float x2921 = 1.0f - x2920;
float x2922 = x2921 * x2920;
float x2924 = x2922 * x2923;
float x2925 = x2919 + x2924;
x1416[x2918] = x2925;

}
for(int x2932=0; x2932 < 20; x2932++) {
int32_t x2933 = x2929;
int32_t x2934 = x2930;
int32_t x2935 = x2931;
int32_t x2936 = x2933;
int32_t x2937 = x2934;
int32_t x2938 = x2935;
for(int x2939=0; x2939 < 50; x2939++) {
int32_t x2940 = x2936;
float x2941 = x1383[x2940];
float x2942 = x1354[x2940];
int32_t x2943 = x2937;
float x2944 = x44[x2943];
int32_t x2945 = x2938;
float x2946 = x1416[x2945];
float x2947 = x2941 + x2946;
x1383[x2940] = x2947;
float x2949 = x50[x2943];
float x2950 = x1354[x2940];
float x2951 = x44[x2943];
float x2952 = x1416[x2945];
float x2953 = x2949 + x2952;
x50[x2943] = x2953;
x2938 += 1;
x2936 += 1;
x2937 += 1;

}
x2931 += 50;
x2929 += 50;

}
for(int x2967=0; x2967 < 20; x2967++) {
int32_t x2968 = x2964;
int32_t x2969 = x2965;
int32_t x2970 = x2966;
int32_t x2971 = x2968;
int32_t x2972 = x2969;
int32_t x2973 = x2970;
for(int x2974=0; x2974 < 50; x2974++) {
int32_t x2975 = x2971;
float x2976 = x1322[x2975];
float x2977 = x1299[x2975];
int32_t x2978 = x2972;
float x2979 = x1327[x2978];
int32_t x2980 = x2973;
float x2981 = x1383[x2980];
float x2982 = x2976 + x2981;
x1322[x2975] = x2982;
float x2984 = x1349[x2978];
float x2985 = x1299[x2975];
float x2986 = x1327[x2978];
float x2987 = x1383[x2980];
float x2988 = x2984 + x2987;
x1349[x2978] = x2988;
x2973 += 1;
x2971 += 1;
x2972 += 1;

}
x2966 += 50;
x2964 += 50;
x2965 += 50;

}
for(int x3000=0; x3000 < 20; x3000++) {
int32_t x3002 = x3000 * 50;
for(int x3001=0; x3001 < 50; x3001++) {
int32_t x3003 = x3002 + x3001;
float x3004 = x1349[x3003];
int32_t x3008 = x3001 * 50;
for(int x3005=0; x3005 < 50; x3005++) {
int32_t x3006 = x3002 + x3005;
float x3007 = x1292[x3006];
int32_t x3009 = x3008 + x3005;
float x3010 = x30[x3009];
float x3011 = x3010 * x3004;
float x3012 = x3007 + x3011;
x1292[x3006] = x3012;
float x3014 = x39[x3009];
float x3015 = x1291[x3006];
float x3016 = x3015 * x3004;
float x3017 = x3014 + x3016;
x39[x3009] = x3017;

}

}

}
for(int x3025=0; x3025 < 20; x3025++) {
int32_t x3027 = x3025 * 50;
int32_t x3031 = x3025 * 26;
for(int x3026=0; x3026 < 50; x3026++) {
int32_t x3028 = x3027 + x3026;
float x3029 = x1322[x3028];
int32_t x3034 = x3026 * 26;
for(int x3030=0; x3030 < 26; x3030++) {
int32_t x3032 = x3031 + x3030;
float x3033 = x1298[x3032];
int32_t x3035 = x3034 + x3030;
float x3036 = x16[x3035];
float x3037 = x3036 * x3029;
float x3038 = x3033 + x3037;
x1298[x3032] = x3038;
float x3040 = x25[x3035];
float x3041 = x1297[x3032];
float x3042 = x3041 * x3029;
float x3043 = x3040 + x3042;
x25[x3035] = x3043;

}

}

}
} else {
float x3052 = 0.0f;
float x3053 = x3052;
float x3054 = x1289[0];
float x3055 = x3053 + x3054;
x3052 = x3055;
float x3057 = x3052;
float* x3058 = (float*)myMalloc(1 * sizeof(float));;
x3058[0] = x3057;
float* x3060 = (float*)myMalloc(1 * sizeof(float));;
for(int x3061=0; x3061 < 1; x3061++) {
x3060[x3061] = 0.0f;

}
float x3065 = x3060[0];
x3060[0] = 1.0f;
float x3067 = x3058[0];
x297[0] = x3067;
// += tensor of dim 0
float x3070 = x3060[0];
float x3071 = x1290[0];
float x3072 = x3071 + x3070;
x1290[0] = x3072;
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
float** x3947 = (float**)myMalloc(6 * sizeof(float*));;
x3947[0] = x329;
x3947[1] = x334;
x3947[2] = x339;
x3947[3] = x345;
x3947[4] = x350;
x3947[5] = x355;
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
float* x375 = (float*)myMalloc(1000 * sizeof(float));;
for(int x376=0; x376 < 20; x376++) {
int32_t x380 = x376 * 26;
int32_t x390 = x376 * 50;
for(int x377=0; x377 < 50; x377++) {
float x378 = 0.0f;
int32_t x383 = x377 * 26;
for(int x379=0; x379 < 26; x379++) {
int32_t x381 = x380 + x379;
float x382 = x373[x381];
int32_t x384 = x383 + x379;
float x385 = x16[x384];
float x386 = x382 * x385;
x378 += x386;

}
float x392 = x378;
int32_t x391 = x390 + x377;
x375[x391] = x392;

}

}
float* x398 = (float*)myMalloc(1000 * sizeof(float));;
for(int x399=0; x399 < 1000; x399++) {
x398[x399] = 0.0f;

}
float* x403 = (float*)myMalloc(1000 * sizeof(float));;
for(int x404=0; x404 < 20; x404++) {
int32_t x408 = x404 * 50;
for(int x405=0; x405 < 50; x405++) {
float x406 = 0.0f;
int32_t x411 = x405 * 50;
for(int x407=0; x407 < 50; x407++) {
int32_t x409 = x408 + x407;
float x410 = x367[x409];
int32_t x412 = x411 + x407;
float x413 = x30[x412];
float x414 = x410 * x413;
x406 += x414;

}
float x419 = x406;
int32_t x418 = x408 + x405;
x403[x418] = x419;

}

}
float* x425 = (float*)myMalloc(1000 * sizeof(float));;
for(int x426=0; x426 < 1000; x426++) {
x425[x426] = 0.0f;

}
float* x430 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x431 = 0;
int32_t x432 = 0;
int32_t x433 = 0;
for(int x434=0; x434 < 20; x434++) {
int32_t x435 = x432;
int32_t x436 = x433;
int32_t x437 = x431;
int32_t x438 = x437;
int32_t x439 = x435;
int32_t x440 = x436;
for(int x441=0; x441 < 50; x441++) {
int32_t x442 = x438;
int32_t x443 = x439;
float x444 = x375[x443];
int32_t x445 = x440;
float x446 = x403[x445];
float x447 = x444 + x446;
x430[x442] = x447;
x438 += 1;
x439 += 1;
x440 += 1;

}
x431 += 50;
x432 += 50;
x433 += 50;

}
float* x459 = (float*)myMalloc(1000 * sizeof(float));;
for(int x460=0; x460 < 1000; x460++) {
x459[x460] = 0.0f;

}
float* x464 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x465 = 0;
int32_t x466 = 0;
int32_t x467 = 0;
for(int x468=0; x468 < 20; x468++) {
int32_t x469 = x466;
int32_t x470 = x467;
int32_t x471 = x465;
int32_t x472 = x471;
int32_t x473 = x469;
int32_t x474 = x470;
for(int x475=0; x475 < 50; x475++) {
int32_t x476 = x472;
int32_t x477 = x473;
float x478 = x430[x477];
int32_t x479 = x474;
float x480 = x44[x479];
float x481 = x478 + x480;
x464[x476] = x481;
x472 += 1;
x473 += 1;
x474 += 1;

}
x465 += 50;
x466 += 50;

}
float* x492 = (float*)myMalloc(1000 * sizeof(float));;
for(int x493=0; x493 < 1000; x493++) {
x492[x493] = 0.0f;

}
float* x497 = (float*)myMalloc(1000 * sizeof(float));;
for(int x498=0; x498 < 1000; x498++) {
float x499 = x464[x498];
float x500 = -1.0f * x499;
double x501 = (double)x500;
double x502 = exp(x501);
float x503 = (float)x502;
float x504 = x503 + 1.0f;
float x505 = 1.0f / x504;
x497[x498] = x505;

}
float* x509 = (float*)myMalloc(1000 * sizeof(float));;
for(int x510=0; x510 < 1000; x510++) {
x509[x510] = 0.0f;

}
float* x514 = (float*)myMalloc(1000 * sizeof(float));;
for(int x515=0; x515 < 20; x515++) {
int32_t x519 = x515 * 26;
int32_t x529 = x515 * 50;
for(int x516=0; x516 < 50; x516++) {
float x517 = 0.0f;
int32_t x522 = x516 * 26;
for(int x518=0; x518 < 26; x518++) {
int32_t x520 = x519 + x518;
float x521 = x373[x520];
int32_t x523 = x522 + x518;
float x524 = x55[x523];
float x525 = x521 * x524;
x517 += x525;

}
float x531 = x517;
int32_t x530 = x529 + x516;
x514[x530] = x531;

}

}
float* x537 = (float*)myMalloc(1000 * sizeof(float));;
for(int x538=0; x538 < 1000; x538++) {
x537[x538] = 0.0f;

}
float* x542 = (float*)myMalloc(1000 * sizeof(float));;
for(int x543=0; x543 < 20; x543++) {
int32_t x547 = x543 * 50;
for(int x544=0; x544 < 50; x544++) {
float x545 = 0.0f;
int32_t x550 = x544 * 50;
for(int x546=0; x546 < 50; x546++) {
int32_t x548 = x547 + x546;
float x549 = x367[x548];
int32_t x551 = x550 + x546;
float x552 = x68[x551];
float x553 = x549 * x552;
x545 += x553;

}
float x558 = x545;
int32_t x557 = x547 + x544;
x542[x557] = x558;

}

}
float* x564 = (float*)myMalloc(1000 * sizeof(float));;
for(int x565=0; x565 < 1000; x565++) {
x564[x565] = 0.0f;

}
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
float x583 = x514[x582];
int32_t x584 = x579;
float x585 = x542[x584];
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
for(int x599=0; x599 < 1000; x599++) {
x598[x599] = 0.0f;

}
float* x603 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x604 = 0;
int32_t x605 = 0;
int32_t x606 = 0;
for(int x607=0; x607 < 20; x607++) {
int32_t x608 = x605;
int32_t x609 = x606;
int32_t x610 = x604;
int32_t x611 = x610;
int32_t x612 = x608;
int32_t x613 = x609;
for(int x614=0; x614 < 50; x614++) {
int32_t x615 = x611;
int32_t x616 = x612;
float x617 = x569[x616];
int32_t x618 = x613;
float x619 = x81[x618];
float x620 = x617 + x619;
x603[x615] = x620;
x611 += 1;
x612 += 1;
x613 += 1;

}
x604 += 50;
x605 += 50;

}
float* x631 = (float*)myMalloc(1000 * sizeof(float));;
for(int x632=0; x632 < 1000; x632++) {
x631[x632] = 0.0f;

}
float* x636 = (float*)myMalloc(1000 * sizeof(float));;
for(int x637=0; x637 < 1000; x637++) {
float x638 = x603[x637];
float x639 = -1.0f * x638;
double x640 = (double)x639;
double x641 = exp(x640);
float x642 = (float)x641;
float x643 = x642 + 1.0f;
float x644 = 1.0f / x643;
x636[x637] = x644;

}
float* x648 = (float*)myMalloc(1000 * sizeof(float));;
for(int x649=0; x649 < 1000; x649++) {
x648[x649] = 0.0f;

}
float* x653 = (float*)myMalloc(1000 * sizeof(float));;
for(int x654=0; x654 < 20; x654++) {
int32_t x658 = x654 * 26;
int32_t x668 = x654 * 50;
for(int x655=0; x655 < 50; x655++) {
float x656 = 0.0f;
int32_t x661 = x655 * 26;
for(int x657=0; x657 < 26; x657++) {
int32_t x659 = x658 + x657;
float x660 = x373[x659];
int32_t x662 = x661 + x657;
float x663 = x127[x662];
float x664 = x660 * x663;
x656 += x664;

}
float x670 = x656;
int32_t x669 = x668 + x655;
x653[x669] = x670;

}

}
float* x676 = (float*)myMalloc(1000 * sizeof(float));;
for(int x677=0; x677 < 1000; x677++) {
x676[x677] = 0.0f;

}
float* x681 = (float*)myMalloc(1000 * sizeof(float));;
for(int x682=0; x682 < 20; x682++) {
int32_t x686 = x682 * 50;
for(int x683=0; x683 < 50; x683++) {
float x684 = 0.0f;
int32_t x689 = x683 * 50;
for(int x685=0; x685 < 50; x685++) {
int32_t x687 = x686 + x685;
float x688 = x367[x687];
int32_t x690 = x689 + x685;
float x691 = x140[x690];
float x692 = x688 * x691;
x684 += x692;

}
float x697 = x684;
int32_t x696 = x686 + x683;
x681[x696] = x697;

}

}
float* x703 = (float*)myMalloc(1000 * sizeof(float));;
for(int x704=0; x704 < 1000; x704++) {
x703[x704] = 0.0f;

}
float* x708 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x709 = 0;
int32_t x710 = 0;
int32_t x711 = 0;
for(int x712=0; x712 < 20; x712++) {
int32_t x713 = x710;
int32_t x714 = x711;
int32_t x715 = x709;
int32_t x716 = x715;
int32_t x717 = x713;
int32_t x718 = x714;
for(int x719=0; x719 < 50; x719++) {
int32_t x720 = x716;
int32_t x721 = x717;
float x722 = x653[x721];
int32_t x723 = x718;
float x724 = x681[x723];
float x725 = x722 + x724;
x708[x720] = x725;
x716 += 1;
x717 += 1;
x718 += 1;

}
x709 += 50;
x710 += 50;
x711 += 50;

}
float* x737 = (float*)myMalloc(1000 * sizeof(float));;
for(int x738=0; x738 < 1000; x738++) {
x737[x738] = 0.0f;

}
float* x742 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x743 = 0;
int32_t x744 = 0;
int32_t x745 = 0;
for(int x746=0; x746 < 20; x746++) {
int32_t x747 = x744;
int32_t x748 = x745;
int32_t x749 = x743;
int32_t x750 = x749;
int32_t x751 = x747;
int32_t x752 = x748;
for(int x753=0; x753 < 50; x753++) {
int32_t x754 = x750;
int32_t x755 = x751;
float x756 = x708[x755];
int32_t x757 = x752;
float x758 = x153[x757];
float x759 = x756 + x758;
x742[x754] = x759;
x750 += 1;
x751 += 1;
x752 += 1;

}
x743 += 50;
x744 += 50;

}
float* x770 = (float*)myMalloc(1000 * sizeof(float));;
for(int x771=0; x771 < 1000; x771++) {
x770[x771] = 0.0f;

}
float* x775 = (float*)myMalloc(1000 * sizeof(float));;
for(int x776=0; x776 < 1000; x776++) {
float x777 = x742[x776];
float x778 = -1.0f * x777;
double x779 = (double)x778;
double x780 = exp(x779);
float x781 = (float)x780;
float x782 = x781 + 1.0f;
float x783 = 1.0f / x782;
x775[x776] = x783;

}
float* x787 = (float*)myMalloc(1000 * sizeof(float));;
for(int x788=0; x788 < 1000; x788++) {
x787[x788] = 0.0f;

}
float* x792 = (float*)myMalloc(1000 * sizeof(float));;
for(int x793=0; x793 < 20; x793++) {
int32_t x797 = x793 * 26;
int32_t x807 = x793 * 50;
for(int x794=0; x794 < 50; x794++) {
float x795 = 0.0f;
int32_t x800 = x794 * 26;
for(int x796=0; x796 < 26; x796++) {
int32_t x798 = x797 + x796;
float x799 = x373[x798];
int32_t x801 = x800 + x796;
float x802 = x91[x801];
float x803 = x799 * x802;
x795 += x803;

}
float x809 = x795;
int32_t x808 = x807 + x794;
x792[x808] = x809;

}

}
float* x815 = (float*)myMalloc(1000 * sizeof(float));;
for(int x816=0; x816 < 1000; x816++) {
x815[x816] = 0.0f;

}
float* x820 = (float*)myMalloc(1000 * sizeof(float));;
for(int x821=0; x821 < 20; x821++) {
int32_t x825 = x821 * 50;
for(int x822=0; x822 < 50; x822++) {
float x823 = 0.0f;
int32_t x828 = x822 * 50;
for(int x824=0; x824 < 50; x824++) {
int32_t x826 = x825 + x824;
float x827 = x367[x826];
int32_t x829 = x828 + x824;
float x830 = x104[x829];
float x831 = x827 * x830;
x823 += x831;

}
float x836 = x823;
int32_t x835 = x825 + x822;
x820[x835] = x836;

}

}
float* x842 = (float*)myMalloc(1000 * sizeof(float));;
for(int x843=0; x843 < 1000; x843++) {
x842[x843] = 0.0f;

}
float* x847 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x848 = 0;
int32_t x849 = 0;
int32_t x850 = 0;
for(int x851=0; x851 < 20; x851++) {
int32_t x852 = x849;
int32_t x853 = x850;
int32_t x854 = x848;
int32_t x855 = x854;
int32_t x856 = x852;
int32_t x857 = x853;
for(int x858=0; x858 < 50; x858++) {
int32_t x859 = x855;
int32_t x860 = x856;
float x861 = x792[x860];
int32_t x862 = x857;
float x863 = x820[x862];
float x864 = x861 + x863;
x847[x859] = x864;
x855 += 1;
x856 += 1;
x857 += 1;

}
x848 += 50;
x849 += 50;
x850 += 50;

}
float* x876 = (float*)myMalloc(1000 * sizeof(float));;
for(int x877=0; x877 < 1000; x877++) {
x876[x877] = 0.0f;

}
float* x881 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x882 = 0;
int32_t x883 = 0;
int32_t x884 = 0;
for(int x885=0; x885 < 20; x885++) {
int32_t x886 = x883;
int32_t x887 = x884;
int32_t x888 = x882;
int32_t x889 = x888;
int32_t x890 = x886;
int32_t x891 = x887;
for(int x892=0; x892 < 50; x892++) {
int32_t x893 = x889;
int32_t x894 = x890;
float x895 = x847[x894];
int32_t x896 = x891;
float x897 = x117[x896];
float x898 = x895 + x897;
x881[x893] = x898;
x889 += 1;
x890 += 1;
x891 += 1;

}
x882 += 50;
x883 += 50;

}
float* x909 = (float*)myMalloc(1000 * sizeof(float));;
for(int x910=0; x910 < 1000; x910++) {
x909[x910] = 0.0f;

}
float* x914 = (float*)myMalloc(1000 * sizeof(float));;
for(int x915=0; x915 < 1000; x915++) {
float x916 = x881[x915];
double x917 = (double)x916;
double x918 = tanh(x917);
float x919 = (float)x918;
x914[x915] = x919;

}
float* x923 = (float*)myMalloc(1000 * sizeof(float));;
for(int x924=0; x924 < 1000; x924++) {
x923[x924] = 0.0f;

}
float* x928 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x929 = 0;
int32_t x930 = 0;
int32_t x931 = 0;
for(int x932=0; x932 < 20; x932++) {
int32_t x933 = x930;
int32_t x934 = x931;
int32_t x935 = x929;
int32_t x936 = x935;
int32_t x937 = x933;
int32_t x938 = x934;
for(int x939=0; x939 < 50; x939++) {
int32_t x940 = x936;
int32_t x941 = x937;
float x942 = x497[x941];
int32_t x943 = x938;
float x944 = x369[x943];
float x945 = x942 * x944;
x928[x940] = x945;
x936 += 1;
x937 += 1;
x938 += 1;

}
x929 += 50;
x930 += 50;
x931 += 50;

}
float* x957 = (float*)myMalloc(1000 * sizeof(float));;
for(int x958=0; x958 < 1000; x958++) {
x957[x958] = 0.0f;

}
float* x962 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x963 = 0;
int32_t x964 = 0;
int32_t x965 = 0;
for(int x966=0; x966 < 20; x966++) {
int32_t x967 = x964;
int32_t x968 = x965;
int32_t x969 = x963;
int32_t x970 = x969;
int32_t x971 = x967;
int32_t x972 = x968;
for(int x973=0; x973 < 50; x973++) {
int32_t x974 = x970;
int32_t x975 = x971;
float x976 = x636[x975];
int32_t x977 = x972;
float x978 = x914[x977];
float x979 = x976 * x978;
x962[x974] = x979;
x970 += 1;
x971 += 1;
x972 += 1;

}
x963 += 50;
x964 += 50;
x965 += 50;

}
float* x991 = (float*)myMalloc(1000 * sizeof(float));;
for(int x992=0; x992 < 1000; x992++) {
x991[x992] = 0.0f;

}
float* x996 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x997 = 0;
int32_t x998 = 0;
int32_t x999 = 0;
for(int x1000=0; x1000 < 20; x1000++) {
int32_t x1001 = x998;
int32_t x1002 = x999;
int32_t x1003 = x997;
int32_t x1004 = x1003;
int32_t x1005 = x1001;
int32_t x1006 = x1002;
for(int x1007=0; x1007 < 50; x1007++) {
int32_t x1008 = x1004;
int32_t x1009 = x1005;
float x1010 = x928[x1009];
int32_t x1011 = x1006;
float x1012 = x962[x1011];
float x1013 = x1010 + x1012;
x996[x1008] = x1013;
x1004 += 1;
x1005 += 1;
x1006 += 1;

}
x997 += 50;
x998 += 50;
x999 += 50;

}
float* x1025 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1026=0; x1026 < 1000; x1026++) {
x1025[x1026] = 0.0f;

}
float* x1030 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1031=0; x1031 < 1000; x1031++) {
float x1032 = x996[x1031];
double x1033 = (double)x1032;
double x1034 = tanh(x1033);
float x1035 = (float)x1034;
x1030[x1031] = x1035;

}
float* x1039 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1040=0; x1040 < 1000; x1040++) {
x1039[x1040] = 0.0f;

}
float* x1044 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x1045 = 0;
int32_t x1046 = 0;
int32_t x1047 = 0;
for(int x1048=0; x1048 < 20; x1048++) {
int32_t x1049 = x1046;
int32_t x1050 = x1047;
int32_t x1051 = x1045;
int32_t x1052 = x1051;
int32_t x1053 = x1049;
int32_t x1054 = x1050;
for(int x1055=0; x1055 < 50; x1055++) {
int32_t x1056 = x1052;
int32_t x1057 = x1053;
float x1058 = x775[x1057];
int32_t x1059 = x1054;
float x1060 = x1030[x1059];
float x1061 = x1058 * x1060;
x1044[x1056] = x1061;
x1052 += 1;
x1053 += 1;
x1054 += 1;

}
x1045 += 50;
x1046 += 50;
x1047 += 50;

}
float* x1073 = (float*)myMalloc(1000 * sizeof(float));;
for(int x1074=0; x1074 < 1000; x1074++) {
x1073[x1074] = 0.0f;

}
float* x1078 = (float*)myMalloc(520 * sizeof(float));;
for(int x1079=0; x1079 < 20; x1079++) {
int32_t x1083 = x1079 * 50;
int32_t x1093 = x1079 * 26;
for(int x1080=0; x1080 < 26; x1080++) {
float x1081 = 0.0f;
int32_t x1086 = x1080 * 50;
for(int x1082=0; x1082 < 50; x1082++) {
int32_t x1084 = x1083 + x1082;
float x1085 = x1044[x1084];
int32_t x1087 = x1086 + x1082;
float x1088 = x163[x1087];
float x1089 = x1085 * x1088;
x1081 += x1089;

}
float x1095 = x1081;
int32_t x1094 = x1093 + x1080;
x1078[x1094] = x1095;

}

}
float* x1101 = (float*)myMalloc(520 * sizeof(float));;
for(int x1103=0; x1103 < 520; x1103++) {
x1101[x1103] = 0.0f;

}
float* x1107 = (float*)myMalloc(520 * sizeof(float));;
int32_t x1108 = 0;
int32_t x1109 = 0;
int32_t x1110 = 0;
for(int x1111=0; x1111 < 20; x1111++) {
int32_t x1112 = x1109;
int32_t x1113 = x1110;
int32_t x1114 = x1108;
int32_t x1115 = x1114;
int32_t x1116 = x1112;
int32_t x1117 = x1113;
for(int x1118=0; x1118 < 26; x1118++) {
int32_t x1119 = x1115;
int32_t x1120 = x1116;
float x1121 = x1078[x1120];
int32_t x1122 = x1117;
float x1123 = x176[x1122];
float x1124 = x1121 + x1123;
x1107[x1119] = x1124;
x1115 += 1;
x1116 += 1;
x1117 += 1;

}
x1108 += 26;
x1109 += 26;

}
float* x1135 = (float*)myMalloc(520 * sizeof(float));;
for(int x1136=0; x1136 < 520; x1136++) {
x1135[x1136] = 0.0f;

}
int* x1140 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x1141=0; x1141 < 20; x1141++) {
int32_t x1142 = x1141 * 20;
int32_t x1143 = x363 + x1142;
int32_t x1144 = x274[x1143];
x1140[x1141] = x1144;

}
float* x1148 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1149 = 0;
for(int x1150=0; x1150 < 20; x1150++) {
float x1151 = -3.4028235E38f;
for(int x1152=0; x1152 < 26; x1152++) {
int32_t x1153 = x1149;
float x1154 = x1107[x1153];
float x1155 = x1151;
bool x1156 = x1154 > x1155;
if (x1156) {
float x1157 = x1107[x1153];
x1151 = x1157;
} else {
}
x1149 += 1;

}
float x1164 = x1151;
x1148[x1150] = x1164;

}
float* x1168 = (float*)myMalloc(520 * sizeof(float));;
for(int x1169=0; x1169 < 520; x1169++) {
x1168[x1169] = 0.0f;

}
int32_t x1173 = 0;
for(int x1174=0; x1174 < 20; x1174++) {
for(int x1175=0; x1175 < 26; x1175++) {
int32_t x1176 = x1173;
float x1177 = x1107[x1176];
float x1178 = x1148[x1174];
float x1179 = x1177 - x1178;
double x1180 = (double)x1179;
double x1181 = exp(x1180);
float x1182 = (float)x1181;
x1168[x1176] = x1182;
x1173 += 1;

}

}
float* x1189 = (float*)myMalloc(20 * sizeof(float));;
for(int x1190=0; x1190 < 20; x1190++) {
x1189[x1190] = 0.0f;

}
for(int x1194=0; x1194 < 20; x1194++) {
int32_t x1195 = x1194;
int32_t x1196 = x1194 * 26;
int32_t x1197 = x1196;
for(int x1198=0; x1198 < 26; x1198++) {
int32_t x1199 = x1195;
float x1200 = x1189[x1199];
int32_t x1201 = x1197;
float x1202 = x1168[x1201];
float x1203 = x1200 + x1202;
x1189[x1199] = x1203;
x1197 += 1;

}

}
x1173 = 0;
for(int x1211=0; x1211 < 20; x1211++) {
float x1212 = x1148[x1211];
float x1213 = x1189[x1211];
double x1214 = (double)x1213;
double x1215 = log(x1214);
float x1216 = (float)x1215;
float x1217 = x1212 + x1216;
for(int x1218=0; x1218 < 26; x1218++) {
int32_t x1219 = x1173;
float x1220 = x1107[x1219];
float x1221 = x1220 - x1217;
x1168[x1219] = x1221;
x1173 += 1;

}

}
float* x1228 = (float*)myMalloc(520 * sizeof(float));;
for(int x1229=0; x1229 < 520; x1229++) {
x1228[x1229] = 0.0f;

}
float* x1233 = (float*)myMalloc(20 * sizeof(float));;
int32_t x1234 = 0;
for(int x1235=0; x1235 < 20; x1235++) {
int32_t x1236 = x1234;
int32_t x1237 = x1140[x1235];
int32_t x1238 = x1236 + x1237;
float x1239 = x1168[x1238];
float x1240 = -1.0f * x1239;
x1233[x1235] = x1240;
x1234 += 26;

}
float* x1245 = (float*)myMalloc(20 * sizeof(float));;
for(int x1246=0; x1246 < 20; x1246++) {
x1245[x1246] = 0.0f;

}
float x1250 = 0.0f;
for(int x1251=0; x1251 < 20; x1251++) {
float x1252 = x1250;
float x1253 = x1233[x1251];
float x1254 = x1252 + x1253;
x1250 = x1254;

}
float x1258 = x1250;
float* x1259 = (float*)myMalloc(1 * sizeof(float));;
x1259[0] = x1258;
float* x1261 = (float*)myMalloc(1 * sizeof(float));;
for(int x1262=0; x1262 < 1; x1262++) {
x1261[x1262] = 0.0f;

}
float* x1266 = (float*)myMalloc(1 * sizeof(float));;
int32_t x1267 = 0;
int32_t x1268 = 0;
int32_t x1269 = 0;
int32_t x1270 = x1267;
int32_t x1271 = x1268;
float x1272 = x365[x1271];
int32_t x1273 = x1269;
float x1274 = x1259[x1273];
float x1275 = x1272 + x1274;
x1266[x1270] = x1275;
x1267 += 1;
float* x1278 = (float*)myMalloc(1 * sizeof(float));;
for(int x1279=0; x1279 < 1; x1279++) {
x1278[x1279] = 0.0f;

}
float** x3077 = (float**)myMalloc(6 * sizeof(float*));;
x3077[0] = x1266;
x3077[1] = x1278;
x3077[2] = x1044;
x3077[3] = x1073;
x3077[4] = x996;
x3077[5] = x1025;
int32_t x1283 = x363 + 1;
x1284(x1283,x3077);
int32_t x3086 = 0;
int32_t x3087 = 0;
int32_t x3088 = 0;
int32_t x3089 = x3086;
float x3090 = x366[x3089];
float x3091 = x365[x3089];
int32_t x3092 = x3087;
float x3093 = x1259[x3092];
int32_t x3094 = x3088;
float x3095 = x1278[x3094];
float x3096 = x3090 + x3095;
x366[x3089] = x3096;
float x3098 = x1261[x3092];
float x3099 = x365[x3089];
float x3100 = x1259[x3092];
float x3101 = x1278[x3094];
float x3102 = x3098 + x3101;
x1261[x3092] = x3102;
x3088 += 1;
// += tensor of dim 0
float x3106 = x1261[0];
for(int x3107=0; x3107 < 20; x3107++) {
float x3108 = x1245[x3107];
float x3109 = x3108 + x3106;
x1245[x3107] = x3109;

}
int32_t x3113 = 0;
for(int x3114=0; x3114 < 20; x3114++) {
int32_t x3115 = x3113;
int32_t x3116 = x1140[x3114];
int32_t x3117 = x3115 + x3116;
float x3118 = x1228[x3117];
float x3119 = x1245[x3114];
float x3120 = -1.0f * x3119;
float x3121 = x3118 + x3120;
x1228[x3117] = x3121;
x3113 += 26;

}
float* x3126 = (float*)myMalloc(20 * sizeof(float));;
for(int x3127=0; x3127 < 20; x3127++) {
x3126[x3127] = 0.0f;

}
for(int x3131=0; x3131 < 20; x3131++) {
int32_t x3132 = x3131;
int32_t x3133 = x3131 * 26;
int32_t x3134 = x3133;
for(int x3135=0; x3135 < 26; x3135++) {
int32_t x3136 = x3132;
float x3137 = x3126[x3136];
int32_t x3138 = x3134;
float x3139 = x1228[x3138];
float x3140 = x3137 + x3139;
x3126[x3136] = x3140;
x3134 += 1;

}

}
int32_t x3147 = 0;
for(int x3148=0; x3148 < 20; x3148++) {
for(int x3149=0; x3149 < 26; x3149++) {
int32_t x3150 = x3147;
float x3151 = x1135[x3150];
float x3152 = x1228[x3150];
float x3153 = x1168[x3150];
float x3157 = x3126[x3148];
double x3154 = (double)x3153;
double x3155 = exp(x3154);
float x3156 = (float)x3155;
float x3158 = x3156 * x3157;
float x3159 = x3152 - x3158;
float x3160 = x3151 + x3159;
x1135[x3150] = x3160;
x3147 += 1;

}

}
int32_t x3167 = 0;
int32_t x3168 = 0;
int32_t x3169 = 0;
for(int x3170=0; x3170 < 20; x3170++) {
int32_t x3171 = x3167;
int32_t x3172 = x3168;
int32_t x3173 = x3169;
int32_t x3174 = x3171;
int32_t x3175 = x3172;
int32_t x3176 = x3173;
for(int x3177=0; x3177 < 26; x3177++) {
int32_t x3178 = x3174;
float x3179 = x1101[x3178];
float x3180 = x1078[x3178];
int32_t x3181 = x3175;
float x3182 = x176[x3181];
int32_t x3183 = x3176;
float x3184 = x1135[x3183];
float x3185 = x3179 + x3184;
x1101[x3178] = x3185;
float x3187 = x182[x3181];
float x3188 = x1078[x3178];
float x3189 = x176[x3181];
float x3190 = x1135[x3183];
float x3191 = x3187 + x3190;
x182[x3181] = x3191;
x3176 += 1;
x3174 += 1;
x3175 += 1;

}
x3169 += 26;
x3167 += 26;

}
for(int x3202=0; x3202 < 20; x3202++) {
int32_t x3204 = x3202 * 26;
int32_t x3208 = x3202 * 50;
for(int x3203=0; x3203 < 26; x3203++) {
int32_t x3205 = x3204 + x3203;
float x3206 = x1101[x3205];
int32_t x3211 = x3203 * 50;
for(int x3207=0; x3207 < 50; x3207++) {
int32_t x3209 = x3208 + x3207;
float x3210 = x1073[x3209];
int32_t x3212 = x3211 + x3207;
float x3213 = x163[x3212];
float x3214 = x3213 * x3206;
float x3215 = x3210 + x3214;
x1073[x3209] = x3215;
float x3217 = x171[x3212];
float x3218 = x1044[x3209];
float x3219 = x3218 * x3206;
float x3220 = x3217 + x3219;
x171[x3212] = x3220;

}

}

}
int32_t x3228 = 0;
int32_t x3229 = 0;
int32_t x3230 = 0;
for(int x3231=0; x3231 < 20; x3231++) {
int32_t x3232 = x3228;
int32_t x3233 = x3229;
int32_t x3234 = x3230;
int32_t x3235 = x3232;
int32_t x3236 = x3233;
int32_t x3237 = x3234;
for(int x3238=0; x3238 < 50; x3238++) {
int32_t x3239 = x3235;
float x3240 = x787[x3239];
float x3241 = x775[x3239];
int32_t x3242 = x3236;
float x3243 = x1030[x3242];
int32_t x3244 = x3237;
float x3245 = x1073[x3244];
float x3246 = x3245 * x3243;
float x3247 = x3240 + x3246;
x787[x3239] = x3247;
float x3249 = x1039[x3242];
float x3250 = x775[x3239];
float x3251 = x1030[x3242];
float x3252 = x1073[x3244];
float x3253 = x3252 * x3250;
float x3254 = x3249 + x3253;
x1039[x3242] = x3254;
x3237 += 1;
x3235 += 1;
x3236 += 1;

}
x3230 += 50;
x3228 += 50;
x3229 += 50;

}
for(int x3266=0; x3266 < 1000; x3266++) {
float x3267 = x1025[x3266];
float x3268 = x1030[x3266];
float x3271 = x1039[x3266];
float x3269 = x3268 * x3268;
float x3270 = 1.0f - x3269;
float x3272 = x3270 * x3271;
float x3273 = x3267 + x3272;
x1025[x3266] = x3273;

}
int32_t x3277 = 0;
int32_t x3278 = 0;
int32_t x3279 = 0;
for(int x3280=0; x3280 < 20; x3280++) {
int32_t x3281 = x3277;
int32_t x3282 = x3278;
int32_t x3283 = x3279;
int32_t x3284 = x3281;
int32_t x3285 = x3282;
int32_t x3286 = x3283;
for(int x3287=0; x3287 < 50; x3287++) {
int32_t x3288 = x3284;
float x3289 = x957[x3288];
float x3290 = x928[x3288];
int32_t x3291 = x3285;
float x3292 = x962[x3291];
int32_t x3293 = x3286;
float x3294 = x1025[x3293];
float x3295 = x3289 + x3294;
x957[x3288] = x3295;
float x3297 = x991[x3291];
float x3298 = x928[x3288];
float x3299 = x962[x3291];
float x3300 = x1025[x3293];
float x3301 = x3297 + x3300;
x991[x3291] = x3301;
x3286 += 1;
x3284 += 1;
x3285 += 1;

}
x3279 += 50;
x3277 += 50;
x3278 += 50;

}
int32_t x3313 = 0;
int32_t x3314 = 0;
int32_t x3315 = 0;
for(int x3316=0; x3316 < 20; x3316++) {
int32_t x3317 = x3313;
int32_t x3318 = x3314;
int32_t x3319 = x3315;
int32_t x3320 = x3317;
int32_t x3321 = x3318;
int32_t x3322 = x3319;
for(int x3323=0; x3323 < 50; x3323++) {
int32_t x3324 = x3320;
float x3325 = x648[x3324];
float x3326 = x636[x3324];
int32_t x3327 = x3321;
float x3328 = x914[x3327];
int32_t x3329 = x3322;
float x3330 = x991[x3329];
float x3331 = x3330 * x3328;
float x3332 = x3325 + x3331;
x648[x3324] = x3332;
float x3334 = x923[x3327];
float x3335 = x636[x3324];
float x3336 = x914[x3327];
float x3337 = x991[x3329];
float x3338 = x3337 * x3335;
float x3339 = x3334 + x3338;
x923[x3327] = x3339;
x3322 += 1;
x3320 += 1;
x3321 += 1;

}
x3315 += 50;
x3313 += 50;
x3314 += 50;

}
int32_t x3351 = 0;
int32_t x3352 = 0;
int32_t x3353 = 0;
for(int x3354=0; x3354 < 20; x3354++) {
int32_t x3355 = x3351;
int32_t x3356 = x3352;
int32_t x3357 = x3353;
int32_t x3358 = x3355;
int32_t x3359 = x3356;
int32_t x3360 = x3357;
for(int x3361=0; x3361 < 50; x3361++) {
int32_t x3362 = x3358;
float x3363 = x509[x3362];
float x3364 = x497[x3362];
int32_t x3365 = x3359;
float x3366 = x369[x3365];
int32_t x3367 = x3360;
float x3368 = x957[x3367];
float x3369 = x3368 * x3366;
float x3370 = x3363 + x3369;
x509[x3362] = x3370;
float x3372 = x370[x3365];
float x3373 = x497[x3362];
float x3374 = x369[x3365];
float x3375 = x957[x3367];
float x3376 = x3375 * x3373;
float x3377 = x3372 + x3376;
x370[x3365] = x3377;
x3360 += 1;
x3358 += 1;
x3359 += 1;

}
x3353 += 50;
x3351 += 50;
x3352 += 50;

}
for(int x3389=0; x3389 < 1000; x3389++) {
float x3390 = x909[x3389];
float x3391 = x914[x3389];
float x3394 = x923[x3389];
float x3392 = x3391 * x3391;
float x3393 = 1.0f - x3392;
float x3395 = x3393 * x3394;
float x3396 = x3390 + x3395;
x909[x3389] = x3396;

}
int32_t x3400 = 0;
int32_t x3401 = 0;
int32_t x3402 = 0;
for(int x3403=0; x3403 < 20; x3403++) {
int32_t x3404 = x3400;
int32_t x3405 = x3401;
int32_t x3406 = x3402;
int32_t x3407 = x3404;
int32_t x3408 = x3405;
int32_t x3409 = x3406;
for(int x3410=0; x3410 < 50; x3410++) {
int32_t x3411 = x3407;
float x3412 = x876[x3411];
float x3413 = x847[x3411];
int32_t x3414 = x3408;
float x3415 = x117[x3414];
int32_t x3416 = x3409;
float x3417 = x909[x3416];
float x3418 = x3412 + x3417;
x876[x3411] = x3418;
float x3420 = x122[x3414];
float x3421 = x847[x3411];
float x3422 = x117[x3414];
float x3423 = x909[x3416];
float x3424 = x3420 + x3423;
x122[x3414] = x3424;
x3409 += 1;
x3407 += 1;
x3408 += 1;

}
x3402 += 50;
x3400 += 50;

}
int32_t x3435 = 0;
int32_t x3436 = 0;
int32_t x3437 = 0;
for(int x3438=0; x3438 < 20; x3438++) {
int32_t x3439 = x3435;
int32_t x3440 = x3436;
int32_t x3441 = x3437;
int32_t x3442 = x3439;
int32_t x3443 = x3440;
int32_t x3444 = x3441;
for(int x3445=0; x3445 < 50; x3445++) {
int32_t x3446 = x3442;
float x3447 = x815[x3446];
float x3448 = x792[x3446];
int32_t x3449 = x3443;
float x3450 = x820[x3449];
int32_t x3451 = x3444;
float x3452 = x876[x3451];
float x3453 = x3447 + x3452;
x815[x3446] = x3453;
float x3455 = x842[x3449];
float x3456 = x792[x3446];
float x3457 = x820[x3449];
float x3458 = x876[x3451];
float x3459 = x3455 + x3458;
x842[x3449] = x3459;
x3444 += 1;
x3442 += 1;
x3443 += 1;

}
x3437 += 50;
x3435 += 50;
x3436 += 50;

}
for(int x3471=0; x3471 < 20; x3471++) {
int32_t x3473 = x3471 * 50;
for(int x3472=0; x3472 < 50; x3472++) {
int32_t x3474 = x3473 + x3472;
float x3475 = x842[x3474];
int32_t x3479 = x3472 * 50;
for(int x3476=0; x3476 < 50; x3476++) {
int32_t x3477 = x3473 + x3476;
float x3478 = x368[x3477];
int32_t x3480 = x3479 + x3476;
float x3481 = x104[x3480];
float x3482 = x3481 * x3475;
float x3483 = x3478 + x3482;
x368[x3477] = x3483;
float x3485 = x112[x3480];
float x3486 = x367[x3477];
float x3487 = x3486 * x3475;
float x3488 = x3485 + x3487;
x112[x3480] = x3488;

}

}

}
for(int x3496=0; x3496 < 20; x3496++) {
int32_t x3498 = x3496 * 50;
int32_t x3502 = x3496 * 26;
for(int x3497=0; x3497 < 50; x3497++) {
int32_t x3499 = x3498 + x3497;
float x3500 = x815[x3499];
int32_t x3505 = x3497 * 26;
for(int x3501=0; x3501 < 26; x3501++) {
int32_t x3503 = x3502 + x3501;
float x3504 = x374[x3503];
int32_t x3506 = x3505 + x3501;
float x3507 = x91[x3506];
float x3508 = x3507 * x3500;
float x3509 = x3504 + x3508;
x374[x3503] = x3509;
float x3511 = x99[x3506];
float x3512 = x373[x3503];
float x3513 = x3512 * x3500;
float x3514 = x3511 + x3513;
x99[x3506] = x3514;

}

}

}
for(int x3522=0; x3522 < 1000; x3522++) {
float x3523 = x770[x3522];
float x3524 = x775[x3522];
float x3527 = x787[x3522];
float x3525 = 1.0f - x3524;
float x3526 = x3525 * x3524;
float x3528 = x3526 * x3527;
float x3529 = x3523 + x3528;
x770[x3522] = x3529;

}
int32_t x3533 = 0;
int32_t x3534 = 0;
int32_t x3535 = 0;
for(int x3536=0; x3536 < 20; x3536++) {
int32_t x3537 = x3533;
int32_t x3538 = x3534;
int32_t x3539 = x3535;
int32_t x3540 = x3537;
int32_t x3541 = x3538;
int32_t x3542 = x3539;
for(int x3543=0; x3543 < 50; x3543++) {
int32_t x3544 = x3540;
float x3545 = x737[x3544];
float x3546 = x708[x3544];
int32_t x3547 = x3541;
float x3548 = x153[x3547];
int32_t x3549 = x3542;
float x3550 = x770[x3549];
float x3551 = x3545 + x3550;
x737[x3544] = x3551;
float x3553 = x158[x3547];
float x3554 = x708[x3544];
float x3555 = x153[x3547];
float x3556 = x770[x3549];
float x3557 = x3553 + x3556;
x158[x3547] = x3557;
x3542 += 1;
x3540 += 1;
x3541 += 1;

}
x3535 += 50;
x3533 += 50;

}
int32_t x3568 = 0;
int32_t x3569 = 0;
int32_t x3570 = 0;
for(int x3571=0; x3571 < 20; x3571++) {
int32_t x3572 = x3568;
int32_t x3573 = x3569;
int32_t x3574 = x3570;
int32_t x3575 = x3572;
int32_t x3576 = x3573;
int32_t x3577 = x3574;
for(int x3578=0; x3578 < 50; x3578++) {
int32_t x3579 = x3575;
float x3580 = x676[x3579];
float x3581 = x653[x3579];
int32_t x3582 = x3576;
float x3583 = x681[x3582];
int32_t x3584 = x3577;
float x3585 = x737[x3584];
float x3586 = x3580 + x3585;
x676[x3579] = x3586;
float x3588 = x703[x3582];
float x3589 = x653[x3579];
float x3590 = x681[x3582];
float x3591 = x737[x3584];
float x3592 = x3588 + x3591;
x703[x3582] = x3592;
x3577 += 1;
x3575 += 1;
x3576 += 1;

}
x3570 += 50;
x3568 += 50;
x3569 += 50;

}
for(int x3604=0; x3604 < 20; x3604++) {
int32_t x3606 = x3604 * 50;
for(int x3605=0; x3605 < 50; x3605++) {
int32_t x3607 = x3606 + x3605;
float x3608 = x703[x3607];
int32_t x3612 = x3605 * 50;
for(int x3609=0; x3609 < 50; x3609++) {
int32_t x3610 = x3606 + x3609;
float x3611 = x368[x3610];
int32_t x3613 = x3612 + x3609;
float x3614 = x140[x3613];
float x3615 = x3614 * x3608;
float x3616 = x3611 + x3615;
x368[x3610] = x3616;
float x3618 = x148[x3613];
float x3619 = x367[x3610];
float x3620 = x3619 * x3608;
float x3621 = x3618 + x3620;
x148[x3613] = x3621;

}

}

}
for(int x3629=0; x3629 < 20; x3629++) {
int32_t x3631 = x3629 * 50;
int32_t x3635 = x3629 * 26;
for(int x3630=0; x3630 < 50; x3630++) {
int32_t x3632 = x3631 + x3630;
float x3633 = x676[x3632];
int32_t x3638 = x3630 * 26;
for(int x3634=0; x3634 < 26; x3634++) {
int32_t x3636 = x3635 + x3634;
float x3637 = x374[x3636];
int32_t x3639 = x3638 + x3634;
float x3640 = x127[x3639];
float x3641 = x3640 * x3633;
float x3642 = x3637 + x3641;
x374[x3636] = x3642;
float x3644 = x135[x3639];
float x3645 = x373[x3636];
float x3646 = x3645 * x3633;
float x3647 = x3644 + x3646;
x135[x3639] = x3647;

}

}

}
for(int x3655=0; x3655 < 1000; x3655++) {
float x3656 = x631[x3655];
float x3657 = x636[x3655];
float x3660 = x648[x3655];
float x3658 = 1.0f - x3657;
float x3659 = x3658 * x3657;
float x3661 = x3659 * x3660;
float x3662 = x3656 + x3661;
x631[x3655] = x3662;

}
int32_t x3666 = 0;
int32_t x3667 = 0;
int32_t x3668 = 0;
for(int x3669=0; x3669 < 20; x3669++) {
int32_t x3670 = x3666;
int32_t x3671 = x3667;
int32_t x3672 = x3668;
int32_t x3673 = x3670;
int32_t x3674 = x3671;
int32_t x3675 = x3672;
for(int x3676=0; x3676 < 50; x3676++) {
int32_t x3677 = x3673;
float x3678 = x598[x3677];
float x3679 = x569[x3677];
int32_t x3680 = x3674;
float x3681 = x81[x3680];
int32_t x3682 = x3675;
float x3683 = x631[x3682];
float x3684 = x3678 + x3683;
x598[x3677] = x3684;
float x3686 = x86[x3680];
float x3687 = x569[x3677];
float x3688 = x81[x3680];
float x3689 = x631[x3682];
float x3690 = x3686 + x3689;
x86[x3680] = x3690;
x3675 += 1;
x3673 += 1;
x3674 += 1;

}
x3668 += 50;
x3666 += 50;

}
int32_t x3701 = 0;
int32_t x3702 = 0;
int32_t x3703 = 0;
for(int x3704=0; x3704 < 20; x3704++) {
int32_t x3705 = x3701;
int32_t x3706 = x3702;
int32_t x3707 = x3703;
int32_t x3708 = x3705;
int32_t x3709 = x3706;
int32_t x3710 = x3707;
for(int x3711=0; x3711 < 50; x3711++) {
int32_t x3712 = x3708;
float x3713 = x537[x3712];
float x3714 = x514[x3712];
int32_t x3715 = x3709;
float x3716 = x542[x3715];
int32_t x3717 = x3710;
float x3718 = x598[x3717];
float x3719 = x3713 + x3718;
x537[x3712] = x3719;
float x3721 = x564[x3715];
float x3722 = x514[x3712];
float x3723 = x542[x3715];
float x3724 = x598[x3717];
float x3725 = x3721 + x3724;
x564[x3715] = x3725;
x3710 += 1;
x3708 += 1;
x3709 += 1;

}
x3703 += 50;
x3701 += 50;
x3702 += 50;

}
for(int x3737=0; x3737 < 20; x3737++) {
int32_t x3739 = x3737 * 50;
for(int x3738=0; x3738 < 50; x3738++) {
int32_t x3740 = x3739 + x3738;
float x3741 = x564[x3740];
int32_t x3745 = x3738 * 50;
for(int x3742=0; x3742 < 50; x3742++) {
int32_t x3743 = x3739 + x3742;
float x3744 = x368[x3743];
int32_t x3746 = x3745 + x3742;
float x3747 = x68[x3746];
float x3748 = x3747 * x3741;
float x3749 = x3744 + x3748;
x368[x3743] = x3749;
float x3751 = x76[x3746];
float x3752 = x367[x3743];
float x3753 = x3752 * x3741;
float x3754 = x3751 + x3753;
x76[x3746] = x3754;

}

}

}
for(int x3762=0; x3762 < 20; x3762++) {
int32_t x3764 = x3762 * 50;
int32_t x3768 = x3762 * 26;
for(int x3763=0; x3763 < 50; x3763++) {
int32_t x3765 = x3764 + x3763;
float x3766 = x537[x3765];
int32_t x3771 = x3763 * 26;
for(int x3767=0; x3767 < 26; x3767++) {
int32_t x3769 = x3768 + x3767;
float x3770 = x374[x3769];
int32_t x3772 = x3771 + x3767;
float x3773 = x55[x3772];
float x3774 = x3773 * x3766;
float x3775 = x3770 + x3774;
x374[x3769] = x3775;
float x3777 = x63[x3772];
float x3778 = x373[x3769];
float x3779 = x3778 * x3766;
float x3780 = x3777 + x3779;
x63[x3772] = x3780;

}

}

}
for(int x3788=0; x3788 < 1000; x3788++) {
float x3789 = x492[x3788];
float x3790 = x497[x3788];
float x3793 = x509[x3788];
float x3791 = 1.0f - x3790;
float x3792 = x3791 * x3790;
float x3794 = x3792 * x3793;
float x3795 = x3789 + x3794;
x492[x3788] = x3795;

}
int32_t x3799 = 0;
int32_t x3800 = 0;
int32_t x3801 = 0;
for(int x3802=0; x3802 < 20; x3802++) {
int32_t x3803 = x3799;
int32_t x3804 = x3800;
int32_t x3805 = x3801;
int32_t x3806 = x3803;
int32_t x3807 = x3804;
int32_t x3808 = x3805;
for(int x3809=0; x3809 < 50; x3809++) {
int32_t x3810 = x3806;
float x3811 = x459[x3810];
float x3812 = x430[x3810];
int32_t x3813 = x3807;
float x3814 = x44[x3813];
int32_t x3815 = x3808;
float x3816 = x492[x3815];
float x3817 = x3811 + x3816;
x459[x3810] = x3817;
float x3819 = x50[x3813];
float x3820 = x430[x3810];
float x3821 = x44[x3813];
float x3822 = x492[x3815];
float x3823 = x3819 + x3822;
x50[x3813] = x3823;
x3808 += 1;
x3806 += 1;
x3807 += 1;

}
x3801 += 50;
x3799 += 50;

}
int32_t x3834 = 0;
int32_t x3835 = 0;
int32_t x3836 = 0;
for(int x3837=0; x3837 < 20; x3837++) {
int32_t x3838 = x3834;
int32_t x3839 = x3835;
int32_t x3840 = x3836;
int32_t x3841 = x3838;
int32_t x3842 = x3839;
int32_t x3843 = x3840;
for(int x3844=0; x3844 < 50; x3844++) {
int32_t x3845 = x3841;
float x3846 = x398[x3845];
float x3847 = x375[x3845];
int32_t x3848 = x3842;
float x3849 = x403[x3848];
int32_t x3850 = x3843;
float x3851 = x459[x3850];
float x3852 = x3846 + x3851;
x398[x3845] = x3852;
float x3854 = x425[x3848];
float x3855 = x375[x3845];
float x3856 = x403[x3848];
float x3857 = x459[x3850];
float x3858 = x3854 + x3857;
x425[x3848] = x3858;
x3843 += 1;
x3841 += 1;
x3842 += 1;

}
x3836 += 50;
x3834 += 50;
x3835 += 50;

}
for(int x3870=0; x3870 < 20; x3870++) {
int32_t x3872 = x3870 * 50;
for(int x3871=0; x3871 < 50; x3871++) {
int32_t x3873 = x3872 + x3871;
float x3874 = x425[x3873];
int32_t x3878 = x3871 * 50;
for(int x3875=0; x3875 < 50; x3875++) {
int32_t x3876 = x3872 + x3875;
float x3877 = x368[x3876];
int32_t x3879 = x3878 + x3875;
float x3880 = x30[x3879];
float x3881 = x3880 * x3874;
float x3882 = x3877 + x3881;
x368[x3876] = x3882;
float x3884 = x39[x3879];
float x3885 = x367[x3876];
float x3886 = x3885 * x3874;
float x3887 = x3884 + x3886;
x39[x3879] = x3887;

}

}

}
for(int x3895=0; x3895 < 20; x3895++) {
int32_t x3897 = x3895 * 50;
int32_t x3901 = x3895 * 26;
for(int x3896=0; x3896 < 50; x3896++) {
int32_t x3898 = x3897 + x3896;
float x3899 = x398[x3898];
int32_t x3904 = x3896 * 26;
for(int x3900=0; x3900 < 26; x3900++) {
int32_t x3902 = x3901 + x3900;
float x3903 = x374[x3902];
int32_t x3905 = x3904 + x3900;
float x3906 = x16[x3905];
float x3907 = x3906 * x3899;
float x3908 = x3903 + x3907;
x374[x3902] = x3908;
float x3910 = x25[x3905];
float x3911 = x373[x3902];
float x3912 = x3911 * x3899;
float x3913 = x3910 + x3912;
x25[x3905] = x3913;

}

}

}
} else {
float x3922 = 0.0f;
float x3923 = x3922;
float x3924 = x365[0];
float x3925 = x3923 + x3924;
x3922 = x3925;
float x3927 = x3922;
float* x3928 = (float*)myMalloc(1 * sizeof(float));;
x3928[0] = x3927;
float* x3930 = (float*)myMalloc(1 * sizeof(float));;
for(int x3931=0; x3931 < 1; x3931++) {
x3930[x3931] = 0.0f;

}
float x3935 = x3930[0];
x3930[0] = 1.0f;
float x3937 = x3928[0];
x297[0] = x3937;
// += tensor of dim 0
float x3940 = x3930[0];
float x3941 = x366[0];
float x3942 = x3941 + x3940;
x366[0] = x3942;
}
};
x360(0,x3947);
float x3956 = x297[0];
int32_t x3957 = x264 % 100;
bool x3958 = x3957 == 0;
if (x3958) {
printf("iter %d, loss %f\n",x264,x3956);
int32_t x3960 = x264 / 100;
double x3961 = (double)x3956;
x258[x3960] = x3961;
} else {
}
for(int x3965=0; x3965 < 1300; x3965++) {
float x3966 = x63[x3965];
bool x3967 = x3966 > 5.0f;
if (x3967) {
x63[x3965] = 5.0f;
} else {
}
float x3971 = x63[x3965];
bool x3972 = x3971 < -5.0f;
if (x3972) {
x63[x3965] = -5.0f;
} else {
}

}
float* x3978 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x3979 = 0;
int32_t x3980 = 0;
int32_t x3981 = 0;
for(int x3982=0; x3982 < 50; x3982++) {
int32_t x3983 = x3980;
int32_t x3984 = x3981;
int32_t x3985 = x3979;
int32_t x3986 = x3985;
int32_t x3987 = x3983;
int32_t x3988 = x3984;
for(int x3989=0; x3989 < 26; x3989++) {
int32_t x3990 = x3986;
int32_t x3991 = x3987;
float x3992 = x63[x3991];
int32_t x3993 = x3988;
float x3994 = x63[x3993];
float x3995 = x3992 * x3994;
x3978[x3990] = x3995;
x3986 += 1;
x3987 += 1;
x3988 += 1;

}
x3979 += 26;
x3980 += 26;
x3981 += 26;

}
for(int x4007=0; x4007 < 1300; x4007++) {
float x4008 = x187[x4007];
float x4009 = x3978[x4007];
float x4010 = x4008 + x4009;
x187[x4007] = x4010;

}
float* x4014 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4015=0; x4015 < 1300; x4015++) {
float x4016 = x63[x4015];
float x4017 = x4016 * 0.1f;
x4014[x4015] = x4017;

}
float* x4021 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4022=0; x4022 < 1300; x4022++) {
float x4023 = x187[x4022];
float x4024 = x4023 + 1.0E-8f;
x4021[x4022] = x4024;

}
float* x4028 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4029=0; x4029 < 1300; x4029++) {
float x4030 = x4021[x4029];
double x4031 = (double)x4030;
double x4032 = sqrt(x4031);
float x4033 = (float)x4032;
x4028[x4029] = x4033;

}
float* x4037 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4038 = 0;
int32_t x4039 = 0;
int32_t x4040 = 0;
for(int x4041=0; x4041 < 50; x4041++) {
int32_t x4042 = x4039;
int32_t x4043 = x4040;
int32_t x4044 = x4038;
int32_t x4045 = x4044;
int32_t x4046 = x4042;
int32_t x4047 = x4043;
for(int x4048=0; x4048 < 26; x4048++) {
int32_t x4049 = x4045;
int32_t x4050 = x4046;
float x4051 = x4014[x4050];
int32_t x4052 = x4047;
float x4053 = x4028[x4052];
float x4054 = x4051 / x4053;
x4037[x4049] = x4054;
x4045 += 1;
x4046 += 1;
x4047 += 1;

}
x4038 += 26;
x4039 += 26;
x4040 += 26;

}
for(int x4066=0; x4066 < 1300; x4066++) {
float x4067 = x55[x4066];
float x4068 = x4037[x4066];
float x4069 = x4067 - x4068;
x55[x4066] = x4069;

}
for(int x4073=0; x4073 < 1300; x4073++) {
float x4074 = x63[x4073];
x63[x4073] = 0.0f;

}
for(int x4078=0; x4078 < 50; x4078++) {
float x4079 = x86[x4078];
bool x4080 = x4079 > 5.0f;
if (x4080) {
x86[x4078] = 5.0f;
} else {
}
float x4084 = x86[x4078];
bool x4085 = x4084 < -5.0f;
if (x4085) {
x86[x4078] = -5.0f;
} else {
}

}
float* x4091 = (float*)myMalloc(50 * sizeof(float));;
int32_t x4092 = 0;
int32_t x4093 = 0;
int32_t x4094 = 0;
for(int x4095=0; x4095 < 50; x4095++) {
int32_t x4096 = x4092;
int32_t x4097 = x4093;
float x4098 = x86[x4097];
int32_t x4099 = x4094;
float x4100 = x86[x4099];
float x4101 = x4098 * x4100;
x4091[x4096] = x4101;
x4092 += 1;
x4093 += 1;
x4094 += 1;

}
for(int x4108=0; x4108 < 50; x4108++) {
float x4109 = x192[x4108];
float x4110 = x4091[x4108];
float x4111 = x4109 + x4110;
x192[x4108] = x4111;

}
float* x4115 = (float*)myMalloc(50 * sizeof(float));;
for(int x4116=0; x4116 < 50; x4116++) {
float x4117 = x86[x4116];
float x4118 = x4117 * 0.1f;
x4115[x4116] = x4118;

}
float* x4122 = (float*)myMalloc(50 * sizeof(float));;
for(int x4123=0; x4123 < 50; x4123++) {
float x4124 = x192[x4123];
float x4125 = x4124 + 1.0E-8f;
x4122[x4123] = x4125;

}
float* x4129 = (float*)myMalloc(50 * sizeof(float));;
for(int x4130=0; x4130 < 50; x4130++) {
float x4131 = x4122[x4130];
double x4132 = (double)x4131;
double x4133 = sqrt(x4132);
float x4134 = (float)x4133;
x4129[x4130] = x4134;

}
float* x4138 = (float*)myMalloc(50 * sizeof(float));;
int32_t x4139 = 0;
int32_t x4140 = 0;
int32_t x4141 = 0;
for(int x4142=0; x4142 < 50; x4142++) {
int32_t x4143 = x4139;
int32_t x4144 = x4140;
float x4145 = x4115[x4144];
int32_t x4146 = x4141;
float x4147 = x4129[x4146];
float x4148 = x4145 / x4147;
x4138[x4143] = x4148;
x4139 += 1;
x4140 += 1;
x4141 += 1;

}
for(int x4155=0; x4155 < 50; x4155++) {
float x4156 = x81[x4155];
float x4157 = x4138[x4155];
float x4158 = x4156 - x4157;
x81[x4155] = x4158;

}
for(int x4162=0; x4162 < 50; x4162++) {
float x4163 = x86[x4162];
x86[x4162] = 0.0f;

}
for(int x4167=0; x4167 < 2500; x4167++) {
float x4168 = x76[x4167];
bool x4169 = x4168 > 5.0f;
if (x4169) {
x76[x4167] = 5.0f;
} else {
}
float x4173 = x76[x4167];
bool x4174 = x4173 < -5.0f;
if (x4174) {
x76[x4167] = -5.0f;
} else {
}

}
float* x4180 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x4181 = 0;
int32_t x4182 = 0;
int32_t x4183 = 0;
for(int x4184=0; x4184 < 50; x4184++) {
int32_t x4185 = x4182;
int32_t x4186 = x4183;
int32_t x4187 = x4181;
int32_t x4188 = x4187;
int32_t x4189 = x4185;
int32_t x4190 = x4186;
for(int x4191=0; x4191 < 50; x4191++) {
int32_t x4192 = x4188;
int32_t x4193 = x4189;
float x4194 = x76[x4193];
int32_t x4195 = x4190;
float x4196 = x76[x4195];
float x4197 = x4194 * x4196;
x4180[x4192] = x4197;
x4188 += 1;
x4189 += 1;
x4190 += 1;

}
x4181 += 50;
x4182 += 50;
x4183 += 50;

}
for(int x4209=0; x4209 < 2500; x4209++) {
float x4210 = x197[x4209];
float x4211 = x4180[x4209];
float x4212 = x4210 + x4211;
x197[x4209] = x4212;

}
float* x4216 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4217=0; x4217 < 2500; x4217++) {
float x4218 = x76[x4217];
float x4219 = x4218 * 0.1f;
x4216[x4217] = x4219;

}
float* x4223 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4224=0; x4224 < 2500; x4224++) {
float x4225 = x197[x4224];
float x4226 = x4225 + 1.0E-8f;
x4223[x4224] = x4226;

}
float* x4230 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4231=0; x4231 < 2500; x4231++) {
float x4232 = x4223[x4231];
double x4233 = (double)x4232;
double x4234 = sqrt(x4233);
float x4235 = (float)x4234;
x4230[x4231] = x4235;

}
float* x4239 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x4240 = 0;
int32_t x4241 = 0;
int32_t x4242 = 0;
for(int x4243=0; x4243 < 50; x4243++) {
int32_t x4244 = x4241;
int32_t x4245 = x4242;
int32_t x4246 = x4240;
int32_t x4247 = x4246;
int32_t x4248 = x4244;
int32_t x4249 = x4245;
for(int x4250=0; x4250 < 50; x4250++) {
int32_t x4251 = x4247;
int32_t x4252 = x4248;
float x4253 = x4216[x4252];
int32_t x4254 = x4249;
float x4255 = x4230[x4254];
float x4256 = x4253 / x4255;
x4239[x4251] = x4256;
x4247 += 1;
x4248 += 1;
x4249 += 1;

}
x4240 += 50;
x4241 += 50;
x4242 += 50;

}
for(int x4268=0; x4268 < 2500; x4268++) {
float x4269 = x68[x4268];
float x4270 = x4239[x4268];
float x4271 = x4269 - x4270;
x68[x4268] = x4271;

}
for(int x4275=0; x4275 < 2500; x4275++) {
float x4276 = x76[x4275];
x76[x4275] = 0.0f;

}
for(int x4280=0; x4280 < 50; x4280++) {
float x4281 = x50[x4280];
bool x4282 = x4281 > 5.0f;
if (x4282) {
x50[x4280] = 5.0f;
} else {
}
float x4286 = x50[x4280];
bool x4287 = x4286 < -5.0f;
if (x4287) {
x50[x4280] = -5.0f;
} else {
}

}
float* x4293 = (float*)myMalloc(50 * sizeof(float));;
int32_t x4294 = 0;
int32_t x4295 = 0;
int32_t x4296 = 0;
for(int x4297=0; x4297 < 50; x4297++) {
int32_t x4298 = x4294;
int32_t x4299 = x4295;
float x4300 = x50[x4299];
int32_t x4301 = x4296;
float x4302 = x50[x4301];
float x4303 = x4300 * x4302;
x4293[x4298] = x4303;
x4294 += 1;
x4295 += 1;
x4296 += 1;

}
for(int x4310=0; x4310 < 50; x4310++) {
float x4311 = x202[x4310];
float x4312 = x4293[x4310];
float x4313 = x4311 + x4312;
x202[x4310] = x4313;

}
float* x4317 = (float*)myMalloc(50 * sizeof(float));;
for(int x4318=0; x4318 < 50; x4318++) {
float x4319 = x50[x4318];
float x4320 = x4319 * 0.1f;
x4317[x4318] = x4320;

}
float* x4324 = (float*)myMalloc(50 * sizeof(float));;
for(int x4325=0; x4325 < 50; x4325++) {
float x4326 = x202[x4325];
float x4327 = x4326 + 1.0E-8f;
x4324[x4325] = x4327;

}
float* x4331 = (float*)myMalloc(50 * sizeof(float));;
for(int x4332=0; x4332 < 50; x4332++) {
float x4333 = x4324[x4332];
double x4334 = (double)x4333;
double x4335 = sqrt(x4334);
float x4336 = (float)x4335;
x4331[x4332] = x4336;

}
float* x4340 = (float*)myMalloc(50 * sizeof(float));;
int32_t x4341 = 0;
int32_t x4342 = 0;
int32_t x4343 = 0;
for(int x4344=0; x4344 < 50; x4344++) {
int32_t x4345 = x4341;
int32_t x4346 = x4342;
float x4347 = x4317[x4346];
int32_t x4348 = x4343;
float x4349 = x4331[x4348];
float x4350 = x4347 / x4349;
x4340[x4345] = x4350;
x4341 += 1;
x4342 += 1;
x4343 += 1;

}
for(int x4357=0; x4357 < 50; x4357++) {
float x4358 = x44[x4357];
float x4359 = x4340[x4357];
float x4360 = x4358 - x4359;
x44[x4357] = x4360;

}
for(int x4364=0; x4364 < 50; x4364++) {
float x4365 = x50[x4364];
x50[x4364] = 0.0f;

}
for(int x4369=0; x4369 < 2500; x4369++) {
float x4370 = x39[x4369];
bool x4371 = x4370 > 5.0f;
if (x4371) {
x39[x4369] = 5.0f;
} else {
}
float x4375 = x39[x4369];
bool x4376 = x4375 < -5.0f;
if (x4376) {
x39[x4369] = -5.0f;
} else {
}

}
float* x4382 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x4383 = 0;
int32_t x4384 = 0;
int32_t x4385 = 0;
for(int x4386=0; x4386 < 50; x4386++) {
int32_t x4387 = x4384;
int32_t x4388 = x4385;
int32_t x4389 = x4383;
int32_t x4390 = x4389;
int32_t x4391 = x4387;
int32_t x4392 = x4388;
for(int x4393=0; x4393 < 50; x4393++) {
int32_t x4394 = x4390;
int32_t x4395 = x4391;
float x4396 = x39[x4395];
int32_t x4397 = x4392;
float x4398 = x39[x4397];
float x4399 = x4396 * x4398;
x4382[x4394] = x4399;
x4390 += 1;
x4391 += 1;
x4392 += 1;

}
x4383 += 50;
x4384 += 50;
x4385 += 50;

}
for(int x4411=0; x4411 < 2500; x4411++) {
float x4412 = x207[x4411];
float x4413 = x4382[x4411];
float x4414 = x4412 + x4413;
x207[x4411] = x4414;

}
float* x4418 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4419=0; x4419 < 2500; x4419++) {
float x4420 = x39[x4419];
float x4421 = x4420 * 0.1f;
x4418[x4419] = x4421;

}
float* x4425 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4426=0; x4426 < 2500; x4426++) {
float x4427 = x207[x4426];
float x4428 = x4427 + 1.0E-8f;
x4425[x4426] = x4428;

}
float* x4432 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4433=0; x4433 < 2500; x4433++) {
float x4434 = x4425[x4433];
double x4435 = (double)x4434;
double x4436 = sqrt(x4435);
float x4437 = (float)x4436;
x4432[x4433] = x4437;

}
float* x4441 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x4442 = 0;
int32_t x4443 = 0;
int32_t x4444 = 0;
for(int x4445=0; x4445 < 50; x4445++) {
int32_t x4446 = x4443;
int32_t x4447 = x4444;
int32_t x4448 = x4442;
int32_t x4449 = x4448;
int32_t x4450 = x4446;
int32_t x4451 = x4447;
for(int x4452=0; x4452 < 50; x4452++) {
int32_t x4453 = x4449;
int32_t x4454 = x4450;
float x4455 = x4418[x4454];
int32_t x4456 = x4451;
float x4457 = x4432[x4456];
float x4458 = x4455 / x4457;
x4441[x4453] = x4458;
x4449 += 1;
x4450 += 1;
x4451 += 1;

}
x4442 += 50;
x4443 += 50;
x4444 += 50;

}
for(int x4470=0; x4470 < 2500; x4470++) {
float x4471 = x30[x4470];
float x4472 = x4441[x4470];
float x4473 = x4471 - x4472;
x30[x4470] = x4473;

}
for(int x4477=0; x4477 < 2500; x4477++) {
float x4478 = x39[x4477];
x39[x4477] = 0.0f;

}
for(int x4482=0; x4482 < 1300; x4482++) {
float x4483 = x25[x4482];
bool x4484 = x4483 > 5.0f;
if (x4484) {
x25[x4482] = 5.0f;
} else {
}
float x4488 = x25[x4482];
bool x4489 = x4488 < -5.0f;
if (x4489) {
x25[x4482] = -5.0f;
} else {
}

}
float* x4495 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4496 = 0;
int32_t x4497 = 0;
int32_t x4498 = 0;
for(int x4499=0; x4499 < 50; x4499++) {
int32_t x4500 = x4497;
int32_t x4501 = x4498;
int32_t x4502 = x4496;
int32_t x4503 = x4502;
int32_t x4504 = x4500;
int32_t x4505 = x4501;
for(int x4506=0; x4506 < 26; x4506++) {
int32_t x4507 = x4503;
int32_t x4508 = x4504;
float x4509 = x25[x4508];
int32_t x4510 = x4505;
float x4511 = x25[x4510];
float x4512 = x4509 * x4511;
x4495[x4507] = x4512;
x4503 += 1;
x4504 += 1;
x4505 += 1;

}
x4496 += 26;
x4497 += 26;
x4498 += 26;

}
for(int x4524=0; x4524 < 1300; x4524++) {
float x4525 = x212[x4524];
float x4526 = x4495[x4524];
float x4527 = x4525 + x4526;
x212[x4524] = x4527;

}
float* x4531 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4532=0; x4532 < 1300; x4532++) {
float x4533 = x25[x4532];
float x4534 = x4533 * 0.1f;
x4531[x4532] = x4534;

}
float* x4538 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4539=0; x4539 < 1300; x4539++) {
float x4540 = x212[x4539];
float x4541 = x4540 + 1.0E-8f;
x4538[x4539] = x4541;

}
float* x4545 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4546=0; x4546 < 1300; x4546++) {
float x4547 = x4538[x4546];
double x4548 = (double)x4547;
double x4549 = sqrt(x4548);
float x4550 = (float)x4549;
x4545[x4546] = x4550;

}
float* x4554 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4555 = 0;
int32_t x4556 = 0;
int32_t x4557 = 0;
for(int x4558=0; x4558 < 50; x4558++) {
int32_t x4559 = x4556;
int32_t x4560 = x4557;
int32_t x4561 = x4555;
int32_t x4562 = x4561;
int32_t x4563 = x4559;
int32_t x4564 = x4560;
for(int x4565=0; x4565 < 26; x4565++) {
int32_t x4566 = x4562;
int32_t x4567 = x4563;
float x4568 = x4531[x4567];
int32_t x4569 = x4564;
float x4570 = x4545[x4569];
float x4571 = x4568 / x4570;
x4554[x4566] = x4571;
x4562 += 1;
x4563 += 1;
x4564 += 1;

}
x4555 += 26;
x4556 += 26;
x4557 += 26;

}
for(int x4583=0; x4583 < 1300; x4583++) {
float x4584 = x16[x4583];
float x4585 = x4554[x4583];
float x4586 = x4584 - x4585;
x16[x4583] = x4586;

}
for(int x4590=0; x4590 < 1300; x4590++) {
float x4591 = x25[x4590];
x25[x4590] = 0.0f;

}
for(int x4595=0; x4595 < 1300; x4595++) {
float x4596 = x99[x4595];
bool x4597 = x4596 > 5.0f;
if (x4597) {
x99[x4595] = 5.0f;
} else {
}
float x4601 = x99[x4595];
bool x4602 = x4601 < -5.0f;
if (x4602) {
x99[x4595] = -5.0f;
} else {
}

}
float* x4608 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4609 = 0;
int32_t x4610 = 0;
int32_t x4611 = 0;
for(int x4612=0; x4612 < 50; x4612++) {
int32_t x4613 = x4610;
int32_t x4614 = x4611;
int32_t x4615 = x4609;
int32_t x4616 = x4615;
int32_t x4617 = x4613;
int32_t x4618 = x4614;
for(int x4619=0; x4619 < 26; x4619++) {
int32_t x4620 = x4616;
int32_t x4621 = x4617;
float x4622 = x99[x4621];
int32_t x4623 = x4618;
float x4624 = x99[x4623];
float x4625 = x4622 * x4624;
x4608[x4620] = x4625;
x4616 += 1;
x4617 += 1;
x4618 += 1;

}
x4609 += 26;
x4610 += 26;
x4611 += 26;

}
for(int x4637=0; x4637 < 1300; x4637++) {
float x4638 = x217[x4637];
float x4639 = x4608[x4637];
float x4640 = x4638 + x4639;
x217[x4637] = x4640;

}
float* x4644 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4645=0; x4645 < 1300; x4645++) {
float x4646 = x99[x4645];
float x4647 = x4646 * 0.1f;
x4644[x4645] = x4647;

}
float* x4651 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4652=0; x4652 < 1300; x4652++) {
float x4653 = x217[x4652];
float x4654 = x4653 + 1.0E-8f;
x4651[x4652] = x4654;

}
float* x4658 = (float*)myMalloc(1300 * sizeof(float));;
for(int x4659=0; x4659 < 1300; x4659++) {
float x4660 = x4651[x4659];
double x4661 = (double)x4660;
double x4662 = sqrt(x4661);
float x4663 = (float)x4662;
x4658[x4659] = x4663;

}
float* x4667 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x4668 = 0;
int32_t x4669 = 0;
int32_t x4670 = 0;
for(int x4671=0; x4671 < 50; x4671++) {
int32_t x4672 = x4669;
int32_t x4673 = x4670;
int32_t x4674 = x4668;
int32_t x4675 = x4674;
int32_t x4676 = x4672;
int32_t x4677 = x4673;
for(int x4678=0; x4678 < 26; x4678++) {
int32_t x4679 = x4675;
int32_t x4680 = x4676;
float x4681 = x4644[x4680];
int32_t x4682 = x4677;
float x4683 = x4658[x4682];
float x4684 = x4681 / x4683;
x4667[x4679] = x4684;
x4675 += 1;
x4676 += 1;
x4677 += 1;

}
x4668 += 26;
x4669 += 26;
x4670 += 26;

}
for(int x4696=0; x4696 < 1300; x4696++) {
float x4697 = x91[x4696];
float x4698 = x4667[x4696];
float x4699 = x4697 - x4698;
x91[x4696] = x4699;

}
for(int x4703=0; x4703 < 1300; x4703++) {
float x4704 = x99[x4703];
x99[x4703] = 0.0f;

}
for(int x4708=0; x4708 < 50; x4708++) {
float x4709 = x122[x4708];
bool x4710 = x4709 > 5.0f;
if (x4710) {
x122[x4708] = 5.0f;
} else {
}
float x4714 = x122[x4708];
bool x4715 = x4714 < -5.0f;
if (x4715) {
x122[x4708] = -5.0f;
} else {
}

}
float* x4721 = (float*)myMalloc(50 * sizeof(float));;
int32_t x4722 = 0;
int32_t x4723 = 0;
int32_t x4724 = 0;
for(int x4725=0; x4725 < 50; x4725++) {
int32_t x4726 = x4722;
int32_t x4727 = x4723;
float x4728 = x122[x4727];
int32_t x4729 = x4724;
float x4730 = x122[x4729];
float x4731 = x4728 * x4730;
x4721[x4726] = x4731;
x4722 += 1;
x4723 += 1;
x4724 += 1;

}
for(int x4738=0; x4738 < 50; x4738++) {
float x4739 = x222[x4738];
float x4740 = x4721[x4738];
float x4741 = x4739 + x4740;
x222[x4738] = x4741;

}
float* x4745 = (float*)myMalloc(50 * sizeof(float));;
for(int x4746=0; x4746 < 50; x4746++) {
float x4747 = x122[x4746];
float x4748 = x4747 * 0.1f;
x4745[x4746] = x4748;

}
float* x4752 = (float*)myMalloc(50 * sizeof(float));;
for(int x4753=0; x4753 < 50; x4753++) {
float x4754 = x222[x4753];
float x4755 = x4754 + 1.0E-8f;
x4752[x4753] = x4755;

}
float* x4759 = (float*)myMalloc(50 * sizeof(float));;
for(int x4760=0; x4760 < 50; x4760++) {
float x4761 = x4752[x4760];
double x4762 = (double)x4761;
double x4763 = sqrt(x4762);
float x4764 = (float)x4763;
x4759[x4760] = x4764;

}
float* x4768 = (float*)myMalloc(50 * sizeof(float));;
int32_t x4769 = 0;
int32_t x4770 = 0;
int32_t x4771 = 0;
for(int x4772=0; x4772 < 50; x4772++) {
int32_t x4773 = x4769;
int32_t x4774 = x4770;
float x4775 = x4745[x4774];
int32_t x4776 = x4771;
float x4777 = x4759[x4776];
float x4778 = x4775 / x4777;
x4768[x4773] = x4778;
x4769 += 1;
x4770 += 1;
x4771 += 1;

}
for(int x4785=0; x4785 < 50; x4785++) {
float x4786 = x117[x4785];
float x4787 = x4768[x4785];
float x4788 = x4786 - x4787;
x117[x4785] = x4788;

}
for(int x4792=0; x4792 < 50; x4792++) {
float x4793 = x122[x4792];
x122[x4792] = 0.0f;

}
for(int x4797=0; x4797 < 2500; x4797++) {
float x4798 = x112[x4797];
bool x4799 = x4798 > 5.0f;
if (x4799) {
x112[x4797] = 5.0f;
} else {
}
float x4803 = x112[x4797];
bool x4804 = x4803 < -5.0f;
if (x4804) {
x112[x4797] = -5.0f;
} else {
}

}
float* x4810 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x4811 = 0;
int32_t x4812 = 0;
int32_t x4813 = 0;
for(int x4814=0; x4814 < 50; x4814++) {
int32_t x4815 = x4812;
int32_t x4816 = x4813;
int32_t x4817 = x4811;
int32_t x4818 = x4817;
int32_t x4819 = x4815;
int32_t x4820 = x4816;
for(int x4821=0; x4821 < 50; x4821++) {
int32_t x4822 = x4818;
int32_t x4823 = x4819;
float x4824 = x112[x4823];
int32_t x4825 = x4820;
float x4826 = x112[x4825];
float x4827 = x4824 * x4826;
x4810[x4822] = x4827;
x4818 += 1;
x4819 += 1;
x4820 += 1;

}
x4811 += 50;
x4812 += 50;
x4813 += 50;

}
for(int x4839=0; x4839 < 2500; x4839++) {
float x4840 = x227[x4839];
float x4841 = x4810[x4839];
float x4842 = x4840 + x4841;
x227[x4839] = x4842;

}
float* x4846 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4847=0; x4847 < 2500; x4847++) {
float x4848 = x112[x4847];
float x4849 = x4848 * 0.1f;
x4846[x4847] = x4849;

}
float* x4853 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4854=0; x4854 < 2500; x4854++) {
float x4855 = x227[x4854];
float x4856 = x4855 + 1.0E-8f;
x4853[x4854] = x4856;

}
float* x4860 = (float*)myMalloc(2500 * sizeof(float));;
for(int x4861=0; x4861 < 2500; x4861++) {
float x4862 = x4853[x4861];
double x4863 = (double)x4862;
double x4864 = sqrt(x4863);
float x4865 = (float)x4864;
x4860[x4861] = x4865;

}
float* x4869 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x4870 = 0;
int32_t x4871 = 0;
int32_t x4872 = 0;
for(int x4873=0; x4873 < 50; x4873++) {
int32_t x4874 = x4871;
int32_t x4875 = x4872;
int32_t x4876 = x4870;
int32_t x4877 = x4876;
int32_t x4878 = x4874;
int32_t x4879 = x4875;
for(int x4880=0; x4880 < 50; x4880++) {
int32_t x4881 = x4877;
int32_t x4882 = x4878;
float x4883 = x4846[x4882];
int32_t x4884 = x4879;
float x4885 = x4860[x4884];
float x4886 = x4883 / x4885;
x4869[x4881] = x4886;
x4877 += 1;
x4878 += 1;
x4879 += 1;

}
x4870 += 50;
x4871 += 50;
x4872 += 50;

}
for(int x4898=0; x4898 < 2500; x4898++) {
float x4899 = x104[x4898];
float x4900 = x4869[x4898];
float x4901 = x4899 - x4900;
x104[x4898] = x4901;

}
for(int x4905=0; x4905 < 2500; x4905++) {
float x4906 = x112[x4905];
x112[x4905] = 0.0f;

}
for(int x4910=0; x4910 < 26; x4910++) {
float x4911 = x182[x4910];
bool x4912 = x4911 > 5.0f;
if (x4912) {
x182[x4910] = 5.0f;
} else {
}
float x4916 = x182[x4910];
bool x4917 = x4916 < -5.0f;
if (x4917) {
x182[x4910] = -5.0f;
} else {
}

}
float* x4923 = (float*)myMalloc(26 * sizeof(float));;
int32_t x4924 = 0;
int32_t x4925 = 0;
int32_t x4926 = 0;
for(int x4927=0; x4927 < 26; x4927++) {
int32_t x4928 = x4924;
int32_t x4929 = x4925;
float x4930 = x182[x4929];
int32_t x4931 = x4926;
float x4932 = x182[x4931];
float x4933 = x4930 * x4932;
x4923[x4928] = x4933;
x4924 += 1;
x4925 += 1;
x4926 += 1;

}
for(int x4940=0; x4940 < 26; x4940++) {
float x4941 = x232[x4940];
float x4942 = x4923[x4940];
float x4943 = x4941 + x4942;
x232[x4940] = x4943;

}
float* x4947 = (float*)myMalloc(26 * sizeof(float));;
for(int x4948=0; x4948 < 26; x4948++) {
float x4949 = x182[x4948];
float x4950 = x4949 * 0.1f;
x4947[x4948] = x4950;

}
float* x4954 = (float*)myMalloc(26 * sizeof(float));;
for(int x4955=0; x4955 < 26; x4955++) {
float x4956 = x232[x4955];
float x4957 = x4956 + 1.0E-8f;
x4954[x4955] = x4957;

}
float* x4961 = (float*)myMalloc(26 * sizeof(float));;
for(int x4962=0; x4962 < 26; x4962++) {
float x4963 = x4954[x4962];
double x4964 = (double)x4963;
double x4965 = sqrt(x4964);
float x4966 = (float)x4965;
x4961[x4962] = x4966;

}
float* x4970 = (float*)myMalloc(26 * sizeof(float));;
int32_t x4971 = 0;
int32_t x4972 = 0;
int32_t x4973 = 0;
for(int x4974=0; x4974 < 26; x4974++) {
int32_t x4975 = x4971;
int32_t x4976 = x4972;
float x4977 = x4947[x4976];
int32_t x4978 = x4973;
float x4979 = x4961[x4978];
float x4980 = x4977 / x4979;
x4970[x4975] = x4980;
x4971 += 1;
x4972 += 1;
x4973 += 1;

}
for(int x4987=0; x4987 < 26; x4987++) {
float x4988 = x176[x4987];
float x4989 = x4970[x4987];
float x4990 = x4988 - x4989;
x176[x4987] = x4990;

}
for(int x4994=0; x4994 < 26; x4994++) {
float x4995 = x182[x4994];
x182[x4994] = 0.0f;

}
for(int x4999=0; x4999 < 1300; x4999++) {
float x5000 = x171[x4999];
bool x5001 = x5000 > 5.0f;
if (x5001) {
x171[x4999] = 5.0f;
} else {
}
float x5005 = x171[x4999];
bool x5006 = x5005 < -5.0f;
if (x5006) {
x171[x4999] = -5.0f;
} else {
}

}
float* x5012 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x5013 = 0;
int32_t x5014 = 0;
int32_t x5015 = 0;
for(int x5016=0; x5016 < 26; x5016++) {
int32_t x5017 = x5014;
int32_t x5018 = x5015;
int32_t x5019 = x5013;
int32_t x5020 = x5019;
int32_t x5021 = x5017;
int32_t x5022 = x5018;
for(int x5023=0; x5023 < 50; x5023++) {
int32_t x5024 = x5020;
int32_t x5025 = x5021;
float x5026 = x171[x5025];
int32_t x5027 = x5022;
float x5028 = x171[x5027];
float x5029 = x5026 * x5028;
x5012[x5024] = x5029;
x5020 += 1;
x5021 += 1;
x5022 += 1;

}
x5013 += 50;
x5014 += 50;
x5015 += 50;

}
for(int x5041=0; x5041 < 1300; x5041++) {
float x5042 = x237[x5041];
float x5043 = x5012[x5041];
float x5044 = x5042 + x5043;
x237[x5041] = x5044;

}
float* x5048 = (float*)myMalloc(1300 * sizeof(float));;
for(int x5049=0; x5049 < 1300; x5049++) {
float x5050 = x171[x5049];
float x5051 = x5050 * 0.1f;
x5048[x5049] = x5051;

}
float* x5055 = (float*)myMalloc(1300 * sizeof(float));;
for(int x5056=0; x5056 < 1300; x5056++) {
float x5057 = x237[x5056];
float x5058 = x5057 + 1.0E-8f;
x5055[x5056] = x5058;

}
float* x5062 = (float*)myMalloc(1300 * sizeof(float));;
for(int x5063=0; x5063 < 1300; x5063++) {
float x5064 = x5055[x5063];
double x5065 = (double)x5064;
double x5066 = sqrt(x5065);
float x5067 = (float)x5066;
x5062[x5063] = x5067;

}
float* x5071 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x5072 = 0;
int32_t x5073 = 0;
int32_t x5074 = 0;
for(int x5075=0; x5075 < 26; x5075++) {
int32_t x5076 = x5073;
int32_t x5077 = x5074;
int32_t x5078 = x5072;
int32_t x5079 = x5078;
int32_t x5080 = x5076;
int32_t x5081 = x5077;
for(int x5082=0; x5082 < 50; x5082++) {
int32_t x5083 = x5079;
int32_t x5084 = x5080;
float x5085 = x5048[x5084];
int32_t x5086 = x5081;
float x5087 = x5062[x5086];
float x5088 = x5085 / x5087;
x5071[x5083] = x5088;
x5079 += 1;
x5080 += 1;
x5081 += 1;

}
x5072 += 50;
x5073 += 50;
x5074 += 50;

}
for(int x5100=0; x5100 < 1300; x5100++) {
float x5101 = x163[x5100];
float x5102 = x5071[x5100];
float x5103 = x5101 - x5102;
x163[x5100] = x5103;

}
for(int x5107=0; x5107 < 1300; x5107++) {
float x5108 = x171[x5107];
x171[x5107] = 0.0f;

}
for(int x5112=0; x5112 < 2500; x5112++) {
float x5113 = x148[x5112];
bool x5114 = x5113 > 5.0f;
if (x5114) {
x148[x5112] = 5.0f;
} else {
}
float x5118 = x148[x5112];
bool x5119 = x5118 < -5.0f;
if (x5119) {
x148[x5112] = -5.0f;
} else {
}

}
float* x5125 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x5126 = 0;
int32_t x5127 = 0;
int32_t x5128 = 0;
for(int x5129=0; x5129 < 50; x5129++) {
int32_t x5130 = x5127;
int32_t x5131 = x5128;
int32_t x5132 = x5126;
int32_t x5133 = x5132;
int32_t x5134 = x5130;
int32_t x5135 = x5131;
for(int x5136=0; x5136 < 50; x5136++) {
int32_t x5137 = x5133;
int32_t x5138 = x5134;
float x5139 = x148[x5138];
int32_t x5140 = x5135;
float x5141 = x148[x5140];
float x5142 = x5139 * x5141;
x5125[x5137] = x5142;
x5133 += 1;
x5134 += 1;
x5135 += 1;

}
x5126 += 50;
x5127 += 50;
x5128 += 50;

}
for(int x5154=0; x5154 < 2500; x5154++) {
float x5155 = x242[x5154];
float x5156 = x5125[x5154];
float x5157 = x5155 + x5156;
x242[x5154] = x5157;

}
float* x5161 = (float*)myMalloc(2500 * sizeof(float));;
for(int x5162=0; x5162 < 2500; x5162++) {
float x5163 = x148[x5162];
float x5164 = x5163 * 0.1f;
x5161[x5162] = x5164;

}
float* x5168 = (float*)myMalloc(2500 * sizeof(float));;
for(int x5169=0; x5169 < 2500; x5169++) {
float x5170 = x242[x5169];
float x5171 = x5170 + 1.0E-8f;
x5168[x5169] = x5171;

}
float* x5175 = (float*)myMalloc(2500 * sizeof(float));;
for(int x5176=0; x5176 < 2500; x5176++) {
float x5177 = x5168[x5176];
double x5178 = (double)x5177;
double x5179 = sqrt(x5178);
float x5180 = (float)x5179;
x5175[x5176] = x5180;

}
float* x5184 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x5185 = 0;
int32_t x5186 = 0;
int32_t x5187 = 0;
for(int x5188=0; x5188 < 50; x5188++) {
int32_t x5189 = x5186;
int32_t x5190 = x5187;
int32_t x5191 = x5185;
int32_t x5192 = x5191;
int32_t x5193 = x5189;
int32_t x5194 = x5190;
for(int x5195=0; x5195 < 50; x5195++) {
int32_t x5196 = x5192;
int32_t x5197 = x5193;
float x5198 = x5161[x5197];
int32_t x5199 = x5194;
float x5200 = x5175[x5199];
float x5201 = x5198 / x5200;
x5184[x5196] = x5201;
x5192 += 1;
x5193 += 1;
x5194 += 1;

}
x5185 += 50;
x5186 += 50;
x5187 += 50;

}
for(int x5213=0; x5213 < 2500; x5213++) {
float x5214 = x140[x5213];
float x5215 = x5184[x5213];
float x5216 = x5214 - x5215;
x140[x5213] = x5216;

}
for(int x5220=0; x5220 < 2500; x5220++) {
float x5221 = x148[x5220];
x148[x5220] = 0.0f;

}
for(int x5225=0; x5225 < 1300; x5225++) {
float x5226 = x135[x5225];
bool x5227 = x5226 > 5.0f;
if (x5227) {
x135[x5225] = 5.0f;
} else {
}
float x5231 = x135[x5225];
bool x5232 = x5231 < -5.0f;
if (x5232) {
x135[x5225] = -5.0f;
} else {
}

}
float* x5238 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x5239 = 0;
int32_t x5240 = 0;
int32_t x5241 = 0;
for(int x5242=0; x5242 < 50; x5242++) {
int32_t x5243 = x5240;
int32_t x5244 = x5241;
int32_t x5245 = x5239;
int32_t x5246 = x5245;
int32_t x5247 = x5243;
int32_t x5248 = x5244;
for(int x5249=0; x5249 < 26; x5249++) {
int32_t x5250 = x5246;
int32_t x5251 = x5247;
float x5252 = x135[x5251];
int32_t x5253 = x5248;
float x5254 = x135[x5253];
float x5255 = x5252 * x5254;
x5238[x5250] = x5255;
x5246 += 1;
x5247 += 1;
x5248 += 1;

}
x5239 += 26;
x5240 += 26;
x5241 += 26;

}
for(int x5267=0; x5267 < 1300; x5267++) {
float x5268 = x247[x5267];
float x5269 = x5238[x5267];
float x5270 = x5268 + x5269;
x247[x5267] = x5270;

}
float* x5274 = (float*)myMalloc(1300 * sizeof(float));;
for(int x5275=0; x5275 < 1300; x5275++) {
float x5276 = x135[x5275];
float x5277 = x5276 * 0.1f;
x5274[x5275] = x5277;

}
float* x5281 = (float*)myMalloc(1300 * sizeof(float));;
for(int x5282=0; x5282 < 1300; x5282++) {
float x5283 = x247[x5282];
float x5284 = x5283 + 1.0E-8f;
x5281[x5282] = x5284;

}
float* x5288 = (float*)myMalloc(1300 * sizeof(float));;
for(int x5289=0; x5289 < 1300; x5289++) {
float x5290 = x5281[x5289];
double x5291 = (double)x5290;
double x5292 = sqrt(x5291);
float x5293 = (float)x5292;
x5288[x5289] = x5293;

}
float* x5297 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x5298 = 0;
int32_t x5299 = 0;
int32_t x5300 = 0;
for(int x5301=0; x5301 < 50; x5301++) {
int32_t x5302 = x5299;
int32_t x5303 = x5300;
int32_t x5304 = x5298;
int32_t x5305 = x5304;
int32_t x5306 = x5302;
int32_t x5307 = x5303;
for(int x5308=0; x5308 < 26; x5308++) {
int32_t x5309 = x5305;
int32_t x5310 = x5306;
float x5311 = x5274[x5310];
int32_t x5312 = x5307;
float x5313 = x5288[x5312];
float x5314 = x5311 / x5313;
x5297[x5309] = x5314;
x5305 += 1;
x5306 += 1;
x5307 += 1;

}
x5298 += 26;
x5299 += 26;
x5300 += 26;

}
for(int x5326=0; x5326 < 1300; x5326++) {
float x5327 = x127[x5326];
float x5328 = x5297[x5326];
float x5329 = x5327 - x5328;
x127[x5326] = x5329;

}
for(int x5333=0; x5333 < 1300; x5333++) {
float x5334 = x135[x5333];
x135[x5333] = 0.0f;

}
for(int x5338=0; x5338 < 50; x5338++) {
float x5339 = x158[x5338];
bool x5340 = x5339 > 5.0f;
if (x5340) {
x158[x5338] = 5.0f;
} else {
}
float x5344 = x158[x5338];
bool x5345 = x5344 < -5.0f;
if (x5345) {
x158[x5338] = -5.0f;
} else {
}

}
float* x5351 = (float*)myMalloc(50 * sizeof(float));;
int32_t x5352 = 0;
int32_t x5353 = 0;
int32_t x5354 = 0;
for(int x5355=0; x5355 < 50; x5355++) {
int32_t x5356 = x5352;
int32_t x5357 = x5353;
float x5358 = x158[x5357];
int32_t x5359 = x5354;
float x5360 = x158[x5359];
float x5361 = x5358 * x5360;
x5351[x5356] = x5361;
x5352 += 1;
x5353 += 1;
x5354 += 1;

}
for(int x5368=0; x5368 < 50; x5368++) {
float x5369 = x252[x5368];
float x5370 = x5351[x5368];
float x5371 = x5369 + x5370;
x252[x5368] = x5371;

}
float* x5375 = (float*)myMalloc(50 * sizeof(float));;
for(int x5376=0; x5376 < 50; x5376++) {
float x5377 = x158[x5376];
float x5378 = x5377 * 0.1f;
x5375[x5376] = x5378;

}
float* x5382 = (float*)myMalloc(50 * sizeof(float));;
for(int x5383=0; x5383 < 50; x5383++) {
float x5384 = x252[x5383];
float x5385 = x5384 + 1.0E-8f;
x5382[x5383] = x5385;

}
float* x5389 = (float*)myMalloc(50 * sizeof(float));;
for(int x5390=0; x5390 < 50; x5390++) {
float x5391 = x5382[x5390];
double x5392 = (double)x5391;
double x5393 = sqrt(x5392);
float x5394 = (float)x5393;
x5389[x5390] = x5394;

}
float* x5398 = (float*)myMalloc(50 * sizeof(float));;
int32_t x5399 = 0;
int32_t x5400 = 0;
int32_t x5401 = 0;
for(int x5402=0; x5402 < 50; x5402++) {
int32_t x5403 = x5399;
int32_t x5404 = x5400;
float x5405 = x5375[x5404];
int32_t x5406 = x5401;
float x5407 = x5389[x5406];
float x5408 = x5405 / x5407;
x5398[x5403] = x5408;
x5399 += 1;
x5400 += 1;
x5401 += 1;

}
for(int x5415=0; x5415 < 50; x5415++) {
float x5416 = x153[x5415];
float x5417 = x5398[x5415];
float x5418 = x5416 - x5417;
x153[x5415] = x5418;

}
for(int x5422=0; x5422 < 50; x5422++) {
float x5423 = x158[x5422];
x158[x5422] = 0.0f;

}
mallocAddr = (void*)x259;

}
double x5430 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x5433 = (long)fopen(x0, "w");
fprintf((FILE *)x5433, "unit: %s\n", "100 iteration");
for(int x5436=0; x5436 < 51; x5436++) {
double x5437 = x258[x5436];
fprintf((FILE *)x5433, "%lf\n", x5437);

}
double x5431 = x257 - x2;
double x5432 = x5430 - x257;
fprintf((FILE *)x5433, "run time: %lf %lf\n", x5431, x5432);
fclose((FILE*)x5433);
// Backend cleanup.
}
/*****************************************
  End of C Generated Code                  
*******************************************/

