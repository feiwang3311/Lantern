#include <fcntl.h>
#include <errno.h>
#include <err.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <functional>
#include <memory>
#include <math.h>
#include <random>

using namespace std;
#ifndef MAP_FILE
#define MAP_FILE MAP_SHARED
#endif
int fsize(int fd) {
	struct stat stat;
	int res = fstat(fd,&stat);
	return stat.st_size;
}
int printll(char* s) {
	while (*s != '\n' && *s != ',' && *s != '\t') {
		putchar(*s++);
	}
	return 0;
}
long hash(char *str0, int len)
{
	unsigned char* str = (unsigned char*)str0;
	unsigned long hash = 5381;
	int c;

	while ((c = *str++) && len--)
		hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

	return hash;
}
int HEAP_SIZE = 1073741826; // 1048576;  //2147483652; //536870912; // 268435456; //2097152;
void *mallocBase = malloc(HEAP_SIZE);
void *mallocAddr = mallocBase;
void *waterMark  = mallocBase;
void* myMalloc(size_t bytes) {
	void* res = mallocAddr;
	mallocAddr += bytes;
	return res;
}

int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
	long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
	result->tv_sec = diff / 1000000;
	result->tv_usec = diff % 1000000;
	return (diff<0);
}



void Snippet(char*);

std::random_device rd{};
std::mt19937 gen{rd()};
std::normal_distribution<> d{0,1};

int main(int argc, char *argv[])
{

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
	printf("Here we go!! Go MNIST!!!!\n");
	srand(42);
	double* x3 = (double*)myMalloc(250 * sizeof(double));
	for(int x5=0; x5 < 250; x5++) {
		double x6 = (double)rand()/RAND_MAX;
		double x7 = x6 - 0.5;
		double x8 = x7 * 0.2;
		x3[x5] = x8;

	}
	double* x12 = (double*)myMalloc(250 * sizeof(double));
	for(int x13=0; x13 < 250; x13++) {
		x12[x13] = 0.0;

	}
	double* x17 = (double*)myMalloc(5000 * sizeof(double));
	for(int x19=0; x19 < 5000; x19++) {
		double x20 = (double)rand()/RAND_MAX;
		double x21 = x20 - 0.5;
		double x22 = x21 * 0.063;
		x17[x19] = x22;

	}
	double* x26 = (double*)myMalloc(5000 * sizeof(double));
	for(int x27=0; x27 < 5000; x27++) {
		x26[x27] = 0.0;

	}
	double* x31 = (double*)myMalloc(16000 * sizeof(double));
	for(int x33=0; x33 < 16000; x33++) {
		double x34 = (double)rand()/RAND_MAX;
		double x35 = x34 - 0.5;
		double x36 = x35 * 0.055;
		x31[x33] = x36;

	}
	double* x40 = (double*)myMalloc(50 * sizeof(double));
	for(int x42=0; x42 < 50; x42++) {
		double x43 = (double)rand()/RAND_MAX;
		double x44 = x43 - 0.5;
		double x45 = x44 * 0.055;
		x40[x42] = x45;

	}
	double* x49 = (double*)myMalloc(16000 * sizeof(double));
	for(int x50=0; x50 < 16000; x50++) {
		x49[x50] = 0.0;

	}
	double* x54 = (double*)myMalloc(50 * sizeof(double));
	for(int x55=0; x55 < 50; x55++) {
		x54[x55] = 0.0;

	}
	double* x59 = (double*)myMalloc(500 * sizeof(double));
	for(int x61=0; x61 < 500; x61++) {
		double x62 = (double)rand()/RAND_MAX;
		double x63 = x62 - 0.5;
		double x64 = x63 * 0.15;
		x59[x61] = x64;

	}
	double* x68 = (double*)myMalloc(10 * sizeof(double));
	for(int x70=0; x70 < 10; x70++) {
		double x71 = (double)rand()/RAND_MAX;
		double x72 = x71 - 0.5;
		double x73 = x72 * 0.05;
		x68[x70] = x73;

	}
	double* x77 = (double*)myMalloc(500 * sizeof(double));
	for(int x78=0; x78 < 500; x78++) {
		x77[x78] = 0.0;

	}
	double* x82 = (double*)myMalloc(10 * sizeof(double));
	for(int x83=0; x83 < 10; x83++) {
		x82[x83] = 0.0;

	}
	struct timeval begin_0, end_0, diff_0;
	gettimeofday(&begin_0, NULL);
	printf("Start normalize\n");
	int32_t x100 = 0;
	int32_t x101 = x100;
	int32_t x102 = x101;
	int32_t x94 = open("../data/bin/mnist_train_target.bin",0);
	printf("I can open the file1\n");
	int64_t x95 = fsize(x94);
	int64_t x97 = x95 / 4LL;
	int32_t x98 = (int32_t)x97;
	int* x96 = (int32_t*)mmap(0, x95, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x94, 0);
	int32_t x89 = open("../data/bin/mnist_train.bin",0);
	int64_t x90 = fsize(x89);
	printf("I can open the file2\n");
	double* x91 = (double*)mmap(0, x90, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x89, 0);
	printf("after mmap\n");
	for(int x104=0; x104 < x98; x104++) {
		int32_t x105 = x102;
		int32_t x107 = x96[x104];
		double* x106 = x91+x105;
		for(int x109=0; x109 < 784; x109++) {
			double x110 = x106[x109];
			double x111 = x110 - 0.1307;
			double x112 = x111 / 0.3081;
			x106[x109] = x112;

		}
		x102 += 784;

	}
	printf("still ok after trying for loop \n");
	int32_t x119 = x102;
	int64_t x92 = x90 / 8LL;
	int32_t x93 = (int32_t)x92;
	bool x120 = x119 == x93;
	if (x120) {
	} else {
		printf("Data length doesn't match\n");
		exit(0);
	}
	int32_t x136 = 0;
	int32_t x137 = x136;
	int32_t x138 = x137;
	int32_t x131 = open("../data/bin/mnist_test_target.bin",0);
	printf("I can open the file3\n");
	int64_t x132 = fsize(x131);
	int64_t x134 = x132 / 4LL;
	int32_t x135 = (int32_t)x134;
	int* x133 = (int32_t*)mmap(0, x132, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x131, 0);
	int32_t x126 = open("../data/bin/mnist_test.bin",0);
	printf("I can open the file4\n");
	int64_t x127 = fsize(x126);
	double* x128 = (double*)mmap(0, x127, PROT_READ | PROT_WRITE, MAP_FILE | MAP_PRIVATE, x126, 0);
	for(int x140=0; x140 < x135; x140++) {
		int32_t x141 = x138;
		int32_t x143 = x133[x140];
		double* x142 = x128+x141;
		for(int x144=0; x144 < 784; x144++) {
			double x145 = x142[x144];
			double x146 = x145 - 0.1307;
			double x147 = x146 / 0.3081;
			x142[x144] = x147;

		}
		x138 += 784;

	}
	int32_t x154 = x138;
	int64_t x129 = x127 / 8LL;
	int32_t x130 = (int32_t)x129;
	bool x155 = x154 == x130;
	if (x155) {
	} else {
		printf("Data length doesn't match\n");
		exit(0);
	}
	gettimeofday(&end_0, NULL);
	timeval_subtract(&diff_0, &end_0, &begin_0);;
	int64_t x163 = ((diff_0.tv_sec * 1000L) + (diff_0.tv_usec/1000L));
	printf("Data normalized in %ldms\n",x163);
	int64_t x165 = (long)mallocAddr;
	int32_t x1012 = x98 / 10;
	double x1017 = (double)x98;
	int64_t x1039 = (int64_t)x98;
	double x1440 = (double)x135;
	for(int x166=0; x166 < 10; x166++) {
		struct timeval begin_1, end_1, diff_1;
		int32_t x168 = 0;
		int32_t x169 = x168;
		int32_t x170 = x169;
		double x171 = 0.0;
		double x172 = x171;
		double x173 = x172;
		int32_t x174 = x166 + 1;
		printf("Start training epoch %d\n",x174);
		gettimeofday(&begin_1, NULL);
		int32_t x177 = 0;
		int32_t x178 = x177;
		int32_t x179 = x178;
		for(int x180=0; x180 < x98; x180++) {
			int32_t x181 = x179;
			int32_t x183 = x96[x180];
			x170 += 1;
			double* x185 = (double*)myMalloc(1 * sizeof(double));
			x185[0] = 0.0;
			double* x187 = (double*)myMalloc(1 * sizeof(double));
			x187[0] = 0.0;
			double* x189 = (double*)myMalloc(1 * sizeof(double));
			for(int x191=0; x191 < 1; x191++) {
				x189[x191] = 0.0;

			}
			double* x195 = (double*)myMalloc(1 * sizeof(double));
			for(int x196=0; x196 < 1; x196++) {
				x195[x196] = 0.0;

			}
			double* x200 = (double*)myMalloc(5760 * sizeof(double));
			for(int x202=0; x202 < 5760; x202++) {
				x200[x202] = 0.0;

			}
			int32_t x206 = 0;
			int32_t x207 = 0;
			double* x182 = x91+x181;
			for(int x208=0; x208 < 10; x208++) {
				int32_t x209 = x207;
				int32_t x210 = x209;
				int32_t x211 = 0;
				int32_t x212 = x206;
				double* x213 = x200+x212;
				for(int x214=0; x214 < 1; x214++) {
					int32_t x215 = x211;
					int32_t x217 = x210;
					double* x218 = x3+x217;
					int32_t x219 = 0;
					int32_t x220 = 0;
					double* x216 = x182+x215;
					for(int x222=0; x222 < 24; x222++) {
						int32_t x223 = x220;
						int32_t x224 = x223;
						for(int x225=0; x225 < 24; x225++) {
							int32_t x226 = 0;
							int32_t x227 = x224;
							int32_t x228 = x227;
							double x229 = 0.0;
							for(int x231=0; x231 < 5; x231++) {
								int32_t x232 = x228;
								int32_t x234 = x226;
								double* x235 = x218+x234;
								double* x233 = x216+x232;
								for(int x236=0; x236 < 5; x236++) {
									double x237 = x233[x236];
									double x238 = x235[x236];
									double x239 = x237 * x238;
									x229 += x239;

								}
								x226 += 5;
								x228 += 28;

							}
							int32_t x247 = x219;
							double x248 = x213[x247];
							double x249 = x229;
							double x250 = x248 + x249;
							x213[x247] = x250;
							x219 += 1;
							x224 += 1;

						}
						x220 += 28;

					}
					x210 += 25;
					x211 += 784;

				}
				x207 += 25;
				x206 += 576;

			}
			double* x267 = (double*)myMalloc(5760 * sizeof(double));
			for(int x268=0; x268 < 5760; x268++) {
				x267[x268] = 0.0;

			}
			double* x272 = (double*)myMalloc(1440 * sizeof(double));
			for(int x274=0; x274 < 1440; x274++) {
				x272[x274] = -1.0E10;

			}
			int32_t* x278 = (int32_t*)myMalloc(1440 * sizeof(int32_t));
			int32_t x279 = 0;
			int32_t x280 = 0;
			for(int x281=0; x281 < 10; x281++) {
				int32_t x282 = x279;
				int32_t x283 = x282;
				for(int x285=0; x285 < 12; x285++) {
					for(int x287=0; x287 < 2; x287++) {
						int32_t x288 = x283;
						int32_t x289 = x288;
						for(int x290=0; x290 < 12; x290++) {
							int32_t x291 = x280;
							double x292 = x200[x291];
							int32_t x293 = x289;
							double x294 = x272[x293];
							bool x295 = x292 > x294;
							if (x295) {
								double x296 = x200[x291];
								x272[x293] = x296;
								x278[x293] = x291;
							} else {
							}
							x280 += 1;
							int32_t x302 = x280;
							double x303 = x200[x302];
							double x304 = x272[x293];
							bool x305 = x303 > x304;
							if (x305) {
								double x306 = x200[x302];
								x272[x293] = x306;
								x278[x293] = x302;
							} else {
							}
							x280 += 1;
							x289 += 1;

						}

					}
					x283 += 12;

				}
				x279 += 144;

			}
			double* x323 = (double*)myMalloc(1440 * sizeof(double));
			for(int x324=0; x324 < 1440; x324++) {
				x323[x324] = 0.0;

			}
			double* x328 = (double*)myMalloc(1440 * sizeof(double));
			for(int x329=0; x329 < 1440; x329++) {
				double x330 = x272[x329];
				bool x331 = x330 < 0.0;
				if (x331) {
					x328[x329] = 0.0;
				} else {
					double x334 = x272[x329];
					x328[x329] = x334;
				}

			}
			double* x340 = (double*)myMalloc(1440 * sizeof(double));
			for(int x341=0; x341 < 1440; x341++) {
				x340[x341] = 0.0;

			}
			double* x345 = (double*)myMalloc(1280 * sizeof(double));
			for(int x347=0; x347 < 1280; x347++) {
				x345[x347] = 0.0;

			}
			int32_t x351 = 0;
			int32_t x352 = 0;
			for(int x354=0; x354 < 20; x354++) {
				int32_t x355 = x352;
				int32_t x356 = x355;
				int32_t x357 = 0;
				int32_t x358 = x351;
				double* x359 = x345+x358;
				for(int x360=0; x360 < 10; x360++) {
					int32_t x361 = x357;
					double* x362 = x328+x361;
					int32_t x363 = x356;
					double* x364 = x17+x363;
					int32_t x365 = 0;
					int32_t x366 = 0;
					for(int x368=0; x368 < 8; x368++) {
						int32_t x369 = x366;
						int32_t x370 = x369;
						for(int x371=0; x371 < 8; x371++) {
							int32_t x372 = 0;
							int32_t x373 = x370;
							int32_t x374 = x373;
							double x375 = 0.0;
							for(int x376=0; x376 < 5; x376++) {
								int32_t x377 = x374;
								double* x378 = x362+x377;
								int32_t x379 = x372;
								double* x380 = x364+x379;
								for(int x381=0; x381 < 5; x381++) {
									double x382 = x378[x381];
									double x383 = x380[x381];
									double x384 = x382 * x383;
									x375 += x384;

								}
								x372 += 5;
								x374 += 12;

							}
							int32_t x392 = x365;
							double x393 = x359[x392];
							double x394 = x375;
							double x395 = x393 + x394;
							x359[x392] = x395;
							x365 += 1;
							x370 += 1;

						}
						x366 += 12;

					}
					x356 += 25;
					x357 += 144;

				}
				x352 += 250;
				x351 += 64;

			}
			double* x412 = (double*)myMalloc(1280 * sizeof(double));
			for(int x413=0; x413 < 1280; x413++) {
				x412[x413] = 0.0;

			}
			double* x417 = (double*)myMalloc(320 * sizeof(double));
			for(int x419=0; x419 < 320; x419++) {
				x417[x419] = -1.0E10;

			}
			int32_t* x423 = (int32_t*)myMalloc(320 * sizeof(int32_t));
			int32_t x424 = 0;
			int32_t x425 = 0;
			for(int x426=0; x426 < 20; x426++) {
				int32_t x427 = x424;
				int32_t x428 = x427;
				for(int x430=0; x430 < 4; x430++) {
					for(int x431=0; x431 < 2; x431++) {
						int32_t x432 = x428;
						int32_t x433 = x432;
						for(int x434=0; x434 < 4; x434++) {
							int32_t x435 = x425;
							double x436 = x345[x435];
							int32_t x437 = x433;
							double x438 = x417[x437];
							bool x439 = x436 > x438;
							if (x439) {
								double x440 = x345[x435];
								x417[x437] = x440;
								x423[x437] = x435;
							} else {
							}
							x425 += 1;
							int32_t x446 = x425;
							double x447 = x345[x446];
							double x448 = x417[x437];
							bool x449 = x447 > x448;
							if (x449) {
								double x450 = x345[x446];
								x417[x437] = x450;
								x423[x437] = x446;
							} else {
							}
							x425 += 1;
							x433 += 1;

						}

					}
					x428 += 4;

				}
				x424 += 16;

			}
			double* x467 = (double*)myMalloc(320 * sizeof(double));
			for(int x468=0; x468 < 320; x468++) {
				x467[x468] = 0.0;

			}
			double* x472 = (double*)myMalloc(320 * sizeof(double));
			for(int x473=0; x473 < 320; x473++) {
				double x474 = x417[x473];
				bool x475 = x474 < 0.0;
				if (x475) {
					x472[x473] = 0.0;
				} else {
					double x478 = x417[x473];
					x472[x473] = x478;
				}

			}
			double* x484 = (double*)myMalloc(320 * sizeof(double));
			for(int x485=0; x485 < 320; x485++) {
				x484[x485] = 0.0;

			}
			double* x489 = (double*)myMalloc(320 * sizeof(double));
			double* x490 = (double*)myMalloc(320 * sizeof(double));
			for(int x491=0; x491 < 320; x491++) {
				double x492 = (double)rand()/RAND_MAX;
				bool x493 = x492 > 0.5;
				if (x493) {
					double x494 = x472[x491];
					double x495 = x494 * 2.0;
					x489[x491] = x495;
					x490[x491] = 2.0;
				} else {
					x489[x491] = 0.0;
					x490[x491] = 0.0;
				}

			}
			double* x505 = (double*)myMalloc(320 * sizeof(double));
			for(int x506=0; x506 < 320; x506++) {
				x505[x506] = 0.0;

			}
			// dot WrappedArray(50, 320) - WrappedArray(320)
			int32_t x511 = 0;
			double* x512 = (double*)myMalloc(50 * sizeof(double));
			for(int x513=0; x513 < 50; x513++) {
				double x514 = 0.0;
				for(int x515=0; x515 < 320; x515++) {
					int32_t x516 = x511;
					double x517 = x31[x516];
					double x518 = x489[x515];
					double x519 = x517 * x518;
					x514 += x519;
					x511 += 1;

				}
				double x524 = x514;
				x512[x513] = x524;

			}
			double* x528 = (double*)myMalloc(50 * sizeof(double));
			for(int x529=0; x529 < 50; x529++) {
				x528[x529] = 0.0;

			}
			double* x533 = (double*)myMalloc(50 * sizeof(double));
			for(int x534=0; x534 < 50; x534++) {
				double x535 = x512[x534];
				double x536 = x40[x534];
				double x537 = x535 + x536;
				x533[x534] = x537;

			}
			double* x541 = (double*)myMalloc(50 * sizeof(double));
			for(int x542=0; x542 < 50; x542++) {
				x541[x542] = 0.0;

			}
			double* x546 = (double*)myMalloc(50 * sizeof(double));
			for(int x547=0; x547 < 50; x547++) {
				double x548 = x533[x547];
				bool x549 = x548 < 0.0;
				if (x549) {
					x546[x547] = 0.0;
				} else {
					double x552 = x533[x547];
					x546[x547] = x552;
				}

			}
			double* x558 = (double*)myMalloc(50 * sizeof(double));
			for(int x559=0; x559 < 50; x559++) {
				x558[x559] = 0.0;

			}
			// dot WrappedArray(10, 50) - WrappedArray(50)
			int32_t x564 = 0;
			double* x565 = (double*)myMalloc(10 * sizeof(double));
			for(int x566=0; x566 < 10; x566++) {
				double x567 = 0.0;
				for(int x568=0; x568 < 50; x568++) {
					int32_t x569 = x564;
					double x570 = x59[x569];
					double x571 = x546[x568];
					double x572 = x570 * x571;
					x567 += x572;
					x564 += 1;

				}
				double x577 = x567;
				x565[x566] = x577;

			}
			double* x581 = (double*)myMalloc(10 * sizeof(double));
			for(int x582=0; x582 < 10; x582++) {
				x581[x582] = 0.0;

			}
			double* x586 = (double*)myMalloc(10 * sizeof(double));
			for(int x587=0; x587 < 10; x587++) {
				double x588 = x565[x587];
				double x589 = x68[x587];
				double x590 = x588 + x589;
				x586[x587] = x590;

			}
			double* x594 = (double*)myMalloc(10 * sizeof(double));
			for(int x595=0; x595 < 10; x595++) {
				x594[x595] = 0.0;

			}
			double x599 = -1.0E10;
			for(int x600=0; x600 < 10; x600++) {
				double x601 = x599;
				double x602 = x586[x600];
				bool x603 = x602 > x601;
				double x604;
				if (x603) {
					x604 = x602;
				} else {
					x604 = x601;
				}
				x599 = x604;

			}
			double x608 = x599;
			double x609 = 0.0;
			for(int x610=0; x610 < 10; x610++) {
				double x611 = x609;
				double x612 = x586[x610];
				double x613 = x599;
				double x614 = x612 - x613;
				double x615 = exp(x614);
				double x616 = x611 + x615;
				x609 = x616;

			}
			double x620 = x609;
			double* x623 = (double*)myMalloc(10 * sizeof(double));
			double x621 = log(x620);
			double x622 = x608 + x621;
			for(int x624=0; x624 < 10; x624++) {
				double x625 = x586[x624];
				double x626 = x625 - x622;
				x623[x624] = x626;

			}
			double* x630 = (double*)myMalloc(10 * sizeof(double));
			for(int x631=0; x631 < 10; x631++) {
				x630[x631] = 0.0;

			}
			double x635 = x623[x183];
			double* x637 = (double*)myMalloc(1 * sizeof(double));
			double x636 = -1.0 * x635;
			x637[0] = x636;
			double* x639 = (double*)myMalloc(1 * sizeof(double));
			for(int x640=0; x640 < 1; x640++) {
				x639[x640] = 0.0;

			}
			for(int x644=0; x644 < 1; x644++) {
				double x645 = x639[x644];
				x639[x644] = 1.0;

			}
			for(int x649=0; x649 < 1; x649++) {
				double x650 = x637[x649];
				x195[x649] = x650;

			}
			double x654 = x639[0];
			double x655 = -1.0 * x654;
			x630[x183] = x655;
			double x657 = 0.0;
			for(int x658=0; x658 < 10; x658++) {
				double x659 = x657;
				double x660 = x630[x658];
				double x661 = x659 + x660;
				x657 = x661;

			}
			double x665 = x657;
			double* x666 = (double*)myMalloc(1 * sizeof(double));
			x666[0] = x665;
			double x668 = x666[0];
			for(int x669=0; x669 < 10; x669++) {
				double x670 = x630[x669];
				double x671 = x623[x669];
				double x672 = exp(x671);
				double x673 = x672 * x668;
				double x674 = x670 - x673;
				x594[x669] = x674;

			}
			// backpropagate +
			for(int x679=0; x679 < 10; x679++) {
				double x680 = x581[x679];
				double x681 = x594[x679];
				double x682 = x680 + x681;
				x581[x679] = x682;

			}
			for(int x686=0; x686 < 10; x686++) {
				double x687 = x82[x686];
				double x688 = x594[x686];
				double x689 = x687 + x688;
				x82[x686] = x689;

			}
			// add_cartesian
			int32_t x694 = 0;
			for(int x695=0; x695 < 10; x695++) {
				for(int x696=0; x696 < 50; x696++) {
					int32_t x697 = x694;
					int32_t x698 = x697 + x696;
					double x699 = x77[x698];
					double x700 = x546[x696];
					double x701 = x581[x695];
					double x702 = x700 * x701;
					double x703 = x699 + x702;
					x77[x698] = x703;

				}
				x694 += 50;

			}
			int32_t x710 = 0;
			for(int x711=0; x711 < 10; x711++) {
				for(int x712=0; x712 < 50; x712++) {
					double x713 = x558[x712];
					int32_t x714 = x710;
					int32_t x715 = x714 + x712;
					double x716 = x59[x715];
					double x717 = x581[x711];
					double x718 = x716 * x717;
					double x719 = x713 + x718;
					x558[x712] = x719;

				}
				x710 += 50;

			}
			for(int x726=0; x726 < 50; x726++) {
				double x727 = x533[x726];
				bool x728 = x727 < 0.0;
				double x731;
				if (x728) {
					x731 = 0.0;
				} else {
					double x729 = x558[x726];
					x731 = x729;
				}
				x541[x726] = x731;

			}
			// backpropagate +
			for(int x736=0; x736 < 50; x736++) {
				double x737 = x528[x736];
				double x738 = x541[x736];
				double x739 = x737 + x738;
				x528[x736] = x739;

			}
			for(int x743=0; x743 < 50; x743++) {
				double x744 = x54[x743];
				double x745 = x541[x743];
				double x746 = x744 + x745;
				x54[x743] = x746;

			}
			// add_cartesian
			int32_t x751 = 0;
			for(int x752=0; x752 < 50; x752++) {
				for(int x753=0; x753 < 320; x753++) {
					int32_t x754 = x751;
					int32_t x755 = x754 + x753;
					double x756 = x49[x755];
					double x757 = x489[x753];
					double x758 = x528[x752];
					double x759 = x757 * x758;
					double x760 = x756 + x759;
					x49[x755] = x760;

				}
				x751 += 320;

			}
			int32_t x767 = 0;
			for(int x768=0; x768 < 50; x768++) {
				for(int x769=0; x769 < 320; x769++) {
					double x770 = x505[x769];
					int32_t x771 = x767;
					int32_t x772 = x771 + x769;
					double x773 = x31[x772];
					double x774 = x528[x768];
					double x775 = x773 * x774;
					double x776 = x770 + x775;
					x505[x769] = x776;

				}
				x767 += 320;

			}
			double* x783 = (double*)myMalloc(320 * sizeof(double));
			for(int x784=0; x784 < 320; x784++) {
				double x785 = x490[x784];
				double x786 = x505[x784];
				double x787 = x785 * x786;
				x783[x784] = x787;

			}
			for(int x791=0; x791 < 320; x791++) {
				double x792 = x484[x791];
				double x793 = x783[x791];
				double x794 = x792 + x793;
				x484[x791] = x794;

			}
			for(int x798=0; x798 < 320; x798++) {
				double x799 = x417[x798];
				bool x800 = x799 < 0.0;
				double x803;
				if (x800) {
					x803 = 0.0;
				} else {
					double x801 = x484[x798];
					x803 = x801;
				}
				x467[x798] = x803;

			}
			for(int x807=0; x807 < 320; x807++) {
				int32_t x808 = x423[x807];
				double x809 = x467[x807];
				x412[x808] = x809;

			}
			int32_t x813 = 0;
			int32_t x814 = 0;
			for(int x815=0; x815 < 20; x815++) {
				int32_t x816 = 0;
				for(int x817=0; x817 < 8; x817++) {
					int32_t x818 = x816;
					int32_t x819 = x818;
					for(int x820=0; x820 < 8; x820++) {
						int32_t x821 = x813;
						double x822 = x412[x821];
						int32_t x823 = x819;
						int32_t x824 = x823;
						int32_t x825 = x814;
						int32_t x826 = x825;
						for(int x827=0; x827 < 10; x827++) {
							int32_t x828 = x824;
							int32_t x829 = x828;
							for(int x830=0; x830 < 5; x830++) {
								for(int x831=0; x831 < 5; x831++) {
									int32_t x832 = x829;
									int32_t x833 = x832 + x831;
									double x834 = x340[x833];
									int32_t x835 = x826;
									double x836 = x17[x835];
									double x837 = x822 * x836;
									double x838 = x834 + x837;
									x340[x833] = x838;
									double x840 = x26[x835];
									double x841 = x328[x833];
									double x842 = x822 * x841;
									double x843 = x840 + x842;
									x26[x835] = x843;
									x826 += 1;

								}
								x829 += 12;

							}
							x824 += 144;

						}
						x819 += 1;
						x813 += 1;

					}
					x816 += 12;

				}
				x814 += 250;

			}
			for(int x864=0; x864 < 1440; x864++) {
				double x865 = x272[x864];
				bool x866 = x865 < 0.0;
				double x869;
				if (x866) {
					x869 = 0.0;
				} else {
					double x867 = x340[x864];
					x869 = x867;
				}
				x323[x864] = x869;

			}
			for(int x873=0; x873 < 1440; x873++) {
				int32_t x874 = x278[x873];
				double x875 = x323[x873];
				x267[x874] = x875;

			}
			int32_t x879 = 0;
			int32_t x880 = 0;
			for(int x881=0; x881 < 10; x881++) {
				int32_t x882 = 0;
				for(int x883=0; x883 < 24; x883++) {
					int32_t x884 = x882;
					int32_t x885 = x884;
					for(int x886=0; x886 < 24; x886++) {
						int32_t x887 = x879;
						double x888 = x267[x887];
						int32_t x889 = x885;
						int32_t x890 = x889;
						int32_t x891 = x880;
						int32_t x892 = x891;
						for(int x893=0; x893 < 1; x893++) {
							int32_t x894 = x890;
							int32_t x895 = x894;
							for(int x896=0; x896 < 5; x896++) {
								for(int x897=0; x897 < 5; x897++) {
									int32_t x898 = x892;
									double x899 = x12[x898];
									int32_t x900 = x895;
									int32_t x901 = x900 + x897;
									double x902 = x182[x901];
									double x903 = x888 * x902;
									double x904 = x899 + x903;
									x12[x898] = x904;
									x892 += 1;

								}
								x895 += 28;

							}
							x890 += 784;

						}
						x885 += 1;
						x879 += 1;

					}
					x882 += 28;

				}
				x880 += 25;

			}
			double x925 = x195[0];
			x173 += x925;
			// Generate code for addMul
			for(int x928=0; x928 < 250; x928++) {
				double x929 = x3[x928];
				double x930 = x12[x928];
				double x931 = -5.0E-4 * x930;
				double x932 = x929 + x931;
				x3[x928] = x932;

			}
			for(int x936=0; x936 < 250; x936++) {
				double x937 = x12[x936];
				x12[x936] = 0.0;

			}
			// Generate code for addMul
			for(int x942=0; x942 < 5000; x942++) {
				double x943 = x17[x942];
				double x944 = x26[x942];
				double x945 = -5.0E-4 * x944;
				double x946 = x943 + x945;
				x17[x942] = x946;

			}
			for(int x950=0; x950 < 5000; x950++) {
				double x951 = x26[x950];
				x26[x950] = 0.0;

			}
			// Generate code for addMul
			for(int x956=0; x956 < 16000; x956++) {
				double x957 = x31[x956];
				double x958 = x49[x956];
				double x959 = -5.0E-4 * x958;
				double x960 = x957 + x959;
				x31[x956] = x960;

			}
			for(int x964=0; x964 < 16000; x964++) {
				double x965 = x49[x964];
				x49[x964] = 0.0;

			}
			// Generate code for addMul
			for(int x970=0; x970 < 50; x970++) {
				double x971 = x40[x970];
				double x972 = x54[x970];
				double x973 = -5.0E-4 * x972;
				double x974 = x971 + x973;
				x40[x970] = x974;

			}
			for(int x978=0; x978 < 50; x978++) {
				double x979 = x54[x978];
				x54[x978] = 0.0;

			}
			// Generate code for addMul
			for(int x984=0; x984 < 500; x984++) {
				double x985 = x59[x984];
				double x986 = x77[x984];
				double x987 = -5.0E-4 * x986;
				double x988 = x985 + x987;
				x59[x984] = x988;

			}
			for(int x992=0; x992 < 500; x992++) {
				double x993 = x77[x992];
				x77[x992] = 0.0;

			}
			// Generate code for addMul
			for(int x998=0; x998 < 10; x998++) {
				double x999 = x68[x998];
				double x1000 = x82[x998];
				double x1001 = -5.0E-4 * x1000;
				double x1002 = x999 + x1001;
				x68[x998] = x1002;

			}
			for(int x1006=0; x1006 < 10; x1006++) {
				double x1007 = x82[x1006];
				x82[x1006] = 0.0;

			}
			int32_t x1011 = x170;
			int32_t x1013 = x1011 % x1012;
			bool x1014 = x1013 == 0;
			if (x1014) {
				double x1019 = x173;
				double x1015 = (double)x1011;
				double x1016 = 100.0 * x1015;
				double x1018 = x1016 / x1017;
				double x1020 = x1019 / x1015;
				printf("Train epoch %d: [%d/%d (%.0f%%)]\tAverage Loss: %.6f\n",x166,x1011,x98,x1018,x1020);
				fflush(stdout);
			} else {
			}
			mallocAddr = (void*)x165;
			x179 += 784;

		}
		int32_t x1029 = x179;
		bool x1030 = x1029 == x93;
		if (x1030) {
		} else {
			printf("Data length doesn't match\n");
			exit(0);
		}
		gettimeofday(&end_1, NULL);
		timeval_subtract(&diff_1, &end_1, &begin_1);;
		int64_t x1038 = ((diff_1.tv_sec * 1000L) + (diff_1.tv_usec/1000L));
		int64_t x1040 = x1038 / x1039;
		printf("Training completed in %ldms (%ld ms/images)\n",x1038,x1040);
		printf("\nStart testing:\n");
		struct timeval begin_2, end_2, diff_2;
		gettimeofday(&begin_2, NULL);
		int32_t x1045 = 0;
		int32_t x1046 = x1045;
		x170 = x1046;
		double x1048 = 0.0;
		double x1049 = x1048;
		double x1050 = x1049;
		int32_t x1051 = 0;
		int32_t x1052 = 0;
		int32_t x1053 = x1052;
		int32_t x1054 = x1053;
		for(int x1055=0; x1055 < x135; x1055++) {
			int32_t x1056 = x1054;
			int32_t x1058 = x133[x1055];
			x170 += 1;
			double* x1060 = (double*)myMalloc(5760 * sizeof(double));
			for(int x1061=0; x1061 < 5760; x1061++) {
				x1060[x1061] = 0.0;

			}
			int32_t x1065 = 0;
			int32_t x1066 = 0;
			double* x1057 = x128+x1056;
			for(int x1067=0; x1067 < 10; x1067++) {
				int32_t x1068 = x1066;
				int32_t x1069 = x1068;
				int32_t x1070 = 0;
				int32_t x1071 = x1065;
				double* x1072 = x1060+x1071;
				for(int x1073=0; x1073 < 1; x1073++) {
					int32_t x1074 = x1070;
					int32_t x1076 = x1069;
					double* x1077 = x3+x1076;
					int32_t x1078 = 0;
					int32_t x1079 = 0;
					double* x1075 = x1057+x1074;
					for(int x1080=0; x1080 < 24; x1080++) {
						int32_t x1081 = x1079;
						int32_t x1082 = x1081;
						for(int x1083=0; x1083 < 24; x1083++) {
							int32_t x1084 = 0;
							int32_t x1085 = x1082;
							int32_t x1086 = x1085;
							double x1087 = 0.0;
							for(int x1088=0; x1088 < 5; x1088++) {
								int32_t x1089 = x1086;
								int32_t x1091 = x1084;
								double* x1092 = x1077+x1091;
								double* x1090 = x1075+x1089;
								for(int x1093=0; x1093 < 5; x1093++) {
									double x1094 = x1090[x1093];
									double x1095 = x1092[x1093];
									double x1096 = x1094 * x1095;
									x1087 += x1096;

								}
								x1084 += 5;
								x1086 += 28;

							}
							int32_t x1104 = x1078;
							double x1105 = x1072[x1104];
							double x1106 = x1087;
							double x1107 = x1105 + x1106;
							x1072[x1104] = x1107;
							x1078 += 1;
							x1082 += 1;

						}
						x1079 += 28;

					}
					x1069 += 25;
					x1070 += 784;

				}
				x1066 += 25;
				x1065 += 576;

			}
			double* x1124 = (double*)myMalloc(1440 * sizeof(double));
			for(int x1125=0; x1125 < 1440; x1125++) {
				x1124[x1125] = -1.0E10;

			}
			int32_t* x1129 = (int32_t*)myMalloc(1440 * sizeof(int32_t));
			int32_t x1130 = 0;
			int32_t x1131 = 0;
			for(int x1132=0; x1132 < 10; x1132++) {
				int32_t x1133 = x1130;
				int32_t x1134 = x1133;
				for(int x1135=0; x1135 < 12; x1135++) {
					for(int x1136=0; x1136 < 2; x1136++) {
						int32_t x1137 = x1134;
						int32_t x1138 = x1137;
						for(int x1139=0; x1139 < 12; x1139++) {
							int32_t x1140 = x1131;
							double x1141 = x1060[x1140];
							int32_t x1142 = x1138;
							double x1143 = x1124[x1142];
							bool x1144 = x1141 > x1143;
							if (x1144) {
								double x1145 = x1060[x1140];
								x1124[x1142] = x1145;
								x1129[x1142] = x1140;
							} else {
							}
							x1131 += 1;
							int32_t x1151 = x1131;
							double x1152 = x1060[x1151];
							double x1153 = x1124[x1142];
							bool x1154 = x1152 > x1153;
							if (x1154) {
								double x1155 = x1060[x1151];
								x1124[x1142] = x1155;
								x1129[x1142] = x1151;
							} else {
							}
							x1131 += 1;
							x1138 += 1;

						}

					}
					x1134 += 12;

				}
				x1130 += 144;

			}
			double* x1172 = (double*)myMalloc(1440 * sizeof(double));
			for(int x1173=0; x1173 < 1440; x1173++) {
				double x1174 = x1124[x1173];
				bool x1175 = x1174 < 0.0;
				if (x1175) {
					x1172[x1173] = 0.0;
				} else {
					double x1178 = x1124[x1173];
					x1172[x1173] = x1178;
				}

			}
			double* x1184 = (double*)myMalloc(1280 * sizeof(double));
			for(int x1185=0; x1185 < 1280; x1185++) {
				x1184[x1185] = 0.0;

			}
			int32_t x1189 = 0;
			int32_t x1190 = 0;
			for(int x1191=0; x1191 < 20; x1191++) {
				int32_t x1192 = x1190;
				int32_t x1193 = x1192;
				int32_t x1194 = 0;
				int32_t x1195 = x1189;
				double* x1196 = x1184+x1195;
				for(int x1197=0; x1197 < 10; x1197++) {
					int32_t x1198 = x1194;
					double* x1199 = x1172+x1198;
					int32_t x1200 = x1193;
					double* x1201 = x17+x1200;
					int32_t x1202 = 0;
					int32_t x1203 = 0;
					for(int x1204=0; x1204 < 8; x1204++) {
						int32_t x1205 = x1203;
						int32_t x1206 = x1205;
						for(int x1207=0; x1207 < 8; x1207++) {
							int32_t x1208 = 0;
							int32_t x1209 = x1206;
							int32_t x1210 = x1209;
							double x1211 = 0.0;
							for(int x1212=0; x1212 < 5; x1212++) {
								int32_t x1213 = x1210;
								double* x1214 = x1199+x1213;
								int32_t x1215 = x1208;
								double* x1216 = x1201+x1215;
								for(int x1217=0; x1217 < 5; x1217++) {
									double x1218 = x1214[x1217];
									double x1219 = x1216[x1217];
									double x1220 = x1218 * x1219;
									x1211 += x1220;

								}
								x1208 += 5;
								x1210 += 12;

							}
							int32_t x1228 = x1202;
							double x1229 = x1196[x1228];
							double x1230 = x1211;
							double x1231 = x1229 + x1230;
							x1196[x1228] = x1231;
							x1202 += 1;
							x1206 += 1;

						}
						x1203 += 12;

					}
					x1193 += 25;
					x1194 += 144;

				}
				x1190 += 250;
				x1189 += 64;

			}
			double* x1248 = (double*)myMalloc(320 * sizeof(double));
			for(int x1249=0; x1249 < 320; x1249++) {
				x1248[x1249] = -1.0E10;

			}
			int32_t* x1253 = (int32_t*)myMalloc(320 * sizeof(int32_t));
			int32_t x1254 = 0;
			int32_t x1255 = 0;
			for(int x1256=0; x1256 < 20; x1256++) {
				int32_t x1257 = x1254;
				int32_t x1258 = x1257;
				for(int x1259=0; x1259 < 4; x1259++) {
					for(int x1260=0; x1260 < 2; x1260++) {
						int32_t x1261 = x1258;
						int32_t x1262 = x1261;
						for(int x1263=0; x1263 < 4; x1263++) {
							int32_t x1264 = x1255;
							double x1265 = x1184[x1264];
							int32_t x1266 = x1262;
							double x1267 = x1248[x1266];
							bool x1268 = x1265 > x1267;
							if (x1268) {
								double x1269 = x1184[x1264];
								x1248[x1266] = x1269;
								x1253[x1266] = x1264;
							} else {
							}
							x1255 += 1;
							int32_t x1275 = x1255;
							double x1276 = x1184[x1275];
							double x1277 = x1248[x1266];
							bool x1278 = x1276 > x1277;
							if (x1278) {
								double x1279 = x1184[x1275];
								x1248[x1266] = x1279;
								x1253[x1266] = x1275;
							} else {
							}
							x1255 += 1;
							x1262 += 1;

						}

					}
					x1258 += 4;

				}
				x1254 += 16;

			}
			double* x1296 = (double*)myMalloc(320 * sizeof(double));
			for(int x1297=0; x1297 < 320; x1297++) {
				double x1298 = x1248[x1297];
				bool x1299 = x1298 < 0.0;
				if (x1299) {
					x1296[x1297] = 0.0;
				} else {
					double x1302 = x1248[x1297];
					x1296[x1297] = x1302;
				}

			}
			// dot WrappedArray(50, 320) - WrappedArray(320)
			int32_t x1309 = 0;
			double* x1310 = (double*)myMalloc(50 * sizeof(double));
			for(int x1311=0; x1311 < 50; x1311++) {
				double x1312 = 0.0;
				for(int x1313=0; x1313 < 320; x1313++) {
					int32_t x1314 = x1309;
					double x1315 = x31[x1314];
					double x1316 = x1296[x1313];
					double x1317 = x1315 * x1316;
					x1312 += x1317;
					x1309 += 1;

				}
				double x1322 = x1312;
				x1310[x1311] = x1322;

			}
			double* x1326 = (double*)myMalloc(50 * sizeof(double));
			for(int x1327=0; x1327 < 50; x1327++) {
				double x1328 = x1310[x1327];
				double x1329 = x40[x1327];
				double x1330 = x1328 + x1329;
				x1326[x1327] = x1330;

			}
			double* x1334 = (double*)myMalloc(50 * sizeof(double));
			for(int x1335=0; x1335 < 50; x1335++) {
				double x1336 = x1326[x1335];
				bool x1337 = x1336 < 0.0;
				if (x1337) {
					x1334[x1335] = 0.0;
				} else {
					double x1340 = x1326[x1335];
					x1334[x1335] = x1340;
				}

			}
			// dot WrappedArray(10, 50) - WrappedArray(50)
			int32_t x1347 = 0;
			double* x1348 = (double*)myMalloc(10 * sizeof(double));
			for(int x1349=0; x1349 < 10; x1349++) {
				double x1350 = 0.0;
				for(int x1351=0; x1351 < 50; x1351++) {
					int32_t x1352 = x1347;
					double x1353 = x59[x1352];
					double x1354 = x1334[x1351];
					double x1355 = x1353 * x1354;
					x1350 += x1355;
					x1347 += 1;

				}
				double x1360 = x1350;
				x1348[x1349] = x1360;

			}
			double* x1364 = (double*)myMalloc(10 * sizeof(double));
			for(int x1365=0; x1365 < 10; x1365++) {
				double x1366 = x1348[x1365];
				double x1367 = x68[x1365];
				double x1368 = x1366 + x1367;
				x1364[x1365] = x1368;

			}
			double x1372 = -1.0E10;
			for(int x1373=0; x1373 < 10; x1373++) {
				double x1374 = x1372;
				double x1375 = x1364[x1373];
				bool x1376 = x1375 > x1374;
				double x1377;
				if (x1376) {
					x1377 = x1375;
				} else {
					x1377 = x1374;
				}
				x1372 = x1377;

			}
			double x1381 = x1372;
			double x1382 = 0.0;
			for(int x1383=0; x1383 < 10; x1383++) {
				double x1384 = x1382;
				double x1385 = x1364[x1383];
				double x1386 = x1372;
				double x1387 = x1385 - x1386;
				double x1388 = exp(x1387);
				double x1389 = x1384 + x1388;
				x1382 = x1389;

			}
			double x1393 = x1382;
			double* x1396 = (double*)myMalloc(10 * sizeof(double));
			double x1394 = log(x1393);
			double x1395 = x1381 + x1394;
			for(int x1397=0; x1397 < 10; x1397++) {
				double x1398 = x1364[x1397];
				double x1399 = x1398 - x1395;
				x1396[x1397] = x1399;

			}
			double x1403 = x1396[x1058];
			double* x1405 = (double*)myMalloc(1 * sizeof(double));
			double x1404 = -1.0 * x1403;
			x1405[0] = x1404;
			double x1407 = x1405[0];
			x1050 += x1407;
			double x1409 = x1396[0];
			double x1410 = x1409;
			int32_t x1411 = 0;
			for(int x1413=1; x1413 < 10; x1413++) {
				double x1414 = x1396[x1413];
				double x1415 = x1410;
				bool x1416 = x1414 > x1415;
				if (x1416) {
					x1411 = x1413;
					double x1418 = x1396[x1413];
					x1410 = x1418;
				} else {
				}

			}
			int32_t x1424 = x1411;
			bool x1425 = x1424 == x1058;
			if (x1425) {
				x1051 += 1;
			} else {
			}
			x1054 += 784;

		}
		int32_t x1432 = x1054;
		bool x1433 = x1432 == x130;
		if (x1433) {
		} else {
			printf("Data length doesn't match\n");
			exit(0);
		}
		double x1439 = x1050;
		int32_t x1442 = x1051;
		gettimeofday(&end_2, NULL);
		timeval_subtract(&diff_2, &end_2, &begin_2);;
		int64_t x1448 = ((diff_2.tv_sec * 1000L) + (diff_2.tv_usec/1000L));
		double x1441 = x1439 / x1440;
		double x1443 = (double)x1442;
		double x1444 = 100.0 * x1443;
		double x1445 = x1444 / x1440;
		printf("Test set: Average loss: %.4f, Acurracy: %d/%d (%.0f) in %ldms\n",x1441,x1442,x135,x1445,x1448);
		printf("\n\n");

	}
}
/*****************************************
  End of C Generated Code                  
 *******************************************/

