
#include <fcntl.h>
#include <errno.h>
#include <err.h>
#include <sys/mman.h>
#include <sys/stat.h>
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
	srand(time(NULL)+0);
	double* x2 = (double*)myMalloc(2 * sizeof(double));
	for(int x4=0; x4 < 2; x4++) {
		double x5 = (double)rand()/RAND_MAX*2.0-1.0;
		x2[x4] = x5;

	}
	double* x9 = (double*)myMalloc(2 * sizeof(double));
	for(int x10=0; x10 < 2; x10++) {
		x9[x10] = 0.0;

	}
	double x14 = x2[0];
	bool x15 = x14 > 0.0;
	if (x15) {
		double* x47 = (double*)myMalloc(2 * sizeof(double));
		for(int x48=0; x48 < 2; x48++) {
			double x49 = x2[x48];
			double x50 = x49 + x49;
			x47[x48] = x50;

		}
		double* x54 = (double*)myMalloc(2 * sizeof(double));
		for(int x55=0; x55 < 2; x55++) {
			x54[x55] = 0.0;

		}
		double** x59 = (double**)myMalloc(2 * sizeof(double*));
		x59[0] = x47;
		x59[1] = x54;
		function<void(double**)> x16 = [&](double** x17) {
			double x20 = 0.0;
			double* x18 = x17[0];
			for(int x21=0; x21 < 2; x21++) {
				double x22 = x18[x21];
				x20 += x22;

			}
			double* x26 = (double*)myMalloc(1 * sizeof(double));
			double x27 = x20;
			x26[0] = x27;
			double* x29 = (double*)myMalloc(1 * sizeof(double));
			for(int x31=0; x31 < 1; x31++) {
				x29[x31] = 0.0;

			}
			for(int x35=0; x35 < 1; x35++) {
				x29[x35] = 1.0;

			}
			double* x19 = x17[1];
			for(int x39=0; x39 < 2; x39++) {
				double x41 = x29[0];
				double x40 = x19[x39];
				double x42 = x40 + x41;
				x19[x39] = x42;

			}
		};
		x16(x59);
		for(int x63=0; x63 < 2; x63++) {
			double x64 = x9[x63];
			double x65 = x54[x63];
			double x66 = x64 + x65;
			x9[x63] = x66;

		}
		for(int x70=0; x70 < 2; x70++) {
			double x71 = x9[x70];
			double x72 = x54[x70];
			double x73 = x71 + x72;
			x9[x70] = x73;

		}
	} else {
		double* x78 = (double*)myMalloc(2 * sizeof(double));
		for(int x79=0; x79 < 2; x79++) {
			double x80 = x2[x79];
			double x81 = x80 * x80;
			x78[x79] = x81;

		}
		double* x85 = (double*)myMalloc(2 * sizeof(double));
		for(int x86=0; x86 < 2; x86++) {
			x85[x86] = 0.0;

		}
		double** x90 = (double**)myMalloc(2 * sizeof(double*));
		x90[0] = x78;
		x90[1] = x85;
		function<void(double**)> x16 = [&](double** x17) {
			double x20 = 0.0;
			double* x18 = x17[0];
			for(int x21=0; x21 < 2; x21++) {
				double x22 = x18[x21];
				x20 += x22;

			}
			double* x26 = (double*)myMalloc(1 * sizeof(double));
			double x27 = x20;
			x26[0] = x27;
			double* x29 = (double*)myMalloc(1 * sizeof(double));
			for(int x31=0; x31 < 1; x31++) {
				x29[x31] = 0.0;

			}
			for(int x35=0; x35 < 1; x35++) {
				x29[x35] = 1.0;

			}
			double* x19 = x17[1];
			for(int x39=0; x39 < 2; x39++) {
				double x41 = x29[0];
				double x40 = x19[x39];
				double x42 = x40 + x41;
				x19[x39] = x42;

			}
		};
		x16(x90);
		double* x94 = (double*)myMalloc(2 * sizeof(double));
		for(int x95=0; x95 < 2; x95++) {
			double x96 = x2[x95];
			double x97 = x85[x95];
			double x98 = x96 * x97;
			x94[x95] = x98;

		}
		for(int x102=0; x102 < 2; x102++) {
			double x103 = x9[x102];
			double x104 = x94[x102];
			double x105 = x103 + x104;
			x9[x102] = x105;

		}
		double* x109 = (double*)myMalloc(2 * sizeof(double));
		for(int x110=0; x110 < 2; x110++) {
			double x111 = x2[x110];
			double x112 = x85[x110];
			double x113 = x111 * x112;
			x109[x110] = x113;

		}
		for(int x117=0; x117 < 2; x117++) {
			double x118 = x9[x117];
			double x119 = x109[x117];
			double x120 = x118 + x119;
			x9[x117] = x120;

		}
	}
	double* x126 = (double*)myMalloc(2 * sizeof(double));
	for(int x127=0; x127 < 2; x127++) {
		x126[x127] = 0.0;

	}
	double* x131 = (double*)myMalloc(2 * sizeof(double));
	for(int x132=0; x132 < 2; x132++) {
		double x133 = x2[x132];
		double x134 = x133 + x133;
		x131[x132] = x134;

	}
	double* x138 = (double*)myMalloc(2 * sizeof(double));
	for(int x139=0; x139 < 2; x139++) {
		x138[x139] = 0.0;

	}
	double x143 = 0.0;
	for(int x144=0; x144 < 2; x144++) {
		double x145 = x131[x144];
		x143 += x145;

	}
	double* x149 = (double*)myMalloc(1 * sizeof(double));
	double x150 = x143;
	x149[0] = x150;
	double* x152 = (double*)myMalloc(1 * sizeof(double));
	for(int x153=0; x153 < 1; x153++) {
		x152[x153] = 0.0;

	}
	for(int x157=0; x157 < 1; x157++) {
		x152[x157] = 1.0;

	}
	for(int x161=0; x161 < 2; x161++) {
		double x162 = x138[x161];
		double x163 = x152[0];
		double x164 = x162 + x163;
		x138[x161] = x164;

	}
	for(int x168=0; x168 < 2; x168++) {
		double x169 = x126[x168];
		double x170 = x138[x168];
		double x171 = x169 + x170;
		x126[x168] = x171;

	}
	for(int x175=0; x175 < 2; x175++) {
		double x176 = x126[x175];
		double x177 = x138[x175];
		double x178 = x176 + x177;
		x126[x175] = x178;

	}
	double* x182 = (double*)myMalloc(2 * sizeof(double));
	for(int x183=0; x183 < 2; x183++) {
		x182[x183] = 0.0;

	}
	double* x187 = (double*)myMalloc(2 * sizeof(double));
	for(int x188=0; x188 < 2; x188++) {
		double x189 = x2[x188];
		double x190 = x189 * x189;
		x187[x188] = x190;

	}
	double* x194 = (double*)myMalloc(2 * sizeof(double));
	for(int x195=0; x195 < 2; x195++) {
		x194[x195] = 0.0;

	}
	double x199 = 0.0;
	for(int x200=0; x200 < 2; x200++) {
		double x201 = x187[x200];
		x199 += x201;

	}
	double* x205 = (double*)myMalloc(1 * sizeof(double));
	double x206 = x199;
	x205[0] = x206;
	double* x208 = (double*)myMalloc(1 * sizeof(double));
	for(int x209=0; x209 < 1; x209++) {
		x208[x209] = 0.0;

	}
	for(int x213=0; x213 < 1; x213++) {
		x208[x213] = 1.0;

	}
	for(int x217=0; x217 < 2; x217++) {
		double x218 = x194[x217];
		double x219 = x208[0];
		double x220 = x218 + x219;
		x194[x217] = x220;

	}
	double* x224 = (double*)myMalloc(2 * sizeof(double));
	for(int x225=0; x225 < 2; x225++) {
		double x226 = x2[x225];
		double x227 = x194[x225];
		double x228 = x226 * x227;
		x224[x225] = x228;

	}
	for(int x232=0; x232 < 2; x232++) {
		double x233 = x182[x232];
		double x234 = x224[x232];
		double x235 = x233 + x234;
		x182[x232] = x235;

	}
	double* x239 = (double*)myMalloc(2 * sizeof(double));
	for(int x240=0; x240 < 2; x240++) {
		double x241 = x2[x240];
		double x242 = x194[x240];
		double x243 = x241 * x242;
		x239[x240] = x243;

	}
	for(int x247=0; x247 < 2; x247++) {
		double x248 = x182[x247];
		double x249 = x239[x247];
		double x250 = x248 + x249;
		x182[x247] = x250;

	}
	if (x15) {
		double x254 = 0.0;
		for(int x255=0; x255 < 2; x255++) {
			double x256 = x9[x255];
			double x257 = x126[x255];
			double x258 = x256 - x257;
			bool x259 = x258 < -1.0E-6;
			bool x260 = x258 > 1.0E-6;
			bool x261 = x259 || x260;
			if (x261) {
				x254 += 1.0;
			} else {
			}

		}
		double x267 = x254;
		bool x269 = x267 == 0.0;
		if (x269) {
		} else {
			printf("ERROR: %s not equal in some data\n","");
		}
	} else {
		double x274 = 0.0;
		for(int x275=0; x275 < 2; x275++) {
			double x276 = x9[x275];
			double x277 = x182[x275];
			double x278 = x276 - x277;
			bool x279 = x278 < -1.0E-6;
			bool x280 = x278 > 1.0E-6;
			bool x281 = x279 || x280;
			if (x281) {
				x274 += 1.0;
			} else {
			}

		}
		double x287 = x274;
		bool x289 = x287 == 0.0;
		if (x289) {
		} else {
			printf("ERROR: %s not equal in some data\n","");
		}
	}
}
/*****************************************
  End of C Generated Code                  
 *******************************************/

