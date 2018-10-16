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
function<void(int32_t,float**)> x490 = [&](int32_t x491,float** x492) {
float** x494 = x492;
float* x495 = x494[0];
float* x496 = x494[1];
float* x497 = x494[2];
float* x498 = x494[3];
int32_t x493 = x491;
bool x499 = x493 < 20;
if (x499) {
int32_t x500 = x493 * 520;
float* x501 = x146+x500;
float* x502 = x168+x500;
// dot: WrappedArray(20, 26), List(26, 50)
float* x504 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x501,26,x14,50,0,x504,50);
float* x506 = (float*)myMalloc(1000 * sizeof(float));;
for(int x507=0; x507 < 1000; x507++) {
x506[x507] = 0.0f;

}
// dot: List(20, 50), List(50, 50)
float* x512 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x497,50,x28,50,0,x512,50);
float* x514 = (float*)myMalloc(1000 * sizeof(float));;
for(int x515=0; x515 < 1000; x515++) {
x514[x515] = 0.0f;

}
float* x519 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x520 = 0;
int32_t x521 = 0;
int32_t x522 = 0;
for(int x523=0; x523 < 20; x523++) {
int32_t x524 = x521;
int32_t x525 = x522;
int32_t x526 = x520;
int32_t x527 = x526;
int32_t x528 = x524;
int32_t x529 = x525;
for(int x530=0; x530 < 50; x530++) {
int32_t x531 = x527;
int32_t x532 = x528;
float x533 = x504[x532];
int32_t x534 = x529;
float x535 = x512[x534];
float x536 = x533 + x535;
x519[x531] = x536;
x527 += 1;
x528 += 1;
x529 += 1;

}
x520 += 50;
x521 += 50;
x522 += 50;

}
float* x548 = (float*)myMalloc(1000 * sizeof(float));;
for(int x549=0; x549 < 1000; x549++) {
x548[x549] = 0.0f;

}
float* x553 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x554 = 0;
int32_t x555 = 0;
int32_t x556 = 0;
for(int x557=0; x557 < 20; x557++) {
int32_t x558 = x555;
int32_t x559 = x556;
int32_t x560 = x554;
int32_t x561 = x560;
int32_t x562 = x558;
int32_t x563 = x559;
for(int x564=0; x564 < 50; x564++) {
int32_t x565 = x561;
int32_t x566 = x562;
float x567 = x519[x566];
int32_t x568 = x563;
float x569 = x42[x568];
float x570 = x567 + x569;
x553[x565] = x570;
x561 += 1;
x562 += 1;
x563 += 1;

}
x554 += 50;
x555 += 50;

}
float* x581 = (float*)myMalloc(1000 * sizeof(float));;
for(int x582=0; x582 < 1000; x582++) {
x581[x582] = 0.0f;

}
float* x586 = (float*)myMalloc(1000 * sizeof(float));;
for(int x587=0; x587 < 1000; x587++) {
float x588 = x553[x587];
double x589 = (double)x588;
double x590 = tanh(x589);
float x591 = (float)x590;
x586[x587] = x591;

}
float* x595 = (float*)myMalloc(1000 * sizeof(float));;
for(int x596=0; x596 < 1000; x596++) {
x595[x596] = 0.0f;

}
// dot: List(20, 50), List(50, 26)
float* x601 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x586,50,x53,26,0,x601,26);
float* x603 = (float*)myMalloc(520 * sizeof(float));;
for(int x604=0; x604 < 520; x604++) {
x603[x604] = 0.0f;

}
float* x608 = (float*)myMalloc(520 * sizeof(float));;
int32_t x609 = 0;
int32_t x610 = 0;
int32_t x611 = 0;
for(int x612=0; x612 < 20; x612++) {
int32_t x613 = x610;
int32_t x614 = x611;
int32_t x615 = x609;
int32_t x616 = x615;
int32_t x617 = x613;
int32_t x618 = x614;
for(int x619=0; x619 < 26; x619++) {
int32_t x620 = x616;
int32_t x621 = x617;
float x622 = x601[x621];
int32_t x623 = x618;
float x624 = x66[x623];
float x625 = x622 + x624;
x608[x620] = x625;
x616 += 1;
x617 += 1;
x618 += 1;

}
x609 += 26;
x610 += 26;

}
float* x636 = (float*)myMalloc(520 * sizeof(float));;
for(int x637=0; x637 < 520; x637++) {
x636[x637] = 0.0f;

}
int* x641 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x642=0; x642 < 20; x642++) {
int32_t x643 = x642 * 20;
int32_t x644 = x493 + x643;
int32_t x645 = x118[x644];
x641[x642] = x645;

}
float* x649 = (float*)myMalloc(20 * sizeof(float));;
int32_t x650 = 0;
for(int x651=0; x651 < 20; x651++) {
float x652 = -3.4028235E38f;
for(int x653=0; x653 < 26; x653++) {
int32_t x654 = x650;
float x655 = x608[x654];
float x656 = x652;
bool x657 = x655 > x656;
if (x657) {
float x658 = x608[x654];
x652 = x658;
} else {
}
x650 += 1;

}
float x665 = x652;
x649[x651] = x665;

}
float* x669 = (float*)myMalloc(520 * sizeof(float));;
for(int x670=0; x670 < 520; x670++) {
x669[x670] = 0.0f;

}
int32_t x674 = 0;
for(int x675=0; x675 < 20; x675++) {
for(int x676=0; x676 < 26; x676++) {
int32_t x677 = x674;
float x678 = x608[x677];
float x679 = x649[x675];
float x680 = x678 - x679;
double x681 = (double)x680;
double x682 = exp(x681);
float x683 = (float)x682;
x669[x677] = x683;
x674 += 1;

}

}
float* x690 = (float*)myMalloc(20 * sizeof(float));;
for(int x691=0; x691 < 20; x691++) {
x690[x691] = 0.0f;

}
for(int x695=0; x695 < 20; x695++) {
int32_t x696 = x695;
int32_t x697 = x695 * 26;
int32_t x698 = x697;
for(int x699=0; x699 < 26; x699++) {
int32_t x700 = x696;
float x701 = x690[x700];
int32_t x702 = x698;
float x703 = x669[x702];
float x704 = x701 + x703;
x690[x700] = x704;
x698 += 1;

}

}
x674 = 0;
for(int x712=0; x712 < 20; x712++) {
float x713 = x649[x712];
float x714 = x690[x712];
double x715 = (double)x714;
double x716 = log(x715);
float x717 = (float)x716;
float x718 = x713 + x717;
for(int x719=0; x719 < 26; x719++) {
int32_t x720 = x674;
float x721 = x608[x720];
float x722 = x721 - x718;
x669[x720] = x722;
x674 += 1;

}

}
float* x729 = (float*)myMalloc(520 * sizeof(float));;
for(int x730=0; x730 < 520; x730++) {
x729[x730] = 0.0f;

}
float* x734 = (float*)myMalloc(20 * sizeof(float));;
int32_t x735 = 0;
for(int x736=0; x736 < 20; x736++) {
int32_t x737 = x735;
int32_t x738 = x641[x736];
int32_t x739 = x737 + x738;
float x740 = x669[x739];
float x741 = -1.0f * x740;
x734[x736] = x741;
x735 += 26;

}
float* x746 = (float*)myMalloc(20 * sizeof(float));;
for(int x747=0; x747 < 20; x747++) {
x746[x747] = 0.0f;

}
float x751 = 0.0f;
for(int x752=0; x752 < 20; x752++) {
float x753 = x751;
float x754 = x734[x752];
float x755 = x753 + x754;
x751 = x755;

}
float x759 = x751;
float* x760 = (float*)myMalloc(1 * sizeof(float));;
x760[0] = x759;
float* x762 = (float*)myMalloc(1 * sizeof(float));;
for(int x763=0; x763 < 1; x763++) {
x762[x763] = 0.0f;

}
float* x767 = (float*)myMalloc(1 * sizeof(float));;
int32_t x768 = 0;
int32_t x769 = 0;
int32_t x770 = 0;
int32_t x771 = x768;
int32_t x772 = x769;
float x773 = x495[x772];
int32_t x774 = x770;
float x775 = x760[x774];
float x776 = x773 + x775;
x767[x771] = x776;
x768 += 1;
float* x779 = (float*)myMalloc(1 * sizeof(float));;
for(int x780=0; x780 < 1; x780++) {
x779[x780] = 0.0f;

}
float** x785 = (float**)myMalloc(4 * sizeof(float*));;
x785[0] = x767;
x785[1] = x779;
x785[2] = x586;
x785[3] = x595;
int32_t x792 = 0;
int32_t x793 = 0;
int32_t x794 = 0;
int32_t x795 = x792;
int32_t x798 = x793;
int32_t x800 = x794;
x794 += 1;
int32_t x819 = 0;
float* x832 = (float*)myMalloc(20 * sizeof(float));;
int32_t x853 = 0;
int32_t x873 = 0;
int32_t x874 = 0;
int32_t x875 = 0;
int32_t x921 = 0;
int32_t x922 = 0;
int32_t x923 = 0;
int32_t x956 = 0;
int32_t x957 = 0;
int32_t x958 = 0;
int32_t x784 = x493 + 1;
x490(x784,x785);
float x796 = x496[x795];
float x797 = x495[x795];
float x799 = x760[x798];
float x801 = x779[x800];
float x802 = x796 + x801;
x496[x795] = x802;
float x804 = x762[x798];
float x805 = x495[x795];
float x806 = x760[x798];
float x807 = x779[x800];
float x808 = x804 + x807;
x762[x798] = x808;
// += tensor of dim 0
float x812 = x762[0];
for(int x813=0; x813 < 20; x813++) {
float x814 = x746[x813];
float x815 = x814 + x812;
x746[x813] = x815;

}
for(int x820=0; x820 < 20; x820++) {
int32_t x821 = x819;
int32_t x822 = x641[x820];
int32_t x823 = x821 + x822;
float x824 = x729[x823];
float x825 = x746[x820];
float x826 = -1.0f * x825;
float x827 = x824 + x826;
x729[x823] = x827;
x819 += 26;

}
for(int x833=0; x833 < 20; x833++) {
x832[x833] = 0.0f;

}
for(int x837=0; x837 < 20; x837++) {
int32_t x838 = x837;
int32_t x839 = x837 * 26;
int32_t x840 = x839;
for(int x841=0; x841 < 26; x841++) {
int32_t x842 = x838;
float x843 = x832[x842];
int32_t x844 = x840;
float x845 = x729[x844];
float x846 = x843 + x845;
x832[x842] = x846;
x840 += 1;

}

}
for(int x854=0; x854 < 20; x854++) {
for(int x855=0; x855 < 26; x855++) {
int32_t x856 = x853;
float x857 = x636[x856];
float x858 = x729[x856];
float x859 = x669[x856];
float x863 = x832[x854];
double x860 = (double)x859;
double x861 = exp(x860);
float x862 = (float)x861;
float x864 = x862 * x863;
float x865 = x858 - x864;
float x866 = x857 + x865;
x636[x856] = x866;
x853 += 1;

}

}
for(int x876=0; x876 < 20; x876++) {
int32_t x877 = x873;
int32_t x878 = x874;
int32_t x879 = x875;
int32_t x880 = x877;
int32_t x881 = x878;
int32_t x882 = x879;
for(int x883=0; x883 < 26; x883++) {
int32_t x884 = x880;
float x885 = x603[x884];
float x886 = x601[x884];
int32_t x887 = x881;
float x888 = x66[x887];
int32_t x889 = x882;
float x890 = x636[x889];
float x891 = x885 + x890;
x603[x884] = x891;
float x893 = x72[x887];
float x894 = x601[x884];
float x895 = x66[x887];
float x896 = x636[x889];
float x897 = x893 + x896;
x72[x887] = x897;
x882 += 1;
x880 += 1;
x881 += 1;

}
x875 += 26;
x873 += 26;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x603,26,x53,26,1,x595,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x586,50,x603,26,1,x61,26);
for(int x910=0; x910 < 1000; x910++) {
float x911 = x581[x910];
float x912 = x586[x910];
float x915 = x595[x910];
float x913 = x912 * x912;
float x914 = 1.0f - x913;
float x916 = x914 * x915;
float x917 = x911 + x916;
x581[x910] = x917;

}
for(int x924=0; x924 < 20; x924++) {
int32_t x925 = x921;
int32_t x926 = x922;
int32_t x927 = x923;
int32_t x928 = x925;
int32_t x929 = x926;
int32_t x930 = x927;
for(int x931=0; x931 < 50; x931++) {
int32_t x932 = x928;
float x933 = x548[x932];
float x934 = x519[x932];
int32_t x935 = x929;
float x936 = x42[x935];
int32_t x937 = x930;
float x938 = x581[x937];
float x939 = x933 + x938;
x548[x932] = x939;
float x941 = x48[x935];
float x942 = x519[x932];
float x943 = x42[x935];
float x944 = x581[x937];
float x945 = x941 + x944;
x48[x935] = x945;
x930 += 1;
x928 += 1;
x929 += 1;

}
x923 += 50;
x921 += 50;

}
for(int x959=0; x959 < 20; x959++) {
int32_t x960 = x956;
int32_t x961 = x957;
int32_t x962 = x958;
int32_t x963 = x960;
int32_t x964 = x961;
int32_t x965 = x962;
for(int x966=0; x966 < 50; x966++) {
int32_t x967 = x963;
float x968 = x506[x967];
float x969 = x504[x967];
int32_t x970 = x964;
float x971 = x512[x970];
int32_t x972 = x965;
float x973 = x548[x972];
float x974 = x968 + x973;
x506[x967] = x974;
float x976 = x514[x970];
float x977 = x504[x967];
float x978 = x512[x970];
float x979 = x548[x972];
float x980 = x976 + x979;
x514[x970] = x980;
x965 += 1;
x963 += 1;
x964 += 1;

}
x958 += 50;
x956 += 50;
x957 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x514,50,x28,50,1,x498,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x497,50,x514,50,1,x37,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x506,50,x14,50,1,x502,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x501,26,x506,50,1,x23,50);
} else {
float x997 = 0.0f;
float x998 = x997;
float x999 = x495[0];
float x1000 = x998 + x999;
x997 = x1000;
float x1002 = x997;
float* x1003 = (float*)myMalloc(1 * sizeof(float));;
x1003[0] = x1002;
float* x1005 = (float*)myMalloc(1 * sizeof(float));;
for(int x1006=0; x1006 < 1; x1006++) {
x1005[x1006] = 0.0f;

}
float x1010 = x1005[0];
x1005[0] = 1.0f;
float x1012 = x1003[0];
x141[0] = x1012;
// += tensor of dim 0
float x1015 = x1005[0];
float x1016 = x496[0];
float x1017 = x1016 + x1015;
x496[0] = x1017;
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
float** x1259 = (float**)myMalloc(4 * sizeof(float*));;
x1259[0] = x173;
x1259[1] = x178;
x1259[2] = x183;
x1259[3] = x189;
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
// dot: WrappedArray(20, 26), List(26, 50)
float* x208 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,26,1,x205,26,x14,50,0,x208,50);
float* x210 = (float*)myMalloc(1000 * sizeof(float));;
for(int x211=0; x211 < 1000; x211++) {
x210[x211] = 0.0f;

}
// dot: WrappedArray(20, 50), List(50, 50)
float* x216 = (float*)myMalloc(1000 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,50,50,1,x201,50,x28,50,0,x216,50);
float* x218 = (float*)myMalloc(1000 * sizeof(float));;
for(int x219=0; x219 < 1000; x219++) {
x218[x219] = 0.0f;

}
float* x223 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x224 = 0;
int32_t x225 = 0;
int32_t x226 = 0;
for(int x227=0; x227 < 20; x227++) {
int32_t x228 = x225;
int32_t x229 = x226;
int32_t x230 = x224;
int32_t x231 = x230;
int32_t x232 = x228;
int32_t x233 = x229;
for(int x234=0; x234 < 50; x234++) {
int32_t x235 = x231;
int32_t x236 = x232;
float x237 = x208[x236];
int32_t x238 = x233;
float x239 = x216[x238];
float x240 = x237 + x239;
x223[x235] = x240;
x231 += 1;
x232 += 1;
x233 += 1;

}
x224 += 50;
x225 += 50;
x226 += 50;

}
float* x252 = (float*)myMalloc(1000 * sizeof(float));;
for(int x253=0; x253 < 1000; x253++) {
x252[x253] = 0.0f;

}
float* x257 = (float*)myMalloc(1000 * sizeof(float));;
int32_t x258 = 0;
int32_t x259 = 0;
int32_t x260 = 0;
for(int x261=0; x261 < 20; x261++) {
int32_t x262 = x259;
int32_t x263 = x260;
int32_t x264 = x258;
int32_t x265 = x264;
int32_t x266 = x262;
int32_t x267 = x263;
for(int x268=0; x268 < 50; x268++) {
int32_t x269 = x265;
int32_t x270 = x266;
float x271 = x223[x270];
int32_t x272 = x267;
float x273 = x42[x272];
float x274 = x271 + x273;
x257[x269] = x274;
x265 += 1;
x266 += 1;
x267 += 1;

}
x258 += 50;
x259 += 50;

}
float* x285 = (float*)myMalloc(1000 * sizeof(float));;
for(int x286=0; x286 < 1000; x286++) {
x285[x286] = 0.0f;

}
float* x290 = (float*)myMalloc(1000 * sizeof(float));;
for(int x291=0; x291 < 1000; x291++) {
float x292 = x257[x291];
double x293 = (double)x292;
double x294 = tanh(x293);
float x295 = (float)x294;
x290[x291] = x295;

}
float* x299 = (float*)myMalloc(1000 * sizeof(float));;
for(int x300=0; x300 < 1000; x300++) {
x299[x300] = 0.0f;

}
// dot: List(20, 50), List(50, 26)
float* x305 = (float*)myMalloc(520 * sizeof(float));;
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 20,26,50,1,x290,50,x53,26,0,x305,26);
float* x307 = (float*)myMalloc(520 * sizeof(float));;
for(int x309=0; x309 < 520; x309++) {
x307[x309] = 0.0f;

}
float* x313 = (float*)myMalloc(520 * sizeof(float));;
int32_t x314 = 0;
int32_t x315 = 0;
int32_t x316 = 0;
for(int x317=0; x317 < 20; x317++) {
int32_t x318 = x315;
int32_t x319 = x316;
int32_t x320 = x314;
int32_t x321 = x320;
int32_t x322 = x318;
int32_t x323 = x319;
for(int x324=0; x324 < 26; x324++) {
int32_t x325 = x321;
int32_t x326 = x322;
float x327 = x305[x326];
int32_t x328 = x323;
float x329 = x66[x328];
float x330 = x327 + x329;
x313[x325] = x330;
x321 += 1;
x322 += 1;
x323 += 1;

}
x314 += 26;
x315 += 26;

}
float* x341 = (float*)myMalloc(520 * sizeof(float));;
for(int x342=0; x342 < 520; x342++) {
x341[x342] = 0.0f;

}
int* x346 = (int32_t*)myMalloc(20 * sizeof(int32_t));;
for(int x347=0; x347 < 20; x347++) {
int32_t x348 = x347 * 20;
int32_t x349 = x197 + x348;
int32_t x350 = x118[x349];
x346[x347] = x350;

}
float* x354 = (float*)myMalloc(20 * sizeof(float));;
int32_t x355 = 0;
for(int x356=0; x356 < 20; x356++) {
float x357 = -3.4028235E38f;
for(int x358=0; x358 < 26; x358++) {
int32_t x359 = x355;
float x360 = x313[x359];
float x361 = x357;
bool x362 = x360 > x361;
if (x362) {
float x363 = x313[x359];
x357 = x363;
} else {
}
x355 += 1;

}
float x370 = x357;
x354[x356] = x370;

}
float* x374 = (float*)myMalloc(520 * sizeof(float));;
for(int x375=0; x375 < 520; x375++) {
x374[x375] = 0.0f;

}
int32_t x379 = 0;
for(int x380=0; x380 < 20; x380++) {
for(int x381=0; x381 < 26; x381++) {
int32_t x382 = x379;
float x383 = x313[x382];
float x384 = x354[x380];
float x385 = x383 - x384;
double x386 = (double)x385;
double x387 = exp(x386);
float x388 = (float)x387;
x374[x382] = x388;
x379 += 1;

}

}
float* x395 = (float*)myMalloc(20 * sizeof(float));;
for(int x396=0; x396 < 20; x396++) {
x395[x396] = 0.0f;

}
for(int x400=0; x400 < 20; x400++) {
int32_t x401 = x400;
int32_t x402 = x400 * 26;
int32_t x403 = x402;
for(int x404=0; x404 < 26; x404++) {
int32_t x405 = x401;
float x406 = x395[x405];
int32_t x407 = x403;
float x408 = x374[x407];
float x409 = x406 + x408;
x395[x405] = x409;
x403 += 1;

}

}
x379 = 0;
for(int x417=0; x417 < 20; x417++) {
float x418 = x354[x417];
float x419 = x395[x417];
double x420 = (double)x419;
double x421 = log(x420);
float x422 = (float)x421;
float x423 = x418 + x422;
for(int x424=0; x424 < 26; x424++) {
int32_t x425 = x379;
float x426 = x313[x425];
float x427 = x426 - x423;
x374[x425] = x427;
x379 += 1;

}

}
float* x434 = (float*)myMalloc(520 * sizeof(float));;
for(int x435=0; x435 < 520; x435++) {
x434[x435] = 0.0f;

}
float* x439 = (float*)myMalloc(20 * sizeof(float));;
int32_t x440 = 0;
for(int x441=0; x441 < 20; x441++) {
int32_t x442 = x440;
int32_t x443 = x346[x441];
int32_t x444 = x442 + x443;
float x445 = x374[x444];
float x446 = -1.0f * x445;
x439[x441] = x446;
x440 += 26;

}
float* x451 = (float*)myMalloc(20 * sizeof(float));;
for(int x452=0; x452 < 20; x452++) {
x451[x452] = 0.0f;

}
float x456 = 0.0f;
for(int x457=0; x457 < 20; x457++) {
float x458 = x456;
float x459 = x439[x457];
float x460 = x458 + x459;
x456 = x460;

}
float x464 = x456;
float* x465 = (float*)myMalloc(1 * sizeof(float));;
x465[0] = x464;
float* x467 = (float*)myMalloc(1 * sizeof(float));;
for(int x468=0; x468 < 1; x468++) {
x467[x468] = 0.0f;

}
float* x472 = (float*)myMalloc(1 * sizeof(float));;
int32_t x473 = 0;
int32_t x474 = 0;
int32_t x475 = 0;
int32_t x476 = x473;
int32_t x477 = x474;
float x478 = x199[x477];
int32_t x479 = x475;
float x480 = x465[x479];
float x481 = x478 + x480;
x472[x476] = x481;
x473 += 1;
float* x484 = (float*)myMalloc(1 * sizeof(float));;
for(int x485=0; x485 < 1; x485++) {
x484[x485] = 0.0f;

}
float** x1022 = (float**)myMalloc(4 * sizeof(float*));;
x1022[0] = x472;
x1022[1] = x484;
x1022[2] = x290;
x1022[3] = x299;
int32_t x489 = x197 + 1;
x490(x489,x1022);
int32_t x1029 = 0;
int32_t x1030 = 0;
int32_t x1031 = 0;
int32_t x1032 = x1029;
float x1033 = x200[x1032];
float x1034 = x199[x1032];
int32_t x1035 = x1030;
float x1036 = x465[x1035];
int32_t x1037 = x1031;
float x1038 = x484[x1037];
float x1039 = x1033 + x1038;
x200[x1032] = x1039;
float x1041 = x467[x1035];
float x1042 = x199[x1032];
float x1043 = x465[x1035];
float x1044 = x484[x1037];
float x1045 = x1041 + x1044;
x467[x1035] = x1045;
x1031 += 1;
// += tensor of dim 0
float x1049 = x467[0];
for(int x1050=0; x1050 < 20; x1050++) {
float x1051 = x451[x1050];
float x1052 = x1051 + x1049;
x451[x1050] = x1052;

}
int32_t x1056 = 0;
for(int x1057=0; x1057 < 20; x1057++) {
int32_t x1058 = x1056;
int32_t x1059 = x346[x1057];
int32_t x1060 = x1058 + x1059;
float x1061 = x434[x1060];
float x1062 = x451[x1057];
float x1063 = -1.0f * x1062;
float x1064 = x1061 + x1063;
x434[x1060] = x1064;
x1056 += 26;

}
float* x1069 = (float*)myMalloc(20 * sizeof(float));;
for(int x1070=0; x1070 < 20; x1070++) {
x1069[x1070] = 0.0f;

}
for(int x1074=0; x1074 < 20; x1074++) {
int32_t x1075 = x1074;
int32_t x1076 = x1074 * 26;
int32_t x1077 = x1076;
for(int x1078=0; x1078 < 26; x1078++) {
int32_t x1079 = x1075;
float x1080 = x1069[x1079];
int32_t x1081 = x1077;
float x1082 = x434[x1081];
float x1083 = x1080 + x1082;
x1069[x1079] = x1083;
x1077 += 1;

}

}
int32_t x1090 = 0;
for(int x1091=0; x1091 < 20; x1091++) {
for(int x1092=0; x1092 < 26; x1092++) {
int32_t x1093 = x1090;
float x1094 = x341[x1093];
float x1095 = x434[x1093];
float x1096 = x374[x1093];
float x1100 = x1069[x1091];
double x1097 = (double)x1096;
double x1098 = exp(x1097);
float x1099 = (float)x1098;
float x1101 = x1099 * x1100;
float x1102 = x1095 - x1101;
float x1103 = x1094 + x1102;
x341[x1093] = x1103;
x1090 += 1;

}

}
int32_t x1110 = 0;
int32_t x1111 = 0;
int32_t x1112 = 0;
for(int x1113=0; x1113 < 20; x1113++) {
int32_t x1114 = x1110;
int32_t x1115 = x1111;
int32_t x1116 = x1112;
int32_t x1117 = x1114;
int32_t x1118 = x1115;
int32_t x1119 = x1116;
for(int x1120=0; x1120 < 26; x1120++) {
int32_t x1121 = x1117;
float x1122 = x307[x1121];
float x1123 = x305[x1121];
int32_t x1124 = x1118;
float x1125 = x66[x1124];
int32_t x1126 = x1119;
float x1127 = x341[x1126];
float x1128 = x1122 + x1127;
x307[x1121] = x1128;
float x1130 = x72[x1124];
float x1131 = x305[x1121];
float x1132 = x66[x1124];
float x1133 = x341[x1126];
float x1134 = x1130 + x1133;
x72[x1124] = x1134;
x1119 += 1;
x1117 += 1;
x1118 += 1;

}
x1112 += 26;
x1110 += 26;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,26,1,x307,26,x53,26,1,x299,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,26,20,1,x290,50,x307,26,1,x61,26);
for(int x1147=0; x1147 < 1000; x1147++) {
float x1148 = x285[x1147];
float x1149 = x290[x1147];
float x1152 = x299[x1147];
float x1150 = x1149 * x1149;
float x1151 = 1.0f - x1150;
float x1153 = x1151 * x1152;
float x1154 = x1148 + x1153;
x285[x1147] = x1154;

}
int32_t x1158 = 0;
int32_t x1159 = 0;
int32_t x1160 = 0;
for(int x1161=0; x1161 < 20; x1161++) {
int32_t x1162 = x1158;
int32_t x1163 = x1159;
int32_t x1164 = x1160;
int32_t x1165 = x1162;
int32_t x1166 = x1163;
int32_t x1167 = x1164;
for(int x1168=0; x1168 < 50; x1168++) {
int32_t x1169 = x1165;
float x1170 = x252[x1169];
float x1171 = x223[x1169];
int32_t x1172 = x1166;
float x1173 = x42[x1172];
int32_t x1174 = x1167;
float x1175 = x285[x1174];
float x1176 = x1170 + x1175;
x252[x1169] = x1176;
float x1178 = x48[x1172];
float x1179 = x223[x1169];
float x1180 = x42[x1172];
float x1181 = x285[x1174];
float x1182 = x1178 + x1181;
x48[x1172] = x1182;
x1167 += 1;
x1165 += 1;
x1166 += 1;

}
x1160 += 50;
x1158 += 50;

}
int32_t x1193 = 0;
int32_t x1194 = 0;
int32_t x1195 = 0;
for(int x1196=0; x1196 < 20; x1196++) {
int32_t x1197 = x1193;
int32_t x1198 = x1194;
int32_t x1199 = x1195;
int32_t x1200 = x1197;
int32_t x1201 = x1198;
int32_t x1202 = x1199;
for(int x1203=0; x1203 < 50; x1203++) {
int32_t x1204 = x1200;
float x1205 = x210[x1204];
float x1206 = x208[x1204];
int32_t x1207 = x1201;
float x1208 = x216[x1207];
int32_t x1209 = x1202;
float x1210 = x252[x1209];
float x1211 = x1205 + x1210;
x210[x1204] = x1211;
float x1213 = x218[x1207];
float x1214 = x208[x1204];
float x1215 = x216[x1207];
float x1216 = x252[x1209];
float x1217 = x1213 + x1216;
x218[x1207] = x1217;
x1202 += 1;
x1200 += 1;
x1201 += 1;

}
x1195 += 50;
x1193 += 50;
x1194 += 50;

}
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,50,50,1,x218,50,x28,50,1,x202,50);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 50,50,20,1,x201,50,x218,50,1,x37,50);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 20,26,50,1,x210,50,x14,50,1,x206,26);
cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 26,50,20,1,x205,26,x210,50,1,x23,50);
} else {
float x1234 = 0.0f;
float x1235 = x1234;
float x1236 = x199[0];
float x1237 = x1235 + x1236;
x1234 = x1237;
float x1239 = x1234;
float* x1240 = (float*)myMalloc(1 * sizeof(float));;
x1240[0] = x1239;
float* x1242 = (float*)myMalloc(1 * sizeof(float));;
for(int x1243=0; x1243 < 1; x1243++) {
x1242[x1243] = 0.0f;

}
float x1247 = x1242[0];
x1242[0] = 1.0f;
float x1249 = x1240[0];
x141[0] = x1249;
// += tensor of dim 0
float x1252 = x1242[0];
float x1253 = x200[0];
float x1254 = x1253 + x1252;
x200[0] = x1254;
}
};
x194(0,x1259);
float x1266 = x141[0];
int32_t x1267 = x108 % 100;
bool x1268 = x1267 == 0;
if (x1268) {
printf("iter %d, loss %f\n",x108,x1266);
int32_t x1270 = x108 / 100;
double x1271 = (double)x1266;
x102[x1270] = x1271;
} else {
}
for(int x1275=0; x1275 < 26; x1275++) {
float x1276 = x72[x1275];
bool x1277 = x1276 > 5.0f;
if (x1277) {
x72[x1275] = 5.0f;
} else {
}
float x1281 = x72[x1275];
bool x1282 = x1281 < -5.0f;
if (x1282) {
x72[x1275] = -5.0f;
} else {
}

}
float* x1288 = (float*)myMalloc(26 * sizeof(float));;
int32_t x1289 = 0;
int32_t x1290 = 0;
int32_t x1291 = 0;
for(int x1292=0; x1292 < 26; x1292++) {
int32_t x1293 = x1289;
int32_t x1294 = x1290;
float x1295 = x72[x1294];
int32_t x1296 = x1291;
float x1297 = x72[x1296];
float x1298 = x1295 * x1297;
x1288[x1293] = x1298;
x1289 += 1;
x1290 += 1;
x1291 += 1;

}
for(int x1305=0; x1305 < 26; x1305++) {
float x1306 = x77[x1305];
float x1307 = x1288[x1305];
float x1308 = x1306 + x1307;
x77[x1305] = x1308;

}
float* x1312 = (float*)myMalloc(26 * sizeof(float));;
for(int x1313=0; x1313 < 26; x1313++) {
float x1314 = x72[x1313];
float x1315 = x1314 * 0.1f;
x1312[x1313] = x1315;

}
float* x1319 = (float*)myMalloc(26 * sizeof(float));;
for(int x1320=0; x1320 < 26; x1320++) {
float x1321 = x77[x1320];
float x1322 = x1321 + 1.0E-8f;
x1319[x1320] = x1322;

}
float* x1326 = (float*)myMalloc(26 * sizeof(float));;
for(int x1327=0; x1327 < 26; x1327++) {
float x1328 = x1319[x1327];
double x1329 = (double)x1328;
double x1330 = sqrt(x1329);
float x1331 = (float)x1330;
x1326[x1327] = x1331;

}
float* x1335 = (float*)myMalloc(26 * sizeof(float));;
int32_t x1336 = 0;
int32_t x1337 = 0;
int32_t x1338 = 0;
for(int x1339=0; x1339 < 26; x1339++) {
int32_t x1340 = x1336;
int32_t x1341 = x1337;
float x1342 = x1312[x1341];
int32_t x1343 = x1338;
float x1344 = x1326[x1343];
float x1345 = x1342 / x1344;
x1335[x1340] = x1345;
x1336 += 1;
x1337 += 1;
x1338 += 1;

}
for(int x1352=0; x1352 < 26; x1352++) {
float x1353 = x66[x1352];
float x1354 = x1335[x1352];
float x1355 = x1353 - x1354;
x66[x1352] = x1355;

}
for(int x1359=0; x1359 < 26; x1359++) {
float x1360 = x72[x1359];
x72[x1359] = 0.0f;

}
for(int x1364=0; x1364 < 1300; x1364++) {
float x1365 = x61[x1364];
bool x1366 = x1365 > 5.0f;
if (x1366) {
x61[x1364] = 5.0f;
} else {
}
float x1370 = x61[x1364];
bool x1371 = x1370 < -5.0f;
if (x1371) {
x61[x1364] = -5.0f;
} else {
}

}
float* x1377 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x1378 = 0;
int32_t x1379 = 0;
int32_t x1380 = 0;
for(int x1381=0; x1381 < 50; x1381++) {
int32_t x1382 = x1379;
int32_t x1383 = x1380;
int32_t x1384 = x1378;
int32_t x1385 = x1384;
int32_t x1386 = x1382;
int32_t x1387 = x1383;
for(int x1388=0; x1388 < 26; x1388++) {
int32_t x1389 = x1385;
int32_t x1390 = x1386;
float x1391 = x61[x1390];
int32_t x1392 = x1387;
float x1393 = x61[x1392];
float x1394 = x1391 * x1393;
x1377[x1389] = x1394;
x1385 += 1;
x1386 += 1;
x1387 += 1;

}
x1378 += 26;
x1379 += 26;
x1380 += 26;

}
for(int x1406=0; x1406 < 1300; x1406++) {
float x1407 = x82[x1406];
float x1408 = x1377[x1406];
float x1409 = x1407 + x1408;
x82[x1406] = x1409;

}
float* x1413 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1414=0; x1414 < 1300; x1414++) {
float x1415 = x61[x1414];
float x1416 = x1415 * 0.1f;
x1413[x1414] = x1416;

}
float* x1420 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1421=0; x1421 < 1300; x1421++) {
float x1422 = x82[x1421];
float x1423 = x1422 + 1.0E-8f;
x1420[x1421] = x1423;

}
float* x1427 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1428=0; x1428 < 1300; x1428++) {
float x1429 = x1420[x1428];
double x1430 = (double)x1429;
double x1431 = sqrt(x1430);
float x1432 = (float)x1431;
x1427[x1428] = x1432;

}
float* x1436 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x1437 = 0;
int32_t x1438 = 0;
int32_t x1439 = 0;
for(int x1440=0; x1440 < 50; x1440++) {
int32_t x1441 = x1438;
int32_t x1442 = x1439;
int32_t x1443 = x1437;
int32_t x1444 = x1443;
int32_t x1445 = x1441;
int32_t x1446 = x1442;
for(int x1447=0; x1447 < 26; x1447++) {
int32_t x1448 = x1444;
int32_t x1449 = x1445;
float x1450 = x1413[x1449];
int32_t x1451 = x1446;
float x1452 = x1427[x1451];
float x1453 = x1450 / x1452;
x1436[x1448] = x1453;
x1444 += 1;
x1445 += 1;
x1446 += 1;

}
x1437 += 26;
x1438 += 26;
x1439 += 26;

}
for(int x1465=0; x1465 < 1300; x1465++) {
float x1466 = x53[x1465];
float x1467 = x1436[x1465];
float x1468 = x1466 - x1467;
x53[x1465] = x1468;

}
for(int x1472=0; x1472 < 1300; x1472++) {
float x1473 = x61[x1472];
x61[x1472] = 0.0f;

}
for(int x1477=0; x1477 < 2500; x1477++) {
float x1478 = x37[x1477];
bool x1479 = x1478 > 5.0f;
if (x1479) {
x37[x1477] = 5.0f;
} else {
}
float x1483 = x37[x1477];
bool x1484 = x1483 < -5.0f;
if (x1484) {
x37[x1477] = -5.0f;
} else {
}

}
float* x1490 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x1491 = 0;
int32_t x1492 = 0;
int32_t x1493 = 0;
for(int x1494=0; x1494 < 50; x1494++) {
int32_t x1495 = x1492;
int32_t x1496 = x1493;
int32_t x1497 = x1491;
int32_t x1498 = x1497;
int32_t x1499 = x1495;
int32_t x1500 = x1496;
for(int x1501=0; x1501 < 50; x1501++) {
int32_t x1502 = x1498;
int32_t x1503 = x1499;
float x1504 = x37[x1503];
int32_t x1505 = x1500;
float x1506 = x37[x1505];
float x1507 = x1504 * x1506;
x1490[x1502] = x1507;
x1498 += 1;
x1499 += 1;
x1500 += 1;

}
x1491 += 50;
x1492 += 50;
x1493 += 50;

}
for(int x1519=0; x1519 < 2500; x1519++) {
float x1520 = x87[x1519];
float x1521 = x1490[x1519];
float x1522 = x1520 + x1521;
x87[x1519] = x1522;

}
float* x1526 = (float*)myMalloc(2500 * sizeof(float));;
for(int x1527=0; x1527 < 2500; x1527++) {
float x1528 = x37[x1527];
float x1529 = x1528 * 0.1f;
x1526[x1527] = x1529;

}
float* x1533 = (float*)myMalloc(2500 * sizeof(float));;
for(int x1534=0; x1534 < 2500; x1534++) {
float x1535 = x87[x1534];
float x1536 = x1535 + 1.0E-8f;
x1533[x1534] = x1536;

}
float* x1540 = (float*)myMalloc(2500 * sizeof(float));;
for(int x1541=0; x1541 < 2500; x1541++) {
float x1542 = x1533[x1541];
double x1543 = (double)x1542;
double x1544 = sqrt(x1543);
float x1545 = (float)x1544;
x1540[x1541] = x1545;

}
float* x1549 = (float*)myMalloc(2500 * sizeof(float));;
int32_t x1550 = 0;
int32_t x1551 = 0;
int32_t x1552 = 0;
for(int x1553=0; x1553 < 50; x1553++) {
int32_t x1554 = x1551;
int32_t x1555 = x1552;
int32_t x1556 = x1550;
int32_t x1557 = x1556;
int32_t x1558 = x1554;
int32_t x1559 = x1555;
for(int x1560=0; x1560 < 50; x1560++) {
int32_t x1561 = x1557;
int32_t x1562 = x1558;
float x1563 = x1526[x1562];
int32_t x1564 = x1559;
float x1565 = x1540[x1564];
float x1566 = x1563 / x1565;
x1549[x1561] = x1566;
x1557 += 1;
x1558 += 1;
x1559 += 1;

}
x1550 += 50;
x1551 += 50;
x1552 += 50;

}
for(int x1578=0; x1578 < 2500; x1578++) {
float x1579 = x28[x1578];
float x1580 = x1549[x1578];
float x1581 = x1579 - x1580;
x28[x1578] = x1581;

}
for(int x1585=0; x1585 < 2500; x1585++) {
float x1586 = x37[x1585];
x37[x1585] = 0.0f;

}
for(int x1590=0; x1590 < 50; x1590++) {
float x1591 = x48[x1590];
bool x1592 = x1591 > 5.0f;
if (x1592) {
x48[x1590] = 5.0f;
} else {
}
float x1596 = x48[x1590];
bool x1597 = x1596 < -5.0f;
if (x1597) {
x48[x1590] = -5.0f;
} else {
}

}
float* x1603 = (float*)myMalloc(50 * sizeof(float));;
int32_t x1604 = 0;
int32_t x1605 = 0;
int32_t x1606 = 0;
for(int x1607=0; x1607 < 50; x1607++) {
int32_t x1608 = x1604;
int32_t x1609 = x1605;
float x1610 = x48[x1609];
int32_t x1611 = x1606;
float x1612 = x48[x1611];
float x1613 = x1610 * x1612;
x1603[x1608] = x1613;
x1604 += 1;
x1605 += 1;
x1606 += 1;

}
for(int x1620=0; x1620 < 50; x1620++) {
float x1621 = x92[x1620];
float x1622 = x1603[x1620];
float x1623 = x1621 + x1622;
x92[x1620] = x1623;

}
float* x1627 = (float*)myMalloc(50 * sizeof(float));;
for(int x1628=0; x1628 < 50; x1628++) {
float x1629 = x48[x1628];
float x1630 = x1629 * 0.1f;
x1627[x1628] = x1630;

}
float* x1634 = (float*)myMalloc(50 * sizeof(float));;
for(int x1635=0; x1635 < 50; x1635++) {
float x1636 = x92[x1635];
float x1637 = x1636 + 1.0E-8f;
x1634[x1635] = x1637;

}
float* x1641 = (float*)myMalloc(50 * sizeof(float));;
for(int x1642=0; x1642 < 50; x1642++) {
float x1643 = x1634[x1642];
double x1644 = (double)x1643;
double x1645 = sqrt(x1644);
float x1646 = (float)x1645;
x1641[x1642] = x1646;

}
float* x1650 = (float*)myMalloc(50 * sizeof(float));;
int32_t x1651 = 0;
int32_t x1652 = 0;
int32_t x1653 = 0;
for(int x1654=0; x1654 < 50; x1654++) {
int32_t x1655 = x1651;
int32_t x1656 = x1652;
float x1657 = x1627[x1656];
int32_t x1658 = x1653;
float x1659 = x1641[x1658];
float x1660 = x1657 / x1659;
x1650[x1655] = x1660;
x1651 += 1;
x1652 += 1;
x1653 += 1;

}
for(int x1667=0; x1667 < 50; x1667++) {
float x1668 = x42[x1667];
float x1669 = x1650[x1667];
float x1670 = x1668 - x1669;
x42[x1667] = x1670;

}
for(int x1674=0; x1674 < 50; x1674++) {
float x1675 = x48[x1674];
x48[x1674] = 0.0f;

}
for(int x1679=0; x1679 < 1300; x1679++) {
float x1680 = x23[x1679];
bool x1681 = x1680 > 5.0f;
if (x1681) {
x23[x1679] = 5.0f;
} else {
}
float x1685 = x23[x1679];
bool x1686 = x1685 < -5.0f;
if (x1686) {
x23[x1679] = -5.0f;
} else {
}

}
float* x1692 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x1693 = 0;
int32_t x1694 = 0;
int32_t x1695 = 0;
for(int x1696=0; x1696 < 26; x1696++) {
int32_t x1697 = x1694;
int32_t x1698 = x1695;
int32_t x1699 = x1693;
int32_t x1700 = x1699;
int32_t x1701 = x1697;
int32_t x1702 = x1698;
for(int x1703=0; x1703 < 50; x1703++) {
int32_t x1704 = x1700;
int32_t x1705 = x1701;
float x1706 = x23[x1705];
int32_t x1707 = x1702;
float x1708 = x23[x1707];
float x1709 = x1706 * x1708;
x1692[x1704] = x1709;
x1700 += 1;
x1701 += 1;
x1702 += 1;

}
x1693 += 50;
x1694 += 50;
x1695 += 50;

}
for(int x1721=0; x1721 < 1300; x1721++) {
float x1722 = x97[x1721];
float x1723 = x1692[x1721];
float x1724 = x1722 + x1723;
x97[x1721] = x1724;

}
float* x1728 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1729=0; x1729 < 1300; x1729++) {
float x1730 = x23[x1729];
float x1731 = x1730 * 0.1f;
x1728[x1729] = x1731;

}
float* x1735 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1736=0; x1736 < 1300; x1736++) {
float x1737 = x97[x1736];
float x1738 = x1737 + 1.0E-8f;
x1735[x1736] = x1738;

}
float* x1742 = (float*)myMalloc(1300 * sizeof(float));;
for(int x1743=0; x1743 < 1300; x1743++) {
float x1744 = x1735[x1743];
double x1745 = (double)x1744;
double x1746 = sqrt(x1745);
float x1747 = (float)x1746;
x1742[x1743] = x1747;

}
float* x1751 = (float*)myMalloc(1300 * sizeof(float));;
int32_t x1752 = 0;
int32_t x1753 = 0;
int32_t x1754 = 0;
for(int x1755=0; x1755 < 26; x1755++) {
int32_t x1756 = x1753;
int32_t x1757 = x1754;
int32_t x1758 = x1752;
int32_t x1759 = x1758;
int32_t x1760 = x1756;
int32_t x1761 = x1757;
for(int x1762=0; x1762 < 50; x1762++) {
int32_t x1763 = x1759;
int32_t x1764 = x1760;
float x1765 = x1728[x1764];
int32_t x1766 = x1761;
float x1767 = x1742[x1766];
float x1768 = x1765 / x1767;
x1751[x1763] = x1768;
x1759 += 1;
x1760 += 1;
x1761 += 1;

}
x1752 += 50;
x1753 += 50;
x1754 += 50;

}
for(int x1780=0; x1780 < 1300; x1780++) {
float x1781 = x14[x1780];
float x1782 = x1751[x1780];
float x1783 = x1781 - x1782;
x14[x1780] = x1783;

}
for(int x1787=0; x1787 < 1300; x1787++) {
float x1788 = x23[x1787];
x23[x1787] = 0.0f;

}
mallocAddr = (void*)x104;

}
double x1795 = ((double)clock() / CLOCKS_PER_SEC);
int64_t x1798 = (long)fopen(x0, "w");
fprintf((FILE *)x1798, "unit: %s\n", "100 iteration");
for(int x1801=0; x1801 < 51; x1801++) {
double x1802 = x102[x1801];
fprintf((FILE *)x1798, "%lf\n", x1802);

}
double x1796 = x103 - x1;
double x1797 = x1795 - x103;
fprintf((FILE *)x1798, "run time: %lf %lf\n", x1796, x1797);
fclose((FILE*)x1798);
}
/*****************************************
  End of C Generated Code                  
*******************************************/

