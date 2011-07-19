#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "mm_malloc.h"

void x264_sort_ssse3(int32_t n, int32_t *dst, int32_t *aux);
void x264_cri(int n, int32_t *data, int32_t *aux);

typedef void (sort_f)(int32_t, int32_t *, int32_t *);


static inline uint64_t read_time(void)
{
    if(sizeof(long)==8)
    {
        uint64_t a, d;
        asm volatile( "rdtsc\n\t" : "=a" (a), "=d" (d) );
        return (d << 32) | (a & 0xffffffff);
    } else {
        uint64_t l;
        asm volatile( "rdtsc\n\t" : "=A" (l) );
        return l;
    }
}
 
#define NOP_CYCLES 65 // time measured by an empty timer on Core2
 
#define START_TIMER \
uint64_t tend;\
uint64_t tstart= read_time();

#define STOP_TIMER(id) {\
tend= read_time();\
{\
    static uint64_t tsum=0;\
    static int tcount=0;\
    static int tskip_count=0;\
    if(tskip_count<2)\
        tskip_count++;\
    else{\
    if(tcount<2 || tend - tstart < 8*tsum/tcount){\
        tsum+= tend - tstart;\
        tcount++;\
    }else\
        tskip_count++;\
    if(((tcount+tskip_count) & (tcount+tskip_count-1)) == 0)\
        printf("%"PRIu64" dezicycles in %s, %d runs, %d skips\n", tsum*10/tcount-NOP_CYCLES*10, id, tcount, tskip_count);\
}}}

#define STOP_TIMER_SUM(id) {\
tend= read_time();\
{\
    static uint64_t tsum=0;\
    static uint64_t tother=0;\
    static uint64_t tend0=0;\
    static int tcount=0;\
    tsum += tend - tstart;\
    if(tcount)\
        tother += tstart - tend0;\
    tend0 = tend;\
    tcount++;\
    if((tcount & (tcount-1)) == 0 && tcount > 4)\
        printf("%"PRIu64"/%"PRIu64" cycles %s, %d runs\n", tsum-NOP_CYCLES*tcount, tother+tsum-NOP_CYCLES*tcount, id, tcount);\
}}

void sub(int n, sort_f f, char *name) {
    int32_t  *d, *aux;
    int i, tests;

    d = _mm_malloc(n * sizeof (int32_t), 64);
    aux = _mm_malloc(n * sizeof (int32_t), 64);

    srandomdev(); // Init random pool
    for (i = 0; i < n; i++)
        d[i] = random() % 10;

#ifdef DEBUG
    for (i = 0; i < 32; i++)
        printf("%i%c", d[i], (i+1) % 4? ' ' : '\n');
    printf("--\n");
#endif

    for (tests = 4; tests; tests--) {
        START_TIMER;
        (*f)(n, d, aux);
        STOP_TIMER("cri");
    }

#ifdef DEBUG
#define TESTME 32
    for (i = 0; i < TESTME; i++)
        d[i] = i % 16; //random() % 10;
    for (i = 0; i < TESTME; i++)
        printf("%X%c", d[i], (i+1) % 4 ? ' ' : '\n');
    printf("\n");
    x264_cri(TESTME, d);
    for (i = 0; i < TESTME; i++)
        printf("%X%c", d[i], (i+1) % 4 ? ' ' : '\n');
#endif

    _mm_free(d);
    _mm_free(aux);
}

int main(int argc, char *argv[]) {

    sub((1<<15), x264_cri, "ssse3");

    return (0);
}
