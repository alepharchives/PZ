//  PF compressor, test methods
//  Copyright (C) 2008-2011  Alejo Sanchez www.ologan.com
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Affero General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Affero General Public License for more details.
//
//  You should have received a copy of the GNU Affero General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>.

//  This file performs tests on the functions of the compressor

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>

typedef __v4si v4si; // For clarity

typedef union {
  int32_t s[4];
  v4si  v;
} v4si_u;


// SSE2 test interfaces
void pz_column_sort_4si_sse2(v4si *v);
void pz_transpose_4si_sse2(v4si *v);
void pz_register_sort_4si_sse2(v4si *v);
void pz_bitonic_sort_4si_sse2(v4si *a, v4si *b);
void pz_bitonic_sort_2x_4si_sse2(v4si *a, v4si *b, v4si *c, v4si *d);
void pz_merge_2l_2x4si_sse2(v4si *s1, v4si *s2);
void pz_merge_parallel_2x2l_2x8si_sse2(v4si *v);
void pz_bitonic_merge_2x16si_sse2(v4si *v);
void pz_merge_2seq_sse2(v4si * restrict dst, v4si * restrict src1,
        v4si * restrict src2, int len);
void pz_register_seq_sort_4si_sse2(v4si *v, int len);

v4si_u *v, *a; // Buffers vector and aux

// Make 4 vectors of 4 32bit signed integers and fill with random
void vec_random(int size) {
    int32_t  *p = (int32_t *) v;
    int      i;

    for (i = 0; i < size; i++)
        *p++ = random() % size;

}

// Check if sequences are sorted
int check_sort(int32_t *a, int sequences, int sequence_length) {
    int i, j, e;

    return -1;
    for (i = 0, j = 0; i < sequences; i++)
        for (j = 0; j < (sequence_length - 1); j++)
            if (a[i * sequence_length + j] > a[i * sequence_length + j + 1])
                break;

    if (i != sequences && j != (sequence_length - 1)) {

        // Print error sequence
        printf("error on sequence %d at position %d\n", i, j);
        e = j; // Store error position
        for (j = 0; j < sequence_length; j++)
            printf("% 3d%s", a[i * sequence_length + j],
                    j == e? "* " : " ");
        printf("\n");

        return -1;

    }

    return 0;

}

// Test sorting columns of 4 vectors of 4 32bit signed integers
int test_column_sort_4() {
    int      i, j;

    vec_random(16); // Add some random numbers

    pz_column_sort_4si_sse2(&v[0].v);

    // Check
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 4; j++) {
            if (v[i].s[j] > v[i + 1].s[j]) {
                printf("test_column_sort_4: error at v[%d].s[%d]\n", i, j);
                printf("  % 2d - % 2d\n", v[i].s[j], v[i + 1].s[j]);
                // Display sorted
                for (i = 0; i < 4; i++)
                    printf("% 2d % 2d % 2d % 2d o\n",
                            v[i].s[0], v[i].s[1], v[i].s[2], v[i].s[3]);
                return (-1);
            }
        }
    }

    return 0;

}

// Test transposing 4 vectors of 4 32bit signed integers
int test_register_sort_4() {

    vec_random(16); // Add some random numbers

    pz_register_sort_4si_sse2(&v[0].v); // Sort 0-3

    return check_sort((int32_t *) v, 2, 4);

}

// Test merge sort of 2 vectors of 4 32bit signed integers
int test_bitonic_sort() {

    vec_random(8); // Add some random numbers

    pz_register_sort_4si_sse2(&v[0].v); // Sort 0-3
    pz_bitonic_sort_4si_sse2(&v[0].v, &v[1].v); // Sort first two together

    return check_sort((int32_t *) v, 1, 8);

}

// Test parallel merge sort of 2 pairs of 2 vectors of 4 32bit signed integers
int test_bitonic_sort_2x() {

    vec_random(16); // Add some random numbers

    pz_register_sort_4si_sse2(&v[0].v); // Sort 0-3
    pz_bitonic_sort_4si_sse2(&v[0].v, &v[1].v); // Sort 0-1
    pz_bitonic_sort_4si_sse2(&v[2].v, &v[3].v); // Sort 2-3

    return check_sort((int32_t *) v, 2, 8);

}

// Test parallel merge sort of 2 pairs of 2 vectors of 4 32bit signed integers
int test_merge_2_pairs() {
    int32_t  a[16], *pa;
    int      i, j;

    vec_random(32); // Add some random numbers

    pz_register_sort_4si_sse2(&v[0].v); // Sort 0-3
    pz_bitonic_sort_4si_sse2(&v[0].v, &v[1].v); // Sort 0-1
    pz_bitonic_sort_4si_sse2(&v[2].v, &v[3].v); // Sort 2-3
    pz_merge_2l_2x4si_sse2((v4si *) &v[0], (v4si *) &v[2]); // Merge 2 seq

    // Move to sequential array
    for (i = 0, pa = a; i < 4; i++)
        for (j = 0; j < 4; j++)
            *pa++ = v[i].s[j];

    // Check
    for (i = 0; i < 15; i++)
        if (a[i] > a[i + 1]) {
            printf("test_merge_2_pairs: error at position %d: %d > %d\n",
                    i, a[i], a[i + 1]);
            break;
        }

    if (i != 15) {

        for (i = 0; i < 4; i++)
            for (j = 0; j < 4; j++)
                printf("% 3d ", v[i].s[j]);
        printf("\n");

        return -1;

    }

    return 0;

}

// Test parallel merge sort of 2 list of 2x2
//    v0   v1   v2   v3      v4   v5   v6   v7
//   aaaa aaaa bbbb bbbb || cccc cccc dddd dddd    (input)
//   xxxx xxxx xxxx xxxx || yyyy yyyy yyyy yyyy    (result)
//
int test_merge_parallel_2list_2pairs() {
    int32_t  a[32], *pa;
    int      i, j;

    vec_random(64); // Add some random numbers

    pz_register_sort_4si_sse2(&v[0].v); // Sort 0-3
    pz_register_sort_4si_sse2(&v[4].v); // Sort 4-7
    pz_bitonic_sort_4si_sse2(&v[0].v, &v[1].v); // Sort 0-1
    pz_bitonic_sort_4si_sse2(&v[2].v, &v[3].v); // Sort 2-3
    pz_bitonic_sort_2x_4si_sse2(&v[0].v, &v[1].v, &v[2].v, &v[3].v);
    pz_bitonic_sort_2x_4si_sse2(&v[4].v, &v[5].v, &v[6].v, &v[7].v);
    pz_merge_2l_2x4si_sse2((v4si *) &v[0], (v4si *) &v[2]); // Merge 0-3
    pz_merge_parallel_2x2l_2x8si_sse2((v4si *) v); // Merge v0-3 v4-7

    // Move to sequential array
    for (i = 0, pa = a; i < 8; i++)
        for (j = 0; j < 4; j++)
            *pa++ = v[i].s[j];

    // Check
    for (i = 0; i < 31; i++)
        if (a[i] > a[i + 1] && i != 15) {
            printf("test_merge_parallel_2list_2pairs: error at position %d:"
                    " %d > %d\n", i, a[i], a[i + 1]);
            break;
        }

    if (i != 31) {

        for (i = 0; i < 8; i++)
            for (j = 0; j < 4; j++)
                printf("% 3d ", v[i].s[j]);
        printf("-\n");

        return -1;

    }

    return 0;

}

// Test merge of 2 lists of 16 32bit signed integers (16x16si network)
int test_merge_16x16() {
    int32_t  a[32], *pa;
    int      i, j;

    vec_random(64); // Add some random numbers

    pz_register_sort_4si_sse2(&v[0].v); // Sort 0-3
    pz_register_sort_4si_sse2(&v[4].v); // Sort 4-7
    pz_bitonic_sort_4si_sse2(&v[0].v, &v[1].v); // Sort 0-1
    pz_bitonic_sort_4si_sse2(&v[2].v, &v[3].v); // Sort 2-3
    pz_bitonic_sort_4si_sse2(&v[4].v, &v[5].v); // Sort 4-5
    pz_bitonic_sort_4si_sse2(&v[6].v, &v[7].v); // Sort 6-7
    pz_merge_2l_2x4si_sse2((v4si *) &v[0], (v4si *) &v[2]); // Merge 0-3
    pz_merge_2l_2x4si_sse2((v4si *) &v[4], (v4si *) &v[6]); // Merge 4-7

    pz_bitonic_merge_2x16si_sse2(&v[0].v); // Test

    // Move to sequential array
    for (i = 0, pa = a; i < 8; i++)
        for (j = 0; j < 4; j++)
            *pa++ = v[i].s[j];

    // Check
    for (i = 0; i < 31; i++)
        if (a[i] > a[i + 1]) {
            printf("test_merge_2_pairs: error at position %d: %d > %d\n",
                    i, a[i], a[i + 1]);
            break;
        }

    if (i != 31) {

        for (i = 0; i < 8; i++)
            for (j = 0; j < 4; j++)
                printf("% 3d ", v[i].s[j]);
        printf("\n");

        return -1;

    }

    return 0;

}

// Test merge of 2 lists of 16 32bit signed integers (16x16si network)
int test_merge_2seq() {
    int32_t  *pa;
    int      i;

    vec_random(32); // Add some random numbers

    pz_register_sort_4si_sse2(&v[0].v); // Sort 0-3
    pz_register_sort_4si_sse2(&v[4].v); // Sort 4-7
    pz_bitonic_sort_4si_sse2(&v[0].v, &v[1].v); // Sort 0-1
    pz_bitonic_sort_4si_sse2(&v[2].v, &v[3].v); // Sort 2-3
    pz_bitonic_sort_4si_sse2(&v[4].v, &v[5].v); // Sort 4-5
    pz_bitonic_sort_4si_sse2(&v[6].v, &v[7].v); // Sort 6-7
    pz_merge_2l_2x4si_sse2((v4si *) &v[0], (v4si *) &v[2]); // Merge 0-3
    pz_merge_2l_2x4si_sse2((v4si *) &v[4], (v4si *) &v[6]); // Merge 4-7

    pz_merge_2seq_sse2(&a[0].v, &v[0].v, &v[4].v, 4);

    // Check
    for (i = 0, pa = (int32_t *) a; i < 31; i++)
        if (pa[i] > pa[i + 1]) {
            printf("test_merge_2_pairs: error at position %d: %d > %d\n",
                    i, pa[i], pa[i + 1]);
            break;
        }

    if (i != 31) {

        for (i = 0, pa = (int32_t *) a; i < 32; i++)
            printf("% 3d%c", pa[i], i == 15? '\n' : ' ');
        printf("\n");
        return -1;

    }

    return 0;

}

// Test register sort of a 32K sequence
int test_sort_registers_32k() {
    int32_t  *pa;
    int      i, j;

    vec_random(32768); // Add some random numbers

    pz_register_seq_sort_4si_sse2(&v[0].v, 32768);

    // Check
    for (i = 0, pa = (int32_t *) v; i < (32768 / 4); i++)
        for (j = 0; j < 3; j++)
            if (pa[i * 4 + j] > pa[i * 4 + j + 1]) {
                printf("test_merge_2_pairs: error at position %d: %d > %d\n",
                    i * 4 + j, pa[i * 4 + j], pa[i * 4 + j + 1]);
            break;
        }

    if (i != (32768 / 4)) {

#if 0
        for (i = 0, pa = (int32_t *) a; i < 32; i++)
            printf("% 3d%c", pa[i], i == 15? '\n' : ' ');
        printf("\n");
#endif
        return -1;

    }

    return 0;

}


int run_test(int (*f)(void), char *name, int reps) {
    int i;

    for (i = 0; i < reps && f() == 0; i++)
        ;
    if (i == reps) {
        printf("%s: %d tests OK\n", name, i);
        return 0;
    } else {
        printf("%s: error on test %d\n", name, i);
        return -1;
    }
}

int main(int argc, char *argv[]) {
    int t = 10000; // Repetitions of the tests

    // Alloc buffers
    v = _mm_malloc(sizeof (*v) * 32768, 16);
    a = _mm_malloc(sizeof (*v) * 32768, 16);

    srandomdev(); // Init random pool from random device

    run_test(test_column_sort_4, "test_column_sort_4", t);
    run_test(test_register_sort_4, "test_register_sort_4", t);
    run_test(test_bitonic_sort, "test_bitonic_sort", t);
    run_test(test_bitonic_sort_2x, "test_bitonic_sort_2x", t);
    run_test(test_merge_2_pairs, "test_merge_2_pairs", t);
    run_test(test_merge_parallel_2list_2pairs,
            "test_merge_parallel_2list_2pairs", t);
    run_test(test_merge_16x16, "test_merge_16x16", t);
    run_test(test_merge_2seq, "test_merge_2seq", t);
    run_test(test_sort_registers_32k, "test_sort_registers_32k", 512);

    _mm_free(v);
    _mm_free(a);

    return 0;

}
