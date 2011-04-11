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
void pz_bitonic_sort_4si_sse2(v4si *v);
void pz_merge_2l_2x4si_sse2(v4si *v);
void pz_merge_parallel_2x2l_2x8si_sse2(v4si *v);
void pz_bitonic_sort_2x_4si_sse2(v4si *v);
void pz_bitonic_merge_2x16si_sse2(v4si *v);
void pz_register_sort_4x4si_full_sse2(v4si *v);

// Make 4 vectors of 4 32bit signed integers and fill with random
v4si_u * get_4x_v4si_random(int size, int32_t m) {
    v4si_u  *v;
    int      i, j;

    v = _mm_malloc(sizeof (*v) * size, 16);
    for (i = 0; i < size; i++)
        for (j = 0; j < 4; j++)
            v[i].s[j] = random() & m; // Smaller number, only testing 4

    return v;

}

// Test sorting columns of 4 vectors of 4 32bit signed integers
int test_column_sort_4() {
    v4si_u   *v;
    int      i, j;

    v = get_4x_v4si_random(4, 3); // Get random 0-3

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

    _mm_free(v);

    return 0;

}

// Test transposing 4 vectors of 4 32bit signed integers
int test_register_sort_4() {
    v4si_u   *v;
    int      i, j;

    v = get_4x_v4si_random(4, 3); // Get random 0-3

    pz_register_sort_4si_sse2(&v[0].v); // Sort 0-3

    // Check
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 3; j++) {
            if (v[i].s[j] > v[i].s[j + 1]) {
                printf("test_register_sort: error at %d:%d\n", i, j);
                printf("  % 2d - % 2d\n", v[i].s[j], v[i].s[j + 1]);
                // Display sorted
                for (i = 0; i < 4; i++)
                    printf("% 3d % 3d % 3d % 3d\n",
                            v[i].s[0], v[i].s[1], v[i].s[2], v[i].s[3]);
                return (-1);
            }
        }
    }

    _mm_free(v);

    return 0;

}

// Test sorting 16 signed integers
int test_register_sort_full_16si() {
    v4si_u   *v;
    int32_t  *vi;
    int      i;

    v = get_4x_v4si_random(4, 15); // Get random 0-15

    pz_register_sort_4x4si_full_sse2(&v[0].v); // Sort 0-3 full

    // Check
    for (i = 1, vi = (int32_t *) v; i < 15; i++) {
        if (vi[i] > vi[i + 1]) {
            printf("test_register_sort_full_16si: error at %d\n", i);
            printf("  % 2d - % 2d\n", vi[i], vi[i + 1]);
        }
    }

    _mm_free(v);

    return 0;

}

// Test merge sort of 2 vectors of 4 32bit signed integers
int test_bitonic_sort() {
    int32_t  a[8], *pa;
    v4si_u   *v;
    int      i, j;

    v = get_4x_v4si_random(4, 15); // Get random 0-3

    pz_register_sort_4si_sse2(&v[0].v); // Sort 0-3
    pz_bitonic_sort_4si_sse2(&v[0].v); // Sort first two together

    // Move to sequential array
    for (i = 0, pa = a; i < 2; i++)
        for (j = 0; j < 4; j++)
            *pa++ = v[i].s[j];

    // Check
    for (i = 0; i < 7; i++)
        if (a[i] > a[i + 1]) {
            printf("%d %d\n", i, j);
            printf("test_bitonic_sort: error at position %d: %d > %d\n",
                    i, a[i], a[i + 1]);
            break;
        }

    _mm_free(v);

    return 0;

}

// Test parallel merge sort of 2 pairs of 2 vectors of 4 32bit signed integers
int test_bitonic_sort_2x() {
    int32_t  a[8], b[8], *pa, *pb;
    v4si_u   *v;
    int      i, j;

    v = get_4x_v4si_random(4, 15); // Get random 0-15

    pz_register_sort_4si_sse2(&v[0].v); // Sort 0-3
    pz_bitonic_sort_4si_sse2(&v[0].v); // Sort 0-1
    pz_bitonic_sort_4si_sse2(&v[2].v); // Sort 2-3

    // Move to sequential array
    for (i = 0, pa = a, pb = b; i < 2; i++)
        for (j = 0; j < 4; j++) {
            *pa++ = v[i].s[j];
            *pb++ = v[i + 2].s[j];
        }

    // Check
    for (i = 0; i < 7; i++)
        if (a[i] > a[i + 1] || b[i] > b[i + 1]) {
            pa = a[i] > a[i + 1] ? pa : pb; // For single printf line
            printf("%d %d\n", i, j);
            printf("test_bitonic_sort: error at position %d: %d > %d\n",
                    i, *pa, pa[1]);
            break;
        }

    _mm_free(v);

    return 0;

}

// Test parallel merge sort of 2 pairs of 2 vectors of 4 32bit signed integers
int test_merge_2_pairs() {
    int32_t  a[16], *pa;
    v4si_u   *v;
    int      i, j;

    v = get_4x_v4si_random(4, 15); // Get random 0-15

    pz_register_sort_4si_sse2(&v[0].v); // Sort 0-3
    pz_bitonic_sort_4si_sse2(&v[0].v); // Sort 0-1
    pz_bitonic_sort_4si_sse2(&v[2].v); // Sort 2-3
    pz_merge_2l_2x4si_sse2((v4si *) v); // Merge 2 adjacent lists of 2 pairs

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

        _mm_free(v);
        return -1;

    }

    _mm_free(v);

    return 0;

}

// Test parallel merge sort of 2 list of 2x2
//    v0   v1   v2   v3      v4   v5   v6   v7
//   aaaa aaaa bbbb bbbb || cccc cccc dddd dddd    (input)
//   xxxx xxxx xxxx xxxx || yyyy yyyy yyyy yyyy    (result)
//
int test_merge_parallel_2list_2pairs() {
    int32_t  a[32], *pa;
    v4si_u   *v;
    int      i, j;

    v = get_4x_v4si_random(8, 15); // Get 8 vectors with random 0-15

    pz_register_sort_4si_sse2(&v[0].v); // Sort 0-3
    pz_register_sort_4si_sse2(&v[4].v); // Sort 4-7
    pz_bitonic_sort_4si_sse2(&v[0].v); // Sort 0-1
    pz_bitonic_sort_4si_sse2(&v[2].v); // Sort 2-3
    pz_bitonic_sort_2x_4si_sse2(&v[0].v);
    pz_bitonic_sort_2x_4si_sse2(&v[4].v);
    pz_merge_2l_2x4si_sse2((v4si *) v); // Merge 2 adjacent lists of 2 pairs
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

        _mm_free(v);
        return -1;

    }

    _mm_free(v);

    return 0;

}

// Test merge of 2 lists of 16 32bit signed integers (16x16si network)
int test_merge_16x16() {
    int32_t  a[32], *pa;
    v4si_u   *v;
    int      i, j;

    v = get_4x_v4si_random(8, 31); // Get random 0-31

    pz_register_sort_4si_sse2(&v[0].v); // Sort 0-3
    pz_register_sort_4si_sse2(&v[4].v); // Sort 4-7
    pz_bitonic_sort_4si_sse2(&v[0].v); // Sort 0-1
    pz_bitonic_sort_4si_sse2(&v[2].v); // Sort 2-3
    pz_bitonic_sort_4si_sse2(&v[4].v); // Sort 4-5
    pz_bitonic_sort_4si_sse2(&v[6].v); // Sort 6-7
    pz_merge_2l_2x4si_sse2((v4si *) &v[0]); // Merge 2 adjacent lists of 2 pairs
    pz_merge_2l_2x4si_sse2((v4si *) &v[4]); // Merge 2 adjacent lists of 2 pairs

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

        _mm_free(v);
        return -1;

    }

    _mm_free(v);

    return 0;

}

int run_test(int (*f)(void), char *name, int reps) {
    int i;

    for (i = 0; i < reps; i++)
        if (f())
            return -1;
    printf("%s: %d tests OK\n", name, i);
    return 0;
}

int main(int argc, char *argv[]) {
    int t = 10000; // Repetitions of the tests

    srandomdev(); // Init random pool from random device

    run_test(test_column_sort_4, "test_column_sort_4", t);
    run_test(test_register_sort_4, "test_register_sort_4", t);
    run_test(test_register_sort_full_16si, "test_register_sort_full_16si", t);
    run_test(test_bitonic_sort, "test_bitonic_sort", t);
    run_test(test_bitonic_sort_2x, "test_bitonic_sort_2x", t);
    run_test(test_merge_2_pairs, "test_merge_2_pairs", t);
    run_test(test_merge_parallel_2list_2pairs,
            "test_merge_parallel_2list_2pairs", t);
    run_test(test_merge_16x16, "test_merge_16x16", t);

    return 0;

}
