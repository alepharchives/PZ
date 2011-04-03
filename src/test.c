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
void pz_sort_4si(v4si *a, v4si *b, v4si *c, v4si *d);
void pz_transpose_4(v4si *a, v4si *b, v4si *c, v4si *d);
void pz_register_sort_4si(v4si *a, v4si *b, v4si *c, v4si *d);
void pz_bitonic_sort_4si(v4si *a, v4si *b);
//void pz_sort_4x4si_each(v4si *a, v4si *b, v4si *c, v4si *d);
//void pz_sort_4x4si(v4si *a, v4si *b, v4si *c, v4si *d);

// Make 4 vectors of 4 32bit signed integers and fill with random
v4si_u * get_4x_v4si_random(int32_t m) {
    v4si_u  *v;
    int      i, j;

    v = _mm_malloc(sizeof (*v) * 4, 16);
    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++)
            v[i].s[j] = random() & m; // Smaller number, only testing 4

    return v;

}

// Test sorting columns of 4 vectors of 4 32bit signed integers
int test_column_sort_4() {
    v4si_u   *v;
    int      i, j;

    v = get_4x_v4si_random(3); // Get random 0-3

    pz_sort_4si(&v[0].v, &v[1].v, &v[2].v, &v[3].v);

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

    v = get_4x_v4si_random(3); // Get random 0-3

    pz_register_sort_4si(&v[0].v, &v[1].v, &v[2].v, &v[3].v);

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

// Test merge sort of 2 vectors of 4 32bit signed integers
int test_bitonic_sort() {
    int32_t  a[8], *pa;
    v4si_u   *v;
    int      i, j;

    v = get_4x_v4si_random(15); // Get random 0-3

    pz_register_sort_4si(&v[0].v, &v[1].v, &v[2].v, &v[3].v); // In register
    pz_bitonic_sort_4si(&v[0].v, &v[1].v); // Sort first two together

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

    v = get_4x_v4si_random(15); // Get random 0-3

    pz_register_sort_4si(&v[0].v, &v[1].v, &v[2].v, &v[3].v); // In register
    pz_bitonic_sort_4si(&v[0].v, &v[1].v); // Sort first pair
    pz_bitonic_sort_4si(&v[2].v, &v[3].v); // Sort second pair

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

int run_test(int (*f)(void), char *name, int reps) {
    int i;

    for (i = 0; i < reps; i++)
        if (f())
            return -1;
    printf("%s: %d tests OK\n", name, i);
    return 0;
}

int main(int argc, char *argv[]) {
    int t = 100; // Repetitions of the tests

    srandomdev(); // Init random pool from random device

    run_test(test_column_sort_4, "test_column_sort_4", t);
    run_test(test_register_sort_4, "test_register_sort_4", t);
    run_test(test_bitonic_sort, "test_bitonic_sort", t);
    run_test(test_bitonic_sort_2x, "test_bitonic_sort_2x", t);

    return 0;

}
