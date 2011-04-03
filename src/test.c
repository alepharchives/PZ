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
void pz_sort_4x4si_each(v4si *a, v4si *b, v4si *c, v4si *d);
void pz_sort_4x4si(v4si *a, v4si *b, v4si *c, v4si *d);

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

// Test sorting columns of 4 vectors of 4 32bit signed integers
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

int main(int argc, char *argv[]) {
    int t = 100; // Repetitions of the tests
    int i;

    srandomdev(); // Init random pool from random device

    for (i = 0; i < t; i++)
        if (test_column_sort_4())
            return -1;
    printf("test_column_sort_4: %d tests OK\n", i);

    for (i = 0; i < t; i++)
        if (test_register_sort_4())
            return -1;
    printf("test_register_sort_4: %d tests OK\n", i);

    return 0;

}
