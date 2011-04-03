//  PF compressor SSE2 methods
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

//  This file implements SSE2 primitives, mostly for sorting


#include <stdint.h>
#include <xmmintrin.h>

// A vector of 4 32bit signed integers (SSE2 128bit register)
typedef __v4si v4si;
typedef __v4sf v4sf;

typedef union {
  int32_t s[4];
  v4si  v;
} v4si_u;

// SSE2 lacks pmin/pmax for 32bit
static void minmax_4si_sse2(v4si *a, v4si *b) {
    v4si mask = __builtin_ia32_pcmpgtd128(*a, *b);
    v4si t = (*a ^ *b) & mask;
    *a ^= t;
    *b ^= t;
}

// In-register sort of 4
static void column_sort_4si_sse2(v4si *a, v4si *b, v4si *c, v4si *d) {

    minmax_4si_sse2(a, c);
    minmax_4si_sse2(b, d);
    minmax_4si_sse2(a, b);
    minmax_4si_sse2(c, d);
    minmax_4si_sse2(b, c);

}

// Transpose 4 vectors of 4 32bit elements
static void transpose_4si_sse2(v4si *a, v4si *b, v4si *c, v4si *d) {

  __v4sf t0 = __builtin_ia32_unpcklps ((__v4sf) *a, (__v4sf) *b);
  __v4sf t1 = __builtin_ia32_unpcklps ((__v4sf) *c, (__v4sf) *d);
  __v4sf t2 = __builtin_ia32_unpckhps ((__v4sf) *a, (__v4sf) *b);
  __v4sf t3 = __builtin_ia32_unpckhps ((__v4sf) *c, (__v4sf) *d);
  *a = (v4si) __builtin_ia32_movlhps (t0, t1);
  *b = (v4si) __builtin_ia32_movhlps (t1, t0);
  *c = (v4si) __builtin_ia32_movlhps (t2, t3);
  *d = (v4si) __builtin_ia32_movhlps (t3, t2);

}
    
// In-register sort of 4 vectors of 32bit signed integers
static void register_sort_4si_sse2(v4si *a, v4si *b, v4si *c, v4si *d) {

    column_sort_4si_sse2(a, b, c, d); // Sort columns
    transpose_4si_sse2(a, b, c, d);   // Transpose (a, b, c, d each sorted)

}

//
// Implementation of a bitonic merge on a 4x4 matrix
//

// Level 1 exchange
static void bitonic_l1_exchange_4si_sse2(v4si *a, v4si *b) {

    __v4sf t0 = __builtin_ia32_movhlps ((__v4sf) *b,(__v4sf)  *a);
    __v4sf t1 = __builtin_ia32_movlhps ((__v4sf) *a,(__v4sf)  *b);
    *b = (v4si) t0;
    *a = (v4si) t1;
}

// Level 2 exchange
static void bitonic_l2_exchange_4si_sse2(v4si *a, v4si *b) {

    __v4sf t0 = __builtin_ia32_unpckhps ((__v4sf) *a, (__v4sf) *b);
    __v4sf t1 = __builtin_ia32_unpcklps ((__v4sf) *a, (__v4sf) *b);
    *a = (v4si) __builtin_ia32_movlhps (t1, t0);
    *b = (v4si) __builtin_ia32_movhlps (t0, t1);

}

// Level 3 exchange
static void bitonic_l3_exchange_4si_sse2(v4si *a, v4si *b) {

    __v4sf t = __builtin_ia32_unpcklps ((__v4sf) *a, (__v4sf) *b);
    *b = (v4si) __builtin_ia32_unpckhps ((__v4sf) *a, (__v4sf) *b);
    *a = (v4si) t;

}

// Bitonic sort for 4 vectors of 4 32bit signed integers
static void bitonic_sort_4si_sse2(v4si *a, v4si *b) {

    *a = __builtin_ia32_pshufd (*a, 0x1B); // Reverse a

    minmax_4si_sse2(a, b);
    bitonic_l1_exchange_4si_sse2(a, b);
    minmax_4si_sse2(a,b);
    bitonic_l2_exchange_4si_sse2(a, b);
    minmax_4si_sse2(a,b);
    bitonic_l3_exchange_4si_sse2(a,b);

}

// Parallel bitonic sort for 2+2 vectors
//   Same as bitonic(a,b) and bitonic(c,d)
//   For latency hiding and better reciprocal throughput
static void bitonic_sort_2x_4si_sse2(v4si *a, v4si *b, v4si *c, v4si *d) {

    *a = __builtin_ia32_pshufd (*a, 0x1B); // Reverse a
    *c = __builtin_ia32_pshufd (*c, 0x1B); // Reverse c

    minmax_4si_sse2(a, b);
    minmax_4si_sse2(c, d);

    bitonic_l1_exchange_4si_sse2(a, b);
    bitonic_l1_exchange_4si_sse2(c, d);

    minmax_4si_sse2(a,b);
    minmax_4si_sse2(c,d);

    bitonic_l2_exchange_4si_sse2(a, b);
    bitonic_l2_exchange_4si_sse2(c, d);

    minmax_4si_sse2(a,b);
    minmax_4si_sse2(c,d);

    bitonic_l3_exchange_4si_sse2(a,b);
    bitonic_l3_exchange_4si_sse2(c,d);

}

// Merge 4 adjacent pairs of sorted registers 
//   since it's only 4 we don't need big aux vector
static void merge_2l_2x4si_sse2(v4si *v) {
    v4si_u  b, d;

    b.v = v[1]; // To compare first elements
    d.v = v[3];

    bitonic_sort_4si_sse2(&v[0], &v[2]); // Merge heads (a, c)

    if (b.s[0] > d.s[0]) {
        v[1] = d.v; // Exchange b and d
        v[3] = b.v;
    }

    bitonic_sort_4si_sse2(&v[1], &v[2]); // Now v[1] is done
    bitonic_sort_4si_sse2(&v[2], &v[3]); // Now v[2] and v[3] are done

}

#ifdef TEST

void pz_sort_4si(v4si *a, v4si *b, v4si *c, v4si *d) {
    column_sort_4si_sse2(a, b, c, d);
}

void pz_transpose_4(v4si *a, v4si *b, v4si *c, v4si *d) {
    transpose_4si_sse2(a, b, c, d);
}

void pz_register_sort_4si(v4si *a, v4si *b, v4si *c, v4si *d) {
    register_sort_4si_sse2(a, b, c, d);
}

void pz_bitonic_sort_4si(v4si *a, v4si *b) {
    bitonic_sort_4si_sse2(a, b);
}

void pz_bitonic_sort_2x_4si(v4si *a, v4si *b, v4si *c, v4si *d) {
    bitonic_sort_2x_4si_sse2(a, b, c, d);
}

void pz_merge_2l_2x4si(v4si *v) {
    merge_2l_2x4si_sse2(v);
}

// External function for sorting 4x4 signed integers
void pz_sort_4x4si(v4si *a, v4si *b, v4si *c, v4si *d) {

    column_sort_4si_sse2(a, b, c, d);

}

#endif
