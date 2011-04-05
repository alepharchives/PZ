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

#ifndef TEST
#  define LOCAL static // No external call, inline
#else
#  define LOCAL        // Empty, allows external call
#endif

// A vector of 4 32bit signed integers (SSE2 128bit register)
typedef __v4si v4si;
typedef __v4sf v4sf;

typedef union {
  int32_t s[4];
  v4si  v;
} v4si_u;

static void swap_sse2(v4si *a, v4si *b) {
    v4si aux = *a;
    *a = *b;
    *b = aux;
}

static void reverse_v4_sse2(v4si *v) {
    *v = __builtin_ia32_pshufd (*v, 0x1B); // abcd -> dcab
}

// SSE2 lacks pmin/pmax for 32bit
LOCAL void minmax_4si_sse2(v4si *a, v4si *b) {
    v4si mask = __builtin_ia32_pcmpgtd128(*a, *b);
    v4si t = (*a ^ *b) & mask;
    *a ^= t;
    *b ^= t;
}

// In-register sort of 4
LOCAL void column_sort_4si_sse2(v4si *a, v4si *b, v4si *c, v4si *d) {

    minmax_4si_sse2(a, c);
    minmax_4si_sse2(b, d);
    minmax_4si_sse2(a, b);
    minmax_4si_sse2(c, d);
    minmax_4si_sse2(b, c);

}

// Transpose 4 vectors of 4 32bit elements
LOCAL void transpose_4si_sse2(v4si *a, v4si *b, v4si *c, v4si *d) {

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
LOCAL void register_sort_4si_sse2(v4si *a, v4si *b, v4si *c, v4si *d) {

    column_sort_4si_sse2(a, b, c, d); // Sort columns
    transpose_4si_sse2(a, b, c, d);   // Transpose (a, b, c, d each sorted)

}

//
// Implementation of a bitonic merge on a 4x4 matrix
//

// Level 1 exchange
LOCAL void bitonic_l1_exchange_4si_sse2(v4si *a, v4si *b) {

    __v4sf t0 = __builtin_ia32_movhlps ((__v4sf) *b,(__v4sf)  *a);
    __v4sf t1 = __builtin_ia32_movlhps ((__v4sf) *a,(__v4sf)  *b);
    *b = (v4si) t0;
    *a = (v4si) t1;
}

// Level 2 exchange
LOCAL void bitonic_l2_exchange_4si_sse2(v4si *a, v4si *b) {

    __v4sf t0 = __builtin_ia32_unpckhps ((__v4sf) *a, (__v4sf) *b);
    __v4sf t1 = __builtin_ia32_unpcklps ((__v4sf) *a, (__v4sf) *b);
    *a = (v4si) __builtin_ia32_movlhps (t1, t0);
    *b = (v4si) __builtin_ia32_movhlps (t0, t1);

}

// Level 3 exchange
LOCAL void bitonic_l3_exchange_4si_sse2(v4si *a, v4si *b) {

    __v4sf t = __builtin_ia32_unpcklps ((__v4sf) *a, (__v4sf) *b);
    *b = (v4si) __builtin_ia32_unpckhps ((__v4sf) *a, (__v4sf) *b);
    *a = (v4si) t;

}

// Bitonic merge 4x4si (2 vectors (registers) of 4 32bit signed integers each)
LOCAL void bitonic_merge_4x4si_sse2(v4si *a, v4si *b) {

    minmax_4si_sse2(a, b);
    bitonic_l1_exchange_4si_sse2(a, b);
    minmax_4si_sse2(a,b);
    bitonic_l2_exchange_4si_sse2(a, b);
    minmax_4si_sse2(a,b);
    bitonic_l3_exchange_4si_sse2(a,b);

}

// Bitonic sort for 2 vectors (registers) of 4 32bit signed integers (each)
LOCAL void bitonic_sort_4si_sse2(v4si *a, v4si *b) {

    reverse_v4_sse2(a);
    bitonic_merge_4x4si_sse2(a, b);

}

// Parallel bitonic sort for 2+2 vectors
//   Same as bitonic(&v[0],&v[1]) and bitonic(&v[2],&v[3])
//   For latency hiding and better reciprocal throughput
//   aaaa bbbb || cccc dddd
//   xxxx xxxx    yyyy yyyy
LOCAL void bitonic_sort_2x_4si_sse2(v4si *v) {

    reverse_v4_sse2(&v[0]);
    reverse_v4_sse2(&v[2]);

    minmax_4si_sse2(&v[0], &v[1]);
    minmax_4si_sse2(&v[2], &v[3]);

    bitonic_l1_exchange_4si_sse2(&v[0], &v[1]);
    bitonic_l1_exchange_4si_sse2(&v[2], &v[3]);

    minmax_4si_sse2(&v[0],&v[1]);
    minmax_4si_sse2(&v[2],&v[3]);

    bitonic_l2_exchange_4si_sse2(&v[0], &v[1]);
    bitonic_l2_exchange_4si_sse2(&v[2], &v[3]);

    minmax_4si_sse2(&v[0],&v[1]);
    minmax_4si_sse2(&v[2],&v[3]);

    bitonic_l3_exchange_4si_sse2(&v[0],&v[1]);
    bitonic_l3_exchange_4si_sse2(&v[2],&v[3]);

}

// Merge 2 lists of 2 vectors
//   aaaa aaaa || bbbb bbbb
//   0123 4567    89AB CDEF
//
LOCAL void merge_2l_2x4si_sse2(v4si *v) {

    reverse_v4_sse2(&v[2]);
    reverse_v4_sse2(&v[3]);
    swap_sse2(&v[2], &v[3]);

    minmax_4si_sse2(&v[0], &v[2]); // L1
    minmax_4si_sse2(&v[1], &v[3]); // L1

    bitonic_merge_4x4si_sse2(&v[0], &v[1]);
    bitonic_merge_4x4si_sse2(&v[2], &v[3]);

}

// Simultaneous merge of 2 pairs of lists of 2x4si each
//    v0   v1   v2   v3      v4   v5   v6   v7
//   aaaa aaaa bbbb bbbb || cccc cccc dddd dddd    (input)
//   xxxx xxxx xxxx xxxx || yyyy yyyy yyyy yyyy    (result)
//
LOCAL void merge_parallel_2x2l_2x4si_sse2(v4si *v) {

    // Prepare by reversing 2nd pairs
    reverse_v4_sse2(&v[2]); // A
    reverse_v4_sse2(&v[3]); // A
    reverse_v4_sse2(&v[6]); // B
    reverse_v4_sse2(&v[7]); // B
    swap_sse2(&v[2], &v[3]); // A
    swap_sse2(&v[6], &v[7]); // B

    minmax_4si_sse2(&v[0], &v[2]); // L1  A
    minmax_4si_sse2(&v[4], &v[6]); // L1  B
    minmax_4si_sse2(&v[1], &v[3]); // L1  A
    minmax_4si_sse2(&v[5], &v[7]); // L1  B

    bitonic_merge_4x4si_sse2(&v[0], &v[1]); // Bitonic 4x4si A1
    bitonic_merge_4x4si_sse2(&v[4], &v[5]); // Bitonic 4x4si B1

    bitonic_merge_4x4si_sse2(&v[2], &v[3]); // Bitonic 4x4si A2
    bitonic_merge_4x4si_sse2(&v[6], &v[7]); // Bitonic 4x4si B2

}
