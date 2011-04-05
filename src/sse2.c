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
//   Same as bitonic(a,b) and bitonic(c,d)
//   For latency hiding and better reciprocal throughput
//   aaaa bbbb || cccc dddd
//   xxxx xxxx    yyyy yyyy
LOCAL void bitonic_sort_2x_4si_sse2(v4si *a, v4si *b, v4si *c, v4si *d) {

    reverse_v4_sse2(a);
    reverse_v4_sse2(c);

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
    v4si_u  *vu = (v4si_u *) v;

    bitonic_sort_4si_sse2(&v[0], &v[2]); // Merge heads (a, b)
    bitonic_sort_4si_sse2(&v[4], &v[6]); // Merge heads (c, d)

    if (vu[1].s[0] > vu[3].s[0]) { // if 2nd a > 2nd b: swap
        v4si aux = v[1]; // move 2nd a to aux
        v[1] = v[3];     // move 2nd b to [1] (where 2nd a was)
        v[3] = aux;      // move aux (2nd a) to [3] (where 2nd b was)
    }

    if (vu[5].s[0] > vu[7].s[0]) { // if 2nd a > 2nd b: swap
        v4si aux = v[5]; // move 2nd c to aux
        v[5] = v[7];     // move 2nd d to [1] (where 2nd c was)
        v[7] = aux;      // move aux (2nd a) to [3] (where 2nd d was)
    }

    bitonic_sort_4si_sse2(&v[1], &v[2]); // Now v[1] is done
    bitonic_sort_4si_sse2(&v[2], &v[3]); // Now v[2] and v[3] are done

    bitonic_sort_4si_sse2(&v[5], &v[6]); // Now v[5] is done
    bitonic_sort_4si_sse2(&v[6], &v[7]); // Now v[6] and v[7] are done

}
