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

static void swap_sse2(v4si *a, v4si *b) {
    v4si aux = *a;
    *a = *b;
    *b = aux;
}

static void reverse_v4_sse2(v4si *a) {
    *a = __builtin_ia32_pshufd (*a, 0x1B); // abcd -> dcab
}

// SSE2 lacks pmin/pmax for 32bit
static void minmax_4si_sse2(v4si *a, v4si *b) {
    v4si mask = __builtin_ia32_pcmpgtd128(*a, *b);
    v4si t = (*a ^ *b) & mask;
    *a ^= t;
    *b ^= t;
}

// In-register sort of 4
static void column_sort_4si_sse2(v4si *v) {

    minmax_4si_sse2(&v[0], &v[2]);
    minmax_4si_sse2(&v[1], &v[3]);
    minmax_4si_sse2(&v[0], &v[1]);
    minmax_4si_sse2(&v[2], &v[3]);
    minmax_4si_sse2(&v[1], &v[2]);

}

// Transpose 4 vectors of 4 32bit elements
static void transpose_4si_sse2(v4si *v) {

  __v4sf t0 = __builtin_ia32_unpcklps ((__v4sf) v[0], (__v4sf) v[1]);
  __v4sf t1 = __builtin_ia32_unpcklps ((__v4sf) v[2], (__v4sf) v[3]);
  __v4sf t2 = __builtin_ia32_unpckhps ((__v4sf) v[0], (__v4sf) v[1]);
  __v4sf t3 = __builtin_ia32_unpckhps ((__v4sf) v[2], (__v4sf) v[3]);
  v[0] = (v4si) __builtin_ia32_movlhps (t0, t1);
  v[1] = (v4si) __builtin_ia32_movhlps (t1, t0);
  v[2] = (v4si) __builtin_ia32_movlhps (t2, t3);
  v[3] = (v4si) __builtin_ia32_movhlps (t3, t2);

}
    
// In-register sort of 4 vectors of 32bit signed integers
static void register_sort_4si_sse2(v4si *v) {

    column_sort_4si_sse2(v); // Sort columns
    transpose_4si_sse2(v);   // Transpose (a, b, c, d each sorted)

}

//
// Implementation of a bitonic merge on a 4x4 matrix
//

// Level 1 exchange
static void bitonic_l1_exchange_4si_sse2(v4si *a, v4si *b) {

    __v4sf t0 = __builtin_ia32_movhlps ((__v4sf) *b,(__v4sf) *a);
    __v4sf t1 = __builtin_ia32_movlhps ((__v4sf) *a,(__v4sf) *b);
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

// Bitonic merge 4x4si (2 vectors (registers) of 4 32bit signed integers each)
static void bitonic_merge_4x4si_sse2(v4si *a, v4si *b) {

    minmax_4si_sse2(a, b);
    bitonic_l1_exchange_4si_sse2(a, b);
    minmax_4si_sse2(a, b);
    bitonic_l2_exchange_4si_sse2(a, b);
    minmax_4si_sse2(a, b);
    bitonic_l3_exchange_4si_sse2(a, b);

}

// Bitonic sort for 2 vectors (registers) of 4 32bit signed integers (each)
static void bitonic_sort_4si_sse2(v4si *a, v4si *b) {

    reverse_v4_sse2(a);
    bitonic_merge_4x4si_sse2(a, b);

}

// Parallel bitonic sort for 2+2 vectors
//   Same as bitonic(&v[0],&v[1]) and bitonic(&v[2],&v[3])
//   For latency hiding and better reciprocal throughput
//   aaaa bbbb || cccc dddd
//   xxxx xxxx    yyyy yyyy
static void bitonic_sort_2x_4si_sse2(v4si *a, v4si *b, v4si *c, v4si *d) {

    reverse_v4_sse2(a);
    reverse_v4_sse2(c);

    minmax_4si_sse2(a, b);
    minmax_4si_sse2(c, d);

    bitonic_l1_exchange_4si_sse2(a, b); // a-b
    bitonic_l1_exchange_4si_sse2(c, d); // c-d

    minmax_4si_sse2(a, b);
    minmax_4si_sse2(c, d);

    bitonic_l2_exchange_4si_sse2(a, b); // a-b
    bitonic_l2_exchange_4si_sse2(c, d); // c-d

    minmax_4si_sse2(a, b);
    minmax_4si_sse2(c, d);

    bitonic_l3_exchange_4si_sse2(a, b); // a-b
    bitonic_l3_exchange_4si_sse2(c, d); // c-d

}

// Merge 2 sequences of 2 vectors
//   aaaa aaaa || bbbb bbbb
//   0123 4567    89AB CDEF
//
static void merge_2l_2x4si_sse2(v4si *s1, v4si *s2) {

    reverse_v4_sse2(&s2[0]);
    reverse_v4_sse2(&s2[1]);
    swap_sse2(&s2[0], &s2[1]);

    minmax_4si_sse2(&s1[0], &s2[0]); // L1
    minmax_4si_sse2(&s1[1], &s2[1]); // L1

    bitonic_merge_4x4si_sse2(&s1[0], &s1[1]); // s1
    bitonic_merge_4x4si_sse2(&s2[0], &s2[1]); // s2

}

static void bitonic_merge_8x8si_sse2(v4si *v) {

    minmax_4si_sse2(&v[0], &v[2]); // L1  A
    minmax_4si_sse2(&v[4], &v[6]); // L1  B
    minmax_4si_sse2(&v[1], &v[3]); // L1  A
    minmax_4si_sse2(&v[5], &v[7]); // L1  B

    bitonic_merge_4x4si_sse2(&v[0], &v[1]); // Bitonic 4x4si A1  0-1
    bitonic_merge_4x4si_sse2(&v[4], &v[5]); // Bitonic 4x4si B1  4-5

    bitonic_merge_4x4si_sse2(&v[2], &v[3]); // Bitonic 4x4si A2  2-3
    bitonic_merge_4x4si_sse2(&v[6], &v[7]); // Bitonic 4x4si B2  6-7

}


// Simultaneous merge of 2 pairs of lists of 2x4si each
//    v0   v1   v2   v3      v4   v5   v6   v7
//   aaaa aaaa bbbb bbbb || cccc cccc dddd dddd    (input)
//   xxxx xxxx xxxx xxxx || yyyy yyyy yyyy yyyy    (result)
//
static void merge_parallel_2x2l_2x8si_sse2(v4si *v) {

    // Prepare by reversing 2nd pairs
    reverse_v4_sse2(&v[2]); // A
    reverse_v4_sse2(&v[3]); // A
    reverse_v4_sse2(&v[6]); // B
    reverse_v4_sse2(&v[7]); // B
    swap_sse2(&v[2], &v[3]); // A
    swap_sse2(&v[6], &v[7]); // B

    bitonic_merge_8x8si_sse2(v); // 8x8si network

}

// Bitonic merge 2 lists of 4 vectors (16x16si network)
//    v0   v1   v2   v3      v4   v5   v6   v7
//   aaaa aaaa aaaa aaaa || bbbb bbbb bbbb bbbb
//   0123 4567 89AB CDEF    0123 4567 89AB CDEF
//
static void bitonic_merge_2x16si_sse2(v4si *v) {

    // Prepare for L1 reversing v4-7
    reverse_v4_sse2(&v[4]);
    reverse_v4_sse2(&v[5]);
    reverse_v4_sse2(&v[6]);
    reverse_v4_sse2(&v[7]);
    swap_sse2(&v[4], &v[7]);
    swap_sse2(&v[5], &v[6]);

    // L1 compare
    minmax_4si_sse2(&v[0], &v[4]);
    minmax_4si_sse2(&v[1], &v[5]);
    minmax_4si_sse2(&v[2], &v[6]);
    minmax_4si_sse2(&v[3], &v[7]);

    bitonic_merge_8x8si_sse2(&v[0]);
    bitonic_merge_8x8si_sse2(&v[4]);

}

#ifdef TEST

// SSE2 test interfaces
void pz_column_sort_4si_sse2(v4si *v) {
    column_sort_4si_sse2(v);
}
void pz_transpose_4si_sse2(v4si *v) {
    transpose_4si_sse2(v);
}
void pz_register_sort_4si_sse2(v4si *v) {
    register_sort_4si_sse2(v);
}
void pz_bitonic_sort_4si_sse2(v4si *a, v4si *b) {
    bitonic_sort_4si_sse2(a, b);
}
void pz_merge_2l_2x4si_sse2(v4si *s1, v4si *s2) {
    merge_2l_2x4si_sse2(s1, s2);
}
void pz_merge_parallel_2x2l_2x8si_sse2(v4si *v) {
    merge_parallel_2x2l_2x8si_sse2(v);
}
void pz_bitonic_sort_2x_4si_sse2(v4si *a, v4si *b, v4si *c, v4si *d) {
    bitonic_sort_2x_4si_sse2(a, b, c, d);
}
void pz_bitonic_merge_2x16si_sse2(v4si *v) {
    bitonic_merge_2x16si_sse2(v);
}
void pz_merge_2seq_sse2(v4si *dst, v4si *src1, v4si *src2, int len) {
    merge_2seq_sse2(dst, src1, src2, len);
}

#endif
