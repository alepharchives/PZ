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

//
// Bitonic full sort of 4 vectors of 4 32bit signed integers
//   Sorts 16 elements
static void sort_4x_4si_sse2(v4si *a, v4si *b, v4si *c, v4si *d) {

    column_sort_4si_sse2(a, b, c, d); // Get partial sort in registers

    bitonic_sort_2x_4si_sse2(a, b, c, d); // Sort (a,b) and (c,d)

    // Merge-exchange both heads (a and c)
    bitonic_sort_4si_sse2(a, c); // Merge sort (a,c)
    // a has lowest 4 elements of original 16
    bitonic_sort_4si_sse2(b, c); // Merge sort (b,c)
    // b has 2nd lowest 4 elements
    bitonic_sort_4si_sse2(c, d); // Merge sort (c,d)
    // c has 3rd lowest 4 elements, d has the higher 4

}

//
// Bitonic full sort of 8 vectors of 32bit signed integers
//   Sorts 32 elements
//   Intercalated sorting of the first 16 and second 16 (paths A, B)
static void sort_8x_4si(v4si *a, v4si *b, v4si *c, v4si *d,
        v4si *e, v4si *f, v4si *g, v4si *h) {

    column_sort_4si_sse2(a, b, c, d); // Column sort path A
    column_sort_4si_sse2(e, f, g, h); // Column sort path B

    bitonic_sort_2x_4si_sse2(a, b, c, d); // Sort path A
    bitonic_sort_2x_4si_sse2(e, f, g, h); // Sort path B

    bitonic_sort_4si_sse2(a, c); // Exchange heads path A
    bitonic_sort_4si_sse2(e, g); // Exchange heads path B
    // a has 4 lowest of original (abcd), e lowest of original (efgh)

    // Merge-exchange b and c
    bitonic_sort_4si_sse2(b, c); // Sort (b,c)
    bitonic_sort_4si_sse2(f, g); // Sort (f,g)
    // b and f have 2nd lowest of each path (A and B respectively)

    bitonic_sort_4si_sse2(c, d); // Merge (c,d)
    bitonic_sort_4si_sse2(g, h); // Merge (g,h)
    // c and g have 3rd lowest of each path (A and B respectively)

    // Finally, merge paths A and B
    bitonic_sort_4si_sse2(a, e); // Merge heads of paths A and B
    bitonic_sort_4si_sse2(b, f);
    bitonic_sort_4si_sse2(c, g);
    bitonic_sort_4si_sse2(d, h);

    // XXX: Should check b < f and so on
}
