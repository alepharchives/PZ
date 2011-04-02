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

#include <stdint.h>
#include <xmmintrin.h>

typedef __v4si v4si; // For readability

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

static void bitonic_l1_exchange_4si_sse2(v4si *a, v4si *b) {

    __v4sf t0 = __builtin_ia32_movhlps ((__v4sf) *b,(__v4sf)  *a);
    __v4sf t1 = __builtin_ia32_movlhps ((__v4sf) *a,(__v4sf)  *b);
    *b = (v4si) t0;
    *a = (v4si) t1;
}

static void bitonic_l2_exchange_4si_sse2(v4si *a, v4si *b) {

    __v4sf t0 = __builtin_ia32_unpckhps ((__v4sf) *a, (__v4sf) *b);
    __v4sf t1 = __builtin_ia32_unpcklps ((__v4sf) *a, (__v4sf) *b);
    *a = (v4si) __builtin_ia32_movlhps (t1, t0);
    *b = (v4si) __builtin_ia32_movhlps (t0, t1);

}

static void bitonic_l3_exchange_4si_sse2(v4si *a, v4si *b) {

    __v4sf t = __builtin_ia32_unpcklps ((__v4sf) *a, (__v4sf) *b);
    *b = (v4si) __builtin_ia32_unpckhps ((__v4sf) *a, (__v4sf) *b);
    *a = (v4si) t;

}

static void bitonic_sort_4si_sse2(v4si *a, v4si *b) {

    *a = __builtin_ia32_pshufd (*a, 0x1B); // Reverse a

    minmax_4si_sse2(a, b);
    bitonic_l1_exchange_4si_sse2(a, b);
    minmax_4si_sse2(a,b);
    bitonic_l2_exchange_4si_sse2(a, b);
    minmax_4si_sse2(a,b);
    bitonic_l3_exchange_4si_sse2(a,b);

}

// Merge 2 pairs in parallel, same as bitonic(a,b) and bitonic(c,d)
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

static void sort_4x_4si_sse2(v4si *a, v4si *b, v4si *c, v4si *d) {

    column_sort_4si_sse2(a, b, c, d);

    bitonic_sort_2x_4si_sse2(a, b, c, d);

    // Merge-exchange both heads (a and c)
    bitonic_sort_4si_sse2(a, c);

    // a is ready, has lowest

    // Merge-exchange b and c
    bitonic_sort_4si_sse2(b, c);
    // b has 2nd lowest now

    // Merge-exchange c and d
    bitonic_sort_4si_sse2(c, d);

}

static void sort_8x_4si(v4si *a, v4si *b, v4si *c, v4si *d,
        v4si *e, v4si *f, v4si *g, v4si *h) {

    column_sort_4si_sse2(a, b, c, d);
    column_sort_4si_sse2(e, f, g, h);

    bitonic_sort_2x_4si_sse2(a, b, c, d);
    bitonic_sort_2x_4si_sse2(e, f, g, h);

    // Merge-exchange both heads (a and c)
    bitonic_sort_4si_sse2(a, c);
    bitonic_sort_4si_sse2(e, g);

    // a lowest of abcd
    // e lowest of efgh

    // Merge-exchange b and c
    bitonic_sort_4si_sse2(b, c);
    bitonic_sort_4si_sse2(f, g);
    // b has 2nd lowest now

    // Merge-exchange c and d
    bitonic_sort_4si_sse2(c, d);
    bitonic_sort_4si_sse2(g, h);

    // Both abcd and efgh sorted, merge both lists
    // Note: imperfect sort, but very good on average
    //       A perfect sort would involve a few conditionals
    //       (Later merges will probably minimize this)

    bitonic_sort_4si_sse2(a, e);
    bitonic_sort_4si_sse2(b, f);
    bitonic_sort_4si_sse2(c, g);
    bitonic_sort_4si_sse2(d, h);

}
