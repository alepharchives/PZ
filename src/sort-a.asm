;;  Copyright (C) 2011  Alejo Sanchez www.ologan.com
;;
;;  This program is free software: you can redistribute it and/or modify
;;  it under the terms of the GNU Affero General Public License as published by
;;  the Free Software Foundation, either version 3 of the License, or
;;  (at your option) any later version.
;;
;;  This program is distributed in the hope that it will be useful,
;;  but WITHOUT ANY WARRANTY; without even the implied warranty of
;;  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;;  GNU Affero General Public License for more details.
;;
;;  You should have received a copy of the GNU Affero General Public License
;;  along with this program.  If not, see <http:;;www.gnu.org/licenses/>.

%include "x86inc.asm"
%include "x86util.asm"

SECTION_RODATA

; for reversing registers of 8 elements
;;const pb_reverse,  db 7, 6, 5, 4, 3, 2, 1, 0

SECTION .text

; minmax + xor exchange, sse2 version
;    mask = min(r1, r2)
;    t0 = (r1 ^ r2) & mask
;    r1 ^= t0   mins
;    r2 ^= t0   maxs
%macro MINMAXD 4 ; %1 reg1 %2 reg2 %3 mask %4 tmp
    mova    m%3, m%1 ; copy for mask
    mova    m%4, m%1 ; copy for tmp
    pcmpgtd m%3, m%2 ; comp dw  %3 has mask
    pxor    m%4, m%2 ; tmp ^= reg2
    pand    m%4, m%3 ; tmp &= mask
    pxor    m%1, m%4 ; reg1 ^= tmp  min
    pxor    m%2, m%4 ; reg2 ^= tmp  max
%endmacro

%macro COLUMNSORT_4x4D 6 ; 2 reg 2 aux
    MINMAXD %1, %3, %5, %6
    MINMAXD %2, %4, %5, %6
    MINMAXD %1, %2, %5, %6
    MINMAXD %3, %4, %5, %6
    MINMAXD %2, %3, %5, %6
%endmacro

; register sort 4 at a time
%macro REGSORT      6 ; uses 6 xmm regs
    COLUMNSORT_4x4D %1, %2, %3, %4, %5, %6 ; .6 cycles/elem
    TRANSPOSE4x4D   %1, %2, %3, %4, %5     ; .3 cycles/elem
%endmacro

;       A4 A3 A2 A1
;       B4 B3 B2 B1
; reg1  A3 A2 A1 A0
; reg2  B3 B2 B1 B0
%macro BITONICL1D 3 ; %1 reg1, %2 reg2, %3 tmp0
    movaps  m%3, m%2 ; save reg2 orig
    movhlps m%2, m%1 ; B1B0A1A0 (reg2 reg1)
    movlhps m%1, m%3 ; B3B2A3A2 (reg1 reg 2)
%endmacro

;       0 1 2 3
;       4 5 6 7
; L2P:  0 4 2 6
; H2P:  1 5 3 7
%macro BITONICL2D 4 ; %1 reg1, %2 reg2, %3 tmp1 %4 tmp2
    mova      m%3, m%1 ; save copy reg1
    unpcklps  m%1, m%2 ; t2 = unpcklps(r1,r2)
    unpckhps  m%3, m%2 ; t1 = unpckhps(r1,r2)
    mova      m%4, m%1 ; save copy reg1
    mova      m%2, m%3 ; r2 = t1
    movlhps   m%1, m%3 ; t2 = movlhps  (t2,t1)
    movhlps   m%2, m%4 ; t1 = movhlps  (t2,t1)
%endmacro

;       0 1 2 3
;       4 5 6 7
; L3P:  0 4 1 5
; H3P:  2 6 3 7
%macro BITONICL3D 3 ; %1 reg1, %2 reg2, %3 tmp1
    mova      m%3, m%1 ; t1 = r1
    unpcklps  m%1, m%2 ; r1 = unpcklps(r1,r2)
    unpckhps  m%3, m%2 ; t1 = unpcklps(t1,r2)
    SWAP      m%3, m%2 ; r2 = t1
%endmacro

;%macro CRI 5 ; takes 2 add
;    mova      m%2, [%1+ 0]       ; load first
;    mova      m%3, [%1+16]       ; load first
;    BITONICL3D  0, 1, 2
;    mova  [%1+ 0], m%2
;    mova  [%1+16], m%3
;%endmacro

%macro BITONIC_MERGED 4 ; takes 2 add
    pshufd      %2, %2, 0x1b      ; reverse second
    MINMAXD     %1, %2, %3, %4
    BITONICL1D  %1, %2, %3
    MINMAXD     %1, %2, %3, %4
    BITONICL2D  %1, %2, %3, %4
    MINMAXD     %1, %2, %3, %4
    BITONICL3D  %1, %2, %3
%endmacro

%macro BITONIC_MERGE_P4_D 6 ; merge %1/%2 and %3/%4, aux %5 %6
    pshufd      %2, %2, 0x1b      ; reverse second
    MINMAXD     %1, %2, %5, %6
    BITONICL1D  %1, %2, %5
    MINMAXD     %1, %2, %5, %6
    BITONICL2D  %1, %2, %5, %6
    MINMAXD     %1, %2, %5, %6
    BITONICL3D  %1, %2, %5
      pshufd      %4, %4, 0x1b      ; reverse second
      MINMAXD     %3, %4, %5, %6
      BITONICL1D  %3, %4, %5
      MINMAXD     %3, %4, %5, %6
      BITONICL2D  %3, %4, %5, %6
      MINMAXD     %3, %4, %5, %6
      BITONICL3D  %3, %4, %5
%endmacro

; void cri(int n, int32_t *elem, int32_t *aux)
cglobal cri, 3,3,16 ; n, src
INIT_XMM
    shr  r0, 5   ; Consume 4x4 elements at a time
.bitonic_loop:
    ; Load 4 xmm regs
    mova      m0, [r1+ 0]       ; load first
    mova      m1, [r1+16]       ; load first
    mova      m2, [r1+32]       ; load first
    mova      m3, [r1+48]       ; load first
    ;; Column sort to obtain 4 sorted registers
    REGSORT 0, 1, 2, 3, 14, 15    ; 4 regs 2 aux
    ; Merge sort first pair of registers
    BITONIC_MERGED 0, 1, 14, 15   ; 2 regs 2 aux
    ; Merge sort second pair of registers
    BITONIC_MERGED 2, 3, 14, 15   ; 2 regs 2 aux
    ; Merge 0-2, 2-min(1,3), 
    ;
    ; Merge pairs 0-1 and 2-3
    ;
    BITONIC_MERGED 0, 2, 14, 15   ; 2 regs 2 aux
    MINMAXD    1, 3, 14, 15        ; lowest to merge with m2
    BITONIC_MERGED 1, 2, 14, 15   ; 2 regs 2 aux
    BITONIC_MERGED 2, 3, 14, 15   ; 2 regs 2 aux
    mova  [r1+ 0], m0
    mova  [r1+16], m1
    mova  [r1+32], m2
    mova  [r1+48], m3
    add        r1, 64  ; 2x4x4 bytes consumed
    dec        r0
    jg .bitonic_loop
    RET

%macro SORTW 1
cglobal sort_%1, 3,4 ; use 1 more reg and less than 6 mmx/sse
    ;; r0=n, r1=d[n], r2=aux[n]
    sub     r2, r1   ; aux as offset of d
    ;shr    r0d, 9   ; divide by 2*8 elems
    shr    r0d, 4    ; // Consume 4x4 elements at a time (n/16*4B)
.regsort_loop:
    add    r1, 64 ; move forward 16 elements (4*mmsize)
    dec    r0d    ; 4 regs done  (sets flag if 0)
    jg     .regsort_loop
    REP_RET       ; RET with Athlon friendly format
%endmacro

INIT_XMM
SORTW ssse3

