#pragma once

#include <cstdint>
#include <immintrin.h>

template<typename __mXXX>
__mXXX afk_mm_lzcnt_epi32(__mXXX vec) {
    uint32_t* in_vec = (uint32_t*)&vec;
    __mXXX out;
    uint32_t* out_vec = (uint32_t*)&out;

    for (int i = 0; i < sizeof(__mXXX) / sizeof(uint32_t); i++)
        out_vec[i] = __lzcnt(in_vec[i]);

    return out;
}

__mmask8 afk_mm256_cmpneq_epi32_mask(__m256i a, __m256i b)
{
    uint32_t* in_vec_a = (uint32_t*)&a;
    uint32_t* in_vec_b = (uint32_t*)&b;
    __mmask8 out = 0;

    for (int i = 0; i < 8; i++)
        if (in_vec_a[0] != in_vec_b[0]) {
            out |= (1 << i);
        }

    return out;
}

__mmask16 afk_mm512_cmpneq_epi32_mask(__m256i a, __m256i b)
{
    uint32_t* in_vec_a = (uint32_t*)&a;
    uint32_t* in_vec_b = (uint32_t*)&b;
    __mmask16 out = 0;

    for (int i = 0; i < 16; i++)
        if (in_vec_a[0] != in_vec_b[0]) {
            out |= (1 << i);
        }

    return out;
}

template<typename __mXXX, typename __mmaskXX>
__m256i afk_mm_mask_compress_epi32(__mXXX dst, __mmaskXX mask, __mXXX src) {
    uint32_t* src_vec = (uint32_t*)&src;
    uint32_t* dst_vec = (uint32_t*)&dst;

    int cnt = 0;
    for (int i = 0; i < 8; i++) {
        if (mask & (1 << i))
            dst_vec[cnt++] = src_vec[i];
    }

    for (int i = cnt; i < sizeof(__mXXX) / sizeof(uint32_t); i++)
        dst_vec[i] = src_vec[i];

    return dst;
}
