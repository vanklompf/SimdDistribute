#pragma once

#include <cstdint>
#include <immintrin.h>


__m256i afk_mm256_lzcnt_epi32(__m256i vec) {
    uint32_t* in_vec = (uint32_t*)&vec;
    __m256i out;
    uint32_t* out_vec = (uint32_t*)&out;

    for (int i = 0; i < sizeof(__m256i) / sizeof(uint32_t); i++)
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

__m256i afk_mm256_mask_compress_epi32(__m256i dst, __mmask8 mask, __m256i src) {
    uint32_t* src_vec = (uint32_t*)&src;
    uint32_t* dst_vec = (uint32_t*)&dst;

    int cnt = 0;
    for (int i = 0; i < 8; i++) {
        if (mask & (1 << i))
            dst_vec[cnt++] = src_vec[i];
    }

    for (int i = cnt; i < sizeof(__m256i) / sizeof(uint32_t); i++)
        dst_vec[i] = src_vec[i];

    return dst;
}
