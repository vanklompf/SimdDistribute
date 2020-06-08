#include <stdint.h>
#include <intrin.h>
#include <immintrin.h>

#include "Queue.h"


__m256i afk_mm256_lzcnt_epi32(__m256i vec) {
  uint32_t* in_vec = (uint32_t*)&vec;
  __m256i out;
  uint32_t* out_vec = (uint32_t*)&out;

  for (int i=0; i<8; i++) 
    out_vec[i] = __lzcnt(in_vec[i]);

  return out;
}

__mmask8 afk_mm256_cmplt_epi32_mask(__m256i a, __m256i b)
{
  uint32_t* in_vec_a = (uint32_t*)&a;
  uint32_t* in_vec_b = (uint32_t*)&b;
  __mmask8 out = 0;

  for (int i = 0; i < 8; i++)
    if (in_vec_a[0] < in_vec_b[0]) out &= (1 << i);

  return out;
}

__m256i afk_mm256_mask_compress_epi32(__m256i dst, __mmask8 mask, __m256i src) {
  uint32_t* src_vec = (uint32_t*)& src;
  uint32_t* dst_vec = (uint32_t*)& dst;

  int cnt = 0;
  for (int i = 0; i < 8; i++) {
    if (mask & (i << i))
      dst_vec[cnt++] = src_vec[i];
  }

  for (int i = cnt; i < 8; i++)
    dst_vec[i] = src_vec[i];

  return dst;
}


void basicDistribute(const uint32_t* bitArray, size_t bitArraySize, Queue** queues) {
  for (auto i = 0; i < bitArraySize; i++) {
    uint32_t mask = bitArray[i];
    while (mask) {
      unsigned long queueId;
      _BitScanForward(&queueId, mask);
      mask &= ~(1U << queueId);
      queues[queueId]->add(i);
    }
  }
}

/*
static void avx256DistributeCompressImpl(const uint32_t* bitArray, size_t bitArraySize, Queue** queues) {
  __m256i cntoffset_vec = _mm256_set1_epi32(31);
  __m256i shift_vec = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  __m256i all_ones_vec = _mm256_set1_epi32(0x7FFFFFFF);

  for (auto i = 0; i < bitArraySize; i+=8) {
    __m256i bitfield_vec = _mm256_loadu_si256((__m256i *)&bitArray[i]);

    __mmask8 res_mask = 0;
    do {
      // count leading zeros
      __m256i res_vec = afk_mm256_lzcnt_epi32(bitfield_vec);

      // prepare clear mask
      __m256i clear_mask_vec = _mm256_srlv_epi32(all_ones_vec, res_vec);

      // clear counted bits
      bitfield_vec = _mm256_and_si256(bitfield_vec, clear_mask_vec);

      // 
      res_mask = afk_mm256_cmplt_epi32_mask(bitfield_vec, cntoffset_vec);

      // 
      res = _mm256_sub_epi32(cntoffset_vec, res);

      res = afk_mm256_mask_compress_epi32(res, res_mask, res);

      __m256i res_idx = afk_mm256_mask_compress_epi32(res_idx, res_mask, shift_vec);

      const auto cnt = _mm_popcnt_u32(res_mask);
      int* res_array = (int*)& res;
      int* idx_array = (int*)& res_idx;
      for (auto x = 0; x < cnt; x++) {
        queues[res_array[x]]->add(i + idx_array[x]);
      }

    } while (res_mask);
  }
}*/

static void avx256DistributeImpl(const uint32_t* bitArray, size_t bitArraySize, Queue** queues) {
  __m256i cntoffset_vec = _mm256_set1_epi32(31);
  __m256i all_ones_vec = _mm256_set1_epi32(0x7FFFFFFF);

  for (auto i = 0; i < bitArraySize; i += 8) {
    __m256i bitfield_vec = _mm256_loadu_si256((__m256i*) &bitArray[i]);

    __mmask8 res_mask = 0;
    bool finished = true;
    do {
      __m256i res_vec = afk_mm256_lzcnt_epi32(bitfield_vec);
      __m256i clear_mask_vec = _mm256_srlv_epi32(all_ones_vec, res_vec);
      bitfield_vec = _mm256_and_si256(bitfield_vec, clear_mask_vec);
      res_vec = _mm256_sub_epi32(cntoffset_vec, res_vec);

      int* res_array = (int*)& res_vec;
      finished = true;
      for (auto x = 0; x < 8; x++) {
        if (res_array[x] >= 0) {
          queues[res_array[x]]->add(i + x);
          finished = false;
        }
      }
    } while (!finished);
  }
}


void avx256Distribute(const uint32_t* bitArray, size_t bitArraySize, Queue** queues) {
  size_t fast_size = bitArraySize & ~0x7;
  size_t reminder_size = bitArraySize & 0x7;

  avx256DistributeImpl(bitArray, fast_size, queues);
  basicDistribute(bitArray + fast_size, reminder_size, queues);
}

/*
void avx256DistributeCompress(const uint32_t* bitArray, size_t bitArraySize, Queue** queues) {
  size_t fast_size = bitArraySize & ~0x7;
  size_t reminder_size = bitArraySize & 0x7;

  avx256DistributeCompressImpl(bitArray, fast_size, queues);
  basicDistribute(bitArray + fast_size, reminder_size, queues);
}*/