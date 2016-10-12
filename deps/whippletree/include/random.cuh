#pragma once
#ifndef RANDOM_CUH_0JOWMKST
#define RANDOM_CUH_0JOWMKST




#include "common.cuh"

#if defined(__CUDACC__)

namespace qrandom
{
  __device__ __inline__
  int warp_rand()
  {
    clock_t m1 = (40009 + clock())/16*19281 + (811*Softshell::smid() + 127*Softshell::warpid()) * 8231;
    clock_t m2 = 36969 * (m1 & 65535) + (m1 >> 16);
    return m2 & 65535;
  }

  __device__ __inline__
  int rand()
  {
    clock_t m1 = (40009 + clock())/16*19281 + (61*threadIdx.x + 811*Softshell::smid() + 127*Softshell::warpid()) * 8231;
    clock_t m2 = 36969 * (m1 & 65535) + (m1 >> 16);
    return m2 & 65535;
  }

  static const int max = 65535;
  static const int Range = 65536;

  __device__ __inline__
  int frand()
  {
   return  rand() / static_cast<float>(max);
  }

  __device__ __inline__
  bool check(int percent)
  {
    return rand() < (max+1)*percent/100;
  }

  __device__ __inline__
  bool warp_check(int percent)
  {
    return warp_rand() < (max+1)*percent/100;
  }

  __device__ __inline__
  bool block_check(int percent)
  {
    __shared__ bool res;
    __syncthreads();
    if(threadIdx.x == 0)
      res = check(percent);
    __syncthreads();
    return res;
  }

}

#endif
#endif /* end of include guard: RANDOM_CUH_0JOWMKST */
