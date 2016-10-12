


#ifndef INCLUDED_CUDA_PTR
#define INCLUDED_CUDA_PTR

#pragma once


template <typename T>
class cuda_ptr
{
private:
  cuda_ptr(const cuda_ptr& p);
  cuda_ptr& operator =(const cuda_ptr& p);

  T* ptr;

  static void release(T* ptr)
  {
    if (ptr != nullptr)
      cudaFree(ptr);
  }

public:
  explicit cuda_ptr(T* ptr = nullptr)
    : ptr(ptr)
  {
  }

  cuda_ptr(cuda_ptr&& p)
    : ptr(p.ptr)
  {
    p.ptr = nullptr;
  }

  ~cuda_ptr()
  {
    release(ptr);
  }

  cuda_ptr& operator =(cuda_ptr&& p)
  {
    std::swap(ptr, p.ptr);
    return *this;
  }

  void release()
  {
    release(ptr);
    ptr = nullptr;
  }

  T** bind()
  {
    release(ptr);
    return &ptr;
  }

  T* unbind()
  {
    T* temp = ptr;
    ptr = nullptr;
    return temp;
  }

  T* operator ->() const { return ptr; }

  T& operator *() const { return *ptr; }

  operator T*() const { return ptr; }

};


#include <memory>
#include <cuda_runtime_api.h>
#include "utils.h"

struct cuda_deleter
{
  void operator()(void* ptr)
  {
    cudaFree(ptr);
  }
};

template <typename T>
inline std::unique_ptr<T, cuda_deleter> cudaAlloc()
{
  void* ptr;
  printf("trying to allocate %.2f MB cuda buffer (%d bytes)\n", sizeof(T) * 1.0 / (1024.0 * 1024.0), sizeof(T));
  CUDA_CHECKED_CALL(cudaMalloc(&ptr, sizeof(T)));
  return std::unique_ptr<T, cuda_deleter>(static_cast<T*>(ptr));
}

template <typename T>
inline std::unique_ptr<T[], cuda_deleter> cudaAllocArray(size_t N)
{
  void* ptr;
  printf("trying to allocate %.2f MB cuda buffer (%d * %d bytes)\n", N * sizeof(T) * 1.0 / (1024.0 * 1024.0), N, sizeof(T));
  CUDA_CHECKED_CALL(cudaMalloc(&ptr, N * sizeof(T)));
  return std::unique_ptr<T[], cuda_deleter>(static_cast<T*>(ptr));
}


#endif  // INCLUDED_CUDA_PTR
