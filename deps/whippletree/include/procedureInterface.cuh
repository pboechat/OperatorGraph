#pragma once

#include <cuda_runtime_api.h>

#include "procedureDescription.h"


class Procedure
{
public:
  //static const int ProcedureId = 0;
  static const int NumThreads = 0;
  static const bool ItemInput = false;
  static const int sharedMemory = 0;
  static const bool InitialProcedure = false;
  typedef int ExpectedData;

  static const char* name() { return "Unnamed_"; }// + std::to_string((unsigned long long)ProcedureId); }
  static std::string algorithmname() { return std::string("UnknownAlgorithm"); }
  

  template<class Q, class Sync>
  static __device__ __inline__ void execute(int threadId, int numThreads, Q* queue, ExpectedData* data, unsigned int* shared) {  }
};

class NoProcedure : public Procedure
{
public:
  //static const int ProcedureId = -1;
  static const int myid = -1;
 
  static const char* name() { return "NoProcedure"; }

  template<class Q, class Sync>
  static __device__ __inline__ void execute(int threadId, int numThreads, Q* queue, ExpectedData* data, unsigned int* shared) 
  {  
    printf("ERROR: NoProcedure executed\n");
  }
};

template<class PROCEDURE>
__device__ __inline__ int getThreadCount()
{
  return PROCEDURE::NumThreads > 0 ? PROCEDURE::NumThreads : (PROCEDURE::ItemInput ? 1 : blockDim.x);
}
template<class PROCEDURE, bool MultiElement>
__device__ __inline__ int getElementCount()
{
  if(!MultiElement && !PROCEDURE::ItemInput)
    return 1;
  return PROCEDURE::NumThreads > 0 ? blockDim.x/PROCEDURE::NumThreads : (PROCEDURE::ItemInput ? blockDim.x : 1);
}
template<class PROCEDURE, bool MultiElement>
__device__ __inline__ int getThreadOffset()
{
  if(!MultiElement && !PROCEDURE::ItemInput)
    return 0;
  return PROCEDURE::NumThreads > 0 ? threadIdx.x/PROCEDURE::NumThreads : (PROCEDURE::ItemInput ? threadIdx.x : 0);
}

