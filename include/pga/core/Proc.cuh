#pragma once

#include "ClockCycleCounter.cuh"
#include "DebugFlags.h"
#include "GlobalVariables.cuh"
#include "Instrumentation.cuh"
#include "Symbol.cuh"
#include "TStdLib.h"

#include <cuda_runtime_api.h>
#include <procedureInterface.cuh>

#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace PGA
{
	template <typename ShapeT, typename OperatorT, unsigned int NumThreadsT = 1>
	class Proc : public ::Procedure
	{
	public:
		typedef OperatorT FirstOperator;
		static_assert(NumThreadsT >= 1, "Num. threads cannot be less than 1");
		typedef Symbol<ShapeT> ExpectedData;
		static const int NumThreads = NumThreadsT;
		static const bool ItemInput = true;

		template <typename QueueT, typename ContextT>
		__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, ExpectedData* symbol, unsigned int* shared)
		{
			int localThreadId = threadId % NumThreads;
			int ProcIdx;
			if (ContextT::Application::UseInstrumentation)
			{
				int subGraphIdx;
				uint64_t t0;
				if (localThreadId == 0)
				{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
					if (symbol->entryIndex < 0 || symbol->entryIndex >= GlobalVars::getNumEntries())
					{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("PGA::Proc::execute(..): invalid entry index [symbol->entryIndex=%d, GlobalVars::getNumEntries()=%d] (CUDA thread %d %d)\n", symbol->entryIndex, GlobalVars::getNumEntries(), threadIdx.x, blockIdx.x);
						asm("trap;");
#else
						throw std::runtime_error(("PGA::Proc::execute(..): invalid entry index [symbol->entryIndex=" + std::to_string(symbol->entryIndex) + ", GlobalVars::getNumEntries()=" + std::to_string(GlobalVars::getNumEntries()) + "]").c_str());
#endif
					}
#endif
					subGraphIdx = GlobalVars::getDispatchTableEntry(symbol->entryIndex).subGraphIndex;
					if (T::IsEnabled<DebugFlags::VertexVisit>::Result)
					{
						ProcIdx = ContextT::Application::ProcedureList::template IndexOf<Proc>::Result;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("Vertex visit enter [ProcIdx=%d, subGraphIdx=%d] (CUDA thread %d %d)\n", ProcIdx, subGraphIdx, threadIdx.x, blockIdx.x);
#else
						std::cout << "Vertex visit enter [ProcIdx=" << ProcIdx << ", subGraphIdx=" << subGraphIdx << "]" << std::endl;
#endif
					}
					t0 = ClockCycleCounter::count();
				}
				FirstOperator::template execute<ContextT, NumThreads>(threadId, numThreads, queue, symbol, shared);
				if (localThreadId == 0)
				{
					uint64_t t1 = ClockCycleCounter::count();;
					Instrumentation::writeSubGraphData(t0, t1, subGraphIdx);
					if (T::IsEnabled<DebugFlags::VertexVisit>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("Vertex visit exit [ProcIdx=%d, subGraphIdx=%d] (CUDA thread %d %d)\n", ProcIdx, subGraphIdx, threadIdx.x, blockIdx.x);
#else
						std::cout << "Vertex visit exit [ProcIdx=" << ProcIdx << ", subGraphIdx=" << subGraphIdx << "]" << std::endl;
#endif
				}
			}
			else
			{
				if (T::IsEnabled<DebugFlags::VertexVisit>::Result)
				{
					if (localThreadId == 0)
					{
						ProcIdx = ContextT::Application::ProcedureList::template IndexOf<Proc>::Result;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("Vertex visit enter [ProcIdx=%d] (CUDA thread %d %d)\n", ProcIdx, threadIdx.x, blockIdx.x);
#else
						std::cout << "Vertex visit enter [ProcIdx=" << ProcIdx << "]" << std::endl;
#endif
					}
				}
				FirstOperator::template execute<ContextT, NumThreads>(threadId, numThreads, queue, symbol, shared);
				if (T::IsEnabled<DebugFlags::VertexVisit>::Result)
				{
					if (localThreadId == 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("Vertex visit exit [ProcIdx=%d] (CUDA thread %d %d)\n", ProcIdx, threadIdx.x, blockIdx.x);
#else
						std::cout << "Vertex visit exit [ProcIdx=" << ProcIdx << "]" << std::endl;
#endif
				}
			}
		}

	};

}
