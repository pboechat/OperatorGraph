#pragma once

#include "ClockCycleCounter.cuh"
#include "DebugFlags.h"
#include "GlobalVariables.cuh"
#include "Instrumentation.cuh"
#include "Shape.cuh"
#include "Symbol.cuh"
#include "TStdLib.h"

#include <cuda_runtime_api.h>
#include <procedureInterface.cuh>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace PGA
{
	namespace Operators
	{
		template <unsigned int OperationCodeT, unsigned int SuccessorOffsetT, int EdgeIndexT = -1>
		class PSCall
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				if (ContextT::Application::UseInstrumentation)
				{
					int localThreadId = threadId % NumThreadsT;
					uint64_t t0;
					int edgeIdx;
					if (localThreadId == 0)
					{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
						if (symbol->predecessor < 0 || symbol->predecessor >= GlobalVars::getNumEntries())
						{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("PGA::Operators::PSCall::execute(..): invalid entry index [symbol->predecessor=%d, GlobalVars::getNumEntries()=%d] (CUDA thread %d %d)\n", symbol->predecessor, GlobalVars::getNumEntries(), threadIdx.x, blockIdx.x);
							asm("trap;");
#else
							throw std::runtime_error(("PGA::Operators::PSCall::execute(..): invalid entry index [symbol->predecessor=" + symbol->predecessor + ", GlobalVars::getNumEntries()=" + std::to_string(GlobalVars::getNumEntries()) + "]").c_str());
#endif
						}
#endif
						auto& entry = GlobalVars::getDispatchTableEntry(symbol->predecessor);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
						if (EdgeIndexT > static_cast<int>(entry.numEdgeIndices))
						{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("PGA::Operators::PSCall::execute(..): out of boundaries edge index access [symbol->predecessor=%d, entry.numEdgeIndices=%d, EdgeIndexT=%d] (CUDA thread %d %d)\n", symbol->predecessor, entry.numEdgeIndices, EdgeIndexT, threadIdx.x, blockIdx.x);
							asm("trap;");
#else
							throw std::runtime_error(("PGA::Operators::PSCall::execute(..): out of boundaries edge index access [symbol->predecessor=" + symbol->predecessor +", entry.numEdgeIndices=" + std::to_string(entry.numEdgeIndices) +", EdgeIndexT=" + std::to_string(EdgeIndexT) + "]").c_str());
#endif
						}
#endif
						edgeIdx = entry.edgeIndices[EdgeIndexT];
						if (T::IsEnabled<DebugFlags::EdgeTraversal>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("Traversing partially-static edge [edgeIdx=%d, symbol->entryIndex=%d, predecessor=%d, OperationCodeT=%d, SuccessorOffsetT=%d] (CUDA thread %d %d)\n", edgeIdx, symbol->entryIndex, symbol->predecessor, OperationCodeT, SuccessorOffsetT, threadIdx.x, blockIdx.x);
#else
							std::cout << "Traversing partially-static edge [edgeIdx=" << edgeIdx << ", symbol->entryIndex=" << symbol->entryIndex << ", predecessor=" << symbol->predecessor << ", OperationCodeT=" << OperationCodeT << ", SuccessorOffsetT=" << SuccessorOffsetT << "]" << std::endl;
#endif
						t0 = ClockCycleCounter::count();
					}
					int phaseIndex = GlobalVars::getDispatchTableEntry(symbol->predecessor).successors[SuccessorOffsetT].phaseIndex;
					ContextT::Application::template dispatchNonTerminal<OperationCodeT>(*symbol, phaseIndex, 0, queue);
					if (localThreadId == 0)
					{
						uint64_t t1 = ClockCycleCounter::count();
						Instrumentation::writeEdgeData(t0, t1, edgeIdx, 2);
					}
				}
				else
				{
					if (T::IsEnabled<DebugFlags::EdgeTraversal>::Result)
					{
						if ((threadId % NumThreadsT) == 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("Traversing partially-static edge [symbol->entryIndex=%d, predecessor=%d, OperationCodeT=%d, SuccessorOffsetT=%d] (CUDA thread %d %d)\n", symbol->entryIndex, symbol->predecessor, OperationCodeT, SuccessorOffsetT, threadIdx.x, blockIdx.x);
#else
							std::cout << "Traversing partially-static edge [symbol->entryIndex" << symbol->entryIndex << ", predecessor=" << symbol->predecessor << ", OperationCodeT=" << OperationCodeT << ", SuccessorOffsetT="<< SuccessorOffsetT <<"]" << std::endl;
#endif
					}
					int phaseIndex = GlobalVars::getDispatchTableEntry(symbol->predecessor).successors[SuccessorOffsetT].phaseIndex;
					ContextT::Application::template dispatchNonTerminal<OperationCodeT>(*symbol, phaseIndex, 0, queue);
				}
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "PSCall";
			}

			__host__ __inline__ static std::string toString()
			{
				return "PSCall<" + std::to_string(OperationCodeT) + ", " + std::to_string(EdgeIndexT) + ">";
			}

		};

		template <unsigned int OperationCodeT, unsigned int SuccessorOffsetT>
		class PSCall < OperationCodeT, SuccessorOffsetT, -1 >
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				static_assert(!ContextT::Application::UseInstrumentation, "PSCall has edge index when it shouldn't have or it misses edge index when it shouldn't miss");
				if (T::IsEnabled<DebugFlags::EdgeTraversal>::Result)
				{
					if ((threadId % NumThreadsT) == 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("Traversing partially-static edge [symbol->entryIndex=%d, predecessor=%d, OperationCodeT=%d, SuccessorOffsetT=%d] (CUDA thread %d %d)\n", symbol->entryIndex, symbol->predecessor, OperationCodeT, SuccessorOffsetT, threadIdx.x, blockIdx.x);
#else
						std::cout << "Traversing partially-static edge [symbol->entryIndex" << symbol->entryIndex << ", predecessor=" << symbol->predecessor << ", OperationCodeT=" << OperationCodeT << ", SuccessorOffsetT=" << SuccessorOffsetT << "]" << std::endl;
#endif
				}
				int phaseIndex = GlobalVars::getDispatchTableEntry(symbol->predecessor).successors[SuccessorOffsetT].phaseIndex;
				ContextT::Application::template dispatchNonTerminal<OperationCodeT>(*symbol, phaseIndex, 0, queue);
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "PSCall";
			}

			__host__ __inline__ static std::string toString()
			{
				return "PSCall<" + std::to_string(OperationCodeT) + ">";
			}

		};

		template <unsigned int OperationCodeT, int PhaseIndexT, int EdgeIndexT = -1, int EntryIndexT = -1>
		class FSCall
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				if (ContextT::Application::UseInstrumentation)
				{
					int localThreadId = threadId % NumThreadsT;
					uint64_t t0;
					int edgeIdx;
					if (localThreadId == 0)
					{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
						if (symbol->predecessor < 0 || symbol->predecessor >= GlobalVars::getNumEntries())
						{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("PGA::Operators::FSCall::execute(..): invalid entry index [symbol->predecessor=%d, GlobalVars::getNumEntries()=%d] (CUDA thread %d %d)\n", symbol->predecessor, GlobalVars::getNumEntries(), threadIdx.x, blockIdx.x);
							asm("trap;");
#else
							throw std::runtime_error(("PGA::Operators::FSCall::execute(..): invalid entry index [symbol->predecessor=" + symbol->predecessor + ", GlobalVars::getNumEntries()=" + std::to_string(GlobalVars::getNumEntries()) + "]").c_str());
#endif
						}
#endif
						auto& entry = GlobalVars::getDispatchTableEntry(symbol->predecessor);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
						if (EdgeIndexT > static_cast<int>(entry.numEdgeIndices))
						{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("PGA::Operators::FSCall::execute(..): out of boundaries edge index access [symbol->predecessor=%d, entry.numEdgeIndices=%d, EdgeIndexT=%d] (CUDA thread %d %d)\n", symbol->predecessor, entry.numEdgeIndices, EdgeIndexT, threadIdx.x, blockIdx.x);
							asm("trap;");
#else
							throw std::runtime_error(("PGA::Operators::FSCall::execute(..): out of boundaries edge index access [symbol->predecessor=" + symbol->predecessor + ", entry.numEdgeIndices=" + std::to_string(entry.numEdgeIndices) + ", EdgeIndexT=" + std::to_string(EdgeIndexT) + "]").c_str());
#endif
						}
#endif
						edgeIdx = entry.edgeIndices[EdgeIndexT];
						if (T::IsEnabled<DebugFlags::EdgeTraversal>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("Traversing fully-static edge [edgeIdx=%d, symbol->entryIndex=%d, predecessor=%d, OperationCodeT=%d] (CUDA thread %d %d)\n", edgeIdx, symbol->entryIndex, symbol->predecessor, OperationCodeT, threadIdx.x, blockIdx.x);
#else
							std::cout << "Traversing fully-static edge [edgeIdx=" << edgeIdx << ", symbol->entryIndex=" << symbol->entryIndex << ", predecessor=" << symbol->predecessor << ", OperationCodeT=" << OperationCodeT << "]" << std::endl;
#endif
						t0 = ClockCycleCounter::count();
					}
					ContextT::Application::template dispatchNonTerminal<OperationCodeT>(*symbol, PhaseIndexT, 0, queue);
					if (localThreadId == 0)
					{
						uint64_t t1 = ClockCycleCounter::count();
						Instrumentation::writeEdgeData(t0, t1, edgeIdx, 1);
					}
				}
				else
				{
					if (T::IsEnabled<DebugFlags::EdgeTraversal>::Result)
					{
						if ((threadId % NumThreadsT) == 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("Traversing fully-static edge [symbol->entryIndex=%d, predecessor=%d, OperationCodeT=%d] (CUDA thread %d %d)\n", symbol->entryIndex, symbol->predecessor, OperationCodeT, threadIdx.x, blockIdx.x);
#else
							std::cout << "Traversing fully-static edge [symbol->entryIndex=" << symbol->entryIndex << ", predecessor=" << symbol->predecessor << ", OperationCodeT=" << OperationCodeT << "]" << std::endl;
#endif
					}
					ContextT::Application::template dispatchNonTerminal<OperationCodeT>(*symbol, PhaseIndexT, 0, queue);
				}
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "FSCall";
			}

			__host__ __inline__ static std::string toString()
			{
				std::string str = "FSCall<" + std::to_string(OperationCodeT);
				str += ", " + std::to_string(EdgeIndexT);
				if (EntryIndexT != -1)
					str += ", " + std::to_string(EntryIndexT);
				str += ">";
				return str;
			}

		};

		template <unsigned int OperationCodeT, int PhaseIndexT, int EntryIndexT>
		class FSCall < OperationCodeT, PhaseIndexT, -1, EntryIndexT >
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				static_assert(!ContextT::Application::UseInstrumentation, "FSCall has edge index when it shouldn't have or it misses edge index when it shouldn't miss");
				if (T::IsEnabled<DebugFlags::EdgeTraversal>::Result)
				{
					if ((threadId % NumThreadsT) == 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("Traversing fully-static edge [symbol->entryIndex=%d, predecessor=%d, OperationCodeT=%d] (CUDA thread %d %d)\n", symbol->entryIndex, symbol->predecessor, OperationCodeT, threadIdx.x, blockIdx.x);
#else
						std::cout << "Traversing fully-static edge [symbol->entryIndex=" << symbol->entryIndex << ", predecessor=" << symbol->predecessor << ", OperationCodeT=" << OperationCodeT << "]" << std::endl;
#endif
				}
				ContextT::Application::template dispatchNonTerminal<OperationCodeT>(*symbol, PhaseIndexT, 0, queue);
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "FSCall";
			}

			__host__ __inline__ static std::string toString()
			{
				std::string str = "FSCall<" + std::to_string(OperationCodeT);
				if (EntryIndexT != -1)
					str += ", " + std::to_string(EntryIndexT);
				str += ">";
				return str;
			}

		};

		template <unsigned int SuccessorOffsetT, int EdgeIndexT = -1>
		class DCall
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				if (ContextT::Application::UseInstrumentation)
				{
					int localThreadId = threadId % NumThreadsT;
					uint64_t t0;
					int edgeIdx;
					if (localThreadId == 0)
					{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
						if (symbol->predecessor < 0 || symbol->predecessor >= GlobalVars::getNumEntries())
						{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("PGA::Operators::DCall::execute(..): invalid entry index [symbol->predecessor=%d, GlobalVars::getNumEntries()=%d] (CUDA thread %d %d)\n", symbol->predecessor, GlobalVars::getNumEntries(), threadIdx.x, blockIdx.x);
							asm("trap;");
#else
							throw std::runtime_error(("PGA::Operators::DCall::execute(..): invalid entry index [symbol->predecessor=" + std::to_string(symbol->predecessor) + ", GlobalVars::getNumEntries()=" + std::to_string(GlobalVars::getNumEntries()) + "]").c_str());
#endif
						}
#endif
						auto& entry = GlobalVars::getDispatchTableEntry(symbol->predecessor);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
						if (EdgeIndexT > static_cast<int>(entry.numEdgeIndices))
						{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("PGA::Operators::DCall::execute(..): out of boundaries edge index access [symbol->predecessor=%d, entry.numEdgeIndices=%d, EdgeIndexT=%d] (CUDA thread %d %d)\n", symbol->predecessor, entry.numEdgeIndices, EdgeIndexT, threadIdx.x, blockIdx.x);
							asm("trap;");
#else
							throw std::runtime_error(("PGA::Operators::DCall::execute(..): out of boundaries edge index access [symbol->predecessor=" + std::to_string(symbol->predecessor) + ", entry.numEdgeIndices=" + std::to_string(entry.numEdgeIndices) + ", EdgeIndexT=" + std::to_string(EdgeIndexT) + "]").c_str());
#endif
						}
#endif
						edgeIdx = entry.edgeIndices[EdgeIndexT];
						if (T::IsEnabled<DebugFlags::EdgeTraversal>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("Traversing dynamic edge [edgeIdx=%d, entryIndex=%d, predecessor=%d, SuccessorOffsetT=%d] (CUDA thread %d %d)\n", edgeIdx, symbol->entryIndex, symbol->predecessor, SuccessorOffsetT, threadIdx.x, blockIdx.x);
#else
							std::cout << "Traversing dynamic edge [edgeIdx=" << edgeIdx <<  ", entryIndex=" << symbol->entryIndex << ", predecessor=" << symbol->predecessor << ", SuccessorOffsetT=" << SuccessorOffsetT << "]" << std::endl;
#endif
						t0 = ClockCycleCounter::count();
					}
					int phaseIndex = GlobalVars::getDispatchTableEntry(symbol->predecessor).successors[SuccessorOffsetT].phaseIndex;
					ContextT::Application::dispatchNonTerminal(*symbol, phaseIndex, 0, queue);
					if (localThreadId == 0)
					{
						uint64_t t1 = ClockCycleCounter::count();
						Instrumentation::writeEdgeData(t0, t1, edgeIdx, 3);
					}
				}
				else
				{
					if (T::IsEnabled<DebugFlags::EdgeTraversal>::Result)
					{
						if ((threadId % NumThreadsT) == 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("Traversing dynamic edge [entryIndex=%d, predecessor=%d, SuccessorOffsetT=%d] (CUDA thread %d %d)\n", symbol->entryIndex, symbol->predecessor, SuccessorOffsetT, threadIdx.x, blockIdx.x);
#else
							std::cout << "Traversing dynamic edge [entryIndex=" << symbol->entryIndex << ", predecessor=" << symbol->predecessor << ", SuccessorOffsetT=" << SuccessorOffsetT << "]" << std::endl;
#endif
					}
					int phaseIndex = GlobalVars::getDispatchTableEntry(symbol->predecessor).successors[SuccessorOffsetT].phaseIndex;
					ContextT::Application::dispatchNonTerminal(*symbol, phaseIndex, 0, queue);
				}
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "DCall";
			}

			__host__ __inline__ static std::string toString()
			{
				std::string str = "DCall<" + std::to_string(SuccessorOffsetT);
				if (EdgeIndexT != -1)
					str += ", " + std::to_string(EdgeIndexT);
				str += ">";
				return str;
			}

		};

		template <unsigned int SuccessorOffsetT>
		class DCall < SuccessorOffsetT, -1 >
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				static_assert(!ContextT::Application::UseInstrumentation, "DCall has edge index when it shouldn't have or it misses edge index when it shouldn't miss");
				if (T::IsEnabled<DebugFlags::EdgeTraversal>::Result)
				{
					if ((threadId % NumThreadsT) == 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("Traversing dynamic edge [entryIndex=%d, predecessor=%d, SuccessorOffsetT=%d] (CUDA thread %d %d)\n", symbol->entryIndex, symbol->predecessor, SuccessorOffsetT, threadIdx.x, blockIdx.x);
#else
						std::cout << "Traversing dynamic edge [entryIndex=" << symbol->entryIndex << ", predecessor=" << symbol->predecessor << ", SuccessorOffsetT=" << SuccessorOffsetT << "]" << std::endl;
#endif
				}
				int phaseIndex = GlobalVars::getDispatchTableEntry(symbol->predecessor).successors[SuccessorOffsetT].phaseIndex;
				ContextT::Application::dispatchNonTerminal(*symbol, phaseIndex, 0, queue);
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "DCall";
			}

			__host__ __inline__ static std::string toString()
			{
				return "DCall<" + std::to_string(SuccessorOffsetT) + ">";
			}

		};

	}

}
