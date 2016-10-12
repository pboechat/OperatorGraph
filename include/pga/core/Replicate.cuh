#pragma once

#include <cstdio>
#include <string>
#include <cuda_runtime_api.h>

#include <math/vector.h>
#include <math/matrix.h>

#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
#include "Shape.cuh"
#include "Symbol.cuh"
#include "Parameters.cuh"
#include "TStdLib.h"
#include "SymbolDecorator.cuh"

namespace PGA
{
	namespace Operators
	{
		template <typename... Params1T>
		class Replicate
		{
		private:
			static const unsigned int NumParameters = sizeof...(Params1T);

			template <typename... Params2T>
			struct ForEachParameter;

			template <typename FirstT, typename... RemainderT>
			struct ForEachParameter < FirstT, RemainderT... >
			{
				template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
				__host__ __device__ __inline__ static void replicate(int threadId, int numThreads, QueueT* queue, const Symbol<ShapeT>* symbol, unsigned int* shared)
				{
					Symbol<ShapeT> newSymbol;
					// NOTE: either run() returns true (which means the new symbol was decorated) and the new symbol's entry index is > -1
					// or run() returns false.
					// this mechanism is used by the graph analysis tool
					if (!SymbolDecorator<FirstT>::run(symbol, &newSymbol) || newSymbol.entryIndex > -1)
					{
						newSymbol = *symbol;
						SymbolDecorator<FirstT>::run(symbol, &newSymbol);
						FirstT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, &newSymbol, shared);
					}
					ForEachParameter<RemainderT...>::template replicate<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared);
				}

				__host__ __inline__ static std::string toString()
				{
					return FirstT::toString() + ", " + ForEachParameter<RemainderT...>::toString();
				}

			};

			template <typename LastT>
			struct ForEachParameter < LastT >
			{
				template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
				__host__ __device__ __inline__ static void replicate(int threadId, int numThreads, QueueT* queue, const Symbol<ShapeT>* symbol, unsigned int* shared)
				{
					Symbol<ShapeT> newSymbol;
					SymbolDecorator<LastT>::run(symbol, &newSymbol);
					if (!SymbolDecorator<LastT>::run(symbol, &newSymbol) || newSymbol.entryIndex > -1)
					{
						newSymbol = *symbol;
						SymbolDecorator<LastT>::run(symbol, &newSymbol);
						LastT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, &newSymbol, shared);
					}
				}

				__host__ __inline__ static std::string toString()
				{
					return LastT::toString();
				}

			};

		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				if (T::IsEnabled<DebugFlags::AllOperators, DebugFlags::Translate>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("Executing replicate with %d params\n", NumParameters);
#else
					std::cout << "Executing replicate with " << NumParameters << " params" << std::endl;
#endif
				ForEachParameter<Params1T...>::template replicate<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared);
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "Replicate";
			}

			__host__ __inline__ static std::string toString()
			{
				return "Replicate<" + ForEachParameter<Params1T...>::toString() + ">";
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename... Params2T>
		class Replicate < PGA::Parameters::DynParams, Params2T... >
		{
		private:
			static_assert(sizeof...(Params2T) == 0, "PGA::Parameters::DynParams can only be used as a single parameter");

		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				auto& entry = GlobalVars::getDispatchTableEntry(symbol->entryIndex);
				auto numParameters = entry.numSuccessors;
				if (T::IsEnabled<DebugFlags::AllOperators, DebugFlags::Subdivide>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("Executing replicate with DynParams [symbol->entryIndex=%d, numParameters=%d] (CUDA thread %d %d)\n", symbol->entryIndex, numParameters, threadIdx.x, blockIdx.x);
#else
					std::cout << "Executing replicate with DynParams [entryIndex=" << symbol->entryIndex << ", numParameters=" << numParameters << "]" << std::endl;
#endif
				for (auto i = 0; i < numParameters; i++)
				{
					symbol->entryIndex = entry.successors[i].entryIndex;
					int phaseIndex = entry.successors[i].phaseIndex;
					ContextT::Application::dispatchNonTerminal(*symbol, phaseIndex, 0, queue);
				}
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "Replicate";
			}

			__host__ __inline__ static std::string toString()
			{
				return "Replicate<DynParams>";
			}

		};

	}

}
