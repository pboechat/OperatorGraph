#pragma once

#include "Call.cuh"
#include "GlobalVariables.cuh"
#include "Symbol.cuh"

#include <cuda_runtime_api.h>

#include <stdexcept>

namespace PGA
{
	namespace Operators
	{
		template <typename T>
		class SymbolDecorator
		{
		public:
			template <typename Shape1, typename Shape2>
			__host__ __device__ __inline__ static bool run(const Symbol<Shape1>* symbol, Symbol<Shape2>* newSymbol)
			{
				newSymbol->entryIndex = symbol->entryIndex;
				newSymbol->predecessor = symbol->predecessor;
				return true;
			}

		};

		template <unsigned int OperationCode, unsigned int SuccessorOffset, int EdgeIndex>
		class SymbolDecorator < PSCall<OperationCode, SuccessorOffset, EdgeIndex> >
		{
		public:
			template <typename T, typename U>
			__host__ __device__ __inline__ static bool run(const Symbol<T>* symbol, Symbol<U>* newSymbol)
			{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 2)
				if (symbol->entryIndex < 0 || symbol->entryIndex >= GlobalVars::getNumEntries())
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("PGA::Operators::SymbolDecorator<PSCall<%d, %d, %d>>::run(..): invalid entry index [symbol->entryIndex=%d, GlobalVars::getNumEntries()=%d] (CUDA thread %d %d)\n", OperationCode, SuccessorOffset, EdgeIndex, symbol->entryIndex, GlobalVars::getNumEntries(), threadIdx.x, blockIdx.x);
					asm("trap;");
#else
					throw std::runtime_error(("PGA::Operators::SymbolDecorator<PSCall<" + std::to_string(OperationCode) + ", " + std::to_string(SuccessorOffset) + ", " + std::to_string(EdgeIndex) + ">>::run(..): invalid entry index [symbol->entryIndex=" + std::to_string(symbol->entryIndex) + ", GlobalVars::getNumEntries()=" + std::to_string(GlobalVars::getNumEntries()) + "]").c_str());
#endif
				}
#endif
				auto& entry = GlobalVars::getDispatchTableEntry(symbol->entryIndex);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 2)
				if (SuccessorOffset >= entry.numSuccessors)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("PGA::Operators::SymbolDecorator<PSCall<%d, %d, %d>>::run(..): out of boundaries successor access [symbol->entryIndex=%d, entry.numSuccessors=%d] (CUDA thread %d %d)\n", OperationCode, SuccessorOffset, EdgeIndex, symbol->entryIndex, entry.numSuccessors, threadIdx.x, blockIdx.x);
					asm("trap;");
#else
					throw std::runtime_error(("PGA::Operators::SymbolDecorator<PSCall<" + std::to_string(OperationCode) + ", " + std::to_string(SuccessorOffset) + ", " + std::to_string(EdgeIndex) + ">>::run(..): out of boundaries successor access [symbol->entryIndex=" + std::to_string(symbol->entryIndex) + ", entry.numSuccessors=" + std::to_string(entry.numSuccessors) + "]").c_str());
#endif
				}
#endif
				newSymbol->predecessor = symbol->entryIndex;
				newSymbol->entryIndex = entry.successors[SuccessorOffset].entryIndex;
				return true;
			}

		};

		template <unsigned int OperationCode, int EdgeIndex, int EntryIndex>
		class SymbolDecorator < FSCall<OperationCode, EdgeIndex, EntryIndex> >
		{
		public:
			template <typename T, typename U>
			__host__ __device__ __inline__ static bool run(const Symbol<T>* symbol, Symbol<U>* newSymbol)
			{
				bool decorateSymbol = EntryIndex != -1;
				if (decorateSymbol)
				{
					newSymbol->predecessor = symbol->entryIndex;
					newSymbol->entryIndex = EntryIndex;
				}
				return decorateSymbol;
			}

		};

		template <unsigned int SuccessorOffset, int EdgeIndex>
		class SymbolDecorator < DCall<SuccessorOffset, EdgeIndex> >
		{
		public:
			template <typename T, typename U>
			__host__ __device__ __inline__ static bool run(const Symbol<T>* symbol, Symbol<U>* newSymbol)
			{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 2)
				if (symbol->entryIndex < 0 || symbol->entryIndex >= GlobalVars::getNumEntries())
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("PGA::Operators::SymbolDecorator<DCall<%d, %d>>::run(..): invalid entry index [symbol->entryIndex=%d, GlobalVars::getNumEntries()=%d] (CUDA thread %d %d)\n", SuccessorOffset, EdgeIndex, symbol->entryIndex, GlobalVars::getNumEntries(), threadIdx.x, blockIdx.x);
					asm("trap;");
#else
					throw std::runtime_error(("PGA::Operators::SymbolDecorator<DCall<" + std::to_string(SuccessorOffset) + ", " + std::to_string(EdgeIndex) + ">>::run(..): invalid entry index [symbol->entryIndex=" + std::to_string(symbol->entryIndex) + ", GlobalVars::getNumEntries()=" + std::to_string(GlobalVars::getNumEntries()) + "]").c_str());
#endif
				}
#endif
				auto& entry = GlobalVars::getDispatchTableEntry(symbol->entryIndex);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 2)
				if (SuccessorOffset >= entry.numSuccessors)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("PGA::Operators::SymbolDecorator<DCall<%d, %d>>::run(..): out of boundaries successor access [symbol->entryIndex=%d, entry.numSuccessors=%d] (CUDA thread %d %d)\n", SuccessorOffset, EdgeIndex, symbol->entryIndex, entry.numSuccessors, threadIdx.x, blockIdx.x);
					asm("trap;");
#else
					throw std::runtime_error(("PGA::Operators::SymbolDecorator<DCall<" + std::to_string(SuccessorOffset) + ", " + std::to_string(EdgeIndex) + ">>::run(..): out of boundaries successor access [symbol->entryIndex=" + std::to_string(symbol->entryIndex) + ", entry.numSuccessors=" + std::to_string(entry.numSuccessors) + "]").c_str());
#endif
				}
#endif
				newSymbol->predecessor = symbol->entryIndex;
				newSymbol->entryIndex = entry.successors[SuccessorOffset].entryIndex;
				return true;
			}

		};

	}

}
