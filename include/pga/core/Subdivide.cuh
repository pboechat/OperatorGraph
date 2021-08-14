#pragma once

#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
#include "Parameters.cuh"
#include "Shape.cuh"
#include "Symbol.cuh"
#include "SymbolDecorator.cuh"
#include "TStdLib.h"

#include <cuda_runtime_api.h>
#include <math/vector.h>

#include <cstdio>
#include <string>

namespace PGA
{
	namespace Operators
	{
		template <typename AxisT, typename... Params1T>
		class Subdivide
		{
		private:
			static_assert(!T::EqualTypes<AxisT, PGA::Parameters::DynParams>::Result, "PGA::Parameters::DynParams can only be used as a single parameter");

			static const unsigned int NumParameters = sizeof...(Params1T);

			template <typename... Params2T>
			struct ForEachParameter;

			template <typename FirstT, typename... RemainderT>
			struct ForEachParameter < FirstT, RemainderT... >
			{
			private:
				static_assert(!T::EqualTypes<FirstT, PGA::Parameters::DynParams>::Result, "PGA::Parameters::DynParams can only be used as a single parameter");

				typedef typename FirstT::Key Factor;
				typedef typename FirstT::Value NextOperator;
				static const unsigned int ParameterIndex = NumParameters - sizeof...(RemainderT)-1;
				static const unsigned int RemainderLength = sizeof...(RemainderT);

			public:
				template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
				__host__ __device__ __inline__ static void subdivide(int threadId, int numThreads, QueueT* queue, const Symbol<ShapeT>* symbol, unsigned int* shared, float relativeToAbsoluteScale, float offset, PGA::Axis axis, Symbol<ShapeT>& newSymbol)
				{
					auto factor = Factor::eval(symbol);
					// NOTE: factor is set to 0 on unused dynamic edges (graph analysis tool)
					if (factor != 0)
					{
						auto newSize = (((factor < 0) * relativeToAbsoluteScale * factor * -1.0f) + ((factor > 0) * factor));
						math::float3 size3(symbol->getSize());
						math::float3 offset3(0.0f);
						offset3[axis] = offset + (newSize * 0.5f);
						size3[axis] = newSize;
						newSymbol.setModel(symbol->getModel4() * math::float4x4::translate(offset3));
						newSymbol.setSize(size3);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						newSymbol.setSeed(symbol->generateNextSeed(__uint2float_rd(ParameterIndex)));
#else
						newSymbol.setSeed(symbol->generateNextSeed(static_cast<float>(ParameterIndex)));
#endif
						SymbolDecorator<NextOperator>::run(symbol, &newSymbol);
						NextOperator::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, &newSymbol, shared);
						offset += newSize;
					}
					ForEachParameter<RemainderT...>::template subdivide<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared, relativeToAbsoluteScale, offset, axis, newSymbol);
				}

				template <typename ShapeT>
				__host__ __device__ __inline__ static void getValuesSums(float& absoluteValuesSum, float& relativeValuesSum, const Symbol<ShapeT>* symbol)
				{
					auto factor = Factor::eval(symbol);
					absoluteValuesSum += (factor > 0) * factor;
					relativeValuesSum += (factor < 0) * (factor * -1.0f);
					ForEachParameter<RemainderT...>::getValuesSums(absoluteValuesSum, relativeValuesSum, symbol);
				}

				__host__ __inline__ static std::string toString()
				{
					return FirstT::toString() + ", " + ForEachParameter<RemainderT...>::toString();
				}

			};

			template <typename LastT>
			struct ForEachParameter < LastT >
			{
			private:
				static_assert(!T::EqualTypes<LastT, PGA::Parameters::DynParams>::Result, "PGA::Parameters::DynParams can only be used as a single parameter");

				typedef typename LastT::Key Factor;
				typedef typename LastT::Value NextOperator;
				static const unsigned int ParameterIndex = NumParameters - 1;

			public:
				template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
				__host__ __device__ __inline__ static void subdivide(int threadId, int numThreads, QueueT* queue, const Symbol<ShapeT>* symbol, unsigned int* shared, float relativeToAbsoluteScale, float offset, PGA::Axis axis, Symbol<ShapeT>& newSymbol)
				{
					float factor = Factor::eval(symbol);
					// NOTE: factor is set to 0 on unused dynamic edges (graph analysis tool)
					if (factor != 0)
					{
						auto newSize = (((factor < 0) * relativeToAbsoluteScale * factor * -1.0f) + ((factor > 0) * factor));
						math::float3 size3(symbol->getSize());
						math::float3 offset3(0.0f);
						offset3[axis] = offset + (newSize * 0.5f);
						size3[axis] = newSize;
						newSymbol.setModel(symbol->getModel4() * math::float4x4::translate(offset3));
						newSymbol.setSize(size3);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						newSymbol.setSeed(symbol->generateNextSeed(__uint2float_rd(ParameterIndex)));
#else
						newSymbol.setSeed(symbol->generateNextSeed(static_cast<float>(ParameterIndex)));
#endif
						SymbolDecorator<NextOperator>::run(symbol, &newSymbol);
						NextOperator::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, &newSymbol, shared);
					}
				}

				template <typename ShapeT>
				__host__ __device__ __inline__ static void getValuesSums(float& absoluteValuesSum, float& relativeValuesSum, const Symbol<ShapeT>* symbol)
				{
					auto factor = Factor::eval(symbol);
					absoluteValuesSum += (factor > 0) * factor;
					relativeValuesSum += (factor < 0) * (factor * -1.0f);
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
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				auto axis = static_cast<PGA::Axis>(__float2uint_rd(AxisT::eval(symbol)));
#else
				auto axis = static_cast<PGA::Axis>(static_cast<unsigned int>(AxisT::eval(symbol)));
#endif
				auto totalSize = symbol->getSize()[axis];
				auto offset = totalSize * -0.5f;
				auto absoluteValuesSum = 0.0f, relativeValuesSum = 0.0f, relativeToAbsoluteScale = 0.0f;
				ForEachParameter<Params1T...>::getValuesSums(absoluteValuesSum, relativeValuesSum, symbol);
				if (relativeValuesSum == 0)
					relativeToAbsoluteScale = 0;
				else
					relativeToAbsoluteScale = (totalSize - absoluteValuesSum) / relativeValuesSum;
				Symbol<ShapeT> newSymbol(*symbol);
				ForEachParameter<Params1T...>::template subdivide<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared, relativeToAbsoluteScale, offset, axis, newSymbol);
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "Subdivide";
			}

			__host__ __inline__ static std::string toString()
			{
				return "Subdivide<" + AxisT::toString() + ", " + ForEachParameter<Params1T...>::toString() + ">";
			}

		};

		template <typename... Params2T>
		class Subdivide < PGA::Parameters::DynParams, Params2T... >
		{
		private:
			static_assert(sizeof...(Params2T) == 0, "PGA::Parameters::DynParams can only be used as a single parameter");

		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				auto& entry = GlobalVars::getDispatchTableEntry(symbol->entryIndex);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				auto axis = static_cast<PGA::Axis>(__float2uint_rd(PGA::Parameters::dynEval(symbol, entry.parameters[0])));
#else
				auto axis = static_cast<PGA::Axis>(static_cast<unsigned int>(PGA::Parameters::dynEval(symbol, entry.parameters[0])));
#endif
				auto numParameters = entry.numParameters;
				if (T::IsEnabled<DebugFlags::AllOperators, DebugFlags::Subdivide>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("Executing Subdivide with DynParams [symbol->entryIndex=%d, axis=%d, numParameters=%d] (CUDA thread %d %d)\n", symbol->entryIndex, axis, numParameters, threadIdx.x, blockIdx.x);
#else
					std::cout << "Executing Subdivide with DynParams [entryIndex=" << symbol->entryIndex << ", axis=" << axis << ", numParameters=" << numParameters << "]" << std::endl;
#endif
				auto totalSize = symbol->getSize()[axis];
				auto offset = totalSize * -0.5f;
				auto absoluteValuesSum = 0.0f, relativeValuesSum = 0.0f, relativeToAbsoluteScale = 0.0f;
				for (auto i = 1u; i < numParameters; i++)
				{
					auto factor = PGA::Parameters::dynEval(symbol, entry.parameters[i]);
					absoluteValuesSum += (factor > 0) * factor;
					relativeValuesSum += (factor < 0) * (factor * -1.0f);
				}
				if (relativeValuesSum == 0)
					relativeToAbsoluteScale = 0;
				else
					relativeToAbsoluteScale = (totalSize - absoluteValuesSum) / relativeValuesSum;
				// NOTE: relativeToAbsoluteScale is less or eq. than 0 when 
				// the sum of absolute values is bigger or eq. the total size
				if (relativeToAbsoluteScale < 0.0f)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("PGA::Operators::Subdivide::execute(): the sum of absolute factors is bigger than the shape size being subdivided (axis=%d, numSuccessors=%d, totalSize=%f, absoluteValuesSum=%f) [CUDA thread %d %d]\n", axis, numParameters - 1,totalSize, absoluteValuesSum, threadIdx.x, blockIdx.x);
					asm("trap;");
#else
					throw std::runtime_error("PGA::Operators::Subdivide::execute(): the sum of absolute factors is bigger than the shape size being subdivided (axis=" + std::to_string(axis) + ", numSuccessors=" + std::to_string(numParameters) + ", totalSize=" + std::to_string(totalSize) + ", absoluteValuesSum=" + std::to_string(absoluteValuesSum) + ")");
#endif
				}
				auto baseSeed = symbol->getSeed();
				auto baseSize = symbol->getSize();
				auto baseModel = symbol->getModel4();
				for (auto i = 1u, j = 0u; i < numParameters; j = i, i++)
				{
					auto factor = PGA::Parameters::dynEval(symbol, entry.parameters[i]);
					auto newSize = (((factor < 0) * relativeToAbsoluteScale * factor * -1.0f) + ((factor > 0) * factor));
					math::float3 size3(baseSize);
					math::float3 offset3(0.0f);
					offset3[axis] = offset + (newSize * 0.5f);
					size3[axis] = newSize;
					symbol->setSize(size3);
					symbol->setModel(baseModel * math::float4x4::translate(offset3));
					symbol->setSeed(Random::nextSeed(baseSeed, j));
					symbol->entryIndex = entry.successors[j].entryIndex;
					auto phaseIndex = entry.successors[j].phaseIndex;
					ContextT::Application::dispatchNonTerminal(*symbol, phaseIndex, 0, queue);
					offset += newSize;
				}
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "Subdivide";
			}

			__host__ __inline__ static std::string toString()
			{
				return "Subdivide<DynParams>";
			}

		};

	}

}
