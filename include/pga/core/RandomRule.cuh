#pragma once

#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
#include "Parameters.cuh"
#include "Shape.cuh"
#include "Symbol.cuh"
#include "SymbolDecorator.cuh"

#include <cuda_runtime_api.h>
#include <math/math.h>

#include <string>

namespace PGA
{
	namespace Operators
	{
		template <typename ChanceT, typename... Params1T>
		class RandomRule
		{
		private:
			static_assert(!T::EqualTypes<ChanceT, PGA::Parameters::DynParams>::Result, "PGA::Parameters::DynParams can only be used as a single parameter");

			template <typename... Params2T>
			struct ForEachParameter;

			template <typename FirstT, typename... RemainderT>
			struct ForEachParameter < FirstT, RemainderT... >
			{
			private:
				static_assert(!T::EqualTypes<FirstT, PGA::Parameters::DynParams>::Result, "PGA::Parameters::DynParams can only be used as a single parameter");

				typedef typename FirstT::Key Probability;
				typedef typename FirstT::Value NextOperator;

			public:
				template <typename ShapeT>
				__host__ __device__ __inline__ static float getProbabilitiesSum(Symbol<ShapeT>* symbol)
				{
					return Probability::eval(symbol) + ForEachParameter<RemainderT...>::getProbabilitiesSum(symbol);
				}

				template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
				__host__ __device__ __inline__ static void doProbabilisticSelection(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared, float chance, float probabilitiesSum, float accProbability)
				{
					float normalizedProbabilty = Probability::eval(symbol) / probabilitiesSum;
					accProbability += normalizedProbabilty;
					// NOTE: probability is set to 0 on unused dynamic edges (graph analysis tool)
					if (normalizedProbabilty != 0 && chance <= accProbability)
					{
						SymbolDecorator<NextOperator>::run(symbol, symbol);
						NextOperator::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared);
					}
					else
					{
						ForEachParameter<RemainderT...>::template doProbabilisticSelection<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared, chance, probabilitiesSum, accProbability);
					}
				}

				__host__ __inline__ static std::string toString()
				{
					return NextOperator::toString() + ", " + ForEachParameter<RemainderT...>::toString();
				}

			};

			template <typename LastT>
			struct ForEachParameter < LastT >
			{
			private:
				static_assert(!T::EqualTypes<LastT, PGA::Parameters::DynParams>::Result, "PGA::Parameters::DynParams can only be used as a single parameter");

				typedef typename LastT::Key Probability;
				typedef typename LastT::Value NextOperator;

			public:
				template <typename ShapeT>
				__host__ __device__ __inline__ static float getProbabilitiesSum(Symbol<ShapeT>* symbol)
				{
					return Probability::eval(symbol);
				}

				template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
				__host__ __device__ __inline__ static void doProbabilisticSelection(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared, float chance, float probabilitiesSum, float accProbability)
				{
					SymbolDecorator<NextOperator>::run(symbol, symbol);
					NextOperator::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared);
				}

				__host__ __inline__ static std::string toString()
				{
					return NextOperator::toString();
				}

			};

		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				float chance = math::frac(ChanceT::eval(symbol));
				float probabilitiesSum = ForEachParameter<Params1T...>::getProbabilitiesSum(symbol);
				ForEachParameter<Params1T...>::template doProbabilisticSelection<ContextT, NumThreadsT>(threadId, numThreads, queue, symbol, shared, chance, probabilitiesSum, 0);
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "RandomRule";
			}

			__host__ __inline__ static std::string toString()
			{
				return "RandomRule<" + ForEachParameter<Params1T...>::toString() + ">";
			}

		};

		template <typename... Params2T>
		class RandomRule < PGA::Parameters::DynParams, Params2T... >
		{
		private:
			static_assert(sizeof...(Params2T) == 0, "PGA::Parameters::DynParams can only be used as a single parameter");

		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				auto& entry = GlobalVars::getDispatchTableEntry(symbol->entryIndex);
				float chance = math::frac(PGA::Parameters::dynEval(symbol, entry.parameters[0]));
				float probabilitiesSum = 0;
				auto numParameters = entry.numParameters;
				for (auto i = 1; i < numParameters; i++)
					probabilitiesSum += PGA::Parameters::dynEval(symbol, entry.parameters[i]);
				float accProbability = 0.0f;
				for (auto i = 1; i < numParameters; i++)
				{
					accProbability += PGA::Parameters::dynEval(symbol, entry.parameters[i]) / probabilitiesSum;
					if (chance <= accProbability)
					{
						symbol->entryIndex = entry.successors[i - 1].entryIndex;
						int phaseIndex = entry.successors[i - 1].phaseIndex;
						ContextT::Application::dispatchNonTerminal(*symbol, phaseIndex, 0, queue);
						break;
					}
				}
			}

		};

	}

}
