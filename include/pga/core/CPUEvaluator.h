#pragma once

#include "CPUBVHConstructor.h"
#include "CPUBaseContext.h"
#include "CUDAException.h"
#include "DebugFlags.h"
#include "Discard.cuh"
#include "DispatchTableEntry.h"
#include "GlobalVariables.cuh"
#include "HighResClock.h"
#include "Instrumentation.cuh"
#include "ShapeGenerator.cuh"
#include "Statistics.h"
#include "Symbol.cuh"
#include "TStdLib.h"

#include <cuda_runtime_api.h>

#include <deque>
#include <map>
#include <memory>
#include <vector>

namespace PGA
{
	namespace CPU
	{
		template <typename QueuesManagerT>
		class BaseQueue
		{
		public:
			virtual bool dequeue(QueuesManagerT* queuesManager) = 0;
			virtual void enqueue(void* data, size_t size) = 0;
			virtual void enqueue(void* data) = 0;

		};

		template <typename ProcedureT, typename QueuesManagerT, typename ContextT>
		class Queue : public BaseQueue<QueuesManagerT>
		{
		private:
			std::deque<typename ProcedureT::ExpectedData> items;

		public:
			virtual bool dequeue(QueuesManagerT* queuesManager)
			{
				if (items.empty())
					return false;
				while (!items.empty())
				{
					auto& item = items.front();
					ContextT::setUpWarp(ProcedureT::NumThreads);
					for (auto threadIdx = 0; threadIdx < ProcedureT::NumThreads; threadIdx++)
					{
						ContextT::setCurrentThreadIdx(threadIdx);
						ProcedureT::template execute<QueuesManagerT, ContextT>(threadIdx, ProcedureT::NumThreads, queuesManager, &item, 0);
					}
					items.pop_front();
				}
				return true;
			}

			virtual void enqueue(void* data, size_t size)
			{
				if (size != sizeof(typename ProcedureT::ExpectedData))
					throw std::runtime_error("enqueueing shape of different type then expected by the procedure");
				enqueue(data);
			}

			virtual void enqueue(void* data)
			{
				auto item = *reinterpret_cast<typename ProcedureT::ExpectedData*>(data);
				items.push_back(item);
			}

		};

		template <typename ProcedureT, unsigned int PhasesT>
		struct MultiPhaseQueueBuilder
		{
		private:
			static const unsigned int PhaseIdx = PhasesT - 1;

		public:
			template <template <class, int> class PhaseTraitsT, typename ContextT, typename QueuesManagerT>
			static void initialize(QueuesManagerT& queuesManager)
			{
				if (PhaseTraitsT<ProcedureT, PhaseIdx>::Enabled)
				{
					std::unique_ptr<Queue<ProcedureT, QueuesManagerT, ContextT>> newQueue(new Queue<ProcedureT, QueuesManagerT, ContextT>());
					queuesManager.initializeQueue(PhaseIdx, std::move(newQueue));
				}
				MultiPhaseQueueBuilder<ProcedureT, PhasesT - 1>::template initialize<PhaseTraitsT, ContextT>(queuesManager);
			}

		};

		template <typename ProcedureT>
		struct MultiPhaseQueueBuilder < ProcedureT, 0 >
		{
			template <template <class, int> class PhaseTraitsT, typename ContextT, typename QueuesManagerT>
			static void initialize(QueuesManagerT& queuesManager)
			{
			}

		};

		template <typename ProcedureListT, unsigned int CountT>
		struct QueuesBuilder
		{
			template <unsigned int NumPhasesT, template <class, int> class PhaseTraitsT, typename ContextT, typename QueuesManagerT>
			static void initializeMultiPhase(QueuesManagerT& queuesManager)
			{
				typedef typename ProcedureListT::template ItemAt<ProcedureListT::Length - CountT>::Result Procedure;
				MultiPhaseQueueBuilder<Procedure, NumPhasesT>::template initialize<PhaseTraitsT, ContextT>(queuesManager);
				QueuesBuilder<ProcedureListT, CountT - 1>::template initializeMultiPhase<NumPhasesT, PhaseTraitsT, ContextT>(queuesManager);
			}

			template <typename ContextT, typename QueuesManagerT>
			static void initializeSinglePhase(QueuesManagerT& queuesManager)
			{
				typedef typename ProcedureListT::template ItemAt<ProcedureListT::Length - CountT>::Result Procedure;
				std::unique_ptr<Queue<Procedure, QueuesManagerT, ContextT>> newQueue(new Queue<Procedure, QueuesManagerT, ContextT>());
				queuesManager.initializeQueue(0, std::move(newQueue));
				QueuesBuilder<ProcedureListT, CountT - 1>::template initializeSinglePhase<ContextT>(queuesManager);
			}

		};

		template <typename ProcedureListT>
		struct QueuesBuilder < ProcedureListT, 0 >
		{
		public:
			template <unsigned int NumPhasesT, template <class, int> class PhaseTraitsT, typename ContextT, typename QueuesManagerT>
			static void initializeMultiPhase(QueuesManagerT& queuesManager)
			{
			}

			template <typename ContextT, typename QueuesManagerT>
			static void initializeSinglePhase(QueuesManagerT& queuesManager)
			{
			}

		};

		template <typename ProcedureListT>
		class RoundRobinQueuesManager
		{
		private:
			typedef RoundRobinQueuesManager<ProcedureListT> Self;
			std::vector<std::vector<std::unique_ptr<BaseQueue<Self>>>> queues;

		public:
			template <typename ProcedureT, typename ShapeT>
			__host__ __device__ void enqueue(Symbol<ShapeT>& symbol, unsigned int phaseIdx)
			{
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 0))
				auto queueIdx = ProcedureListT::template IndexOf<ProcedureT>::Result;
				static_assert(T::EqualTypes<typename ProcedureT::ExpectedData, Symbol<ShapeT>>::Result, "enqueueing shape of different type then expected by the procedure");
				// FIXME: checking invariants
				if (queues.size() <= phaseIdx)
					throw std::runtime_error("queues.size() <= phaseIdx");
				// FIXME: checking invariants
				if (queues[phaseIdx].size() <= queueIdx)
					throw std::runtime_error("queues[phaseIdx].size() <= queueIdx");
				queues[phaseIdx][queueIdx]->enqueue((void*)&symbol);
#endif
			}

			template <typename ShapeT>
			__host__ __device__ void enqueue(int queueIdx, Symbol<ShapeT>& symbol, unsigned int phaseIdx)
			{
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 0))
				// FIXME: checking invariants
				if (queues.size() <= phaseIdx)
					throw std::runtime_error("queues.size() <= phaseIdx");
				// FIXME: checking invariants
				if (queues[phaseIdx].size() <= queueIdx)
					throw std::runtime_error("queues[phaseIdx].size() <= queueIdx");
				// NOTE: do size checking
				queues[phaseIdx][queueIdx]->enqueue((void*)&symbol, sizeof(Symbol<ShapeT>));
#endif
			}

			bool dequeueAll()
			{
				bool proceed = false;
				for (auto& phaseQueues : queues)
					for (auto& queue : phaseQueues)
						proceed |= queue->dequeue(this);
				return proceed;
			}

			bool dequeueAll(unsigned int phaseIdx)
			{
				bool proceed = false;
				for (auto& queue : queues[phaseIdx])
					proceed |= queue->dequeue(this);
				return proceed;
			}

			template <typename QueueT>
			void initializeQueue(unsigned int phaseIdx, std::unique_ptr<QueueT>&& queue)
			{
				if (queues.size() <= phaseIdx)
					queues.resize(phaseIdx + 1);
				queues[phaseIdx].emplace_back(std::move(queue));
			}

		};

		template <typename ProcedureListT, typename GenFuncFilterT, bool UseInstrumentationT>
		struct BaseSymbolManager
		{
			typedef ProcedureListT ProcedureList;
			static const bool UseInstrumentation = UseInstrumentationT;

			template <unsigned int OperatorCodeT, typename ShapeT, typename QueueManagerT>
			__host__ __device__ __inline__ static void dispatchNonTerminal(Symbol<ShapeT>& symbol, unsigned int phaseIdx, unsigned char ruleTagId, QueueManagerT* queueManager)
			{
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 0))
				if (T::IsEnabled<DebugFlags::SymbolDispatch>::Result)
					std::cout << "Non-terminal symbol dispatched (OperatorCodeT=" << OperatorCodeT << ", symbol.entryIndex=" << symbol.entryIndex << ", shapeType=" << Shapes::GetName<ShapeT>::Result() << ", phaseIdx=" << phaseIdx << ", ruleTagId=" << static_cast<unsigned int>(ruleTagId) << ")" << std::endl;
				typedef typename ProcedureListT::template ItemAt<OperatorCodeT>::Result Procedure;
				if (!T::EqualTypes<Operators::Discard, typename Procedure::FirstOperator>::Result)
					queueManager->template enqueue<Procedure>(symbol, phaseIdx);
#endif
			}

			template <typename ShapeT, typename QueueManagerT>
			__host__ __device__ __inline__ static void dispatchNonTerminal(Symbol<ShapeT>& symbol, unsigned int phaseIdx, unsigned char ruleTagId, QueueManagerT* queueManager)
			{
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 0))
				const DispatchTableEntry& dispatchTableEntry = Host::DispatchTable[symbol.entryIndex];
				if (T::IsEnabled<DebugFlags::SymbolDispatch>::Result)
					std::cout << "Non-terminal symbol dispatched (symbol.entryIndex=" << symbol.entryIndex << ", shapeType=" << Shapes::GetName<ShapeT>::Result() << ", operatorCode=" << dispatchTableEntry.operatorCode << ", phaseIdx=" << phaseIdx << ", ruleTagId=" << static_cast<unsigned int>(ruleTagId) << ")" << std::endl;
				queueManager->enqueue(dispatchTableEntry.operatorCode, symbol, phaseIdx);
#endif
			}

			template <unsigned int OperatorCodeT, typename ShapeT, typename QueueManagerT>
			__host__ __device__ __inline__ static void dispatchAxiom(ShapeT& shape, QueueManagerT* queueManager, int entryIndex = -1)
			{
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 0))
				Symbol<ShapeT> symbol(shape);
				symbol.entryIndex = entryIndex;
				if (T::IsEnabled<DebugFlags::SymbolDispatch>::Result)
					std::cout << "Axiom dispatched [compile-time resolution] (OperatorCodeT=" << OperatorCodeT << ", shapeType=" << Shapes::GetName<ShapeT>::Result() << ", entryIndex=" << entryIndex << ")" << std::endl;
				typedef typename ProcedureListT::template ItemAt<OperatorCodeT>::Result Procedure;
				if (!T::EqualTypes<Operators::Discard, typename Procedure::FirstOperator>::Result)
					queueManager->template enqueue<Procedure>(symbol, 0);
#endif
			}

			template <typename ShapeT, typename QueueManagerT>
			__host__ __device__ __inline__ static void dispatchAxiom(ShapeT& shape, int entryIndex, QueueManagerT* queueManager)
			{
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 0))
				Symbol<ShapeT> symbol(shape);
				symbol.entryIndex = entryIndex;
				const DispatchTableEntry& dispatchTableEntry = Host::DispatchTable[entryIndex];
				if (T::IsEnabled<DebugFlags::SymbolDispatch>::Result)
					std::cout << "Axiom dispatched [run-time resolution] (entryIndex=" << entryIndex << ", shapeType=" << Shapes::GetName<ShapeT>::Result() << ", operatorCode=" << dispatchTableEntry.operatorCode << ")" << std::endl;
				queueManager->enqueue(dispatchTableEntry.operatorCode, symbol, 0);
#endif
			}

			template <unsigned int GenFuncIdxT, typename ShapeT, typename... ArgsT>
			__host__ __device__ __inline__ static void dispatchTerminal(Symbol<ShapeT>& symbol, ArgsT... args)
			{
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 0))
				if (T::IsEnabled<DebugFlags::SymbolDispatch>::Result)
					std::cout << "Terminal symbol dispatched [serial] (entryIndex=" << symbol.entryIndex << ", shapeType=" << Shapes::GetName<ShapeT>::Result() << ", GenFuncIdxT=" << GenFuncIdxT << ")" << std::endl;
				ShapeGenerator<ShapeT, false>::template run<GenerationFunction<GenFuncFilterT, GenFuncIdxT>>(symbol, args...);
#endif
			}

			template <unsigned int GenFuncIdxT, typename ContextT, unsigned int NumThreadsT, typename ShapeT, typename... ArgsT>
			__host__ __device__ __inline__ static void dispatchTerminalInParallel(int localThreadId, Symbol<ShapeT>& symbol, ArgsT... args)
			{
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ == 0))
				static_assert(NumThreadsT >= ShapeGenerator<ShapeT, true>::NumThreads, "using less threads than the required by the shape generation function");
				if (T::IsEnabled<DebugFlags::SymbolDispatch>::Result)
					std::cout << "Terminal symbol dispatched [parallel] (entryIndex=" << symbol.entryIndex << ", shapeType=" << Shapes::GetName<ShapeT>::Result() << ", GenFuncIdxT=" << GenFuncIdxT << ", NumThreadsT=" << NumThreadsT << ")" << std::endl;
				ShapeGenerator<ShapeT, true>::template run<GenerationFunction<GenFuncFilterT, GenFuncIdxT>, ContextT>(localThreadId, symbol, args...);
#endif
			}

		};

		struct DefaultConfiguration
		{
			static const unsigned int MaxDerivationSteps = 100000;

		};

		template
		<
			typename ProcedureListT,
			typename AxiomGeneratorT,
			typename GenFuncFilterT,
			unsigned int NumPhasesT,
			template <class, int> class PhaseTraitsT,
			bool UseInstrumentationT = false,
			unsigned int NumSubgraphsT = 0,
			unsigned int NumEdgesT = 0,
			typename ConfigurationT = DefaultConfiguration
		>
		class MultiPhaseEvaluator
		{
		public:
			typedef BaseSymbolManager<ProcedureListT, GenFuncFilterT, UseInstrumentationT> SymbolManager;

			struct Context : public BaseContext
			{
				typedef SymbolManager Application;

			};

		private:
			RoundRobinQueuesManager<ProcedureListT> queuesManager;

		public:
			double execute()
			{
				return execute(AxiomGeneratorT::getNumAxioms());
			}

			double execute(unsigned int numAxioms)
			{
				for (auto id = 0; id < numAxioms; id++)
					AxiomGeneratorT::template generateAxiom<SymbolManager>(id, &queuesManager);
				//ContextSensitivity::IntermediateSymbols::reset();
				unsigned int derivationStep[NumPhasesT];
				double executionTimes[NumPhasesT];
				double totalExecutionTime = 0;
				for (unsigned int i = 0, j = NumPhasesT - 1; i < NumPhasesT; i++, j--)
				{
					//if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					//	PGA_CUDA_checkedCall(cudaMemcpyToSymbol(ContextSensitivity::Device::NumCollisionChecks, &numCollisionChecks, sizeof(unsigned int)));
					derivationStep[i] = 0;
					auto start = HighResClock::now();
					while (queuesManager.dequeueAll(i) && ++derivationStep[i] < ConfigurationT::MaxDerivationSteps);
					auto end = HighResClock::now();
					double executionTime = (end - start).count() / (double)PGA::HighResClock::period::den;
					//if (j > 0)
					//	executionTime += ContextSensitivity::CPU::BVHConstructor::construct(ContextSensitivity::Constants::SceneBoundaryMin, ContextSensitivity::Constants::SceneBoundaryMax);
					totalExecutionTime += executionTime;
					executionTimes[i] = executionTime;
				}
				if (T::IsEnabled<Statistics::Execution>::Result)
				{
					std::cout << "************************************************************" << std::endl;
					std::cout << "CPU Multi-Phased Execution Statistics:" << std::endl;
					std::cout << "************************************************************" << std::endl;
					std::cout << "Num. Axioms: " << numAxioms << std::endl;
					std::cout << "Elapsed Time (Total): " << totalExecutionTime << "s" << std::endl;
					for (auto i = 0; i < NumPhasesT; i++)
					{
						std::cout << "Phase " << i << std::endl;
						std::cout << "Elapsed Time: " << executionTimes[i] << "s" << std::endl;
						std::cout << "Num. Derivation Steps: " << derivationStep[i] << std::endl;
						//	if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
						//	{
						//		PGA_CUDA_checkedCall(cudaMemcpyFromSymbol(&numCollisionChecks, ContextSensitivity::Device::NumCollisionChecks, sizeof(unsigned int)));
						//		std::cout << "[Phase " << i << "] Num. Collision Checks: " << numCollisionChecks << std::endl;
						//	}
					}
				}
				return totalExecutionTime;
			}

			void initialize(DispatchTableEntry* dispatchTable, size_t numEntries)
			{
				Host::NumEntries = numEntries;
				if (dispatchTable != nullptr)
				{
					size_t size = numEntries * sizeof(DispatchTableEntry);
					Host::DispatchTable = (DispatchTableEntry*)malloc(size);
					memcpy(Host::DispatchTable, dispatchTable, size);
				}
				QueuesBuilder<ProcedureListT, ProcedureListT::Length>::template initializeMultiPhase<NumPhasesT, PhaseTraitsT, Context>(queuesManager);
			}

			void release()
			{
				if (Host::DispatchTable)
					free(Host::DispatchTable);
			}

		};

		template
		<
			typename ProcedureListT,
			typename AxiomGeneratorT,
			typename GenFuncFilterT,
			bool UseInstrumentationT = false,
			unsigned int NumSubgraphsT = 0,
			unsigned int NumEdgesT = 0,
			typename ConfigurationT = DefaultConfiguration
		>
		class SinglePhaseEvaluator
		{
		public:
			typedef BaseSymbolManager<ProcedureListT, GenFuncFilterT, UseInstrumentationT> SymbolManager;

			struct Context : public BaseContext
			{
				typedef SymbolManager Application;

			};

		private:
			RoundRobinQueuesManager<ProcedureListT> queuesManager;

		public:
			double execute()
			{
				return execute(AxiomGeneratorT::getNumAxioms());
			}

			double execute(unsigned int numAxioms)
			{
				for (auto id = 0; id < numAxioms; id++)
					AxiomGeneratorT::template generateAxiom<SymbolManager>(id, &queuesManager);
				auto derivationStep = 0;
				auto start = HighResClock::now();
				while (queuesManager.dequeueAll() && ++derivationStep < ConfigurationT::MaxDerivationSteps);
				auto end = HighResClock::now();
				double totalExecutionTime = (end - start).count() / (double)PGA::HighResClock::period::den;
				if (T::IsEnabled<Statistics::Execution>::Result)
				{
					std::cout << "************************************************************" << std::endl;
					std::cout << "CPU Single-Phase Execution Statistics:" << std::endl;
					std::cout << "************************************************************" << std::endl;
					std::cout << "Num. Axioms: " << numAxioms << std::endl;
					std::cout << "Elapsed Time: " << totalExecutionTime << "s" << std::endl;
					std::cout << "Num. Derivation Steps: " << derivationStep << std::endl;
				}
				return totalExecutionTime;
			}

			void initialize(DispatchTableEntry* dispatchTable, size_t numEntries)
			{
				Host::NumEntries = numEntries;
				size_t size = numEntries * sizeof(DispatchTableEntry);
				Host::DispatchTable = (DispatchTableEntry*)malloc(size);
				memcpy(Host::DispatchTable, dispatchTable, size);
				QueuesBuilder<ProcedureListT, ProcedureListT::Length>::template initializeSinglePhase<Context>(queuesManager);
			}

			void release()
			{
				if (Host::DispatchTable)
					free(Host::DispatchTable);
			}

		};

	}

}
