#pragma once

#include <memory>
#include <cuda_runtime_api.h>

#include <techniqueKernels.cuh>
#include <techniqueDynamicParallelism.cuh>
#include <techniqueMegakernel.cuh>
#include <queueDistLocks.cuh>
#include <queueShared.cuh>
#include <queuingPerProc.cuh>

#include "DebugFlags.h"
#include "CUDAException.h"
#include "Statistics.h"
#include "DispatchTableEntry.h"
#include "GlobalVariables.cuh"
#include "Symbol.cuh"
#include "GPUBVHConstructor.cuh"
#include "IntermediateSymbolsBufferAdapter.cuh"
#include "ShapeGenerator.cuh"
#include "Discard.cuh"
#include "TStdLib.h"
#include "GPUTechnique.h"
#include "Instrumentation.cuh"

#define SAME_SHAPES() \
	T::EqualTypes<ShapeT, typename GetSymbolShape<typename ProcedureListT::ItemAt<LengthT - 1>::Result::ExpectedData>::Result>::Result

namespace PGA
{
	namespace GPU
	{
		////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename Procedure>
		struct SharedQueueTraits
		{
			static const int QueueSize = 0;

		};

		////////////////////////////////////////////////////////////////////////////////////////////////////
		template <template <class> class QueueT, typename ProcInfoT, typename CustomTypeT, Technique Technique>
		struct GetTechniqueType;

		template <template <class> class QueueT, typename ProcInfoT, typename CustomTypeT>
		struct GetTechniqueType<QueueT, ProcInfoT, CustomTypeT, Technique::KERNELS>
		{
			typedef KernelLaunches::TechniqueStreams<QueueT, ProcInfoT, CustomTypeT> Type;

		};

		template <template <class> class QueueT, typename ProcInfoT, typename CustomTypeT>
		struct GetTechniqueType<QueueT, ProcInfoT, CustomTypeT, Technique::DYN_PAR>
		{
			typedef DynamicParallelism::TechniqueQueuedNoCopy<QueueT, ProcInfoT, CustomTypeT> Type;

		};

		template <template <class> class QueueT, typename ProcInfoT, typename CustomTypeT>
		struct GetTechniqueType<QueueT, ProcInfoT, CustomTypeT, Technique::MEGAKERNEL>
		{
			typedef Megakernel::SimplePointed16336<QueueT, ProcInfoT, CustomTypeT> Type;

		};

		template <template <class> class QueueT, typename ProcInfoT, typename CustomTypeT>
		struct GetTechniqueType<QueueT, ProcInfoT, CustomTypeT, Technique::MEGAKERNEL_LOCAL_QUEUES>
		{
			typedef Megakernel::SimplePointed16336<QueueT, ProcInfoT, CustomTypeT> Type;

		};

		////////////////////////////////////////////////////////////////////////////////////////////////////
		template <Technique Technique, typename ProcInfoT, unsigned int QueueSizeT, unsigned int MaxSharedMemoryT>
		struct QueueSelector
		{
			typedef typename PerProcedureQueueTyping<QueueDistLocksOpt_t, QueueSizeT, false>::template Type<ProcInfoT> Type;

		};

		template <typename ProcInfoT1, unsigned int QueueSizeT, unsigned int MaxSharedMemoryT>
		struct QueueSelector<Technique::MEGAKERNEL_LOCAL_QUEUES, ProcInfoT1, QueueSizeT, MaxSharedMemoryT>
		{
			template <typename ProcInfoT2>
			struct GlobalQueue : PerProcedureQueueTyping<QueueDistLocksOpt_t, QueueSizeT, false>::template Type<ProcInfoT2> {};

			template <typename ProcInfoT2>
			struct SharedQueue : SharedStaticQueue<ProcInfoT2, MaxSharedMemoryT, SharedQueueTraits, true> {};

			typedef SharedCombinerQueue<ProcInfoT1, GlobalQueue, SharedQueue> Type;

		};

		////////////////////////////////////////////////////////////////////////////////////////////////////
		template <Technique TechniqueT, typename ProcInfoT1, typename CustomTypeT, unsigned int QueueSizeT, unsigned int MaxSharedMemoryT, bool UseInstrumentationT /* false */, unsigned int NumSubgraphsT, unsigned int NumEdgesT>
		struct AbstractTechniqueFactory
		{
			template <typename ProcInfoT2>
			struct QueueForwarder : QueueSelector<TechniqueT, ProcInfoT2, QueueSizeT, MaxSharedMemoryT>::Type {};

			typedef typename GetTechniqueType<QueueForwarder, ProcInfoT1, CustomTypeT, TechniqueT>::Type TechniqueType;

			static std::unique_ptr<TechniqueType, technique_deleter> create()
			{
				return std::unique_ptr<TechniqueType, technique_deleter>(new TechniqueType());
			}

		};

		template <Technique TechniqueT, typename ProcInfoT1, typename CustomTypeT1, unsigned int QueueSizeT, unsigned int MaxSharedMemoryT, unsigned int NumSubgraphsT, unsigned int NumEdgesT>
		struct AbstractTechniqueFactory<TechniqueT, ProcInfoT1, CustomTypeT1, QueueSizeT, MaxSharedMemoryT, true, NumSubgraphsT, NumEdgesT>
		{
			template <template <typename> class QueueT, typename ProcInfoT2, typename CustomTypeT2>
			struct TechniqueForwarder : GetTechniqueType<QueueT, ProcInfoT2, CustomTypeT2, TechniqueT>::Type {};

			template <typename ProcInfoT2>
			struct QueueForwarder : QueueSelector<TechniqueT, ProcInfoT2, QueueSizeT, MaxSharedMemoryT>::Type {};

			typedef PGA::Instrumentation::NoSchedTechniqueWrapper<TechniqueForwarder, QueueForwarder, ProcInfoT1, CustomTypeT1> TechniqueType;

			static std::unique_ptr<TechniqueType, technique_deleter> create()
			{
				PGA::Instrumentation::NoSchedMothership& getMothership(unsigned int, unsigned int);
				return std::unique_ptr<TechniqueType, technique_deleter>(new TechniqueType(getMothership(NumSubgraphsT, NumEdgesT)));
			}

		};

		////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename ListT, unsigned int CountT>
		struct ProcInfoListBuilder_Iterator
		{
			struct Result : public ProcInfo<typename ListT::template ItemAt<CountT - 1>::Result, typename ProcInfoListBuilder_Iterator<ListT, CountT - 1>::Result> {};

		};

		template <typename ListT>
		struct ProcInfoListBuilder_Iterator < ListT, 0 >
		{
			typedef ProcInfoEnd Result;

		};

		template <typename ListT, unsigned int NumPhasesT, template <class, int> class PhaseTraitsT, bool MultiphasedT>
		struct ProcInfoListBuilder_Selector
		{
			typedef typename ProcInfoListBuilder_Iterator<ListT, ListT::Length>::Result Result;

		};

		template <typename ListT, unsigned int NumPhasesT, template <class, int> class PhaseTraitsT>
		struct ProcInfoListBuilder_Selector < ListT, NumPhasesT, PhaseTraitsT, true >
		{
			typedef ProcInfoMultiPhase<NumPhasesT, PhaseTraitsT, NoPriority, typename ProcInfoListBuilder_Iterator<ListT, ListT::Length>::Result> Result;
		};

		template <typename ListT, unsigned int NumPhasesT, template <class, int> class PhaseTraitsT>
		struct ProcInfoListBuilder
		{
			typedef typename ProcInfoListBuilder_Selector<ListT, NumPhasesT, PhaseTraitsT, (NumPhasesT > 1)>::Result Result;

		};

		////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename ProcedureListT, unsigned int LengthT>
		struct ProcedureMatcher
		{
			template <typename ShapeT, typename QueueT>
			__device__ __inline__ static typename T::EnableIf<SAME_SHAPES(), bool>::Result enqueue(int operatorCode, unsigned int phaseIndex, unsigned char ruleTagId, Symbol<ShapeT>& shape, QueueT* queue);

			template <typename ShapeT, typename QueueT>
			__device__ __inline__ static typename T::EnableIf<!SAME_SHAPES(), bool>::Result enqueue(int operatorCode, unsigned int phaseIndex, unsigned char ruleTagId, Symbol<ShapeT>& shape, QueueT* queue);

			template <typename ShapeT, typename QueueT>
			__device__ __inline__ static typename T::EnableIf<SAME_SHAPES(), bool>::Result enqueueInitial(int operatorCode, unsigned int phaseIndex, unsigned char ruleTagId, Symbol<ShapeT>& shape, QueueT* queue);

			template <typename ShapeT, typename QueueT>
			__device__ __inline__ static typename T::EnableIf<!SAME_SHAPES(), bool>::Result enqueueInitial(int operatorCode, unsigned int phaseIndex, unsigned char ruleTagId, Symbol<ShapeT>& shape, QueueT* queue);

		};

		template <typename ProcedureListT>
		struct ProcedureMatcher < ProcedureListT, 0 >
		{
			template <typename ShapeT, typename QueueT>
			__device__ __inline__ static bool enqueue(int operatorCode, unsigned int phaseIndex, unsigned char ruleTagId, Symbol<ShapeT>& shape, QueueT* queue)
			{
				return false;
			}

			template <typename ShapeT, typename QueueT>
			__device__ __inline__ static bool enqueueInitial(int operatorCode, unsigned int phaseIndex, unsigned char ruleTagId, Symbol<ShapeT>& shape, QueueT* queue)
			{
				return false;
			}

		};

		template <typename ProcedureListT, unsigned int LengthT>
		template <typename ShapeT, typename QueueT>
		__device__ __inline__ typename T::EnableIf<SAME_SHAPES(), bool>::Result ProcedureMatcher<ProcedureListT, LengthT>::enqueue(int operatorCode, unsigned int phaseIndex, unsigned char ruleTagId, Symbol<ShapeT>& shape, QueueT* queue)
		{
			if (operatorCode == (LengthT - 1))
			{
				typedef typename ProcedureListT::template ItemAt<LengthT - 1>::Result Procedure;
				if (!T::EqualTypes<Operators::Discard, Procedure>::Result)
				{
					// NOTE: The value stored differs from 0 only when the current rule is in the intermediate symbols list and has a rule tag id (see also StaticDispatchTableBuilder.cuh)
					if (ruleTagId > 0)
						ContextSensitivity::PerShapeIntermediateSymbolsBufferAdapter<ShapeT>::store(ruleTagId, shape);
					queue->template enqueue<Procedure>(shape, phaseIndex);
				}
				return true;
			}
			return ProcedureMatcher<ProcedureListT, LengthT - 1>::enqueue(operatorCode, phaseIndex, ruleTagId, shape, queue);
		}

		template <typename ProcedureListT, unsigned int LengthT>
		template <typename ShapeT, typename QueueT>
		__device__ __inline__ typename T::EnableIf<SAME_SHAPES(), bool>::Result ProcedureMatcher<ProcedureListT, LengthT>::enqueueInitial(int operatorCode, unsigned int phaseIndex, unsigned char ruleTagId, Symbol<ShapeT>& shape, QueueT* queue)
		{
			if (operatorCode == (LengthT - 1))
			{
				typedef typename ProcedureListT::template ItemAt<LengthT - 1>::Result Procedure;
				if (!T::EqualTypes<Operators::Discard, Procedure>::Result)
				{
					// NOTE: The value stored differs from 0 only when the current rule is in the intermediate symbols list and has a rule tag id (see also StaticDispatchTableBuilder.cuh)
					if (ruleTagId > 0)
						ContextSensitivity::PerShapeIntermediateSymbolsBufferAdapter<ShapeT>::store(ruleTagId, shape);
					queue->template enqueueInitial<Procedure>(shape, phaseIndex);
				}
				return true;
			}
			return ProcedureMatcher<ProcedureListT, LengthT - 1>::enqueueInitial(operatorCode, phaseIndex, ruleTagId, shape, queue);
		}

		template <typename ProcedureListT, unsigned int LengthT>
		template <typename ShapeT, typename QueueT>
		__device__ __inline__ typename T::EnableIf<!SAME_SHAPES(), bool>::Result ProcedureMatcher<ProcedureListT, LengthT>::enqueue(int operatorCode, unsigned int phaseIndex, unsigned char ruleTagId, Symbol<ShapeT>& shape, QueueT* queue)
		{
			return ProcedureMatcher<ProcedureListT, LengthT - 1>::enqueue(operatorCode, phaseIndex, ruleTagId, shape, queue);
		}

		template <typename ProcedureListT, unsigned int LengthT>
		template <typename ShapeT, typename QueueT>
		__device__ __inline__ typename T::EnableIf<!SAME_SHAPES(), bool>::Result ProcedureMatcher<ProcedureListT, LengthT>::enqueueInitial(int operatorCode, unsigned int phaseIndex, unsigned char ruleTagId, Symbol<ShapeT>& shape, QueueT* queue)
		{
			return ProcedureMatcher<ProcedureListT, LengthT - 1>::enqueueInitial(operatorCode, phaseIndex, ruleTagId, shape, queue);
		}

		template <typename ProcedureListT, typename GenFuncFilterT, bool UseInstrumentationT>
		struct BaseSymbolManager
		{
			typedef ProcedureListT ProcedureList;
			static const bool UseInstrumentation = UseInstrumentationT;

			template <unsigned int OperatorCodeT, typename ShapeT, typename QueueT>
			__host__ __device__ __inline__ static void dispatchNonTerminal(Symbol<ShapeT>& symbol, unsigned int phaseIndex, unsigned char ruleTagId, QueueT* queue)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				if (ruleTagId > 0)
					ContextSensitivity::PerShapeIntermediateSymbolsBufferAdapter<ShapeT>::store(ruleTagId, symbol);
				if (T::IsEnabled<DebugFlags::SymbolDispatch>::Result)
					printf("Non-terminal symbol dispatched (OperatorCodeT=%d, symbol.entryIndex=%d, shapeType=%s, phaseIndex=%d, ruleTagId=%d)\n", OperatorCodeT, symbol.entryIndex, Shapes::GetName<ShapeT>::Result(), QueueT::Phase, static_cast<unsigned int>(ruleTagId));
				typedef typename ProcedureListT::template ItemAt<OperatorCodeT>::Result Procedure;
				if (!T::EqualTypes<Operators::Discard, typename Procedure::FirstOperator>::Result)
					queue->template enqueue<Procedure>(symbol, phaseIndex);
#endif
			}

			template <typename ShapeT, typename QueueT>
			__host__ __device__ __inline__ static void dispatchNonTerminal(Symbol<ShapeT>& symbol, unsigned int phaseIndex, unsigned char ruleTagId, QueueT* queue)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				const DispatchTableEntry& dispatchTableEntry = Device::DispatchTable[symbol.entryIndex];
				if (T::IsEnabled<DebugFlags::SymbolDispatch>::Result)
					printf("Non-terminal symbol dispatched (symbol.entryIndex=%d, shapeType=%s, operatorCode=%d, phaseIndex=%d, ruleTagId=%d)\n", symbol.entryIndex, Shapes::GetName<ShapeT>::Result(), dispatchTableEntry.operatorCode, QueueT::Phase, static_cast<unsigned int>(ruleTagId));
				if (!ProcedureMatcher<ProcedureListT, ProcedureListT::Length>::enqueue(dispatchTableEntry.operatorCode, phaseIndex, ruleTagId, symbol, queue))
				{
					printf("PGA::GPU::BaseSymbolManager::dispatchNonTerminal(): missing procedure declaration in the procedure list [dispatchTableIndex=%d, shapeType=%s, operatorCode=%d, phaseIndex=%d, ruleTagId=%d]\n", symbol.entryIndex, Shapes::GetName<ShapeT>::Result(), dispatchTableEntry.operatorCode, QueueT::Phase, static_cast<unsigned int>(ruleTagId));
					asm("trap;");
				}
#endif
			}

			template <unsigned int OperatorCodeT, typename ShapeT, typename QueueT>
			__host__ __device__ __inline__ static void dispatchAxiom(ShapeT& shape, QueueT* queue, int entryIndex = -1)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				Symbol<ShapeT> symbol(shape);
				symbol.entryIndex = entryIndex;
				if (T::IsEnabled<DebugFlags::SymbolDispatch>::Result)
					printf("Axiom dispatched [compile-time resolution] (OperatorCodeT=%d, shapeType=%s, entryIndex=%d)\n", OperatorCodeT, Shapes::GetName<ShapeT>::Result(), entryIndex);
				typedef typename ProcedureListT::template ItemAt<OperatorCodeT>::Result Procedure;
				static_assert(T::EqualTypes<ShapeT, GetSymbolShape<typename Procedure::ExpectedData>::Result>::Result, "enqueueing for a procedure with different shape type");
				if (!T::EqualTypes<Operators::Discard, typename Procedure::FirstOperator>::Result)
					queue->template enqueue<Procedure>(symbol, 0);
#endif
			}

			template <typename ShapeT, typename QueueT>
			__host__ __device__ __inline__ static void dispatchAxiom(ShapeT& shape, int entryIndex, QueueT* queue)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				Symbol<ShapeT> symbol(shape);
				symbol.entryIndex = entryIndex;
				const DispatchTableEntry& dispatchTableEntry = Device::DispatchTable[entryIndex];
				if (T::IsEnabled<DebugFlags::SymbolDispatch>::Result)
					printf("Axiom dispatched [run-time resolution] (entryIndex=%d, shapeType=%s, operatorCode=%d)\n", entryIndex, Shapes::GetName<ShapeT>::Result(), dispatchTableEntry.operatorCode);
				if (!ProcedureMatcher<ProcedureListT, ProcedureListT::Length>::enqueueInitial(dispatchTableEntry.operatorCode, 0, 0, symbol, queue))
				{
					printf("PGA::GPU::BaseSymbolManager::dispatchAxiom(): missing procedure declaration in the procedure list [dispatchTableIndex=%d, shapeType=%s, operatorCode=%d]\n", symbol.entryIndex, Shapes::GetName<ShapeT>::Result(), dispatchTableEntry.operatorCode);
					asm("trap;");
				}
#endif
			}

			template <unsigned int GenFuncIdxT, typename ShapeT, typename... ArgsT>
			__host__ __device__ __inline__ static void dispatchTerminal(Symbol<ShapeT>& symbol, ArgsT... args)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				if (T::IsEnabled<DebugFlags::SymbolDispatch>::Result)
					printf("Terminal symbol dispatched [serial] (entryIndex=%d, shapeType=%s, GenFuncIdxT=%d)\n", symbol.entryIndex, Shapes::GetName<ShapeT>::Result(), GenFuncIdxT);
				ShapeGenerator<ShapeT, false>::template run<GenerationFunction<GenFuncFilterT, GenFuncIdxT>>(symbol, args...);
#endif
			}

			template <unsigned int GenFuncIdxT, typename ContextT, unsigned int NumThreadsT, typename ShapeT, typename... ArgsT>
			__device__ __inline__ static void dispatchTerminalInParallel(int localThreadId, Symbol<ShapeT>& symbol, ArgsT... args)
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				static_assert(NumThreadsT >= ShapeGenerator<ShapeT, true>::NumThreads, "using less threads than the required by the shape generation function");
				if (T::IsEnabled<DebugFlags::SymbolDispatch>::Result)
				{
					if (localThreadId == 0)
						printf("Terminal symbol dispatched [parallel] (entryIndex=%d, shapeType=%s, GenFuncIdxT=%d, NumThreadsT=%d)\n", symbol.entryIndex, Shapes::GetName<ShapeT>::Result(), GenFuncIdxT, NumThreadsT);
				}
				ShapeGenerator<ShapeT, true>::template run<GenerationFunction<GenFuncFilterT, GenFuncIdxT>, ContextT>(localThreadId, symbol, args...);
#endif
			}

		};

		////////////////////////////////////////////////////////////////////////////////////////////////////
		template <typename AxiomGeneratorT, typename SymbolManagerT>
		struct InitFunc
		{
			static const bool reuseInit = false;

			template <typename QueueT>
			__device__ __inline__ static void init(QueueT* queue, int id)
			{
				AxiomGeneratorT::template generateAxiom<SymbolManagerT>(id, queue);
			}

		};

		////////////////////////////////////////////////////////////////////////////////////////////////////
		class Evaluator
		{
		protected:
			DispatchTableEntry* deviceDispatchTable;

		public:
			Evaluator() : deviceDispatchTable(0) {}

			virtual double execute() = 0;
			virtual double execute(unsigned int numAxioms) = 0;

			virtual void initialize(DispatchTableEntry* dispatchTable, size_t numEntries)
			{
				if (dispatchTable != nullptr)
				{
					PGA_CUDA_checkedCall(cudaMalloc((void **)&deviceDispatchTable, numEntries * sizeof(DispatchTableEntry)));
					PGA_CUDA_checkedCall(cudaMemcpy(deviceDispatchTable, dispatchTable, numEntries * sizeof(DispatchTableEntry), cudaMemcpyHostToDevice));
					PGA_CUDA_checkedCall(cudaMemcpyToSymbol(Device::DispatchTable, &deviceDispatchTable, sizeof(DispatchTableEntry*)));
				}
				PGA_CUDA_checkedCall(cudaMemcpyToSymbol(Device::NumEntries, &numEntries, sizeof(unsigned int)));
			}

			virtual void release()
			{
				if (deviceDispatchTable)
				{
					PGA_CUDA_checkedCall(cudaFree(deviceDispatchTable));
					deviceDispatchTable = 0;
				}
			}

		};

		////////////////////////////////////////////////////////////////////////////////////////////////////
		struct DefaultConfiguration
		{
			static const PGA::GPU::Technique Technique = PGA::GPU::Technique::MEGAKERNEL;
			static const unsigned int QueueSize = 1024 * 128;
			static const unsigned int MaxSharedMemory = 0;

		};

		////////////////////////////////////////////////////////////////////////////////////////////////////
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
		class MultiPhaseEvaluator : public Evaluator
		{
		public:
			typedef BaseSymbolManager<ProcedureListT, GenFuncFilterT, UseInstrumentationT> SymbolManager;

		private:
			typedef typename ProcInfoListBuilder<ProcedureListT, NumPhasesT, PhaseTraitsT>::Result ProcInfoList;
			typedef AbstractTechniqueFactory<ConfigurationT::Technique, ProcInfoList, SymbolManager, ConfigurationT::QueueSize, ConfigurationT::MaxSharedMemory, UseInstrumentationT, NumSubgraphsT, NumEdgesT> TechniqueFactory;
			std::unique_ptr<typename TechniqueFactory::TechniqueType, technique_deleter> gpuTechnique;

		public:
			virtual double execute()
			{
				return execute(AxiomGeneratorT::getNumAxioms());
			}

			virtual double execute(unsigned int numAxioms)
			{
				gpuTechnique->template insertIntoQueue<InitFunc<AxiomGeneratorT, SymbolManager>>(numAxioms);
				ContextSensitivity::IntermediateSymbolsBufferAdapter::reset();
				double executionTimes[NumPhasesT];
				double totalExecutionTime = 0;
				for (unsigned int i = 0, j = NumPhasesT - 1; i < NumPhasesT; i++, j--)
				{
					//if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					//	PGA_CUDA_checkedCall(cudaMemcpyToSymbol(ContextSensitivity::Device::NumCollisionChecks, &numCollisionChecks, sizeof(unsigned int)));
					double executionTime = gpuTechnique->execute(i);
					if (j > 0)
						executionTime += ContextSensitivity::GPU::BVHConstructor::construct(ContextSensitivity::Constants::SceneBoundaryMin, ContextSensitivity::Constants::SceneBoundaryMax);
					totalExecutionTime += executionTime;
					executionTimes[i] = executionTime;
				}
				if (T::IsEnabled<Statistics::Execution>::Result)
				{
					std::cout << "************************************************************" << std::endl;
					std::cout << "GPU Multi-Phased Execution Statistics:" << std::endl;
					std::cout << "************************************************************" << std::endl;
					std::cout << "Num. Axioms: " << numAxioms << std::endl;
					std::cout << "Elapsed Time (Total): " << totalExecutionTime << "s" << std::endl;
					for (auto i = 0; i < NumPhasesT; i++)
					{
						std::cout << "Phase " << i;
						std::cout << "Elapsed Time: " << executionTimes[i] << "s" << std::endl;
//						if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
//						{
//							PGA_CUDA_checkedCall(cudaMemcpyFromSymbol(&numCollisionChecks, ContextSensitivity::Device::NumCollisionChecks, sizeof(unsigned int)));
//							std::cout << "[Phase " << i << "] Num. Collision Checks: " << numCollisionChecks << std::endl;
//						}
					}
				}
				return totalExecutionTime;
			}

			virtual void initialize(DispatchTableEntry* dispatchTable, size_t numEntries)
			{
				Evaluator::initialize(dispatchTable, numEntries);
				ContextSensitivity::IntermediateSymbolsBufferAdapter::initialize();
				gpuTechnique = TechniqueFactory::create();
				gpuTechnique->init();
			}

			virtual void release()
			{
				Evaluator::release();
				ContextSensitivity::IntermediateSymbolsBufferAdapter::release();
				gpuTechnique->release();
			}

		};

		////////////////////////////////////////////////////////////////////////////////////////////////////
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
		class SinglePhaseEvaluator : public Evaluator
		{
		public:
			typedef BaseSymbolManager<ProcedureListT, GenFuncFilterT, UseInstrumentationT> SymbolManager;

			template <typename ProcedureT, int PhaseIndexT>
			struct PermissivePhaseTraits
			{
				static const bool Active = true;

			};

		private:
			typedef typename ProcInfoListBuilder<ProcedureListT, 1, PermissivePhaseTraits>::Result ProcInfoList;
			typedef AbstractTechniqueFactory<ConfigurationT::Technique, ProcInfoList, SymbolManager, ConfigurationT::QueueSize, ConfigurationT::MaxSharedMemory, UseInstrumentationT, NumSubgraphsT, NumEdgesT> TechniqueFactory;
			std::unique_ptr<typename TechniqueFactory::TechniqueType, technique_deleter> gpuTechnique;

		public:
			virtual double execute()
			{
				return execute(AxiomGeneratorT::getNumAxioms());
			}

			virtual double execute(unsigned int numAxioms)
			{
				gpuTechnique->template insertIntoQueue<InitFunc<AxiomGeneratorT, SymbolManager>>(numAxioms);
				double totalExecutionTime = gpuTechnique->execute(0);
				if (T::IsEnabled<Statistics::Execution>::Result)
				{
					std::cout << "************************************************************" << std::endl;
					std::cout << "GPU Single-Phase Execution Statistics:" << std::endl;
					std::cout << "************************************************************" << std::endl;
					std::cout << "Num. Axioms: " << numAxioms << std::endl;
					std::cout << "Elapsed Time: " << totalExecutionTime << "s" << std::endl;
				}
				return totalExecutionTime;
			}

			virtual void initialize(DispatchTableEntry* dispatchTable, size_t numEntries)
			{
				Evaluator::initialize(dispatchTable, numEntries);
				gpuTechnique = TechniqueFactory::create();
				gpuTechnique->init();
			}

			virtual void release()
			{
				Evaluator::release();
				gpuTechnique->release();
			}

		};

	}

}
