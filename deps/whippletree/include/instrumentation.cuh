#ifndef INCLUDED_INSTRUMENTATION_CUH
#define INCLUDED_INSTRUMENTATION_CUH

#pragma once

#include <cstdint>
#include <memory>
#include <map>
#include <string>
#include <iostream>

#include "cuda_memory.h"
#include "common.cuh"

#include "procedureInterface.cuh"
#include "queueInterface.cuh"
#include "procinfoTemplate.cuh"
#include "techniqueInterface.h"

#include "instrumentation.h"

namespace Instrumentation
{
	namespace device
	{
		template <class PROC>
		class ProcedureWrapper;

		template<class PROCINFO>
		struct WrappedProcInfo;
	}
}

template <class QUEUE>
struct getProcInfoFromQueue;

template <template <class> class QUEUE, class ProcInfo>
struct getProcInfoFromQueue<QUEUE<ProcInfo> >
{
	typedef ProcInfo type;
};


template <template <class> class QUEUE, class ProcInfo, int Phase>
struct getProcInfoFromQueue< CurrentMultiphaseQueue<MultiPhaseQueue<ProcInfo, QUEUE>, Phase > >
{
	typedef ProcInfo type;
};



namespace Instrumentation
{

#if defined(__CUDACC__)
	namespace device
	{
		typedef int int32;
		typedef long long int64;
		typedef unsigned int uint32;
		typedef unsigned long long uint64;

		const size_t MAX_NUM_MULTIPROCESSORS = 16;
		const int maxNumConcurrentWorkpackages = 8;
		const int maxDifferentProcEnqueues = 8; //must be a multiple of 4!

		const size_t N = 16U * 1024U * 1024U;
		__constant__ ulonglong2* buffer;
		__constant__ long long clock_calibration_offsets[MAX_NUM_MULTIPROCESSORS];
		__device__ uint32 buffer_counter = 0U;


		/*

		Procedure:

		|63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37|36|35|34|33|32|31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10| 9| 8| 7| 6| 5| 4| 3| 2| 1| 0|
		|                                                                                          timestamp                                                                                            |
		|63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37|36|35|34|33|32|31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10| 9| 8| 7| 6| 5| 4| 3| 2| 1| 0|
		|                                                                                          timestamp                                                                                            |
		|63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37|36|35|34|33|32|31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10| 9| 8| 7| 6| 5| 4| 3| 2| 1| 0|
		|             pid                |           mpid           |            active_threads         |                                     overhead counter                                          |
		|63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37|36|35|34|33|32|31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10| 9| 8| 7| 6| 5| 4| 3| 2| 1| 0|
		|                                         dequeue time                                          |xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx| num following enqueues|

		*/

		/*
		|63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37|36|35|34|33|32|31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10| 9| 8| 7| 6| 5| 4| 3| 2| 1| 0|
		|             pid                |                              num                             |             pid                |                              num                             |
		|63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37|36|35|34|33|32|31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10| 9| 8| 7| 6| 5| 4| 3| 2| 1| 0|
		|             pid                |                              num                             |             pid                |                              num                             |
		*/

		__shared__ volatile int num_threads;
		//const int num_threads = 1024;
		__shared__ volatile uint64 t0[maxNumConcurrentWorkpackages];
		__shared__ volatile uint32 overhead_counter[maxNumConcurrentWorkpackages];
		__shared__ volatile uint64 dequeue_start;

		const int maxDifferentProcEnqeueusBlock = maxNumConcurrentWorkpackages*maxDifferentProcEnqueues;
		__shared__ volatile unsigned int enqueueCounters[2*maxDifferentProcEnqeueusBlock];


		template <unsigned int start, unsigned int length>
		__device__ inline unsigned long long packField(unsigned long long v)
		{
			return (v & ~(~0ULL << length)) << start;
		}

		template <int ProcID, int NumThreads>
		__device__ inline void writeInstrumentationData(uint64 t1, int active_threads)
		{
			// count matching enqueue counters
			unsigned int mygroup = threadIdx.x / NumThreads;
			int myNumEnqueues = 0;
			for(int i = 0; i < maxDifferentProcEnqueues; ++i)
			{
				if(enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+i)] != 0)
					++myNumEnqueues;
			}
			uint reqBlocks = 2 + (myNumEnqueues + 3) / 4;
			uint i = atomicAdd(&buffer_counter, reqBlocks);
			if (i + reqBlocks < N)
			{
				unsigned int mpid = Softshell::smid();
				buffer[i] = make_ulonglong2(t0[threadIdx.x / num_threads] + clock_calibration_offsets[mpid], t1 + clock_calibration_offsets[mpid]);
				buffer[i + 1] = make_ulonglong2(packField<53, 11>(ProcID) |
					packField<44, 8>(mpid) |
					packField<32, 12>(active_threads) |
					packField<0, 32>(overhead_counter[threadIdx.x / num_threads]/((num_threads+31)/32)),
					packField<32, 32>(t0[threadIdx.x / num_threads] - dequeue_start) | 
					packField<0, 8>(myNumEnqueues));

				for(int j = 0; j < myNumEnqueues; j+=4)
				{
					//printf("%d %d writing to global:%d %d, %d %d, %d %d, %d %d\n",blockIdx.x,threadIdx.x, 
					//enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+0)+0] - 1, enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+0)+1],
					//enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+1)+0] - 1, enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+1)+1],
					//enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+2)+0] - 1, enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+2)+1],
					//enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+3)+0] - 1, enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+3)+1]);
					buffer[i + 2 + j] = make_ulonglong2(packField<53, 11>(enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+1)+0] - 1) | packField<32, 21>(enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+1)+1] ) | 
						packField<21, 11>(enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+0)+0] - 1) | packField<0, 21>(enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+0)+1] ) , 
						packField<53, 11>(enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+3)+0] - 1) | packField<32, 21>(enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+3)+1] ) | 
						packField<21, 11>(enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+2)+0] - 1) | packField<0, 21>(enqueueCounters[2*(maxDifferentProcEnqueues*mygroup+2)+1] ) );
				}
			}
		}

		__device__ void enqueueWrite(int en)
		{			 
			if(__any(en) && __any(1-en))
				return;
			if(__popc(Softshell::lanemask_lt() & __ballot(1)) == 0)
				atomicAdd((uint32*)&overhead_counter[threadIdx.x / num_threads],(1-2*en)*clock());
			__threadfence_block();

			//{
			//	if(en)
			//		atomicAdd((uint32*)&overhead_counter[threadIdx.x / num_threads],-clock());
			//	else
			//		atomicAdd((uint32*)&overhead_counter[threadIdx.x / num_threads],clock());
			//	__threadfence_block();
			//}
		}
		template<int NumThreads>
		__device__ void enqueueBegin(unsigned int procId)
		{
			int myGroup = threadIdx.x / num_threads;
			// get num enqueues
			unsigned int mask = __ballot(1);
			int num = __popc(mask) / NumThreads;
			// only leader writes
			if(__popc(Softshell::lanemask_lt() & mask) == 0)
			{
				//find / create matching entry
				for(int i = 0; i < maxDifferentProcEnqueues; ++i)
				{
					unsigned int prevProcId = atomicCAS(const_cast<unsigned int*>(enqueueCounters + 2*(maxDifferentProcEnqueues*myGroup + i)), 0, procId+1);
					//printf("%d %d atomicCAS @ %d, wanted: %d, was: %d\n",blockIdx.x,threadIdx.x, 2*(maxDifferentProcEnqueues*myGroup + i), procId+1,prevProcId);
					if(prevProcId == 0 || prevProcId == procId+1)
					{
						int c = atomicAdd(const_cast<unsigned int*>(enqueueCounters + 2*(maxDifferentProcEnqueues*myGroup + i) + 1), num);
						//printf("%d %d wrote %d for %d @ %d: %d\n",blockIdx.x,threadIdx.x, num, procId+1,2*(maxDifferentProcEnqueues*myGroup + i),c+num);

						break;
					}
				}
			}

			enqueueWrite(1);

			//atomicAdd((int64*)&overhead_counter[threadIdx.x / num_threads],-clock64());
			//__threadfence_block();
		}

		__device__ void enqueueEnd()
		{
			enqueueWrite(0);

			//atomicAdd((int64*)&overhead_counter[threadIdx.x / num_threads],clock64());
			//__threadfence_block();
		}

		__device__ void dequeueBegin()
		{
			if(threadIdx.x == 0)
				dequeue_start = clock64();
		}

		template <class Sync, bool ItemInput>
		struct ProcedureWrapperSync
		{
			static __device__ __inline__ void sync(int numThreads)
			{
				Softshell::syncthreads(3,(numThreads+31)/32*32);
			}
		};

		template <class Sync>
		struct ProcedureWrapperSync<Sync, false>
		{
			static __device__ __inline__ void sync(int numThreads)
			{
				Sync::sync();
			}
		};


		template <class PROC>
		class ProcedureWrapper : public PROC
		{
		public:
			static const int NumThreads = PROC::NumThreads;
			static const bool ItemInput = PROC::ItemInput;
			static const int sharedMemory = PROC::sharedMemory;
			static const bool InitialProcedure = PROC::InitialProcedure;
			typedef typename PROC::ExpectedData ExpectedData;

			template<class Q, class Context>
			static __device__ __inline__ void execute(int threadId, int numThreads, Q* queue, ExpectedData* data, unsigned int* shared_mem)
			{
				// clear enqueue data
				for(int i = threadId; i < 2*maxDifferentProcEnqueues; i+= numThreads)
				{
					int group = ItemInput ? 0 : threadIdx.x / NumThreads;
					//printf("%d %d clearing @ %d\n",blockIdx.x,threadIdx.x, 2*maxDifferentProcEnqueues*group + i);
					enqueueCounters[2*maxDifferentProcEnqueues*group + i] = 0;
				}

				num_threads = numThreads;
				if (threadId == 0)
				{
					//num_threads = numThreads; //(PROC::ItemInput || PROC::NumThreads == 0) ? blockDim.x :	PROC::NumThreads;
					t0[threadIdx.x / numThreads] = clock64();
					overhead_counter[threadIdx.x / numThreads] = 0UL;
					__threadfence_block();
				}

				ProcedureWrapperSync<Context, ItemInput>::sync(numThreads);
				PROC::template execute<Q, Context>(threadId, numThreads, queue, data, shared_mem);
				ProcedureWrapperSync<Context, ItemInput>::sync(numThreads);
				if (threadId == 0)
					writeInstrumentationData< 
					findProcId<
					typename getProcInfoFromQueue<Q>::type,
					ProcedureWrapper>::value,
					NumThreads>(clock64(), numThreads);
			}
		};

		template<class PROCINFO>
		struct nextWrappedProcInfo;

		template<class PROCINFO>
		struct WrappedProcInfoSelector;

		template<template <typename head, typename tail> class PI, class TProc, class TNext>
		struct nextWrappedProcInfo<PI<TProc, TNext> >
		{
			typedef typename WrappedProcInfoSelector<TNext>::WrappedProcInfo type;
		};

		template<template <typename head, typename tail> class PI, class TProc>
		struct nextWrappedProcInfo<PI<TProc, ProcInfoEnd> >
		{
			typedef ProcInfoEnd type;
		};


		template<class PROCINFO>
		struct WrappedProcInfoSelector
		{

			struct WrappedProcInfo : public PROCINFO
			{
				typedef ProcedureWrapper<typename PROCINFO::Procedure> Procedure;
				typedef typename nextWrappedProcInfo<PROCINFO>::type Next;
				typedef PROCINFO ProcInfo;
			};
		};


		template<class Proc>
		struct UnwrapProc
		{
			typedef Proc type;
		};

		template<class Proc>
		struct UnwrapProc<ProcedureWrapper<Proc>>
		{
			typedef Proc type;
		};

		template<int TNumPhases, template<class /*Proc*/, int /*Phase*/> class TPhaseTraits, template <int /*Phase*/> class TPriority, class TProcInfo>
		struct WrappedProcInfoSelector<ProcInfoMultiPhase<TNumPhases, TPhaseTraits, TPriority, TProcInfo> >
		{
			template<class Proc, int Phase>
			struct WrappedPhaseTraits
			{
				static const bool Active = TPhaseTraits<typename UnwrapProc<Proc>::type,Phase>::Active;
			};

			struct WrappedProcInfo : public ProcInfoMultiPhase<TNumPhases, WrappedPhaseTraits, TPriority, typename WrappedProcInfoSelector<TProcInfo>::WrappedProcInfo >
			{  };
		};

		


	}
}
#endif

//make sure traits for the shared queue are passed through the instrumentation
template< template<typename> class SQTraits,class Procedure>
struct GetTraitQueueSize;

template<template<typename> class SQTraits, class Procedure>
struct GetTraitQueueSize<SQTraits,Instrumentation::device::ProcedureWrapper<Procedure> > : public GetTraitQueueSize<SQTraits,Procedure>
{ };


namespace Instrumentation
{
	template <typename ProcInfo>
	struct BuildProcedureDesc;

	template <>
	struct BuildProcedureDesc<ProcInfoEnd>
	{
		static void gen(std::vector<ProcedureDescription>& desc) {}
	};

	template <typename ProcInfo>
	struct BuildProcedureDesc
	{
		static void gen(std::vector<ProcedureDescription>& desc)
		{
			desc.push_back(ProcedureDescription::generate<typename ProcInfo::Procedure, findProcId<ProcInfo, typename ProcInfo::Procedure>::value>());
			BuildProcedureDesc<typename ProcInfo::Next>::gen(desc);
		}
	};


	template <template <template <class> class /*Queue*/, class /*ProcInfo*/, class /*ApplicationContext*/> class RealTechnique, template <class> class QUEUE, /*class INITFUNC, */class PROCINFO, class ApplicationContext>
	class TechniqueWrapper
	{
	public:
		template<class ProcInfo>
		class WrapperQueue : public QUEUE<ProcInfo>
		{
		private:
			typedef QUEUE<ProcInfo> Q;

		public:

			template<class PROCEDURE>
			__inline__ __device__ bool enqueueInitial(typename PROCEDURE::ExpectedData data)
			{
				//device::enqueueBegin<1>(findProcId<typename getProcInfoFromQueue<Q>::type, device::ProcedureWrapper<PROCEDURE> >::value);
				bool res = Q::template enqueueInitial<device::ProcedureWrapper<PROCEDURE> >(data);
				//device::enqueueEnd();
				return res;
			}

			template<class PROCEDURE>
			__inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData data)
			{
				device::enqueueBegin<1>(findProcId<typename getProcInfoFromQueue<Q>::type, device::ProcedureWrapper<PROCEDURE> >::value);
				bool res = Q::template enqueue<device::ProcedureWrapper<PROCEDURE> >(data);
				device::enqueueEnd();
				return res;
			}

			template<int threads, class PROCEDURE>
			__inline__ __device__ bool enqueue(typename PROCEDURE::ExpectedData* data)
			{
				device::enqueueBegin<threads>(findProcId<typename getProcInfoFromQueue<Q>::type, device::ProcedureWrapper<PROCEDURE> >::value);
				bool res = Q::template enqueue<threads, device::ProcedureWrapper<PROCEDURE> >(data);
				device::enqueueEnd();
				return res;
			}

			template<class PROCEDURE>
			__inline__ __device__ typename PROCEDURE::ExpectedData* reserveSpot()
			{
				device::enqueueBegin<1>(findProcId<typename getProcInfoFromQueue<Q>::type, device::ProcedureWrapper<PROCEDURE> >::value);
				typename PROCEDURE::ExpectedData* res = Q::template reserveSpot <device::ProcedureWrapper<PROCEDURE> > ();
				device::enqueueEnd();
				return res;
			}

			template<int threads, class PROCEDURE>
			__inline__ __device__ typename PROCEDURE::ExpectedData* reserveSpot()
			{
				device::enqueueBegin<threads>(findProcId<typename getProcInfoFromQueue<Q>::type, device::ProcedureWrapper<PROCEDURE> >::value);
				typename PROCEDURE::ExpectedData* res = Q::template reserveSpot <threads, device::ProcedureWrapper<PROCEDURE> > ();
				device::enqueueEnd();
				return res;
			}

			template<class PROCEDURE>
			__inline__ __device__ void completeSpot(typename PROCEDURE::ExpectedData* spot)
			{
				device::enqueueBegin<1>(findProcId<typename getProcInfoFromQueue<Q>::type, device::ProcedureWrapper<PROCEDURE> >::value);
				Q::template completeSpot <1, device::ProcedureWrapper<PROCEDURE> > (spot);
				device::enqueueEnd();
			}

			template<int threads, class PROCEDURE>
			__inline__ __device__ void completeSpot(typename PROCEDURE::ExpectedData* spot)
			{
				device::enqueueBegin<threads>(findProcId<typename getProcInfoFromQueue<Q>::type, device::ProcedureWrapper<PROCEDURE> >::value);
				Q::template completeSpot <threads, device::ProcedureWrapper<PROCEDURE> > (spot);
				device::enqueueEnd();
			}


			template<bool MultiProcedure>
			__inline__ __device__ int dequeue(void*& data, int*& procId, int maxShared = -1)
			{
				device::dequeueBegin();
				return Q::template dequeue<MultiProcedure>(data,procId,maxShared);
			}

			template<bool MultiProcedure>
			__inline__ __device__ int dequeueSelected(void*& data, int procId, int maxNum = -1)
			{
				device::dequeueBegin();
				return	Q::template dequeueSelected<MultiProcedure>(data,procId,this->maxShared);
			}

			template<bool MultiProcedure>
			__inline__ __device__ int dequeueStartRead(void*& data, int*& procId, int maxShared = -1)
			{
				device::dequeueBegin();
				return	Q::template dequeueStartRead<MultiProcedure>(data,procId,maxShared);
			}

			template<class PROCEDURE>
			__inline__ __device__ int reserveRead(int maxNum = -1)
			{
				device::dequeueBegin();
				return	Q::template reserveRead<PROCEDURE>(maxNum);
			}

		};

	private:
		typedef RealTechnique<WrapperQueue, /*INITFUNC, */ typename device::WrappedProcInfoSelector<PROCINFO>::WrappedProcInfo, ApplicationContext> TechniqueType;

		std::unique_ptr<TechniqueType, technique_deleter> realTechnique;
		Mothership& mothership;
		std::vector<ProcedureDescription> proceduresDescriptions;

	public:

		TechniqueWrapper(Mothership& mothership) : 
			realTechnique(new TechniqueType()),
			mothership(mothership)
		{
			BuildProcedureDesc<PROCINFO>::gen(proceduresDescriptions);
		}

		void init()
		{
			realTechnique->init();
		}


		void resetQueue()
		{
			realTechnique->resetQueue();
		}

		void recordQueue()
		{
			realTechnique->recordQueue();
		}

		void restoreQueue()
		{
			realTechnique->restoreQueue();
		}

		template<class InsertFunc>
		void insertIntoQueue(int num)
		{
			realTechnique->insertIntoQueue<InsertFunc>(num);
		}

		std::string name() const
		{
			return "instrumented " + realTechnique->name();
		}

		std::string actualTechniqueName() const
		{
			return realTechnique->name();
		}

		void release()
		{
			delete this;
		}


		double execute(int phase = 0, double timelimitInMs = 0)
		{
			mothership.prepare();
			double time = realTechnique->execute(phase, timelimitInMs);
			mothership.computeTimings(proceduresDescriptions);
			return time;
		}


		template<int Phase, int TimeLimitInKCycles>
		double execute()
		{
			mothership.prepare();
			double time = realTechnique->execute<Phase,TimeLimitInKCycles>();
			mothership.computeTimings(proceduresDescriptions);
			return time;
		}

		template<int Phase>
		double execute()
		{
			mothership.prepare();
			double time = realTechnique->execute<Phase>();
			mothership.computeTimings(proceduresDescriptions);
			return time;
		}
	};

#if defined(__CUDACC__)
	__global__ void initClockCalibrationOffsets(long long* offsets)
	{
		uint smid = Softshell::smid();
		//if (threadIdx.x == 0)
		offsets[smid] = 16777216-static_cast<long long>(clock64());
	}
#endif

	void Mothership::calibrateClockCounters()
	{
		initClockCalibrationOffsets<<<prop.multiProcessorCount, 1, prop.sharedMemPerBlock>>>((long long*)d_clock_calibration_offsets.get());
		CUDA_CHECKED_CALL(cudaMemcpyToSymbol(device::clock_calibration_offsets, d_clock_calibration_offsets.get(), 8 * prop.multiProcessorCount, 0, cudaMemcpyDeviceToDevice));
	}
	
	Mothership::Mothership()
		: d_buffer(cudaAllocArray<ulonglong2>(device::N)),
		processor(nullptr)
	{
		int device;
		CUDA_CHECKED_CALL(cudaGetDevice(&device));
		CUDA_CHECKED_CALL(cudaGetDeviceProperties(&prop, device));

		CUDA_CHECKED_CALL(cudaEventCreate(&t0));
		CUDA_CHECKED_CALL(cudaEventCreate(&t1));

		d_clock_calibration_offsets = cudaAllocArray<std::int64_t>(prop.multiProcessorCount);

		ulonglong2* ptr = d_buffer.get();
		CUDA_CHECKED_CALL(cudaMemcpyToSymbol(device::buffer, &ptr, sizeof(ptr)));

		for (unsigned int i = 0; i < numBuffers; ++i)
		{
			buffers[0] = std::unique_ptr<unsigned char[]>(new unsigned char[16 * device::N]);
		}

		CUDA_CHECKED_CALL(cudaEventRecord(t0));
	}

	void Mothership::prepare()
	{
		if (processor)
		{
			calibrateClockCounters();
		}

		std::uint32_t count = 0;
		CUDA_CHECKED_CALL(cudaMemcpyToSymbol(device::buffer_counter, &count, sizeof(count)));

		cudaEventRecord(t1);
	}

	void Mothership::computeTimings(const std::vector<ProcedureDescription>& proceduresDescriptions)
	{
		float baseTime;
		CUDA_CHECKED_CALL(cudaEventElapsedTime(&baseTime, t0, t1));
		if (processor)
		{
			std::uint32_t count;
			CUDA_CHECKED_CALL(cudaMemcpyFromSymbol(&count, device::buffer_counter, sizeof(count)));
			if (count >= device::N)
			{
				std::cerr << "!WARNING! instrumentation buffer overflow detected\n";
				count = device::N;
			}
			size_t bufferSize = sizeof(ulonglong2) * count;
			CUDA_CHECKED_CALL(cudaMemcpy(&buffers[0][0], d_buffer.get(), bufferSize, cudaMemcpyDeviceToHost));
			processData(reinterpret_cast<const unsigned char*>(&buffers[0][0]), bufferSize, baseTime, proceduresDescriptions);
		}
	}
}

#endif	// INCLUDED_INSTRUMENTATION_CUH
