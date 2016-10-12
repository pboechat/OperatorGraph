#pragma once

#include <cstdint>
#include <memory>
#include <map>
#include <string>
#include <sstream>
#include <iostream>

#include <procedureInterface.cuh>
#include <queueInterface.cuh>
#include <procinfoTemplate.cuh>
#include <queuingMultiPhase.cuh>
#include <techniqueInterface.h>

#include "CUDAException.h"
#include "Experiment.h"
#include "Instrumentation.h"

static const char* events[] = {
	"atom_count",
	"gld_request",
	"gst_request",

};

namespace PGA
{
	namespace Instrumentation
	{
		namespace Device
		{
			const size_t N = 16U * 1024U * 1024U;

			__constant__ InstrumentationData* edgeData;
			__constant__ InstrumentationData* subGraphData;

		}

		namespace Host
		{
			InstrumentationData* edgeData;
			InstrumentationData* subGraphData;

		}

		__host__ __device__ __inline__ void writeEdgeData(unsigned long long t0, unsigned long long t1, unsigned int edgeIdx, unsigned int callType)
		{
			auto duration = t1 - t0;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			if (Device::edgeData[edgeIdx].callType == 0)
				Device::edgeData[edgeIdx].callType = callType;
			atomicAdd(&Device::edgeData[edgeIdx].numTraversals, 1);
			atomicAdd(&Device::edgeData[edgeIdx].duration, duration);
#else
			if (Host::edgeData[edgeIdx].callType == 0)
				Host::edgeData[edgeIdx].callType = callType;
			Host::edgeData[edgeIdx].numTraversals++;
			Host::edgeData[edgeIdx].duration += duration;
#endif
		}

		__host__ __device__ __inline__ void writeSubGraphData(unsigned long long t0, unsigned long long t1, unsigned int sgIdx)
		{
			auto duration = t1 - t0;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			atomicAdd(&Device::subGraphData[sgIdx].numTraversals, 1);
			atomicAdd(&Device::subGraphData[sgIdx].duration, duration);
#else
			Device::subGraphData[sgIdx].numTraversals++;
			Device::subGraphData[sgIdx].duration += duration;
#endif
		}

	}

}

__global__ void initInstrumentationData(unsigned int numSubgraphs, unsigned int numEdges)
{
	for (unsigned int sgIdx = 0; sgIdx < numSubgraphs; ++sgIdx)
	{
		PGA::Instrumentation::Device::subGraphData[sgIdx].idx = sgIdx;
		PGA::Instrumentation::Device::subGraphData[sgIdx].callType = 0;
		PGA::Instrumentation::Device::subGraphData[sgIdx].duration = 0;
		PGA::Instrumentation::Device::subGraphData[sgIdx].numTraversals = 0;
	}

	for (unsigned int edgeIdx = 0; edgeIdx < numEdges; ++edgeIdx)
	{
		PGA::Instrumentation::Device::edgeData[edgeIdx].idx = edgeIdx;
		PGA::Instrumentation::Device::edgeData[edgeIdx].callType = 0;
		PGA::Instrumentation::Device::edgeData[edgeIdx].duration = 0;
		PGA::Instrumentation::Device::edgeData[edgeIdx].numTraversals = 0;
	}
}

namespace PGA
{
	namespace Instrumentation
	{
		template <template <template <class> class /*Queue*/, class /*ProcInfo*/, class /*ApplicationContext*/> class RealTechnique, template <class> class Queue, /*class INITFUNC, */class ProcedureList, class ApplicationContext>
		class NoSchedTechniqueWrapper
		{
		private:
			typedef RealTechnique<Queue, ProcedureList, ApplicationContext> TechniqueType;

			std::unique_ptr<TechniqueType, technique_deleter> realTechnique;
			NoSchedMothership& mothership;

		public:
			NoSchedTechniqueWrapper(NoSchedMothership& mothership) : realTechnique(new TechniqueType()), mothership(mothership)
			{
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
				return realTechnique->name() + " instrumented for edges and subgraphs";
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
				double elapsedTime = 0;
				std::stringstream cuptiOutput;
				Experiment experiment(events, std::extent<decltype(events)>::value);
				mothership.prepare();
				auto results = experiment.conduct([&]()
				{
					elapsedTime = realTechnique->execute(phase, timelimitInMs);
				});
				if (!results.empty())
				{
					cuptiOutput << "phase=" << std::to_string(phase) << "\n";

					for (int i = 0; i < std::extent<decltype(events)>::value; ++i)
					{
						auto v = results[i].extrapolate();
						cuptiOutput << results[i].event_name << ": " << v << '\n';
					}
				}
				mothership.computeTimings(cuptiOutput.str());
				return elapsedTime;
			}

			template <int Phase, int TimeLimitInKCycles>
			double execute()
			{
				double elapsedTime = 0;
				std::stringstream cuptiOutput;
				Experiment experiment(events, std::extent<decltype(events)>::value);
				mothership.prepare();
				auto results = experiment.conduct([&]()
				{
					elapsedTime = realTechnique->execute<Phase, TimeLimitInKCycles>();
				});
				if (!results.empty())
				{
					cuptiOutput << "phase=" << std::to_string(Phase) << "\n";
					for (int i = 0; i < std::extent<decltype(events)>::value; ++i)
					{
						auto v = results[i].extrapolate();
						cuptiOutput << results[i].event_name << ": " << v << '\n';
					}
				}
				mothership.computeTimings(cuptiOutput.str());
				return elapsedTime;
			}

			template <int Phase>
			double execute()
			{
				double elapsedTime = 0;
				std::stringstream cuptiOutput;
				Experiment experiment(events, std::extent<decltype(events)>::value);
				mothership.prepare();
				auto results = experiment.conduct([&]()
				{
					elapsedTime = realTechnique->execute<Phase>();
				});
				if (!results.empty())
				{
					cuptiOutput << "phase=" << std::to_string(Phase) << "\n";

					for (int i = 0; i < std::extent<decltype(events)>::value; ++i)
					{
						auto v = results[i].extrapolate();
						cuptiOutput << results[i].event_name << ": " << v << '\n';
					}
				}
				mothership.computeTimings(cuptiOutput.str());
				return elapsedTime;
			}

		};

        ////////////////////////////////////////////////////////////////////////////////////////////////////
		void NoSchedMothership::init(unsigned int numSubGraphs, unsigned int numEdges)
		{
			this->numSubGraphs = numSubGraphs;
			this->numEdges = numEdges;

			d_edgeData = cudaAllocArray<InstrumentationData>(numEdges);
			d_subGraphData = cudaAllocArray<InstrumentationData>(numSubGraphs);

			int device;
			PGA_CUDA_checkedCall(cudaGetDevice(&device));
			PGA_CUDA_checkedCall(cudaGetDeviceProperties(&prop, device));

			InstrumentationData* edgeDataPtr = d_edgeData.get();
			PGA_CUDA_checkedCall(cudaMemcpyToSymbol(Device::edgeData, &edgeDataPtr, sizeof(edgeDataPtr)));

			InstrumentationData* subgraphDataPtr = d_subGraphData.get();
			PGA_CUDA_checkedCall(cudaMemcpyToSymbol(Device::subGraphData, &subgraphDataPtr, sizeof(subgraphDataPtr)));

			for (unsigned int i = 0; i < NumBuffers; ++i)
				buffers[i] = std::unique_ptr<unsigned char[]>(new unsigned char[16 * Device::N]);
		}

		void NoSchedMothership::prepare()
		{
			initInstrumentationData<<<1, 1, 1>>>(numSubGraphs, numEdges);
			PGA_CUDA_checkedCall(cudaDeviceSynchronize());
		}

		void NoSchedMothership::computeTimings(const std::string& cuptiOutput)
		{
			auto edges = processors.find("edges");
			if (edges != processors.end())
			{
				std::uint32_t edgeDataBufferSize = sizeof(InstrumentationData) * numEdges;
				PGA_CUDA_checkedCall(cudaMemcpy(&buffers[0][0], d_edgeData.get(), edgeDataBufferSize, cudaMemcpyDeviceToHost));
				processEdgeData(reinterpret_cast<const unsigned char*>(&buffers[0][0]), numEdges, prop.clockRate);
			}

			auto subgraphs = processors.find("subgraphs");
			if (subgraphs != processors.end())
			{
				std::uint32_t subGraphDataBufferSize = sizeof(InstrumentationData) * numSubGraphs;
				PGA_CUDA_checkedCall(cudaMemcpy(&buffers[1][0], d_subGraphData.get(), subGraphDataBufferSize, cudaMemcpyDeviceToHost));
				processSubGraphData(reinterpret_cast<const unsigned char*>(&buffers[1][0]), numSubGraphs, prop.clockRate);
			}

			auto cupti = processors.find("cupti");
			if (cupti != processors.end())
				processCuptiData(cuptiOutput);
		}

	}

}
