#pragma once

#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

#include <cuda_memory.h>
#include <techniqueInterface.h>

#include "CUDAException.h"

namespace PGA
{
	namespace Instrumentation
	{
		struct GPUInfo
		{
			int cc_major;
			int cc_minor;
			int multiprocessor_count;
			int warp_size;
			int max_thread_per_mp;
			int max_threads_per_block;
			int max_regs_per_block;
			size_t shared_memory_per_block;
			size_t total_constant_memory;
			size_t total_global_memory;
			unsigned long long clock_rate;

		};

		class DataProcessor
		{
		public:
			virtual void attach(const char* deviceName, const GPUInfo& gpuInfo) = 0;
			virtual void processData(const unsigned char* buffer, size_t bufferSize, int clockRate) = 0;
			virtual const std::string getName() const = 0;

		};

		struct InstrumentationData
		{
			unsigned int idx;
			unsigned int callType;
			unsigned int numTraversals;
			unsigned long long duration;

		};

		class NoSchedMothership
		{
		private:
			static const unsigned int NumBuffers = 2;

			unsigned int numEdges;
			unsigned int numSubGraphs;

			std::unique_ptr<InstrumentationData[], cuda_deleter> d_edgeData;
			std::unique_ptr<InstrumentationData[], cuda_deleter> d_subGraphData;

			cudaDeviceProp prop;
			std::map<std::string, DataProcessor*> processors;
			std::unique_ptr<unsigned char[]> buffers[NumBuffers];

			NoSchedMothership(const NoSchedMothership&) = delete;
			NoSchedMothership& operator =(const NoSchedMothership&) = delete;

			void processEdgeData(const unsigned char* buffer, size_t numEntries, int clockRate)
			{
				processors.at("edges")->processData(reinterpret_cast<const unsigned char*>(&buffer[0]), numEntries, clockRate);
			}

			void processCuptiData(const std::string& data)
			{
				processors.at("cupti")->processData(reinterpret_cast<const unsigned char*>(data.c_str()), 0, 0);
			}

			void processSubGraphData(const unsigned char* buffer, size_t numEntries, int clockRate)
			{
				processors.at("subgraphs")->processData(reinterpret_cast<const unsigned char*>(&buffer[0]), numEntries, clockRate);
			}

		public:
            NoSchedMothership() = default;
			void init(unsigned int num_procedures, unsigned int num_edges);
            void prepare();
			void computeTimings(const std::string& cuptiOutput);
			void attach(DataProcessor* processor)
			{
				processors[processor->getName()] = processor;
				if (processor)
				{
					GPUInfo info;
					info.cc_major = prop.major;
					info.cc_minor = prop.minor;
					info.multiprocessor_count = prop.multiProcessorCount;
					info.warp_size = prop.warpSize;
					info.max_thread_per_mp = prop.maxThreadsPerMultiProcessor;
					info.max_threads_per_block = prop.maxThreadsPerBlock;
					info.max_regs_per_block = prop.regsPerBlock;
					info.shared_memory_per_block = prop.sharedMemPerBlock;
					info.total_constant_memory = prop.totalConstMem;
					info.total_global_memory = prop.totalGlobalMem;
					info.clock_rate = prop.clockRate * 1000ULL;
					processor->attach(prop.name, info);
				}
			}

		};

	}

}
