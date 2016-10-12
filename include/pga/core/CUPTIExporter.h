#pragma once

#include <fstream>
#include <iosfwd>
#include <string>

#include "Instrumentation.h"

namespace PGA
{
	namespace Instrumentation
	{
		class CUPTIExporter : public DataProcessor
		{
		protected:
			std::ostream& file;
			std::string name;

			CUPTIExporter(const CUPTIExporter&) = delete;
			CUPTIExporter& operator =(const CUPTIExporter&) = delete;

		public:
			CUPTIExporter(std::ostream& file, std::string name)
				: file(file), name(name)
			{
			}

			virtual void attach(const char* device_name, const Instrumentation::GPUInfo& gpuInfo)
			{
				file << "name;cc_major;cc_minor;num_multiprocessors;warp_size;max_threads_per_mp;max_threads_per_block;regs_per_block;shared_memory;constant_memory;global_memory;clock_rate\n"
					<< device_name << ';'
					<< gpuInfo.cc_major << ';'
					<< gpuInfo.cc_minor << ';'
					<< gpuInfo.multiprocessor_count << ';'
					<< gpuInfo.warp_size << ';'
					<< gpuInfo.max_thread_per_mp << ';'
					<< gpuInfo.max_threads_per_block << ';'
					<< gpuInfo.max_regs_per_block << ';'
					<< gpuInfo.shared_memory_per_block << ';'
					<< gpuInfo.total_constant_memory << ';'
					<< gpuInfo.total_global_memory << ';'
					<< gpuInfo.clock_rate << '\n'
					<< std::endl;
			}

			virtual void processData(const unsigned char* buffer, size_t bufferSize, int clockRate)
			{
				file << buffer;
			}

			virtual const std::string getName() const { return name; }

		};

	}

}
