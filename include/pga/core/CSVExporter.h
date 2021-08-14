#pragma once

#include "Instrumentation.h"
#include "InstrumentationPackUtils.h"

#include <fstream>
#include <iosfwd>
#include <string>

namespace PGA
{
	namespace Instrumentation
	{
		class CSVExporter : public DataProcessor
		{
		protected:
			std::ostream& file;
			std::string name;

			CSVExporter(const CSVExporter&) = delete;
			CSVExporter& operator =(const CSVExporter&) = delete;

		public:
			CSVExporter(std::ostream& file, std::string name)
				: file(file), name(name)
			{
			}

			virtual void attach(const char* deviceName, const Instrumentation::GPUInfo& gpuInfo)
			{
				file << "name;cc_major;cc_minor;num_multiprocessors;warp_size;max_threads_per_mp;max_threads_per_block;regs_per_block;shared_memory;constant_memory;global_memory;clock_rate\n"
					<< deviceName << ';'
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
				file << "t0;t1;pid;mpid;active_threads;overhead_counter;dequeue_time;{enqueue_stats_pid;enqueue_stats_count}" << std::endl;
				for (const unsigned char* data = buffer; data < buffer + bufferSize;)
				{
					int num_enqueue_stats = PGA::Instrumentation::PackUtils::unpack_num_enqueue_stats(data);
					file << PGA::Instrumentation::PackUtils::unpack_t0(data) << ';'
						<< PGA::Instrumentation::PackUtils::unpack_t1(data) << ';'
						<< PGA::Instrumentation::PackUtils::unpack_pid(data) << ';'
						<< PGA::Instrumentation::PackUtils::unpack_mpid(data) << ';'
						<< PGA::Instrumentation::PackUtils::unpack_active_threads(data) << ';'
						<< PGA::Instrumentation::PackUtils::unpack_overhead_counter(data) << ';'
						<< PGA::Instrumentation::PackUtils::unpack_dequeue_time(data);
					if (num_enqueue_stats > 0)
					{
						for (int i = 0; i < num_enqueue_stats; ++i)
							file << ';' << PGA::Instrumentation::PackUtils::unpack_enqueue_stat_pid(data, i) << ';' << PGA::Instrumentation::PackUtils::unpack_enqueue_stat_count(data, i);
					}
					file << std::endl;
					data = PGA::Instrumentation::PackUtils::next_element(data, num_enqueue_stats);
				}
				file << std::endl;
			}

			virtual const std::string getName() const { return name; }

		};

	}

}
