#ifndef INCLUDED_INSTRUMENTATION_DATA_PROCESSOR_H
#define INCLUDED_INSTRUMENTATION_DATA_PROCESSOR_H

#pragma once

#include "interface.h"
#include "instrumentation.h"

#include <cstdint>
#include <vector>

namespace Instrumentation
{
	unsigned long long unpack_t0(const unsigned char* data);
	unsigned long long unpack_t1(const unsigned char* data);
	unsigned int unpack_pid(const unsigned char* data);
	unsigned int unpack_mpid(const unsigned char* data);
	unsigned int unpack_active_threads(const unsigned char* data);
	unsigned int unpack_overhead_counter(const unsigned char* data);
	unsigned int unpack_dequeue_time(const unsigned char* data);
	unsigned int unpack_num_enqueue_stats(const unsigned char* data);
	unsigned int unpack_enqueue_stat_pid(const unsigned char* data, int i);
	unsigned int unpack_enqueue_stat_count(const unsigned char* data, int i);
	const unsigned char* next_element(const unsigned char* data, int num_enqueue_stats);

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

	class INTERFACE DataProcessor
	{
	protected:
		DataProcessor() {}
		DataProcessor(const DataProcessor&) {}
		DataProcessor& operator =(const DataProcessor&) { return *this; }
		~DataProcessor() {}

	public:
		virtual void attach(const char* device_name, const GPUInfo& gpu_info) = 0;
		virtual void processData(const unsigned char* buffer, size_t bufferSize, float baseTime, const std::vector<ProcedureDescription>& proceduresDescriptions) = 0;

	};
}

#endif  // INCLUDED_INSTRUMENTATION_DATA_PROCESSOR_H
