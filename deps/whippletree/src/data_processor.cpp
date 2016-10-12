


#include <iostream>

#include "..\include\data_processor.h"
#include "..\include\CSVExporter.h"
#include "..\include\procedureDescription.h"


namespace
{
	template <unsigned int start, unsigned int length>
	inline std::uint64_t unpackField(std::uint64_t v)
	{
		return (v >> start) & ~(~0ULL << length);
	}

	template <unsigned int start, unsigned int length>
	inline std::uint32_t unpackField(std::uint32_t v)
	{
		return (v >> start) & ~(~0UL << length);
	}

	template <int i>
	const std::uint64_t& get_element(const void* data)
	{
		return *(static_cast<const std::uint64_t*>(data) + i);
	}
}

namespace Instrumentation
{
	unsigned long long unpack_t0(const unsigned char* data)
	{
		return get_element<0>(data);
	}

	unsigned long long unpack_t1(const unsigned char* data)
	{
		return get_element<1>(data);
	}

	unsigned int unpack_pid(const unsigned char* data)
	{
		return static_cast<int>(unpackField<53, 11>(get_element<2>(data)));
	}

	unsigned int unpack_mpid(const unsigned char* data)
	{
		return static_cast<int>(unpackField<44, 8>(get_element<2>(data)));
	}

	unsigned int unpack_active_threads(const unsigned char* data)
	{
		return static_cast<int>(unpackField<32, 12>(get_element<2>(data)));
	}

	unsigned int unpack_overhead_counter(const unsigned char* data)
	{
		return static_cast<int>(unpackField<0, 32>(get_element<2>(data)));
	}

	unsigned int unpack_dequeue_time(const unsigned char* data)
	{
		return static_cast<int>(unpackField<32, 32>(get_element<3>(data)));
	}

	unsigned int unpack_num_enqueue_stats(const unsigned char* data)
	{
		return static_cast<int>(unpackField<0, 8>(get_element<3>(data)));
	}

	unsigned int unpack_enqueue_stat_pid(const unsigned char* data, int i)
	{
		return unpackField<21, 11>(*(reinterpret_cast<const uint32_t*>(data) + 8 + i));
	}

	unsigned int unpack_enqueue_stat_count(const unsigned char* data, int i)
	{
		return unpackField<0, 21>(*(reinterpret_cast<const uint32_t*>(data) + 8 + i));
	}

	const unsigned char* next_element(const unsigned char* data, int num_enqueue_stats)
	{
		return data + 32 + (num_enqueue_stats + 3) / 4 * 16;
	}


	CSVExporter::CSVExporter(std::ostream& file)
		: file(file)
	{
	}

	void CSVExporter::attach(const char* device_name, const Instrumentation::GPUInfo& gpu_info)
	{
		file << "name;cc_major;cc_minor;num_multiprocessors;warp_size;max_threads_per_mp;max_threads_per_block;regs_per_block;shared_memory;constant_memory;global_memory;clock_rate\n"
			<< device_name << ';'
			<< gpu_info.cc_major << ';'
			<< gpu_info.cc_minor << ';'
			<< gpu_info.multiprocessor_count << ';'
			<< gpu_info.warp_size << ';'
			<< gpu_info.max_thread_per_mp << ';'
			<< gpu_info.max_threads_per_block << ';'
			<< gpu_info.max_regs_per_block << ';'
			<< gpu_info.shared_memory_per_block << ';'
			<< gpu_info.total_constant_memory << ';'
			<< gpu_info.total_global_memory << ';'
			<< gpu_info.clock_rate << '\n'
			<< std::endl;
	}

	void CSVExporter::processData(const unsigned char* buffer, size_t bufferSize, float baseTime, const std::vector<ProcedureDescription>& proceduresDescriptions)
	{
		file << "name;pid;num_threads;item_input;input_size;shared_memory" << std::endl;
		for (auto p = proceduresDescriptions.rbegin(); p != proceduresDescriptions.rend(); ++p)
		{
			file << p->name << ';' << p->ProcedureId << ';' << p->NumThreads << ';' << p->ItemInput << ';' << p->inputSize << ';' << p->sharedMemory << '\n';
		}
		file << std::endl;

		file << "t0;t1;pid;mpid;active_threads;overhead_counter;dequeue_time;{enqueue_stats_pid;enqueue_stats_count}" << std::endl;

		for (const unsigned char* data = buffer; data < buffer + bufferSize;)
		{
			int num_enqueue_stats = Instrumentation::unpack_num_enqueue_stats(data);

			file << Instrumentation::unpack_t0(data) << ';'
				<< Instrumentation::unpack_t1(data) << ';'
				<< Instrumentation::unpack_pid(data) << ';'
				<< Instrumentation::unpack_mpid(data) << ';'
				<< Instrumentation::unpack_active_threads(data) << ';'
				<< Instrumentation::unpack_overhead_counter(data) << ';'
				<< Instrumentation::unpack_dequeue_time(data);

			if (num_enqueue_stats > 0)
			{
				for (int i = 0; i < num_enqueue_stats; ++i)
					file << ';'<< Instrumentation::unpack_enqueue_stat_pid(data, i) << ';'<< Instrumentation::unpack_enqueue_stat_count(data, i);
			}
			file << std::endl;

			data = Instrumentation::next_element(data, num_enqueue_stats);
		}

		file << std::endl;
	}
}
