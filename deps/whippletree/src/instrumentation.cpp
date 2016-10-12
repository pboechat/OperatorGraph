#include "..\include\instrumentation.h"

#include "..\include\data_processor.h"

namespace Instrumentation
{
	void Mothership::attach(DataProcessor* processor)
	{
		Mothership::processor = processor;

		if(processor)
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

	void Mothership::processData(const unsigned char* buffer, size_t buffer_size, float base_time, const std::vector<ProcedureDescription>& proc_desc)
	{
		processor->processData(reinterpret_cast<const unsigned char*>(&buffer[0]), buffer_size, base_time, proc_desc);
	}

}
