#ifndef INCLUDED_INSTRUMENTATION_H
#define INCLUDED_INSTRUMENTATION_H

#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include "utils.h"
#include "cuda_memory.h"

#include "interface.h"
//#include "procedureInterface.cuh"
#include "techniqueInterface.h"
#include "procedureDescription.h"

//class ProcedureDescription;

namespace Instrumentation
{
	class DataProcessor;

	class Mothership
	{
	private:
		static const unsigned int numBuffers = 1;

		cudaEvent_t t0;
		cudaEvent_t t1;
		std::unique_ptr<ulonglong2[], cuda_deleter> d_buffer;
		std::unique_ptr<std::int64_t[], cuda_deleter> d_clock_calibration_offsets;
		cudaDeviceProp prop;
		DataProcessor* processor;
		std::unique_ptr<unsigned char[]> buffers[numBuffers];

		Mothership(const Mothership&);
		Mothership& operator =(const Mothership&);

		void calibrateClockCounters();
		void processData(const unsigned char* buffer, size_t buffer_size, float base_time, const std::vector<ProcedureDescription>& proc_desc);

	public:
		void attach(DataProcessor* processor);
		void prepare();
		void computeTimings(const std::vector<ProcedureDescription>& proc_desc);
		void changeSceneName(const char* scene_name);

		Mothership();

	};
}

#endif  // INCLUDED_INSTRUMENTATION_H
