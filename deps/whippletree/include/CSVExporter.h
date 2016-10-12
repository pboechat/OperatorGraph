#ifndef INCLUDED_INSTRUMENTATION_CSV_EXPORTER_H
#define INCLUDED_INSTRUMENTATION_CSV_EXPORTER_H

#pragma once

#include <iosfwd>

#include "data_processor.h"

namespace Instrumentation
{
	class CSVExporter : public DataProcessor
	{
	protected:
		CSVExporter(const CSVExporter&);
		CSVExporter& operator =(const CSVExporter&);

		std::ostream& file;

	public:
		CSVExporter(std::ostream& file);

		virtual void attach(const char* device_name, const Instrumentation::GPUInfo& gpu_info);
		virtual void processData(const unsigned char* buffer, size_t bufferSize, float baseTime, const std::vector<ProcedureDescription>& proceduresDescriptions);

	};
}

#endif  // INCLUDED_INSTRUMENTATION_CSV_EXPORTER_H
