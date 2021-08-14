#pragma once

#include "Instrumentation.h"

#include <fstream>
#include <iosfwd>
#include <string>

namespace PGA
{
	namespace Instrumentation
	{
		class SubGraphExporter : public DataProcessor
		{
		protected:
			std::ostream& file;
			std::string name;

			SubGraphExporter(const SubGraphExporter&) = delete;
			SubGraphExporter& operator =(const SubGraphExporter&) = delete;

		public:
			SubGraphExporter(std::ostream& file, const std::string& name)
				: file(file), name(name)
			{
			}

			virtual void attach(const char* deviceName, const Instrumentation::GPUInfo& gpuInfo)
			{
			}

			virtual void processData(const unsigned char* buffer, size_t bufferSize, int clockRate)
			{
				file << "clock_rate;" << clockRate << ";" << std::endl;
				for (auto i = 0; i < bufferSize * sizeof(InstrumentationData); i += sizeof(InstrumentationData))
				{
					const InstrumentationData* subGraphData = reinterpret_cast<const InstrumentationData*>(buffer + i);
					if (subGraphData->numTraversals > 0)
						file << subGraphData->idx << ";" << subGraphData->duration << ";" << subGraphData->numTraversals << ";" << std::endl;
					else
						file << subGraphData->idx << ";0;0;" << std::endl;
				}
			}

			virtual const std::string getName() const { return name; }

		};

	}

}
