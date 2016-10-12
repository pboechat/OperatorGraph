#pragma once

#include <fstream>
#include <iosfwd>
#include <string>

#include "Instrumentation.h"

namespace PGA
{
	namespace Instrumentation
	{
		class EdgeExporter : public DataProcessor
		{
		protected:
			std::ostream& file;
			std::string name;

			EdgeExporter(const EdgeExporter&) = delete;
			EdgeExporter& operator =(const EdgeExporter&) = delete;

		public:
			EdgeExporter(std::ostream& file, const std::string& name)
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
					const InstrumentationData* edgeData = reinterpret_cast<const InstrumentationData*>(buffer + i);
					if (edgeData->numTraversals > 0)
						file << edgeData->idx << ";" << edgeData->callType << ";" << edgeData->duration << ";" << edgeData->numTraversals << ";" << std::endl;
					else
						file << edgeData->idx << ";" << edgeData->callType << ";0;0;" << std::endl;
				}
			}

			virtual const std::string getName() const { return name; }

		};

	}

}
