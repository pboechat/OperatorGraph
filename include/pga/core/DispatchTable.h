#pragma once

#include "DispatchTableEntry.h"
#include "ParameterType.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <memory>
#include <vector>
#include <stdexcept>

namespace PGA
{
	struct DispatchTable
	{
		struct Parameter
		{
			ParameterType type;
			std::vector<double> values;

			Parameter() : type(ParameterType::PT_SCALAR) {}
			Parameter(ParameterType type, const std::vector<double>& values) : type(type), values(values.begin(), values.end()) {}
			Parameter(ParameterType type, const std::initializer_list<double>& values) : type(type), values(values.begin(), values.end()) {}

			void fill(DispatchTableEntry::Parameter& parameter) const
			{
				parameter.type = type;
				if (values.size() > Constants::MaxNumParameterValues)
					throw std::runtime_error("values.size() > PGA::Constants::MaxNumParameterValues");
				memset(parameter.value, 0, sizeof(float) * Constants::MaxNumParameterValues);
				for (unsigned int i = 0; i < values.size(); i++)
					parameter.value[i] = static_cast<float>(values[i]);
			}

		};

		struct Successor
		{
			int entryIndex;
			int phaseIndex;

			Successor() : entryIndex(-1), phaseIndex(-1) {}
			Successor(int entryIndex, int phaseIndex) : entryIndex(entryIndex), phaseIndex(phaseIndex) {}

			void fill(DispatchTableEntry::Successor& successor) const
			{
				successor.entryIndex = entryIndex;
				successor.phaseIndex = phaseIndex;
			}

		};

		struct Entry
		{
			size_t operatorCode;
			std::vector<Parameter> parameters;
			std::vector<Successor> successors;
			std::vector<int> edgeIndices;
			int subGraphIndex;

			Entry() : operatorCode(0) {}
			Entry(size_t operatorCode, const std::initializer_list<Parameter>& parameters, const std::initializer_list<Successor>& successors, const std::initializer_list<int>& edgeIndices = {}, int subGraphIndex = -1)
				: operatorCode(operatorCode), parameters(parameters.begin(), parameters.end()), successors(successors.begin(), successors.end()), edgeIndices(edgeIndices.begin(), edgeIndices.end()), subGraphIndex(subGraphIndex) {}

			void fill(DispatchTableEntry& entry) const
			{
				entry.operatorCode = static_cast<unsigned int>(operatorCode);
				if (parameters.size() > Constants::MaxNumParameters)
					throw std::runtime_error("parameters.size() > PGA::Constants::MaxNumParameters");
				entry.numParameters = static_cast<unsigned int>(parameters.size());
				memset(entry.parameters, 0, sizeof(DispatchTableEntry::Parameter) * Constants::MaxNumParameters);
				for (size_t i = 0; i < parameters.size(); i++)
					parameters[i].fill(entry.parameters[i]);
				if (successors.size() > Constants::MaxNumSuccessors)
					throw std::runtime_error("successors.size() > PGA::Constants::MaxNumSuccessors");
				entry.numSuccessors = static_cast<unsigned int>(successors.size());
				memset(entry.successors, -1, sizeof(DispatchTableEntry::Successor) * Constants::MaxNumSuccessors);
				for (size_t i = 0; i < successors.size(); i++)
					successors[i].fill(entry.successors[i]);
				if (edgeIndices.size() > Constants::MaxNumSuccessors)
					throw std::runtime_error("edgeIndices.size() > PGA::Constants::MaxNumSuccessors");
				entry.numEdgeIndices = static_cast<unsigned int>(edgeIndices.size());
				memset(entry.edgeIndices, -1, sizeof(int) * Constants::MaxNumSuccessors);
				for (size_t i = 0; i < edgeIndices.size(); i++)
					entry.edgeIndices[i] = edgeIndices[i];
				entry.subGraphIndex = subGraphIndex;
			}

		};

		std::vector<Entry> entries;

		DispatchTable() = default;
		DispatchTable(const std::initializer_list<Entry>& entries) : entries(entries.begin(), entries.end()) {}

		std::unique_ptr<DispatchTableEntry[]> toDispatchTableEntriesPtr()
		{
			std::unique_ptr<DispatchTableEntry[]> ptr(new DispatchTableEntry[entries.size()]);
			for (size_t i = 0; i < entries.size(); i++)
				entries[i].fill(ptr[i]);
			return ptr;
		}

	};

}