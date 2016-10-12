#pragma once

#include <string>

namespace ParseUtils
{
	long parseArg(const std::string& arg, long def);
	unsigned long parseArg(const std::string& arg, unsigned long def);
	long long parseArg(const std::string& arg, long long def);
	unsigned int parseArg(const std::string& arg, unsigned int def);
	double parseArg(const std::string& arg, double def);
	bool parseHeuristicsTestSpecs(const std::string& value,
		HeuristicsTestSpecifications& specs,
		double defaultE = 4.0,
		ConfidenceLevel defaultConfidenceLevel = ConfidenceLevel::CL_95,
		double defaultStdDeviation = 0.5);

}
