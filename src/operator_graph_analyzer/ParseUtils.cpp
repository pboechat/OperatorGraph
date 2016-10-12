#include <cstdlib>
#include <vector>

#include <pga/core/StringUtils.h>

#include "HeuristicsTest.h"
#include "ParseUtils.h"

namespace ParseUtils
{
	long parseArg(const std::string& arg, long def)
	{
		if (arg == "0") return 0L;
		auto c = atol(arg.c_str());
		return c;
	}

	unsigned long parseArg(const std::string& arg, unsigned long def)
	{
		if (arg == "0") return 0uL;
		auto c = atol(arg.c_str());
		if (c < 0uL) return def;
		return static_cast<unsigned long>(c);
	}

	long long parseArg(const std::string& arg, long long def)
	{
		if (arg == "0") return 0L;
		auto c = atoll(arg.c_str());
		return c;
	}

	unsigned int parseArg(const std::string& arg, unsigned int def)
	{
		if (arg == "0") return 0L;
		auto c = atoi(arg.c_str());
		if (c < 0) return def;
		return c;
	}

	double parseArg(const std::string& arg, double def)
	{
		if (arg == "0") return 0L;
		auto c = atof(arg.c_str());
		return c;
	}

	bool parseHeuristicsTestSpecs(const std::string& value, 
		HeuristicsTestSpecifications& specs, 
		double defaultE, 
		ConfidenceLevel defaultConfidenceLevel,
		double defaultStdDeviation)
	{
		if (value == "")
			return false;
		std::vector<std::string> tokens;
		PGA::StringUtils::split(value, ',', tokens);
		if (tokens.size() != 3)
			return false;
		specs.errorMargin = parseArg(tokens[0], defaultE);
		specs.confidenceLevel = static_cast<ConfidenceLevel>(parseArg(tokens[1], static_cast<unsigned int>(defaultConfidenceLevel)));
		specs.standardDeviation = parseArg(tokens[2], defaultStdDeviation);
		return true;
	}

}
