#pragma once

#include <string>
#include <memory>
#include <fstream>
#include <map>

#include <pga/core/CSVExporter.h>
#include <pga/core/CuptiExporter.h>
#include <pga/core/EdgeExporter.h>
#include <pga/core/SubGraphExporter.h>

class LogManager
{
private:
	std::unique_ptr<PGA::Instrumentation::CSVExporter> schedulingExporter;
	std::unique_ptr<PGA::Instrumentation::EdgeExporter> edgeExporter;
	std::unique_ptr<PGA::Instrumentation::CUPTIExporter> cuptiExporter;
	std::unique_ptr<PGA::Instrumentation::SubGraphExporter> subGraphExporter;
	std::map<std::string, std::ofstream> loggers;
	std::string baseDir;
	// NOTE: unsafe!!!
	PGA::Instrumentation::NoSchedMothership* mothership;

	LogManager(const LogManager&) = delete;
	LogManager& operator =(const LogManager&) = delete;

public:
	LogManager() : mothership(nullptr), schedulingExporter(nullptr), edgeExporter(nullptr), cuptiExporter(nullptr)
	{
	}

	~LogManager()
	{
		for (auto& entry : loggers)
		{
			entry.second.flush();
			entry.second.close();
		}
	}

	void initializeForSchedulingInstrumentation(const std::string& sinkName, const std::string& fileName)
	{
		std::ofstream& instrumentationSink = addLogger(sinkName, fileName);
		schedulingExporter = std::unique_ptr<PGA::Instrumentation::CSVExporter>(new PGA::Instrumentation::CSVExporter(instrumentationSink, sinkName));
		instrumentationSink << fileName << std::endl << std::endl;
	}

	void initializeForEdgeInstrumentation(const std::string& sinkName, const std::string& fileName)
	{
		std::ofstream& instrumentationSink = addLogger(sinkName, fileName);
		edgeExporter = std::unique_ptr<PGA::Instrumentation::EdgeExporter>(new PGA::Instrumentation::EdgeExporter(instrumentationSink, sinkName));
		instrumentationSink << fileName << std::endl << std::endl;
		if (mothership == nullptr) return;
		// TODO:
		mothership->attach(edgeExporter.get());
	}

	void initializeForCuptiInstrumentation(const std::string& sinkName, const std::string& fileName)
	{
		std::ofstream& instrumentationSink = addLogger(sinkName, fileName);
		cuptiExporter = std::unique_ptr<PGA::Instrumentation::CUPTIExporter>(new PGA::Instrumentation::CUPTIExporter(instrumentationSink, sinkName));
		instrumentationSink << fileName << std::endl << std::endl;
		if (mothership == nullptr) return;
		// TODO:
		mothership->attach(cuptiExporter.get());
	}

	void initializeForSubgraphInstrumentation(const std::string& sinkName, const std::string& fileName)
	{
		std::ofstream& instrumentationSink = addLogger(sinkName, fileName);
		subGraphExporter = std::unique_ptr<PGA::Instrumentation::SubGraphExporter>(new PGA::Instrumentation::SubGraphExporter(instrumentationSink, sinkName));
		instrumentationSink << fileName << std::endl << std::endl;
		if (mothership == nullptr) return;
		// TODO:
		mothership->attach(subGraphExporter.get());
	}

	void finalizeForInstrumentation()
	{
		schedulingExporter = nullptr;
		edgeExporter = nullptr;
		cuptiExporter = nullptr;
		subGraphExporter = nullptr;
	}

	void attach(PGA::Instrumentation::NoSchedMothership* mothership)
	{
		this->mothership = mothership;
		attachExporters();
	}

	inline std::ofstream& addLogger(const std::string& name, const std::string& fileName)
	{
		std::ofstream& out = loggers[name];
		out.flush();
		out.close();
		out.open((baseDir + fileName).c_str());
		return out;
	}

	template <typename T>
	void write(const std::string& loggerName, T value)
	{
		loggers[loggerName] << value << std::endl;
	}

	void setBaseDir(const std::string& dir)
	{
		baseDir = dir;
	}

	void attachExporters()
	{
		if (mothership == nullptr) return;
		// FIXME: passing plain pointer forward (unsafe)
		if (edgeExporter != nullptr)
			mothership->attach(edgeExporter.get());
		if (cuptiExporter != nullptr)
			mothership->attach(cuptiExporter.get());
		if (subGraphExporter != nullptr)
			mothership->attach(subGraphExporter.get());
	}



};
