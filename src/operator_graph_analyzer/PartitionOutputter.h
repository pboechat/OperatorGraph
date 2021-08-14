#pragma once

#include "ConnectionVisitor.h"
#include "DotGraphVisitor.h"

#include <pga/compiler/Graph.h>
#include <pga/core/GPUTechnique.h>

#include <ostream>
#include <string>

struct PartitionOutputter : PGA::Compiler::Graph::ComputePartitionCallback
{
	PartitionOutputter(unsigned int optimization,
		bool instrumented, 
		long minMatches, 
		long maxProcs, 
		long long queuesMem, 
		long long itemSize, 
		const std::string& templateCode);

	virtual bool operator()(size_t i, PGA::Compiler::Graph::PartitionPtr& partition);
	virtual void processPartition(size_t i, PGA::Compiler::Graph::PartitionPtr& partition) = 0;

protected:
	std::string suffix;
	unsigned int optimization;
	bool instrumented;
	long minMatches;
	long maxProcs;
	long long queuesMem;
	long long itemSize;
	std::string templateCode;

};

struct FilesPartitionOutputter : PartitionOutputter
{
	FilesPartitionOutputter(const std::string& outputDir,
		unsigned int optimization,
		bool instrumented,
		long minMatches,
		long maxProcs,
		long long queuesMem,
		long long itemSize,
		const std::string& templateCode);

	virtual void processPartition(size_t i, PGA::Compiler::Graph::PartitionPtr& partition);

private:
	std::string outputDir;

};

struct StreamsPartitionOutputter : PartitionOutputter
{
	StreamsPartitionOutputter(std::ostream& connStream,
		std::ostream& dotStream,
		std::ostream& srcStream,
		unsigned int optimization,
		bool instrumented,
		long minMatches,
		long maxProcs,
		long long queuesMem,
		long long itemSize,
		const std::string& templateCode);

	virtual void processPartition(size_t i, PGA::Compiler::Graph::PartitionPtr& partition);

private:
	std::ostream& connStream;
	std::ostream& dotStream;
	std::ostream& srcStream;

};