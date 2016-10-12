#include <sstream>
#include <fstream>
#include <windows.h>

#include <pga/core/StringUtils.h>
#include <pga/compiler/CodeGenerator.h>

#include "PartitionOutputter.h"

//////////////////////////////////////////////////////////////////////////
PartitionOutputter::PartitionOutputter(
	unsigned int optimization, 
	bool instrumented, 
	long minMatches, 
	long maxProcs, 
	long long queuesMem, 
	long long itemSize, 
	const std::string& templateCode) : 
	optimization(optimization),
	instrumented(instrumented),
	minMatches(minMatches),
	maxProcs(maxProcs),
	queuesMem(queuesMem),
	itemSize(itemSize),
	templateCode(templateCode)
{
	suffix = ("_o" + std::to_string(optimization)) + ((instrumented) ? "_i" : "");
}

bool PartitionOutputter::operator()(size_t i, PGA::Compiler::Graph::PartitionPtr& partition)
{
	if (maxProcs != -1 && maxProcs < partition->matchGroups->size())
		return false;

	if (minMatches != -1 && minMatches > partition->numMatches())
		return false;

	processPartition(i, partition);

	return true;
}

//////////////////////////////////////////////////////////////////////////
FilesPartitionOutputter::FilesPartitionOutputter(
	const std::string& outputDir, 
	unsigned int optimization, 
	bool instrumented, 
	long minMatches, 
	long maxProcs, 
	long long queuesMem, 
	long long itemSize, 
	const std::string& templateCode) : 
	PartitionOutputter(optimization, instrumented, minMatches, maxProcs, queuesMem, itemSize, templateCode), 
	outputDir(outputDir)
{
}

void FilesPartitionOutputter::processPartition(size_t i, PGA::Compiler::Graph::PartitionPtr& partition)
{
	static bool firstTime = true;
	static bool failSilently = false;

	if (failSilently)
		return;

	if (firstTime)
	{
		std::string dataDir(outputDir);
		if (!CreateDirectory(outputDir.c_str(), 0) && GetLastError() != ERROR_ALREADY_EXISTS)
		{
			std::cerr << "error creating directory: " << outputDir << std::endl;
			failSilently = true;
		}
		firstTime = false;
	}

	auto uid = partition->getUid();
	auto idxStr = std::to_string(i);

	std::string baseFileName = outputDir + "/partition_" + idxStr + suffix;

	ConnectionVisitor connVisitor;
	partition->traverse(connVisitor);
	std::ofstream connFile;
	connFile.open(baseFileName + ".conn");
	connFile << connVisitor;
	connFile.close();

	DotGraphVisitor dotVisitor;
	partition->traverse(dotVisitor);
	std::ofstream dotFile;
	dotFile.open(baseFileName + ".dot");
	dotFile << dotVisitor;
	dotFile.close();

	bool staticFirstProcedure;
	std::stringstream out;
	PGA::Compiler::CodeGenerator::fromPartition(out, partition, true, instrumented, staticFirstProcedure);

	std::string content(templateCode);
	PGA::StringUtils::replaceAll(content, "<code>", out.str());
	PGA::StringUtils::replaceAll(content, "<uid>", uid);
	PGA::StringUtils::replaceAll(content, "<idx>", idxStr);
	PGA::StringUtils::replaceAll(content, "<operatorCode>", (staticFirstProcedure) ? "0" : "-1");
	PGA::StringUtils::replaceAll(content, "<entryIdx>", (staticFirstProcedure) ? "-1" : "0");
	PGA::StringUtils::replaceAll(content, "<numSubgraphs>", std::to_string(partition->subGraphs.size()));
	auto queueMem = queuesMem / static_cast<long long>(partition->matchGroups->size());
	long long queueSize;
	if (itemSize > 0)
		queueSize = queueMem / itemSize;
	else
		queueSize = 0;
	// NOTE: vc++ doesn't support arrays bigger than 2GB!!!
	if (queueSize > 2147483647LL)
		queueSize = 2147483647LL;
	PGA::StringUtils::replaceAll(content, "<queueSize>", std::to_string(queueSize));

	std::ofstream srcFile;
	srcFile.open(baseFileName + ".cuh");
	srcFile << content;
	srcFile.close();
}

//////////////////////////////////////////////////////////////////////////
StreamsPartitionOutputter::StreamsPartitionOutputter(
	std::ostream& connStream, 
	std::ostream& dotStream, 
	std::ostream& srcStream, 
	unsigned int optimization, 
	bool instrumented, 
	long minMatches, 
	long maxProcs, 
	long long queuesMem, 
	long long itemSize, 
	const std::string& templateCode) : 
	PartitionOutputter(optimization, instrumented, minMatches, maxProcs, queuesMem, itemSize, templateCode), 
	connStream(connStream), 
	dotStream(dotStream), 
	srcStream(srcStream)
{
}

void StreamsPartitionOutputter::processPartition(size_t i, PGA::Compiler::Graph::PartitionPtr& partition)
{
	auto uid = partition->getUid();

	ConnectionVisitor connVisitor;
	partition->traverse(connVisitor);
	connStream << connVisitor;

	DotGraphVisitor dotVisitor;
	partition->traverse(dotVisitor);
	dotStream << dotVisitor;

	bool staticFirstProcedure;
	std::stringstream out;
	PGA::Compiler::CodeGenerator::fromPartition(out, partition, true, instrumented, staticFirstProcedure);

	std::string content(templateCode);
	PGA::StringUtils::replaceAll(content, "<code>", out.str());
	PGA::StringUtils::replaceAll(content, "<uid>", uid);
	PGA::StringUtils::replaceAll(content, "<idx>", std::to_string(i));
	PGA::StringUtils::replaceAll(content, "<operatorCode>", (staticFirstProcedure) ? "0" : "-1");
	PGA::StringUtils::replaceAll(content, "<entryIdx>", (staticFirstProcedure) ? "-1" : "0");
	PGA::StringUtils::replaceAll(content, "<numSubgraphs>", std::to_string(partition->subGraphs.size()));
	auto avgQueueSize = queuesMem / static_cast<long long>(partition->matchGroups->size());
	auto avgQueueMem = (avgQueueSize * itemSize);
	// NOTE: vc++ doesn't support arrays bigger than 2GB!!!
	if (avgQueueMem > 2147483648LL)
		avgQueueSize = 2147483648LL / itemSize;
	PGA::StringUtils::replaceAll(content, "<avgQueueSize>", std::to_string(avgQueueSize));

	srcStream << content;
}
