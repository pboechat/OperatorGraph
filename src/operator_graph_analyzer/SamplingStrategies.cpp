#include "CommonGraphOperations.h"
#include "HeuristicsTest.h"
#include "SamplingStrategies.h"

#include <iostream>

SamplingStrategy::SamplingStrategy(const std::set<size_t>& alwaysCutEdges,
	const std::set<size_t>& neverCutEdges, 
	PGA::Compiler::Graph& graph) : 
		alwaysCutEdges(alwaysCutEdges),
		neverCutEdges(neverCutEdges), 
		graph(graph)
{
}

void SamplingStrategy::computeSamples(long seed, const std::set<size_t>& graphAlwaysCutEdges, const std::set<size_t>& nonFixedEdges)
{
	if (seed < 0)
		generator.seed(static_cast<unsigned long>(std::chrono::system_clock::now().time_since_epoch().count()));
	else
		generator.seed(seed);

	std::map<size_t, std::set<size_t>> parents;
	graph.getParents(parents);
	std::map<size_t, float> edgeCutChances;
	for (auto& entry : parents)
		edgeCutChances.emplace(entry.first, 1.0f - 0.5f / static_cast<float>(entry.second.size()));

	size_t currSample = 0;
	while (currSample < samplingSize)
	{
		if (sample(graphAlwaysCutEdges,
			nonFixedEdges,
			edgeCutChances))
			currSample++;
	}
}

RandomSampling::RandomSampling(const std::set<size_t>& alwaysCutEdges, 
	const std::set<size_t>& neverCutEdges, 
	PGA::Compiler::Graph& graph, 
	bool matchGroups,
	PGA::Compiler::Graph::ComputePartitionCallback& callback,
	size_t samplingSize) : 
		SamplingStrategy(alwaysCutEdges, neverCutEdges, graph),
		matchGroups(matchGroups),
		callback(callback)
{
	this->samplingSize = samplingSize;
}

void RandomSampling::computeSize(size_t numNonFixedEdges)
{
	// do nothing
}

bool RandomSampling::sample(const std::set<size_t>& graphAlwaysCutEdges, 
	const std::set<size_t>& nonFixedEdges, 
	std::map<size_t, float>& edgeCutChances)
{
	std::set<size_t> cutEdges(alwaysCutEdges.begin(), alwaysCutEdges.end());
	cutEdges.insert(graphAlwaysCutEdges.begin(), graphAlwaysCutEdges.end());
	for (auto edgeIdx : nonFixedEdges)
	{
		size_t outIdx = graph.edgeOut(edgeIdx);
		bool cut = distribution(generator) <= edgeCutChances[outIdx];
		if (cut)
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (cutEdges.find(edgeIdx) != cutEdges.end())
				throw std::runtime_error("cutEdges.find(edgeIdx) != cutEdges.end()");
#endif
			cutEdges.insert(edgeIdx);
		}
	}
	enforceR3(graph, cutEdges);
	return graph.computePartition(*this, matchGroups, cutEdges);
}

bool RandomSampling::operator()(size_t i, PGA::Compiler::Graph::PartitionPtr& partition)
{
	if (acceptedSamples.find(partition->getUid()) != acceptedSamples.end())
		return false;
	auto uid = partition->getUid();
	for (auto edgeIdx : alwaysCutEdges)
		if (uid.at(edgeIdx) == '0')
			return false;
	for (auto edgeIdx : neverCutEdges)
		if (uid.at(edgeIdx) == '1')
			return false;
	if (!callback(acceptedSamples.size(), partition))
		return false;
	acceptedSamples.insert(partition->getUid());
#ifdef PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE
	std::cout << "sample " << acceptedSamples.size() << " accepted:" << std::endl << partition->getUid().c_str() << std::endl;
#endif
	return true;
}

HeuristicsTestSampling::HeuristicsTestSampling(const std::set<size_t>& alwaysCutEdges, 
	const std::set<size_t>& neverCutEdges, 
	PGA::Compiler::Graph& graph, 
	bool matchGroups, 
	PGA::Compiler::Graph::ComputePartitionCallback& callback, 
	HeuristicsTestSpecifications& heuristicsTestSpecifications) :
	SamplingStrategy(alwaysCutEdges, neverCutEdges, graph),
	samplesTester(matchGroups, alwaysCutEdges, neverCutEdges, callback, graph),
	heuristicsTestSpecifications(heuristicsTestSpecifications)
{
}

void HeuristicsTestSampling::computeSize(size_t numNonFixedEdges)
{
	size_t populationSize = static_cast<size_t>(pow(2, numNonFixedEdges));
	samplingSize = heuristicsTestSpecifications.calculateSampleSize(populationSize);
}

bool HeuristicsTestSampling::sample(const std::set<size_t>& graphAlwaysCutEdges, 
	const std::set<size_t>& nonFixedEdges, 
	std::map<size_t, float>& edgeCutChances)
{
	std::set<size_t> cutEdges1(alwaysCutEdges.begin(), alwaysCutEdges.end());
	std::set<size_t> cutEdges2(neverCutEdges.begin(), neverCutEdges.end());
	cutEdges1.insert(graphAlwaysCutEdges.begin(), graphAlwaysCutEdges.end());
	cutEdges2.insert(graphAlwaysCutEdges.begin(), graphAlwaysCutEdges.end());
	for (auto edgeIdx : nonFixedEdges)
	{
		size_t outIdx = graph.edgeOut(edgeIdx);
		bool cut = distribution(generator) <= edgeCutChances[outIdx];
		if (cut)
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (cutEdges1.find(edgeIdx) != cutEdges1.end())
				throw std::runtime_error("cutEdges1.find(edgeIdx) != cutEdges1.end()");
			if (cutEdges2.find(edgeIdx) != cutEdges2.end())
				throw std::runtime_error("cutEdges2.find(edgeIdx) != cutEdges2.end()");
#endif
			cutEdges1.insert(edgeIdx);
			cutEdges2.insert(edgeIdx);
		}
	}
	enforceR3(graph, cutEdges1);
	enforceR3(graph, cutEdges2);
	return samplesTester.accept(cutEdges1, cutEdges2);
}
