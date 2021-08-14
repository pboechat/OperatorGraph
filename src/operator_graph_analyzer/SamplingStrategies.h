#pragma once

#include <pga/compiler/Graph.h>

#include <chrono>
#include <map>
#include <random>
#include <set>

struct SamplingStrategy
{
	SamplingStrategy(const std::set<size_t>& alwaysCutEdges,
		const std::set<size_t>& neverCutEdges,
		PGA::Compiler::Graph& graph);

	virtual void computeSize(size_t numNonFixedEdges) = 0;
	void computeSamples(long seed,
		const std::set<size_t>& graphAlwaysCutEdges,
		const std::set<size_t>& nonFixedEdges);

protected:
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution;
	const std::set<size_t>& alwaysCutEdges;
	const std::set<size_t>& neverCutEdges;
	PGA::Compiler::Graph& graph;
	size_t samplingSize;

	virtual bool sample(const std::set<size_t>& graphAlwaysCutEdges,
		const std::set<size_t>& nonFixedEdges,
		std::map<size_t, float>& edgeCutChances) = 0;

};

struct RandomSampling : public SamplingStrategy, public PGA::Compiler::Graph::ComputePartitionCallback
{
	RandomSampling(const std::set<size_t>& alwaysCutEdges,
		const std::set<size_t>& neverCutEdges,
		PGA::Compiler::Graph& graph, 
		bool matchGroups,
		PGA::Compiler::Graph::ComputePartitionCallback& callback,
		size_t samplingSize);

	virtual void computeSize(size_t numNonFixedEdges);

	virtual bool operator()(size_t i, PGA::Compiler::Graph::PartitionPtr& partition);

protected:
	virtual bool sample(const std::set<size_t>& graphAlwaysCutEdges,
		const std::set<size_t>& nonFixedEdges,
		std::map<size_t, float>& edgeCutChances);

private:
	bool matchGroups;
	PGA::Compiler::Graph::ComputePartitionCallback& callback;
	std::set<std::string> acceptedSamples;
	bool success;

};

struct HeuristicsTestSampling : public SamplingStrategy
{
	HeuristicsTestSampling(const std::set<size_t>& alwaysCutEdges,
		const std::set<size_t>& neverCutEdges,
		PGA::Compiler::Graph& graph,
		bool matchGroups,
		PGA::Compiler::Graph::ComputePartitionCallback& callback,
		HeuristicsTestSpecifications& heuristicsTestSpecifications);

	virtual void computeSize(size_t numNonFixedEdges);

protected:
	virtual bool sample(const std::set<size_t>& graphAlwaysCutEdges,
		const std::set<size_t>& nonFixedEdges,
		std::map<size_t, float>& edgeCutChances);

private:
	HeuristicsSamplesTester samplesTester;
	HeuristicsTestSpecifications& heuristicsTestSpecifications;

};