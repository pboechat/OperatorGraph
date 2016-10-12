#pragma once

#include <set>

#include <pga/compiler/Graph.h>

enum ConfidenceLevel
{
	CL_90,
	CL_95,
	CL_99

};

struct HeuristicsTestSpecifications
{
	double errorMargin;
	ConfidenceLevel confidenceLevel;
	double standardDeviation;

	HeuristicsTestSpecifications(double errorMargin, ConfidenceLevel confidenceLevel, double standardDeviation);

	size_t calculateSampleSize(size_t populationSize) const;

};

struct HeuristicsSamplesTester : PGA::Compiler::Graph::ComputePartitionCallback
{
	HeuristicsSamplesTester(
		bool matchGroups,
		const std::set<size_t>& alwaysCutEdges,
		const std::set<size_t>& neverCutEdges,
		PGA::Compiler::Graph::ComputePartitionCallback& callback,
		PGA::Compiler::Graph& graph);

	bool accept(const std::set<size_t>& cutEdges1, const std::set<size_t>& cutEdges2);
	virtual bool operator()(size_t i, PGA::Compiler::Graph::PartitionPtr& partition);


private:
	std::string lastAcceptedSample;
	std::set<std::string> acceptedSamples;
	int phase;
	std::set<size_t> cutEdges2;
	bool matchGroups;
	const std::set<size_t>& alwaysCutEdges;
	const std::set<size_t>& neverCutEdges;
	PGA::Compiler::Graph::ComputePartitionCallback& callback;
	PGA::Compiler::Graph& graph;

	bool isInvalid(PGA::Compiler::Graph::PartitionPtr& partition) const;
	void accept(PGA::Compiler::Graph::PartitionPtr& partition);

};