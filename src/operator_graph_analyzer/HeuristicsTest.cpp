#include <cmath>
#include <iostream>

#include "HeuristicsTest.h"

//////////////////////////////////////////////////////////////////////////
HeuristicsSamplesTester::HeuristicsSamplesTester(bool matchGroups, const std::set<size_t>& alwaysCutEdges, const std::set<size_t>& neverCutEdges, PGA::Compiler::Graph::ComputePartitionCallback& callback, PGA::Compiler::Graph& graph) :
	matchGroups(matchGroups),
	alwaysCutEdges(alwaysCutEdges),
	neverCutEdges(neverCutEdges),
	callback(callback),
	graph(graph)
{
}

bool HeuristicsSamplesTester::accept(const std::set<size_t>& cutEdges1, const std::set<size_t>& cutEdges2)
{
	phase = 0;
	this->cutEdges2.clear();
	this->cutEdges2.insert(cutEdges2.begin(), cutEdges2.end());
	graph.computePartition(*this, matchGroups, cutEdges1);
	return phase == 3;
}

void HeuristicsSamplesTester::accept(PGA::Compiler::Graph::PartitionPtr& partition)
{
	callback(acceptedSamples.size(), partition);
	lastAcceptedSample = partition->getUid();
	acceptedSamples.insert(lastAcceptedSample);
}

bool HeuristicsSamplesTester::operator()(size_t i, PGA::Compiler::Graph::PartitionPtr& partition)
{
	if (acceptedSamples.find(partition->getUid()) != acceptedSamples.end())
		return false;

	if (phase == 0)
	{
		if (isInvalid(partition))
			return false;

		phase = 1;
		graph.computePartition(*this, matchGroups, cutEdges2);

		if (phase == 2)
		{
#ifdef PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE
			std::cout << "pair " << (acceptedSamples.size() / 2) << " accepted:" << std::endl << partition->getUid().c_str() << std::endl << lastAcceptedSample.c_str() << std::endl << std::endl;
#endif
			accept(partition);
			phase = 3;
		}
	}
	else if (phase == 1)
	{
		if (isInvalid(partition))
			return false;

		accept(partition);
		phase = 2;
	}
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
	else
		throw std::runtime_error("invalid phase");
#endif
	return true;
}

bool HeuristicsSamplesTester::isInvalid(PGA::Compiler::Graph::PartitionPtr& partition) const
{
	auto uid = partition->getUid();
	if (phase == 0)
	{
		for (auto edgeIdx : alwaysCutEdges)
			if (uid.at(edgeIdx) == '0')
				return true;
		for (auto edgeIdx : neverCutEdges)
			if (uid.at(edgeIdx) == '1')
				return true;
	}
	else if (phase == 1)
	{
		for (auto edgeIdx : alwaysCutEdges)
			if (uid.at(edgeIdx) == '1')
				return true;
		for (auto edgeIdx : neverCutEdges)
			if (uid.at(edgeIdx) == '0')
				return true;
	}
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
	else
		throw std::runtime_error("invalid phase");
#endif
	return false;
}

//////////////////////////////////////////////////////////////////////////
HeuristicsTestSpecifications::HeuristicsTestSpecifications(double errorMargin, ConfidenceLevel confidenceLevel, double standardDeviation) : errorMargin(errorMargin), confidenceLevel(confidenceLevel), standardDeviation(standardDeviation)
{
}

size_t HeuristicsTestSpecifications::calculateSampleSize(size_t populationSize) const
{
	double zVal;
	switch (confidenceLevel)
	{
	case CL_90:
		zVal = 1.645;
		break;
	case CL_95:
		zVal = 1.96;
		break;
	case CL_99:
		zVal = 2.576;
		break;
	default:
		// FIXME: checking invariants
		throw std::runtime_error("unknown confidence level");
	}
	double confidenceInterval = errorMargin / 100.0;
	double sampleSize = ((zVal * zVal) * 0.25) / pow(confidenceInterval, 2.0);
	sampleSize = sampleSize / (1.0 + (sampleSize - 1.0) / static_cast<double>(populationSize));
	return static_cast<size_t>(sampleSize);
}

