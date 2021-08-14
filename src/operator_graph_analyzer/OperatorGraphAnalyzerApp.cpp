#include "CommonGraphOperations.h"
#include "DotGraphVisitor.h"
#include "HeuristicsTest.h"
#include "OperatorGraphAnalyzerApp.h"
#include "ParseUtils.h"
#include "PartitionOutputter.h"
#include "SamplingStrategies.h"
#include "SearchHeuristics.h"

#include <pga/compiler/Context.h>
#include <pga/compiler/DispatchTableGenerator.h>
#include <pga/compiler/Graph.h>
#include <pga/compiler/Logger.h>
#include <pga/compiler/Parser.h>
#include <pga/core/EnumUtils.h>
#include <pga/core/GPUTechnique.h>
#include <pga/core/StringUtils.h>

#include <cmath>
#include <fstream>
#include <initializer_list>
#include <streambuf>
#include <string>
#include <vector>

void abort_(const std::string& message = "")
{
	if (!message.empty())
		std::cout << message << std::endl;
	system("pause");
	exit(-1);
}

bool containsExactly(std::set<size_t>& set, const std::initializer_list<size_t>& list) /* */
{
	if (list.size() != set.size()) return false;
	for (auto item : list)
		if (set.find(item) == set.end())
			return false;
	return true;
}

template <typename T>
bool containsExactly(std::map<size_t, T>& map, const std::initializer_list<size_t>& list) /* */
{
	if (list.size() != map.size()) return false;
	for (auto item : list)
		if (map.find(item) == map.end())
			return false;
	return true;
}

const long long DEF_QMEM = 2150L;
const long DEF_GRIDX = 16;
const long DEF_GRIDY = 16;
const long DEF_MINST = 5000L;
const long DEF_MIND = 1500000L;
const long DEF_MVERT = 1000000L;
const auto DEF_ITEM_SIZE = 96LL;
const auto DEF_E = 4.0;
const auto DEF_CL = ConfidenceLevel::CL_95;
const auto DEF_STD = 0.5;
const auto DEF_SEED = -1L;
const auto MATCH_GROUPS = 1u;
const auto OPTIMIZE_CODE = 2u;

PGA::Compiler::ProcedureList procedures = {
	// BOX
	{ PGA::Compiler::TRANSLATE, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::ROTATE, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::SCALE, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::COMPSPLIT, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::REPEAT, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::SUBDIV, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::DISCARD, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::IF, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::IFSIZELESS, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::IFCOLLIDES, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::GENERATE, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::GENERATE, PGA::Compiler::BOX, 1 },
	{ PGA::Compiler::STOCHASTIC, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONVEX_POLYGON, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_POLYGON, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, PGA::Compiler::BOX, 0 },
	{ PGA::Compiler::SWAPSIZE, PGA::Compiler::BOX, 0 },

	// DYNAMIC_CONVEX_RIGHT_PRISM
	{ PGA::Compiler::TRANSLATE, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::ROTATE, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::SCALE, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::COMPSPLIT, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::REPEAT, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::SUBDIV, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::DISCARD, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::IF, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::IFSIZELESS, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::IFCOLLIDES, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::GENERATE, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::GENERATE, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 1 },
	{ PGA::Compiler::STOCHASTIC, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONVEX_POLYGON, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_POLYGON, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ PGA::Compiler::SWAPSIZE, PGA::Compiler::DYNAMIC_CONVEX_RIGHT_PRISM, 0 },

	// DYNAMIC_RIGHT_PRISM
	{ PGA::Compiler::TRANSLATE, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::ROTATE, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::SCALE, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::COMPSPLIT, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::REPEAT, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::SUBDIV, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::DISCARD, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::IF, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::IFSIZELESS, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::IFCOLLIDES, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::GENERATE, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::GENERATE, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 1 },
	{ PGA::Compiler::STOCHASTIC, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONVEX_POLYGON, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_POLYGON, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },
	{ PGA::Compiler::SWAPSIZE, PGA::Compiler::DYNAMIC_RIGHT_PRISM, 0 },

	// QUAD
	{ PGA::Compiler::TRANSLATE, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::ROTATE, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::SCALE, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::EXTRUDE, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::REPEAT, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::SUBDIV, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::DISCARD, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::IF, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::IFSIZELESS, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::IFCOLLIDES, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::GENERATE, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::GENERATE, PGA::Compiler::QUAD, 1 },
	{ PGA::Compiler::STOCHASTIC, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONVEX_POLYGON, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_POLYGON, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, PGA::Compiler::QUAD, 0 },
	{ PGA::Compiler::SWAPSIZE, PGA::Compiler::QUAD, 0 },

	// DYNAMIC_CONVEX_POLYGON
	{ PGA::Compiler::TRANSLATE, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::ROTATE, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::SCALE, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::EXTRUDE, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::REPEAT, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::SUBDIV, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::DISCARD, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::IF, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::IFSIZELESS, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::IFCOLLIDES, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::GENERATE, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::GENERATE, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 1 },
	{ PGA::Compiler::STOCHASTIC, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONVEX_POLYGON, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_POLYGON, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },
	{ PGA::Compiler::SWAPSIZE, PGA::Compiler::DYNAMIC_CONVEX_POLYGON, 0 },

	// DYNAMIC_POLYGON
	{ PGA::Compiler::TRANSLATE, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::ROTATE, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::SCALE, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::EXTRUDE, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::REPEAT, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::SUBDIV, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::DISCARD, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::IF, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::IFSIZELESS, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::IFCOLLIDES, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::GENERATE, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::GENERATE, PGA::Compiler::DYNAMIC_POLYGON, 1 },
	{ PGA::Compiler::STOCHASTIC, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONVEX_POLYGON, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_POLYGON, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, PGA::Compiler::DYNAMIC_POLYGON, 0 },
	{ PGA::Compiler::SWAPSIZE, PGA::Compiler::DYNAMIC_POLYGON, 0 },

};

int OperatorGraphAnalyzerApp::run(unsigned int argc, const char** argv)
{
    std::string pgaSourceFilePath;
	std::string templateFilePath;
	std::string out;
	std::string partitionUid;
	bool baseGraph = false;
	bool instrumented = false;
	auto optimization = 0u;
	auto maxProcs = -1L;
	auto minMatches = -1L;
	auto oneToOneOperatorEdgesHeuristic = false;
	auto randomRuleHeuristic = false;
	auto subtreeOrderHeuristicSize = -1L;
	auto maxNumVertices = DEF_MVERT, maxNumIndices = DEF_MIND, maxNumInstances = DEF_MINST;
	auto gridX = DEF_GRIDX, gridY = DEF_GRIDY;
	auto queuesMem = DEF_QMEM * 1024LL * 1024LL;
	auto itemSize = DEF_ITEM_SIZE;
	std::string pgaSourceCode;
	auto testHeuristics = false;
	auto randomSampling = 0u;
	auto seed = DEF_SEED;
	auto technique = PGA::GPU::MEGAKERNEL;
	HeuristicsTestSpecifications heuristicsTestSpecs(DEF_E, DEF_CL, DEF_STD);
	for (unsigned int i = 1; i < argc; i++)
	{
		std::vector<std::string> kvp;
		PGA::StringUtils::split(argv[i], '=', kvp);
		if (kvp.size() < 2) continue;
		if (kvp[0] == "path")
			pgaSourceFilePath = kvp[1];
		else if (kvp[0] == "src")
			pgaSourceCode = kvp[1];
		else if (kvp[0] == "templ")
			templateFilePath = kvp[1];
		else if (kvp[0] == "out")
			out = kvp[1];
		else if (kvp[0] == "bgraph")
			baseGraph = (kvp[1] != "false" && kvp[1] != "0");
		else if (kvp[0] == "opt")
			optimization = ParseUtils::parseArg(kvp[1], 0u);
		else if (kvp[0] == "instr")
			instrumented = (kvp[1] != "false" && kvp[1] != "0");
		else if (kvp[0] == "h1to1")
			oneToOneOperatorEdgesHeuristic = (kvp[1] != "false" && kvp[1] != "0");
		else if (kvp[0] == "hrr")
			randomRuleHeuristic = (kvp[1] != "false" && kvp[1] != "0");
		else if (kvp[0] == "hsto")
			subtreeOrderHeuristicSize = ParseUtils::parseArg(kvp[1], -1L);
		else if (kvp[0] == "puid")
			partitionUid = kvp[1];
		else if (kvp[0] == "mprocs")
			maxProcs = ParseUtils::parseArg(kvp[1], -1L);
		else if (kvp[0] == "mmatches")
			minMatches = ParseUtils::parseArg(kvp[1], -1L);
		else if (kvp[0] == "mvert")
			maxNumVertices = ParseUtils::parseArg(kvp[1], DEF_MVERT);
		else if (kvp[0] == "mind")
			maxNumIndices = ParseUtils::parseArg(kvp[1], DEF_MIND);
		else if (kvp[0] == "minst")
			maxNumInstances = ParseUtils::parseArg(kvp[1], DEF_MINST);
		else if (kvp[0] == "gridx")
			gridX = ParseUtils::parseArg(kvp[1], DEF_GRIDX);
		else if (kvp[0] == "gridy")
			gridY = ParseUtils::parseArg(kvp[1], DEF_GRIDY);
		else if (kvp[0] == "qmem")
			queuesMem = ParseUtils::parseArg(kvp[1], DEF_QMEM) * 1024LL * 1024LL;
		else if (kvp[0] == "isize")
			itemSize = ParseUtils::parseArg(kvp[1], DEF_ITEM_SIZE);
		else if (kvp[0] == "htspecs")
			testHeuristics = ParseUtils::parseHeuristicsTestSpecs(kvp[1], heuristicsTestSpecs, DEF_E, DEF_CL, DEF_STD);
		else if (kvp[0] == "rsampl")
			randomSampling = ParseUtils::parseArg(kvp[1], 0u);
		else if (kvp[0] == "seed")
			seed = ParseUtils::parseArg(kvp[1], DEF_SEED);
		else if (kvp[0] == "tech")
			technique = static_cast<PGA::GPU::Technique>(ParseUtils::parseArg(kvp[1], (unsigned int)PGA::GPU::MEGAKERNEL));
	}

	if (randomSampling > 0 && testHeuristics)
		abort_("random samples and heuristics testing are mutually exclusive");

	if (pgaSourceFilePath.empty() && pgaSourceCode.empty()) /* */
		abort_("pga source unspecified");

	if (templateFilePath.empty())
		abort_("template file unspecified");

	if (pgaSourceCode.empty())
	{
		std::ifstream pgaSourceFile(pgaSourceFilePath);
		pgaSourceCode = std::string(std::istreambuf_iterator<char>(pgaSourceFile), std::istreambuf_iterator<char>());
		pgaSourceFile.close();
	}

	std::ifstream templateFile(templateFilePath);
	std::string templateCode((std::istreambuf_iterator<char>(templateFile)), std::istreambuf_iterator<char>());
	templateFile.close();

	std::vector<PGA::Compiler::Axiom> axioms;
	std::vector<PGA::Compiler::Rule> rules;
	PGA::Compiler::Logger logger;
	if (!PGA::Compiler::Parser::parse(pgaSourceCode, logger, axioms, rules))
	{
		abort_("[COMPILER ERRORS] " + logger[PGA::Compiler::Logger::LL_ERROR].str());
	}
	if (logger.hasMessages(PGA::Compiler::Logger::LL_WARNING))
		std::cout << "[COMPILER WARNINGS] " << logger[PGA::Compiler::Logger::LL_WARNING].str() << std::endl;

	if (axioms.empty() || rules.empty())
		throw std::runtime_error("axioms.empty() || rules.empty()");

	PGA::Compiler::Context context(axioms[0], rules);
	std::set<size_t> alwaysCutEdges;
	context.baseGraph.computePhases(alwaysCutEdges);
	std::cout << "no. edges: " << context.baseGraph.numEdges() << std::endl;

	DotGraphVisitor visitor;
	context.baseGraph.depthFirst(visitor);
	std::ofstream dotFile;
	dotFile.open(out + "/base_graph.dot");
	dotFile << visitor;
	dotFile.close();

	if (baseGraph)
	{
#ifdef _DEBUG
		system("pause");
#endif
		return EXIT_SUCCESS;
	}

	std::string genFuncCountersStr = "{ ";
	auto it = context.genFuncCounters.begin();
	for (auto i = 0; i < context.genFuncCounters.size() - 1; i++, it++)
		genFuncCountersStr += "{ " + std::to_string(it->first) +
		", " + std::to_string(it->second) + " }, ";
	genFuncCountersStr += "{ " + std::to_string(it->first) + ", " + std::to_string(it->second) + " }";
	genFuncCountersStr += " }";

	PGA::StringUtils::replaceAll(templateCode, "<genFuncCounters>", genFuncCountersStr);
	PGA::StringUtils::replaceAll(templateCode, "<numEdges>", std::to_string(context.baseGraph.numEdges()));
	PGA::StringUtils::replaceAll(templateCode, "<optimization>", std::to_string(optimization));
	PGA::StringUtils::replaceAll(templateCode, "<instrumented>", ((instrumented) ? "true" : "false"));
	PGA::StringUtils::replaceAll(templateCode, "<gridX>", std::to_string(gridX));
	PGA::StringUtils::replaceAll(templateCode, "<gridY>", std::to_string(gridY));
	PGA::StringUtils::replaceAll(templateCode, "<maxNumVertices>", std::to_string(maxNumVertices));
	PGA::StringUtils::replaceAll(templateCode, "<maxNumIndices>", std::to_string(maxNumIndices));
	PGA::StringUtils::replaceAll(templateCode, "<maxNumInstances>", std::to_string(maxNumInstances));
	PGA::StringUtils::replaceAll(templateCode, "<technique>", "PGA::GPU::" + PGA::EnumUtils::toString(technique));

	std::unique_ptr<PartitionOutputter> outputter;
	if (out.empty())
		outputter = std::unique_ptr<PartitionOutputter>(new StreamsPartitionOutputter(
			std::cout, 
			std::cout, 
			std::cout, 
			optimization, 
			instrumented, 
			minMatches, 
			maxProcs, 
			queuesMem, 
			itemSize, 
			templateCode));
	else
		outputter = std::unique_ptr<PartitionOutputter>(new FilesPartitionOutputter(
			out, 
			optimization, 
			instrumented, 
			minMatches, 
			maxProcs, 
			queuesMem, 
			itemSize, 
			templateCode));

	if (partitionUid.empty())
	{
		if (testHeuristics && !oneToOneOperatorEdgesHeuristic && !randomRuleHeuristic && (subtreeOrderHeuristicSize < 1))
		{
			std::cout << "WARNING: no heuristics to test!" << std::endl;
			testHeuristics = false;
		}

		std::map<size_t, size_t> replacementMappings;
		std::vector<std::set<size_t>> replacementGroups;

		std::set<size_t> neverCutEdges;
		if (randomRuleHeuristic)
		{
			SearchHeuristics::RandomRuleHeuristic randomRuleHeuristic(alwaysCutEdges);
			context.baseGraph.depthFirst(randomRuleHeuristic);
		}
		if (subtreeOrderHeuristicSize > -1)
		{
			SearchHeuristics::SubtreeOrderHeuristic subtreeOrderHeuristic(static_cast<size_t>(subtreeOrderHeuristicSize), alwaysCutEdges, neverCutEdges);
			context.baseGraph.breadthFirst(subtreeOrderHeuristic);
		}
		if (oneToOneOperatorEdgesHeuristic)
		{
			SearchHeuristics::OneToOneOperatorEdgesHeuristic oneToOneOperatorEdgesHeuristic(alwaysCutEdges, neverCutEdges);
			context.baseGraph.depthFirst(oneToOneOperatorEdgesHeuristic);
		}

		enforceR3(context.baseGraph, alwaysCutEdges);

		std::cout << "no. always-cut edges (heuristics): " << alwaysCutEdges.size() << std::endl;
		std::cout << "no. never-cut edges (heuristics): " << neverCutEdges.size() << std::endl;
#ifdef PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE
		if (!alwaysCutEdges.empty())
		{
			std::cout << "always-cut edges (heuristics): ";
			for (auto alwaysCutEdge : alwaysCutEdges)
				std::cout << alwaysCutEdge << " ";
			std::cout << std::endl;
		}
		if (!neverCutEdges.empty())
		{
			std::cout << "never-cut edges (heuristics): ";
			for (auto neverCutEdge : neverCutEdges)
				std::cout << neverCutEdge << " ";
			std::cout << std::endl;
		}
#endif
		std::unique_ptr<SamplingStrategy> samplingStrategy;
		if (randomSampling > 0)
			samplingStrategy = std::unique_ptr<SamplingStrategy>(new RandomSampling(alwaysCutEdges,
			neverCutEdges,
			context.baseGraph, 
			(optimization & MATCH_GROUPS) != 0,
			*outputter.get(),
			randomSampling));
		else if (testHeuristics)
			samplingStrategy = std::unique_ptr<SamplingStrategy>(new HeuristicsTestSampling(
				alwaysCutEdges, 
				neverCutEdges, 
				context.baseGraph,
				(optimization & MATCH_GROUPS) != 0, 
				*outputter.get(), 
				heuristicsTestSpecs));

		if (samplingStrategy != nullptr)
		{
			// TODO: anticipate edges interdependencies possibly leads to random sampling rejection!
			auto numEdges = context.baseGraph.numEdges();
			std::set<size_t> graphAlwaysCutEdges, graphNeverCutEdges, fixedEdges;
			context.baseGraph.findAlwaysCutEdges(graphAlwaysCutEdges);
			context.baseGraph.findNeverCutEdges(graphNeverCutEdges);
			fixedEdges.insert(graphAlwaysCutEdges.begin(), graphAlwaysCutEdges.end());
			fixedEdges.insert(graphNeverCutEdges.begin(), graphNeverCutEdges.end());
			fixedEdges.insert(alwaysCutEdges.begin(), alwaysCutEdges.end());
			fixedEdges.insert(neverCutEdges.begin(), neverCutEdges.end());
			samplingStrategy->computeSize(numEdges - fixedEdges.size());
			std::set<size_t> nonFixedEdges;
#ifdef PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE
			std::string mask1, mask2;
#endif
			for (size_t edgeIdx = 0; edgeIdx < numEdges; edgeIdx++)
			{
				bool alwaysCut = alwaysCutEdges.find(edgeIdx) != alwaysCutEdges.end();
				bool neverCut = neverCutEdges.find(edgeIdx) != neverCutEdges.end();
				bool graphAlwaysCut = graphAlwaysCutEdges.find(edgeIdx) != graphAlwaysCutEdges.end();
				bool graphNeverCut = graphNeverCutEdges.find(edgeIdx) != graphNeverCutEdges.end();
				if (alwaysCut && neverCut)
					throw std::runtime_error(("there are conflicting heuristics over edge " + std::to_string(edgeIdx)).c_str());
				// FIXME: checking invariants
				if (graphAlwaysCut && graphNeverCut)
					throw std::runtime_error(("edge " + std::to_string(edgeIdx) + " is considered both never and always-cut (which is impossible!)").c_str());
#ifdef PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE
				if (alwaysCut && graphNeverCut)
					std::cout << "edge " << edgeIdx << " is considered an always-cut edge by some heuristic, but it should always be considered a never-cut" << std::endl;
				if (neverCut && graphAlwaysCut)
					std::cout << "edge " << edgeIdx << " is considered a never-cut edge by some heuristic, but it should always be considered an always-cut" << std::endl;
				if (neverCut && graphNeverCut)
					std::cout << "edge " << edgeIdx << " is considered a never-cut by some heuristic, but it should always be a never-cut, so the edge will not be considered in the heuristics test" << std::endl;
#endif
				if (graphAlwaysCut || graphNeverCut || alwaysCut || neverCut)
				{
#ifdef PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE
					if (graphAlwaysCut)
					{
						mask1 += "1";
						mask2 += "1";
					}
					else if (graphNeverCut)
					{
						mask1 += "0";
						mask2 += "0";
					}
					else if (alwaysCut)
					{
						mask1 += "1";
						mask2 += "0";
					}
					else
					{
						mask1 += "0";
						mask2 += "1";
					}
#endif
					continue;
				}
				nonFixedEdges.insert(edgeIdx);
#ifdef PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE
				mask1 += "_";
				mask2 += "_";
#endif
			}
#ifdef PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE
			std::cout << std::endl << "partition mask: " << std::endl << mask1 << std::endl;
			std::cout << std::endl << "antagonistic partition mask: " << std::endl << mask2 << std::endl << std::endl;
#endif
			samplingStrategy->computeSamples(seed, graphAlwaysCutEdges, nonFixedEdges);
		}
		else
		{
			context.baseGraph.computePartitions(
				*outputter.get(),
				(optimization & MATCH_GROUPS) != 0,
				alwaysCutEdges,
				neverCutEdges,
				replacementMappings,
				replacementGroups
			);
		}
	}
	else
	{
		std::set<size_t> cutEdges;
		if (partitionUid == "all_cut")
		{
			for (auto i = 0; i < context.baseGraph.numEdges(); i++)
				cutEdges.insert(i);
		}
		else if (partitionUid == "none_cut")
		{
			// do nothing
		}
		else if (partitionUid == "designers_choice")
		{
			cutEdges = context.ruleEdges;
		}
		else
		{
			cutEdges = PGA::Compiler::Graph::Partition::cutEdgesFromUid(partitionUid);
		}
		std::cout << "partition asked: " << PGA::Compiler::Graph::Partition::uidFromCutEdges(context.baseGraph.numEdges(), cutEdges) << std::endl;
		std::set<size_t> alwaysCutEdges;
		std::set<size_t> neverCutEdges;
		context.baseGraph.findAlwaysCutEdges(alwaysCutEdges);
		context.baseGraph.findNeverCutEdges(neverCutEdges, alwaysCutEdges);
		for (auto edge : alwaysCutEdges)
		{
			if (neverCutEdges.find(edge) != neverCutEdges.end())
				// FIXME: checking invariants
				throw std::runtime_error("neverCutEdges.find(edge) != neverCutEdges.end()");
			auto it = cutEdges.find(edge);
			if (it == cutEdges.end())
				cutEdges.insert(edge);
		}
		for (auto edge : neverCutEdges)
		{
			if (alwaysCutEdges.find(edge) != alwaysCutEdges.end())
				// FIXME: checking invariants
				throw std::runtime_error("alwaysCutEdges.find(edge) != alwaysCutEdges.end()");
			auto it = cutEdges.find(edge);
			if (it != cutEdges.end())
				cutEdges.erase(it);
		}
		std::cout << "partition enforced: " << PGA::Compiler::Graph::Partition::uidFromCutEdges(context.baseGraph.numEdges(), cutEdges) << std::endl;
		if (!context.baseGraph.computePartition(
			*outputter.get(),
			(optimization & MATCH_GROUPS) != 0,
			cutEdges
			))
			std::cout << "couldn't produce a valid partition" << std::endl;
	}

#if defined(_WIN32) && defined(_DEBUG)
	system("pause");
#endif

	return EXIT_SUCCESS;
}