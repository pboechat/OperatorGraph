#pragma once

#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <deque>
#include <map>

#include <pga/core/DispatchTable.h>
#include <pga/compiler/ShapeType.h>
#include <pga/compiler/OperatorType.h>
#include <pga/compiler/ProcedureList.h>
#include <pga/compiler/Logger.h>
#include <pga/compiler/Axiom.h>
#include <pga/compiler/Rule.h>
#include <pga/compiler/Parameters.h>
#include <pga/compiler/Symbol.h>
#include <pga/compiler/Operator.h>
#include <pga/compiler/Graph.h>
#include <pga/compiler/GraphVisitor.h>

namespace PGA
{
	namespace Compiler
	{
		struct DispatchTableGenerator
		{
			std::unique_ptr<DispatchTable> dispatchTable;
			std::map<std::string, size_t> symbolToEntryIdx;
			std::map<std::string, ShapeType> symbolToInputShape;
			std::set<std::string> terminalSymbols;
			std::unordered_set<SingleOperatorProcedure> proceduresUsed;

			bool fromRuleSet(const std::vector<Axiom>& axioms, const std::vector<Rule>& rules, const ProcedureList& procedures, Logger& logger);
			bool fromBaseGraph(const PGA::Compiler::Graph& graph, const ProcedureList& procedureList, Logger& logger);

		private:
			struct BaseGraphVisitor : PGA::Compiler::GraphVisitor
			{
				BaseGraphVisitor(DispatchTableGenerator& parent, const ProcedureList& procedureList, Logger& logger) : parent(parent), procedureList(procedureList), logger(logger), entryIndex(-1) {}
				virtual bool visit(size_t i, const Vertex_LW& vertex);
				virtual bool visit(size_t i, const Edge_LW& edge, const Vertex_LW& src, const Vertex_LW& dst);

			private:
				DispatchTableGenerator& parent;
				const ProcedureList& procedureList;
				Logger& logger;
				int entryIndex;

			};

			bool visitSuccessor(size_t entryIdx, ShapeType inputShape, const std::shared_ptr<Successor>& successor, const ProcedureList& procedures, Logger& logger, std::deque<std::shared_ptr<Symbol>>& symbolsToVisit, const std::set<std::string>& visitedSymbols);
			bool visitOperator(size_t entryIdx, ShapeType inputShape, const std::shared_ptr<Operator>& successor, const ProcedureList& procedures, Logger& logger, std::deque<std::shared_ptr<Symbol>>& symbolsToVisit, const std::set<std::string>& visitedSymbols);
			bool visitSymbol(size_t entryIdx, ShapeType inputShape, const std::shared_ptr<Symbol>& symbol, Logger& logger, std::deque<std::shared_ptr<Symbol>>& symbolsToVisit, const std::set<std::string>& visitedSymbols);
			bool visitTerminalSymbol(size_t entryIdx, ShapeType inputShape, const std::shared_ptr<Symbol>& symbol, const ProcedureList& procedures, Logger& logger, const std::set<std::string>& visitedSymbols);
			void buildColliderEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase);
			void buildTranslateEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase);
			void buildRotateEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase);
			void buildScaleEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase);
			void buildExtrudeEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase);
			void buildComponentSplitEntry(DispatchTable::Entry& newEntry, const std::vector<size_t>& successorEntries, std::vector<unsigned int> succPhases);
			void buildSubdivideEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, const std::vector<std::shared_ptr<Parameter>>& factors, const std::vector<size_t>& successorEntries, std::vector<unsigned int> succPhases);
			void buildRepeatEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase);
			void buildDiscardEntry(DispatchTable::Entry& newEntry);
			void buildIfEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdxIfTrue, size_t succEntryIdxIfFalse, std::vector<unsigned int> succPhases);
			void buildIfCollidesEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdxIfTrue, size_t succEntryIdxIfFalse, std::vector<unsigned int> succPhases);
			void buildIfSizeLessEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdxIfTrue, size_t succEntryIdxIfFalse, std::vector<unsigned int> succPhases);
			void buildRandomRuleEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, const std::vector<std::shared_ptr<Parameter>>& factors, const std::vector<size_t>& successorEntries, std::vector<unsigned int> succPhases);
			void buildSetAsDynamicConvexPolygonEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase);
			void buildSetAsDynamicConvexRightPrismEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase);
			void buildSetAsDynamicConcavePolygonEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase);
			void buildSetAsDynamicConcaveRightPrismEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase);
			void buildGenerateEntry(DispatchTable::Entry& newEntry, const std::vector<double>& termAttrs = std::vector<double>());
			void buildSwapSizeEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase);
			void buildReplicateEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, const std::vector<std::shared_ptr<Parameter>>& factors, const std::vector<size_t>& succEntryIdxs, std::vector<unsigned int> succPhases);
			void buildEntryFromGraph(int entryIndex, long procIdx, const Operator* op, const std::vector<size_t> succEntryIdxs, std::vector<unsigned int> succPhases);
			friend struct BaseGraphVisitor;

		};

	}

}
