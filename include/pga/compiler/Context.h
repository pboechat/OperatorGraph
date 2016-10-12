#pragma once

#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <vector>

#include <pga/compiler/ShapeType.h>
#include <pga/compiler/Axiom.h>
#include <pga/compiler/Rule.h>
#include <pga/compiler/Parameters.h>
#include <pga/compiler/Edge.h>
#include <pga/compiler/Vertex.h>
#include <pga/compiler/Graph.h>

namespace PGA
{
	namespace Compiler
	{
		struct Context
		{
			Graph baseGraph;
			std::map<size_t, size_t> genFuncCounters;
			std::set<size_t> ruleEdges;
			std::set<std::string> terminalSymbols;

			Context(const Axiom& axiom, const std::vector<Rule>& rules);

		private:
			Axiom axiom;
			std::vector<std::shared_ptr<Vertex>> vertices;
			std::vector<std::shared_ptr<Edge>> edges;

			void traverseSymbol(const std::string& symbol, ShapeType shapeType, std::map<std::string, Rule>& ruleMap, size_t succIdx, std::shared_ptr<Parameter>& succParam, size_t srcVertexIdx, std::unordered_map<std::string, ShapeType>& symbols, std::unordered_map<size_t, std::shared_ptr<Vertex>>& visited, std::unordered_map<std::string, size_t>& visitedNamedTerminals, bool ruleEdge);
			void traverseOperator(std::shared_ptr<Operator>& op, ShapeType shapeType, std::map<std::string, Rule>& ruleMap, size_t succIdx, std::shared_ptr<Parameter>& succParam, size_t parVertexIdx, std::unordered_map<std::string, ShapeType>& symbols, std::unordered_map<size_t, std::shared_ptr<Vertex>>& visited, std::unordered_map<std::string, size_t>& visitedNamedTerminals, bool ruleEdge);
			bool addNewVertex(std::shared_ptr<Operator>& op, ShapeType shapeType, size_t succIdx, std::shared_ptr<Parameter>& succParam, size_t parVertexIdx, size_t& vertexIdx, bool loop, std::unordered_map<size_t, std::shared_ptr<Vertex>>& visited, bool ruleEdge);

		};

	}

}