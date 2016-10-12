#include <chrono>
#include <algorithm>
#include <stdexcept>

#include <pga/compiler/Operator.h>
#include <pga/compiler/Symbol.h>
#include <pga/compiler/Context.h>

namespace PGA
{
	namespace Compiler
	{
		Context::Context(const Axiom& axiom, const std::vector<Rule>& rules) : axiom(axiom)
		{
			std::map<std::string, Rule> ruleMap;
			for (auto& rule : rules)
				ruleMap[rule.symbol] = rule;
			std::shared_ptr<Parameter> succParam(nullptr);
			std::unordered_map<std::string, ShapeType> symbols;
			std::unordered_map<size_t, std::shared_ptr<Vertex>> visited;
			std::unordered_map<std::string, size_t> visitedNamedTerminals;
			traverseSymbol(axiom.symbol, axiom.shapeType, ruleMap, 0, succParam, 0, symbols, visited, visitedNamedTerminals, false);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (vertices.empty())
				throw std::runtime_error("vertices.empty()");
#endif
			baseGraph = Graph(vertices, edges);
		}

		void Context::traverseSymbol(const std::string& symbol, ShapeType inputShape, std::map<std::string, Rule>& ruleMap, size_t succIdx, std::shared_ptr<Parameter>& succParam, size_t parVertexIdx, std::unordered_map<std::string, ShapeType>& symbols, std::unordered_map<size_t, std::shared_ptr<Vertex>>& visited, std::unordered_map<std::string, size_t>& visitedNamedTerminals, bool ruleEdge /*= false*/)
		{
			auto it1 = symbols.find(symbol);
			if (it1 == symbols.end())
			{
				symbols.emplace(symbol, inputShape);
				auto it2 = ruleMap.find(symbol);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it2 == ruleMap.end())
					throw std::runtime_error("rule " + symbol + " does not exist!");
#endif
				auto& rule = it2->second;
				if (rule.successor->getType() == SuccessorType::SYMBOL)
				{
					auto nextSymbol = std::dynamic_pointer_cast<Symbol>(rule.successor);
					auto it3 = symbols.find(nextSymbol->name);
					if (it3 != symbols.end())
					{
						if (it3->second != inputShape)
							throw std::runtime_error("it3->second != shapeType");
					}
					if (ruleMap.find(nextSymbol->name) != ruleMap.end())
						traverseSymbol(nextSymbol->name, inputShape, ruleMap, succIdx, succParam, parVertexIdx, symbols, visited, visitedNamedTerminals, ruleEdge);
					else
					{
						symbols.erase(symbol);
						return;
					}
				}
				else
				{
					auto nextOp = std::dynamic_pointer_cast<Operator>(rule.successor);
					nextOp->shapeType = inputShape;
					traverseOperator(nextOp, inputShape, ruleMap, succIdx, succParam, parVertexIdx, symbols, visited, visitedNamedTerminals, ruleEdge);
				}
			}
			else
			{
				// NOTE: loop found
				if (it1->second != inputShape)
					throw std::runtime_error("it1->second != shapeType");
				auto it2 = ruleMap.find(symbol);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it2 == ruleMap.end())
					throw std::runtime_error("rule " + symbol + " does not exist!");
#endif
				auto nextOp = std::dynamic_pointer_cast<Operator>(it2->second.successor);
				nextOp->shapeType = Operator::nextShapeType(nextOp->type, inputShape, 0);
				// NOTE: stop operator traversal at loop
				size_t unusedVertexIdx;
				addNewVertex(nextOp, it1->second, succIdx, succParam, parVertexIdx, unusedVertexIdx, true, visited, true);
				return;
			}
			symbols.erase(symbol);
		}

		void Context::traverseOperator(std::shared_ptr<Operator>& op, ShapeType inputShape, std::map<std::string, Rule>& ruleMap, size_t succIdx, std::shared_ptr<Parameter>& succParam, size_t parVertexIdx, std::unordered_map<std::string, ShapeType>& symbols, std::unordered_map<size_t, std::shared_ptr<Vertex>>& visited, std::unordered_map<std::string, size_t>& visitedNamedTerminals, bool ruleEdge)
		{
			size_t currVertexIdx;
			// NOTE: stop operator traversal on visit of an existing edge
			if (!addNewVertex(op, inputShape, succIdx, succParam, parVertexIdx, currVertexIdx, false, visited, ruleEdge))
				return;

			for (size_t i = 0; i < op->successors.size(); i++)
			{
				ShapeType nextShapeType = Operator::nextShapeType(op->type, inputShape, i);

				std::shared_ptr<Parameter> succParam;
				if (i < op->successorParams.size())
					succParam = op->successorParams[i];

				if (op->successors[i]->getType() == SuccessorType::SYMBOL)
				{
					auto nextSymbol = std::dynamic_pointer_cast<Symbol>(op->successors[i]);
					if (ruleMap.find(nextSymbol->name) != ruleMap.end())
						traverseSymbol(nextSymbol->name, nextShapeType, ruleMap, i, succParam, currVertexIdx, symbols, visited, visitedNamedTerminals, true);
					else
					{
						auto it2 = visitedNamedTerminals.find(nextSymbol->name);
						if (it2 != visitedNamedTerminals.end())
						{
							auto nextVertex = visited.at(it2->second);
							// FIXME:
							if (nextVertex->shapeType != nextShapeType)
								throw std::runtime_error("nextVertex->shapeType != nextShapeType");
							size_t unusedVertexIdx;
							addNewVertex(nextVertex->op, nextShapeType, i, succParam, currVertexIdx, unusedVertexIdx, false, visited, true);
						}
						else
						{
							auto newOp = std::make_shared<Operator>();
							newOp->id = nextSymbol->id;
							newOp->myrule = nextSymbol->myrule;
							newOp->shapeType = nextShapeType;
							newOp->type = OperatorType::GENERATE;
							auto& termAttrs = nextSymbol->terminalAttributes;
							newOp->terminalAttributes = termAttrs;
							size_t genFuncIdx;
							size_t termIdx;
							auto it1 = termAttrs.begin();
							if (it1 != termAttrs.end())
							{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
								if (*it1 < 0)
									throw std::runtime_error("*it1 < 0");
#endif
								genFuncIdx = static_cast<size_t>(*it1);
								newOp->genFuncIdx = static_cast<long>(genFuncIdx);
								if (++it1 != termAttrs.end())
								{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
									if (*it1 < 0)
										throw std::runtime_error("*it1 < 0");
#endif
									termIdx = static_cast<size_t>(*it1);
									auto it2 = genFuncCounters.find(genFuncIdx);
									if (it2 != genFuncCounters.end())
									{
										if (termIdx >= it2->second)
											it2->second = termIdx + 1;
									}
									else
										genFuncCounters[genFuncIdx] = termIdx + 1;
									newOp->termAttrs.resize(termAttrs.size() - 1);
									std::copy(it1, termAttrs.end(), newOp->termAttrs.begin());
								}
								else
								{
									termIdx = 0;
									auto it = genFuncCounters.find(genFuncIdx);
									if (it == genFuncCounters.end())
										genFuncCounters[genFuncIdx] = 0;
									newOp->termAttrs.push_back(static_cast<double>(termIdx));
								}
							}
							else
							{
								genFuncIdx = 0;
								newOp->genFuncIdx = 0L;
								termIdx = 0;
								auto it = genFuncCounters.find(genFuncIdx);
								if (it == genFuncCounters.end())
									genFuncCounters[genFuncIdx] = 0;
								newOp->termAttrs.push_back(static_cast<double>(termIdx));
							}
#ifdef VERBOSE
							std::cout << "terminal " << nextSymbol->name << " [genFuncIdx=" << genFuncIdx << ", termIdx=" << termIdx << "]" << std::endl;
#endif
							visitedNamedTerminals.emplace(nextSymbol->name, newOp->id);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
							if (terminalSymbols.find(nextSymbol->name) != terminalSymbols.end())
								throw std::runtime_error("terminalSymbols.find(nextSymbol->name) != terminalSymbols.end()");
#endif
							terminalSymbols.insert(nextSymbol->name);
							size_t unusedVertexIdx;
							addNewVertex(newOp, nextShapeType, i, succParam, currVertexIdx, unusedVertexIdx, false, visited, true);
						}
					}
				}
				else
				{
					auto nextOp = std::dynamic_pointer_cast<Operator>(op->successors[i]);
					nextOp->shapeType = nextShapeType;
					// NOTE: stop operator traversal on Discard
					if (nextOp->type == OperatorType::DISCARD)
					{
						size_t unusedVertexIdx;
						addNewVertex(nextOp, nextShapeType, i, succParam, currVertexIdx, unusedVertexIdx, false, visited, false);
					}
					else
					{
						traverseOperator(nextOp, nextShapeType, ruleMap, i, succParam, currVertexIdx, symbols, visited, visitedNamedTerminals, false);
					}
				}
			}
		}

		bool Context::addNewVertex(std::shared_ptr<Operator>& op, ShapeType shapeType, size_t succIdx, std::shared_ptr<Parameter>& succParam, size_t parVertexIdx, size_t& vertexIdx, bool loop, std::unordered_map<size_t, std::shared_ptr<Vertex>>& visited, bool ruleEdge)
		{
			auto it1 = visited.find(op->id);
			if (it1 != visited.end())
			{
				auto it2 = visited.find(op->id);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it2 == visited.end())
					throw std::runtime_error("it2 == visited.end()");
#endif
				auto dstVertex = it2->second;
				auto parVertex = vertices[parVertexIdx];
				auto edgeIdx = edges.size();
				auto newEdge = std::make_shared<Edge>(edgeIdx, parVertex, dstVertex, succIdx, succParam, loop);
				dstVertex->incomingEdges.push_back(newEdge);
				parVertex->outgoingEdges.push_back(newEdge);
				edges.push_back(newEdge);
				if (ruleEdge)
					ruleEdges.insert(edgeIdx);
				vertexIdx = dstVertex->index;
				return false;
			}
			else
			{
				vertexIdx = vertices.size();
				auto newVertex = std::shared_ptr<Vertex>(new Vertex(vertexIdx, op, shapeType));
				vertices.push_back(newVertex);
				visited.insert(std::make_pair(op->id, newVertex));
				if (vertexIdx > 0)
				{
					auto parVertex = vertices[parVertexIdx];
					auto edgeIdx = edges.size();
					auto newEdge = std::make_shared<Edge>(edgeIdx, parVertex, newVertex, succIdx, succParam, loop);
					parVertex->outgoingEdges.push_back(newEdge);
					newVertex->incomingEdges.push_back(newEdge);
					newVertex->dist_from_root = parVertex->dist_from_root + 1;
					edges.push_back(newEdge);
					if (ruleEdge)
						ruleEdges.insert(edgeIdx);
				}
				return true;
			}
		}

	}

}
