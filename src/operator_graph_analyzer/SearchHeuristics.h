#pragma once

#include <pga/compiler/Graph.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <map>
#include <set>
#include <vector>

namespace SearchHeuristics
{
	struct OneToOneOperatorEdgesHeuristic : PGA::Compiler::GraphVisitor
	{
		std::set<size_t>& alwaysCutEdges;
		std::set<size_t>& neverCutEdges;

		OneToOneOperatorEdgesHeuristic(std::set<size_t>& alwaysCutEdges, std::set<size_t>& neverCutEdges) : alwaysCutEdges(alwaysCutEdges), neverCutEdges(neverCutEdges) {}

		virtual bool visit(size_t i, const PGA::Compiler::Vertex_LW& vertex)
		{
			return true;
		}

		virtual bool visit(size_t i, const PGA::Compiler::Edge_LW& edge, const PGA::Compiler::Vertex_LW& src, const PGA::Compiler::Vertex_LW& dst)
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (isTerminal(src.operatorType()))
				throw std::runtime_error("src can never be a terminal op.");
#endif
			auto dstOpType = dst.operatorType();
			if (is1To1(src.operatorType()) &&
				dst.numIncomingEdges() == 1 &&
				(is1To1(dstOpType) || dstOpType == PGA::Compiler::GENERATE))
			{
				if (alwaysCutEdges.find(i) != alwaysCutEdges.end())
				{
#ifdef PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE
					std::cout << "partitioning has one or more conflicting heuristics for edge " << i << " (1to1)" << std::endl;
#endif
					return true;
				}
				neverCutEdges.insert(i);
			}
			return true;
		}

	private:
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
		inline bool isTerminal(PGA::Compiler::OperatorType opType) const
		{
			return opType == PGA::Compiler::GENERATE ||
				opType == PGA::Compiler::DISCARD;
		}
#endif

		inline bool is1To1(PGA::Compiler::OperatorType opType) const
		{
			return opType == PGA::Compiler::ROTATE ||
				opType == PGA::Compiler::TRANSLATE ||
				opType == PGA::Compiler::SCALE ||
				opType == PGA::Compiler::EXTRUDE ||
				opType == PGA::Compiler::SET_AS_DYNAMIC_CONVEX_POLYGON ||
				opType == PGA::Compiler::SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM || 
				opType == PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_POLYGON || 
				opType == PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_POLYGON;
		}

	};

	struct SubtreeOrderHeuristic : PGA::Compiler::GraphVisitor
	{
		struct Node
		{
			std::map<size_t, std::weak_ptr<Node>> parents;
			size_t src;
			std::map<size_t, std::weak_ptr<Node>> children;

			Node(size_t src) : src(src) {}
			~Node() {}

			size_t order() const
			{
				return order_phase1();
			}

			void append(size_t edgeIdx, std::weak_ptr<Node> node)
			{
				children.emplace(std::make_pair(edgeIdx, node));
			}

			bool operator == (const Node& node) const
			{
				return src == node.src;
			}

			bool operator != (const Node& node) const
			{
				return !operator==(node);
			}

			void rollback(std::vector<std::shared_ptr<Node>>& nodes,
				std::set<size_t>& neverCutEdges,
				std::set<size_t>& newNeverCutEdges)
			{
				rollback_phase1(neverCutEdges, newNeverCutEdges);
				rollback_phase2(nodes, neverCutEdges, newNeverCutEdges);
			}

			bool hasCommonAncestry(std::weak_ptr<Node> other) const
			{
				auto o = other.lock();
				if (src == o->src)
					return true;
				if (parents.empty())
				{
					for (auto& otherParent : o->parents)
					{
						if (hasCommonAncestry(otherParent.second))
							return true;
					}
				}
				else
				{
					for (auto& parent : parents)
					{
						for (auto& otherParent : o->parents)
							if (parent.second.lock()->hasCommonAncestry(otherParent.second))
								return true;
					}
				}
				return false;
			}

			bool addParent(size_t edgeIdx, std::weak_ptr<Node> newParent)
			{
				auto p0 = newParent.lock();
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (p0 == nullptr)
					throw std::runtime_error("p0 == nullptr");
#endif
				for (auto& parent : parents)
				{
					auto p1 = parent.second.lock();
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (p1 == nullptr)
						throw std::runtime_error("p1 == nullptr");
#endif
					if (p0 == p1)
						return false;
				}
				parents.emplace(std::make_pair(edgeIdx, newParent));
				return true;
			}

		private:
			bool isParent(const Node* node) const
			{
				for (auto& parent : parents)
					if (*parent.second.lock().get() == *node)
						return true;
				return false;
			}

			size_t order_phase1() const
			{
				if (parents.empty())
					return order_phase2();
				size_t acc = 0;
				for (auto& parent : parents)
					acc += parent.second.lock()->order_phase1();
				return acc;
			}

			size_t order_phase2(std::set<size_t>& visited) const
			{
				if (visited.find(src) != visited.end())
					return 0;
				visited.insert(src);
				size_t acc = 1;
				for (auto& conn : children)
					acc += conn.second.lock()->order_phase2(visited);
				return acc;
			}

			size_t order_phase2() const
			{
				std::set<size_t> visited;
				return order_phase2(visited);
			}

			void rollback_phase2(std::vector<std::shared_ptr<Node>>& nodes,
				std::set<size_t>& neverCutEdges,
				std::set<size_t>& newNeverCutEdges,
				long connIdx,
				std::set<size_t>& visited)
			{
				if (visited.find(src) != visited.end())
					return;

				visited.insert(src);

				auto it1 = std::find_if(nodes.begin(), nodes.end(), NodeComparer(src));

#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it1 == nodes.end())
					throw std::runtime_error("it1 == nodes.end()");
#endif

				nodes.erase(it1);

				if (connIdx != -1)
				{
					auto edgeIdx = static_cast<size_t>(connIdx);
					auto it1 = newNeverCutEdges.find(edgeIdx);
					if (it1 != newNeverCutEdges.end())
					{
						newNeverCutEdges.erase(it1);
						auto it3 = neverCutEdges.find(edgeIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (it3 == neverCutEdges.end())
							throw std::runtime_error("it3 == neverCutEdges.end()");
#endif
						neverCutEdges.erase(it3);
					}
				}

				for (auto& conn : children)
					conn.second.lock()->rollback_phase2(nodes, neverCutEdges, newNeverCutEdges, static_cast<long>(conn.first), visited);
			}

			void rollback_phase2(std::vector<std::shared_ptr<Node>>& nodes,
							std::set<size_t>& neverCutEdges,
							std::set<size_t>& newNeverCutEdges,
							long connIdx = -1)
			{
				std::set<size_t> visited;
				rollback_phase2(nodes, neverCutEdges, newNeverCutEdges, connIdx, visited);
			}

			void rollback_phase1(std::set<size_t>& neverCutEdges, std::set<size_t>& newNeverCutEdges)
			{
				for (auto& parent : parents)
				{
					auto p0 = parent.second.lock();
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (p0 == nullptr)
						throw std::runtime_error("p0 == nullptr");
#endif
					auto edgeIdx = parent.first;
					auto it1 = p0->children.find(edgeIdx);
					p0->children.erase(it1);
					auto it2 = newNeverCutEdges.find(edgeIdx);
					if (it2 != newNeverCutEdges.end())
					{
						newNeverCutEdges.erase(it2);
						auto it3 = neverCutEdges.find(edgeIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (it3 == neverCutEdges.end())
							throw std::runtime_error("it3 == neverCutEdges.end()");
#endif
						neverCutEdges.erase(it3);
					}
				}
			}

		};

		struct NodeComparer
		{
			NodeComparer(size_t baseLine) : baseLine(baseLine) {}

			bool operator()(const std::shared_ptr<Node>& a) const
			{
				return a->src == baseLine;
			}

		private:
			size_t baseLine;

		};

		size_t maxSubtreeOrder;
		std::set<size_t>& alwaysCutEdges;
		std::set<size_t>& neverCutEdges;
		std::set<size_t> newNeverCutEdges;
		std::vector<std::shared_ptr<Node>> nodes;

		SubtreeOrderHeuristic(size_t maxSubtreeOrder, std::set<size_t>& alwaysCutEdges, std::set<size_t>& neverCutEdges) : maxSubtreeOrder(maxSubtreeOrder), alwaysCutEdges(alwaysCutEdges), neverCutEdges(neverCutEdges) {}

		virtual bool visit(size_t i, const PGA::Compiler::Vertex_LW& vertex)
		{
			return true;
		}

		virtual bool visit(size_t i, const PGA::Compiler::Edge_LW& edge, const PGA::Compiler::Vertex_LW& src, const PGA::Compiler::Vertex_LW& dst)
		{
			auto it1 = std::find_if(nodes.begin(), nodes.end(), NodeComparer(src));
			if (it1 != nodes.end())
			{
				tryToAppendNode(*it1, src, dst, i);
				return true;
			}

			if (src.operatorType() != PGA::Compiler::SUBDIV &&
				src.operatorType() != PGA::Compiler::COMPSPLIT &&
				src.operatorType() != PGA::Compiler::IFSIZELESS)
				return true;

			if (alwaysCutEdges.find(i) != alwaysCutEdges.end())
			{
#ifdef PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE
				std::cout << "partitioning has one or more conflicting heuristics for edge " << i << " (sto)" << std::endl;
#endif
				return true;
			}

			tryToAppendNode(createNode(src), src, dst, i);
			return true;
		}

		void tryToAppendNode(std::shared_ptr<Node> srcSubtree, const PGA::Compiler::Vertex_LW& src, const PGA::Compiler::Vertex_LW& dst, size_t i)
		{
			std::shared_ptr<Node> dstSubtree;
			bool hasOtherParent = dst.hasOtherParent(src);
			auto it1 = std::find_if(nodes.begin(), nodes.end(), NodeComparer(dst));
			if (it1 != nodes.end())
			{
				dstSubtree = *it1;
				if (hasOtherParent)
				{
					if (srcSubtree->hasCommonAncestry(dstSubtree))
					{
						dstSubtree->addParent(i, static_cast<std::weak_ptr<Node>>(srcSubtree));
						srcSubtree->append(i, static_cast<std::weak_ptr<Node>>(dstSubtree));
						if (neverCutEdges.find(i) == neverCutEdges.end())
							newNeverCutEdges.insert(i);
						neverCutEdges.insert(i);
#if PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE
						std::cout << "added " << static_cast<size_t>(dst) << " to " << static_cast<size_t>(src) << " subtree again [edgeIdx=" << i << "] (sto)" << std::endl;
#endif
					}
					else
					{
						dstSubtree->rollback(nodes, neverCutEdges, newNeverCutEdges);
#if PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE
						std::cout << "rolled back " << static_cast<size_t>(dst) << " subtree! [edgeIdx=" << i << "] (sto)" << std::endl;
#endif
						return;
					}
				}
			}

			if (srcSubtree->order() < maxSubtreeOrder)
			{
				if (hasOtherParent)
				{
					std::set<size_t> dstParents;
					dst.getParents(dstParents);
					for (auto& dstParent : dstParents)
					{
						if (dstParent == src) continue;
						auto it2 = std::find_if(nodes.begin(), nodes.end(), NodeComparer(dstParent));
						if (it2 == nodes.end())
							return;
						auto dstParentSubtree = *it2;
						if (!srcSubtree->hasCommonAncestry(dstParentSubtree))
							return;
					}
				}

				if (dstSubtree == nullptr)
					dstSubtree = createNode(dst);
				dstSubtree->addParent(i, static_cast<std::weak_ptr<Node>>(srcSubtree));
				srcSubtree->append(i, static_cast<std::weak_ptr<Node>>(dstSubtree));
				if (neverCutEdges.find(i) == neverCutEdges.end())
					newNeverCutEdges.insert(i);
				neverCutEdges.insert(i);
#if PGA_OPERATOR_GRAPH_ANALYZER_VERBOSE
				std::cout << "added " << static_cast<size_t>(dst) << " to " << static_cast<size_t>(src) << " subtree [edgeIdx=" << i << "] (sto)" << std::endl;
#endif
			}
		}

		std::shared_ptr<Node> createNode(const PGA::Compiler::Vertex_LW& src)
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			auto it1 = std::find_if(nodes.begin(), nodes.end(), NodeComparer(src));
			if (it1 != nodes.end())
				throw std::runtime_error("it1 != nodes.end()");
#endif
			auto nodeIdx = nodes.size();
			nodes.emplace_back(new Node(src));
			return nodes[nodeIdx];
		}

	};

	struct RandomRuleHeuristic : PGA::Compiler::GraphVisitor
	{
		std::set<size_t>& alwaysCutEdges;

		RandomRuleHeuristic(std::set<size_t>& alwaysCutEdges) : alwaysCutEdges(alwaysCutEdges) {}

		virtual bool visit(size_t i, const PGA::Compiler::Vertex_LW& vertex)
		{
			return true;
		}

		virtual bool visit(size_t i, const PGA::Compiler::Edge_LW& edge, const PGA::Compiler::Vertex_LW& src, const PGA::Compiler::Vertex_LW& dst)
		{
			if (src.operatorType() != PGA::Compiler::STOCHASTIC)
				return true;
			alwaysCutEdges.insert(i);
			return true;
		}

	};

}
