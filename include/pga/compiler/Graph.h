#pragma once

#include <pga/compiler/Edge.h>
#include <pga/compiler/GraphVisitor.h>
#include <pga/compiler/MatchGroupVisitor.h>
#include <pga/compiler/OperatorType.h>
#include <pga/compiler/Parameters.h>
#include <pga/compiler/PartitionVisitor.h>
#include <pga/compiler/ShapeType.h>
#include <pga/compiler/Vertex.h>

#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace PGA
{
	namespace Compiler
	{
		struct Graph
		{
			struct Partition
			{
				struct SubGraph
				{
					struct MatchResult
					{
						std::map<size_t, size_t> vertexMap;
						std::map<size_t, size_t> edgeMap;
						std::map<size_t, std::set<size_t>> vertexParametersMap;
						std::map<size_t, std::set<size_t>> vertexTermAttrMap;
						std::set<size_t> edgesWithMatchingParam;
						std::set<size_t> visited;

					private:
						MatchResult& operator+=(const MatchResult& other);

						friend struct SubGraph;

					};

					inline size_t size() const
					{
						return vertices.size();
					}

					int match(const SubGraph& other, MatchResult& matchResult, size_t i = 0, size_t j = 0) const;

				private:
					const Graph* source;
					std::vector<size_t> vertices;
					std::vector<std::vector<std::pair<size_t, size_t>>> adjacenciesList;

					SubGraph(const Graph* source);
					long vertexIndex(size_t i) const;
					bool hasCycle(size_t vertexIdx = 0, std::set<size_t> visited = std::set<size_t>()) const;
					bool hasEdge(size_t in, size_t out) const;
					static void increasingSequences(const std::vector<std::vector<size_t>>& all, std::vector<std::vector<size_t>>& sequences, size_t i = 0, std::vector<size_t> sequence = std::vector<size_t>());
					bool hasVertex(size_t vertexIdx) const;
					size_t addVertex(size_t vertexIdx);
					void makeEdge(size_t in, size_t out, size_t ref);

					friend struct Partition;
					friend struct Graph;

				};

				struct MatchGroups
				{
					struct MatchGroup
					{
						struct Match
						{
							size_t dstIdx;

						private:
							const MatchGroup* group;
							std::map<size_t, size_t> vertexMap;
							std::map<size_t, size_t> edgeMap;
							std::map<size_t, std::set<size_t>> vertexParametersMap;
							std::map<size_t, std::set<size_t>> vertexTermAttrsMap;
							std::set<size_t> edgesWithMatchingParam;

							Match(const MatchGroup* group,
								size_t dstIdx,
								const SubGraph::MatchResult& result);

							friend struct MatchGroup;

						};

						struct VertexDiff
						{
							void getEqParams(std::map<size_t, std::weak_ptr<Parameter>>& eqVertParams) const;
							void getDiffParams(size_t i, std::map<size_t, std::weak_ptr<Parameter>>& diffVertParams) const;
							void getDiffTermAttrs(size_t i, std::map<size_t, double>& diffTermAttrs) const;

						private:
							OperatorType opType;
							ShapeType shapeType;
							int phase;
							long genFuncIdx;
							size_t numParams;
							std::map<size_t, std::weak_ptr<Parameter>> eqParams;
							std::map<size_t, std::map<size_t, std::weak_ptr<Parameter>>> diffParams;
							size_t numTermAttrs;
							std::map<size_t, double> eqTermAttrs;
							std::map<size_t, std::map<size_t, double>> diffTermAttrs;
							size_t numEdges;

							VertexDiff(
								OperatorType opType,
								ShapeType shapeType,
								int phase,
								long genFuncIdx,
								size_t numParams,
								const std::map<size_t, std::weak_ptr<Parameter>>& eqParams,
								const std::map<size_t, std::map<size_t, std::weak_ptr<Parameter>>>& diffParams,
								size_t numTermAttrs,
								const std::map<size_t, double>& eqTermAttrs,
								const std::map<size_t, std::map<size_t, double>>& diffTermAttrs,
								size_t numEdges
								);

							friend struct MatchGroup;

						};

						struct EdgeDiff
						{
							bool requiresParam;
							bool hasDynParam;
							std::map<size_t, size_t> matchesMap;
							std::map<size_t, size_t> cutMap;
							std::map<size_t, std::weak_ptr<Parameter>> paramMap;
							bool common;

							EdgeDiff(
								const MatchGroup* source,
								bool requiresParam,
								bool hasDynParam,
								const std::map<size_t, size_t>& matchesMap,
								const std::map<size_t, size_t>& cutEdgesMap,
								const std::map<size_t, std::weak_ptr<Parameter>>& paramMap,
								bool common
								);

							inline bool isCut() const
							{
								return !cutMap.empty();
							}

						private:
							const MatchGroup* source;

						};

						size_t srcIdx;

						inline size_t size() const
						{
							return matchesMap.size();
						}

					private:
						struct EdgeMapValueComparer
						{
							EdgeMapValueComparer(size_t baseLine) : baseLine(baseLine) {}

							bool operator()(const std::pair<size_t, size_t>& entry)
							{
								return entry.second == baseLine;
							}

						private:
							size_t baseLine;

						};

						MatchGroups* source;
						std::map<size_t, Match> matchesMap;
						std::map<size_t, VertexDiff> verticesDiff;
						std::map<std::pair<size_t, size_t>, EdgeDiff> edgesDiff;

						MatchGroup(MatchGroups* source, size_t srcIdx);

						void addMatch(size_t dstIdx, const SubGraph::MatchResult& matchResult);
						void traverse(MatchGroupVisitor& visitor, size_t dstIdx, size_t srcVertexIdx, std::set<size_t>& visited) const;
						void computeDiffs();

						friend struct Partition;
						friend struct MatchGroups;

					};

					std::map<size_t, size_t> mapping;

					inline size_t size() const
					{
						return groups.size();
					}

					void traverse(MatchGroupVisitor& visitor, bool groupsOnly = true) const;

				private:
					const Partition* partition;
					std::map<size_t, MatchGroup> groups;

					MatchGroups(const Partition* partition);

					MatchGroup& addGroup(size_t srcIdx);
					void addMapping(size_t srcIdx, size_t dstIdx);
					void computeDiffs();

					friend struct Partition;
				};

				std::vector<SubGraph> subGraphs;
				std::unique_ptr<MatchGroups> matchGroups;

				inline void traverse(PartitionVisitor& visitor) const
				{
					std::set<size_t> visited;
					traverse(visitor, visited, 0, 0);
				}
				inline std::string getUid() const
				{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (uid.empty())
						throw std::runtime_error("uid.empty()");
#endif
					return uid;
				}
				inline size_t numMatches() const
				{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (matchGroups == nullptr)
						throw std::runtime_error("matchGroups == nullptr");
#endif
					size_t acc = 0;
					for (auto& entry : matchGroups->groups)
						if (entry.second.size() > 0)
							acc++;
					return acc;
				}
				bool isCutEdge(size_t edgeIdx) const;
				static std::string uidFromCutEdges(size_t numEdges, const std::set<size_t>& cutEdges);
				static std::set<size_t> cutEdgesFromUid(const std::string& partitionUid);

			private:
				struct TraversalHasher
				{
					size_t operator()(const std::pair<size_t, size_t> &p) const
					{
						return p.first ^ p.second;
					}

				};

				std::string uid;
				const Graph* source;
				std::map<size_t, size_t> cutEdges;
				std::unordered_map<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>, TraversalHasher> traversals;

				Partition(const Graph* source);
				void computeUid();
				void computeMatchGroups();
				void createEmptyMatchGroups();
				bool violatesR1() const;
				bool violatesR3() const;
				size_t newSubgraph();
				void addTraversal(size_t subgraphIndex, size_t vertexIndex, size_t newSubgraphIndex, size_t edgeIndex);
				void traverse(PartitionVisitor& visitor, std::set<size_t>& visited, size_t g, size_t i) const;
				friend struct Graph;

			};

			typedef std::unique_ptr<Partition> PartitionPtr;

			struct ComputePartitionCallback
			{
				virtual bool operator()(size_t i, PartitionPtr& arg) = 0;

			};

			Graph() = default;
			Graph(const Graph& other);
			Graph(std::vector<std::shared_ptr<Vertex>>& vertices, std::vector<std::shared_ptr<Edge>>& edges);

			void computePhases(std::set<size_t>& alwaysCutEdges);
			bool computePartition(ComputePartitionCallback& callback, bool matchGroups, const std::set<size_t>& cutEdgesIdxs) const;
			void computePartitions(ComputePartitionCallback& callback, bool matchGroups, const std::set<size_t>& alwaysCutEdges = std::set<size_t>(), const std::set<size_t>& neverCutEdges = std::set<size_t>(), const std::map<size_t, size_t>& replacementMapping = std::map<size_t, size_t>(), const std::vector<std::set<size_t>>& replacementGroups = std::vector<std::set<size_t>>()) const;
			void findInVertices(size_t i, std::vector<size_t>& ins) const;
			void findInEdges(size_t i, std::set<size_t>& ins) const;
			inline bool depthFirst(GraphVisitor& visitor, size_t vertexIdx = 0) const
			{
				std::set<size_t> visited, ancestry;
				return depthFirst(visitor, vertexIdx, visited, ancestry);
			}
			bool breadthFirst(GraphVisitor& visitor) const;
			bool operator == (const Graph& other) const;
			void findNeverCutEdges(std::set<size_t>& neverCutEdges, const std::set<size_t>& alwaysCutEdges = std::set<size_t>()) const;
			void findAlwaysCutEdges(std::set<size_t>& alwaysCutEdges, const std::set<size_t>& neverCutEdges = std::set<size_t>()) const;
			void getParents(std::map<size_t, std::set<size_t>>& parents) const;
			inline size_t numEdges() const
			{
				return edges.size();
			}
			inline size_t edgeOut(size_t edgeIdx) const
			{
				return edges[edgeIdx].out;
			}
			inline size_t numVertices() const
			{
				return vertices.size();
			}

		private:
			template <bool In>
			struct AdjacencyComparer
			{
				AdjacencyComparer(size_t baseLine) : baseLine(baseLine) {}

				bool operator() (const std::pair<size_t, size_t>& arg)
				{
					if (In)
						return arg.first == baseLine;
					else
						return arg.second == baseLine;
				}

			private:
				size_t baseLine;

			};

#ifdef EDGES_INTERDEPENDENCIES
			struct FindEdgesInterdependencies : GraphVisitor
			{
				std::vector<std::set<size_t>>& edgesInterdependencies;
				std::map<size_t, size_t> vertexToGroupIdxs;

				FindEdgesInterdependencies(std::vector<std::set<size_t>>& edgesInterdependencies) : edgesInterdependencies(edgesInterdependencies) {}

				virtual void visit(size_t i, const PGA::Compiler::Vertex_LW& vertex)
				{
				}

				virtual void visit(size_t i, const PGA::Compiler::Edge_LW& edge, const PGA::Compiler::Vertex_LW& src, const PGA::Compiler::Vertex_LW& dst)
				{
					if (dst.numIncomingEdges() < 2)
						return;

					auto it = vertexToGroupIdxs.find(dst);
					size_t j;
					if (it == vertexToGroupIdxs.end())
					{
						j = edgesInterdependencies.size();
						vertexToGroupIdxs[dst] = j;
						edgesInterdependencies.emplace_back();
					}
					else
						j = it->second;
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (edgesInterdependencies[j].find(i) != edgesInterdependencies[j].end())
						throw std::runtime_error("repeated interdependent edge idx");
#endif
					edgesInterdependencies[j].insert(i);
				}

			};
#endif

			struct FindSingleParentDiscardEdges : GraphVisitor
			{
				std::set<size_t>& neverCutEdges;
				const std::set<size_t>& alwaysCutEdges;

				FindSingleParentDiscardEdges(std::set<size_t>& neverCutEdges, const std::set<size_t>& alwaysCutEdges = std::set<size_t>()) : neverCutEdges(neverCutEdges), alwaysCutEdges(alwaysCutEdges) {}

				virtual bool visit(size_t i, const PGA::Compiler::Vertex_LW& vertex)
				{
					return true;
				}

				virtual bool visit(size_t i, const PGA::Compiler::Edge_LW& edge, const PGA::Compiler::Vertex_LW& src, const PGA::Compiler::Vertex_LW& dst)
				{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (src.operatorType() == PGA::Compiler::GENERATE ||
						src.operatorType() == PGA::Compiler::DISCARD)
						throw std::runtime_error("invalid op. type for src");
#endif
					if (dst.operatorType() == PGA::Compiler::DISCARD)
					{
						if (dst.numIncomingEdges() > 1 && dst.hasOtherParent(src))
							return true;

						if (alwaysCutEdges.find(i) != alwaysCutEdges.end())
						{
							//std::cout << "there's a conflicting heuristic considering an edge leading to a single-parent discard as an always-cut edge (edgeIdx=" << i << ")" << std::endl;
							return true;
						}
						neverCutEdges.insert(i);
					}
					return true;
				}

			};

			struct GetParents : GraphVisitor
			{
				std::map<size_t, std::set<size_t>>& parents;

				GetParents(std::map<size_t, std::set<size_t>>& parents) : parents(parents) {}

				virtual bool visit(size_t i, const PGA::Compiler::Vertex_LW& vertex)
				{
					return true;
				}

				virtual bool visit(size_t i, const PGA::Compiler::Edge_LW& edge, const PGA::Compiler::Vertex_LW& src, const PGA::Compiler::Vertex_LW& dst)
				{
					parents[dst].insert(src);
					return true;
				}

			};

			struct FindCycleEdges : GraphVisitor
			{
				const std::set<size_t>& neverCutEdges;
				std::set<size_t>& alwaysCutEdges;

				FindCycleEdges(const std::set<size_t>& neverCutEdges, std::set<size_t>& alwaysCutEdges = std::set<size_t>()) : neverCutEdges(neverCutEdges), alwaysCutEdges(alwaysCutEdges) {}

				virtual bool visit(size_t i, const PGA::Compiler::Vertex_LW& vertex)
				{
					return true;
				}

				virtual bool visit(size_t i, const PGA::Compiler::Edge_LW& edge, const PGA::Compiler::Vertex_LW& src, const PGA::Compiler::Vertex_LW& dst)
				{
					return true;
				}

				virtual void visitCycle(size_t i, const PGA::Compiler::Edge_LW& edge)
				{
					if (neverCutEdges.find(i) != neverCutEdges.end())
					{
						throw std::runtime_error("loop edge set as a never-cut edge");
					}
					alwaysCutEdges.insert(i);
				}

			};

			struct CollectVertices : GraphVisitor
			{
				std::set<size_t>& vertices;

				CollectVertices(std::set<size_t>& vertices) : vertices(vertices) {}

				virtual bool visit(size_t i, const PGA::Compiler::Vertex_LW& vertex)
				{
					vertices.insert(vertex);
					return true;
				}

				virtual bool visit(size_t i, const PGA::Compiler::Edge_LW& edge, const PGA::Compiler::Vertex_LW& src, const PGA::Compiler::Vertex_LW& dst)
				{
					return true;
				}

			};

			std::vector<Vertex_LW> vertices;
			std::vector<Edge_LW> edges;
			std::vector<std::vector<std::pair<size_t, size_t>>> adjacenciesList;

			bool depthFirst(GraphVisitor& visitor, size_t i, std::set<size_t>& visited, std::set<size_t> ancestry) const;
			size_t addVertex(const Vertex_LW& vertex);
			size_t computePartition(const std::set<size_t>& cutEdgesIdxs, PartitionPtr& partition, size_t subgraphIndex, size_t vertexIndex, size_t edgeIdx, std::map<size_t, size_t>& visited, std::map<size_t, size_t> ancestry) const;

		};

	}

}
