#include <pga/compiler/Graph.h>
#include <pga/compiler/Operator.h>
#include <pga/compiler/PhaseVisitor.h>

#ifdef VERBOSE
#include <chrono>
#endif
#include <algorithm>
#include <iterator>
#include <deque>
#include <tuple>
#ifdef _DEBUG
#include <initializer_list>
#endif

namespace PGA
{
	namespace Compiler
	{
		//////////////////////////////////////////////////////////////////////////
		Graph::Graph(const Graph& other) : vertices(other.vertices.begin(), other.vertices.end()), edges(other.edges.begin(), other.edges.end()), adjacenciesList(other.adjacenciesList.begin(), other.adjacenciesList.end())
		{
		}

		Graph::Graph(std::vector<std::shared_ptr<Vertex>>& vertices, std::vector<std::shared_ptr<Edge>>& edges)
		{
			for (auto& vertex : vertices)
				this->vertices.emplace_back((std::weak_ptr<Vertex>)vertex);
			adjacenciesList.resize(vertices.size());
			for (auto& edge : edges)
			{
				auto it = std::find(vertices.begin(), vertices.end(), edge->srcVertex);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it == vertices.end())
					throw std::runtime_error("it == vertices.end()");
#endif
				auto in = std::distance(vertices.begin(), it);
				it = std::find(vertices.begin(), vertices.end(), edge->dstVertex);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it == vertices.end())
					throw std::runtime_error("it == vertices.end()");
#endif
				auto out = std::distance(vertices.begin(), it);
				adjacenciesList[in].emplace_back(out, this->edges.size());
				this->edges.emplace_back((std::weak_ptr<Edge>)edge, in, out);
			}
		}

		bool Graph::operator==(const Graph& other) const
		{
			if (vertices.size() != other.vertices.size()) return false;
			if (adjacenciesList.size() != other.adjacenciesList.size()) return false;
			for (size_t i = 0; i < vertices.size(); i++)
			{
				auto it1 = std::find(other.vertices.begin(), other.vertices.end(), vertices[i]);
				if (it1 == other.vertices.end())
					return false;
				size_t j = (size_t)std::distance(other.vertices.begin(), it1);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (j > other.vertices.size())
					throw std::runtime_error("j > other.vertices.size()");
#endif
				auto& otherAdjacencies = other.adjacenciesList[j];
				for (auto& adjacency : adjacenciesList[i])
				{
					auto it2 = std::find(other.vertices.begin(), other.vertices.end(), vertices[adjacency.first]);
					if (it2 == other.vertices.end())
						return false;
					size_t k = (size_t)std::distance(other.vertices.begin(), it2);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (k > other.vertices.size())
						throw std::runtime_error("k > other.vertices.size()");
#endif
					if (std::find_if(otherAdjacencies.begin(), otherAdjacencies.end(), AdjacencyComparer<true>(k)) == otherAdjacencies.end())
						return false;
				}
			}
			return true;
		}

		size_t Graph::addVertex(const Vertex_LW& vertex)
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (std::find(vertices.begin(), vertices.end(), vertex) != vertices.end())
				throw std::runtime_error("std::find(vertices.begin(), vertices.end(), vertex) != vertices.end()");
#endif
			auto i = vertices.size();
			vertices.push_back(vertex);
			adjacenciesList.resize(vertices.size());
			return i;
		}

		bool Graph::depthFirst(GraphVisitor& visitor, size_t i, std::set<size_t>& visited, std::set<size_t> ancestry) const
		{
			if (!visitor.visit(i, vertices[i]))
				return false;
			visited.insert(i);
			ancestry.insert(i);
			for (auto& adjacency : adjacenciesList[i])
			{
				auto& edge = edges[adjacency.second];
				if (!visitor.visit(adjacency.second, edge, vertices[edge.in], vertices[edge.out]))
					return false;
				if (visited.find(adjacency.first) != visited.end())
				{
					if (ancestry.find(adjacency.first) != ancestry.end())
						visitor.visitCycle(adjacency.second, edge);
					continue;
				}
				if (!depthFirst(visitor, adjacency.first, visited, ancestry))
					return false;
			}
			return true;
		}

		bool Graph::breadthFirst(GraphVisitor& visitor) const
		{
			std::deque<std::tuple<long, long, long>> stack;
			stack.emplace_back(-1, 0, -1);
			std::set<size_t> visited;
			while (!stack.empty())
			{
				auto curr = stack.front();
				stack.pop_front();
				auto srcVertexIdx = std::get<0>(curr);
				auto dstVertexIdx = std::get<1>(curr);
				auto edgeIdx = std::get<2>(curr);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (dstVertexIdx == -1)
					throw std::runtime_error("dstVertexIdx == -1");
#endif
				if (edgeIdx != -1)
				{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (srcVertexIdx == -1)
						throw std::runtime_error("srcVertexIdx == -1");
#endif
					if (!visitor.visit(static_cast<size_t>(edgeIdx), edges[edgeIdx], vertices[srcVertexIdx], vertices[dstVertexIdx]))
						return false;
				}
				if (!visitor.visit(dstVertexIdx, vertices[dstVertexIdx]))
					return false;
				visited.insert(dstVertexIdx);
				for (auto& adjacency : adjacenciesList[dstVertexIdx])
				{
					if (visited.find(adjacency.first) != visited.end())
						continue;
					stack.emplace_back(dstVertexIdx, static_cast<long>(adjacency.first), static_cast<long>(adjacency.second));
				}
			}
			return true;
		}

		void Graph::computePhases(std::set<size_t>& alwaysCutEdges)
		{
			PhaseVisitor pv;
			this->depthFirst(pv);

			if (vertices.at(0).getOperator()->type == OperatorType::IFCOLLIDES)
				throw std::runtime_error("first operator can not be an IfCollides");

			int predecessor_phase = 0;

			std::deque<std::tuple<long, long, long>> stack;
			std::deque<std::tuple<long, long, long>> waiting;
			std::set<size_t> visited;
			std::map<long, unsigned int> pop_count;
			std::vector<long> unreachable;

			stack.emplace_back(-1, 0, -1);
			while (!stack.empty())
			{
				auto curr = stack.front();
				stack.pop_front();
				auto srcVertexIdx = std::get<0>(curr);
				auto dstVertexIdx = std::get<1>(curr);
				auto edgeIdx = std::get<2>(curr);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (dstVertexIdx == -1)
					throw std::runtime_error("dstVertexIdx == -1");
#endif

				Operator* op = vertices.at(dstVertexIdx).getOperator();

				if (srcVertexIdx != -1)
					predecessor_phase = vertices.at(srcVertexIdx).getOperator()->phase;
				else
					predecessor_phase = 0;

				if (op->type == OperatorType::IFCOLLIDES)
				{
					unsigned int collider_id = static_cast<unsigned int>(op->operatorParams.at(0).get()->at(0));
					std::vector<std::tuple<long, long, long>> unvisited;
					int phase_max = -1;
					if (pv.colliders.count(collider_id) != 0)
					{
						for (auto it = pv.colliders.at(collider_id).begin(); it != pv.colliders.at(collider_id).end(); ++it)
						{
							phase_max = std::max(phase_max, it->getOperator()->phase);
							if (visited.find(*it) == visited.end())
								unvisited.emplace_back(0L, static_cast<long>(static_cast<int>(*it)), -1L);
						}
						if (unvisited.size() > 0)
						{
							if (pop_count.count(dstVertexIdx) < 1)
								waiting.emplace_back(srcVertexIdx, dstVertexIdx, edgeIdx);
							else
								unreachable.push_back(dstVertexIdx);

							if (stack.empty())
							{
								if (waiting.size() > 0)
								{
									auto waitingVertex = waiting.front();
									waiting.pop_front();
									auto srcVertexIdx = std::get<0>(waitingVertex);
									auto dstVertexIdx = std::get<1>(waitingVertex);
									auto edgeIdx = std::get<2>(waitingVertex);
									if (pop_count.count(dstVertexIdx) == 0)
										pop_count.insert(std::make_pair(dstVertexIdx, 1));
									else
										pop_count.at(dstVertexIdx)++;
									stack.emplace_front(srcVertexIdx, dstVertexIdx, edgeIdx);
								}
							}
							continue;
						}
						else
						{
							op->phase = phase_max + 1;
							alwaysCutEdges.insert(edgeIdx);
						}
					}
					else
					{
						std::cout << "Warning: No collider for IfCollides(" << collider_id << ")" << std::endl;
					}

				}
				else
				{
					op->phase = predecessor_phase;
				}

#ifdef VERBOSE
				//if (op->type == OperatorType::IFCOLLIDES || op->type == OperatorType::COLLIDER)
				//	std::cout << "idx=" << size_t(vertices.at(dstVertexIdx)) << " name=" << PGA::Compiler::toString(op->type) 
				//	<< " ID=" << static_cast<unsigned int>(op->operatorParams.at(0).get()->at(0)) << " phase=" << op->phase << std::endl;
				//else
				//	std::cout << "idx=" << size_t(vertices.at(dstVertexIdx)) << " name=" << PGA::Compiler::toString(op->type) << " phase=" << op->phase << std::endl;
#endif

				visited.insert(dstVertexIdx);
				for (auto& adjacency : adjacenciesList[dstVertexIdx])
				{
					if (visited.find(adjacency.first) != visited.end())
						continue;
					stack.emplace_front(dstVertexIdx, static_cast<long>(adjacency.first), static_cast<long>(adjacency.second));
				}

				if (stack.empty())
				{
					if (waiting.size() > 0)
					{
						auto waitingVertex = waiting.front();
						waiting.pop_front();
						auto srcVertexIdx = std::get<0>(waitingVertex);
						auto dstVertexIdx = std::get<1>(waitingVertex);
						auto edgeIdx = std::get<2>(waitingVertex);
						if (pop_count.count(dstVertexIdx) == 0)
							pop_count.insert(std::make_pair(dstVertexIdx, 1));
						else
							pop_count.at(dstVertexIdx)++;
						stack.emplace_front(srcVertexIdx, dstVertexIdx, edgeIdx);
					}
				}
			}

#ifdef VERBOSE
			if (unreachable.size() > 0)
			{
				std::cout << "Error: IfCollides with ID(s) ";
				for (auto x : unreachable)
				{
					std::cout << static_cast<unsigned int>(vertices.at(x).getOperator()->operatorParams.at(0).get()->at(0)) << " ";
				}
				std::cout << "have unresolvable dependencies" << std::endl;
			}
#endif
		}

		size_t Graph::computePartition(const std::set<size_t>& cutEdgesIdxs, PartitionPtr& partition, size_t subgraphIndex, size_t vertexIndex, size_t edgeIdx, std::map<size_t, size_t>& visited, std::map<size_t, size_t> ancestry) const
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (visited.find(vertexIndex) != visited.end())
				throw std::runtime_error("visited.find(i) != visited.end()");

			if (ancestry.find(vertexIndex) != ancestry.end())
				throw std::runtime_error("ancestry.find(i) != ancestry.end()");
#endif
			auto sgVertexIndex = partition->subGraphs[subgraphIndex].addVertex(vertexIndex);
			visited[vertexIndex] = subgraphIndex;
			ancestry[vertexIndex] = subgraphIndex;
			for (auto& adjacency : adjacenciesList[vertexIndex])
			{
				auto adjacentVertexIndex = adjacency.first;
				auto adjacentEdgeIndex = adjacency.second;

				auto it1 = ancestry.find(adjacency.first);
				if (it1 != ancestry.end())
				{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (visited.find(adjacency.first) == visited.end())
						throw std::runtime_error("visited.find(adjacency.first) == visited.end()");
#endif
					partition->addTraversal(subgraphIndex, sgVertexIndex, it1->second, adjacentEdgeIndex);
					edgeIdx++;
					continue;
				}

				auto it2 = visited.find(adjacency.first);
				if (it2 != visited.end())
				{
					if (it2->second != subgraphIndex)
					{
						partition->addTraversal(subgraphIndex, sgVertexIndex, it2->second, adjacentEdgeIndex);
					}
					else
					{
						auto otherSgVertexIndex = partition->subGraphs[subgraphIndex].vertexIndex(adjacentVertexIndex);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (otherSgVertexIndex == -1)
							throw std::runtime_error("otherSgVertexIndex == -1");
#endif
						partition->subGraphs[subgraphIndex].makeEdge(sgVertexIndex, static_cast<size_t>(otherSgVertexIndex), adjacentEdgeIndex);
					}
					edgeIdx++;
					continue;
				}

				auto prevSgVertexIndex = partition->subGraphs[subgraphIndex].vertices.size();
				if (cutEdgesIdxs.find(edgeIdx) != cutEdgesIdxs.end())
				{
					auto newSubgraphIndex = partition->newSubgraph();
					partition->addTraversal(subgraphIndex, sgVertexIndex, newSubgraphIndex, adjacentEdgeIndex);
					edgeIdx = computePartition(cutEdgesIdxs, partition, newSubgraphIndex, adjacentVertexIndex, edgeIdx + 1, visited, ancestry);
				}
				else
				{
					edgeIdx = computePartition(cutEdgesIdxs, partition, subgraphIndex, adjacentVertexIndex, edgeIdx + 1, visited, ancestry);
					partition->subGraphs[subgraphIndex].makeEdge(sgVertexIndex, prevSgVertexIndex, adjacentEdgeIndex);
				}
			}
			return edgeIdx;
		}

		void Graph::findInVertices(size_t i, std::vector<size_t>& ins) const
		{
			for (size_t j = 0; j < adjacenciesList.size(); j++)
			{
				if (i == j) continue;
				for (auto& adjacency : adjacenciesList[j])
				{
					if (adjacency.first == i)
					{
						ins.push_back(j);
						break;
					}
				}
			}
		}

		void Graph::findInEdges(size_t i, std::set<size_t>& ins) const
		{
			for (size_t j = 0; j < adjacenciesList.size(); j++)
			{
				if (i == j) continue;
				for (auto& adjacency : adjacenciesList[j])
				{
					if (adjacency.first == i)
					{
						ins.insert(adjacency.second);
						break;
					}
				}
			}
		}

		void Graph::computePartitions(ComputePartitionCallback& callback, bool matchGroups, const std::set<size_t>& pAlwaysCutEdges, const std::set<size_t>& pNeverCutEdges, const std::map<size_t, size_t>& replacementMapping, const std::vector<std::set<size_t>>& replacementGroups) const
		{
			std::set<std::string> uids;
			std::vector<size_t> idxs;
#ifdef VERBOSE
			std::set<std::string> r1Violators;
			std::set<std::string> r3Violators;
			size_t numRepeatedPartitions = 0;
#endif
			std::set<size_t> alwaysCutEdges(pAlwaysCutEdges.begin(), pAlwaysCutEdges.end());
			std::set<size_t> neverCutEdges(pNeverCutEdges.begin(), pNeverCutEdges.end());
			findAlwaysCutEdges(alwaysCutEdges, neverCutEdges);
			findNeverCutEdges(neverCutEdges, alwaysCutEdges);
			for (auto i = 0; i < edges.size(); i++)
			{
				if (neverCutEdges.find(i) != neverCutEdges.end()) continue;
				if (alwaysCutEdges.find(i) != alwaysCutEdges.end()) continue;
				idxs.push_back(i);
			}
			long numIdxs = static_cast<long>(idxs.size());
			std::vector<long> stack(numIdxs + 1);
			size_t pos;
			stack[0] = -1;
			pos = 0;
#ifdef VERBOSE
			std::cout << "no. of edges indices: " << numIdxs << std::endl;
			auto start = std::chrono::system_clock::now();
#endif
			while (true)
			{
				if (stack[pos] < numIdxs - 1)
				{
					stack[pos + 1] = stack[pos] + 1;
					pos++;
				}
				else
					++stack[--pos];

				if (pos == 0)
					break;

				std::set<size_t> cutEdgesIdxs(alwaysCutEdges.begin(), alwaysCutEdges.end());
				for (auto i = 1; i <= pos; i++)
					cutEdgesIdxs.insert(idxs[stack[i]]);

				PartitionPtr newPartition(new Partition(this));
				auto subgraphIndex = newPartition->newSubgraph();
				std::map<size_t, size_t> visited;
				std::map<size_t, size_t> ancestry;
				computePartition(cutEdgesIdxs, newPartition, subgraphIndex, 0, 0, visited, ancestry);
				if (newPartition->violatesR1())
				{
#ifdef VERBOSE
					newPartition->computeUid();
					r1Violators.insert(newPartition->getUid());
#endif
					continue;
				}
#ifdef VERBOSE
				else
					newPartition->computeUid();
#endif

				if (newPartition->violatesR3())
				{
#ifdef VERBOSE
					r3Violators.insert(newPartition->getUid());
#endif
					continue;
				}
#ifndef VERBOSE
				newPartition->computeUid();
#endif
				if (uids.find(newPartition->getUid()) != uids.end())
				{
#ifdef VERBOSE
					numRepeatedPartitions++;
#endif
					continue;
				}
				if (matchGroups)
					newPartition->computeMatchGroups();
				else
					newPartition->createEmptyMatchGroups();
				// DEBUG:
				std::cout << newPartition->getUid() << std::endl;
				if (callback(uids.size(), newPartition))
					uids.insert(newPartition->getUid());
			}
#ifdef VERBOSE
			auto end = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsedTime = end - start;
			std::cout << "Partitions computation time: " << elapsedTime.count() << "s" << std::endl;
			std::cout << "No. partitions: " << uids.size() << std::endl;
			std::cout << "No. partitions that violated R1: " << r1Violators.size() << std::endl;
			std::cout << "No. partitions that violated R3: " << r3Violators.size() << std::endl;
			std::cout << "No. repeated partitions: " << numRepeatedPartitions << std::endl;
#endif
		}

		bool Graph::computePartition(ComputePartitionCallback& callback, bool matchGroups, const std::set<size_t>& cutEdgesIdxs) const
		{
			PartitionPtr newPartition(new Partition(this));
			auto subgraphIndex = newPartition->newSubgraph();
			std::map<size_t, size_t> visited;
			std::map<size_t, size_t> ancestry;
			computePartition(cutEdgesIdxs, newPartition, subgraphIndex, 0, 0, visited, ancestry);
			if (newPartition->violatesR1()) return false;
			if (newPartition->violatesR3()) return false;
			newPartition->computeUid();
			if (matchGroups)
				newPartition->computeMatchGroups();
			else
				newPartition->createEmptyMatchGroups();
			return callback(0, newPartition);
		}

		void Graph::findNeverCutEdges(std::set<size_t>& neverCutEdges, const std::set<size_t>& alwaysCutEdges /*= std::set<size_t>()*/) const
		{
			FindSingleParentDiscardEdges visitor(neverCutEdges, alwaysCutEdges);
			depthFirst(visitor);
		}

		void Graph::findAlwaysCutEdges(std::set<size_t>& alwaysCutEdges, const std::set<size_t>& neverCutEdges /*= std::set<size_t>()*/) const
		{
			std::set<size_t> cycleEdges;
			FindCycleEdges visitor(neverCutEdges, cycleEdges);
			depthFirst(visitor);
			alwaysCutEdges.insert(cycleEdges.begin(), cycleEdges.end());
			std::set<size_t> r3Violations;
			for (auto edgeIdx : cycleEdges)
			{
				auto& edge = edges[edgeIdx];
				/*std::vector<size_t> parents;
				findInVertices(edge.out, parents);
				for (auto parentIdx : parents)
				{
				// ignoring the source vertex of the cycle
				if (parentIdx == edge.in)
				continue;
				for (auto& adjacency : adjacenciesList[parentIdx])
				alwaysCutEdges.insert(adjacency.second);
				}*/
				std::set<size_t> subgraph;
				CollectVertices visitor(subgraph);
				depthFirst(visitor, edge.out);
				for (auto vertexIdx : subgraph)
				{
					// ignore subgraph root
					if (vertexIdx == edge.out)
						continue;
					std::vector<size_t> parents;
					findInVertices(vertexIdx, parents);
					for (auto parentIdx : parents)
					{
						if (subgraph.find(parentIdx) == subgraph.end())
							r3Violations.insert(vertexIdx);
					}
				}
			}
			for (auto vertexIdx : r3Violations)
			{
				std::set<size_t> inEdges;
				findInEdges(vertexIdx, inEdges);
				for (auto edgeIdx : inEdges)
					alwaysCutEdges.insert(edgeIdx);
			}
		}

		void Graph::getParents(std::map<size_t, std::set<size_t>>& parents) const
		{
			depthFirst(GetParents(parents));
		}

		//////////////////////////////////////////////////////////////////////////
		Graph::Partition::Partition(const Graph* source) : source(source), matchGroups(nullptr)
		{
		}

		bool Graph::Partition::violatesR1() const
		{
			for (auto& subGraph : subGraphs)
			{
				if (subGraph.hasCycle())
					return true;
			}
			return false;
		}

		// detects base graph sub-graphs with non-root vertices that have incoming edges in the base graph
		bool Graph::Partition::violatesR3() const
		{
			for (auto& subGraph : subGraphs)
			{
				for (auto i = 1; i < subGraph.vertices.size(); i++)
				{
					std::vector<size_t> ins;
					source->findInVertices(subGraph.vertices[i], ins);
					for (const auto& in : ins)
					{
						auto j = subGraph.vertexIndex(in);
						if (j == -1)
							return true;
						if (!subGraph.hasEdge(j, i))
							return true;
					}
				}
			}
			return false;
		}

		void Graph::Partition::traverse(PartitionVisitor& visitor, std::set<size_t>& visited, size_t subGraphIndex, size_t sgVertexIndex) const
		{
			if (sgVertexIndex == 0)
				visitor.enterSubgraph(subGraphIndex);
			auto vertexIndex = subGraphs[subGraphIndex].vertices[sgVertexIndex];
			auto vertex = source->vertices[vertexIndex];
			visitor.visitEnter(vertexIndex, sgVertexIndex, vertex);
			visited.insert(vertexIndex);
			for (auto& adjacency : subGraphs[subGraphIndex].adjacenciesList[sgVertexIndex])
			{
				auto& edge = source->edges[adjacency.second];
				visitor.visit(adjacency.second, edge, edge.in, source->vertices[edge.in], edge.out, source->vertices[edge.out], subGraphIndex, subGraphIndex, false);
				auto nextVertexIndex = adjacency.first;
				if (visited.find(edge.out) != visited.end())
					continue;
				traverse(visitor, visited, subGraphIndex, nextVertexIndex);
			}
			auto it = traversals.find(std::make_pair(subGraphIndex, sgVertexIndex));
			if (it == traversals.end())
			{
				visitor.visitLeave(vertexIndex, sgVertexIndex, vertex);
				if (sgVertexIndex == 0)
					visitor.leaveSubgraph(subGraphIndex);
			}
			else
			{
				for (auto& cutEdge : it->second)
				{
					auto otherSubGraphIndex = cutEdge.first;
					auto& edge = source->edges[cutEdge.second];
					visitor.visit(cutEdge.second, edge, edge.in, source->vertices[edge.in], edge.out, source->vertices[edge.out], subGraphIndex, otherSubGraphIndex, true);
				}
				visitor.visitLeave(vertexIndex, sgVertexIndex, vertex);
				if (sgVertexIndex == 0)
					visitor.leaveSubgraph(subGraphIndex);
				for (auto& cutEdge : it->second)
				{
					auto otherSubGraphIndex = cutEdge.first;
					auto& edge = source->edges[cutEdge.second];
					if (visited.find(edge.out) != visited.end())
						continue;
					traverse(visitor, visited, otherSubGraphIndex, 0);
				}
			}
		}

		size_t Graph::Partition::newSubgraph()
		{
			auto i = subGraphs.size();
			subGraphs.emplace_back(SubGraph(source));
			return i;
		}

		void Graph::Partition::addTraversal(size_t subgraph1Index, size_t vertexIndex, size_t subgraph2Index, size_t edgeIndex)
		{
			cutEdges.insert(std::make_pair(edgeIndex, subgraph2Index));
			traversals[std::make_pair(subgraph1Index, vertexIndex)].push_back(std::make_pair(subgraph2Index, edgeIndex));
		}

		void Graph::Partition::computeMatchGroups()
		{
			matchGroups = std::unique_ptr<MatchGroups>(new MatchGroups(this));
			std::set<size_t> visited;
			for (auto i = 0; i < subGraphs.size(); i++)
			{
				if (visited.find(i) != visited.end()) continue;
				visited.insert(i);
				auto& matchGroup = matchGroups->addGroup(i);
				for (auto j = i + 1; j < subGraphs.size(); j++)
				{
					if (visited.find(j) != visited.end()) continue;
					SubGraph::MatchResult matchResult;
					int score;
					if ((score = subGraphs[i].match(subGraphs[j], matchResult)) > -1)
					{
						visited.insert(j);
						matchGroup.addMatch(j, matchResult);
					}
				}
			}
			matchGroups->computeDiffs();
		}

		std::string Graph::Partition::uidFromCutEdges(size_t numEdges, const std::set<size_t>& cutEdges)
		{
			std::string uid;
			for (auto i = 0; i < numEdges; i++)
			{
				auto it = cutEdges.find(i);
				if (it == cutEdges.end())
					uid += "0";
				else
					uid += "1";
			}
			return uid;
		}

		std::set<size_t> Graph::Partition::cutEdgesFromUid(const std::string& partitionUid)
		{
			std::set<size_t> cutEdgesIdxs;
			for (auto i = 0; i < partitionUid.size(); i++)
				if (partitionUid[i] == '1')
					cutEdgesIdxs.insert(i);
			return cutEdgesIdxs;
		}

		void Graph::Partition::computeUid()
		{
			auto numEdges = source->numEdges();
			for (auto i = 0; i < numEdges; i++)
			{
				auto it = cutEdges.find(i);
				if (it == cutEdges.end())
					uid += "0";
				else
					uid += "1";
			}
			//uid = uidFromCutEdges(source->numEdges(), cutEdges);
		}

		bool Graph::Partition::isCutEdge(size_t edgeIdx) const
		{
			return cutEdges.find(edgeIdx) != cutEdges.end();
		}

		void Graph::Partition::createEmptyMatchGroups()
		{
			matchGroups = std::unique_ptr<MatchGroups>(new MatchGroups(this));
			std::set<size_t> visited;
			for (auto i = 0; i < subGraphs.size(); i++)
			{
				if (visited.find(i) != visited.end()) continue;
				visited.insert(i);
				matchGroups->addGroup(i);
			}
			matchGroups->computeDiffs();
		}

		//////////////////////////////////////////////////////////////////////////
		Graph::Partition::SubGraph::SubGraph(const Graph* source) : source(source)
		{
		}

		long Graph::Partition::SubGraph::vertexIndex(size_t i) const
		{
			auto it = std::find(vertices.begin(), vertices.end(), i);
			if (it == vertices.end()) return -1;
			return (long)std::distance(vertices.begin(), it);
		}

		bool Graph::Partition::SubGraph::hasCycle(size_t vertexIdx /*= 0*/, std::set<size_t> visited /*= std::set<size_t>()*/) const
		{
			if (visited.find(vertexIdx) != visited.end())
				return true;
			visited.insert(vertexIdx);
			for (auto& adjacency : adjacenciesList[vertexIdx])
			{
				if (hasCycle(adjacency.first, std::set<size_t>(visited.begin(), visited.end())))
					return true;
			}
			return false;
		}

		bool Graph::Partition::SubGraph::hasEdge(size_t in, size_t out) const
		{
			if (in >= vertices.size()) return false;
			if (out >= vertices.size()) return false;
			// TODO: can be optimized
			for (auto& adjacency : adjacenciesList[in])
				if (adjacency.first == out)
					return true;
			return false;
		}

		int Graph::Partition::SubGraph::match(const SubGraph& other, MatchResult& matchResult, size_t i, size_t j) const
		{
			auto k = vertices[i];
			auto l = other.vertices[j];

			auto& v0 = source->vertices[k];
			auto& v1 = other.source->vertices[l];
			if (v0.isDiff(v1)) return -1;
			if (adjacenciesList[i].size() != other.adjacenciesList[j].size()) return -1;

			int matchingScore = static_cast<int>(v0.getCommonParams(v1, matchResult.vertexParametersMap[k]));
			matchingScore += static_cast<int>(v0.getCommonTermParams(v1, matchResult.vertexTermAttrMap[k]));

			if (adjacenciesList[i].empty())
			{
				matchResult.vertexMap.insert(std::make_pair(k, l));
				matchResult.visited.insert(j);
				return matchingScore;
			}

#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (v0.operatorType() != v1.operatorType())
				throw std::runtime_error("v0.operatorType() != v1.operatorType()");
#endif

			bool adjacencyMatchingOrderMatters = v0.operatorType() != SUBDIV && v0.operatorType() != REPEAT && v0.operatorType() != STOCHASTIC;
			std::vector<std::vector<size_t>> allMatchingSequences(adjacenciesList[i].size());
			std::map<size_t, std::map<size_t, std::pair<int, MatchResult>>> childrenMatchResults;
			for (size_t adjacency1Idx = 0; adjacency1Idx < adjacenciesList[i].size(); adjacency1Idx++)
			{
				auto& adjacency1 = adjacenciesList[i][adjacency1Idx];
				long originalAdjacency1Pos = -1;
				if (adjacencyMatchingOrderMatters)
				{
					// NOTE: getting the position of adjacency1 edge in the "original" adjacencies list for current vertex
					auto it1 = std::find_if(source->adjacenciesList[k].begin(), source->adjacenciesList[k].end(), AdjacencyComparer<false>(adjacency1.second));
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (it1 == source->adjacenciesList[k].end())
						throw std::runtime_error("it1 == source->adjacenciesList[k].end()");
#endif
					originalAdjacency1Pos = static_cast<long>(std::distance(source->adjacenciesList[k].begin(), it1));
				}
				for (size_t adjacency2Idx = 0; adjacency2Idx < other.adjacenciesList[j].size(); adjacency2Idx++)
				{
					auto& adjacency2 = other.adjacenciesList[j][adjacency2Idx];
					// NOTE: do not match an adjacency twice
					if (matchResult.visited.find(adjacency2.first) != matchResult.visited.end())
						continue;
					MatchResult childMatchResult;
					int childMatchingScore = match(other, childMatchResult, adjacency1.first, adjacency2.first);
					if (childMatchingScore == -1) continue;
					// NOTE: if adjacency order matters, only accept adjacency matches that correspond to 
					// edges at the same position in both "original" adjacencies lists
					if (adjacencyMatchingOrderMatters)
					{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (originalAdjacency1Pos == -1)
							throw std::runtime_error("originalAdjacency1Pos == -1");
#endif
						auto it1 = std::find_if(source->adjacenciesList[l].begin(), source->adjacenciesList[l].end(), AdjacencyComparer<false>(adjacency2.second));
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (it1 == source->adjacenciesList[l].end())
							throw std::runtime_error("it1 == source->adjacenciesList[l].end()");
#endif
						auto originalAdjacency2Pos = static_cast<long>(std::distance(source->adjacenciesList[l].begin(), it1));
						if (originalAdjacency1Pos != originalAdjacency2Pos)
							continue;
					}
					auto& e0 = source->edges[adjacency1.second];
					auto& e1 = other.source->edges[adjacency2.second];
					auto p0 = e0.getParameter().lock();
					auto p1 = e1.getParameter().lock();
					if (p0 != nullptr && p1 != nullptr && p0->isEqual(p1.get()))
					{
						childMatchingScore++;
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (childMatchResult.edgesWithMatchingParam.find(adjacency1.second) != childMatchResult.edgesWithMatchingParam.end())
							throw std::runtime_error("childMatchResult.edgesWithMatchingParam.find(adjacency1.second) != childMatchResult.edgesWithMatchingParam.end()");
#endif
						childMatchResult.edgesWithMatchingParam.insert(adjacency1.second);
					}
					childrenMatchResults[adjacency1Idx][adjacency2Idx] = std::make_pair(childMatchingScore, childMatchResult);
					allMatchingSequences[adjacency1Idx].push_back(adjacency2Idx);
				}
				if (allMatchingSequences[adjacency1Idx].empty())
					return -1;
			}

			std::vector<std::vector<size_t>> validMatchingSequences;
			SubGraph::increasingSequences(allMatchingSequences, validMatchingSequences);

			if (validMatchingSequences.empty())
				return -1;

			size_t bestMatchingSequenceIdx;
			int highestChildMatchingScore = -1;
			for (size_t matchingSequenceIdx = 0; matchingSequenceIdx < validMatchingSequences.size(); matchingSequenceIdx++)
			{
				auto& matchingSequence = validMatchingSequences[matchingSequenceIdx];
				int childMatchingScore = 0;
				for (size_t adjacencyIdx = 0; adjacencyIdx < matchingSequence.size(); adjacencyIdx++)
				{
					auto otherAdjacencyIdx = matchingSequence[adjacencyIdx];
					auto it1 = childrenMatchResults.find(adjacencyIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (it1 == childrenMatchResults.end())
						throw std::runtime_error("it1 == childrenMatchResults.end()");
#endif
					auto it2 = it1->second.find(otherAdjacencyIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (it2 == it1->second.end())
						throw std::runtime_error("it2 == it1->second.end()");
#endif
					childMatchingScore += it2->second.first;
				}
				if (childMatchingScore > highestChildMatchingScore)
				{
					highestChildMatchingScore = childMatchingScore;
					bestMatchingSequenceIdx = matchingSequenceIdx;
				}
			}

			matchingScore += highestChildMatchingScore;

			auto& matchingSequence = validMatchingSequences[bestMatchingSequenceIdx];
			for (size_t adjacencyIdx = 0; adjacencyIdx < matchingSequence.size(); adjacencyIdx++)
			{
				auto otherAdjacencyIdx = matchingSequence[adjacencyIdx];
				auto it1 = childrenMatchResults.find(adjacencyIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it1 == childrenMatchResults.end())
					throw std::runtime_error("it1 == childrenMatchResults.end()");
#endif
				auto it2 = it1->second.find(otherAdjacencyIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it2 == it1->second.end())
					throw std::runtime_error("it2 == it1->second.end()");
#endif
				matchResult += it2->second.second;
				matchResult.edgeMap.insert(
					std::make_pair(
					adjacenciesList[i][adjacencyIdx].second,
					other.adjacenciesList[j][adjacencyIdx].second
					)
					);
			}

			matchResult.vertexMap.insert(std::make_pair(k, l));
			matchResult.visited.insert(j);

			return matchingScore;
		}

		void Graph::Partition::SubGraph::increasingSequences(const std::vector<std::vector<size_t>>& all, std::vector<std::vector<size_t>>& sequences, size_t i, std::vector<size_t> sequence)
		{
			if (i == 0)
			{
				if (all.empty())
					return;
				for (auto j : all[0])
					increasingSequences(all, sequences, 1, { j });
				return;
			}
			else if (i == all.size())
			{
				sequences.push_back(sequence);
				return;
			}
			auto& next = all[i];
			auto it = std::upper_bound(next.begin(), next.end(), sequence.back());
			while (it != next.end())
			{
				std::vector<size_t> newSequence(sequence.begin(), sequence.end());
				newSequence.push_back(*it);
				increasingSequences(all, sequences, i + 1, newSequence);
				it++;
			}
		}

		bool Graph::Partition::SubGraph::hasVertex(size_t vertexIdx) const
		{
			return std::find(vertices.begin(), vertices.end(), vertexIdx) != vertices.end();
		}

		size_t Graph::Partition::SubGraph::addVertex(size_t vertexIdx)
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (hasVertex(vertexIdx))
				throw std::runtime_error("hasVertex(vertexIdx)");
#endif
			auto i = vertices.size();
			vertices.push_back(vertexIdx);
			adjacenciesList.resize(vertices.size());
			return i;
		}

		void Graph::Partition::SubGraph::makeEdge(size_t in, size_t out, size_t ref)
		{
			adjacenciesList[in].push_back(std::make_pair(out, ref));
		}

		//////////////////////////////////////////////////////////////////////////
		Graph::Partition::MatchGroups::MatchGroup::MatchGroup(MatchGroups* source, size_t srcIdx) : source(source), srcIdx(srcIdx)
		{
		}

		void Graph::Partition::MatchGroups::MatchGroup::addMatch(size_t dstIdx, const SubGraph::MatchResult& matchResult)
		{
			matchesMap.emplace(std::make_pair(dstIdx, Match(this, dstIdx, matchResult)));
			source->addMapping(srcIdx, dstIdx);
		}

		void Graph::Partition::MatchGroups::MatchGroup::computeDiffs()
		{
			auto& sgVerticesIdxs = source->partition->subGraphs[srcIdx].vertices;
			for (size_t i = 0; i < sgVerticesIdxs.size(); i++)
			{
				auto vertexIdx = sgVerticesIdxs[i];
				auto& vertex = source->partition->source->vertices[vertexIdx];
				auto numVertParams = vertex.numParams();
				std::vector<std::weak_ptr<Parameter>> vertParams;
				vertex.getParams(vertParams);
				std::map<size_t, std::weak_ptr<Parameter>> eqVertParams;
				std::map<size_t, std::map<size_t, std::weak_ptr<Parameter>>> diffVertParams;
				for (auto j = 0; j < numVertParams; j++)
				{
					bool found = true;
					for (auto& entry : matchesMap)
					{
						auto& vertexParametersMap = entry.second.vertexParametersMap[vertexIdx];
						if (vertexParametersMap.find(j) == vertexParametersMap.end())
						{
							found = false;
							break;
						}
					}
					if (found)
						eqVertParams[j] = vertParams[j];
					else
						diffVertParams[srcIdx][j] = vertParams[j];
				}
				std::vector<double> vertTermParams;
				vertex.getTermAttrs(vertTermParams);
				auto numVertTermParams = vertTermParams.size();
				for (auto& entry : matchesMap)
				{
					auto it = entry.second.vertexMap.find(vertexIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (it == entry.second.vertexMap.end())
						throw std::runtime_error("it == entry.second.vertexMap.end()");
#endif
					auto otherVertNumTermParams = source->partition->source->vertices[it->second].numTermAttrs();
					if (otherVertNumTermParams > numVertTermParams)
						numVertTermParams = otherVertNumTermParams;
				}
				std::map<size_t, double> eqVertTermAttrs;
				std::map<size_t, std::map<size_t, double>> diffVertTermAttrs;
				for (auto j = 0; j < numVertTermParams; j++)
				{
					bool found = true;
					for (auto& entry : matchesMap)
					{
						auto it = entry.second.vertexTermAttrsMap.find(vertexIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (it == entry.second.vertexTermAttrsMap.end())
							throw std::runtime_error("it == entry.second.vertexTermAttrsMap.end()");
#endif
						if (it->second.find(j) == it->second.end())
						{
							found = false;
							break;
						}
					}
					if (found)
						eqVertTermAttrs[j] = vertTermParams[j];
					else
						diffVertTermAttrs[srcIdx][j] = vertTermParams[j];
				}
				for (auto& entry : matchesMap)
				{
					auto it = entry.second.vertexMap.find(vertexIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (it == entry.second.vertexMap.end())
						throw std::runtime_error("it == entry.second.vertexMap.end()");
#endif
					std::vector<double> otherVertTermAttrs;
					source->partition->source->vertices[it->second].getTermAttrs(otherVertTermAttrs);
					for (size_t j = 0; j < numVertTermParams; j++)
						if (eqVertTermAttrs.find(j) == eqVertTermAttrs.end())
							diffVertTermAttrs[entry.first][j] = otherVertTermAttrs[j];
				}
				auto& adjacencyList = source->partition->source->adjacenciesList[vertexIdx];
				std::vector<long> edgesPos(adjacencyList.size());
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				std::map<size_t, size_t> minParamIdxMap;
#endif
				for (size_t j = 0, k = 0; j < adjacencyList.size(); j++, k++)
				{
					auto edgeIdx = adjacencyList[j].second;
					edgesPos[k] = static_cast<long>(edgeIdx);
					size_t l = k;
					for (auto& entry : matchesMap)
					{
						auto& match = entry.second;
						auto it2 = match.vertexMap.find(vertexIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (it2 == match.vertexMap.end())
							throw std::runtime_error("it1 == match.vertexMap.end()");
#endif
						auto otherVertexIdx = it2->second;
						auto& otherVertex = source->partition->source->vertices[otherVertexIdx];
						std::vector<std::weak_ptr<Parameter>> otherVertParams;
						otherVertex.getParams(otherVertParams);
						diffVertParams[entry.first];
						for (size_t k = 0; k < numVertParams; k++)
							if (eqVertParams.find(k) == eqVertParams.end())
								diffVertParams[entry.first][k] = otherVertParams[k];
						auto& otherAdjacencyList = source->partition->source->adjacenciesList[otherVertexIdx];
						auto it3 = match.edgeMap.find(edgeIdx);
						if (it3 == match.edgeMap.end()) continue;
						auto it4 = std::find_if(otherAdjacencyList.begin(), otherAdjacencyList.end(), AdjacencyComparer<false>(it3->second));
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (it4 == otherAdjacencyList.end())
							throw std::runtime_error("it2 == otherAdjacencyList.end()");
#endif
						auto m = static_cast<size_t>(std::distance(otherAdjacencyList.begin(), it4));
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						// NOTE: Checking if edges are mapped in increasing order.
						// In other words, the index of a edge mapping can never be smaller then the index of the next edge mapping.
						// i.e:
						//
						//		S											S
						//    /   \         should never be mapped to     /	  \
												//	T       R									R'		T'
						//
						//     because T' mapping idx is 0 while R' mapping idx is 1
						//
						auto it5 = minParamIdxMap.find(entry.first);
						if (it5 != minParamIdxMap.end())
						{
							if (it5->second > m)
								throw std::runtime_error("it5->second > m");
						}
						minParamIdxMap[entry.first] = m;
#endif
						if (l < m)
							l = m;
					}
					if (l == k) continue;
					auto it1 = edgesPos.begin();
					std::advance(it1, k);
					edgesPos.insert(it1, l - k, -1);
					k = l;
				}

				std::map<size_t, std::map<size_t, size_t>> edgesMap;
				std::set<size_t> edgeWithDynParam;
				for (auto& adjacency : adjacencyList)
				{
					auto edgeIdx = adjacency.second;
					bool found = true;
					for (auto& entry : matchesMap)
					{
						auto& match = entry.second;
						if (match.edgesWithMatchingParam.find(edgeIdx) == match.edgesWithMatchingParam.end())
						{
							found = false;
							break;
						}
					}
					if (!found)
						edgeWithDynParam.insert(edgeIdx);
					auto it1 = std::find(edgesPos.begin(), edgesPos.end(), edgeIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (it1 == edgesPos.end())
						throw std::runtime_error("it1 == edgesPos.end()");
#endif
					auto j = std::distance(edgesPos.begin(), it1);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (edgesMap.find(j) != edgesMap.end())
						throw std::runtime_error("edgesMap.find(j) != edgesMap.end()");
#endif
					edgesMap[j].insert(std::make_pair(srcIdx, edgeIdx));
				}

				for (auto& it1 : matchesMap)
				{
					auto dstIdx = it1.first;
					auto& match = it1.second;
					auto it2 = match.vertexMap.find(vertexIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (it2 == match.vertexMap.end())
						throw std::runtime_error("it2 == match.vertexMap.end()");
#endif
					auto otherVertexIdx = it2->second;
					auto otherSgVertexIdx = source->partition->subGraphs[match.dstIdx].vertexIndex(otherVertexIdx);
					if (otherSgVertexIdx == -1) continue;
					size_t j = 0, k = 0;
					auto& otherAdjacencyList = source->partition->source->adjacenciesList[otherVertexIdx];
					while (j < otherAdjacencyList.size())
					{
						auto edgeIdx = otherAdjacencyList[j].second;
						auto it3 = std::find_if(match.edgeMap.begin(), match.edgeMap.end(), EdgeMapValueComparer(edgeIdx));
						if (it3 != match.edgeMap.end())
						{
							auto it4 = std::find(edgesPos.begin(), edgesPos.end(), it3->first);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
							if (it4 == edgesPos.end())
								throw std::runtime_error("it4 == edgesPos.end()");
#endif
							k = std::distance(edgesPos.begin(), it4);
						}
						edgesMap[k].insert(std::make_pair(dstIdx, edgeIdx));
						j++, k++;
					}
				}

				auto numMatches = matchesMap.size() + 1;
				for (auto& it1 : edgesMap)
				{
					auto key = std::make_pair(vertexIdx, it1.first);
					std::map<size_t, size_t> cutMap;
					std::map<size_t, std::weak_ptr<Parameter>> paramMap;
					bool requiresParam = Operator::requiresEdgeParameter(vertex.operatorType());
					bool cutEdge;
					bool dynParam;
					if (it1.second.size() == numMatches)
					{
						auto it2 = it1.second.find(srcIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (it2 == it1.second.end())
							throw std::runtime_error("it2 == it1.second.end()");
#endif
						auto edgeIdx = it2->second;
						cutEdge = source->partition->cutEdges.find(edgeIdx) != source->partition->cutEdges.end();
						dynParam = (requiresParam && edgeWithDynParam.find(edgeIdx) != edgeWithDynParam.end());
					}
					else
					{
						cutEdge = true;
						dynParam = requiresParam;
					}

					for (auto it2 : it1.second)
					{
						auto sgIdx = it2.first;
						auto corrEdgeIdx = it2.second;
						paramMap.insert(std::make_pair(sgIdx, source->partition->source->edges[corrEdgeIdx].getParameter()));
						if (cutEdge)
						{
							auto it3 = source->partition->cutEdges.find(corrEdgeIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
							if (it3 == source->partition->cutEdges.end())
								throw std::runtime_error("it3 == partition->cutEdgesIdxs.end()");
#endif
							cutMap.emplace(*it3);
						}
					}

					bool common = false;
					if (cutEdge)
					{
						bool eqOutIdxs = false;
						if (cutMap.size() == numMatches)
						{
							auto it = cutMap.begin();
							auto it2 = source->mapping.find(it->second);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
							if (it2 == source->mapping.end())
								throw std::runtime_error("it2 == source->mapping.end()");
#endif
							size_t outIdx = it2->second;
							eqOutIdxs = true;
							while (++it != cutMap.end())
							{
								auto it2 = source->mapping.find(it->second);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
								if (it2 == source->mapping.end())
									throw std::runtime_error("it2 == source->mapping.end()");
#endif
								if (it2->second != outIdx)
								{
									eqOutIdxs = false;
									break;
								}
							}
						}
						if (eqOutIdxs)
							common = true;
					}
					else
					{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (it1.second.size() != numMatches)
							throw std::runtime_error("it1.second.size() != numMatches");
#endif
						bool eqOutIdxs = true;
						auto it3 = it1.second.begin();
						size_t outIdx = it3->second;
						while (++it3 != it1.second.end())
						{
							if (it3->second != outIdx)
							{
								eqOutIdxs = false;
								break;
							}
						}
						if (eqOutIdxs)
							common = true;
					}
					edgesDiff.emplace(std::make_pair(key, EdgeDiff(this, requiresParam, dynParam, it1.second, cutMap, paramMap, common)));
				}
				verticesDiff.emplace(std::make_pair(vertexIdx, VertexDiff(vertex.operatorType(), vertex.shapeType(), vertex.phase(), vertex.getGenFuncIdx(), numVertParams, eqVertParams, diffVertParams, numVertTermParams, eqVertTermAttrs, diffVertTermAttrs, edgesMap.size())));
			}
		}

		void Graph::Partition::MatchGroups::MatchGroup::traverse(MatchGroupVisitor& visitor, size_t dstIdx, size_t srcInVertexIdx, std::set<size_t>& visited) const
		{
			auto it1 = verticesDiff.find(srcInVertexIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (it1 == verticesDiff.end())
				throw std::runtime_error("it1 == verticesDiff.end()");
#endif
			auto& vertexDiff = it1->second;
			size_t dstInVertexIdx;
			if (srcIdx != dstIdx)
			{
				auto it2 = matchesMap.find(dstIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it2 == matchesMap.end())
					throw std::runtime_error("it2 == matchesMap.end()");
#endif
				auto& match = it2->second;
				auto it3 = match.vertexMap.find(srcInVertexIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it3 == match.vertexMap.end())
					throw std::runtime_error("it3 == match.vertexMap.end()");
#endif
				dstInVertexIdx = it3->second;
			}
			else
				dstInVertexIdx = srcInVertexIdx;

			std::map<size_t, std::weak_ptr<Parameter>> diffVertParams;
			vertexDiff.getDiffParams(dstIdx, diffVertParams);
			std::map<size_t, double> diffTermParams;
			vertexDiff.getDiffTermAttrs(dstIdx, diffTermParams);
			visitor.visitVertex(
				dstIdx,
				dstInVertexIdx,
				vertexDiff.opType,
				vertexDiff.shapeType,
				vertexDiff.phase,
				vertexDiff.genFuncIdx,
				vertexDiff.numParams,
				vertexDiff.eqParams,
				diffVertParams,
				vertexDiff.numTermAttrs,
				vertexDiff.eqTermAttrs,
				diffTermParams,
				vertexDiff.numEdges
				);

			visited.insert(srcInVertexIdx);
			for (auto i = 0; i < vertexDiff.numEdges; i++)
			{
				auto it4 = edgesDiff.find(std::make_pair(srcInVertexIdx, i));
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it4 == edgesDiff.end())
					throw std::runtime_error("it4 == edgesDiff.end()");
#endif
				auto& edgeDiff = it4->second;

				auto it5 = edgeDiff.matchesMap.find(srcIdx);
				long srcEdgeIdx;
				if (it5 != edgeDiff.matchesMap.end())
					srcEdgeIdx = static_cast<long>(it5->second);
				else
					srcEdgeIdx = -1;

				auto it6 = edgeDiff.matchesMap.find(dstIdx);
				long dstEdgeIdx;
				if (it6 != edgeDiff.matchesMap.end())
					dstEdgeIdx = static_cast<long>(it6->second);
				else
					dstEdgeIdx = -1;

				if (edgeDiff.isCut())
				{
					std::weak_ptr<Parameter> edgeParam;
					if (srcIdx == dstIdx && srcEdgeIdx != -1)
						edgeParam = source->partition->source->edges[srcEdgeIdx].getParameter();
					else if (srcIdx != dstIdx && dstEdgeIdx != -1)
						edgeParam = source->partition->source->edges[dstEdgeIdx].getParameter();

					long dstSgIdx;
					if (dstEdgeIdx != -1)
					{
						auto it7 = edgeDiff.cutMap.find(dstEdgeIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (it7 == edgeDiff.cutMap.end())
							throw std::runtime_error("it7 == edgeDiff.cutEdgesMap.end()");
#endif
						dstSgIdx = static_cast<long>(it7->second);
					}
					else
						dstSgIdx = -1;

					visitor.visitEdge(
						dstIdx,
						dstInVertexIdx,
						dstSgIdx,
						dstEdgeIdx,
						true,
						edgeDiff.common,
						edgeDiff.requiresParam,
						edgeDiff.hasDynParam,
						edgeParam,
						i);
				}
				else
				{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (srcEdgeIdx == -1)
						throw std::runtime_error("srcEdgeIdx == -1");
#endif
					auto& srcEdge = source->partition->source->edges[srcEdgeIdx];
					auto srcOutVertexIdx = srcEdge.out;
					size_t dstOutVertexIdx;
					std::weak_ptr<Parameter> edgeParam;
					if (srcIdx != dstIdx)
					{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (dstEdgeIdx == -1)
							throw std::runtime_error("srcIdx != dstIdx && dstEdgeIdx == -1");
#endif
						auto& dstEdge = source->partition->source->edges[dstEdgeIdx];
						dstOutVertexIdx = dstEdge.out;
						edgeParam = dstEdge.getParameter();
					}
					else
					{
						dstOutVertexIdx = srcOutVertexIdx;
						edgeParam = srcEdge.getParameter();
					}

					visitor.visitEdge(
						dstIdx,
						dstInVertexIdx,
						static_cast<long>(dstOutVertexIdx),
						dstEdgeIdx,
						false,
						edgeDiff.common,
						edgeDiff.requiresParam,
						edgeDiff.hasDynParam,
						edgeParam,
						i);

					if (visited.find(srcOutVertexIdx) != visited.end())
						continue;

					traverse(visitor, dstIdx, srcOutVertexIdx, visited);
				}
			}
		}

		//////////////////////////////////////////////////////////////////////////
		Graph::Partition::MatchGroups::MatchGroup::Match::Match(
			const MatchGroup* group,
			size_t dstIdx,
			const SubGraph::MatchResult& result
			) :
			group(group),
			dstIdx(dstIdx),
			vertexMap(result.vertexMap.begin(), result.vertexMap.end()),
			edgeMap(result.edgeMap.begin(), result.edgeMap.end()),
			vertexParametersMap(result.vertexParametersMap.begin(), result.vertexParametersMap.end()),
			vertexTermAttrsMap(result.vertexTermAttrMap.begin(), result.vertexTermAttrMap.end()),
			edgesWithMatchingParam(result.edgesWithMatchingParam.begin(), result.edgesWithMatchingParam.end())
		{
		}

		//////////////////////////////////////////////////////////////////////////
		Graph::Partition::MatchGroups::MatchGroup::VertexDiff::VertexDiff(
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
			size_t numEdges)
			:
			opType(opType),
			shapeType(shapeType),
			phase(phase),
			genFuncIdx(genFuncIdx),
			numParams(numParams),
			eqParams(eqParams.begin(), eqParams.end()),
			numTermAttrs(numTermAttrs),
			eqTermAttrs(eqTermAttrs.begin(), eqTermAttrs.end()),
			diffTermAttrs(diffTermAttrs.begin(), diffTermAttrs.end()),
			diffParams(diffParams.begin(), diffParams.end()),
			numEdges(numEdges)
		{
		}

		void Graph::Partition::MatchGroups::MatchGroup::VertexDiff::getEqParams(std::map<size_t, std::weak_ptr<Parameter>>& eqParams) const
		{
			eqParams.insert(this->eqParams.begin(), this->eqParams.end());
		}

		void Graph::Partition::MatchGroups::MatchGroup::VertexDiff::getDiffParams(size_t i, std::map<size_t, std::weak_ptr<Parameter>>& diffParams) const
		{
			auto it = this->diffParams.find(i);
			if (it == this->diffParams.end())
				return;
			diffParams.insert(it->second.begin(), it->second.end());
		}

		void Graph::Partition::MatchGroups::MatchGroup::VertexDiff::getDiffTermAttrs(size_t i, std::map<size_t, double>& diffTermAttrs) const
		{
			auto it = this->diffTermAttrs.find(i);
			if (it == this->diffTermAttrs.end())
				return;
			diffTermAttrs.insert(it->second.begin(), it->second.end());
		}

		//////////////////////////////////////////////////////////////////////////
		Graph::Partition::MatchGroups::MatchGroup::EdgeDiff::EdgeDiff(
			const MatchGroup* source,
			bool requiresParam,
			bool hasDynParam,
			const std::map<size_t, size_t>& matchesMap,
			const std::map<size_t, size_t>& cutMap,
			const std::map<size_t, std::weak_ptr<Parameter>>& paramMap,
			bool common
			)
			:
			source(source),
			requiresParam(requiresParam),
			hasDynParam(hasDynParam),
			matchesMap(matchesMap.begin(), matchesMap.end()),
			cutMap(cutMap.begin(), cutMap.end()),
			paramMap(paramMap.begin(), paramMap.end()),
			common(common)
		{
		}

		//////////////////////////////////////////////////////////////////////////
		Graph::Partition::SubGraph::MatchResult& Graph::Partition::SubGraph::MatchResult::operator+=(const MatchResult& other)
		{
			vertexMap.insert(other.vertexMap.begin(), other.vertexMap.end());
			vertexParametersMap.insert(other.vertexParametersMap.begin(), other.vertexParametersMap.end());
			vertexTermAttrMap.insert(other.vertexTermAttrMap.begin(), other.vertexTermAttrMap.end());
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			auto size = edgesWithMatchingParam.size();
#endif
			edgesWithMatchingParam.insert(other.edgesWithMatchingParam.begin(), other.edgesWithMatchingParam.end());
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (edgesWithMatchingParam.size() != (size + other.edgesWithMatchingParam.size()))
				throw std::runtime_error("edgesWithMatchingParam.size() != (size + other.edgesWithMatchingParam.size())");
#endif
			edgeMap.insert(other.edgeMap.begin(), other.edgeMap.end());
			visited.insert(other.visited.begin(), other.visited.end());
			return *this;
		}

		void Graph::Partition::MatchGroups::addMapping(size_t srcIdx, size_t dstIdx)
		{
			mapping[dstIdx] = srcIdx;
		}

		void Graph::Partition::MatchGroups::computeDiffs()
		{
			for (auto& it : groups)
				it.second.computeDiffs();
		}

		void Graph::Partition::MatchGroups::traverse(MatchGroupVisitor& visitor, bool groupsOnly) const
		{
			if (groupsOnly)
			{
				for (auto& entry : groups)
				{
					auto& group = entry.second;
					std::set<size_t> visited;
					group.traverse(visitor, group.srcIdx, partition->subGraphs[group.srcIdx].vertices[0], visited);
				}
			}
			else
			{
				for (size_t sgIdx = 0; sgIdx < partition->subGraphs.size(); sgIdx++)
				{
					auto it = mapping.find(sgIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (it == mapping.end())
						throw std::runtime_error("it == mapping.end()");
#endif
					auto it2 = groups.find(it->second);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (it2 == groups.end())
						throw std::runtime_error("it2 = groups.find(it->second)");
#endif
					auto& group = it2->second;
					std::set<size_t> visited;
					group.traverse(visitor, sgIdx, partition->subGraphs[group.srcIdx].vertices[0], visited);
				}
			}
		}

		Graph::Partition::MatchGroups::MatchGroup& Graph::Partition::MatchGroups::addGroup(size_t srcIdx)
		{
			auto it = groups.find(srcIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (it != groups.end())
				throw std::runtime_error("it != groups.end()");
#endif
			groups.insert(std::make_pair(srcIdx, MatchGroup(this, srcIdx)));
			mapping[srcIdx] = srcIdx;
			it = groups.find(srcIdx);
			return it->second;
		}

		Graph::Partition::MatchGroups::MatchGroups(const Partition* partition) : partition(partition)
		{
		}

	}

}
