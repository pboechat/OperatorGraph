#include <math/vector.h>
#include <pga/compiler/CodeGenerator.h>
#include <pga/compiler/EnumUtils.h>
#include <pga/compiler/MatchGroupVisitor.h>
#include <pga/compiler/OperatorType.h>
#include <pga/compiler/Parameters.h>
#include <pga/compiler/ShapeType.h>
#include <pga/core/GeometryUtils.h>
#include <pga/core/GlobalConstants.h>

#include <cassert>
#include <cmath>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace PGA
{
	namespace Compiler
	{
		struct Connection_
		{
			long edgeIdx;
			bool cutEdge;
			bool commonEdge;
			bool requiresParam;
			bool dynParam;
			std::weak_ptr<Parameter> param;
			long out;
			bool staticRef;
			Connection_(long edgeIdx, bool cutEdge, bool commonEdge, bool requiresParam, bool dynParam, const std::weak_ptr<Parameter>& param, long out) : edgeIdx(edgeIdx), cutEdge(cutEdge), commonEdge(commonEdge), requiresParam(requiresParam), dynParam(dynParam), param(param), out(out), staticRef(false) {}

			inline bool isFullyStatic() const
			{
				return !cutEdge || (cutEdge && commonEdge && staticRef);
			}

		};

		struct Operator_
		{
			OperatorType type;
			size_t vertexIdx;
			ShapeType shapeType;
			int phase;
			long genFuncIdx;
			size_t numParams;
			std::map<size_t, std::weak_ptr<Parameter>> params;
			size_t numTermAttrs;
			std::map<size_t, double> termAttrs;
			size_t numEdges;
			std::map<size_t, Connection_> connections;

			inline bool isFullyStatic() const
			{
				if (numParams != params.size()) return false;
				if (numTermAttrs != termAttrs.size()) return false;
				for (auto& it : connections)
				{
					auto& conn = it.second;
					if (!conn.isFullyStatic())
						return false;
				}
				return true;
			}

		};

		struct Procedure_
		{
			std::map<size_t, Operator_> vertexIdxToOp;
			std::vector<std::tuple<size_t, size_t, size_t>> referrers;

			size_t numThreads() const
			{
				if (vertexIdxToOp.size() != 1)
					return 1;

				auto& op = vertexIdxToOp.begin()->second;

				if (op.genFuncIdx != 0)
					return 1;

				if (op.type == OperatorType::GENERATE && op.shapeType == ShapeType::BOX)
					return 16;
				else if (op.type == OperatorType::COMPSPLIT && op.shapeType == ShapeType::BOX)
					return 16;

				return 1;
			}

			bool isParallel() const
			{
				return (numThreads() > 1);
			}

			inline bool isFullyStatic() const
			{
				for (auto it : vertexIdxToOp)
				{
					auto& op = it.second;
					if (!op.isFullyStatic())
						return false;
				}
				return true;
			}

		};

		struct ProcedureListCodeGenerator : MatchGroupVisitor /* */
		{
			std::map<size_t, size_t> opCodeMap;

			ProcedureListCodeGenerator(const std::map<size_t, size_t>& subGraphMap, bool optimized, bool instrumented) : subGraphMap(subGraphMap), optimized(optimized), instrumented(instrumented) {}

			virtual void visitVertex(
				size_t sgIdx,
				size_t vertexIdx,
				OperatorType opType,
				ShapeType shapeType,
				int phase,
				long genFuncIdx,
				size_t numParams,
				const std::map<size_t, std::weak_ptr<Parameter>>& eqParams,
				const std::map<size_t, std::weak_ptr<Parameter>>& diffParams,
				size_t numTermAttrs,
				const std::map<size_t, double>& eqTermParams,
				const std::map<size_t, double>& diffTermParams,
				size_t numEdges
				)
			{
				auto& proc = getProc(sgIdx);
				auto it2 = proc.vertexIdxToOp.find(vertexIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it2 != proc.vertexIdxToOp.end())
					throw std::runtime_error("it2 != proc.vertexIdxToOp.end()");
#endif
				auto& op = proc.vertexIdxToOp[vertexIdx];
				op.vertexIdx = vertexIdx;
				op.type = opType;
				op.shapeType = shapeType;
				op.phase = phase;
				op.genFuncIdx = genFuncIdx;
				op.numParams = numParams;
				op.params.clear();
				op.params.insert(eqParams.begin(), eqParams.end());
				op.numTermAttrs = numTermAttrs;
				op.termAttrs.insert(eqTermParams.begin(), eqTermParams.end());
				op.numEdges = numEdges;
			}

			virtual void visitEdge(
				size_t sgIdx,
				size_t inIdx,
				long outIdx,
				long edgeIdx,
				bool cutEdge,
				bool commonEdge,
				bool requiresParam,
				bool dynParam,
				const std::weak_ptr<Parameter>& param,
				size_t i)
			{
				auto& proc = findProc(sgIdx);
				auto it2 = proc.vertexIdxToOp.find(inIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it2 == proc.vertexIdxToOp.end())
					throw std::runtime_error("it2 == proc.vertexIdxToOp.end()");
#endif
				auto& op = it2->second;
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (op.connections.find(i) != op.connections.end())
					throw std::runtime_error("op.connections.find(i) != op.connections.end()");
#endif
				op.connections.emplace(std::make_pair(i, Connection_(edgeIdx, cutEdge, commonEdge, requiresParam, dynParam, param, outIdx)));
				if (cutEdge && commonEdge)
				{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (outIdx == -1)
						throw std::runtime_error("outIdx == -1");
#endif
					auto it3 = subGraphMap.find(outIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (it3 == subGraphMap.end())
						throw std::runtime_error("it3 == subGraphMap.end()");
#endif
					auto currSgIdx = it3->second;
					auto& otherProc = getProc(currSgIdx);
					otherProc.referrers.push_back(std::make_tuple(sgIdx, inIdx, i));
				}
			}

			void resolveFullyStaticReferences(std::set<size_t>& fullyStaticSgIdxs)
			{
				if (!optimized)
					return;

				while (true)
				{
					std::set<size_t> newFullyStaticSgIdxs;
					for (auto& it : opCodeMap)
					{
						if (fullyStaticSgIdxs.find(it.first) != fullyStaticSgIdxs.end()) continue;
						auto& proc = procList[it.second];
						if (proc.isFullyStatic())
							newFullyStaticSgIdxs.insert(it.first);
					}
					if (newFullyStaticSgIdxs.empty()) break;
					fullyStaticSgIdxs.insert(newFullyStaticSgIdxs.begin(), newFullyStaticSgIdxs.end());
					for (auto newFullyStaticSgIdx : newFullyStaticSgIdxs)
					{
						auto& fullyStaticProc = findProc(newFullyStaticSgIdx);
						for (auto referrer : fullyStaticProc.referrers)
						{
							auto sgIdx = std::get<0>(referrer);
							auto inIdx = std::get<1>(referrer);
							auto i = std::get<2>(referrer);
							auto& referrerProc = findProc(sgIdx);
							auto it1 = referrerProc.vertexIdxToOp.find(inIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
							if (it1 == referrerProc.vertexIdxToOp.end())
								throw std::runtime_error("it1 != otherProc.vertexIdxToOp.end()");
#endif
							auto& referrerOp = it1->second;
							auto it2 = referrerOp.connections.find(i);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
							if (it2 == referrerOp.connections.end())
								throw std::runtime_error("it2 != op.connections.end()");
#endif
							auto& referrerConn = it2->second;
							referrerConn.staticRef = true;
						}
					}
				}
			}

			void print(std::ostream& out, const std::map<size_t, size_t>& entryMap, bool inlineProcedures = false) const
			{
				if (inlineProcedures)
				{
					out << "struct ProcedureList : T::List<" << std::endl;
					for (auto it1 = procList.begin(); it1 != procList.end(); it1++)
					{
#ifdef PGA_GENERATE_ANNOTATED_CODE
						out << "/* procedure[" << std::distance(procList.begin(), it1) << "]= */ ";
#endif
						auto& proc = *it1;
						auto& op = proc.vertexIdxToOp.begin()->second;
						out << "Proc<" << PGA::Compiler::EnumUtils::toString(op.shapeType) << ", ";
						std::vector<std::tuple<long, long, size_t>> dynParams;
						std::vector<long> dynSuccs;
						std::vector<long> succs;
						print_visitOp(out, entryMap, proc, op, dynParams, dynSuccs, succs);
						out << ", " << proc.numThreads() << ">, " << std::endl;
					}
					out << "> {};" << std::endl;
				}
				else
				{
					for (auto it1 = procList.begin(); it1 != procList.end(); it1++)
					{
						auto& proc = *it1;
						auto& op = proc.vertexIdxToOp.begin()->second;
						out << "struct P" << (it1 - procList.begin()) << " : public ";
						std::vector<std::tuple<long, long, size_t>> dynParams;
						std::vector<long> dynSuccs;
						std::vector<long> succs;
						out << "Proc<" << PGA::Compiler::EnumUtils::toString(op.shapeType) << ", ";
						print_visitOp(out, entryMap, proc, op, dynParams, dynSuccs, succs);
						out << ", " << proc.numThreads() << "> {}; " << std::endl;
					}

					out << "struct ProcedureList : T::List<" << std::endl;
					for (auto it1 = procList.begin(); it1 != procList.end(); it1++)
					{
						auto& proc = *it1;
						auto& op = proc.vertexIdxToOp.begin()->second;
						out << "P" << (it1 - procList.begin()) << ", " << std::endl;
					}
					out << "> {};" << std::endl;
				}
			}

		private:
			const std::map<size_t, size_t>& subGraphMap;
			bool optimized;
			bool instrumented;
			std::vector<Procedure_> procList;

			Procedure_& getProc(size_t sgIdx)
			{
				auto it1 = opCodeMap.find(sgIdx);
				if (it1 == opCodeMap.end())
				{
					auto opCode = procList.size();
					opCodeMap.insert(std::make_pair(sgIdx, opCode));
					procList.emplace_back(Procedure_());
					return procList.back();
				}
				else
					return procList[it1->second];
			}

			Procedure_& findProc(size_t sgIdx)
			{
				auto it1 = opCodeMap.find(sgIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it1 == opCodeMap.end())
					throw std::runtime_error("it1 == opCodeMap.end()");
#endif
				return procList[it1->second];
			}

			void print_visitComponentSplit(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				// TODO:
				out << "ComponentSplit<false, ";
				if (op.numParams != 0)
					throw std::runtime_error("op.numParams != 0");
				if (op.numEdges != 3)
					throw std::runtime_error("op.numEdges != 3");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 1, dynParams, dynSuccs, succs);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 2, dynParams, dynSuccs, succs);
				out << ">";
			}

			void print_visitDiscard(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "Discard";
				if (op.numParams != 0)
					throw std::runtime_error("op.numParams != 0");
				if (op.numEdges != 0)
					throw std::runtime_error("op.numEdges != 0");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
			}

			void print_visitExtrude(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "Extrude<";
				if (op.numParams != 2)
					throw std::runtime_error("op.numParams != 2");
				if (op.numEdges != 1)
					throw std::runtime_error("op.numEdges != 1");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				print_visitParam(out, op, 0, dynParams);
				out << ", ";
				print_visitParam(out, op, 1, dynParams);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ">";
			}

			void print_visitCollider(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "Collider<";
				if (op.numParams != 1)
					throw std::runtime_error("op.numParams != 1");
				if (op.numEdges != 1)
					throw std::runtime_error("op.numEdges != 1");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				print_visitParam(out, op, 0, dynParams);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ">";
			}

			void print_visitGenerate(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "Generate<" << (proc.isParallel() ? "true" : "false") << ", " << op.genFuncIdx;
				if (op.numParams != 0)
					throw std::runtime_error("op.numParams != 0");
				if (op.numEdges != 0)
					throw std::runtime_error("op.numEdges != 0");
				if (op.numTermAttrs > 0)
				{
					out << ", ";
					for (size_t i = 0; i < op.numTermAttrs - 1; i++)
					{
						print_visitTermAttrs(out, op, i, dynParams);
						out << ", ";
					}
					print_visitTermAttrs(out, op, op.numTermAttrs - 1, dynParams);
				}
				out << ">";
			}

			void print_visitIf(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "If<";
				if (op.numParams != 1)
					throw std::runtime_error("op.numParams != 1");
				if (op.numEdges != 2)
					throw std::runtime_error("op.numEdges != 2");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				print_visitParam(out, op, 0, dynParams);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 1, dynParams, dynSuccs, succs);
				out << ">";
			}

			void print_visitIfSizeLess(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "IfSizeLess<";
				if (op.numParams != 2)
					throw std::runtime_error("op.numParams != 2");
				if (op.numEdges != 2)
					throw std::runtime_error("op.numEdges != 2");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				print_visitParam(out, op, 0, dynParams);
				out << ", ";
				print_visitParam(out, op, 1, dynParams);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 1, dynParams, dynSuccs, succs);
				out << ">";
			}

			void print_visitIfCollides(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "IfCollides<";
				if (op.numParams != 1)
					throw std::runtime_error("op.numParams != 1");
				if (op.numEdges != 2)
					throw std::runtime_error("op.numEdges != 2");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				print_visitParam(out, op, 0, dynParams);
				out << ", ";
				print_visitParam(out, op, 1, dynParams);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 1, dynParams, dynSuccs, succs);
				out << ">";
			}

			void print_visitRandomRule(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "RandomRule<";
				if (op.numParams != 1)
					throw std::runtime_error("op.numParams != 1");
				if (op.numEdges == 0)
					throw std::runtime_error("op.numEdges == 0");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				print_visitParam(out, op, 0, dynParams);
				out << ", ";
				for (auto i = 0; i < op.numEdges - 1; i++)
				{
					print_visitEdge(out, entryMap, proc, op, i, dynParams, dynSuccs, succs);
					out << ", ";
				}
				print_visitEdge(out, entryMap, proc, op, op.numEdges - 1, dynParams, dynSuccs, succs);
				out << ">";
			}

			void print_visitRepeat(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				// TODO:
				out << "Repeat<false, ";
				if (op.numParams != 3)
					throw std::runtime_error("op.numParams != 3");
				if (op.numEdges != 1)
					throw std::runtime_error("op.numEdges != 1");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				// TODO:
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				print_visitParam(out, op, 0, dynParams);
				out << ", ";
				print_visitParam(out, op, 1, dynParams);
				out << ", ";
				print_visitParam(out, op, 2, dynParams);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ">";
			}

			void print_visitRotate(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "Rotate<";
				if (op.numParams != 3)
					throw std::runtime_error("op.numParams != 3");
				if (op.numEdges != 1)
					throw std::runtime_error("op.numEdges != 1");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				print_visitParam(out, op, 0, dynParams);
				out << ", ";
				print_visitParam(out, op, 1, dynParams);
				out << ", ";
				print_visitParam(out, op, 2, dynParams);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ">";
			}

			void print_visitScale(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "Scale<";
				if (op.numParams != 3)
					throw std::runtime_error("op.numParams != 3");
				if (op.numEdges != 1)
					throw std::runtime_error("op.numEdges != 1");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				print_visitParam(out, op, 0, dynParams);
				out << ", ";
				print_visitParam(out, op, 1, dynParams);
				out << ", ";
				print_visitParam(out, op, 2, dynParams);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ">";
			}

			void print_visitSetAsDynamicConvexPolygon(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "SetAsDynamicConvexPolygon<";
				if (op.numParams < 3)
					throw std::runtime_error("op.numParams < 3");
				if (op.numEdges != 1)
					throw std::runtime_error("op.numEdges != 1");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				std::stringstream tmpOut;
				print_visitParam(tmpOut, op, 0, dynParams);
				tmpOut << ", ";
				if (optimized && op.numParams == op.params.size())
				{
					std::vector<math::float2> vertices;
					auto it = op.params.begin();
					while (it != op.params.end())
					{
						auto actualParam = it->second.lock();
						if (actualParam->getType() != ParameterType::PT_VEC2)
							throw std::runtime_error("actualParam->getType() != ParameterType::PT_VEC2");
						vertices.emplace_back(math::float2(static_cast<float>(actualParam->at(0)), static_cast<float>(actualParam->at(1))));
						it++;
					}
					std::vector<math::float2> orderedVertices;
					GeometryUtils::orderVertices_CCW(vertices, orderedVertices);
					for (auto i = 0; i < orderedVertices.size() - 1; i++)
					{
						print_paramVec2(tmpOut, orderedVertices[i]);
						tmpOut << ", ";
					}
				}
				else
				{
					for (auto i = 0; i < op.numParams - 1; i++)
					{
						print_visitParam(tmpOut, op, i, dynParams);
						tmpOut << ", ";
					}
					print_visitParam(tmpOut, op, op.numParams - 1, dynParams);
				}
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ", " << tmpOut.str();
				out << ">";
			}

			void print_visitSetAsDynamicConvexRightPrism(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "SetAsDynamicConvexRightPrism<";
				if (op.numParams < 3)
					throw std::runtime_error("op.numParams < 3");
				if (op.numEdges != 1)
					throw std::runtime_error("op.numEdges != 1");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				std::stringstream tmpOut;
				print_visitParam(tmpOut, op, 0, dynParams);
				tmpOut << ", ";
				if (optimized && op.numParams == op.params.size())
				{
					std::vector<math::float2> vertices;
					auto it = op.params.begin();
					while (it != op.params.end())
					{
						auto actualParam = it->second.lock();
						if (actualParam->getType() != ParameterType::PT_VEC2)
							throw std::runtime_error("actualParam->getType() != ParameterType::PT_VEC2");
						vertices.emplace_back(math::float2(static_cast<float>(actualParam->at(0)), static_cast<float>(actualParam->at(1))));
						it++;
					}
					std::vector<math::float2> orderedVertices;
					GeometryUtils::orderVertices_CCW(vertices, orderedVertices);
					for (auto i = 0; i < orderedVertices.size() - 1; i++)
					{
						print_paramVec2(tmpOut, orderedVertices[i]);
						tmpOut << ", ";
					}
				}
				else
				{
					for (auto i = 0; i < op.numParams - 1; i++)
					{
						print_visitParam(tmpOut, op, i, dynParams);
						tmpOut << ", ";
					}
					print_visitParam(tmpOut, op, op.numParams - 1, dynParams);
				}
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ", " << tmpOut.str();
				out << ">";
			}

			void print_visitSetAsDynamicConcavePolygon(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "SetAsDynamicConcavePolygon<";
				if (op.numParams < 3)
					throw std::runtime_error("op.numParams < 3");
				if (op.numEdges != 1)
					throw std::runtime_error("op.numEdges != 1");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				std::stringstream tmpOut;
				print_visitParam(tmpOut, op, 0, dynParams);
				tmpOut << ", ";
				if (optimized && op.numParams == op.params.size())
				{
					std::vector<math::float2> vertices;
					auto it = op.params.begin();
					while (it != op.params.end())
					{
						auto actualParam = it->second.lock();
						if (actualParam->getType() != ParameterType::PT_VEC2)
							throw std::runtime_error("actualParam->getType() != ParameterType::PT_VEC2");
						vertices.emplace_back(math::float2(static_cast<float>(actualParam->at(0)), static_cast<float>(actualParam->at(1))));
						it++;
					}
					std::vector<math::float2> nonCollinearVertices;
					GeometryUtils::removeCollinearPoints(vertices, nonCollinearVertices);
					for (auto i = 0; i < nonCollinearVertices.size() - 1; i++)
					{
						print_paramVec2(tmpOut, nonCollinearVertices[i]);
						tmpOut << ", ";
					}
					print_paramVec2(tmpOut, nonCollinearVertices[nonCollinearVertices.size() - 1]);
				}
				else
				{
					for (auto i = 0; i < op.numParams - 1; i++)
					{
						print_visitParam(tmpOut, op, i, dynParams);
						tmpOut << ", ";
					}
					print_visitParam(tmpOut, op, op.numParams - 1, dynParams);
				}
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ", " << tmpOut.str();
				out << ">";
			}

			void print_visitSetAsDynamicConcaveRightPrism(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "SetAsDynamicConcaveRightPrism<";
				if (op.numParams < 3)
					throw std::runtime_error("op.numParams < 3");
				if (op.numEdges != 1)
					throw std::runtime_error("op.numEdges != 1");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				std::stringstream tmpOut;
				print_visitParam(tmpOut, op, 0, dynParams);
				tmpOut << ", ";
				if (optimized && op.numParams == op.params.size())
				{
					std::vector<math::float2> vertices;
					auto it = op.params.begin();
					while (it != op.params.end())
					{
						auto actualParam = it->second.lock();
						if (actualParam->getType() != ParameterType::PT_VEC2)
							throw std::runtime_error("actualParam->getType() != ParameterType::PT_VEC2");
						vertices.emplace_back(math::float2(static_cast<float>(actualParam->at(0)), static_cast<float>(actualParam->at(1))));
						it++;
					}
					std::vector<math::float2> nonCollinearVertices;
					GeometryUtils::removeCollinearPoints(vertices, nonCollinearVertices);
					for (auto i = 0; i < nonCollinearVertices.size() - 1; i++)
					{
						print_paramVec2(tmpOut, nonCollinearVertices[i]);
						tmpOut << ", ";
					}
					print_paramVec2(tmpOut, nonCollinearVertices[nonCollinearVertices.size() - 1]);
				}
				else
				{
					for (auto i = 0; i < op.numParams - 1; i++)
					{
						print_visitParam(tmpOut, op, i, dynParams);
						tmpOut << ", ";
					}
					print_visitParam(tmpOut, op, op.numParams - 1, dynParams);
				}
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ", " << tmpOut.str();
				out << ">";
			}

			void print_visitSubdivide(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "Subdivide<";
				if (op.numParams != 1)
					throw std::runtime_error("op.numParams != 1");
				if (op.numEdges == 0)
					throw std::runtime_error("op.numEdges == 0");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				print_visitParam(out, op, 0, dynParams);
				out << ", ";
				for (auto i = 0; i < op.numEdges - 1; i++)
				{
					print_visitEdge(out, entryMap, proc, op, i, dynParams, dynSuccs, succs);
					out << ", ";
				}
				print_visitEdge(out, entryMap, proc, op, op.numEdges - 1, dynParams, dynSuccs, succs);
				out << ">";
			}

			void print_visitTranslate(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "Translate<";
				if (op.numParams != 3)
					throw std::runtime_error("op.numParams != 3");
				if (op.numEdges != 1)
					throw std::runtime_error("op.numEdges != 1");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				print_visitParam(out, op, 0, dynParams);
				out << ", ";
				print_visitParam(out, op, 1, dynParams);
				out << ", ";
				print_visitParam(out, op, 2, dynParams);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ">";
			}

			void print_visitSwapSize(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "SwapSize<";
				if (op.numParams != 3)
					throw std::runtime_error("op.numParams != 3");
				if (op.numEdges != 1)
					throw std::runtime_error("op.numEdges != 1");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				print_visitParam(out, op, 0, dynParams);
				out << ", ";
				print_visitParam(out, op, 1, dynParams);
				out << ", ";
				print_visitParam(out, op, 2, dynParams);
				out << ", ";
				print_visitEdge(out, entryMap, proc, op, 0, dynParams, dynSuccs, succs);
				out << ">";
			}

			void print_visitReplicate(
				std::ostream& out,
				const std::map<size_t, size_t>& entryMap,
				const Procedure_& proc,
				const Operator_& op,
				std::vector<std::tuple<long, long, size_t>>& dynParams,
				std::vector<long>& dynSuccs,
				std::vector<long>& succs) const
			{
				out << "Replicate<";
				if (op.numParams != 0)
					throw std::runtime_error("op.numParams != 0");
				if (op.numEdges == 0)
					throw std::runtime_error("op.numEdges == 0");
				if (op.numTermAttrs != 0)
					throw std::runtime_error("op.numTermAttrs != 0");
				if (proc.isParallel())
					throw std::runtime_error("proc.isParallel()");
				for (auto i = 0; i < op.numEdges - 1; i++)
				{
					print_visitEdge(out, entryMap, proc, op, i, dynParams, dynSuccs, succs);
					out << ", ";
				}
				print_visitEdge(out, entryMap, proc, op, op.numEdges - 1, dynParams, dynSuccs, succs);
				out << ">";
			}

			void print_visitOp(std::ostream& out, const std::map<size_t, size_t>& entryMap, const Procedure_& proc, const Operator_& op, std::vector<std::tuple<long, long, size_t>>& dynParams, std::vector<long>& dynSuccs, std::vector<long>& succs) const
			{
				switch (op.type)
				{
				case TRANSLATE:
					print_visitTranslate(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case ROTATE:
					print_visitRotate(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case SCALE:
					print_visitScale(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case EXTRUDE:
					print_visitExtrude(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case COMPSPLIT:
					print_visitComponentSplit(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case SUBDIV:
					print_visitSubdivide(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case REPEAT:
					print_visitRepeat(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case DISCARD:
					print_visitDiscard(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case IF:
					print_visitIf(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case IFSIZELESS:
					print_visitIfSizeLess(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case GENERATE:
					print_visitGenerate(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case STOCHASTIC:
					print_visitRandomRule(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case SET_AS_DYNAMIC_CONVEX_POLYGON:
					print_visitSetAsDynamicConvexPolygon(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM:
					print_visitSetAsDynamicConvexRightPrism(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case SET_AS_DYNAMIC_CONCAVE_POLYGON:
					print_visitSetAsDynamicConcavePolygon(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM:
					print_visitSetAsDynamicConcaveRightPrism(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case COLLIDER:
					print_visitCollider(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case IFCOLLIDES:
					print_visitIfCollides(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case SWAPSIZE:
					print_visitSwapSize(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				case REPLICATE:
					print_visitReplicate(out, entryMap, proc, op, dynParams, dynSuccs, succs);
					break;
				default:
					throw std::runtime_error("PGA::Compiler::CodeGenerator::ProcedureListCodeGenerator::print_visitOp(..): unknown operator type [op.type=" + std::to_string(op.type) + "]");
				}
			}

			void print_paramVec2(std::ostream& out, const math::float2& vertex) const
			{
				out << "Vec2<" << static_cast<int>(vertex.x) << ", " << static_cast<int>(vertex.y) << ">";
			}

			void print_visitParam(std::ostream& out, const Operator_& op, size_t i, std::vector<std::tuple<long, long, size_t>>& dynParams) const
			{
				if (optimized)
				{
					auto it = op.params.find(i);
					if (it != op.params.end())
					{
						it->second.lock()->print(out, false);
						return;
					}
				}
				out << "DynParam<" << getDynParamIdx(dynParams, static_cast<long>(op.vertexIdx), -1, i) << ">";
			}

			void print_visitTermAttrs(std::ostream& out, const Operator_& op, size_t i, std::vector<std::tuple<long, long, size_t>>& dynParams) const
			{
				if (optimized)
				{
					auto it = op.termAttrs.find(i);
					if (it != op.termAttrs.end())
					{
						int value;
						unsigned int NOoM;
						Scalar::getValueAndNegativeOrderOfMagnitude(it->second, value, NOoM);
						if (NOoM > 0)
							out << "Scalar<" << value << ", " << NOoM << ">";
						else
							out << "Scalar<" << value << ">";
						return;
					}
				}
				out << "DynParam<" << getDynParamIdx(dynParams, static_cast<long>(op.vertexIdx), -1, i) << ">";
			}

			size_t getSuccIdx(std::vector<long>& list, long edgeIdx) const
			{
				if (edgeIdx != -1)
				{
					auto it = std::find(list.begin(), list.end(), edgeIdx);
					if (it != list.end())
						return std::distance(list.begin(), it);
				}
				auto succIdx = list.size();
				list.push_back(edgeIdx);
				return succIdx;
			}

			size_t getDynParamIdx(std::vector<std::tuple<long, long, size_t>>& list, long vertexIdx, long edgeIdx, size_t i) const
			{
				if ((vertexIdx == -1) ^ (edgeIdx == -1))
				{
					auto it = std::find(list.begin(), list.end(), std::make_tuple(vertexIdx, edgeIdx, i));
					if (it != list.end())
						return std::distance(list.begin(), it);
				}
				auto paramIdx = list.size();
				list.push_back(std::make_tuple(vertexIdx, edgeIdx, i));
				return paramIdx;
			}

			void print_visitEdge(std::ostream& out, const std::map<size_t, size_t>& entryMap, const Procedure_& proc, const Operator_& op, size_t i, std::vector<std::tuple<long, long, size_t>>& dynParams, std::vector<long>& dynSuccs, std::vector<long>& succs) const
			{
				auto it2 = op.connections.find(i);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it2 == op.connections.end())
					throw std::runtime_error("it2 == op.connections.end()");
#endif
				auto conn = it2->second;
				if (conn.requiresParam)
				{
					out << "T::Pair<";
					if (!conn.dynParam && optimized)
					{
						auto edgeParam = conn.param.lock();
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (edgeParam == nullptr)
							throw std::runtime_error("edgeParam == nullptr");
#endif
						edgeParam->print(out);
					}
					else
						out << "DynParam<" << getDynParamIdx(dynParams, -1, conn.edgeIdx, i) << ">";
					out << ", ";
				}

				if (conn.cutEdge)
				{
					if (optimized)
					{
						if (conn.staticRef)
						{
							auto it1 = opCodeMap.find(conn.out);
							if (it1 == opCodeMap.end())
							{
								auto it2 = subGraphMap.find(conn.out);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
								if (it2 == subGraphMap.end())
									throw std::runtime_error("it2 == subGraphMap.end()");
#endif
								it1 = opCodeMap.find(it2->second);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
								if (it1 == opCodeMap.end())
									throw std::runtime_error("it1 == opCodeMap.end()");
#endif
							}
							auto opCode = it1->second;
							auto& otherProc = procList[it1->second];
							if (otherProc.isFullyStatic())
							{
								if (instrumented)
								{
									auto it2 = entryMap.find(conn.out);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
									if (it2 == entryMap.end())
										throw std::runtime_error("it2 == entryMap.end()");
#endif
									out << "FSCall<" << opCode << ", " << otherProc.vertexIdxToOp.begin()->second.phase << ", " << getSuccIdx(succs, conn.edgeIdx) << ", " << it2->second << ">";
								}
								else
									out << "FSCall<" << opCode << ", " << otherProc.vertexIdxToOp.begin()->second.phase << ">";
							}
							else
							{
								if (instrumented)
									out << "PSCall<" << opCode << ", " << getSuccIdx(dynSuccs, conn.edgeIdx) << ", " << getSuccIdx(succs, conn.edgeIdx) << ">";
								else
									out << "PSCall<" << opCode << ", " << getSuccIdx(dynSuccs, conn.edgeIdx) << ">";
							}
						}
						else
						{
							if (conn.commonEdge)
							{
								auto it1 = opCodeMap.find(static_cast<size_t>(conn.out));
								if (it1 == opCodeMap.end())
								{
									auto it2 = subGraphMap.find(static_cast<size_t>(conn.out));
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
									if (it2 == subGraphMap.end())
										throw std::runtime_error("it2 == subGraphMap.end()");
#endif
									it1 = opCodeMap.find(it2->second);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
									if (it1 == opCodeMap.end())
										throw std::runtime_error("it1 == opCodeMap.end()");
#endif
								}
								auto opCode = it1->second;

#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
								auto& otherProc = procList[it1->second];
								if (otherProc.isFullyStatic())
									throw std::runtime_error("otherProc.isFullyStatic()");
#endif
								if (instrumented)
									out << "PSCall<" << opCode << ", " << getSuccIdx(dynSuccs, conn.edgeIdx) << ", " << getSuccIdx(succs, conn.edgeIdx) << ">";
								else
									out << "PSCall<" << opCode << ", " << getSuccIdx(dynSuccs, conn.edgeIdx) << ">";
							}
							else
							{
								if (instrumented)
									out << "DCall<" << getSuccIdx(dynSuccs, conn.edgeIdx) << ", " << getSuccIdx(succs, conn.edgeIdx) << ">";
								else
									out << "DCall<" << getSuccIdx(dynSuccs, conn.edgeIdx) << ">";
							}
						}
					}
					else
					{
						if (instrumented)
							out << "DCall<" << getSuccIdx(dynSuccs, conn.edgeIdx) << ", " << getSuccIdx(succs, conn.edgeIdx) << ">";
						else
							out << "DCall<" << getSuccIdx(dynSuccs, conn.edgeIdx) << ">";
					}
				}
				else
				{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (conn.out == -1)
						throw std::runtime_error("conn.out == -1");
#endif
					auto it3 = proc.vertexIdxToOp.find(conn.out);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (it3 == proc.vertexIdxToOp.end())
						throw std::runtime_error("it3 == proc.vertexIdxToOp.end()");
#endif
					print_visitOp(out, entryMap, proc, it3->second, dynParams, dynSuccs, succs);
				}
				if (conn.requiresParam)
					out << ">";
			}

		};

		struct DispatchTableCodeGenerator : MatchGroupVisitor
		{
			std::map<size_t, size_t> entryMap;

			DispatchTableCodeGenerator(const std::map<size_t, size_t>& subGraphMap, const std::map<size_t, size_t>& opCodeMap, const std::set<size_t>& fullyStaticSgIdxs, bool optimized, bool instrumented) : subGraphMap(subGraphMap), opCodeMap(opCodeMap), fullyStaticSgIdxs(fullyStaticSgIdxs), optimized(optimized), instrumented(instrumented) {}

			virtual void visitVertex(
				size_t sgIdx,
				size_t vertexIdx,
				OperatorType opType,
				ShapeType shapeType,
				int phase,
				long genFuncIdx,
				size_t numParams,
				const std::map<size_t, std::weak_ptr<Parameter>>& eqParams,
				const std::map<size_t, std::weak_ptr<Parameter>>& diffParams,
				size_t numTermParams,
				const std::map<size_t, double>& eqTermParams,
				const std::map<size_t, double>& diffTermParams,
				size_t numEdges
				)
			{
				auto it1 = subGraphMap.find(sgIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it1 == subGraphMap.end())
					throw std::runtime_error("it2 == subGraphMap.end()");
#endif
				auto currSgIdx = it1->second;
				if (!instrumented && sgIdx != currSgIdx && fullyStaticSgIdxs.find(currSgIdx) != fullyStaticSgIdxs.end())
					return;
				auto it2 = opCodeMap.find(currSgIdx);
				auto opCode = it2->second;
				auto it3 = entryMap.find(sgIdx);
				size_t entryIdx;
				if (it3 == entryMap.end())
				{
					entryIdx = entryMap.size();
					Entry newEntry;
					newEntry.opCode = opCode;
					newEntry.phase = phase;
					entries.emplace_back(newEntry);
					entryMap[sgIdx] = entryIdx;
				}
				else
					entryIdx = it3->second;
				auto& entry = entries[entryIdx];
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (entry.opCode != opCode)
					throw std::runtime_error("entry.opCode != opCode");
#endif
				if (optimized)
				{
					if ((opType == SET_AS_DYNAMIC_CONVEX_POLYGON || opType == SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM) &&
						diffParams.size() == numParams)
					{
						std::vector<math::float2> unorderedVertices;
						auto it = diffParams.begin();
						while (++it != diffParams.end())
						{
							auto actualParam = it->second.lock();
							if (actualParam->getType() != ParameterType::PT_VEC2)
								throw std::runtime_error("PGA::Compiler::CodeGenerator::FromPartition::DispatchTableCodeGenerator::visitVertex(): actualParam->getType() != ParameterType::PT_VEC2");
							unorderedVertices.emplace_back(math::float2(static_cast<float>(actualParam->at(0)), static_cast<float>(actualParam->at(1))));
						}
						std::vector<math::float2> orderedVertices;
						GeometryUtils::orderVertices_CCW(unorderedVertices, orderedVertices);
						for (auto& vertex : orderedVertices)
						{
							auto tmpParam = std::shared_ptr<Vec2>(new Vec2((double)vertex.x, (double)vertex.y));
							entry.tmpParams.emplace_back(tmpParam);
							entry.params.emplace_back(tmpParam);
						}
					}
					else
					{
						for (auto& diffParam : diffParams)
							entry.params.emplace_back(diffParam.second);
						for (auto& diffTermParam : diffTermParams)
						{
							auto termParam = std::shared_ptr<Parameter>(new Scalar(diffTermParam.second));
							entry.termAttrs.emplace_back(termParam);
							entry.params.emplace_back(termParam);
						}
					}
				}
				else
				{
					if (opType == SET_AS_DYNAMIC_CONVEX_POLYGON || opType == SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM)
					{
						std::vector<math::float2> unorderedVertices;
						for (auto i = 0; i < numParams; i++)
						{
							auto it = eqParams.find(i);
							if (it == eqParams.end())
							{
								it = diffParams.find(i);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
								if (it == diffParams.end())
									throw std::runtime_error("it == diffParams.end()");
#endif
							}
							auto actualParam = it->second.lock();
							if (actualParam->getType() != ParameterType::PT_VEC2)
								throw std::runtime_error("PGA::Compiler::CodeGenerator::FromPartition::DispatchTableCodeGenerator::visitVertex(): actualParam->getType() != ParameterType::PT_VEC2");
							unorderedVertices.emplace_back(math::float2(static_cast<float>(actualParam->at(0)), static_cast<float>(actualParam->at(1))));
						}
						std::vector<math::float2> orderedVertices;
						GeometryUtils::orderVertices_CCW(unorderedVertices, orderedVertices);
						for (auto& vertex : orderedVertices)
						{
							auto tmpParam = std::shared_ptr<Vec2>(new Vec2((double)vertex.x, (double)vertex.y));
							entry.tmpParams.emplace_back(tmpParam);
							entry.params.emplace_back(tmpParam);
						}
					}
					else
					{
						for (auto i = 0; i < numParams; i++)
						{
							auto it = eqParams.find(i);
							if (it == eqParams.end())
							{
								it = diffParams.find(i);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
								if (it == diffParams.end())
									throw std::runtime_error("it == diffParams.end()");
#endif
							}
							entry.params.emplace_back(it->second);
						}
						for (auto i = 0; i < numTermParams; i++)
						{
							auto it = eqTermParams.find(i);
							if (it == eqTermParams.end())
							{
								it = diffTermParams.find(i);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
								if (it == diffTermParams.end())
									throw std::runtime_error("it == diffTermParams.end()");
#endif
							}
							auto attrParam = std::shared_ptr<Scalar>(new Scalar(it->second));
							entry.termAttrs.emplace_back(attrParam);
							entry.params.emplace_back(attrParam);
						}
					}
				}
			}

			virtual void visitEdge(
				size_t sgIdx,
				size_t inIdx,
				long outIdx,
				long edgeIdx,
				bool cutEdge,
				bool commonEdge,
				bool requiresParam,
				bool dynParam,
				const std::weak_ptr<Parameter>& param,
				size_t i
				)
			{
				auto it1 = subGraphMap.find(sgIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it1 == subGraphMap.end())
					throw std::runtime_error("it2 == subGraphMap.end()");
#endif
				auto currSgIdx = it1->second;
				if (!instrumented && sgIdx != currSgIdx && fullyStaticSgIdxs.find(currSgIdx) != fullyStaticSgIdxs.end())
					return;
				auto it2 = entryMap.find(sgIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it2 == entryMap.end())
					throw std::runtime_error("it2 == dstIdxToEntry.end()");
#endif
				auto& entry = entries[it2->second];
				if (requiresParam && (dynParam || !optimized))
					entry.params.emplace_back(param);

				if (cutEdge)
					entry.cutEdges.emplace_back(CutEdge(edgeIdx, outIdx, commonEdge));
			}

			void print(std::ostream& out) const
			{
				out << "DispatchTable dispatchTable = {" << std::endl;
				for (auto it1 = entryMap.begin(); it1 != entryMap.end(); it1++)
				{
					auto& entry = entries[it1->second];
					auto entryIdx = std::distance(entryMap.begin(), it1);
#ifdef PGA_GENERATE_ANNOTATED_CODE
					out << "/* entries[" << entryIdx << "]= */";
#endif
					out << "{ ";
#ifdef PGA_GENERATE_ANNOTATED_CODE
					out << "/* procIdx= */";
#endif
					out << entry.opCode << ", ";
#ifdef PGA_GENERATE_ANNOTATED_CODE
					out << "/* parameters[" << entry.params.size() << "]= */";
#endif
					out << "{ ";
					if (!entry.params.empty())
					{
						auto it2 = entry.params.begin();
						for (size_t j = 0; j < entry.params.size() - 1; j++, it2++)
						{
#ifdef PGA_GENERATE_ANNOTATED_CODE
							out << "/* " << j << " */ ";
#endif
							writeParameter(out, *it2);
							out << ", ";
						}
#ifdef PGA_GENERATE_ANNOTATED_CODE
						out << "/* " << (entry.params.size() - 1) << " */ ";
#endif
						writeParameter(out, *it2);
					}
					out << " }, ";
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (entry.cutEdges.size() != entry.successors.size())
						throw std::runtime_error("entry.cutEdges.size() != entry.successors.size()");
#endif
					std::vector<std::string> successorStrs;
					for (auto& successor : entry.successors)
					{
						if (successor.ignore) continue;
						// TODO: add phase support
						//successorStrs.emplace_back("{ " + std::to_string(successor.entryIdx) + ", 0 }");
						successorStrs.emplace_back("{ " + std::to_string(successor.entryIdx) + ", " + std::to_string(successor.phaseIndex) + " }");
					}
#ifdef PGA_GENERATE_ANNOTATED_CODE
					out << "/* successors[" << successorStrs.size() << "]= */";
#endif
					out << "{ ";
					if (!successorStrs.empty())
					{
						for (size_t j = 0; j < successorStrs.size() - 1; j++)
						{
#ifdef PGA_GENERATE_ANNOTATED_CODE
							out << "/* " << j << " */ ";
#endif
							out << successorStrs[j] << ", ";
						}
#ifdef PGA_GENERATE_ANNOTATED_CODE
						out << "/* " << successorStrs.size() - 1 << " */ ";
#endif
						out << successorStrs[successorStrs.size() - 1];
					}
					out << " }";
					if (instrumented)
					{
						out << ", ";
#ifdef PGA_GENERATE_ANNOTATED_CODE
						out << "/* edgeIndices[" << entry.cutEdges.size() << "]= */";
#endif
						out << "{ ";
						std::vector<std::string> edgeIdxsStrs;
						for (auto& cutEdge : entry.cutEdges)
						{
							auto edgeIdxStr = std::to_string(cutEdge.edgeIdx);
							edgeIdxsStrs.emplace_back(edgeIdxStr);
						}
						if (!edgeIdxsStrs.empty())
						{
							for (size_t j = 0; j < edgeIdxsStrs.size() - 1; j++)
								out << edgeIdxsStrs[j] << ", ";
							out << edgeIdxsStrs[edgeIdxsStrs.size() - 1];
						}
						out << " }, ";
#ifdef PGA_GENERATE_ANNOTATED_CODE
						out << "/* subgraphIndex= */";
#endif
						out << it1->first;
					}
					out << " }, " << std::endl;
				}
				out << " };" << std::endl;
			}

			void resolveSuccessors()
			{
				for (auto& entry : entries)
				{
					for (auto& cutEdge : entry.cutEdges)
					{
						long entryIdx;
						bool ignore;
						if (cutEdge.out == -1)
						{
							entryIdx = -1;
							ignore = optimized && cutEdge.common;
						}
						else
						{
							auto outIdx = static_cast<size_t>(cutEdge.out);
							auto it1 = entryMap.find(outIdx);
							if (it1 == entryMap.end())
							{
								auto it2 = subGraphMap.find(outIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
								if (it2 == subGraphMap.end())
									throw std::runtime_error("it2 == subGraphMap.end()");
#endif
								it1 = entryMap.find(it2->second);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
								if (it1 == entryMap.end())
									throw std::runtime_error("it1 == entryMap.end()");
#endif
							}
							entryIdx = static_cast<long>(it1->second);
							ignore = false;
							if (optimized && cutEdge.common)
							{
								auto it2 = subGraphMap.find(outIdx);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
								if (it2 == subGraphMap.end())
									throw std::runtime_error("it2 == subGraphMap.end()");
#endif
								auto it3 = fullyStaticSgIdxs.find(it2->second);
								if (it3 != fullyStaticSgIdxs.end())
									ignore = true;
							}
						}
						if (entryIdx >= 0)
							entry.successors.emplace_back(Successor(entryIdx, entries.at(entryIdx).phase, ignore));
						else
							entry.successors.emplace_back(Successor(entryIdx, entry.phase, ignore));
						//entry.successors.emplace_back(Successor(entryIdx, 0, ignore));
					}
				}
			}

		private:
			struct CutEdge
			{
				long edgeIdx;
				long out;
				bool common;

				CutEdge(long edgeIdx, long out, bool common) : edgeIdx(edgeIdx), out(out), common(common) {}

			};

			struct Successor
			{
				long entryIdx;
				int phaseIndex;
				bool ignore;

				Successor(long entryIdx, int phaseIndex, bool ignore) : entryIdx(entryIdx), phaseIndex(phaseIndex), ignore(ignore) {}

			};

			struct Entry
			{
				size_t opCode;
				int phase;
				std::vector<std::shared_ptr<Parameter>> tmpParams;
				std::vector<std::shared_ptr<Parameter>> termAttrs;
				std::vector<std::weak_ptr<Parameter>> params;
				std::vector<CutEdge> cutEdges;
				std::vector<Successor> successors;

			};

			const std::map<size_t, size_t>& subGraphMap;
			const std::map<size_t, size_t>& opCodeMap;
			const std::set<size_t>& fullyStaticSgIdxs;
			bool optimized;
			bool instrumented;
			std::vector<Entry> entries;

			void writeParameter(std::ostream& out, const std::weak_ptr<Parameter>& param) const
			{
				auto p = param.lock();
				if (p == nullptr)
				{
					out << "{ PT_SCALAR, { 0 } }";
					return;
				}
				out << "{ ";
				switch (p->getType())
				{
				case ParameterType::PT_SCALAR:
					out << "PT_SCALAR";
					break;
				case ParameterType::PT_RAND:
					out << "PT_RAND";
					break;
				case ParameterType::PT_SHAPE_ATTR:
					out << "PT_SHAPE_ATTR";
					break;
				case ParameterType::PT_EXP:
					out << "PT_EXP";
					break;
				case ParameterType::PT_VEC2:
					out << "PT_VEC2";
					break;
				default:
					throw std::runtime_error("PGA::Compiler::CodeGenerator::DispatchTableCodeGenerator::writeParameter(): unknown parameter type");
				}
				out << ", { ";
				if (p->getType() == ParameterType::PT_EXP)
				{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (p->size() > 1)
						throw std::runtime_error("p->size() > 1");
#endif
					unsigned int parameterLength = p->getParameterLength();
					unsigned int bufferLength = parameterLength + 1;
					if (bufferLength > Constants::MaxNumParameterValues)
						throw std::runtime_error("PGA::Compiler::CodeGenerator::DispatchTableCodeGenerator::writeParameter(): bufferLength > PGA::Constants::MaxNumParameterValues");
					float values[Constants::MaxNumParameterValues];
					values[0] = static_cast<float>(parameterLength);
					p->encode(values + 1, 0);
					for (unsigned int i = 0; i < bufferLength - 1; i++)
						out << values[i] << ", ";
					out << values[bufferLength - 1];
				}
				else
				{
					if (p->size() > 0)
					{
						for (auto i = 0; i < p->size() - 1; i++)
							out << p->at(i) << ", ";
						out << p->at(p->size() - 1);
					}
				}
				out << " } }";
			}

		};

		void CodeGenerator::fromPartition(std::ostream& out, const Graph::PartitionPtr& partition)
		{
			bool unused;
			fromPartition(out, partition, unused);
		}

		void CodeGenerator::fromPartition(std::ostream& out, const Graph::PartitionPtr& partition, bool optimized, bool instrumented)
		{
			bool unused;
			fromPartition(out, partition, optimized, instrumented, unused);
		}

		void CodeGenerator::fromPartition(std::ostream& out, const Graph::PartitionPtr& partition, bool& staticFirstProcedure)
		{
			fromPartition(out, partition, false, false, staticFirstProcedure);
		}

		void CodeGenerator::fromPartition(std::ostream& out, const Graph::PartitionPtr& partition, bool optimized, bool instrumented, bool& staticFirstProcedure)
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (partition == nullptr)
				throw std::runtime_error("partition == nullptr");
			if (partition->matchGroups == nullptr)
				throw std::runtime_error("partition.matchGroups == nullptr");
#endif
			ProcedureListCodeGenerator procedureCodeGenerator(partition->matchGroups->mapping, optimized, instrumented);
			partition->matchGroups->traverse(procedureCodeGenerator);
			std::set<size_t> fullyStaticSgIdxs;
			procedureCodeGenerator.resolveFullyStaticReferences(fullyStaticSgIdxs);

			int firstSgIdx = -1;
			for (auto& entry : procedureCodeGenerator.opCodeMap)
			{
				if (entry.second == 0)
				{
					firstSgIdx = static_cast<int>(entry.first);
					break;
				}
			}
			assert(firstSgIdx > -1);
			staticFirstProcedure = fullyStaticSgIdxs.find(firstSgIdx) != fullyStaticSgIdxs.end();

			DispatchTableCodeGenerator dispatchTableCodeGenerator(partition->matchGroups->mapping, procedureCodeGenerator.opCodeMap, fullyStaticSgIdxs, optimized, instrumented);
			partition->matchGroups->traverse(dispatchTableCodeGenerator, false);
			dispatchTableCodeGenerator.resolveSuccessors();

			procedureCodeGenerator.print(out, dispatchTableCodeGenerator.entryMap); /* */
			dispatchTableCodeGenerator.print(out); /* */
		}

	}

}