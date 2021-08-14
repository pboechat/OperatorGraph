#include <pga/compiler/DispatchTableGenerator.h>
#include <pga/compiler/EnumUtils.h>
#include <pga/compiler/Logger.h>
#include <pga/core/DispatchTableEntry.h>
#include <pga/core/GeometryUtils.h>

#include <algorithm>
#include <stdexcept>

namespace PGA
{
	namespace Compiler
	{
		struct SymbolComparer
		{
			SymbolComparer(const std::shared_ptr<Symbol>& baseLine) : baseLine(baseLine) {}

			bool operator()(const std::shared_ptr<Symbol>& entry)
			{
				return entry->name == baseLine->name;
			}

		private:
			const std::shared_ptr<Symbol>& baseLine;

		};

		bool DispatchTableGenerator::visitSuccessor(size_t entryIdx, ShapeType shapeType, const std::shared_ptr<Successor>& successor, const ProcedureList& procedures, Logger& logger, std::deque<std::shared_ptr<Symbol>>& symbolsToVisit, const std::set<std::string>& visitedSymbols)
		{
			if (successor->getType() == SuccessorType::SYMBOL)
				return visitSymbol(entryIdx, shapeType, std::dynamic_pointer_cast<Symbol>(successor), logger, symbolsToVisit, visitedSymbols);
			else
				return visitOperator(entryIdx, shapeType, std::dynamic_pointer_cast<Operator>(successor), procedures, logger, symbolsToVisit, visitedSymbols);
		}

		bool DispatchTableGenerator::visitSymbol(size_t entryIdx, ShapeType shapeType, const std::shared_ptr<Symbol>& symbol, Logger& logger, std::deque<std::shared_ptr<Symbol>>& symbolsToVisit, const std::set<std::string>& visitedSymbols)
		{
			auto it1 = symbolToInputShape.find(symbol->name);
			if (it1 != symbolToInputShape.end())
			{
				if (it1->second != shapeType)
				{
					logger.addMessage(Logger::LL_ERROR, "%s expects shape of type %s but is actually getting a shape of type %s", symbol->name.c_str(), PGA::Compiler::EnumUtils::toString(it1->second).c_str(), PGA::Compiler::EnumUtils::toString(shapeType).c_str());
					return false;
				}
			}
			else
				symbolToInputShape.emplace(symbol->name, shapeType);
			if (visitedSymbols.find(symbol->name) == visitedSymbols.end() &&
				std::find_if(symbolsToVisit.begin(), symbolsToVisit.end(), SymbolComparer(symbol)) == symbolsToVisit.end())
				symbolsToVisit.emplace_back(symbol);
			return true;
		}

		bool DispatchTableGenerator::visitOperator(size_t entryIdx, ShapeType inputShape, const std::shared_ptr<Operator>& op, const ProcedureList &procedureList, Logger& logger, std::deque<std::shared_ptr<Symbol>>& symbolsToVisit, const std::set<std::string>& visitedSymbols)
		{
			size_t genFuncIdx = 0;
			std::vector<double> termAttrs;
			if (op->type == GENERATE)
			{
				size_t termIdx;
				auto it1 = op->terminalAttributes.begin();
				if (it1 != op->terminalAttributes.end())
				{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (*it1 < 0)
						throw std::runtime_error("*it1 < 0");
#endif
					genFuncIdx = static_cast<size_t>(*it1);
					if (++it1 != op->terminalAttributes.end())
					{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (*it1 < 0)
							throw std::runtime_error("*it1 < 0");
#endif
						termIdx = static_cast<size_t>(*it1);
						termAttrs.resize(op->terminalAttributes.size() - 1);
						std::copy(it1, op->terminalAttributes.end(), termAttrs.begin());
					}
					else
					{
						termIdx = 0;
						termAttrs.push_back(static_cast<double>(termIdx));
					}
				}
				else
				{
					genFuncIdx = 0;
					termIdx = 0;
					termAttrs.push_back(static_cast<double>(termIdx));
				}
#ifdef VERBOSE
				std::cout << "terminal [generate_operator] [genFuncIdx=" << genFuncIdx << ", termIdx=" << termIdx << ", inputShape=" << PGA::Compiler::EnumUtils::toString(inputShape) << "]" << std::endl;
#endif
			}

			long procIdx = procedureList.indexOf(SingleOperatorProcedure(op->type, inputShape, genFuncIdx));
			if (procIdx == -1)
			{
				logger.addMessage(Logger::LL_ERROR, "Unsupported procedure (operatorType=%s, inputShape=%s, genFuncIdx=%d)", PGA::Compiler::EnumUtils::toString(op->type).c_str(), PGA::Compiler::EnumUtils::toString(inputShape).c_str(), genFuncIdx);
				return false;
			}
			proceduresUsed.emplace(SingleOperatorProcedure(op->type, inputShape, genFuncIdx));

			dispatchTable->entries[entryIdx].operatorCode = static_cast<size_t>(procIdx);

			std::vector<size_t> succEntryIdxs;
			std::vector<unsigned int> succPhases;
			for (auto i = 0; i < op->successors.size(); i++)
			{
				size_t succEntryIdx;
				// NOTE: the next phase is either the successor phase (if succPhase > 0), or the current phase (if currPhase > 0) or 0 (if currPhase < 0)
				auto currPhase = op->phase;
				auto succPhase = op->successors[i]->phase;
				unsigned int nextPhase = ((succPhase < 0) ? ((currPhase < 0) ? 0 : currPhase) : succPhase);
				if (op->successors[i]->getType() == SuccessorType::SYMBOL)
				{
					auto symbol = std::dynamic_pointer_cast<Symbol>(op->successors[i]);
					std::string nextSymbolStr = symbol->name;
					auto it1 = symbolToEntryIdx.find(nextSymbolStr);
					if (it1 == symbolToEntryIdx.end())
					{
						symbolsToVisit.emplace_back(symbol);
						auto nextShapeType = Operator::nextShapeType(op->type, inputShape, i);
						symbolToInputShape.emplace(nextSymbolStr, nextShapeType);
						succEntryIdx = dispatchTable->entries.size();
						dispatchTable->entries.resize(succEntryIdx + 1);
						symbolToEntryIdx.emplace(nextSymbolStr, succEntryIdx);
					}
					else
						succEntryIdx = it1->second;
				}
				else
				{
					succEntryIdx = dispatchTable->entries.size();
					dispatchTable->entries.resize(succEntryIdx + 1);
				}
				succEntryIdxs.push_back(succEntryIdx);
				succPhases.push_back(nextPhase);
			}

			switch (op->type)
			{
			case TRANSLATE:
				buildTranslateEntry(dispatchTable->entries[entryIdx], op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case ROTATE:
				buildRotateEntry(dispatchTable->entries[entryIdx], op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case SCALE:
				buildScaleEntry(dispatchTable->entries[entryIdx], op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case COLLIDER:
				buildColliderEntry(dispatchTable->entries[entryIdx], op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case EXTRUDE:
				buildExtrudeEntry(dispatchTable->entries[entryIdx], op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case COMPSPLIT:
				buildComponentSplitEntry(dispatchTable->entries[entryIdx], succEntryIdxs, succPhases);
				break;
			case SUBDIV:
				buildSubdivideEntry(dispatchTable->entries[entryIdx], op->operatorParams, op->successorParams, succEntryIdxs, succPhases);
				break;
			case REPEAT:
				buildRepeatEntry(dispatchTable->entries[entryIdx], op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case DISCARD:
				buildDiscardEntry(dispatchTable->entries[entryIdx]);
				break;
			case IF:
				buildIfEntry(dispatchTable->entries[entryIdx], op->operatorParams, succEntryIdxs[0], succEntryIdxs[1], succPhases);
				break;
			case IFCOLLIDES:
				buildIfCollidesEntry(dispatchTable->entries[entryIdx], op->operatorParams, succEntryIdxs[0], succEntryIdxs[1], succPhases);
				break;
			case IFSIZELESS:
				buildIfSizeLessEntry(dispatchTable->entries[entryIdx], op->operatorParams, succEntryIdxs[0], succEntryIdxs[1], succPhases);
				break;
			case STOCHASTIC:
				buildRandomRuleEntry(dispatchTable->entries[entryIdx], op->operatorParams, op->successorParams, succEntryIdxs, succPhases);
				break;
			case SET_AS_DYNAMIC_CONVEX_POLYGON:
				buildSetAsDynamicConvexPolygonEntry(dispatchTable->entries[entryIdx], op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM:
				buildSetAsDynamicConvexRightPrismEntry(dispatchTable->entries[entryIdx], op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case SET_AS_DYNAMIC_CONCAVE_POLYGON:
				buildSetAsDynamicConcavePolygonEntry(dispatchTable->entries[entryIdx], op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM:
				buildSetAsDynamicConcaveRightPrismEntry(dispatchTable->entries[entryIdx], op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case GENERATE:
				buildGenerateEntry(dispatchTable->entries[entryIdx], termAttrs);
				break;
			case SWAPSIZE:
				buildSwapSizeEntry(dispatchTable->entries[entryIdx], op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case REPLICATE:
				buildReplicateEntry(dispatchTable->entries[entryIdx], op->operatorParams, op->successorParams, succEntryIdxs, succPhases);
				break;
			default:
				throw std::runtime_error("PGA::Compiler::DispatchTableGenerator::visitOperator(..): unknown operator type [op->type=" + std::to_string(op->type) + "]");
			}

			for (auto i = 0; i < op->successors.size(); i++)
				if (!visitSuccessor(succEntryIdxs[i], Operator::nextShapeType(op->type, inputShape, i), op->successors[i], procedureList, logger, symbolsToVisit, visitedSymbols))
					return false;

			return true;
		}

		bool DispatchTableGenerator::visitTerminalSymbol(size_t entryIdx, ShapeType inputShape, const std::shared_ptr<Symbol>& symbol, const ProcedureList& procedures, Logger& logger, const std::set<std::string>& visitedSymbols)
		{
			size_t genFuncIdx;
			size_t termIdx;
			std::vector<double> termAttrs;
			auto it1 = symbol->terminalAttributes.begin();
			if (it1 != symbol->terminalAttributes.end())
			{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (*it1 < 0)
					throw std::runtime_error("*it1 < 0");
#endif
				genFuncIdx = static_cast<size_t>(*it1);
				if (++it1 != symbol->terminalAttributes.end())
				{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (*it1 < 0)
						throw std::runtime_error("*it1 < 0");
#endif
					termIdx = static_cast<size_t>(*it1);
					termAttrs.resize(symbol->terminalAttributes.size() - 1);
					std::copy(it1, symbol->terminalAttributes.end(), termAttrs.begin());
				}
				else
				{
					termIdx = 0;
					termAttrs.push_back(static_cast<double>(termIdx));
				}
			}
			else
			{
				genFuncIdx = 0;
				termIdx = 0;
				termAttrs.push_back(static_cast<double>(termIdx));
			}
#ifdef VERBOSE
			std::cout << "terminal " << symbol->name << " [genFuncIdx=" << genFuncIdx << ", termIdx=" << termIdx << ", inputShape=" << PGA::Compiler::EnumUtils::toString(inputShape) << "]" << std::endl;
#endif
			long procIdx = procedures.indexOf(SingleOperatorProcedure(GENERATE, inputShape, genFuncIdx));
			if (procIdx == -1)
			{
				logger.addMessage(Logger::LL_ERROR, "Unsupported procedure (operatorType=Generate, inputShape=%s, genFuncIdx=%d)", PGA::Compiler::EnumUtils::toString(inputShape).c_str(), genFuncIdx);
				return false;
			}
			proceduresUsed.emplace(SingleOperatorProcedure(GENERATE, inputShape, genFuncIdx));
			dispatchTable->entries[entryIdx].operatorCode = static_cast<size_t>(procIdx);
			buildGenerateEntry(dispatchTable->entries[entryIdx], termAttrs);
			terminalSymbols.insert(symbol->name);
			return true;
		}

		void DispatchTableGenerator::buildGenerateEntry(DispatchTable::Entry& newEntry, const std::vector<double>& termAttrs)
		{
			// NOTE: we only support up to 5 terminal attributes so far (6 = 5 actual terminal attributes + terminal idx)
			if (termAttrs.size() > 6)
				throw std::runtime_error("DPT Builder: too many generate attributes");
			newEntry.parameters.resize((termAttrs.empty()) ? 1 : termAttrs.size());
			newEntry.parameters[0].type = ParameterType::PT_SCALAR;
			// NOTE: the first item on the terminal attributes list is the terminal idx
			newEntry.parameters[0].values.push_back((termAttrs.empty()) ? 0.0f : static_cast<float>(termAttrs[0]));
			for (size_t i = 1; i < termAttrs.size(); i++)
			{
				newEntry.parameters[i].type = ParameterType::PT_SCALAR;
				newEntry.parameters[i].values.push_back(termAttrs[i]);
			}
		}

		void DispatchTableGenerator::buildTranslateEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase)
		{
			if (params.size() < 3)
				throw std::runtime_error("PGA::Compiler::DispatchTableGenerator::buildTranslateEntry(): expected 3 parameters, got " + std::to_string(params.size()));
			newEntry.parameters.resize(3);
			newEntry.parameters[0] = params[0]->toDispatchTableParameter();
			newEntry.parameters[1] = params[1]->toDispatchTableParameter();
			newEntry.parameters[2] = params[2]->toDispatchTableParameter();
			newEntry.successors.resize(1);
			newEntry.successors[0].entryIndex = static_cast<int>(succEntryIdx);
			newEntry.successors[0].phaseIndex = succPhase;
		}

		void DispatchTableGenerator::buildRotateEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase)
		{
			if (params.size() < 3)
				throw std::runtime_error("PGA::Compiler::DispatchTableGenerator::buildRotateEntry(): expected 3 parameters, got " + std::to_string(params.size()));
			newEntry.parameters.resize(3);
			newEntry.parameters[0] = params[0]->toDispatchTableParameter();
			newEntry.parameters[1] = params[1]->toDispatchTableParameter();
			newEntry.parameters[2] = params[2]->toDispatchTableParameter();
			newEntry.successors.resize(1);
			newEntry.successors[0].entryIndex = static_cast<int>(succEntryIdx);
			newEntry.successors[0].phaseIndex = succPhase;
		}

		void DispatchTableGenerator::buildScaleEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase)
		{
			if (params.size() < 3)
				throw std::runtime_error("PGA::Compiler::DispatchTableGenerator::buildScaleEntry(): expected 3 parameters, got " + std::to_string(params.size()));
			newEntry.parameters.resize(3);
			newEntry.parameters[0] = params[0]->toDispatchTableParameter();
			newEntry.parameters[1] = params[1]->toDispatchTableParameter();
			newEntry.parameters[2] = params[2]->toDispatchTableParameter();
			newEntry.successors.resize(1);
			newEntry.successors[0].entryIndex = static_cast<int>(succEntryIdx);
			newEntry.successors[0].phaseIndex = succPhase;
		}

		void DispatchTableGenerator::buildColliderEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase)
		{
			if (params.empty())
				throw std::runtime_error("PGA::Compiler::DispatchTableGenerator::buildColliderEntry(): expected 1 parameter, got 0");
			newEntry.parameters.resize(1);
			newEntry.parameters[0] = params[0]->toDispatchTableParameter();
			newEntry.successors.resize(1);
			newEntry.successors[0].entryIndex = static_cast<int>(succEntryIdx);
			newEntry.successors[0].phaseIndex = succPhase;
		}

		void DispatchTableGenerator::buildExtrudeEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase)
		{
			if (params.size() < 2)
				throw std::runtime_error("PGA::Compiler::DispatchTableGenerator::buildExtrudeEntry(): expected 2 parameters, got " + params.size());
			newEntry.parameters.resize(2);
			// NOTE: currently not using param[0] (axis) 
			newEntry.parameters[0] = params[0]->toDispatchTableParameter();
			newEntry.parameters[1] = params[1]->toDispatchTableParameter();
			newEntry.successors.resize(1);
			newEntry.successors[0].entryIndex = static_cast<int>(succEntryIdx);
			newEntry.successors[0].phaseIndex = succPhase;
		}

		void DispatchTableGenerator::buildComponentSplitEntry(DispatchTable::Entry& newEntry, const std::vector<size_t>& succEntryIdxs, std::vector<unsigned int> succPhases)
		{
			newEntry.successors.resize(succEntryIdxs.size());
			for (size_t i = 0; i < succEntryIdxs.size(); i++)
			{
				newEntry.successors[i].entryIndex = static_cast<int>(succEntryIdxs[i]);
				newEntry.successors[i].phaseIndex = succPhases[i];
			}
		}

		void DispatchTableGenerator::buildSubdivideEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, const std::vector<std::shared_ptr<Parameter>>& factors, const std::vector<size_t>& succEntryIdxs, std::vector<unsigned int> succPhases)
		{
			newEntry.parameters.resize(factors.size() + 1);
			newEntry.parameters[0].type = ParameterType::PT_SCALAR;
			if (newEntry.parameters[0].values.size() > 0)
				newEntry.parameters[0].values.at(0) = static_cast<float>(params[0]->at(0));
			else
				newEntry.parameters[0].values.push_back(static_cast<float>(params[0]->at(0)));
			float relativeValuesSum = 0.0f;
			float absoluteValuesSum = 0.0f;
			for (size_t i = 0, j = 1; i < factors.size(); i = j, j++)
			{
				float value = static_cast<float>(factors[i]->at(0));
				newEntry.parameters[j].type = ParameterType::PT_SCALAR;
				newEntry.parameters[j].values.push_back(value);
				newEntry.parameters[j].values.push_back(absoluteValuesSum);
				newEntry.parameters[j].values.push_back(relativeValuesSum);
				if (value > 0) absoluteValuesSum += value;
				else relativeValuesSum -= value;
			}
			newEntry.parameters[0].values.push_back(absoluteValuesSum);
			newEntry.parameters[0].values.push_back(relativeValuesSum);
			newEntry.successors.resize(succEntryIdxs.size());
			for (size_t i = 0; i < succEntryIdxs.size(); i++)
			{
				newEntry.successors[i].entryIndex = static_cast<int>(succEntryIdxs[i]);
				newEntry.successors[i].phaseIndex = succPhases[i];
			}
		}

		void DispatchTableGenerator::buildRepeatEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase)
		{
			if (params.size() < 3)
				throw std::runtime_error("PGA::Compiler::DispatchTableGenerator::buildRepeatEntry(): expected 3 parameters, got " + std::to_string(params.size()));
			newEntry.parameters.resize(3);
			newEntry.parameters[0] = params[0]->toDispatchTableParameter();
			newEntry.parameters[1] = params[1]->toDispatchTableParameter();
			newEntry.parameters[2] = params[2]->toDispatchTableParameter();
			newEntry.successors.resize(1);
			newEntry.successors[0].entryIndex = static_cast<int>(succEntryIdx);
			newEntry.successors[0].phaseIndex = succPhase;
		}

		void DispatchTableGenerator::buildDiscardEntry(DispatchTable::Entry& newEntry)
		{
		}

		void DispatchTableGenerator::buildIfEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdxIfTrue, size_t succEntryIdxIfFalse, std::vector<unsigned int> succPhases)
		{
			if (params.empty())
				throw std::runtime_error("PGA::Compiler::DispatchTableGenerator::buildIfEntry(): expected 1 parameter, got 0");
			newEntry.parameters.resize(1);
			newEntry.parameters[0] = params[0]->toDispatchTableParameter();
			newEntry.successors.resize(2);
			newEntry.successors[0].entryIndex = static_cast<int>(succEntryIdxIfTrue);
			newEntry.successors[0].phaseIndex = succPhases[0];
			newEntry.successors[1].entryIndex = static_cast<int>(succEntryIdxIfFalse);
			newEntry.successors[1].phaseIndex = succPhases[0];
		}

		void DispatchTableGenerator::buildIfCollidesEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdxIfTrue, size_t succEntryIdxIfFalse, std::vector<unsigned int> succPhases)		
		{
			if (params.empty())
				throw std::runtime_error("PGA::Compiler::DispatchTableGenerator::buildIfCollidesEntry(): expected 1 parameter, got 0");
			newEntry.parameters.resize(1);
			newEntry.parameters[0] = params[0]->toDispatchTableParameter();
			newEntry.successors.resize(2);
			newEntry.successors[0].entryIndex = static_cast<int>(succEntryIdxIfTrue);
			newEntry.successors[0].phaseIndex = succPhases[0];
			newEntry.successors[1].entryIndex = static_cast<int>(succEntryIdxIfFalse);
			newEntry.successors[1].phaseIndex = succPhases[0];
		}

		void DispatchTableGenerator::buildIfSizeLessEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdxIfTrue, size_t succEntryIdxIfFalse, std::vector<unsigned int> succPhases)
		{
			if (params.size() < 2)
				throw std::runtime_error("PGA::Compiler::DispatchTableGenerator::buildIfSizeLessEntry(): expected 2 parameters, got " + params.size());
			newEntry.parameters.resize(2);
			newEntry.parameters[0] = params[0]->toDispatchTableParameter();
			newEntry.parameters[1] = params[1]->toDispatchTableParameter();
			newEntry.successors.resize(2);
			newEntry.successors[0].entryIndex = static_cast<int>(succEntryIdxIfTrue);
			newEntry.successors[0].phaseIndex = succPhases[0];
			newEntry.successors[1].entryIndex = static_cast<int>(succEntryIdxIfFalse);
			newEntry.successors[1].phaseIndex = succPhases[0];
		}

		void DispatchTableGenerator::buildSetAsDynamicConvexPolygonEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase)
		{
			std::vector<math::float2> vertices;
			for (unsigned int i = 0; i < params.size(); i++)
				vertices.push_back(math::float2(static_cast<float>(params[i]->at(0)), static_cast<float>(params[i]->at(1))));
			std::vector<math::float2> orderedPoints;
			newEntry.parameters.resize(params.size());
			GeometryUtils::orderVertices_CCW(vertices, orderedPoints);
			for (unsigned int i = 0; i < orderedPoints.size(); i++)
			{
				newEntry.parameters[i].type = ParameterType::PT_VEC2;
				newEntry.parameters[i].values.push_back(orderedPoints[i].x);
				newEntry.parameters[i].values.push_back(orderedPoints[i].y);
			}
			newEntry.successors.resize(1);
			newEntry.successors[0].entryIndex = static_cast<int>(succEntryIdx);
			newEntry.successors[0].phaseIndex = succPhase;
		}

		void DispatchTableGenerator::buildSetAsDynamicConvexRightPrismEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase)
		{
			std::vector<math::float2> vertices;
			for (unsigned int i = 0; i < params.size(); i++)
				vertices.push_back(math::float2(static_cast<float>(params[i]->at(0)), static_cast<float>(params[i]->at(1))));
			std::vector<math::float2> orderedPoints;
			GeometryUtils::orderVertices_CCW(vertices, orderedPoints);
			newEntry.parameters.resize(params.size());
			for (unsigned int i = 0; i < orderedPoints.size(); i++)
			{
				newEntry.parameters[i].type = ParameterType::PT_VEC2;
				newEntry.parameters[i].values.push_back(orderedPoints[i].x);
				newEntry.parameters[i].values.push_back(orderedPoints[i].y);
			}
			newEntry.successors.resize(1);
			newEntry.successors[0].entryIndex = static_cast<int>(succEntryIdx);
			newEntry.successors[0].phaseIndex = succPhase;
		}

		void DispatchTableGenerator::buildSetAsDynamicConcavePolygonEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase)
		{
			std::vector<math::float2> vertices;
			for (unsigned int i = 0; i < params.size(); i++)
				vertices.push_back(math::float2(static_cast<float>(params[i]->at(0)), static_cast<float>(params[i]->at(1))));
			std::vector<math::float2> nonCollinearVertices;
			GeometryUtils::removeCollinearPoints(vertices, nonCollinearVertices);
			newEntry.parameters.resize(params.size());
			for (unsigned int i = 0; i < nonCollinearVertices.size(); i++)
			{
				newEntry.parameters[i].type = ParameterType::PT_VEC2;
				newEntry.parameters[i].values.push_back(nonCollinearVertices[i].x);
				newEntry.parameters[i].values.push_back(nonCollinearVertices[i].y);
			}
			newEntry.successors.resize(1);
			newEntry.successors[0].entryIndex = static_cast<int>(succEntryIdx);
			newEntry.successors[0].phaseIndex = succPhase;
		}

		void DispatchTableGenerator::buildSetAsDynamicConcaveRightPrismEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase)
		{
			std::vector<math::float2> vertices;
			for (unsigned int i = 0; i < params.size(); i++)
				vertices.push_back(math::float2(static_cast<float>(params[i]->at(0)), static_cast<float>(params[i]->at(1))));
			std::vector<math::float2> nonCollinearVertices;
			GeometryUtils::removeCollinearPoints(vertices, nonCollinearVertices);
			newEntry.parameters.resize(params.size());
			for (unsigned int i = 0; i < nonCollinearVertices.size(); i++)
			{
				newEntry.parameters[i].type = ParameterType::PT_VEC2;
				newEntry.parameters[i].values.push_back(nonCollinearVertices[i].x);
				newEntry.parameters[i].values.push_back(nonCollinearVertices[i].y);
			}
			newEntry.successors.resize(1);
			newEntry.successors[0].entryIndex = static_cast<int>(succEntryIdx);
			newEntry.successors[0].phaseIndex = succPhase;
		}

		void DispatchTableGenerator::buildSwapSizeEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, size_t succEntryIdx, unsigned int succPhase)
		{
			if (params.size() < 3)
				throw std::runtime_error("PGA::Compiler::DispatchTableGenerator::buildSwapSizeEntry(): expected 3 parameters, got " + params.size());
			newEntry.parameters.resize(3);
			newEntry.parameters[0] = params[0]->toDispatchTableParameter();
			newEntry.parameters[1] = params[1]->toDispatchTableParameter();
			newEntry.parameters[2] = params[2]->toDispatchTableParameter();
			newEntry.successors.resize(1);
			newEntry.successors[0].entryIndex = static_cast<int>(succEntryIdx);
			newEntry.successors[0].phaseIndex = succPhase;
		}

		void DispatchTableGenerator::buildReplicateEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params, const std::vector<std::shared_ptr<Parameter>>& factors, const std::vector<size_t>& succEntryIdxs, std::vector<unsigned int> succPhases)
		{
			newEntry.successors.resize(succEntryIdxs.size());
			for (size_t i = 0; i < succEntryIdxs.size(); i++)
			{
				newEntry.successors[i].entryIndex = static_cast<int>(succEntryIdxs[i]);
				newEntry.successors[i].phaseIndex = succPhases[i];
			}
		}

		void DispatchTableGenerator::buildRandomRuleEntry(DispatchTable::Entry& newEntry, const std::vector<std::shared_ptr<Parameter>>& params,
			const std::vector<std::shared_ptr<Parameter>>& factors, const std::vector<size_t>& succEntryIdxs, std::vector<unsigned int> succPhases)
		{
			newEntry.parameters.resize(factors.size() + 1);
			newEntry.parameters[0] = params[0]->toDispatchTableParameter();
			for (unsigned int i = 0; i < factors.size(); i++)
				newEntry.parameters[i + 1] = factors[i]->toDispatchTableParameter();
			newEntry.successors.resize(succEntryIdxs.size());
			for (size_t i = 0; i < succEntryIdxs.size(); i++)
			{
				newEntry.successors[i].entryIndex = static_cast<int>(succEntryIdxs[i]);
				newEntry.successors[i].phaseIndex = succPhases[i];
			}
		}

		bool DispatchTableGenerator::fromRuleSet(const std::vector<Axiom>& axioms, const std::vector<Rule>& rules, const ProcedureList& procedures, Logger& logger)
		{
			dispatchTable = std::unique_ptr<DispatchTable>(new DispatchTable());
			symbolToEntryIdx.clear();
			symbolToInputShape.clear();
			terminalSymbols.clear();

			std::map<std::string, Rule> ruleMap;
			for (auto& rule : rules)
				ruleMap[rule.symbol] = rule;

			std::deque<std::shared_ptr<Symbol>> symbolsToVisit;
			for (auto& axiom : axioms)
			{
				symbolToInputShape.emplace(axiom.symbol, axiom.shapeType);
				auto it1 = ruleMap.find(axiom.symbol);
				if (it1 != ruleMap.end())
					symbolsToVisit.emplace_back(std::shared_ptr<Symbol>(new Symbol(axiom.symbol)));
				else
					logger.addMessage(Logger::LL_WARNING, "Axiom '%s' doesn't have a matching rule", axiom.symbol.c_str());
			}

			std::set<std::string> visitedSymbols;
			while (!symbolsToVisit.empty())
			{
				auto symbol = symbolsToVisit.front();
				symbolsToVisit.pop_front();
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (visitedSymbols.find(symbol->name) != visitedSymbols.end())
					throw std::runtime_error("visitedSymbols.find(symbol->name) != visitedSymbols.end()");
#endif
				size_t entryIdx;
				auto it1 = symbolToEntryIdx.find(symbol->name);
				if (it1 == symbolToEntryIdx.end())
				{
					entryIdx = dispatchTable->entries.size();
					dispatchTable->entries.resize(entryIdx + 1);
					symbolToEntryIdx.emplace(symbol->name, entryIdx);
				}
				else
					entryIdx = it1->second;

				auto it2 = symbolToInputShape.find(symbol->name);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
				if (it2 == symbolToInputShape.end())
					throw std::runtime_error("it2 == symbolToInputShape.end()");
#endif

				auto it3 = ruleMap.find(symbol->name);
				if (it3 == ruleMap.end())
				{
					if (!visitTerminalSymbol(entryIdx, it2->second, symbol, procedures, logger, visitedSymbols))
						return false;
				}
				else
				{
					if (!visitSuccessor(entryIdx, it2->second, it3->second.successor, procedures, logger, symbolsToVisit, visitedSymbols))
						return false;
				}
				visitedSymbols.emplace(symbol->name);
			}

			return true;
		}

		bool DispatchTableGenerator::fromBaseGraph(const PGA::Compiler::Graph& graph, const ProcedureList& procedureList, Logger& logger)
		{
			dispatchTable = std::unique_ptr<DispatchTable>(new DispatchTable());
			dispatchTable->entries.resize(graph.numVertices());
			symbolToEntryIdx.clear();
			symbolToInputShape.clear();
			terminalSymbols.clear();
			BaseGraphVisitor visitor(*this, procedureList, logger);
			return graph.depthFirst(visitor);
		}

		bool DispatchTableGenerator::BaseGraphVisitor::visit(size_t i, const Edge_LW& edge, const Vertex_LW& src, const Vertex_LW& dst)
		{
			return true;
		}

		bool DispatchTableGenerator::BaseGraphVisitor::visit(size_t i, const Vertex_LW& vertex)
		{
			std::vector<size_t> succEntryIdxs;
			std::vector<unsigned int> succPhases;
			entryIndex = static_cast<int>(i);
			Operator* op = vertex.getOperator();
			std::vector<Vertex*> children;
			vertex.getChildVertices(children);
			for (auto it = children.begin(); it != children.end(); ++it)
			{
				Vertex* child = *it;
				succEntryIdxs.push_back(child->index);
				succPhases.push_back(child->op->phase);
			}
			size_t genFuncIdx = 0;
			if (op->type == GENERATE)
			{
				size_t termIdx;
				auto it1 = op->terminalAttributes.begin();
				if (it1 != op->terminalAttributes.end())
				{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
					if (*it1 < 0)
						throw std::runtime_error("*it1 < 0");
#endif
					genFuncIdx = static_cast<size_t>(*it1);
					if (++it1 != op->terminalAttributes.end())
					{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
						if (*it1 < 0)
							throw std::runtime_error("*it1 < 0");
#endif
						termIdx = static_cast<size_t>(*it1);
					}
				}
				else
				{
					genFuncIdx = 0;
					termIdx = 0;
				}
#ifdef VERBOSE
				std::cout << "terminal Vertex" << i << " [genFuncIdx=" << genFuncIdx << ", termIdx=" << termIdx << "]" << std::endl;
#endif
			}
			long procIdx = procedureList.indexOf(SingleOperatorProcedure(op->type, op->shapeType, genFuncIdx));
			if (procIdx == -1)
			{
				logger.addMessage(Logger::LL_ERROR, "Unsupported procedure (operatorType=%s, shapeType=%s, genFuncIdx=%d)", PGA::Compiler::EnumUtils::toString(op->type).c_str(), PGA::Compiler::EnumUtils::toString(op->shapeType).c_str(), genFuncIdx);
				return false;
			}
			parent.proceduresUsed.emplace(SingleOperatorProcedure(op->type, op->shapeType, genFuncIdx));
			parent.buildEntryFromGraph(entryIndex, procIdx, op, succEntryIdxs, succPhases);
			return true;
		}

		void DispatchTableGenerator::buildEntryFromGraph(int entryIndex, long procIdx, const Operator* op, const std::vector<size_t> succEntryIdxs, std::vector<unsigned int> succPhases)
		{
			DispatchTable::Entry& newEntry = dispatchTable->entries[entryIndex];
			dispatchTable->entries[entryIndex].operatorCode = static_cast<size_t>(procIdx);
			switch (op->type)
			{
			case TRANSLATE:
				buildTranslateEntry(newEntry, op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case ROTATE:
				buildRotateEntry(newEntry, op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case SCALE:
				buildScaleEntry(newEntry, op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case COLLIDER:
				buildColliderEntry(newEntry, op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case EXTRUDE:
				buildExtrudeEntry(newEntry, op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case COMPSPLIT:
				buildComponentSplitEntry(newEntry, succEntryIdxs, succPhases);
				break;
			case SUBDIV:
				buildSubdivideEntry(newEntry, op->operatorParams, op->successorParams, succEntryIdxs, succPhases);
				break;
			case REPEAT:
				buildRepeatEntry(newEntry, op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case DISCARD:
				buildDiscardEntry(newEntry);
				break;
			case IF:
				buildIfEntry(newEntry, op->operatorParams, succEntryIdxs[0], succEntryIdxs[1], succPhases);
				break;
			case IFCOLLIDES:
				buildIfCollidesEntry(newEntry, op->operatorParams, succEntryIdxs[0], succEntryIdxs[1], succPhases);
				break;
			case IFSIZELESS:
				buildIfSizeLessEntry(newEntry, op->operatorParams, succEntryIdxs[0], succEntryIdxs[1], succPhases);
				break;
			case STOCHASTIC:
				buildRandomRuleEntry(newEntry, op->operatorParams, op->successorParams, succEntryIdxs, succPhases);
				break;
			case SET_AS_DYNAMIC_CONVEX_POLYGON:
				buildSetAsDynamicConvexPolygonEntry(newEntry, op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM:
				buildSetAsDynamicConvexRightPrismEntry(newEntry, op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case SET_AS_DYNAMIC_CONCAVE_POLYGON:
				buildSetAsDynamicConcavePolygonEntry(newEntry, op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM:
				buildSetAsDynamicConcaveRightPrismEntry(newEntry, op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case GENERATE:
				buildGenerateEntry(newEntry, op->termAttrs);
				break;
			case SWAPSIZE:
				buildSwapSizeEntry(newEntry, op->operatorParams, succEntryIdxs[0], succPhases[0]);
				break;
			case REPLICATE:
				buildReplicateEntry(newEntry, op->operatorParams, op->successorParams, succEntryIdxs, succPhases);
				break;
			default:
				throw std::runtime_error("PGA::Compiler::DispatchTableGenerator::buildEntryFromGraph(..): unknown operator type [op->type=" + std::to_string(op->type) + "]");
			}
		}

	}

}
