#include <string>
#include <map>
#include <tuple>

#include <boost/variant/get.hpp>

#include <pga/core/Axis.h>
#include <pga/compiler/EnumUtils.h>
#include <pga/compiler/Symbol.h>

#include "ParserUtils.h"
#include "ParserSymbol.h"
#include "ParserOperator.h"

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct OperatorDescription
			{
				int numParameters;
				int numSuccessors;
				bool hasSuccessorParameters;

			};

			//////////////////////////////////////////////////////////////////////////
			// NOTE: Has to follow the same order of the enumerator OperatorType
			static OperatorDescription operatorDescriptions[] =
			{
				/* RESERVED */ { 0, 0, false },
				/* TRANSLATE */ { 3, 1, false },
				/* ROTATE */ { 3, 1, false },
				/* SCALE */ { 3, 1, false },
				/* EXTRUDE */ { 1, 1, false },
				/* COMPSPLIT */ { 0, 3, false },
				/* SUBDIV */ { 1, -1, true },
				/* REPEAT */ { 3, 1, false },
				/* DISCARD */ { 0, 0, false },
				/* IF */ { 1, 2, false },
				/* IFSIZELESS */ { 2, 2, false },
				/* IFCOLLIDES */ { 1, 3, false },
				/* GENERATE */ { 0, 0, false },
				/* STOCHASTIC */ { 1, -1, true },
				/* SET_AS_DYNAMIC_CONVEX_POLYGON */ { 3, 1, false },
				/* SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM */ { 3, 1, false },
				/* SET_AS_DYNAMIC_CONCAVE_POLYGON */ { 3, 1, false },
				/* SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM */ { 3, 1, false},
				/* COLLIDER */ { 1, 1, false },
				/* SWAPSIZE */ { 3, 1, false },
				/* REPLICATE */ { 0, -1, false }
			};

			//////////////////////////////////////////////////////////////////////////
			std::shared_ptr<PGA::Compiler::Parameter> createOperatorParameter(Operand operand, OperatorType operatorType, size_t idx)
			{
				switch (operatorType)
				{
				case PGA::Compiler::TRANSLATE:
					if (idx >= 3)
						throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): Translate accepts only 3 parameters");
					return toParameter(operand);
				case PGA::Compiler::ROTATE:
					if (idx >= 3)
						throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): Rotate accepts only 3 parameters");
					return toParameter(operand);
				case PGA::Compiler::SCALE:
					if (idx >= 3)
						throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): Scale accepts only 3 parameters");
					return toParameter(operand);
				case PGA::Compiler::EXTRUDE:
					// FIXME:
					if (idx == 0)
						return toParameter(operand);
					else
						throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): Extrude accepts only 1 parameter");
					break;
				case PGA::Compiler::COMPSPLIT:
					throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): CompSplit accepts no parameters");
					break;
				case PGA::Compiler::SUBDIV:
					if (idx >= 1)
						throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): SubDiv accepts only 1 parameter");
					return toParameter(operand);
				case PGA::Compiler::REPEAT:
					if (idx < 3)
						return toParameter(operand);
					else
						throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): Repeat accepts only 3 parameter");
					break;
				case PGA::Compiler::DISCARD:
					throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): Discard accepts no parameters");
					break;
				case PGA::Compiler::IF:
					if (idx == 0)
						return toParameter(operand);
					else
						throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): If accepts only 1 parameter");
					break;
				case PGA::Compiler::IFSIZELESS:
					if (idx < 2)
						return toParameter(operand);
					else
						throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): IfSizeLess accepts only 2 parameter");
					break;
				case PGA::Compiler::IFCOLLIDES:
					if (idx == 0)
						return toParameter(operand);
					else
						throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): IfCollides accepts only 1 parameter");
					break;
				case PGA::Compiler::GENERATE:
					throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): Generate accepts no parameters");
					break;
				case PGA::Compiler::STOCHASTIC:
					if (idx == 0)
						return toParameter(operand);
					else
						throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): RandomRule accepts only 1 parameter");
					break;
				case PGA::Compiler::SET_AS_DYNAMIC_CONVEX_POLYGON:
				case PGA::Compiler::SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM:
				case PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_POLYGON:
				case PGA::Compiler::SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM:
					return toParameter(operand);
				case PGA::Compiler::COLLIDER:
					if (idx == 0)
						return toParameter(operand);
					else
						throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): Collider accepts only 1 parameter");
					break;
				case PGA::Compiler::SWAPSIZE:
					if (idx >= 3)
						throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): SwapSize accepts only 3 parameters");
					return toParameter(operand);
				case PGA::Compiler::REPLICATE:
					throw std::runtime_error("PGA::Compiler::Grammar::createOperatorParameter(): Replicate accepts no parameters");
					break;
				default:
					break;
				}
				// FIXME:
				return std::shared_ptr<PGA::Compiler::Parameter>();
			}

			//////////////////////////////////////////////////////////////////////////
			void Operator::convertToAbstraction
			(
				std::vector<PGA::Compiler::Terminal>& terminals, 
				PGA::Compiler::Operator& operator_, 
				PGA::Compiler::Rule& rule
			) const
			{
				OperatorType operatorType = static_cast<OperatorType>(this->operator_);
				operator_.type = operatorType;

				if (operatorType == OperatorType::IFCOLLIDES)
					rule.containsIfCollide = true;

				for (auto param_it = parameters.begin(); param_it != parameters.end(); ++param_it)
					operator_.operatorParams.push_back(createOperatorParameter(*param_it, operatorType, std::distance(parameters.begin(), param_it)));

				// FIXME: adding missing fields that should be declared in the grammar...
				if (operatorType == OperatorType::EXTRUDE)
					operator_.operatorParams.insert(operator_.operatorParams.begin(), std::shared_ptr<PGA::Compiler::Parameter>((new PGA::Compiler::Axis(Z))));

				for (auto it1 = successors.begin(); it1 != successors.end(); ++it1)
				{
					for (auto it2 = it1->parameters.begin(); it2 != it1->parameters.end(); ++it2)
						operator_.successorParams.push_back(createOperatorParameter(*it2, operatorType, std::distance(it1->parameters.begin(), it2)));

					size_t successorIdx = operator_.successors.size();
					std::string terminalName;
					switch (it1->successor.which())
					{
					case 0:
					{
						terminalName = trim(boost::get<Symbol>(it1->successor).symbol);
						std::shared_ptr<PGA::Compiler::Symbol> newSymbol(new PGA::Compiler::Symbol(terminalName));
						newSymbol->id = nextSuccessorIdx();
						newSymbol->myrule = operator_.myrule;
						for (auto it3 = terminals.begin(); it3 != terminals.end(); ++it3)
						{
							if (std::find(it3->symbols.begin(), it3->symbols.end(), terminalName) != it3->symbols.end())
							{
								for (const auto& termAttr : it3->parameters)
									newSymbol->terminalAttributes.push_back(termAttr);
								break;
							}
						}
						operator_.successors.push_back(newSymbol);
						break;
					}
					case 1:
					{
						Operator op = boost::get<Operator>(it1->successor);
						std::shared_ptr<PGA::Compiler::Operator> newOperator(new PGA::Compiler::Operator());
						newOperator->id = nextSuccessorIdx();
						newOperator->myrule = operator_.myrule;
						terminalName = operator_.myrule;
						for (auto it3 = terminals.begin(); it3 != terminals.end(); ++it3)
						{
							if (std::find(it3->symbols.begin(), it3->symbols.end(), terminalName) != it3->symbols.end())
							{
								newOperator->type = OperatorType::GENERATE;
								for (const auto& termAttr : it3->parameters)
									newOperator->terminalAttributes.push_back(termAttr);
								break;
							}
						}
						operator_.successors.push_back(newOperator);
						op.convertToAbstraction(terminals, *newOperator, rule);
						break;
					}
					}
				}
			}

			bool Operator::check(Logger& logger)
			{
				bool successor = true;
				auto operatorType = static_cast<OperatorType>(operator_);
				auto operatorDescription = operatorDescriptions[operatorType];

				if (parameters.size() < operatorDescription.numParameters)
				{
					logger.addMessage(Logger::LL_ERROR, "(line = %d, column = %d) %s requires at least %d parameters. %d parameters given", line + 1, column, PGA::Compiler::EnumUtils::toString(operatorType).c_str(), operatorDescription.numParameters, parameters.size());
					successor = false;
				}

				if (operatorDescription.numSuccessors != -1 && operatorDescription.numSuccessors != successors.size())
				{
					logger.addMessage(Logger::LL_ERROR, "(line = %d, column = %d) %s requires %d successors but %d successors were given", line + 1, column, PGA::Compiler::EnumUtils::toString(operatorType).c_str(), operatorDescription.numSuccessors, parameters.size());
					successor = false;
				}

				for (auto it = successors.begin(); it != successors.end(); ++it)
				{
					if (operatorDescription.hasSuccessorParameters)
					{
						if (it->parameters.size() == 0)
						{
							logger.addMessage(Logger::LL_ERROR, "(line = %d, column = %d) operator %s requires successor parameters", it->line + 1, it->column, PGA::Compiler::EnumUtils::toString(operatorType).c_str());
							successor = false;
						}
					}
					else
					{
						if (it->parameters.size() > 0)
						{
							logger.addMessage(Logger::LL_ERROR, "(line = %d, column = %d) operator %s cannot have successor parameters", it->line + 1, it->column, PGA::Compiler::EnumUtils::toString(operatorType).c_str());
							successor = false;
						}
					}
				}

				for (auto it = successors.begin(); it != successors.end(); ++it)
				{
					if (it->successor.which() == 1)
					{
						Operator succOp = boost::get<Operator>(it->successor);
						successor = succOp.check(logger);
						return successor;
					}
				}

				return successor;
			}

		}

	}

}