#include "ParserOperator.h"
#include "ParserProductionRule.h"
#include "ParserSymbol.h"
#include "ParserUtils.h"

#include <boost/variant/get.hpp>
#include <pga/compiler/Operator.h>
#include <pga/compiler/Symbol.h>

#include <algorithm>
#include <cctype>
#include <functional>

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			bool ProductionRule::check(Logger& logger)
			{
				double probabilitySum = 0;
				unsigned int else_idx = 0;

				for (auto it = successors.begin(); it != successors.end(); ++it)
				{
					double tmp_prob = 0;
					unsigned int idx = static_cast<unsigned int>(std::distance(successors.begin(), it));
					switch (it->which())
					{
					case 0:
					{
						Symbol& sym = boost::get<Symbol>(*it);
						tmp_prob = sym.probability;
						if (tmp_prob == -1)
							if (else_idx != 0)
							{
							logger.addMessage(Logger::LL_ERROR, "(line = %d, column = %d) multiple *else* branches in stochastic rule (%s)", line + 1, column, symbol.c_str());
							return false;
							}
							else
								else_idx = idx;
						else
							probabilitySum += tmp_prob;
						break;
					}
					case 1:
					{
						Operator& op = boost::get<Operator>(*it);
						tmp_prob = op.probability;
						if (tmp_prob == -1)
							if (else_idx != 0)
							{
							logger.addMessage(Logger::LL_ERROR, "(line = %d, column = %d) multiple *else* branches in stochastic rule (%s)", line + 1, column, symbol.c_str());
							return false;
							}
							else
								else_idx = idx;
						else
						{
							probabilitySum += tmp_prob;

							if (op.check(logger) == false)
								return false;

						}
						break;
					}
					}
				}

				if (probabilitySum >= 100)
				{
					logger.addMessage(Logger::LL_ERROR, "probability sum of stochastic rule(%s) is %d (more than 100%)", symbol.c_str(), probabilitySum);
					return false;
				}
				else
				{
					double else_prob = 100 - probabilitySum;
					switch (successors.at(else_idx).which())
					{
					case 0:
						boost::get<Symbol>(successors.at(else_idx)).probability = else_prob;
						break;
					case 1:
						boost::get<Operator>(successors.at(else_idx)).probability = else_prob;
						break;
					}
				}

				return true;
			}

			void ProductionRule::convertToAbstraction(std::vector<PGA::Compiler::Terminal>& terminals, PGA::Compiler::Rule& rule) const
			{
				rule.symbol = trim(symbol);

				double probabilitySum = 0;
				switch (successors.at(0).which())
				{
				case 0:
					probabilitySum = boost::get<Symbol>(successors.at(0)).probability;
					break;
				case 1:
					probabilitySum = boost::get<Operator>(successors.at(0)).probability;
					break;
				}

				// TODO: Maybe support more rule successors (e.g. A --> B C D;)
				// For this, changes to the abstraction in ProductionRule.h need to be made
				// but this is not supported anyway (see next comment)
				if ((probabilitySum == 100) && (successors.size() == 1))
				{
					for (auto it = successors.begin(); it != successors.end(); ++it)
					{
						switch (it->which())
						{
						case 0: // shouldn't happen: RuleA --> RuleB produces cpp code with only CallRule<RuleB>, which is not supported afaik (but this works in the editor)
							rule.successor = std::shared_ptr<PGA::Compiler::Successor>(new PGA::Compiler::Symbol(boost::get<Symbol>(*it).symbol));
							rule.successor->id = nextSuccessorIdx();
							rule.successor->myrule = symbol;
							break;
						case 1:
							Operator op = boost::get<Operator>(*it);
							PGA::Compiler::Operator* newOperator = new PGA::Compiler::Operator();
							newOperator->myrule = symbol;
							newOperator->id = nextSuccessorIdx();
							if (op.operator_ == OperatorType::GENERATE)
							{
								for (auto it = terminals.begin(); it != terminals.end(); ++it)
								{
									// The 'if' below is true when there's a RuleA --> Generate()
									if (std::find(it->symbols.begin(), it->symbols.end(), symbol) != it->symbols.end())
									{
										newOperator->type = OperatorType::GENERATE;
										for (const auto& termAttr : it->parameters)
											newOperator->terminalAttributes.push_back(termAttr);
										break;
									}
								}
							}

							rule.successor = std::shared_ptr<PGA::Compiler::Successor>(newOperator);
							op.convertToAbstraction(terminals, *newOperator, rule);
							break;
						}
					}
				}
				else
				{
					Operator stochasticOperator;
					stochasticOperator.operator_ = STOCHASTIC;
					for (auto it = successors.begin(); it != successors.end(); ++it)
					{
						ParameterizedSuccessor parameterizedSuccessor(*it);
						switch (it->which())
						{
						case 0:
							parameterizedSuccessor.parameters.push_back(boost::get<Symbol>(*it).probability);
							break;
						case 1:
							parameterizedSuccessor.parameters.push_back(boost::get<Operator>(*it).probability);
							break;
						}
						stochasticOperator.successors.push_back(parameterizedSuccessor);
					}
					PGA::Compiler::Operator* newOperator = new PGA::Compiler::Operator();
					newOperator->id = nextSuccessorIdx();
					newOperator->myrule = symbol;
					rule.successor = std::shared_ptr<PGA::Compiler::Successor>(newOperator);
					stochasticOperator.convertToAbstraction(terminals, *newOperator, rule);
				}

			}

		}

	}

}