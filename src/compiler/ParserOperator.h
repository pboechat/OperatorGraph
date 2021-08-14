#pragma once

#include "ParserElement.h"
#include "ParserExpression.h"
#include "ParserOperand.h"
#include "ParserParameterizedSuccessor.h"
#include "Terminal.h"

#include <pga/compiler/Logger.h>
#include <pga/compiler/Operator.h>
#include <pga/compiler/OperatorType.h>
#include <pga/compiler/Rule.h>

#include <list>
#include <memory>
#include <string>
#include <vector>

// NOTE: needed here!
namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			struct Operator : Element
			{
				Operator() : operator_(0), probability(0) { }
				unsigned int operator_;
				std::vector<Operand> parameters;
				std::list<ParameterizedSuccessor> successors;
				double probability;

				void convertToAbstraction(std::vector<PGA::Compiler::Terminal>& terminals, PGA::Compiler::Operator& operator_, PGA::Compiler::Rule& rule) const;
				//void print(int level) const;
				bool check(Logger& logger);

			};

			std::shared_ptr<PGA::Compiler::Parameter> createOperatorParameter(Operand operand, OperatorType operatorType, size_t idx);

		}

	}

}