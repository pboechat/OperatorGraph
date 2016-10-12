#include "ParserResult.h"

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			bool Result::check(Logger& logger)
			{
				if (axioms.size() < 1)
				{
					logger.addMessage(Logger::LL_ERROR, "No axioms to start with");
					return false;
				}
				else
				{
					for (auto axiom = axioms.begin(); axiom != axioms.end(); ++axiom)
					{
						if ((axiom->shapeType == ShapeType::DYNAMIC_CONVEX_POLYGON ||
							axiom->shapeType == ShapeType::DYNAMIC_POLYGON ||
							axiom->shapeType == ShapeType::DYNAMIC_CONVEX_RIGHT_PRISM ||
							axiom->shapeType == ShapeType::DYNAMIC_RIGHT_PRISM) &&
							axiom->vertices.size() < 3)
						{
							logger.addMessage(Logger::LL_ERROR, "Dynamic(ConvexPolygon/RightPrism) axiom declaration requires least 3 vertices");
							return false;
						}
					}
				}

				for (auto it = rules.begin(); it != rules.end(); ++it)
					if (!it->check(logger))
						return false;

				return true;
			}

			void Result::convertToAbstraction
			(
				std::vector<PGA::Compiler::Axiom>& axioms, 
				std::vector<PGA::Compiler::Rule>& rules, 
				std::vector<PGA::Compiler::Terminal>& terminals
			) const
			{
				axioms.reserve(axioms.size());
				for (auto it = this->axioms.begin(); it != this->axioms.end(); ++it)
				{
					PGA::Compiler::Axiom axiom;
					axiom.symbol = it->name;
					axiom.shapeType = static_cast<ShapeType>(it->shapeType);
					axiom.vertices.insert(axiom.vertices.begin(), it->vertices.begin(), it->vertices.end());
					axioms.push_back(axiom);
				}

				terminals.reserve(this->terminals.size());
				for (auto it1 = this->terminals.begin(); it1 != this->terminals.end(); ++it1)
				{
					PGA::Compiler::Terminal terminal;
					terminal.idx = static_cast<unsigned int>(std::distance(this->terminals.begin(), it1)) + 1;
					for (const auto& parameter : it1->parameters)
						terminal.parameters.push_back(parameter);
					for (const auto& parameter : it1->symbols)
						terminal.symbols.push_back(parameter);
					terminals.push_back(terminal);
				}

				rules.reserve(this->rules.size());
				for (auto it = this->rules.begin(); it != this->rules.end(); ++it)
				{
					rules.push_back(PGA::Compiler::Rule());
					it->convertToAbstraction(terminals, rules.back());
				}
			}

		}

	}

}