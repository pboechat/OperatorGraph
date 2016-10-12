#pragma once

#include <pga/compiler/Successor.h>
#include <pga/compiler/Parameters.h>
#include <pga/compiler/ShapeType.h>
#include <pga/compiler/OperatorType.h>

#include <vector>
#include <memory>
#include <initializer_list>

namespace PGA
{
	namespace Compiler
	{
		struct Operator : public Successor
		{
			OperatorType type;
			std::vector<std::shared_ptr<Parameter>> operatorParams;
			std::vector<std::shared_ptr<Parameter>> successorParams;
			std::vector<std::shared_ptr<Successor>> successors;

			// NOTE: used only by Context.cpp (Analysis)
			long genFuncIdx;
			std::vector<double> termAttrs;

			Operator() : type(OperatorType(0)), genFuncIdx(-1) {}
			Operator(OperatorType type) : type(type) {}
			Operator(OperatorType type, const std::initializer_list<std::shared_ptr<Parameter>>& operatorParams, const std::initializer_list<std::shared_ptr<Parameter>>& successorParams, const std::initializer_list<std::shared_ptr<Successor>>& successors) :
				type(type), operatorParams(operatorParams.begin(), operatorParams.end()), successorParams(successorParams.begin(), successorParams.end()), successors(successors.begin(), successors.end()) {}
			virtual ~Operator() {}
			virtual SuccessorType getType() const {	return SuccessorType::OPERATOR;	}
			static ShapeType nextShapeType(OperatorType opType, ShapeType shapeType, size_t succIdx);
			static bool requiresEdgeParameter(OperatorType opType);

		};

	}

}
