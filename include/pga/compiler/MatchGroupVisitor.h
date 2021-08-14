#pragma once

#include <pga/compiler/OperatorType.h>
#include <pga/compiler/Parameters.h>
#include <pga/compiler/ShapeType.h>

#include <memory>
#include <vector>

namespace PGA
{
	namespace Compiler
	{
		struct MatchGroupVisitor
		{
			virtual void visitVertex(
				size_t sgIdx,
				size_t vertexIdx,
				PGA::Compiler::OperatorType opType,
				PGA::Compiler::ShapeType shapeType,
				int phase,
				long genFuncIdx,
				size_t numParams,
				const std::map<size_t, std::weak_ptr<PGA::Compiler::Parameter>>& eqParams,
				const std::map<size_t, std::weak_ptr<PGA::Compiler::Parameter>>& diffParams,
				size_t numTermAttrs,
				const std::map<size_t, double>& eqTermAttrs,
				const std::map<size_t, double>& diffTermAttrs,
				size_t numEdges) = 0;
			virtual void visitEdge(
				size_t sgIdx,
				size_t inIdx,
				long outIdx,
				long edgeIdx,
				bool cutEdge,
				bool commonEdge,
				bool requiresParam,
				bool dynParam,
				const std::weak_ptr<PGA::Compiler::Parameter>& param,
				size_t i) = 0;

		};

	}

}