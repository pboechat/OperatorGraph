#pragma once

#include <pga/compiler/Vertex.h>
#include <pga/compiler/Edge.h>

namespace PGA
{
	namespace Compiler
	{
		struct GraphVisitor
		{
			virtual bool visit(size_t i, const PGA::Compiler::Vertex_LW& vertex) = 0;
			virtual bool visit(size_t i, const PGA::Compiler::Edge_LW& edge, const PGA::Compiler::Vertex_LW& src, const PGA::Compiler::Vertex_LW& dst) = 0;
			virtual void visitCycle(size_t i, const PGA::Compiler::Edge_LW& edge) {}

		};

	}

}
