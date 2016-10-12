#pragma once

#include <pga/compiler/Vertex.h>

namespace PGA
{
	namespace Compiler
	{
		struct PartitionVisitor
		{
			virtual void enterSubgraph(size_t i) = 0;
			virtual void leaveSubgraph(size_t i) = 0;
			virtual void visitEnter(size_t i, size_t j, const Vertex_LW& vertex) = 0;
			virtual void visit(size_t edgeIdx, 
				const Edge_LW& edge, 
				size_t inIdx, 
				const Vertex_LW& src, 
				size_t outIdx, 
				const Vertex_LW& dst, 
				size_t srcSgIndex, 
				size_t dstSgIndex,
				bool cutEdge) = 0;
			virtual void visitLeave(size_t i, size_t j, const Vertex_LW& vertex) = 0;

		};

	}

}