#pragma once

#include <map>
#include <vector>
#include <string>

#include <pga/compiler/GraphVisitor.h>
#include <pga/compiler/Edge.h>
#include <pga/compiler/Vertex.h>

namespace PGA
{
	namespace Compiler
	{
		struct PhaseVisitor : GraphVisitor
		{
			typedef std::map<size_t, std::vector<Vertex_LW>> VertexDependency;
			typedef std::map<unsigned int, VertexDependency> VertexDependencies;

			VertexDependencies visitedColliders;
			VertexDependencies visitedIfCollides;
			std::map<unsigned int, std::vector<Vertex_LW>> colliders;

			virtual bool visit(size_t i, const Vertex_LW& vertex);
			virtual bool visit(size_t i, const Edge_LW& edge, const Vertex_LW& src, const Vertex_LW& dst);

		};

	}

}