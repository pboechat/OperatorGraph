#include "CommonGraphOperations.h"

//////////////////////////////////////////////////////////////////////////
struct VerticesSharedByCutEdges : PGA::Compiler::GraphVisitor
{
	const std::set<size_t>& alwaysCutEdges;
	std::set<size_t>& verticesSharedByAlwaysCutEdges;

	VerticesSharedByCutEdges(std::set<size_t>& alwaysCutEdges, std::set<size_t>& verticesSharedByAlwaysCutEdges) : alwaysCutEdges(alwaysCutEdges), verticesSharedByAlwaysCutEdges(verticesSharedByAlwaysCutEdges) {}

	virtual bool visit(size_t i, const PGA::Compiler::Vertex_LW& vertex)
	{
		return true;
	}

	virtual bool visit(size_t i, const PGA::Compiler::Edge_LW& edge, const PGA::Compiler::Vertex_LW& src, const PGA::Compiler::Vertex_LW& dst)
	{
		if (alwaysCutEdges.find(edge) != alwaysCutEdges.end())
			verticesSharedByAlwaysCutEdges.insert(dst);
		return true;
	}

};

//////////////////////////////////////////////////////////////////////////
void enforceR3(PGA::Compiler::Graph& graph, std::set<size_t>& cutEdges)
{
	std::set<size_t> vertices;
	VerticesSharedByCutEdges visitor(cutEdges, vertices);
	graph.depthFirst(visitor);
	std::set<size_t> inEdges;
	for (auto vertexIdx : vertices)
		graph.findInEdges(vertexIdx, inEdges);
	cutEdges.insert(inEdges.begin(), inEdges.end());
}