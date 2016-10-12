#pragma once

#include <pga/compiler/Graph.h>

#include <sstream>
#include <stack>
#include <vector>
#include <string>

struct DotGraphVisitor : PGA::Compiler::GraphVisitor, PGA::Compiler::PartitionVisitor
{
	DotGraphVisitor();
	void clear();
	virtual void enterSubgraph(size_t i);
	virtual void leaveSubgraph(size_t i);
	virtual bool visit(size_t i, const PGA::Compiler::Vertex_LW& vertex);
	virtual void visitEnter(size_t i, size_t j, const PGA::Compiler::Vertex_LW& vertex);
	virtual void visitLeave(size_t i, size_t j, const PGA::Compiler::Vertex_LW& vertex);
	virtual bool visit(size_t i, const PGA::Compiler::Edge_LW& edge, const PGA::Compiler::Vertex_LW& src, const PGA::Compiler::Vertex_LW& dst);
	virtual void visit(size_t i, const PGA::Compiler::Edge_LW& edge, size_t j, const PGA::Compiler::Vertex_LW& src, size_t k, const PGA::Compiler::Vertex_LW& dst, size_t srcSgIndex = 0, size_t dstSgIndex = 0, bool cutEdge = false);

	friend std::ostream& operator<<(std::ostream& out, const DotGraphVisitor& obj);

private:
	std::vector<size_t> visitedNodes;
	std::stack<size_t> s;
	std::vector<std::string> streams;

	void visitEdge(size_t i, const PGA::Compiler::Vertex_LW &src, const PGA::Compiler::Vertex_LW &dst, const PGA::Compiler::Edge_LW &edge, bool cutEdge);

};

std::ostream& operator<<(std::ostream& out, const DotGraphVisitor& obj);
