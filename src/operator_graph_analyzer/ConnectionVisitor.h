#pragma once

#include <pga/compiler/Edge.h>
#include <pga/compiler/PartitionVisitor.h>
#include <pga/compiler/Vertex.h>

#include <map>
#include <ostream>
#include <string>
#include <vector>

struct ConnectionVisitor : PGA::Compiler::PartitionVisitor
{
	ConnectionVisitor();
	virtual void enterSubgraph(size_t i);
	virtual void leaveSubgraph(size_t i);
	virtual void visitEnter(size_t i, size_t j, const PGA::Compiler::Vertex_LW& vertex);
	virtual void visitLeave(size_t i, size_t j, const PGA::Compiler::Vertex_LW& vertex);
	virtual void visit(size_t i, const PGA::Compiler::Edge_LW& edge, size_t j, const PGA::Compiler::Vertex_LW& src, size_t k, const PGA::Compiler::Vertex_LW& dst, size_t srcSgIndex, size_t dstSgIndex, bool cutEdge);

	friend std::ostream& operator<<(std::ostream& out, const ConnectionVisitor& obj);

private:
	std::map<size_t, std::vector<size_t>> outConnections;

};

std::ostream& operator<<(std::ostream& out, const ConnectionVisitor& obj);