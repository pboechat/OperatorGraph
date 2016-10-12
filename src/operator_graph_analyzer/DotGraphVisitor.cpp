#include "DotGraphVisitor.h"
//#define SIMPLIFIED_REP

DotGraphVisitor::DotGraphVisitor()
{
	s.push(0);
	streams.resize(1);
}

bool DotGraphVisitor::visit(size_t i, const PGA::Compiler::Vertex_LW& vertex)
{
	auto& source = streams[s.top()];
	std::stringstream stream;
	stream << source;
#ifdef SIMPLIFIED_REP
	stream << vertex.uniqueName() << "[label=\"";
#else
	stream << vertex.uniqueName() << "[label=\"" << i << ": ";
#endif
	vertex.print(stream, true, false);
	stream << "\"]" << std::endl;
	source = stream.str();
	return true;
}

void DotGraphVisitor::visitEnter(size_t i, size_t j, const PGA::Compiler::Vertex_LW& vertex)
{
	auto& source = streams[s.top()];
	std::stringstream stream;
	stream << source;
#ifdef SIMPLIFIED_REP
	stream << vertex.uniqueName() << "[label=\"";
#else
	stream << vertex.uniqueName() << "[label=\"" << i << "[" << j << "]" << ": ";
#endif
	vertex.print(stream, true, false);
	stream << "\"]" << std::endl;
	source = stream.str();
}

void DotGraphVisitor::visitLeave(size_t i, size_t j, const PGA::Compiler::Vertex_LW& vertex)
{
}

bool DotGraphVisitor::visit(size_t i, const PGA::Compiler::Edge_LW& edge, const PGA::Compiler::Vertex_LW& src, const PGA::Compiler::Vertex_LW& dst)
{
	visitEdge(i, src, dst, edge, false);
	return true;
}

void DotGraphVisitor::visit(size_t i, const PGA::Compiler::Edge_LW& edge, size_t j, const PGA::Compiler::Vertex_LW& src, size_t k, const PGA::Compiler::Vertex_LW& dst, size_t srcSgIndex, size_t dstSgIndex, bool cutEdge)
{
	visitEdge(i, src, dst, edge, cutEdge);
}

void DotGraphVisitor::enterSubgraph(size_t i)
{
	if ((i + 1) >= streams.size())
		streams.resize(i + 2);
	s.push(i + 1);
	auto& source = streams[s.top()];
	std::stringstream stream;
	stream << source;
	stream << "subgraph cluster_" << i << " {" << std::endl << "label=\"subgraph_" << i << "\"" << std::endl;
	source = stream.str();
}

void DotGraphVisitor::leaveSubgraph(size_t i)
{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
	if (i >= streams.size())
		throw std::runtime_error("i >= dot.size()");
#endif
	auto& source = streams[s.top()];
	std::stringstream stream;
	stream << source;
	stream << "}" << std::endl;
	source = stream.str();
	s.pop();
}

void DotGraphVisitor::visitEdge(size_t i, const PGA::Compiler::Vertex_LW &src, const PGA::Compiler::Vertex_LW &dst, const PGA::Compiler::Edge_LW &edge, bool cutEdge)
{
	auto& source = streams[0];
	std::stringstream stream;
	stream << source;
	stream << src.uniqueName() << " -> " << dst.uniqueName();
#ifdef SIMPLIFIED_REP
	if (cutEdge)
		stream << "[color=red];" << std::endl;
	else
		stream << ";" << std::endl;
#else
	stream << "[label=\"" << i << ": ";
	edge.print(stream);
	stream << "\"";
	if (cutEdge)
		stream << ", color=red";
	stream << "];" << std::endl;
#endif
	source = stream.str();
}

void DotGraphVisitor::clear()
{
	streams.clear();
	while (!s.empty())
		s.pop();
	s.push(0);
	streams.resize(1);
}

std::ostream& operator<<(std::ostream& out, const DotGraphVisitor& obj)
{
	out << "digraph operator_graph {" << std::endl;
	for (size_t i = 1; i < obj.streams.size(); i++)
		out << obj.streams[i] << std::endl;
	out << obj.streams[0] << std::endl;
	out << std::endl << "}";
	return out;
}

