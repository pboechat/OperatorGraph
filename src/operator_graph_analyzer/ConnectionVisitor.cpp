#include "ConnectionVisitor.h"

ConnectionVisitor::ConnectionVisitor()
{
}

void ConnectionVisitor::visitEnter(size_t i, size_t j, const PGA::Compiler::Vertex_LW& vertex)
{
}

void ConnectionVisitor::visitLeave(size_t i, size_t j, const PGA::Compiler::Vertex_LW& vertex)
{
}

void ConnectionVisitor::visit(size_t i, const PGA::Compiler::Edge_LW& edge, size_t j, const PGA::Compiler::Vertex_LW& src, size_t k, const PGA::Compiler::Vertex_LW& dst, size_t srcSgIndex, size_t dstSgIndex, bool cutEdge)
{
	if (!cutEdge)
		return;
	outConnections[srcSgIndex].push_back(i);
}

void ConnectionVisitor::enterSubgraph(size_t i)
{
}

void ConnectionVisitor::leaveSubgraph(size_t i)
{
}

std::ostream& operator<<(std::ostream& out, const ConnectionVisitor& obj)
{
	for (auto& entry : obj.outConnections)
	{
		out << entry.first << "=";
		auto it = entry.second.begin();
		for (auto i = 0; i < entry.second.size() - 1; i++, it++)
		{
			out << (*it) << ", ";
		}
		out << (*it) << std::endl;
	}
	return out;
}

