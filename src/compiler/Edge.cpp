#include <string>

#include <pga/compiler/Edge.h>

namespace PGA
{
	namespace Compiler
	{
		Edge::Edge(size_t index, std::shared_ptr<Vertex> srcVertex, std::shared_ptr<Vertex> dstVertex, size_t succIdx, std::shared_ptr<Parameter> param, bool loop)
			: index(index), srcVertex(srcVertex), dstVertex(dstVertex), succIdx(succIdx), param(param), loop(loop)
		{
		}

		std::string Edge::uniqueName() const
		{
			return "edge_" + std::to_string(index);
		}

		std::weak_ptr<Parameter> Edge::getParameter() const
		{
			return static_cast<std::weak_ptr<Parameter>>(param);
		}

		void Edge::print(std::ostream& out) const
		{
			if (param != nullptr)
			{
				out << "[" << succIdx << "]: ";
				param->print(out, true);
			}
			else
				out << " - ";
		}

		//////////////////////////////////////////////////////////////////////////
		size_t Edge_LW::counter = 0;

	}

}