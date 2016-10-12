#include <pga/compiler/Operator.h>
#include <pga/compiler/PhaseVisitor.h>

namespace PGA
{
	namespace Compiler
	{
		bool PhaseVisitor::visit(size_t i, const Vertex_LW& vertex)
		{
			Operator* op = vertex.getOperator();
			if (op->type == OperatorType::COLLIDER)
			{
				unsigned int id = static_cast<unsigned int>(op->operatorParams.at(0)->at(0));
				std::pair<int, size_t> colliderVertexID(id, vertex);
				if (colliders.count(id) != 0)
					colliders.at(id).push_back(vertex);
				else
					colliders.insert(std::make_pair(id, std::vector<Vertex_LW>({ vertex })));
			}
			return true;
		}

		bool PhaseVisitor::visit(size_t i, const Edge_LW& edge, const Vertex_LW& src, const Vertex_LW& dst)
		{
			return true;
		}

	}

}