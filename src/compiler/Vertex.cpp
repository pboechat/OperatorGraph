#include <sstream>
#include <iomanip>
#include <algorithm>
#include <exception>

#include <math/math.h>

#include <pga/compiler/Vertex.h>
#include <pga/compiler/Graph.h>
#include <pga/compiler/OperatorType.h>
#include <pga/compiler/EnumUtils.h>
#include <pga/core/GlobalConstants.h>
#include <pga/core/DispatchTableEntry.h>

namespace PGA
{
	namespace Compiler
	{
		std::string VARIABLES[] =
		{
			"X",
			"Y",
			"Z",
			"U",
			"V",
			"W",
		};

		Vertex::Vertex(size_t index, std::shared_ptr<Operator> op, ShapeType shapeType) : index(index), op(op), shapeType(shapeType), dist_from_root(0)
		{
		}

		void Vertex::print(std::ostream& out, bool plain, bool useVariables /*= false*/, const std::string& startParams /*= "("*/, const std::string& endParams /*= ")"*/) const
		{
			out << std::setprecision(ParameterPrecision);
			out << EnumUtils::toString(op->type);
			if (!op->operatorParams.empty())
			{
				out << startParams;
				for (size_t i = 0; i < op->operatorParams.size() - 1; i++)
				{
					if (useVariables)
						out << VARIABLES[i];
					else
						op->operatorParams[i]->print(out, plain);
					out << ", ";
				}
				auto j = op->operatorParams.size() - 1;
				op->operatorParams[j]->print(out, plain);
				out << endParams;
			}
		}

		void Vertex::getParams(std::vector<std::weak_ptr<Parameter>>& params) const
		{
			for (auto& operatorParam : op->operatorParams)
				params.emplace_back(operatorParam);
		}

		size_t Vertex::getCommonParams(const Vertex& other, std::set<size_t>& commonParams) const
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (op->operatorParams.size() != other.op->operatorParams.size())
				throw std::runtime_error("PGA::Compiler::Vertex::getCommonParams(..): trying to get common parameters between vertices with different number of parameters");
#endif
			for (size_t i = 0; i < op->operatorParams.size(); i++)
				if (op->operatorParams[i]->isEqual(other.op->operatorParams[i].get()))
					commonParams.insert(i);
			return commonParams.size();
		}

		std::string Vertex::uniqueName() const
		{
			return ((char)((index % 25) + 65)) + std::to_string(index / 25);
		}

		bool Vertex::isDiff(const Vertex& other) const
		{
			if (op->type != other.op->type)
				return true;
			if (op->operatorParams.size() != other.op->operatorParams.size())
				return true;
			if (shapeType != other.shapeType)
				return true;
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (op->type != GENERATE && op->genFuncIdx != -1)
				throw std::runtime_error("op->type != GENERATE && op->genFuncIdx != -1");
			if (op->type == GENERATE && op->genFuncIdx == -1)
				throw std::runtime_error("op->type == GENERATE && op->genFuncIdx == -1");
			if (other.op->type != GENERATE && other.op->genFuncIdx != -1)
				throw std::runtime_error("other.op->type != GENERATE && other.op->genFuncIdx != -1");
			if (other.op->type == GENERATE && other.op->genFuncIdx == -1)
				throw std::runtime_error("other.op->type == GENERATE && other.op->genFuncIdx == -1");
#endif
			if (op->genFuncIdx != other.op->genFuncIdx)
				return true;
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (op->type != GENERATE && !op->termAttrs.empty())
				throw std::runtime_error("op->type != GENERATE && !op->termAttrs.empty()");
			if (op->type == GENERATE && op->termAttrs.empty())
				throw std::runtime_error("op->type == GENERATE && op->termAttrs.empty()");
			if (other.op->type != GENERATE && !other.op->termAttrs.empty())
				throw std::runtime_error("other.op->type != GENERATE && !other.op->termAttrs.empty()");
			if (other.op->type == GENERATE && other.op->termAttrs.empty())
				throw std::runtime_error("other.op->type == GENERATE && other.op->termAttrs.empty()");
#endif
			return false;
		}

		size_t Vertex::getCommonTermParams(const Vertex& other, std::set<size_t>& commonTermParams) const
		{
			size_t size = math::min(op->termAttrs.size(), other.op->termAttrs.size());
			for (size_t i = 0; i < size; i++)
				if (op->termAttrs[i] == other.op->termAttrs[i])
					commonTermParams.insert(i);
			return commonTermParams.size();
		}

		void Vertex::getTermAttrs(std::vector<double>& termAttrs) const
		{
			termAttrs.insert(termAttrs.end(), op->termAttrs.begin(), op->termAttrs.end());
		}

		bool Vertex::hasOtherParent(const Vertex* parent) const
		{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 1)
			if (parent == nullptr)
				throw std::runtime_error("parent == nullptr");
#endif
			for (auto incomingEdge : incomingEdges)
			{
				// NOTE: bad design: comparing pointers!!!
				if (incomingEdge.lock().get()->srcVertex.get() != parent)
					return true;
			}

			return false;
		}

		void Vertex::getParents(std::set<size_t>& parents) const
		{
			for (auto incomingEdge : incomingEdges)
				parents.insert(incomingEdge.lock().get()->srcVertex->index);
		}

		size_t Vertex::getDistanceFromRoot() const
		{
			return dist_from_root;
		}

		void Vertex::getParentVertices(std::vector<Vertex*>& parents) const
		{
			for (auto incomingEdge : incomingEdges)
				parents.push_back(incomingEdge.lock().get()->srcVertex.get());

		}

		void Vertex::getChildVertices(std::vector<Vertex*>& children) const
		{
			for (auto outgoingEdge : outgoingEdges)
				children.push_back(outgoingEdge.lock().get()->dstVertex.get());

		}

	}

}

