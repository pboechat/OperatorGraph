#pragma once

#include <pga/compiler/Operator.h>
#include <pga/compiler/ShapeType.h>

#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <vector>

namespace PGA
{
	namespace Compiler
	{
		struct Edge;

		struct Vertex
		{
			static const unsigned int ParameterPrecision = 3;

			size_t index;
			size_t dist_from_root;
			std::vector<std::weak_ptr<Edge>> incomingEdges;
			std::vector<std::weak_ptr<Edge>> outgoingEdges;
			std::shared_ptr<Operator> op;
			ShapeType shapeType;

			Vertex(size_t id, std::shared_ptr<Operator> op, ShapeType shapeType);

			void print(std::ostream& out, bool plain, bool dynamic = false, const std::string& startParams = "(", const std::string& endParams = ")") const;
			void getParams(std::vector<std::weak_ptr<Parameter>>& params) const;
			void getTermAttrs(std::vector<double>& termParams) const;
			size_t getCommonParams(const Vertex& other, std::set<size_t>& commonParams) const;
			size_t getCommonTermParams(const Vertex& other, std::set<size_t>& commonTermValues) const;
			std::string uniqueName() const;
			bool isDiff(const Vertex& other) const;
			bool hasOtherParent(const Vertex* parent) const;
			void getParents(std::set<size_t>& parents) const;
			size_t getDistanceFromRoot() const;
			void getParentVertices(std::vector<Vertex*>& parents) const;
			void getChildVertices(std::vector<Vertex*>& children) const;

		};

		struct Vertex_LW
		{
			Vertex_LW(std::weak_ptr<Vertex> ptr) : /*uid(counter++), */ptr(ptr)
			{
			}

			inline bool operator == (const Vertex_LW& other) const
			{
				//return uid == other.uid;
				return ptr.lock()->index == other.ptr.lock()->index;
			}

			inline bool operator != (const Vertex_LW& other) const
			{
				return !operator==(other);
			}

			inline size_t numIncomingEdges() const
			{
				return ptr.lock()->incomingEdges.size();
			}

			inline OperatorType operatorType() const
			{
				return ptr.lock()->op->type;
			}

			inline ShapeType shapeType() const
			{
				return ptr.lock()->shapeType;
			}

			inline int phase() const
			{
				return ptr.lock()->op->phase;
			}

			inline long getGenFuncIdx() const
			{
				return ptr.lock()->op->genFuncIdx;
			}

			inline size_t numParams() const
			{
				return ptr.lock()->op->operatorParams.size();
			}

			inline void getParams(std::vector<std::weak_ptr<Parameter>>& params) const
			{
				ptr.lock()->getParams(params);
			}

			inline void getTermAttrs(std::vector<double>& termParams) const
			{
				return ptr.lock()->getTermAttrs(termParams);
			}

			inline size_t numTermAttrs() const
			{
				return ptr.lock()->op->termAttrs.size();
			}

			inline void print(std::ostream& out, bool plain, bool useVariables = false, const std::string& startParams = "(", const std::string& endParams = ")") const
			{
				ptr.lock()->print(out, plain, useVariables, startParams, endParams);
			}

			inline std::string uniqueName() const
			{
				return ptr.lock()->uniqueName();
			}

			inline size_t getCommonParams(const Vertex_LW& other, std::set<size_t>& commonParams) const
			{
				return ptr.lock()->getCommonParams(*other.ptr.lock().get(), commonParams);
			}

			inline size_t getCommonTermParams(const Vertex_LW& other, std::set<size_t>& commonTermParams) const
			{
				return ptr.lock()->getCommonTermParams(*other.ptr.lock().get(), commonTermParams);
			}

			operator size_t() const
			{
				return ptr.lock()->index;
			}

			operator Vertex*() const
			{
				return ptr.lock().get();
			}

			bool isDiff(const Vertex_LW& other) const
			{
				return ptr.lock()->isDiff(*other.ptr.lock().get());
			}

			inline bool hasOtherParent(const Vertex_LW& parent) const
			{
				return ptr.lock()->hasOtherParent(parent.ptr.lock().get());
			}

			inline void getParents(std::set<size_t>& parents) const
			{
				ptr.lock()->getParents(parents);
			}

			inline size_t	getDistanceFromRoot() const
			{
				return ptr.lock()->getDistanceFromRoot();
			}

			inline Operator* getOperator() const
			{
				return ptr.lock()->op.get();
			}

			inline void getParentVertices(std::vector<Vertex*>& parents) const
			{
				ptr.lock()->getParentVertices(parents);
			}

			inline void getChildVertices(std::vector<Vertex*>& children) const
			{
				ptr.lock()->getChildVertices(children);
			}
		private:
			std::weak_ptr<Vertex> ptr;

		};

	}

}