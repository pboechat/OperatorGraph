#pragma once

#include <map>
#include <set>
#include <memory>
#include <ostream>

#include <pga/compiler/Parameters.h>
#include <pga/compiler/Vertex.h>

namespace PGA
{
	namespace Compiler
	{
		//////////////////////////////////////////////////////////////////////////
		struct Edge
		{
			size_t index;
			std::shared_ptr<Vertex> srcVertex;
			std::shared_ptr<Vertex> dstVertex;
			size_t succIdx;
			std::shared_ptr<Parameter> param;
			bool loop;

			Edge(size_t index, std::shared_ptr<Vertex> srcVertex, std::shared_ptr<Vertex> dstVertex, size_t succIdx, std::shared_ptr<Parameter> succParam, bool loop);
			std::weak_ptr<Parameter> getParameter() const;
			void print(std::ostream& out) const;
			std::string uniqueName() const;

		};

		//////////////////////////////////////////////////////////////////////////
		struct Edge_LW
		{
			size_t in;
			size_t out;

			Edge_LW(std::weak_ptr<Edge> ptr, size_t in, size_t out) : uid(counter++), ptr(ptr), in(in), out(out)
			{
			}

			Edge_LW(size_t in, size_t out) : uid(counter++), in(in), out(out)
			{
			}

			inline std::string uniqueName() const
			{
				return ptr.lock()->uniqueName();
			}

			inline void print(std::ostream& out) const
			{
				ptr.lock()->print(out);
			}

			std::weak_ptr<Parameter> getParameter() const
			{
				return ptr.lock()->getParameter();
			}

			operator size_t() const
			{
				return uid;
			}

		private:
			static size_t counter;
			size_t uid;
			std::weak_ptr<Edge> ptr;

		};

	}

}