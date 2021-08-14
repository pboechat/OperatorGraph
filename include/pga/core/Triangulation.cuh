#pragma once

#include "GeometryUtils.h"
#include "LinkedList.cuh"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace PGA
{
	namespace Triangulation
	{
		__host__ __device__ __inline__ float side(const math::float2& a, const math::float2& b)
		{
			return (a.x * b.y - a.y * b.x);
		}

		__host__ __device__ __inline__ float angle(const math::float2& a, const math::float2& b)
		{
			// NOTE: due to imprecision, dot(a,b)/(length(a)*length(b)) can lead to values
			// slightly off the interval [-1,1]
			float c = acos(math::clamp(dot(a, b) / (length(a) * length(b)), -1.0f, 1.0f));
			if (side(a, b) < EPSILON)
				c = 2 * math::constants<float>::pi() - c;
			return c;
		}

		template <unsigned int N>
		__host__ __device__ __inline__ bool isEar(
			math::float2 vertices[N],
			unsigned int numVertices,
			DoublyLinkedList<unsigned int, N>& reflexes,
			int i0,
			int i1,
			int i2)
		{
			const auto& v0 = vertices[i0];
			const auto& v1 = vertices[i1];
			const auto& v2 = vertices[i2];
			auto a = v1 - v0, b = v2 - v1, c = v0 - v2;
			// forward search on reflex vertices list
			auto r = reflexes.front();
			while (r != -1)
			{
				auto i = reflexes[r];
				if (i != i0 && i != i1 && i != i2)
				{
					auto v3 = vertices[i];
					if (side(a, v3 - v0) > -EPSILON && side(b, v3 - v1) > -EPSILON && side(c, v3 - v2) > -EPSILON)
					// NOTE: right-handed
					//if (side(a, v3 - v0) < EPSILON && side(b, v3 - v1) < EPSILON && side(c, v3 - v2) < EPSILON)
						return false;
				}
				r = reflexes.next(r);
			}
			return true;
		}

		template <unsigned int N>
		__host__ __device__ __inline__ bool isReflex(math::float2 vertices[N], unsigned int numVertices, int i0, int i1, int i2)
		{
			const auto& v0 = vertices[i0];
			const auto& v1 = vertices[i1];
			const auto& v2 = vertices[i2];
			auto a = v2 - v1, b = v0 - v1;
			// NOTE: right-handed
			//auto a = v0 - v1, b = v2 - v1;
			auto d = angle(a, b);
			return d >= (math::constants<float>::pi() - EPSILON);
		}

		template <unsigned int N>
		__host__ __device__ __inline__ void update(
			math::float2 vertices[N],
			unsigned int numVertices,
			CircularDoublyLinkedList<unsigned int, N>& adjacencies,
			DoublyLinkedList<unsigned int, N>& reflexes,
			DoublyLinkedList<unsigned int, N>& ears,
			unsigned int i1)
		{
			auto a = adjacencies.backwardSearch(i1);
			auto i0 = adjacencies.previous(a);
			auto i2 = adjacencies.next(a);
			if (isReflex<N>(vertices, numVertices, i0, i1, i2))
			{
				if (reflexes.backwardSearch(i1) != -1)
					return;
				reflexes.addFront(i1);
				auto e = ears.backwardSearch(i1);
				if (e != -1)
					ears.remove(e);
			}
			else if (isEar<N>(vertices, numVertices, reflexes, i0, i1, i2))
			{
				// add i1 to the ears list
				if (ears.backwardSearch(i1) != -1)
					return;
				ears.addFront(i1);
				// if i1 was a reflex vertex, remove it from the list
				auto r = reflexes.backwardSearch(i1);
				if (r != -1)
					reflexes.remove(r);
			}
			else
			{
				auto r = reflexes.backwardSearch(i1);
				if (r != -1)
					reflexes.remove(r);
				else
				{
					auto e = ears.backwardSearch(i1);
					if (e != -1)
						ears.remove(e);
				}
			}
		}

		template <unsigned int N>
		__host__ __device__ __inline__ bool earClipping(math::float2 vertices[N], unsigned int numVertices, unsigned int indices[(N - 2) * 3])
		{
			static_assert(N >= 3, "N < 3");
			CircularDoublyLinkedList<unsigned int, N> adjacencies;
			DoublyLinkedList<unsigned int, N> reflexes;
			DoublyLinkedList<unsigned int, N> ears;
			for (unsigned int i1 = 0, i0 = numVertices - 1; i1 < numVertices; i0 = i1, i1++)
			{
				auto i2 = (i1 + 1) % numVertices;
				if (isReflex<N>(vertices, numVertices, i0, i1, i2))
					reflexes.addBack(i1);
				adjacencies.addBack(i1);
			}
			for (unsigned int i1 = 0, i0 = numVertices - 1; i1 < numVertices; i0 = i1, i1++)
			{
				if (reflexes.forwardSearch(i1) != -1)
					continue;
				auto i2 = (i1 + 1) % numVertices;
				if (isEar<N>(vertices, numVertices, reflexes, i0, i1, i2))
					ears.addBack(i1);
			}
			unsigned int t = 0;
			do
			{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
				if (ears.empty())
				{
/*#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("PGA::GeometryUtils::earClipping(): ears.empty() [numVertices=%d, vertices={");
					for (auto i = 0; i < numVertices; i++)
					{
						printf("(%f, %f)", vertices[i].x, vertices[i].y);
						if (i < numVertices - 1)
							printf(", ");
					}
					printf("}] (CUDA thread %d %d)\n", threadIdx.x, blockIdx.x);
					asm("trap;");
#else
					std::string verticesStr;
					for (unsigned int i = 0; i < numVertices; i++)
					{
						verticesStr += "(" + std::to_string(vertices[i].x) + ", " + std::to_string(vertices[i].y) + ")";
						if (i < numVertices - 1)
							verticesStr += ", ";
					}
					throw std::runtime_error("PGA::GeometryUtils::earClipping(): ears.empty() [numVertices=" + std::to_string(numVertices) + ", vertices={" + verticesStr + "}]");
#endif*/
					return false;
				}
#endif
				// assign last ear to i0 and remove it
				auto e = ears.back();
				auto i1 = ears[e];
				ears.remove(e);
				// assign the adjacencies of the last ear to i1 and it2 and remove it 
				// from the adjacency list
				auto a = adjacencies.backwardSearch(i1);
				auto i0 = adjacencies.previous(a);
				auto i2 = adjacencies.next(a);
				adjacencies.remove(a);
				// add the indices to the list
				indices[t++] = i0;
				indices[t++] = i1;
				indices[t++] = i2;
				if (adjacencies.size() > 3)
				{
					// update lists for adjacent vertices
					update<N>(vertices, numVertices, adjacencies, reflexes, ears, i0);
					update<N>(vertices, numVertices, adjacencies, reflexes, ears, i2);
				}
			} while (adjacencies.size() >= 3);

			return true;
		}

	}

}
