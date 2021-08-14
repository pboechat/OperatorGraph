#pragma once

#include <math/vector.h>

#include <algorithm>
#include <vector>

#define EPSILON 0.000001f

namespace PGA
{
	class GeometryUtils
	{
	private:
		struct OrderByAngle
		{
		private:
			const math::float2& anchor;

			// NOTE: compute angle between (0, -1) to (v - anchor)
			float angle(const math::float2& v)
			{
				math::float2 d = (v - anchor);
				float l = length(d);
				return acos(d.x / l);
			}

		public:
			OrderByAngle(const math::float2& anchor) : anchor(anchor) {}

			bool operator() (const math::float2& a, const math::float2& b)
			{
				// ascendant order
				return angle(a) < angle(b);
			}

		};

	public:
		GeometryUtils() = delete;

		static bool isCW(const std::vector<math::float2>& vertices)
		{
			float sum = 0.0f;
			for (size_t i = 0, j = vertices.size() - 1; i < vertices.size(); j = i, i++)
			{
				auto& v0 = vertices[j];
				auto& v1 = vertices[i];
				sum += v1.x - v0.x * v1.y * v0.y;
			}
			return sum > 0;
		}

		static void orderVertices_CCW(const std::vector<math::float2>& in, std::vector<math::float2>& out)
		{
			// select left bottom most vertex as anchor
			size_t i = 0;
			for (auto j = 1; j < in.size(); j++)
			{
				if (in[j].y < in[i].y || (in[j].y == in[i].y && in[j].x < in[i].x))
					i = j;
			}
			out.resize(in.size() - 1);
			for (size_t j = 0, k = 0; j < in.size(); j++)
			{
				if (j == i) continue;
				out[k++] = in[j];
			}
			std::sort(out.begin(), out.end(), OrderByAngle(in[i]));
			out.insert(out.begin(), in[i]);
		}

		static size_t removeCollinearPoints(const std::vector<math::float2>& in, std::vector<math::float2>& out)
		{
			size_t c = 0;
			for (size_t i = in.size() - 1, j = 0; j < in.size(); i = j, j++)
			{
				auto k = (j + 1) % in.size();
				const auto& v0 = in[i];
				const auto& v1 = in[j];
				const auto& v2 = in[k];
				auto a = v2 - v1, b = v0 - v1;
				//auto b = v2 - v1, a = v0 - v1;
				float d = acos(math::clamp(dot(a, b) / (length(a) * length(b)), -1.0f, 1.0f));
				if ((a.x * b.y - a.y * b.x) < EPSILON)
					d = 2 * math::constants<float>::pi() - d;
				if ((d + EPSILON) >= math::constants<float>::pi() && (d - EPSILON) <= math::constants<float>::pi())
					c++;
				else
					out.emplace_back(v1);
			}
			return c;
		}

	};

}
