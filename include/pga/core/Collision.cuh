#pragma once

#include <cuda_runtime_api.h>

#include <math/vector.h>
#include <math/matrix.h>

#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "ContextSensitivityDeviceVariables.cuh"
#include "Shapes.cuh"
#include "AABB.cuh"
#include "TStdLib.h"

namespace PGA
{
	namespace ContextSensitivity
	{
		struct Collision
		{
			__host__ __device__ __inline__ static bool check(const AABB& aabb, const Shapes::Box& box)
			{
				return check((Shapes::Box)aabb, box);
			}

			__host__ __device__ __inline__ static bool check(const AABB& aabb, const Shapes::Quad& quad)
			{
				return check((Shapes::Box)aabb, quad);
			}

			__host__ __device__ __inline__ static bool check(const AABB& aabb, const Shapes::Sphere& sphere)
			{
				return check((Shapes::Box)aabb, sphere);
			}

			__host__ __device__ __inline__ static bool check(const Shapes::Box& aBox, const Shapes::Box& bBox)
			{
				if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					atomicInc(&Device::NumCollisionChecks, INT_MAX);
#else
					Host::NumCollisionChecks = (Host::NumCollisionChecks + 1) % INT_MAX;
#endif
				}

				const math::float3x4& A = aBox.getModel();
				const math::float3x4& B = bBox.getModel();

				math::float3 a = aBox.getHalfExtents();
				math::float3 b = bBox.getHalfExtents();
				math::float3 v = bBox.getPosition() - aBox.getPosition();
				math::float3 T(dot(v, A.column1()), dot(v, A.column2()), dot(v, A.column3()));

				float R[3][3];
				float ra, rb, t;
				unsigned int i, k;

				R[0][0] = dot(A.column1(), B.column1());
				R[0][1] = dot(A.column1(), B.column2());
				R[0][2] = dot(A.column1(), B.column3());
				R[1][0] = dot(A.column2(), B.column1());
				R[1][1] = dot(A.column2(), B.column2());
				R[1][2] = dot(A.column2(), B.column3());
				R[2][0] = dot(A.column3(), B.column1());
				R[2][1] = dot(A.column3(), B.column2());
				R[2][2] = dot(A.column3(), B.column3());

				for (i = 0; i < 3; i++)
				{
					ra = a[i];
					rb = b[0] * fabs(R[i][0]) + b[1] * fabs(R[i][1]) + b[2] * fabs(R[i][2]);
					t = fabs(T[i]);
					if (t > ra + rb)
					{
						if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
						{
							printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
								aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
								bBox.getPosition().x, bBox.getPosition().y, bBox.getPosition().z, bBox.getSize().x, bBox.getSize().y, bBox.getSize().z);
						}

						return false;
					}
				}

				for (k = 0; k < 3; k++)
				{
					ra = a[0] * fabs(R[0][k]) + a[1] * fabs(R[1][k]) + a[2] * fabs(R[2][k]);
					rb = b[k];
					t = fabs(T[0] * R[0][k] + T[1] * R[1][k] + T[2] * R[2][k]);
					if (t > ra + rb)
					{
						if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
						{
							printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
								aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
								bBox.getPosition().x, bBox.getPosition().y, bBox.getPosition().z, bBox.getSize().x, bBox.getSize().y, bBox.getSize().z);
						}

						return false;
					}
				}

				// 9 cross products

				// L = A0 x B0
				ra = a[1] * fabs(R[2][0]) + a[2] * fabs(R[1][0]);
				rb = b[1] * fabs(R[0][2]) + b[2] * fabs(R[0][1]);
				t = fabs(T[2] * R[1][0] - T[1] * R[2][0]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bBox.getPosition().x, bBox.getPosition().y, bBox.getPosition().z, bBox.getSize().x, bBox.getSize().y, bBox.getSize().z);
					}

					return false;
				}

				// L = A0 x B1
				ra = a[1] * fabs(R[2][1]) + a[2] * fabs(R[1][1]);
				rb = b[0] * fabs(R[0][2]) + b[2] * fabs(R[0][0]);
				t = fabs(T[2] * R[1][1] - T[1] * R[2][1]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bBox.getPosition().x, bBox.getPosition().y, bBox.getPosition().z, bBox.getSize().x, bBox.getSize().y, bBox.getSize().z);
					}

					return false;
				}

				// L = A0 x B2
				ra = a[1] * fabs(R[2][2]) + a[2] * fabs(R[1][2]);
				rb = b[0] * fabs(R[0][1]) + b[1] * fabs(R[0][0]);
				t = fabs(T[2] * R[1][2] - T[1] * R[2][2]);

				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bBox.getPosition().x, bBox.getPosition().y, bBox.getPosition().z, bBox.getSize().x, bBox.getSize().y, bBox.getSize().z);
					}

					return false;
				}

				// L = A1 x B0
				ra = a[0] * fabs(R[2][0]) + a[2] * fabs(R[0][0]);
				rb = b[1] * fabs(R[1][2]) + b[2] * fabs(R[1][1]);
				t = fabs(T[0] * R[2][0] - T[2] * R[0][0]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bBox.getPosition().x, bBox.getPosition().y, bBox.getPosition().z, bBox.getSize().x, bBox.getSize().y, bBox.getSize().z);
					}

					return false;
				}

				// L = A1 x B1
				ra = a[0] * fabs(R[2][1]) + a[2] * fabs(R[0][1]);
				rb = b[0] * fabs(R[1][2]) + b[2] * fabs(R[1][0]);
				t = fabs(T[0] * R[2][1] - T[2] * R[0][1]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bBox.getPosition().x, bBox.getPosition().y, bBox.getPosition().z, bBox.getSize().x, bBox.getSize().y, bBox.getSize().z);
					}

					return false;
				}

				// L = A1 x B2
				ra = a[0] * fabs(R[2][2]) + a[2] * fabs(R[0][2]);
				rb = b[0] * fabs(R[1][1]) + b[1] * fabs(R[1][0]);
				t = fabs(T[0] * R[2][2] - T[2] * R[0][2]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bBox.getPosition().x, bBox.getPosition().y, bBox.getPosition().z, bBox.getSize().x, bBox.getSize().y, bBox.getSize().z);
					}

					return false;
				}

				// L = A2 x B0
				ra = a[0] * fabs(R[1][0]) + a[1] * fabs(R[0][0]);
				rb = b[1] * fabs(R[2][2]) + b[2] * fabs(R[2][1]);
				t = fabs(T[1] * R[0][0] - T[0] * R[1][0]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bBox.getPosition().x, bBox.getPosition().y, bBox.getPosition().z, bBox.getSize().x, bBox.getSize().y, bBox.getSize().z);
					}

					return false;
				}

				// L = A2 x B1
				ra = a[0] * fabs(R[1][1]) + a[1] * fabs(R[0][1]);
				rb = b[0] * fabs(R[2][2]) + b[2] * fabs(R[2][0]);
				t = fabs(T[1] * R[0][1] - T[0] * R[1][1]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bBox.getPosition().x, bBox.getPosition().y, bBox.getPosition().z, bBox.getSize().x, bBox.getSize().y, bBox.getSize().z);
					}

					return false;
				}

				// L = A2 x B2
				ra = a[0] * fabs(R[1][2]) + a[1] * fabs(R[0][2]);
				rb = b[0] * fabs(R[2][1]) + b[1] * fabs(R[2][0]);
				t = fabs(T[1] * R[0][2] - T[0] * R[1][2]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bBox.getPosition().x, bBox.getPosition().y, bBox.getPosition().z, bBox.getSize().x, bBox.getSize().y, bBox.getSize().z);
					}

					return false;
				}

				// no separating axis found, the two boxes overlap

				if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
				{
					printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == true\n",
						aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
						bBox.getPosition().x, bBox.getPosition().y, bBox.getPosition().z, bBox.getSize().x, bBox.getSize().y, bBox.getSize().z);
				}

				return true;
			}

			__host__ __device__ __inline__ static bool check(const Shapes::Box& aBox, const Shapes::Quad& bQuad)
			{
				if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					atomicInc(&Device::NumCollisionChecks, INT_MAX);
#else
					Host::NumCollisionChecks = (Host::NumCollisionChecks + 1) % INT_MAX;
#endif
				}

				const math::float3x4& A = aBox.getModel();
				const math::float3x4& B = bQuad.getModel();

				math::float3 a = aBox.getHalfExtents();
				math::float3 b = bQuad.getSize() * 0.5f;
				math::float3 v = bQuad.getPosition() - aBox.getPosition();
				math::float3 T(dot(v, A.column1()), dot(v, A.column2()), dot(v, A.column3()));

				float R[3][3];
				float ra, rb, t;
				unsigned int i, k;

				R[0][0] = dot(A.column1(), B.column1());
				R[0][1] = dot(A.column1(), B.column2());
				R[0][2] = dot(A.column1(), B.column3());
				R[1][0] = dot(A.column2(), B.column1());
				R[1][1] = dot(A.column2(), B.column2());
				R[1][2] = dot(A.column2(), B.column3());
				R[2][0] = dot(A.column3(), B.column1());
				R[2][1] = dot(A.column3(), B.column2());
				R[2][2] = dot(A.column3(), B.column3());

				for (i = 0; i < 3; i++)
				{
					ra = a[i];
					rb = b[0] * fabs(R[i][0]) + b[1] * fabs(R[i][1]);
					t = fabs(T[i]);
					if (t > ra + rb)
					{
						if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
						{
							printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
								aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
								bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
						}

						return false;
					}
				}

				for (k = 0; k < 3; k++)
				{
					ra = a[0] * fabs(R[0][k]) + a[1] * fabs(R[1][k]) + a[2] * fabs(R[2][k]);
					rb = b[k];
					t = fabs(T[0] * R[0][k] + T[1] * R[1][k] + T[2] * R[2][k]);
					if (t > ra + rb)
					{
						if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
						{
							printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
								aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
								bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
						}

						return false;
					}
				}

				// 9 cross products

				// L = A0 x B0
				ra = a[1] * fabs(R[2][0]) + a[2] * fabs(R[1][0]);
				rb = b[1] * fabs(R[0][2]);
				t = fabs(T[2] * R[1][0] - T[1] * R[2][0]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A0 x B1
				ra = a[1] * fabs(R[2][1]) + a[2] * fabs(R[1][1]);
				rb = b[0] * fabs(R[0][2]);
				t = fabs(T[2] * R[1][1] - T[1] * R[2][1]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A0 x B2
				ra = a[1] * fabs(R[2][2]) + a[2] * fabs(R[1][2]);
				rb = b[0] * fabs(R[0][1]) + b[1] * fabs(R[0][0]);
				t = fabs(T[2] * R[1][2] - T[1] * R[2][2]);

				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A1 x B0
				ra = a[0] * fabs(R[2][0]) + a[2] * fabs(R[0][0]);
				rb = b[1] * fabs(R[1][2]);
				t = fabs(T[0] * R[2][0] - T[2] * R[0][0]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A1 x B1
				ra = a[0] * fabs(R[2][1]) + a[2] * fabs(R[0][1]);
				rb = b[0] * fabs(R[1][2]);
				t = fabs(T[0] * R[2][1] - T[2] * R[0][1]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A1 x B2
				ra = a[0] * fabs(R[2][2]) + a[2] * fabs(R[0][2]);
				rb = b[0] * fabs(R[1][1]) + b[1] * fabs(R[1][0]);
				t = fabs(T[0] * R[2][2] - T[2] * R[0][2]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A2 x B0
				ra = a[0] * fabs(R[1][0]) + a[1] * fabs(R[0][0]);
				rb = b[1] * fabs(R[2][2]);
				t = fabs(T[1] * R[0][0] - T[0] * R[1][0]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A2 x B1
				ra = a[0] * fabs(R[1][1]) + a[1] * fabs(R[0][1]);
				rb = b[0] * fabs(R[2][2]);
				t = fabs(T[1] * R[0][1] - T[0] * R[1][1]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A2 x B2
				ra = a[0] * fabs(R[1][2]) + a[1] * fabs(R[0][2]);
				rb = b[0] * fabs(R[2][1]) + b[1] * fabs(R[2][0]);
				t = fabs(T[1] * R[0][2] - T[0] * R[1][2]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// no separating axis found, the two boxes overlap

				if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
				{
					printf("[Collision] Box[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == true\n",
						aBox.getPosition().x, aBox.getPosition().y, aBox.getPosition().z, aBox.getSize().x, aBox.getSize().y, aBox.getSize().z,
						bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
				}

				return true;
			}

			__host__ __device__ __inline__ static bool check(const Shapes::Box& aBox, const Shapes::Sphere& bSphere)
			{
				if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					atomicInc(&Device::NumCollisionChecks, INT_MAX);
#else
					Host::NumCollisionChecks = (Host::NumCollisionChecks + 1) % INT_MAX;
#endif
				}

				// TODO:
				return false;
			}

			__host__ __device__ __inline__ static bool check(const Shapes::Quad& aQuad, const Shapes::Quad& bQuad)
			{
				if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					atomicInc(&Device::NumCollisionChecks, INT_MAX);
#else
					Host::NumCollisionChecks = (Host::NumCollisionChecks + 1) % INT_MAX;
#endif
				}

				const math::float3x4& A = aQuad.getModel();
				const math::float3x4& B = bQuad.getModel();

				math::float3 a = aQuad.getSize() * 0.5f;
				math::float3 b = bQuad.getSize() * 0.5f;
				math::float3 v = bQuad.getPosition() - aQuad.getPosition();
				math::float3 T(dot(v, A.column1()), dot(v, A.column2()), dot(v, A.column3()));

				float R[3][3];
				float ra, rb, t;
				unsigned int i, k;

				R[0][0] = dot(A.column1(), B.column1());
				R[0][1] = dot(A.column1(), B.column2());
				R[0][2] = dot(A.column1(), B.column3());
				R[1][0] = dot(A.column2(), B.column1());
				R[1][1] = dot(A.column2(), B.column2());
				R[1][2] = dot(A.column2(), B.column3());
				R[2][0] = dot(A.column3(), B.column1());
				R[2][1] = dot(A.column3(), B.column2());
				R[2][2] = dot(A.column3(), B.column3());

				for (i = 0; i < 3; i++)
				{
					ra = a[i];
					rb = b[0] * fabs(R[i][0]) + b[1] * fabs(R[i][1]);
					t = fabs(T[i]);
					if (t > ra + rb)
					{
						if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
						{
							printf("[Collision] Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
								aQuad.getPosition().x, aQuad.getPosition().y, aQuad.getPosition().z, aQuad.getSize().x, aQuad.getSize().y, aQuad.getSize().z,
								bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
						}

						return false;
					}
				}

				for (k = 0; k < 3; k++)
				{
					ra = a[0] * fabs(R[0][k]) + a[1] * fabs(R[1][k]);
					rb = b[k];
					t = fabs(T[0] * R[0][k] + T[1] * R[1][k] + T[2] * R[2][k]);
					if (t > ra + rb)
					{
						if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
						{
							printf("[Collision] Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
								aQuad.getPosition().x, aQuad.getPosition().y, aQuad.getPosition().z, aQuad.getSize().x, aQuad.getSize().y, aQuad.getSize().z,
								bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
						}

						return false;
					}
				}

				// 9 cross products

				// L = A0 x B0
				ra = a[1] * fabs(R[2][0]);
				rb = b[1] * fabs(R[0][2]);
				t = fabs(T[2] * R[1][0] - T[1] * R[2][0]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aQuad.getPosition().x, aQuad.getPosition().y, aQuad.getPosition().z, aQuad.getSize().x, aQuad.getSize().y, aQuad.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A0 x B1
				ra = a[1] * fabs(R[2][1]);
				rb = b[0] * fabs(R[0][2]);
				t = fabs(T[2] * R[1][1] - T[1] * R[2][1]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aQuad.getPosition().x, aQuad.getPosition().y, aQuad.getPosition().z, aQuad.getSize().x, aQuad.getSize().y, aQuad.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A0 x B2
				ra = a[1] * fabs(R[2][2]);
				rb = b[0] * fabs(R[0][1]) + b[1] * fabs(R[0][0]);
				t = fabs(T[2] * R[1][2] - T[1] * R[2][2]);

				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aQuad.getPosition().x, aQuad.getPosition().y, aQuad.getPosition().z, aQuad.getSize().x, aQuad.getSize().y, aQuad.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A1 x B0
				ra = a[0] * fabs(R[2][0]);
				rb = b[1] * fabs(R[1][2]);
				t = fabs(T[0] * R[2][0] - T[2] * R[0][0]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aQuad.getPosition().x, aQuad.getPosition().y, aQuad.getPosition().z, aQuad.getSize().x, aQuad.getSize().y, aQuad.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A1 x B1
				ra = a[0] * fabs(R[2][1]);
				rb = b[0] * fabs(R[1][2]);
				t = fabs(T[0] * R[2][1] - T[2] * R[0][1]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aQuad.getPosition().x, aQuad.getPosition().y, aQuad.getPosition().z, aQuad.getSize().x, aQuad.getSize().y, aQuad.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A1 x B2
				ra = a[0] * fabs(R[2][2]);
				rb = b[0] * fabs(R[1][1]) + b[1] * fabs(R[1][0]);
				t = fabs(T[0] * R[2][2] - T[2] * R[0][2]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aQuad.getPosition().x, aQuad.getPosition().y, aQuad.getPosition().z, aQuad.getSize().x, aQuad.getSize().y, aQuad.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A2 x B0
				ra = a[0] * fabs(R[1][0]) + a[1] * fabs(R[0][0]);
				rb = b[1] * fabs(R[2][2]);
				t = fabs(T[1] * R[0][0] - T[0] * R[1][0]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aQuad.getPosition().x, aQuad.getPosition().y, aQuad.getPosition().z, aQuad.getSize().x, aQuad.getSize().y, aQuad.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A2 x B1
				ra = a[0] * fabs(R[1][1]) + a[1] * fabs(R[0][1]);
				rb = b[0] * fabs(R[2][2]);
				t = fabs(T[1] * R[0][1] - T[0] * R[1][1]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aQuad.getPosition().x, aQuad.getPosition().y, aQuad.getPosition().z, aQuad.getSize().x, aQuad.getSize().y, aQuad.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// L = A2 x B2
				ra = a[0] * fabs(R[1][2]) + a[1] * fabs(R[0][2]);
				rb = b[0] * fabs(R[2][1]) + b[1] * fabs(R[2][0]);
				t = fabs(T[1] * R[0][2] - T[0] * R[1][2]);
				if (t > ra + rb)
				{
					if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
					{
						printf("[Collision] Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == false\n",
							aQuad.getPosition().x, aQuad.getPosition().y, aQuad.getPosition().z, aQuad.getSize().x, aQuad.getSize().y, aQuad.getSize().z,
							bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
					}

					return false;
				}

				// no separating axis found, the two boxes overlap

				if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
				{
					printf("[Collision] Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] & Quad[position=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f)] == true\n",
						aQuad.getPosition().x, aQuad.getPosition().y, aQuad.getPosition().z, aQuad.getSize().x, aQuad.getSize().y, aQuad.getSize().z,
						bQuad.getPosition().x, bQuad.getPosition().y, bQuad.getPosition().z, bQuad.getSize().x, bQuad.getSize().y, bQuad.getSize().z);
				}

				return true;
			}

			__host__ __device__ __inline__ static bool check(const Shapes::Quad& aQuad, const Shapes::Box& bBox)
			{
				return check(bBox, aQuad);
			}

			__host__ __device__ __inline__ static bool check(const Shapes::Quad& aQuad, const Shapes::Sphere& bSphere)
			{
				if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					atomicInc(&Device::NumCollisionChecks, INT_MAX);
#else
					Host::NumCollisionChecks = (Host::NumCollisionChecks + 1) % INT_MAX;
#endif
				}

				// TODO:
				return false;
			}

			__host__ __device__ static bool check(const Shapes::Sphere& aSphere, const Shapes::Sphere& bSphere)
			{
				if (T::IsEnabled<DebugFlags::CollisionCheck>::Result)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					atomicInc(&Device::NumCollisionChecks, INT_MAX);
#else
					Host::NumCollisionChecks = (Host::NumCollisionChecks + 1) % INT_MAX;
#endif
				}

				// TODO:
				return false;
			}

			__host__ __device__ static bool check(const Shapes::Sphere& aSphere, const Shapes::Box& bBox)
			{
				return check(bBox, aSphere);
			}

			__host__ __device__ static bool check(const Shapes::Sphere& aSphere, const Shapes::Quad& bQuad)
			{
				return check(bQuad, aSphere);
			}

		};

	}

}
