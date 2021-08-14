#pragma once

#include "Axis.h"
#include "DebugFlags.h"
#include "GlobalVariables.cuh"
#include "Shapes.cuh"
#include "Symbol.cuh"
#include "SymbolDecorator.cuh"
#include "TStdLib.h"

#include <cuda_runtime_api.h>
#include <math/matrix.h>
#include <math/vector.h>

#include <string>

namespace PGA
{
	namespace Operators
	{
		namespace
		{
			template <typename ShapeT>
			struct FaceSplitter
			{
				__host__ __device__ __inline__ static auto splitSideFace(const ShapeT& shape, unsigned int faceIndex) -> decltype(shape.newSideFace())
				{
					math::float3 size = shape.getSize();
					math::float2x2 scale = math::float2x2::scale(size.xz());
					math::float2 p0 = scale * shape.getVertex((faceIndex == 0) ? shape.getNumSides() - 1 : faceIndex - 1);
					math::float2 p1 = scale * shape.getVertex(faceIndex);
					p0.y = -p0.y;
					p1.y = -p1.y;
					math::float2 d = p1 - p0;
					float x = length(d);
					math::float2 offset = p0 + (p1 - p0) * 0.5f;
					d = d * (1.0f / x);
					math::float4x4 facing = math::float4x4(
						d.x, 0.0f, -d.y, 0.0f,
						0.0f, 1.0f, 0.0f, 0.0f,
						d.y, 0.0f, d.x, 0.0f,
						0.0f, 0.0f, 0.0f, 1.0f
					);
					auto face = shape.newSideFace();
					face.setModel(shape.getModel4() * math::float4x4::translate(math::float3(offset.x, 0.0f, offset.y)) * facing);
					face.setSize(math::float3(x, size.y, 1.0f));
					// NOTE: next seed modifier ranges from 2 to num. sides + 2 for side faces 
					// (0 and 1 are reserved for top and bottom faces respectively)
					face.setSeed(shape.generateNextSeed(faceIndex + 2.0f));
					face.setCustomAttribute(shape.getCustomAttribute());
					return face;
				}

				__host__ __device__ __inline__ static auto splitCapFace(const ShapeT& shape, bool isBottom) -> decltype(shape.newCapFace(isBottom))
				{
					math::float3 size = shape.getSize();
					float Sign = 1.0f - 2.0f * isBottom;
					auto face = shape.newCapFace(isBottom);
					face.setModel(shape.getModel4() *
						math::float4x4::translate(math::float3(0.0f, size.y * Sign * 0.5f, 0.0f)) *
						math::float4x4(
							1.0f, 0.0f, 0.0f, 0.0f,
							0.0f, 0.0f, Sign, 0.0f,
							0.0f, -Sign, 0.0f, 0.0f,
							0.0f, 0.0f, 0.0f, 1.0f
						)
					);
					face.setSize(math::float3(size.xz(), 1.0f));
					// NOTE: next seed modifier is 0 for top and 1 for bottom faces
					face.setSeed(shape.generateNextSeed((isBottom) ? 1.0f : 0.0f));
					face.setCustomAttribute(shape.getCustomAttribute());
					return face;
				}

			};

			template <>
			struct FaceSplitter < Shapes::Box >
			{
				// NOTE:
				// 0 => Back
				// 1 => Front
				// 2 => Left
				// 3 => Right
				__host__ __device__ __inline__ static Shapes::Quad splitSideFace(const Shapes::Box& shape, unsigned int faceIndex)
				{
					math::float4x4 model = shape.getModel4();
					math::float3 size = shape.getSize();
					math::float3 halfExtents = size * 0.5f;
					Shapes::Quad face;
					switch (faceIndex)
					{
					case 0: // Back
						face.setModel(model * math::float4x4::translate(math::float3(0.0f, 0.0f, -halfExtents.z)) *
							math::float4x4(
								-1.0f, 0.0f, 0.0f, 0.0f,
								0.0f, 1.0f, 0.0f, 0.0f,
								0.0f, 0.0f, -1.0f, 0.0f,
								0.0f, 0.0f, 0.0f, 1.0f
							)
						);
						face.setSize(math::float3(size.xy(), 1.0f));
						break;
					case 1: // Front
						face.setModel(model * math::float4x4::translate(math::float3(0.0f, 0.0f, halfExtents.z)));
						face.setSize(math::float3(size.xy(), 1.0f));
						break;
					case 2: // Left
						face.setModel(model * math::float4x4::translate(math::float3(-halfExtents.x, 0.0f, 0.0f)) *
							math::float4x4(
								0.0f, 0.0f, -1.0f, 0.0f,
								0.0f, 1.0f, 0.0f, 0.0f,
								1.0f, 0.0f, 0.0f, 0.0f,
								0.0f, 0.0f, 0.0f, 1.0f
							)
						);
						face.setSize(math::float3(size.zy(), 1.0f));
						break;
					case 3: // Right
						face.setModel(model * math::float4x4::translate(math::float3(halfExtents.x, 0.0f, 0.0f)) *
							math::float4x4(
								0.0f, 0.0f, 1.0f, 0.0f,
								0.0f, 1.0f, 0.0f, 0.0f,
								-1.0f, 0.0f, 0.0f, 0.0f,
								0.0f, 0.0f, 0.0f, 1.0f
							)
						);
						face.setSize(math::float3(size.zy(), 1.0f));
						break;
					}
					// NOTE: details in FaceSplitter<ShapeT>::splitSideFace(..)
					face.setSeed(shape.generateNextSeed(faceIndex + 2.0f));
					face.setCustomAttribute(shape.getCustomAttribute());
					return face;
				}

				__host__ __device__ __inline__ static Shapes::Quad splitCapFace(const Shapes::Box& shape, bool isBottom)
				{
					math::float4x4 model = shape.getModel4();
					math::float3 size = shape.getSize();
					math::float3 halfExtents = shape.getHalfExtents();
					Shapes::Quad face;
					face.invert = isBottom;
					if (isBottom) // Bottom
					{
						face.setModel(model * math::float4x4::translate(math::float3(0.0f, -halfExtents.y, 0.0f)) *
							math::float4x4(
								1.0f, 0.0f, 0.0f, 0.0f,
								0.0f, 0.0f, -1.0f, 0.0f,
								0.0f, 1.0f, 0.0f, 0.0f,
								0.0f, 0.0f, 0.0f, 1.0f
							)
						);
						face.setSize(math::float3(size.xz(), 1.0f));
					}
					else // Top
					{
						face.setModel(model * math::float4x4::translate(math::float3(0.0f, halfExtents.y, 0.0f)) *
							math::float4x4(
								1.0f, 0.0f, 0.0f, 0.0f,
								0.0f, 0.0f, 1.0f, 0.0f,
								0.0f, -1.0f, 0.0f, 0.0f,
								0.0f, 0.0f, 0.0f, 1.0f
							)
						);
						face.setSize(math::float3(size.xz(), 1.0f));
					}
					// NOTE: details in FaceSplitter<ShapeT>::splitCapFace(..)
					face.setSeed(shape.generateNextSeed((isBottom) ? 1.0f : 0.0f));
					face.setCustomAttribute(shape.getCustomAttribute());
					return face;
				}

			};

			template <typename ShapeT>
			struct ParallelFaceSplitter;

			template <>
			struct ParallelFaceSplitter < Shapes::Box >
			{
				__host__ __device__ __inline__ static Shapes::Quad splitFace(const Shapes::Box& shape, int faceIndex)
				{
					// NOTE: correspondence between faceIndex and face direction
					// 0 => Back
					// 1 => Front
					// 2 => Left
					// 3 => Right
					// 4 => Bottom
					// 5 => Top

					float sign = ((faceIndex & 0x1) * 2.0f) - 1.0f;
					float a = (float)(faceIndex < 2);
					float b1 = (faceIndex & 0x2) * 0.5f;
					float b2 = (faceIndex & 0x4) * 0.25f;

					math::float3 n(b1 * sign, b2 * sign, a * sign);
					math::float3 y(0.0f, 1.0f - b2, b2 * -sign);
					math::float3 x = cross(y, n);

					math::float4x4 model = shape.getModel4();
					math::float4x4 rotation(
						x.x, y.x, n.x, 0.0f,
						x.y, y.y, n.y, 0.0f,
						x.z, y.z, n.z, 0.0f,
						0.0f, 0.0f, 0.0f, 1.0f
					);

					math::float3 size = shape.getSize();
					math::float4x4 translation = math::float4x4::translate(size * n * 0.5f);

					math::float3 s0(a + b2, 0.0f, b1);
					math::float3 s1(0.0f, 1.0f - b2, b2);

					Shapes::Quad face;

					face.setModel(model * translation * rotation);
					face.setSize(math::float3(dot(size, abs(s0)), dot(size, abs(s1)), 1.0f));
					// NOTE: further details in FaceSplitter<ShapeT>::splitSideFace(..) and FaceSplitter<ShapeT>::splitCapFace(..)
					unsigned int modifier = (faceIndex + 2) % 6;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					face.setSeed(shape.generateNextSeed(__float2uint_rd(modifier)));
#else
					face.setSeed(shape.generateNextSeed(static_cast<float>(modifier)));
#endif
					face.setCustomAttribute(shape.getCustomAttribute());
					return face;
				}

			};

		}

		template <bool ParallelT /* false */, typename TopOperatorT, typename BottomOperatorT, typename SidesOperatorT>
		class ComponentSplit
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				auto numSides = symbol->getNumSides();
				for (unsigned int faceIndex = 0; faceIndex < numSides; faceIndex++)
				{
					Symbol<typename Shapes::GetSideFaceType<ShapeT>::Result> sideNewSymbol(FaceSplitter<ShapeT>::splitSideFace(*symbol, faceIndex));
					SymbolDecorator<SidesOperatorT>::run(symbol, &sideNewSymbol);
					SidesOperatorT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, &sideNewSymbol, shared);
				}
				Symbol<typename Shapes::GetCapFaceType<ShapeT>::Result> bottomNewSymbol(FaceSplitter<ShapeT>::splitCapFace(*symbol, true));
				SymbolDecorator<BottomOperatorT>::run(symbol, &bottomNewSymbol);
				BottomOperatorT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, &bottomNewSymbol, shared);

				Symbol<typename Shapes::GetCapFaceType<ShapeT>::Result> topNewSymbol(FaceSplitter<ShapeT>::splitCapFace(*symbol, false));
				SymbolDecorator<TopOperatorT>::run(symbol, &topNewSymbol);
				TopOperatorT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, &topNewSymbol, shared);
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "ComponentSplit";
			}

			__host__ __inline__ static std::string toString()
			{
				return "ComponentSplit<false, " + TopOperatorT::toString() + ", " + BottomOperatorT::toString() + ", " + SidesOperatorT::toString() + ">";
			}

		};

		template <typename TopOperatorT, typename BottomOperatorT, typename SidesOperatorT>
		class ComponentSplit < true, TopOperatorT, BottomOperatorT, SidesOperatorT >
		{
		private:
			template <unsigned int NumFacesT, unsigned int FaceIndexT = 0>
			struct ForEachFace
			{
			private:
				typedef typename T::If<(FaceIndexT == NumFacesT - 1), TopOperatorT, typename T::If<(FaceIndexT == NumFacesT - 2), BottomOperatorT, SidesOperatorT>::Result>::Result NextOperator;

			public:
				template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename FaceShape>
				__host__ __device__ __inline__ static void dispatchInSerial(int localThreadId, int threadId, int numThreads, QueueT* queue, FaceShape& face, unsigned int* shared)
				{
					if (localThreadId == FaceIndexT)
					{
						// NOTE: It's necessary to create new object for output, because it has a different shape type
						Symbol<FaceShape> newSymbol(face);
						SymbolDecorator<NextOperator>::run(face, &newSymbol);
						NextOperator::template execute<ContextT>(threadId, numThreads, queue, &newSymbol, shared);
					}
					ForEachFace<NumFacesT, FaceIndexT + 1>::template dispatchInSerial<ContextT>(localThreadId, threadId, numThreads, queue, face, shared);
				}

			};

			template <unsigned int NumFacesT>
			struct ForEachFace <NumFacesT, NumFacesT >
			{
				template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename FaceShape>
				__host__ __device__ __inline__ static void dispatchInSerial(int localThreadId, int threadId, int numThreads, QueueT* queue, FaceShape& face, unsigned int* shared) {}

			};

		public:
			// NOTE: Currently only works with boxes
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				int localThreadId = threadId % numThreads;
				const auto NumFacesT = Shapes::GetNumFaces<ShapeT>::Result;
				// NOTE: 1 thread per face
				if (localThreadId < NumFacesT)
				{
					auto face = ParallelFaceSplitter<ShapeT>::splitFace(*symbol, localThreadId);
					ForEachFace<NumFacesT>::template dispatchInSerial<ContextT>(localThreadId, threadId, numThreads, queue, face, shared);
				}
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "ComponentSplit";
			}

			__host__ __inline__ static std::string toString()
			{
				return "ComponentSplit<true, " + TopOperatorT::toString() + ", " + BottomOperatorT::toString() + ", " + SidesOperatorT::toString() + ">";
			}

		};

	}

}
