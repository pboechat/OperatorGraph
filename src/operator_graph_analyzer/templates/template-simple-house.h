#pragma once

#include "ScenesCommons.cuh"

#include <pga/core/GlobalConstants.h>
#include <pga/core/DispatchTable.h>
#include <pga/core/Proc.cuh>
#include <pga/core/Operators.cuh>
#include <pga/core/Shapes.cuh>
#include <pga/core/Parameters.cuh>
#include <pga/core/Axis.h>
#include <pga/core/TStdLib.h>
#include <pga/core/Grid.cuh>
#include <pga/rendering/SingleTerminalTraits.cuh>
#include <pga/rendering/ShapeMesh.cuh>


#include <cuda_runtime_api.h>

#include <string>
#include <map>

using namespace PGA;
using namespace PGA::Operators;
using namespace PGA::Shapes;
using namespace PGA::Parameters;
using namespace PGA::AxiomGenerators;
using namespace PGA::Rendering;


namespace Scene
{
	const unsigned int QueueSize = <avgQueueSize>;
	const std::map<std::size_t, std::size_t> GenFuncCounters = <genFuncCounters>;
	const unsigned int NumEdges = <numEdges>;
	const unsigned int NumSubgraphs = <numSubgraphs>;
	const bool Instrumented = <instrumented>;
	const unsigned int MaxNumAxioms = <gridY> * <gridX>;

<code>

	const unsigned int NumPhases = 1;

	struct Controller : Scenes::GriddedScene < <gridY>, <gridX> >
	{
		template <unsigned int AxiomId>
		struct AxiomTraits : Grid::DefaultAxiomTraits < Box, 0 >
		{
			__device__ __inline__ static math::float3 getSize(const Box& shape)
			{
				return math::float3(1.0f, 1.0f, 1.0f);
			}

		};

		typedef Grid::Dynamic<AxiomTraits, 1> AxiomGenerator;

		struct TerminalsTraits
		{
			__host__ __device__ __inline__ static unsigned int getMaxNumVertices(unsigned int terminalIndex)
			{
				switch (terminalIndex)
				{
				case 0:
					return 400 * 4 * MaxNumAxioms;
					break;
				case 1:
					return 1 * 4 * MaxNumAxioms;
					break;
				case 2:
					return 400 * 4 * MaxNumAxioms;
					break;
				default:
					// FIXME: checking invariants
					//throw std::exception("getMaxNumVertices: unknown terminalIndex");
					break;

				}

			}

			__host__ __device__ __inline__ static unsigned int getMaxNumIndices(unsigned int terminalIndex)
			{
				switch (terminalIndex)
				{
				case 0:
					return 400 * 6 * MaxNumAxioms;
					break;
				case 1:
					return 1 * 6 * MaxNumAxioms;
					break;
				case 2:
					return 400 * 6 * MaxNumAxioms;
					break;
				default:
					// FIXME: checking invariants
					//throw std::exception("getMaxNumIndices: unknown terminalIndex");
					return 0;

				}
			}

			__host__ __device__ __inline__ static unsigned int getMaxNumElements(unsigned int terminalIndex)
			{
				switch (terminalIndex)
				{
				case 0:
					return 400 * MaxNumAxioms;
					break;
				case 1:
					return 1 * MaxNumAxioms;
					break;
				case 2:
					return 400 * MaxNumAxioms;
					break;
				default:
					// FIXME: checking invariants
					//throw std::exception("getMaxNumInstances: unknown terminalIndex");
					return 0;

				}
			}

			__host__ __inline__ static std::unique_ptr<InstanceMesh> createInstanceMesh(unsigned int terminalIndex)
			{
				math::float2 vertices[3] = { { -0.25, -0.5 }, { 0.25, 0 }, { -0.25, 0.5 } };
				switch (terminalIndex)
				{
				case 0:
					return std::unique_ptr<InstanceMesh>(new ShapeMesh<Quad>());
					break;
				case 1:
					//return std::unique_ptr<InstanceMesh>(new ShapeMesh<DynamicConvexPolygon<3> >(vertices, 3));
					return std::unique_ptr<InstanceMesh>(new ShapeMesh<DynamicRightPrism<PGA::Constants::MaxNumSides> >(DynamicRightPrism<PGA::Constants::MaxNumSides>(vertices, 3)));
					break;
				case 2:
					return std::unique_ptr<InstanceMesh>(new ShapeMesh<Quad>());
					break;
				default:
					// FIXME: checking invariants
					throw std::exception("createInstanceMesh: unknown terminalIndex");

				}
				
			}

			__host__ __inline__ static std::unique_ptr<Material> createTriangleMeshMaterial(unsigned int terminalIndex)
			{
				std::unique_ptr<Material> material(new Material(std::unique_ptr<Shaders::ColorShader>(new Shaders::ColorShader())));
				return material;
			}

			__host__ __inline__ static std::unique_ptr<Material> createInstanceMaterial(unsigned int terminalIndex)
			{
				std::unique_ptr<Material> material(new Material(std::unique_ptr<Shaders::ColorShader>(new Shaders::ColorShader())));
				return material;
			}

		};

		static std::string name()
		{
			//return std::string("partition_<idx>_<uid>_") + std::to_string(getNumElements()) + ((<optimized>) ? "_o" : "") + ((Instrumented) ? "_i" : "");
			return std::string("partition_<idx>_") + std::to_string(getNumElements()) + ((<optimized>) ? "_o" : "") + ((Instrumented) ? "_i" : "");
		}
		
	};

}
