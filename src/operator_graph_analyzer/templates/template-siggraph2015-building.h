#pragma once

#include "ScenesCommons.cuh"

#include <cuda_runtime_api.h>
#include <pga/core/Axis.h>
#include <pga/core/DispatchTable.h>
#include <pga/core/GlobalConstants.h>
#include <pga/core/Grid.cuh>
#include <pga/core/Operators.cuh>
#include <pga/core/Parameters.cuh>
#include <pga/core/Proc.cuh>
#include <pga/core/Shapes.cuh>
#include <pga/core/TStdLib.h>
#include <pga/rendering/ColorShader.h>
#include <pga/rendering/ShapeMesh.cuh>

#include <map>
#include <string>

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
				return math::float3(10.0f, 20.0f, 10.0f);
			}

		};

		typedef Grid::Dynamic<AxiomTraits, 1> AxiomGenerator;

		struct TerminalsTraits
		{
			__host__ __device__ __inline__ static unsigned int getMaxNumVertices(unsigned int terminalIndex)
			{
				return 0;
			}

			__host__ __device__ __inline__ static unsigned int getMaxNumIndices(unsigned int terminalIndex)
			{
				return 0;
			}

			__host__ __device__ __inline__ static unsigned int getMaxNumElements(unsigned int terminalIndex)
			{
				switch (terminalIndex)
				{
				case 0:
					return 4 * MaxNumAxioms;
				case 1:
					return 12 * MaxNumAxioms;
				case 2:
					return 464 * MaxNumAxioms;
				case 3:
					return 476 * MaxNumAxioms;
				case 4:
					return 6 * MaxNumAxioms;
				case 5:
					return 1276 * MaxNumAxioms;
				case 6:
					return 116 * MaxNumAxioms;
				default:
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
#if defined(CHECK_INVARIANTS) && (CHECK_INVARIANTS == 2 || CHECK_INVARIANTS == 3)
					printf("TerminalTraits::getMaxNumElements(..): invalid terminal index [terminalIndex=%d] (CUDA thread %d %d)\n", terminalIndex, threadIdx.x, blockIdx.x);
					asm("trap;");
#endif
#else
#if defined(CHECK_INVARIANTS) && (CHECK_INVARIANTS == 1 || CHECK_INVARIANTS == 3)
					throw std::runtime_error(("TerminalTraits::getMaxNumElements(..): invalid terminal index [terminalIndex=" + std::to_string(terminalIndex) + "]").c_str());
#endif
#endif
					return 0;
				}
			}

			__host__ __inline__ static std::unique_ptr<InstanceMesh> createInstanceMesh(unsigned int terminalIndex)
			{
				switch (terminalIndex)
				{
				case 1:
				case 4:
				case 5:
				case 6:
					return std::unique_ptr<InstanceMesh>(new ShapeMesh<Box>());
				case 0:
				case 2:
				case 3:
					return std::unique_ptr<InstanceMesh>(new ShapeMesh<Quad>());
				default:
#if defined(CHECK_INVARIANTS) && (CHECK_INVARIANTS == 1 || CHECK_INVARIANTS == 3)
					throw std::runtime_error(("TerminalTraits::createInstanceMesh(..): invalid terminal index [terminalIndex=" + std::to_string(terminalIndex) + "]").c_str());
#endif
					return nullptr;
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
