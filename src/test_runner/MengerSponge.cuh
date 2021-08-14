#pragma once

#include "SceneUtils.cuh"

#include <cuda_runtime_api.h>
#include <pga/core/Core.h>
#include <pga/core/Grid.cuh>
#include <pga/core/GPUTechnique.h>

#include <map>
#include <string>

using namespace PGA;
using namespace PGA::Operators;
using namespace PGA::Shapes;
using namespace PGA::Parameters;

namespace Scene
{
	const unsigned int MaxIterations = 5;
	const std::map<size_t, size_t> GenFuncCounters = { { 1, 1 } };
	const bool Instrumented = false;
	const unsigned int NumEdges = 1;
	const unsigned int NumSubgraphs = 1;
	const unsigned int QueueSize = 1000000 /* queue items */;
	const GPU::Technique Technique = GPU::Technique::MEGAKERNEL;

	struct ProcedureList : T::List <
		/* procedure[0]= */ Proc<Box, Subdivide<DynParam<0>, T::Pair<DynParam<1>, DCall<0>>, T::Pair<DynParam<2>, DCall<1>>, T::Pair<DynParam<3>, DCall<2>>>, 1>,
		/* procedure[1]= */ Proc<Box, Discard, 1>,
		/* procedure[2]= */ Proc<Box, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>, 1>,
		/* procedure[3]= */ Proc<Box, Generate<false, 1 /* Instanced Triangle Mesh */, DynParam<0>>, 1>,
	> {};

	DispatchTable dispatchTable = {
		/* entries[0]= */{ /* procIdx= */2, /* parameters[2]= */{ /* 0 */{ PT_SCALAR, { 0 } }, /* 1 */{ PT_SCALAR, { 1 } } }, /* successors[2]= */{ /* 0 */{ 1, 0 }, /* 1 */{ 2, 0 } } },
		/* entries[1]= */{ /* procIdx= */3, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 0 } } }, /* successors[0]= */{} },
		/* entries[2]= */{ /* procIdx= */0, /* parameters[4]= */{ /* 0 */{ PT_SCALAR, { 0 } }, /* 1 */{ PT_SCALAR, { -1 } }, /* 2 */{ PT_SCALAR, { -1 } }, /* 3 */{ PT_SCALAR, { -1 } } }, /* successors[3]= */{ /* 0 */{ 3, 0 }, /* 1 */{ 7, 0 }, /* 2 */{ 3, 0 } } },
		/* entries[3]= */{ /* procIdx= */0, /* parameters[4]= */{ /* 0 */{ PT_SCALAR, { 1 } }, /* 1 */{ PT_SCALAR, { -1 } }, /* 2 */{ PT_SCALAR, { -1 } }, /* 3 */{ PT_SCALAR, { -1 } } }, /* successors[3]= */{ /* 0 */{ 4, 0 }, /* 1 */{ 5, 0 }, /* 2 */{ 4, 0 } } },
		/* entries[4]= */{ /* procIdx= */0, /* parameters[4]= */{ /* 0 */{ PT_SCALAR, { 2 } }, /* 1 */{ PT_SCALAR, { -1 } }, /* 2 */{ PT_SCALAR, { -1 } }, /* 3 */{ PT_SCALAR, { -1 } } }, /* successors[3]= */{ /* 0 */{ 0, 0 }, /* 1 */{ 0, 0 }, /* 2 */{ 0, 0 } } },
		/* entries[5]= */{ /* procIdx= */0, /* parameters[4]= */{ /* 0 */{ PT_SCALAR, { 2 } }, /* 1 */{ PT_SCALAR, { -1 } }, /* 2 */{ PT_SCALAR, { -1 } }, /* 3 */{ PT_SCALAR, { -1 } } }, /* successors[3]= */{ /* 0 */{ 0, 0 }, /* 1 */{ 6, 0 }, /* 2 */{ 0, 0 } } },
		/* entries[6]= */{ /* procIdx= */1, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[7]= */{ /* procIdx= */0, /* parameters[4]= */{ /* 0 */{ PT_SCALAR, { 1 } }, /* 1 */{ PT_SCALAR, { -1 } }, /* 2 */{ PT_SCALAR, { -1 } }, /* 3 */{ PT_SCALAR, { -1 } } }, /* successors[3]= */{ /* 0 */{ 5, 0 }, /* 1 */{ 8, 0 }, /* 2 */{ 5, 0 } } },
		/* entries[8]= */{ /* procIdx= */1, /* parameters[0]= */{}, /* successors[0]= */{} },
	};

	template <unsigned int AxiomId>
	struct AxiomTraits : Grid::DefaultAxiomTraits< Box >
	{
		__host__ __device__ __inline__ static int getEntryIndex()
		{
			return 2;
		}

		__host__ __device__ __inline__ static math::float3 getSize(const Box& shape)
		{
			float size = static_cast<float>(math::pow(3, (SceneUtils::GlobalVars::getAttribute(0) % MaxIterations)));
			return math::float3(size, size, size);
		}

	};

	typedef Grid::Static<AxiomTraits, 1, 1, 1> AxiomGenerator;

	struct Controller : public SceneUtils::AttributeSceneController<1>
	{
		static const bool IsInitializable = false;
		static const bool IsGridded = false;

	};

	struct Configurator
	{
		static const bool IsFile = false;
		static std::string value()
		{
			return "<?xml version=\"1.0\" encoding=\"UTF - 8\"?>				"
				   "<Configuration>												"
				   "<InstancedTriangleMesh										"
				   "	index=\"0\"												"
				   "	maxNumElements=\"3200000\"								"
				   "	type=\"0\"												"
				   "	shape=\"Box\"											"
				   ">															"
				   "	<Material type=\"0\">									"
				   "		<Attribute name=\"color0\" value=\"(1,1,1,1)\" />	"
				   "	</Material>												"
				   "</InstancedTriangleMesh>									"
				   "</Configuration>											";
		}

	};

	std::string testName()
	{
		return std::string("menger_sponge_") + std::to_string(Grid::GlobalVars::getNumElements()) + ((Instrumented) ? "_i" : "");
	}

	std::string sceneName()
	{
		return std::string("menger_sponge") + ((Instrumented) ? "_i" : "");
	}

}
