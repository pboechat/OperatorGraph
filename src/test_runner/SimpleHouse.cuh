#pragma once

#include "SceneUtils.cuh"

#include <cuda_runtime_api.h>
#include <pga/core/Core.h>
#include <pga/core/Grid.cuh>

#include <map>
#include <string>

using namespace PGA;
using namespace PGA::Operators;
using namespace PGA::Shapes;
using namespace PGA::Parameters;

namespace Scene
{
	const std::map<size_t, size_t> GenFuncCounters = { { 0, 1 } };
	const bool Instrumented = false;
	const unsigned int NumEdges = 1;
	const unsigned int NumSubgraphs = 1;
	const unsigned int QueueSize = 1024 * 128;
	const GPU::Technique Technique = GPU::Technique::MEGAKERNEL;

	struct ProcedureList : T::List<
		/* 0 */ Proc<Box, Repeat<false, AxisParam<Y>, Scalar<1000, 3>, RepeatModeParam<ADJUST_TO_FILL>, DCall<0> /* 1 */>>,
		/* 1 */ Proc<Box, ComponentSplit<false, DCall<0> /* 6 */, DCall<1> /* 5 */, DCall<2> /* 2 */>>,
		/* 2 */ Proc<Quad, Repeat<false, AxisParam<X>, Scalar<500, 3>, RepeatModeParam<ADJUST_TO_FILL>, DCall<0> /* 3 */>>,
		/* 3 */ Proc<Quad,
			Subdivide<AxisParam<X>,
				T::Pair<Scalar<-1000, 3>, DCall<0>>, // 6
				T::Pair<Scalar<-2000, 3>, DCall<1>>, // 4
				T::Pair<Scalar<-1000, 3>, DCall<0>>  // 6
			>
		>,
		/* 4 */ Proc<Quad,
			Subdivide<AxisParam<Y>, 
				T::Pair<Scalar<-1000, 3>, DCall<0>>, // 6
				T::Pair<Scalar<-2000, 3>, DCall<1>>, // 5
				T::Pair<Scalar<-1000, 3>, DCall<0>>  // 6
			>
		>,
		/* 5 */ Proc<Quad, Discard>,
		/* 6 */ Proc<Quad, Extrude<AxisParam<Z>, Scalar<100, 3>, DCall<0>> /* 7 */>,
		/* 7 */ Proc<Box, Generate<false, 1  /* Instanced Triangle Mesh */>>,
	> {};

	DispatchTable dispatchTable = {
		/* 0 */{ 0, {  }, { { 1, 0 } } },
		/* 1 */{ 1, {  }, { { 6, 0 }, { 5, 0 }, { 2, 0 } } },
		/* 2 */{ 2, {  }, { { 3, 0 } } },
		/* 3 */{ 3, {  }, { { 6, 0 }, { 4, 0 }, { 6, 0 } } },
		/* 4 */{ 4, {  }, { { 6, 0 }, { 5, 0 }, { 6, 0 } } },
		/* 5 */{ 5, {  }, {  } },
		/* 6 */{ 6, {  }, { { 7, 0 } } },
		/* 7 */{ 7, {  }, {  } },
	};

	template <unsigned int AxiomId>
	struct AxiomTraits : Grid::DefaultAxiomTraits < Box >
	{
		__host__ __device__ __inline__ static int getEntryIndex()
		{
			return 0;
		}

		__host__ __device__ __inline__ static math::float3 getSize(const Box& shape)
		{
			return math::float3(2.0f, 5.0f, 2.0f);
		}

	};

	typedef Grid::Dynamic<AxiomTraits, 1> AxiomGenerator;

	struct Controller : SceneUtils::GriddedSceneController < 16, 16 >
	{
		static const bool IsInitializable = false;
		static const bool HasAttributes = false;

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
		return std::string("simple_house_") + std::to_string(Grid::GlobalVars::getNumElements()) + ((Instrumented) ? "_i" : "");
	}

	std::string sceneName()
	{
		return std::string("simple_house") + ((Instrumented) ? "_i" : "");
	}

}
