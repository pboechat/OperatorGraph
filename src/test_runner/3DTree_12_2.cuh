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
	const std::map<size_t, size_t> GenFuncCounters = { { 1, 2 } };
	const unsigned int NumEdges = 35;
	const unsigned int NumSubgraphs = 6;
	const bool Instrumented = false;
	const unsigned int QueueSize = 3195660;
	const GPU::Technique Technique = GPU::Technique::MEGAKERNEL;

	struct P0 : public Proc<Box, Replicate<Scale<Scalar<-1000, 3>, Scalar<-4000, 3>, Scalar<-1000, 3>, FSCall<1, 0>>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<2500, 3>, Multi >, Scalar<0>, PSCall<2, 0>>>, 1> {};
	struct P1 : public Proc<Box, Generate<false, 1, Scalar<0>>, 1> {};
	struct P2 : public Proc<Box, IfSizeLess<AxisParam<X>, Scalar<3000, 3>, FSCall<3, 0>, Replicate<Translate<Exp<ShapeSize<Y>, Scalar<-162, 3>, Multi >, Exp<ShapeSize<Y>, Scalar<-27, 3>, Multi >, Scalar<0>, Rotate<Scalar<0>, Scalar<94740, 3>, Scalar<18950, 3>, Scale<Scalar<-850, 3>, Scalar<-1000, 3>, Scalar<-850, 3>, PSCall<4, 0>>>>, Translate<Exp<ShapeSize<Y>, Scalar<162, 3>, Multi >, Exp<ShapeSize<Y>, Scalar<-27, 3>, Multi >, Scalar<0>, Rotate<Scalar<0>, Scalar<132630, 3>, Scalar<-18950, 3>, Scale<Scalar<-850, 3>, Scalar<-1000, 3>, Scalar<-850, 3>, PSCall<4, 1>>>>>>, 1> {};
	struct P3 : public Proc<Box, Replicate<Rotate<Scalar<0>, Scalar<90000, 3>, Scalar<0>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<500, 3>, Multi >, Scalar<0>, Rotate<Scalar<0>, Scalar<0>, Scalar<45000, 3>, Translate<Scalar<-11313, 3>, Scalar<4686, 3>, Scalar<0>, Scale<Scalar<32000, 3>, Scalar<32000, 3>, Scalar<32000, 3>, FSCall<5, 0>>>>>>, Rotate<Scalar<0>, Scalar<135000, 3>, Scalar<0>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<500, 3>, Multi >, Scalar<0>, Rotate<Scalar<0>, Scalar<0>, Scalar<45000, 3>, Translate<Scalar<-11313, 3>, Scalar<4686, 3>, Scalar<0>, Scale<Scalar<32000, 3>, Scalar<32000, 3>, Scalar<32000, 3>, FSCall<5, 0>>>>>>, Rotate<Scalar<0>, Scalar<315000, 3>, Scalar<0>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<500, 3>, Multi >, Scalar<0>, Rotate<Scalar<0>, Scalar<0>, Scalar<45000, 3>, Translate<Scalar<-11313, 3>, Scalar<4686, 3>, Scalar<0>, Scale<Scalar<32000, 3>, Scalar<32000, 3>, Scalar<32000, 3>, FSCall<5, 0>>>>>>>, 1> {};
	struct P4 : public Proc<Box, Replicate<FSCall<1, 0>, Translate<Scalar<0>, ShapeSize<Y>, Scalar<0>, PSCall<2, 0>>>, 1> {};
	struct P5 : public Proc<Box, Generate<false, 1, Scalar<1000, 3>>, 1> {};
	struct ProcedureList : T::List<
		P0,
		P1,
		P2,
		P3,
		P4,
		P5,
	> {};
	DispatchTable dispatchTable = {
		/* entries[0]= */{ /* procIdx= */0, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 2, 0 } } },
		/* entries[1]= */{ /* procIdx= */1, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[2]= */{ /* procIdx= */2, /* parameters[0]= */{}, /* successors[2]= */{ /* 0 */{ 5, 0 }, /* 1 */{ 5, 0 } } },
		/* entries[3]= */{ /* procIdx= */3, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[4]= */{ /* procIdx= */5, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[5]= */{ /* procIdx= */4, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 2, 0 } } },
	};


	template <unsigned int AxiomIdT>
	struct AxiomTraits : Grid::DefaultAxiomTraits < Box, -1 >
	{
		__host__ __device__ __inline__ static math::float3 getSize(const Box& shape)
		{
			return math::float3(16, 16, 16);
		}

		__host__ __device__ __inline__ static math::float3 getPosition(int row, int column, const Box& shape)
		{
			return math::float3(column * 64.0f - 8.0f, 0.0f, row * 64.0f - 8.0f);
		}

		__host__ __device__ __inline__ static int getEntryIndex()
		{
			return 0;
		}

	};

	typedef Grid::Dynamic<AxiomTraits, 1> AxiomGenerator;

	struct Controller : SceneUtils::GriddedSceneController< 16, 16 >
	{
		static const bool IsInitializable = false;
		static const bool HasAttributes = false;

	};

	struct Configurator
	{
		static const bool IsFile = false;
		static std::string value()
		{
			return
				"<?xml version=\"1.0\" encoding=\"UTF-8\"?>                 "
				"<Configuration>                                            "
				"    <ModelRootPath                                         "
				"        path=\"models/3d_tree/\"                           "
				"    />                                                     "
				"                                                           "
				"    <TextureRootPath                                       "
				"        path=\"textures/3d_tree/\"                         "
				"    />                                                     "
				"                                                           "
				"    <!-- TRUNK (x4095) -->                                 "
				"    <InstancedTriangleMesh                                 "
				"        index=\"0\"                                        "
				"        maxNumElements=\"262080\"                          "
				"        type=\"1\"                                         "
				"        modelPath=\"cylinder16.obj\"                       "
				"    >                                                      "
				"        <Material type=\"1\">                              "
				"            <Attribute name=\"tex0\" value=\"trunk.png\" />"
				"            <Attribute name=\"uvScale0\" value=\"(1,4)\" />"
				"        </Material>                                        "
				"    </InstancedTriangleMesh>                               "
				"                                                           "
				"    <!-- LEAF (x6144)-->                                   "
				"    <InstancedTriangleMesh                                 "
				"        index=\"1\"                                        "
				"        maxNumElements=\"393216\"                          "
				"        type=\"1\"                                         "
				"        modelPath=\"leaf.obj\"                             "
				"    >                                                      "
				"        <Material type=\"1\">                              "
				"            <Attribute name=\"tex0\" value=\"leaf.png\" /> "
				"        </Material>                                        "
				"    </InstancedTriangleMesh>                               "
				"                                                           "
				"</Configuration>                                           ";
		}

	};

	std::string testName()
	{
		return std::string("3d_tree_12_2_") + std::to_string(Grid::GlobalVars::getNumElements()) + ((Instrumented) ? "_i" : "");
	}

	std::string sceneName()
	{
		return std::string("3d_tree_12_2") + ((Instrumented) ? "_i" : "");
	}

}