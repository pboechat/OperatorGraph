#pragma once

#include <map>
#include <string>
#include <math.h>
#include <cuda_runtime_api.h>

#include <pga/core/Core.h>
#include <pga/core/Grid.cuh>

using namespace PGA;
using namespace PGA::Operators;
using namespace PGA::Shapes;
using namespace PGA::Parameters;

namespace Scene
{
	const std::map<size_t, size_t> GenFuncCounters = { { 1, 9 } };
	const unsigned int NumEdges = 67;
	const unsigned int NumSubgraphs = 34;
	const bool Instrumented = false;
	const unsigned int QueueSize = 1369568;
	const GPU::Technique Technique = GPU::Technique::KERNELS;

	struct P0 : public Proc<Box, RandomRule<ShapeSeed, T::Pair<DynParam<0>, DCall<0>>, T::Pair<DynParam<1>, DCall<1>>>, 1> {};
	struct P1 : public Proc<Box, Subdivide<AxisParam<Y>, T::Pair<Scalar<-1000, 3>, PSCall<0, 0>>, T::Pair<Scalar<3000, 3>, Scale<Scalar<-800, 3>, Scalar<-1000, 3>, Scalar<-800, 3>, PSCall<0, 1>>>>, 1> {};
	struct P2 : public Proc<Box, Subdivide<AxisParam<Y>, T::Pair<Scalar<-1000, 3>, PSCall<3, 0>>, T::Pair<Scalar<1000, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-4000, 3>, Scale<Scalar<-1066, 3>, Scalar<-1000, 3>, Scalar<-1066, 3>, PSCall<4, 1>>>, T::Pair<Scalar<-1000, 3>, Scale<Scalar<-1066, 3>, Scalar<-1000, 3>, Scalar<-1133, 3>, PSCall<4, 2>>>>>>, 1> {};
	struct P3 : public Proc<Box, ComponentSplit<false, Generate<false, 1, Scalar<4000, 3>>, Discard, RandomRule<ShapeCustomAttribute, T::Pair<Scalar<800, 3>, PSCall<5, 0>>, T::Pair<Scalar<200, 3>, PSCall<6, 1>>>>, 1> {};
	struct P4 : public Proc<Box, Generate<false, 1, DynParam<0>>, 1> {};
	struct P5 : public Proc<Quad, Repeat<false, AxisParam<X>, Scalar<5000, 3>, RepeatModeParam<ADJUST_TO_FILL>, Subdivide<AxisParam<X>, T::Pair<Scalar<1250, 3>, PSCall<7, 0>>, T::Pair<Scalar<-1000, 3>, Repeat<false, AxisParam<Y>, Scalar<3000, 3>, RepeatModeParam<ANCHOR_TO_END>, PSCall<8, 1>>>, T::Pair<Scalar<1250, 3>, PSCall<7, 2>>>>, 1> {};
	struct P6 : public Proc<Quad, Repeat<false, AxisParam<X>, Scalar<2500, 3>, RepeatModeParam<ADJUST_TO_FILL>, Repeat<false, AxisParam<Y>, Scalar<3000, 3>, RepeatModeParam<ANCHOR_TO_END>, RandomRule<ShapeCustomAttribute, T::Pair<Scalar<333, 3>, PSCall<9, 0>>, T::Pair<Scalar<333, 3>, PSCall<10, 1>>, T::Pair<Scalar<334, 3>, PSCall<11, 2>>>>>, 1> {};
	struct P7 : public Proc<Quad, Extrude<AxisParam<Z>, Scalar<333, 3>, Translate<Scalar<0>, Scalar<166, 3>, Scalar<0>, RandomRule<ShapeCustomAttribute, T::Pair<Scalar<333, 3>, PSCall<4, 0>>, T::Pair<Scalar<333, 3>, PSCall<4, 1>>, T::Pair<Scalar<333, 3>, PSCall<4, 2>>>>>, 1> {};
	struct P8 : public Proc<Quad, RandomRule<ShapeCustomAttribute, T::Pair<Scalar<250, 3>, PSCall<9, 0>>, T::Pair<Scalar<250, 3>, PSCall<10, 1>>, T::Pair<Scalar<250, 3>, PSCall<11, 2>>, T::Pair<Scalar<250, 3>, PSCall<11, 3>>>, 1> {};
	struct P9 : public Proc<Quad, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, PSCall<12, 0>>, T::Pair<Scalar<2000, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-1000, 3>, PSCall<12, 1>>, T::Pair<Scalar<1500, 3>, PSCall<12, 2>>, T::Pair<Scalar<-1000, 3>, PSCall<12, 3>>>>, T::Pair<Scalar<-1000, 3>, PSCall<12, 4>>>, 1> {};
	struct P10 : public Proc<Quad, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, PSCall<12, 0>>, T::Pair<Scalar<850, 3>, PSCall<11, 1>>, T::Pair<Scalar<300, 3>, Extrude<AxisParam<Z>, Scalar<300, 3>, Generate<false, 1, Scalar<6000, 3>>>>, T::Pair<Scalar<850, 3>, PSCall<11, 2>>, T::Pair<Scalar<-1000, 3>, PSCall<12, 3>>>, 1> {};
	struct P11 : public Proc<Quad, Subdivide<DynParam<0>, T::Pair<DynParam<1>, DCall<0>>, T::Pair<DynParam<2>, DCall<1>>, T::Pair<DynParam<3>, DCall<2>>, T::Pair<DynParam<4>, DCall<3>>, T::Pair<DynParam<5>, DCall<4>>, T::Pair<DynParam<6>, DCall<5>>, T::Pair<DynParam<7>, DCall<6>>>, 1> {};
	struct P12 : public Proc<Quad, Generate<false, 1, DynParam<0>>, 1> {};
	struct P13 : public Proc<Quad, Scale<Scalar<-1000, 3>, Scalar<-1000, 3>, Scalar<300, 3>, Generate<false, 1, Scalar<7000, 3>>>, 1> {};
	struct ProcedureList : T::List<
		P0,
		P1,
		P2,
		P3,
		P4,
		P5,
		P6,
		P7,
		P8,
		P9,
		P10,
		P11,
		P12,
		P13,
	> {};
	DispatchTable dispatchTable = {
		/* entries[0]= */{ /* procIdx= */0, /* parameters[2]= */{ /* 0 */{ PT_SCALAR, { 0.5 } }, /* 1 */{ PT_SCALAR, { 0.5 } } }, /* successors[2]= */{ /* 0 */{ 1, 0 }, /* 1 */{ 2, 0 } } },
		/* entries[1]= */{ /* procIdx= */1, /* parameters[0]= */{}, /* successors[2]= */{ /* 0 */{ 2, 0 }, /* 1 */{ 2, 0 } } },
		/* entries[2]= */{ /* procIdx= */0, /* parameters[2]= */{ /* 0 */{ PT_SCALAR, { 0.8 } }, /* 1 */{ PT_SCALAR, { 0.2 } } }, /* successors[2]= */{ /* 0 */{ 3, 0 }, /* 1 */{ 4, 0 } } },
		/* entries[3]= */{ /* procIdx= */2, /* parameters[0]= */{}, /* successors[3]= */{ /* 0 */{ 4, 0 }, /* 1 */{ 24, 0 }, /* 2 */{ 24, 0 } } },
		/* entries[4]= */{ /* procIdx= */3, /* parameters[0]= */{}, /* successors[2]= */{ /* 0 */{ 5, 0 }, /* 1 */{ 23, 0 } } },
		/* entries[5]= */{ /* procIdx= */5, /* parameters[0]= */{}, /* successors[3]= */{ /* 0 */{ 6, 0 }, /* 1 */{ 10, 0 }, /* 2 */{ 6, 0 } } },
		/* entries[6]= */{ /* procIdx= */7, /* parameters[0]= */{}, /* successors[3]= */{ /* 0 */{ 7, 0 }, /* 1 */{ 8, 0 }, /* 2 */{ 9, 0 } } },
		/* entries[7]= */{ /* procIdx= */4, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 0 } } }, /* successors[0]= */{} },
		/* entries[8]= */{ /* procIdx= */4, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 1 } } }, /* successors[0]= */{} },
		/* entries[9]= */{ /* procIdx= */4, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 2 } } }, /* successors[0]= */{} },
		/* entries[10]= */{ /* procIdx= */8, /* parameters[0]= */{}, /* successors[4]= */{ /* 0 */{ 11, 0 }, /* 1 */{ 14, 0 }, /* 2 */{ 17, 0 }, /* 3 */{ 20, 0 } } },
		/* entries[11]= */{ /* procIdx= */9, /* parameters[0]= */{}, /* successors[5]= */{ /* 0 */{ 12, 0 }, /* 1 */{ 12, 0 }, /* 2 */{ 13, 0 }, /* 3 */{ 12, 0 }, /* 4 */{ 12, 0 } } },
		/* entries[12]= */{ /* procIdx= */12, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 5 } } }, /* successors[0]= */{} },
		/* entries[13]= */{ /* procIdx= */12, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 8 } } }, /* successors[0]= */{} },
		/* entries[14]= */{ /* procIdx= */10, /* parameters[0]= */{}, /* successors[4]= */{ /* 0 */{ 12, 0 }, /* 1 */{ 15, 0 }, /* 2 */{ 15, 0 }, /* 3 */{ 12, 0 } } },
		/* entries[15]= */{ /* procIdx= */11, /* parameters[8]= */{ /* 0 */{ PT_SCALAR, { 1 } }, /* 1 */{ PT_SCALAR, { -1 } }, /* 2 */{ PT_SCALAR, { 0.7 } }, /* 3 */{ PT_SCALAR, { 0.1 } }, /* 4 */{ PT_SCALAR, { 0.7 } }, /* 5 */{ PT_SCALAR, { -1 } }, /* 6 */{ PT_SCALAR, { 0 } }, /* 7 */{ PT_SCALAR, { 0 } } }, /* successors[7]= */{ /* 0 */{ 12, 0 }, /* 1 */{ 16, 0 }, /* 2 */{ 12, 0 }, /* 3 */{ 16, 0 }, /* 4 */{ 12, 0 }, /* 5 */{ -1, 0 }, /* 6 */{ -1, 0 } } },
		/* entries[16]= */{ /* procIdx= */12, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 8 } } }, /* successors[0]= */{} },
		/* entries[17]= */{ /* procIdx= */11, /* parameters[8]= */{ /* 0 */{ PT_SCALAR, { 0 } }, /* 1 */{ PT_SCALAR, { -1 } }, /* 2 */{ PT_SCALAR, { 0.4 } }, /* 3 */{ PT_SCALAR, { 0.1 } }, /* 4 */{ PT_SCALAR, { 1 } }, /* 5 */{ PT_SCALAR, { 0.1 } }, /* 6 */{ PT_SCALAR, { 0.4 } }, /* 7 */{ PT_SCALAR, { -1 } } }, /* successors[7]= */{ /* 0 */{ 12, 0 }, /* 1 */{ 18, 0 }, /* 2 */{ 12, 0 }, /* 3 */{ 18, 0 }, /* 4 */{ 12, 0 }, /* 5 */{ 18, 0 }, /* 6 */{ 12, 0 } } },
		/* entries[18]= */{ /* procIdx= */11, /* parameters[8]= */{ /* 0 */{ PT_SCALAR, { 1 } }, /* 1 */{ PT_SCALAR, { -1 } }, /* 2 */{ PT_SCALAR, { 1.25 } }, /* 3 */{ PT_SCALAR, { -1 } }, /* 4 */{ PT_SCALAR, { 0 } }, /* 5 */{ PT_SCALAR, { 0 } }, /* 6 */{ PT_SCALAR, { 0 } }, /* 7 */{ PT_SCALAR, { 0 } } }, /* successors[7]= */{ /* 0 */{ 12, 0 }, /* 1 */{ 19, 0 }, /* 2 */{ 12, 0 }, /* 3 */{ -1, 0 }, /* 4 */{ -1, 0 }, /* 5 */{ -1, 0 }, /* 6 */{ -1, 0 } } },
		/* entries[19]= */{ /* procIdx= */12, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 8 } } }, /* successors[0]= */{} },
		/* entries[20]= */{ /* procIdx= */11, /* parameters[8]= */{ /* 0 */{ PT_SCALAR, { 1 } }, /* 1 */{ PT_SCALAR, { 1 } }, /* 2 */{ PT_SCALAR, { -1 } }, /* 3 */{ PT_SCALAR, { 0 } }, /* 4 */{ PT_SCALAR, { 0 } }, /* 5 */{ PT_SCALAR, { 0 } }, /* 6 */{ PT_SCALAR, { 0 } }, /* 7 */{ PT_SCALAR, { 0 } } }, /* successors[7]= */{ /* 0 */{ 21, 0 }, /* 1 */{ 22, 0 }, /* 2 */{ -1, 0 }, /* 3 */{ -1, 0 }, /* 4 */{ -1, 0 }, /* 5 */{ -1, 0 }, /* 6 */{ -1, 0 } } },
		/* entries[21]= */{ /* procIdx= */13, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[22]= */{ /* procIdx= */12, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 8 } } }, /* successors[0]= */{} },
		/* entries[23]= */{ /* procIdx= */6, /* parameters[0]= */{}, /* successors[3]= */{ /* 0 */{ 11, 0 }, /* 1 */{ 14, 0 }, /* 2 */{ 17, 0 } } },
		/* entries[24]= */{ /* procIdx= */4, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 3 } } }, /* successors[0]= */{} },
	};

	template <unsigned int AxiomIdT>
	struct AxiomTraits : Grid::DefaultAxiomTraits < Box, -1 >
	{
		__host__ __device__ __inline__ static float getHeight(const Box& shape)
		{
			return (round(3.0f * shape.getSeed()) + 2.0f) * 15.0f;
		}

		__host__ __device__ __inline__ static math::float3 getSize(const Box& shape)
		{
			auto side = (round(2.0f * shape.getSeed()) + 2.0f) * 5.0f;
			return math::float3(side, getHeight(shape), side);
		}

		__host__ __device__ __inline__ static math::float3 getPosition(int row, int column, const Box& shape)
		{
			return math::float3((column*28.0f) + 14.0f, getHeight(shape)*0.5f, (row*28.0f) + 14.0f);
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
		static const bool IsFile = true;
		static std::string value()
		{
			return "mc_skyscrapers_configuration.xml";
		}

	};

	std::string testName()
	{
		return std::string("mc_skyscrapers_") + std::to_string(Grid::GlobalVars::getNumElements()) + ((Instrumented) ? "_i" : "");
	}

	std::string sceneName()
	{
		return std::string("mc_skyscrapers") + ((Instrumented) ? "_i" : "");
	}

}