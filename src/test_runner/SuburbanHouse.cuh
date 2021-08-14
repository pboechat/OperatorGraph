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
	const std::map<size_t, size_t> GenFuncCounters = { { 1, 13 } };
	const unsigned int NumEdges = 133;
	const unsigned int NumSubgraphs = 27;
	const bool Instrumented = false;
	const unsigned int QueueSize = 684784;
	const GPU::Technique Technique = GPU::Technique::KERNELS;

	struct P0 : public Proc<Box, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, Translate<Scalar<0>, Scalar<0>, Exp<ShapeSize<Z>, Scalar<-300, 3>, Multi >, Scale<Scalar<-1000, 3>, Scalar<-1000, 3>, Scalar<-400, 3>, ComponentSplit<false, Extrude<AxisParam<Z>, Scalar<3000, 3>, Rotate<Scalar<90000, 3>, Scalar<0>, Scalar<0>, PSCall<1, 0>>>, Discard, PSCall<2, 1>>>>>, T::Pair<Scalar<-1000, 3>, Translate<Exp<ShapeSize<X>, Scalar<-500, 3>, Multi >, Scalar<0>, Exp<ShapeSize<Z>, Scalar<200, 3>, Multi >, Scale<Scalar<-2000, 3>, Scalar<-1000, 3>, Scalar<-600, 3>, PSCall<3, 2>>>>, T::Pair<Scalar<-1000, 3>, Translate<Exp<ShapeSize<X>, Scalar<-200, 3>, Multi >, Exp<ShapeSize<Y>, Scalar<-250, 3>, Multi >, Exp<ShapeSize<Z>, Scalar<100, 3>, Multi >, Scale<Scalar<-800, 3>, Scalar<-500, 3>, Scalar<-650, 3>, PSCall<4, 3>>>>>, 1> {};
	struct P1 : public Proc<Box, SwapSize<AxisParam<X>, AxisParam<Z>, AxisParam<Y>, PSCall<5, 0>>, 1> {};
	struct P2 : public Proc<Quad, Subdivide<AxisParam<Y>, T::Pair<Scalar<6000, 3>, If<Exp<ShapeNormal<Z>, Scalar<-1000, 3>, Eq >, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, PSCall<10, 0>>, T::Pair<Scalar<2000, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<1000, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-1000, 3>, PSCall<11, 1>>, T::Pair<Scalar<-1000, 3>, PSCall<11, 2>>>>, T::Pair<Scalar<3000, 3>, Generate<false, 1, Scalar<0>>>, T::Pair<Scalar<1200, 3>, Extrude<AxisParam<Z>, Scalar<750, 3>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<500, 3>, Multi >, Scalar<0>, Rotate<Scalar<0>, Scalar<0>, Scalar<180000, 3>, SetAsDynamicConvexRightPrism<ComponentSplit<false, Rotate<Scalar<0>, Scalar<180000, 3>, Scalar<0>, Generate<false, 1, Scalar<8000, 3>>>, Generate<false, 1, Scalar<3000, 3>>, If<Exp<ShapeNormal<Y>, Scalar<-1000, 3>, Eq >, FSCall<12, 0>, PSCall<7, 3>>>, Vec2<0, 500>, Vec2<-499, -499>, Vec2<499, -499>, >>>>>, T::Pair<Scalar<-1000, 3>, PSCall<8, 4>>>>, T::Pair<Scalar<-1000, 3>, PSCall<10, 5>>>, PSCall<10, 6>>>, T::Pair<Scalar<-1000, 3>, Repeat<false, AxisParam<Y>, Scalar<4000, 3>, RepeatModeParam<ADJUST_TO_FILL>, PSCall<13, 7>>>>, 1> {};
	struct P3 : public Proc<Box, ComponentSplit<false, Extrude<AxisParam<Z>, Scalar<3000, 3>, Rotate<Scalar<90000, 3>, Scalar<90000, 3>, Scalar<0>, PSCall<18, 0>>>, Discard, Subdivide<AxisParam<Y>, T::Pair<Scalar<6000, 3>, PSCall<16, 1>>, T::Pair<Scalar<-1000, 3>, Repeat<false, AxisParam<Y>, Scalar<4000, 3>, RepeatModeParam<ADJUST_TO_FILL>, PSCall<13, 2>>>>>, 1> {};
	struct P4 : public Proc<Box, ComponentSplit<false, Extrude<AxisParam<Z>, Scalar<3000, 3>, Rotate<Scalar<90000, 3>, Scalar<0>, Scalar<0>, SwapSize<AxisParam<X>, AxisParam<Z>, AxisParam<Y>, PSCall<19, 0>>>>, Discard, If<Exp<ShapeNormal<Z>, Scalar<-1000, 3>, Eq >, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, PSCall<8, 1>>, T::Pair<Scalar<4500, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<3000, 3>, Subdivide<AxisParam<X>, T::Pair<Scalar<300, 3>, PSCall<14, 2>>, T::Pair<Scalar<-1000, 3>, PSCall<20, 3>>, T::Pair<Scalar<300, 3>, PSCall<14, 4>>>>, T::Pair<Scalar<-1000, 3>, PSCall<8, 5>>>>, T::Pair<Scalar<-1000, 3>, PSCall<8, 6>>>, PSCall<8, 7>>>, 1> {};
	struct P5 : public Proc<Box, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<350, 3>, Multi >, Exp<ShapeSize<Z>, Scalar<666, 3>, Multi >, Scale<Scalar<-1000, 3>, Scalar<-1700, 3>, Scalar<-1000, 3>, SetAsDynamicConvexRightPrism<ComponentSplit<false, FSCall<6, 0>, FSCall<6, 0>, If<Exp<ShapeNormal<Y>, Scalar<-1000, 3>, Eq >, Discard, Subdivide<AxisParam<Y>, T::Pair<Scalar<-600, 3>, PSCall<7, 0>>, T::Pair<Scalar<-400, 3>, PSCall<8, 1>>>>>, Vec2<0, 500>, Vec2<-499, -499>, Vec2<499, -499>, >>>, 1> {};
	struct P6 : public Proc<DynamicPolygon<PGA::Constants::MaxNumSides, true>, Generate<false, 1, Scalar<6000, 3>>, 1> {};
	struct P7 : public Proc<Quad, If<DynParam<0>, PSCall<9, 0>, PSCall<9, 1>>, 1> {};
	struct P8 : public Proc<Quad, Generate<false, 1, DynParam<0>>, 1> {};
	struct P9 : public Proc<Quad, Translate<DynParam<0>, Scalar<0>, Scalar<0>, Scale<Exp<ShapeSize<X>, Scalar<300, 3>, Add >, Scalar<-1000, 3>, Scalar<-1000, 3>, PSCall<8, 0>>>, 1> {};
	struct P10 : public Proc<Quad, Repeat<false, AxisParam<X>, Scalar<3000, 3>, RepeatModeParam<ADJUST_TO_FILL>, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, PSCall<8, 0>>, T::Pair<Scalar<1400, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<1500, 3>, PSCall<8, 1>>, T::Pair<Scalar<300, 3>, PSCall<14, 2>>, T::Pair<Scalar<2500, 3>, PSCall<8, 3>>, T::Pair<Scalar<-1000, 3>, PSCall<8, 4>>>>, T::Pair<Scalar<-1000, 3>, PSCall<8, 5>>>>, 1> {};
	struct P11 : public Proc<Quad, Extrude<AxisParam<Z>, DynParam<0>, FSCall<15, 0>>, 1> {};
	struct P12 : public Proc<Quad, Subdivide<AxisParam<X>, T::Pair<Scalar<200, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<200, 3>, Extrude<AxisParam<Z>, Scalar<3000, 3>, Translate<Scalar<0>, Exp<Scalar<3000, 3>, Scalar<500, 3>, Multi >, Scalar<0>, Generate<false, 1, Scalar<12000, 3>>>>>, T::Pair<Scalar<-1000, 3>, Discard>>>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<200, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<200, 3>, Extrude<AxisParam<Z>, Scalar<3000, 3>, Translate<Scalar<0>, Exp<Scalar<3000, 3>, Scalar<500, 3>, Multi >, Scalar<0>, Generate<false, 1, Scalar<12000, 3>>>>>, T::Pair<Scalar<-1000, 3>, Discard>>>>, 1> {};
	struct P13 : public Proc<Quad, Subdivide<AxisParam<Y>, T::Pair<Scalar<300, 3>, Extrude<AxisParam<Z>, Scalar<100, 3>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<500, 3>, Multi >, Scalar<0>, Generate<false, 1, Scalar<1000, 3>>>>>, T::Pair<Scalar<-1000, 3>, PSCall<16, 0>>>, 1> {};
	struct P14 : public Proc<Quad, Extrude<AxisParam<Z>, DynParam<0>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<500, 3>, Multi >, Scalar<0>, Generate<false, 1, DynParam<1>>>>, 1> {};
	struct P15 : public Proc<Box, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<500, 3>, Multi >, Scalar<0>, Generate<false, 1, Scalar<7000, 3>>>, 1> {};
	struct P16 : public Proc<Quad, Repeat<false, AxisParam<X>, Scalar<3000, 3>, RepeatModeParam<ADJUST_TO_FILL>, PSCall<17, 0>>, 1> {};
	struct P17 : public Proc<Quad, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, PSCall<8, 0>>, T::Pair<Scalar<1400, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<750, 3>, PSCall<8, 1>>, T::Pair<Scalar<300, 3>, PSCall<14, 2>>, T::Pair<Scalar<2500, 3>, PSCall<8, 3>>, T::Pair<Scalar<-1000, 3>, PSCall<8, 4>>>>, T::Pair<Scalar<-1000, 3>, PSCall<8, 5>>>, 1> {};
	struct P18 : public Proc<Box, SwapSize<AxisParam<Z>, AxisParam<X>, AxisParam<Y>, Translate<Scalar<0>, Scalar<0>, Exp<ShapeSize<Z>, Scalar<666, 3>, Multi >, SetAsDynamicConvexRightPrism<ComponentSplit<false, FSCall<6, 0>, FSCall<6, 0>, If<Exp<ShapeNormal<Y>, Scalar<-1000, 3>, Eq >, Discard, PSCall<7, 0>>>, Vec2<0, 500>, Vec2<-499, -499>, Vec2<499, -499>, >>>, 1> {};
	struct P19 : public Proc<Box, Translate<Scalar<0>, Scalar<0>, Exp<ShapeSize<Z>, Scalar<-500, 3>, Multi >, SetAsDynamicConvexRightPrism<ComponentSplit<false, FSCall<6, 0>, FSCall<6, 0>, If<Exp<ShapeNormal<Y>, Scalar<-1000, 3>, Eq >, Discard, PSCall<7, 0>>>, Vec2<0, 500>, Vec2<-499, -499>, Vec2<499, -499>, >>, 1> {};
	struct P20 : public Proc<Quad, Subdivide<AxisParam<Y>, T::Pair<Scalar<-1000, 3>, Generate<false, 1, Scalar<2000, 3>>>, T::Pair<Scalar<300, 3>, PSCall<14, 0>>>, 1> {};
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
		P14,
		P15,
		P16,
		P17,
		P18,
		P19,
		P20,
	> {};
	DispatchTable dispatchTable = {
		/* entries[0]= */{ /* procIdx= */0, /* parameters[0]= */{}, /* successors[4]= */{ /* 0 */{ 1, 0 }, /* 1 */{ 8, 0 }, /* 2 */{ 20, 0 }, /* 3 */{ 22, 0 } } },
		/* entries[1]= */{ /* procIdx= */1, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 2, 0 } } },
		/* entries[2]= */{ /* procIdx= */5, /* parameters[0]= */{}, /* successors[2]= */{ /* 0 */{ 4, 0 }, /* 1 */{ 6, 0 } } },
		/* entries[3]= */{ /* procIdx= */6, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[4]= */{ /* procIdx= */7, /* parameters[1]= */{ /* 0 */{ PT_EXP, { 3, -2.52435e-029, 0, -1.18112e+010 } } }, /* successors[2]= */{ /* 0 */{ 5, 0 }, /* 1 */{ 7, 0 } } },
		/* entries[5]= */{ /* procIdx= */9, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 0.15 } } }, /* successors[1]= */{ /* 0 */{ 6, 0 } } },
		/* entries[6]= */{ /* procIdx= */8, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 5 } } }, /* successors[0]= */{} },
		/* entries[7]= */{ /* procIdx= */9, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { -0.15 } } }, /* successors[1]= */{ /* 0 */{ 6, 0 } } },
		/* entries[8]= */{ /* procIdx= */2, /* parameters[0]= */{}, /* successors[8]= */{ /* 0 */{ 9, 0 }, /* 1 */{ 13, 0 }, /* 2 */{ 15, 0 }, /* 3 */{ 4, 0 }, /* 4 */{ 10, 0 }, /* 5 */{ 9, 0 }, /* 6 */{ 9, 0 }, /* 7 */{ 17, 0 } } },
		/* entries[9]= */{ /* procIdx= */10, /* parameters[0]= */{}, /* successors[6]= */{ /* 0 */{ 10, 0 }, /* 1 */{ 10, 0 }, /* 2 */{ 11, 0 }, /* 3 */{ 12, 0 }, /* 4 */{ 10, 0 }, /* 5 */{ 10, 0 } } },
		/* entries[10]= */{ /* procIdx= */8, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 8 } } }, /* successors[0]= */{} },
		/* entries[11]= */{ /* procIdx= */14, /* parameters[2]= */{ /* 0 */{ PT_SCALAR, { 0.2 } }, /* 1 */{ PT_SCALAR, { 10 } } }, /* successors[0]= */{} },
		/* entries[12]= */{ /* procIdx= */8, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 9 } } }, /* successors[0]= */{} },
		/* entries[13]= */{ /* procIdx= */11, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 0.75 } } }, /* successors[0]= */{} },
		/* entries[14]= */{ /* procIdx= */15, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[15]= */{ /* procIdx= */11, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 0.5 } } }, /* successors[0]= */{} },
		/* entries[16]= */{ /* procIdx= */12, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[17]= */{ /* procIdx= */13, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 18, 0 } } },
		/* entries[18]= */{ /* procIdx= */16, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 19, 0 } } },
		/* entries[19]= */{ /* procIdx= */17, /* parameters[0]= */{}, /* successors[6]= */{ /* 0 */{ 10, 0 }, /* 1 */{ 10, 0 }, /* 2 */{ 11, 0 }, /* 3 */{ 12, 0 }, /* 4 */{ 10, 0 }, /* 5 */{ 10, 0 } } },
		/* entries[20]= */{ /* procIdx= */3, /* parameters[0]= */{}, /* successors[3]= */{ /* 0 */{ 21, 0 }, /* 1 */{ 18, 0 }, /* 2 */{ 17, 0 } } },
		/* entries[21]= */{ /* procIdx= */18, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 4, 0 } } },
		/* entries[22]= */{ /* procIdx= */4, /* parameters[0]= */{}, /* successors[8]= */{ /* 0 */{ 23, 0 }, /* 1 */{ 10, 0 }, /* 2 */{ 25, 0 }, /* 3 */{ 26, 0 }, /* 4 */{ 25, 0 }, /* 5 */{ 10, 0 }, /* 6 */{ 10, 0 }, /* 7 */{ 10, 0 } } },
		/* entries[23]= */{ /* procIdx= */19, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 24, 0 } } },
		/* entries[24]= */{ /* procIdx= */7, /* parameters[1]= */{ /* 0 */{ PT_EXP, { 3, -3.9443e-029, 0, -1.2348e+010 } } }, /* successors[2]= */{ /* 0 */{ 5, 0 }, /* 1 */{ 7, 0 } } },
		/* entries[25]= */{ /* procIdx= */14, /* parameters[2]= */{ /* 0 */{ PT_SCALAR, { 0.15 } }, /* 1 */{ PT_SCALAR, { 11 } } }, /* successors[0]= */{} },
		/* entries[26]= */{ /* procIdx= */20, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 25, 0 } } },
	};

	template <unsigned int AxiomIdT>
	struct AxiomTraits : Grid::DefaultAxiomTraits < Box, -1 >
	{
		__host__ __device__ __inline__ static math::float3 getSize(const Box& shape)
		{
			return math::float3(30, 10, 16);
		}

		__host__ __device__ __inline__ static math::float3 getPosition(int row, int column, const Box& shape)
		{
			return math::float3(column * 36.0f - 15.0f, 5.0f, row * 36.0f - 8.0f);
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
			return "suburban_house_configuration.xml";
		}

	};

	std::string testName()
	{
		return std::string("suburban_house_") + std::to_string(Grid::GlobalVars::getNumElements()) + ((Instrumented) ? "_i" : "");
	}

	std::string sceneName()
	{
		return std::string("suburban_house") + ((Instrumented) ? "_i" : "");
	}

}