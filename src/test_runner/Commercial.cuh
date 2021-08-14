#pragma once

#include <cuda_runtime_api.h>
#include <pga/core/Core.h>
#include <pga/core/Grid.cuh>

#include <map>
#include <math.h>
#include <string>

using namespace PGA;
using namespace PGA::Operators;
using namespace PGA::Shapes;
using namespace PGA::Parameters;

namespace Scene
{
	const std::map<size_t, size_t> GenFuncCounters = { { 1, 8 } };
	const unsigned int NumEdges = 152;
	const unsigned int NumSubgraphs = 39;
	const bool Instrumented = false;
	const unsigned int QueueSize = 491640;
	const GPU::Technique Technique = GPU::Technique::KERNELS;

	struct P0 : public Proc<Box, Subdivide<AxisParam<Y>, T::Pair<Scalar<15000, 3>, PSCall<1, 0>>, T::Pair<Scalar<-600, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-500, 3>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<2000, 3>, Div >, Scalar<0>, Scale<Scalar<-1000, 3>, Scalar<-2000, 3>, Scalar<-1000, 3>, PSCall<2, 1>>>>, T::Pair<Scalar<-500, 3>, PSCall<3, 2>>>>, T::Pair<Scalar<-400, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-1000, 3>, PSCall<4, 3>>, T::Pair<Scalar<400, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-500, 3>, PSCall<3, 4>>, T::Pair<Scalar<-500, 3>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<-2000, 3>, Div >, Scalar<0>, Scale<Scalar<-1000, 3>, Scalar<-2000, 3>, Scalar<-1000, 3>, PSCall<5, 5>>>>>>>>>, 1> {};
	struct P1 : public Proc<Box, Subdivide<AxisParam<Y>, T::Pair<Scalar<-500, 3>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<2000, 3>, Div >, Scalar<0>, Scale<Scalar<-1000, 3>, Scalar<-2000, 3>, Scalar<-1000, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<5000, 3>, ComponentSplit<false, Discard, Discard, PSCall<6, 0>>>, T::Pair<Scalar<-1000, 3>, PSCall<2, 1>>>>>>, T::Pair<Scalar<-500, 3>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<-2000, 3>, Div >, Scalar<0>, Scale<Scalar<-1000, 3>, Scalar<-2000, 3>, Scalar<-1000, 3>, PSCall<7, 2>>>>>, 1> {};
	struct P2 : public Proc<Box, ComponentSplit<false, Discard, Discard, DCall<0>>, 1> {};
	struct P3 : public Proc<Box, Translate<Scalar<0>, DynParam<0>, Scalar<0>, Scale<Scalar<-1000, 3>, Scalar<-2000, 3>, Scalar<-1000, 3>, DCall<0>>>, 1> {};
	struct P4 : public Proc<Box, Subdivide<DynParam<0>, T::Pair<DynParam<1>, DCall<0>>, T::Pair<DynParam<2>, DCall<1>>>, 1> {};
	struct P5 : public Proc<Box, RandomRule<ShapeSeed, T::Pair<Scalar<333, 3>, PSCall<19, 0>>, T::Pair<Scalar<333, 3>, PSCall<4, 1>>, T::Pair<Scalar<334, 3>, FSCall<20, 0>>>, 1> {};
	struct P6 : public Proc<Quad, Subdivide<AxisParam<X>, T::Pair<DynParam<0>, PSCall<8, 0>>, T::Pair<DynParam<1>, FSCall<9, 0>>, T::Pair<DynParam<2>, DCall<1>>, T::Pair<DynParam<3>, DCall<2>>, T::Pair<DynParam<4>, DCall<3>>>, 1> {};
	struct P7 : public Proc<Box, Scale<Scalar<-999, 3>, Scalar<-1000, 3>, Scalar<-999, 3>, PSCall<16, 0>>, 1> {};
	struct P8 : public Proc<Quad, Extrude<AxisParam<Z>, Scalar<400, 3>, ComponentSplit<false, PSCall<10, 0>, PSCall<10, 1>, PSCall<10, 2>>>, 1> {};
	struct P9 : public Proc<Quad, IfSizeLess<AxisParam<Y>, Scalar<4000, 3>, Discard, Repeat<false, AxisParam<Y>, Scalar<4000, 3>, RepeatModeParam<ADJUST_TO_FILL>, FSCall<11, 0>>>, 1> {};
	struct P10 : public Proc<Quad, Generate<false, 1, DynParam<0>>, 1> {};
	struct P11 : public Proc<Quad, IfSizeLess<AxisParam<X>, Scalar<4000, 3>, Discard, Repeat<false, AxisParam<X>, Scalar<4000, 3>, RepeatModeParam<ADJUST_TO_FILL>, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<2800, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<3000, 3>, Scale<Scalar<-900, 3>, Scalar<-900, 3>, Scalar<-1000, 3>, Generate<false, 1, Scalar<7000, 3>>>>, T::Pair<Scalar<-1000, 3>, Discard>>>, T::Pair<Scalar<-1000, 3>, Discard>>>>, 1> {};
	struct P12 : public Proc<Quad, Extrude<AxisParam<Z>, Scalar<1000, 3>, ComponentSplit<false, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, PSCall<10, 0>>, T::Pair<Scalar<4000, 3>, PSCall<13, 1>>, T::Pair<Scalar<-1000, 3>, PSCall<10, 2>>>, Discard, PSCall<10, 3>>>, 1> {};
	struct P13 : public Proc<Quad, Subdivide<AxisParam<X>, T::Pair<Scalar<-200, 3>, PSCall<10, 0>>, T::Pair<Scalar<-600, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-20, 3>, PSCall<10, 1>>, T::Pair<Scalar<-780, 3>, PSCall<10, 2>>, T::Pair<Scalar<-200, 3>, PSCall<10, 3>>>>, T::Pair<Scalar<-200, 3>, PSCall<10, 4>>>, 1> {};
	struct P14 : public Proc<Quad, RandomRule<ShapeCustomAttribute, T::Pair<Scalar<333, 3>, PSCall<6, 0>>, T::Pair<Scalar<333, 3>, PSCall<15, 1>>, T::Pair<Scalar<334, 3>, PSCall<6, 2>>>, 1> {};
	struct P15 : public Proc<Quad, IfSizeLess<AxisParam<X>, Scalar<12000, 3>, PSCall<6, 0>, Subdivide<AxisParam<X>, T::Pair<Scalar<1000, 3>, PSCall<8, 1>>, T::Pair<Scalar<-1000, 3>, FSCall<9, 0>>, T::Pair<Scalar<1000, 3>, PSCall<8, 2>>, T::Pair<Scalar<3000, 3>, FSCall<9, 0>>, T::Pair<Scalar<1000, 3>, PSCall<8, 3>>, T::Pair<Scalar<-1000, 3>, FSCall<9, 0>>, T::Pair<Scalar<1000, 3>, PSCall<8, 4>>>>, 1> {};
	struct P16 : public Proc<Box, ComponentSplit<false, Generate<false, 1, Scalar<5000, 3>>, Discard, PSCall<10, 0>>, 1> {};
	struct P17 : public Proc<Box, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<2000, 3>, Div >, Scalar<0>, Scale<Scalar<-1000, 3>, Scalar<-2000, 3>, Scalar<-1000, 3>, ComponentSplit<false, Discard, Discard, PSCall<14, 0>>>>, 1> {};
	struct P18 : public Proc<Quad, Extrude<AxisParam<Z>, Scalar<300, 3>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<-2000, 3>, Div >, Scalar<0>, ComponentSplit<false, PSCall<10, 0>, PSCall<10, 1>, PSCall<10, 2>>>>, 1> {};
	struct P19 : public Proc<Box, Subdivide<AxisParam<X>, T::Pair<Scalar<1000, 3>, PSCall<21, 0>>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<1000, 3>, Subdivide<AxisParam<Z>, T::Pair<Scalar<1000, 3>, PSCall<22, 1>>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<1000, 3>, Discard>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<1000, 3>, PSCall<22, 2>>>>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<1000, 3>, PSCall<21, 3>>>, 1> {};
	struct P20 : public Proc<Box, Translate<Scalar<0>, ShapeSize<Y>, Scalar<0>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<-500, 3>, Multi >, Scalar<0>, Rotate<Scalar<0>, Scalar<0>, Scalar<90000, 3>, Scale<Rand<1000, 3000, 3, 3>, ShapeSize<X>, Scalar<-1000, 3>, Translate<Exp<ShapeSize<X>, Scalar<500, 3>, Multi >, Scalar<0>, Scalar<0>, Generate<false, 1, Scalar<4000, 3>>>>>>>, 1> {};
	struct P21 : public Proc<Box, Subdivide<AxisParam<Z>, T::Pair<Scalar<1000, 3>, PSCall<22, 0>>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<1000, 3>, PSCall<22, 1>>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<1000, 3>, PSCall<22, 2>>>, 1> {};
	struct P22 : public Proc<Box, Scale<Scalar<-1000, 3>, Scalar<-2000, 3>, Scalar<-1000, 3>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<750, 3>, Multi >, Scalar<0>, ComponentSplit<false, PSCall<10, 0>, Discard, PSCall<10, 1>>>>, 1> {};
	struct P23 : public Proc<Box, Subdivide<AxisParam<Z>, T::Pair<DynParam<0>, DCall<0>>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<DynParam<1>, DCall<1>>>, 1> {};
	struct P24 : public Proc<Box, Scale<Scalar<3500, 3>, Scalar<1500, 3>, Scalar<2500, 3>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<2000, 3>, Div >, Scalar<0>, PSCall<25, 0>>>, 1> {};
	struct P25 : public Proc<Box, ComponentSplit<false, PSCall<10, 0>, Discard, Generate<false, 1, Scalar<0>>>, 1> {};
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
		P21,
		P22,
		P23,
		P24,
		P25,
	> {};
	DispatchTable dispatchTable = {
		/* entries[0]= */{ /* procIdx= */0, /* parameters[0]= */{}, /* successors[6]= */{ /* 0 */{ 1, 0 }, /* 1 */{ 19, 0 }, /* 2 */{ 20, 0 }, /* 3 */{ 21, 0 }, /* 4 */{ 24, 0 }, /* 5 */{ 27, 0 } } },
		/* entries[1]= */{ /* procIdx= */1, /* parameters[0]= */{}, /* successors[3]= */{ /* 0 */{ 2, 0 }, /* 1 */{ 11, 0 }, /* 2 */{ 17, 0 } } },
		/* entries[2]= */{ /* procIdx= */6, /* parameters[5]= */{ /* 0 */{ PT_SCALAR, { 1 } }, /* 1 */{ PT_SCALAR, { -1 } }, /* 2 */{ PT_SCALAR, { 5 } }, /* 3 */{ PT_SCALAR, { -1 } }, /* 4 */{ PT_SCALAR, { 1 } } }, /* successors[4]= */{ /* 0 */{ 3, 0 }, /* 1 */{ 7, 0 }, /* 2 */{ 5, 0 }, /* 3 */{ 3, 0 } } },
		/* entries[3]= */{ /* procIdx= */8, /* parameters[0]= */{}, /* successors[3]= */{ /* 0 */{ 4, 0 }, /* 1 */{ 4, 0 }, /* 2 */{ 4, 0 } } },
		/* entries[4]= */{ /* procIdx= */10, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 2 } } }, /* successors[0]= */{} },
		/* entries[5]= */{ /* procIdx= */9, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[6]= */{ /* procIdx= */11, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[7]= */{ /* procIdx= */12, /* parameters[0]= */{}, /* successors[4]= */{ /* 0 */{ 8, 0 }, /* 1 */{ 9, 0 }, /* 2 */{ 8, 0 }, /* 3 */{ 8, 0 } } },
		/* entries[8]= */{ /* procIdx= */10, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 6 } } }, /* successors[0]= */{} },
		/* entries[9]= */{ /* procIdx= */13, /* parameters[0]= */{}, /* successors[5]= */{ /* 0 */{ 8, 0 }, /* 1 */{ 8, 0 }, /* 2 */{ 10, 0 }, /* 3 */{ 8, 0 }, /* 4 */{ 8, 0 } } },
		/* entries[10]= */{ /* procIdx= */10, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 3 } } }, /* successors[0]= */{} },
		/* entries[11]= */{ /* procIdx= */2, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 12, 0 } } },
		/* entries[12]= */{ /* procIdx= */14, /* parameters[0]= */{}, /* successors[3]= */{ /* 0 */{ 13, 0 }, /* 1 */{ 14, 0 }, /* 2 */{ 16, 0 } } },
		/* entries[13]= */{ /* procIdx= */6, /* parameters[5]= */{ /* 0 */{ PT_SCALAR, { 1 } }, /* 1 */{ PT_SCALAR, { -1 } }, /* 2 */{ PT_SCALAR, { 1 } }, /* 3 */{ PT_SCALAR, { 0 } }, /* 4 */{ PT_SCALAR, { 0 } } }, /* successors[4]= */{ /* 0 */{ 3, 0 }, /* 1 */{ 3, 0 }, /* 2 */{ -1, 0 }, /* 3 */{ -1, 0 } } },
		/* entries[14]= */{ /* procIdx= */15, /* parameters[0]= */{}, /* successors[5]= */{ /* 0 */{ 15, 0 }, /* 1 */{ 3, 0 }, /* 2 */{ 3, 0 }, /* 3 */{ 3, 0 }, /* 4 */{ 3, 0 } } },
		/* entries[15]= */{ /* procIdx= */6, /* parameters[5]= */{ /* 0 */{ PT_SCALAR, { 1 } }, /* 1 */{ PT_SCALAR, { -1 } }, /* 2 */{ PT_SCALAR, { 1 } }, /* 3 */{ PT_SCALAR, { 0 } }, /* 4 */{ PT_SCALAR, { 0 } } }, /* successors[4]= */{ /* 0 */{ 3, 0 }, /* 1 */{ 3, 0 }, /* 2 */{ -1, 0 }, /* 3 */{ -1, 0 } } },
		/* entries[16]= */{ /* procIdx= */6, /* parameters[5]= */{ /* 0 */{ PT_SCALAR, { 1 } }, /* 1 */{ PT_SCALAR, { -1 } }, /* 2 */{ PT_SCALAR, { 1 } }, /* 3 */{ PT_SCALAR, { -1 } }, /* 4 */{ PT_SCALAR, { 1 } } }, /* successors[4]= */{ /* 0 */{ 3, 0 }, /* 1 */{ 3, 0 }, /* 2 */{ 5, 0 }, /* 3 */{ 3, 0 } } },
		/* entries[17]= */{ /* procIdx= */7, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 18, 0 } } },
		/* entries[18]= */{ /* procIdx= */16, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 8, 0 } } },
		/* entries[19]= */{ /* procIdx= */2, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 12, 0 } } },
		/* entries[20]= */{ /* procIdx= */3, /* parameters[1]= */{ /* 0 */{ PT_EXP, { 3, -4.18096e-029, 4.65661e-010, -8.58993e+009 } } }, /* successors[1]= */{ /* 0 */{ 17, 0 } } },
		/* entries[21]= */{ /* procIdx= */4, /* parameters[3]= */{ /* 0 */{ PT_SCALAR, { 1 } }, /* 1 */{ PT_SCALAR, { -0.5 } }, /* 2 */{ PT_SCALAR, { -0.5 } } }, /* successors[2]= */{ /* 0 */{ 22, 0 }, /* 1 */{ 23, 0 } } },
		/* entries[22]= */{ /* procIdx= */17, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 12, 0 } } },
		/* entries[23]= */{ /* procIdx= */3, /* parameters[1]= */{ /* 0 */{ PT_EXP, { 3, -4.18096e-029, 4.65661e-010, -8.58993e+009 } } }, /* successors[1]= */{ /* 0 */{ 17, 0 } } },
		/* entries[24]= */{ /* procIdx= */3, /* parameters[1]= */{ /* 0 */{ PT_EXP, { 3, -4.18096e-029, 2.52435e-029, -8.58993e+009 } } }, /* successors[1]= */{ /* 0 */{ 25, 0 } } },
		/* entries[25]= */{ /* procIdx= */2, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 26, 0 } } },
		/* entries[26]= */{ /* procIdx= */18, /* parameters[0]= */{}, /* successors[3]= */{ /* 0 */{ 8, 0 }, /* 1 */{ 8, 0 }, /* 2 */{ 8, 0 } } },
		/* entries[27]= */{ /* procIdx= */5, /* parameters[0]= */{}, /* successors[2]= */{ /* 0 */{ 28, 0 }, /* 1 */{ 31, 0 } } },
		/* entries[28]= */{ /* procIdx= */19, /* parameters[0]= */{}, /* successors[4]= */{ /* 0 */{ 29, 0 }, /* 1 */{ 30, 0 }, /* 2 */{ 30, 0 }, /* 3 */{ 29, 0 } } },
		/* entries[29]= */{ /* procIdx= */21, /* parameters[0]= */{}, /* successors[3]= */{ /* 0 */{ 30, 0 }, /* 1 */{ 30, 0 }, /* 2 */{ 30, 0 } } },
		/* entries[30]= */{ /* procIdx= */22, /* parameters[0]= */{}, /* successors[2]= */{ /* 0 */{ 8, 0 }, /* 1 */{ 8, 0 } } },
		/* entries[31]= */{ /* procIdx= */4, /* parameters[3]= */{ /* 0 */{ PT_SCALAR, { 0 } }, /* 1 */{ PT_SCALAR, { -1 } }, /* 2 */{ PT_SCALAR, { -1 } } }, /* successors[2]= */{ /* 0 */{ 32, 0 }, /* 1 */{ 36, 0 } } },
		/* entries[32]= */{ /* procIdx= */23, /* parameters[2]= */{ /* 0 */{ PT_SCALAR, { -1 } }, /* 1 */{ PT_SCALAR, { 0 } } }, /* successors[2]= */{ /* 0 */{ 33, 0 }, /* 1 */{ -1, 0 } } },
		/* entries[33]= */{ /* procIdx= */24, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 34, 0 } } },
		/* entries[34]= */{ /* procIdx= */25, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 35, 0 } } },
		/* entries[35]= */{ /* procIdx= */10, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 1 } } }, /* successors[0]= */{} },
		/* entries[36]= */{ /* procIdx= */23, /* parameters[2]= */{ /* 0 */{ PT_SCALAR, { 0 } }, /* 1 */{ PT_SCALAR, { -1 } } }, /* successors[2]= */{ /* 0 */{ -1, 0 }, /* 1 */{ 33, 0 } } },
		/* entries[37]= */{ /* procIdx= */20, /* parameters[0]= */{}, /* successors[0]= */{} },
	};

	template <unsigned int AxiomIdT>
	struct AxiomTraits : Grid::DefaultAxiomTraits < Box, -1 >
	{
		__host__ __device__ __inline__ static math::float3 getSize(const Box& shape)
		{
			return math::float3(30, 45, 30);
		}

		__host__ __device__ __inline__ static math::float3 getPosition(int row, int column, const Box& shape)
		{
			return math::float3(column * 40.0f - 15.0f, 22.5f, row * 40.0f - 15.0f);
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
			return "commercial_configuration.xml";
		}

	};

	std::string testName()
	{
		return std::string("commercial_") + std::to_string(Grid::GlobalVars::getNumElements()) + ((Instrumented) ? "_i" : "");
	}

	std::string sceneName()
	{
		return std::string("commercial") + ((Instrumented) ? "_i" : "");
	}

}