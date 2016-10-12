#pragma once

#include <map>
#include <string>
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
	const unsigned int NumEdges = 118;
	const unsigned int NumSubgraphs = 34;
	const bool Instrumented = false;
	const unsigned int QueueSize = 1027176;
	const GPU::Technique Technique = GPU::Technique::KERNELS;

	struct P0 : public Proc<Box, Subdivide<AxisParam<Y>, T::Pair<Scalar<-600, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-500, 3>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<2000, 3>, Div >, Scalar<0>, Scale<Scalar<-1000, 3>, Scalar<-2000, 3>, Scalar<-1000, 3>, PSCall<1, 0>>>>, T::Pair<Scalar<-500, 3>, PSCall<2, 1>>>>, T::Pair<Scalar<-400, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-500, 3>, PSCall<3, 2>>, T::Pair<Scalar<-500, 3>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<-2000, 3>, Div >, Scalar<0>, Scale<Scalar<-1000, 3>, Scalar<-2000, 3>, Scalar<-1000, 3>, PSCall<4, 3>>>>>>, T::Pair<Scalar<2000, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-900, 3>, ComponentSplit<false, Discard, Discard, PSCall<5, 4>>>, T::Pair<Scalar<-100, 3>, PSCall<6, 5>>>>>, 1> {};
	struct P1 : public Proc<Box, Subdivide<AxisParam<Y>, T::Pair<Scalar<5000, 3>, ComponentSplit<false, Discard, Discard, PSCall<7, 0>>>, T::Pair<Scalar<-1000, 3>, ComponentSplit<false, Discard, Discard, PSCall<8, 1>>>>, 1> {};
	struct P2 : public Proc<Box, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<-2000, 3>, Div >, Scalar<0>, Scale<Scalar<-1000, 3>, Scalar<-2000, 3>, Scalar<-1000, 3>, PSCall<4, 0>>>, 1> {};
	struct P3 : public Proc<Box, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<2000, 3>, Div >, Scalar<0>, Scale<Scalar<-1000, 3>, Scalar<-2000, 3>, Scalar<-1000, 3>, ComponentSplit<false, Discard, Discard, PSCall<8, 0>>>>, 1> {};
	struct P4 : public Proc<Box, Scale<Scalar<-999, 3>, Scalar<-1000, 3>, Scalar<-999, 3>, PSCall<12, 0>>, 1> {};
	struct P5 : public Proc<Quad, Generate<false, 1, DynParam<0>>, 1> {};
	struct P6 : public Proc<Box, SetAsDynamicConvexRightPrism<Translate<Scalar<0>, Scalar<2400, 3>, Scalar<0>, Rotate<Scalar<90000, 3>, Scalar<0>, Scalar<0>, Scale<ShapeSize<X>, ShapeSize<Z>, Scalar<5000, 3>, PSCall<13, 0>>>>, Vec2<0, 500>, Vec2<-499, -499>, Vec2<499, -499>, >, 1> {};
	struct P7 : public Proc<Quad, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, IfSizeLess<AxisParam<Y>, Scalar<5000, 3>, Discard, Repeat<false, AxisParam<Y>, Scalar<5000, 3>, RepeatModeParam<ADJUST_TO_FILL>, IfSizeLess<AxisParam<X>, Scalar<5000, 3>, Discard, Repeat<false, AxisParam<X>, Scalar<5000, 3>, RepeatModeParam<ADJUST_TO_FILL>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<3000, 3>, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<3000, 3>, PSCall<5, 0>>, T::Pair<Scalar<-1000, 3>, Discard>>>, T::Pair<Scalar<-1000, 3>, Discard>>>>>>>, T::Pair<Scalar<5000, 3>, Extrude<AxisParam<Z>, Scalar<1000, 3>, ComponentSplit<false, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, PSCall<5, 1>>, T::Pair<Scalar<4000, 3>, Subdivide<AxisParam<X>, T::Pair<Scalar<-200, 3>, PSCall<5, 2>>, T::Pair<Scalar<-600, 3>, PSCall<9, 3>>, T::Pair<Scalar<-200, 3>, PSCall<5, 4>>>>, T::Pair<Scalar<-1000, 3>, PSCall<5, 5>>>, Discard, PSCall<5, 6>>>>, T::Pair<Scalar<-1000, 3>, IfSizeLess<AxisParam<Y>, Scalar<5000, 3>, Discard, Repeat<false, AxisParam<Y>, Scalar<5000, 3>, RepeatModeParam<ADJUST_TO_FILL>, IfSizeLess<AxisParam<X>, Scalar<5000, 3>, Discard, Repeat<false, AxisParam<X>, Scalar<5000, 3>, RepeatModeParam<ADJUST_TO_FILL>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<3000, 3>, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<3000, 3>, PSCall<5, 0>>, T::Pair<Scalar<-1000, 3>, Discard>>>, T::Pair<Scalar<-1000, 3>, Discard>>>>>>>>, 1> {};
	struct P8 : public Proc<Quad, IfSizeLess<AxisParam<Y>, Scalar<8000, 3>, Discard, Repeat<false, AxisParam<Y>, Scalar<6000, 3>, RepeatModeParam<ADJUST_TO_FILL>, Subdivide<AxisParam<Y>, T::Pair<Scalar<1000, 3>, Discard>, T::Pair<Scalar<-1000, 3>, IfSizeLess<AxisParam<X>, Scalar<8000, 3>, Discard, Repeat<false, AxisParam<X>, Scalar<8000, 3>, RepeatModeParam<ADJUST_TO_FILL>, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, Subdivide<AxisParam<X>, T::Pair<Scalar<-100, 3>, Discard>, T::Pair<Scalar<-900, 3>, Subdivide<AxisParam<Y>, T::Pair<Scalar<-1000, 3>, Extrude<AxisParam<Z>, Scalar<1500, 3>, Translate<Scalar<0>, Scalar<750, 3>, Scalar<0>, ComponentSplit<false, Generate<false, 1, Scalar<3000, 3>>, Discard, If<Exp<ShapeNormal<Y>, Scalar<1000, 3>, Eq >, Discard, If<Exp<ShapeNormal<Y>, Scalar<-1000, 3>, Eq >, Generate<false, 1, Scalar<8000, 3>>, Generate<false, 1, Scalar<3000, 3>>>>>>>>, T::Pair<Scalar<-1000, 3>, Translate<Scalar<0>, Exp<ShapeSize<Y>, Scalar<-2000, 3>, Div >, Scalar<0>, Scale<Scalar<-1000, 3>, Exp<ShapeSize<Y>, Scalar<2000, 3>, Multi >, Scalar<-1000, 3>, FSCall<10, 0>>>>>>, T::Pair<Scalar<-100, 3>, Discard>>>, T::Pair<Scalar<-1000, 3>, PSCall<11, 0>>>>>>>>>, 1> {};
	struct P9 : public Proc<Quad, Subdivide<AxisParam<Y>, T::Pair<Scalar<-20, 3>, PSCall<5, 0>>, T::Pair<Scalar<-780, 3>, Generate<false, 1, Scalar<2000, 3>>>, T::Pair<Scalar<-200, 3>, PSCall<5, 1>>>, 1> {};
	struct P10 : public Proc<Quad, Subdivide<AxisParam<X>, T::Pair<Scalar<-100, 3>, Discard>, T::Pair<Scalar<-900, 3>, Generate<false, 1, Scalar<1000, 3>>>, T::Pair<Scalar<-100, 3>, Discard>>, 1> {};
	struct P11 : public Proc<Quad, Subdivide<AxisParam<Y>, T::Pair<Scalar<-2000, 3>, Discard>, T::Pair<Scalar<-500, 3>, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<-3000, 3>, Extrude<AxisParam<Z>, Scalar<300, 3>, Translate<Scalar<0>, Scalar<150, 3>, Scalar<0>, Generate<false, 1, Scalar<0>>>>>, T::Pair<Scalar<-1000, 3>, Discard>>>, T::Pair<Scalar<-2000, 3>, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, Discard>, T::Pair<Scalar<-3000, 3>, PSCall<5, 0>>, T::Pair<Scalar<-1000, 3>, Discard>>>>, 1> {};
	struct P12 : public Proc<Box, ComponentSplit<false, Discard, Discard, PSCall<5, 0>>, 1> {};
	struct P13 : public Proc<DynamicRightPrism<PGA::Constants::MaxNumSides, true>, ComponentSplit<false, Generate<false, 1, Scalar<7000, 3>>, Generate<false, 1, Scalar<7000, 3>>, If<Exp<ShapeNormal<Y>, Scalar<-1000, 3>, Eq >, Discard, If<Exp<ShapeNormal<X>, Scalar<0>, Lt >, Translate<Scalar<500, 3>, Scalar<0>, Scalar<0>, Scale<Exp<ShapeSize<X>, Scalar<1000, 3>, Add >, Scalar<-1000, 3>, Scalar<-1000, 3>, PSCall<5, 0>>>, Translate<Scalar<-500, 3>, Scalar<0>, Scalar<0>, Scale<Exp<ShapeSize<X>, Scalar<1000, 3>, Add >, Scalar<-1000, 3>, Scalar<-1000, 3>, PSCall<5, 1>>>>>>, 1> {};
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
		/* entries[0]= */{ /* procIdx= */0, /* parameters[0]= */{}, /* successors[6]= */{ /* 0 */{ 1, 0 }, /* 1 */{ 9, 0 }, /* 2 */{ 12, 0 }, /* 3 */{ 10, 0 }, /* 4 */{ 4, 0 }, /* 5 */{ 13, 0 } } },
		/* entries[1]= */{ /* procIdx= */1, /* parameters[0]= */{}, /* successors[2]= */{ /* 0 */{ 2, 0 }, /* 1 */{ 6, 0 } } },
		/* entries[2]= */{ /* procIdx= */7, /* parameters[0]= */{}, /* successors[7]= */{ /* 0 */{ 3, 0 }, /* 1 */{ 4, 0 }, /* 2 */{ 4, 0 }, /* 3 */{ 5, 0 }, /* 4 */{ 4, 0 }, /* 5 */{ 4, 0 }, /* 6 */{ 4, 0 } } },
		/* entries[3]= */{ /* procIdx= */5, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 5 } } }, /* successors[0]= */{} },
		/* entries[4]= */{ /* procIdx= */5, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 4 } } }, /* successors[0]= */{} },
		/* entries[5]= */{ /* procIdx= */9, /* parameters[0]= */{}, /* successors[2]= */{ /* 0 */{ 4, 0 }, /* 1 */{ 4, 0 } } },
		/* entries[6]= */{ /* procIdx= */8, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 8, 0 } } },
		/* entries[7]= */{ /* procIdx= */10, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[8]= */{ /* procIdx= */11, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 3, 0 } } },
		/* entries[9]= */{ /* procIdx= */2, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 10, 0 } } },
		/* entries[10]= */{ /* procIdx= */4, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 11, 0 } } },
		/* entries[11]= */{ /* procIdx= */12, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 4, 0 } } },
		/* entries[12]= */{ /* procIdx= */3, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 6, 0 } } },
		/* entries[13]= */{ /* procIdx= */6, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 14, 0 } } },
		/* entries[14]= */{ /* procIdx= */13, /* parameters[0]= */{}, /* successors[2]= */{ /* 0 */{ 15, 0 }, /* 1 */{ 15, 0 } } },
		/* entries[15]= */{ /* procIdx= */5, /* parameters[1]= */{ /* 0 */{ PT_SCALAR, { 6 } } }, /* successors[0]= */{} },
	};

	template <unsigned int AxiomIdT>
	struct AxiomTraits : Grid::DefaultAxiomTraits < Box, -1 >
	{
		__host__ __device__ __inline__ static math::float3 getSize(const Box& shape)
		{
			return math::float3(24, 34, 24);
		}

		__host__ __device__ __inline__ static math::float3 getPosition(int row, int column, const Box& shape)
		{
			return math::float3(column * 32.0f - 12.0f, 17.0f, row * 32.0f - 12.0f);
		}

		__host__ __device__ __inline__ static int getEntryIndex()
		{
			return 0;
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
		static const bool IsFile = true;
		static std::string value()
		{
			return "balcony_configuration.xml";
		}

	};

	std::string testName()
	{
		return std::string("balcony_") + std::to_string(Grid::GlobalVars::getNumElements()) + ((Instrumented) ? "_i" : "");
	}

	std::string sceneName()
	{
		return std::string("balcony") + ((Instrumented) ? "_i" : "");
	}

}