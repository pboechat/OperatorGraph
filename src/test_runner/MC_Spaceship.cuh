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
	const std::map<size_t, size_t> GenFuncCounters = { { 1, 3 } };
	const unsigned int NumEdges = 51;
	const unsigned int NumSubgraphs = 14;
	const bool Instrumented = false;
	const unsigned int QueueSize = 1369568;
	const GPU::Technique Technique = GPU::Technique::MEGAKERNEL;

	struct P0 : public Proc<Box, Replicate<FSCall<1, 0>, PSCall<2, 0>, FSCall<3, 0>>, 1> {};
	struct P1 : public Proc<Box, Replicate<FSCall<4, 0>, Rotate<Scalar<0>, Scalar<0>, Scalar<180000, 3>, FSCall<5, 0>>, FSCall<5, 0>>, 1> {};
	struct P2 : public Proc<Box, Translate<Scalar<0>, Scalar<0>, Scalar<8000, 3>, Replicate<FSCall<6, 0>, Rotate<Scalar<0>, Scalar<0>, Scalar<180000, 3>, FSCall<7, 0>>, FSCall<7, 0>, PSCall<8, 0>>>, 1> {};
	struct P3 : public Proc<Box, Translate<Scalar<0>, Scalar<0>, Scalar<16000, 3>, Replicate<Translate<Scalar<0>, Scalar<0>, Scalar<-500, 3>, Scale<Scalar<3000, 3>, Scalar<3000, 3>, Scalar<7000, 3>, FSCall<10, 0>>>, Translate<Scalar<0>, Scalar<0>, Scalar<4500, 3>, FSCall<11, 0>>>>, 1> {};
	struct P4 : public Proc<Box, Generate<false, 1, Scalar<0>>, 1> {};
	struct P5 : public Proc<Box, Translate<Scalar<7500, 3>, Scalar<0>, Scalar<1500, 3>, Scale<Scalar<10000, 3>, Scalar<1000, 3>, Scalar<10010, 3>, Subdivide<AxisParam<X>, T::Pair<Scalar<1430, 3>, Scale<Scalar<1430, 3>, Scalar<1430, 3>, Scalar<4000, 3>, Translate<Scalar<0>, Scalar<0>, Scalar<-2000, 3>, FSCall<4, 0>>>>, T::Pair<Scalar<1430, 3>, Scale<Scalar<1430, 3>, Scalar<1430, 3>, Scalar<7000, 3>, Translate<Scalar<0>, Scalar<0>, Scalar<-500, 3>, FSCall<6, 0>>>>, T::Pair<Scalar<4290, 3>, Scale<Scalar<4290, 3>, Scalar<1430, 3>, Scalar<3000, 3>, Translate<Scalar<0>, Scalar<0>, Scalar<2000, 3>, FSCall<4, 0>>>>, T::Pair<Scalar<2860, 3>, Scale<Scalar<2860, 3>, Scalar<2860, 3>, Scalar<9000, 3>, Translate<Scalar<0>, Scalar<0>, Scalar<500, 3>, FSCall<6, 0>>>>>>>, 1> {};
	struct P6 : public Proc<Box, Generate<false, 1, Scalar<1000, 3>>, 1> {};
	struct P7 : public Proc<Box, Replicate<Translate<Scalar<2750, 3>, Scalar<0>, Scalar<0>, Scale<Scalar<500, 3>, Scalar<500, 3>, Scalar<4000, 3>, FSCall<6, 0>>>, Translate<Scalar<4250, 3>, Scalar<0>, Scalar<1600, 3>, Scale<Scalar<2500, 3>, Scalar<2500, 3>, Scalar<6400, 3>, FSCall<6, 0>>>>, 1> {};
	struct P8 : public Proc<Box, Translate<Scalar<0>, Scalar<3333, 3>, Scalar<0>, Scale<Scalar<2500, 3>, Scalar<2500, 3>, Scalar<6400, 3>, PSCall<9, 0>>>, 1> {};
	struct P9 : public Proc<Box, IfSizeLess<AxisParam<Y>, Scalar<500, 3>, Discard, Subdivide<AxisParam<Y>, T::Pair<Scalar<500, 3>, FSCall<4, 0>>, T::Pair<Scalar<-1000, 3>, Scale<Scalar<-800, 3>, Scalar<-1000, 3>, Scalar<-800, 3>, PSCall<9, 0>>>>>, 1> {};
	struct P10 : public Proc<Box, Subdivide<AxisParam<X>, T::Pair<Scalar<-1000, 3>, FSCall<12, 0>>, T::Pair<Scalar<-1000, 3>, FSCall<12, 0>>>, 1> {};
	struct P11 : public Proc<Box, Scale<Scalar<3000, 3>, Scalar<3000, 3>, Scalar<3000, 3>, FSCall<13, 0>>, 1> {};
	struct P12 : public Proc<Box, Subdivide<AxisParam<Y>, T::Pair<Scalar<-1000, 3>, FSCall<6, 0>>, T::Pair<Scalar<-1000, 3>, FSCall<6, 0>>>, 1> {};
	struct P13 : public Proc<Box, Generate<false, 1, Scalar<2000, 3>>, 1> {};
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
		/* entries[0]= */{ /* procIdx= */0, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 5, 0 } } },
		/* entries[1]= */{ /* procIdx= */1, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[2]= */{ /* procIdx= */4, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[3]= */{ /* procIdx= */5, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[4]= */{ /* procIdx= */6, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[5]= */{ /* procIdx= */2, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 7, 0 } } },
		/* entries[6]= */{ /* procIdx= */7, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[7]= */{ /* procIdx= */8, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 8, 0 } } },
		/* entries[8]= */{ /* procIdx= */9, /* parameters[0]= */{}, /* successors[1]= */{ /* 0 */{ 8, 0 } } },
		/* entries[9]= */{ /* procIdx= */3, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[10]= */{ /* procIdx= */10, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[11]= */{ /* procIdx= */12, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[12]= */{ /* procIdx= */11, /* parameters[0]= */{}, /* successors[0]= */{} },
		/* entries[13]= */{ /* procIdx= */13, /* parameters[0]= */{}, /* successors[0]= */{} },
	};


	template <unsigned int AxiomIdT>
	struct AxiomTraits : Grid::DefaultAxiomTraits < Box, -1 >
	{
		__host__ __device__ __inline__ static math::float3 getSize(const Box& shape)
		{
			return math::float3(5, 4.5f, 8);
		}

		__host__ __device__ __inline__ static math::float3 getPosition(int row, int column, const Box& shape)
		{
			return math::float3(column * 32.0f - 6.0f, 2.25f, row * 32.0f - 4.0f);
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
			return "mc_spaceship_configuration.xml";
		}

	};

	std::string testName()
	{
		return std::string("mc_spaceship_") + std::to_string(Grid::GlobalVars::getNumElements()) + ((Instrumented) ? "_i" : "");
	}

	std::string sceneName()
	{
		return std::string("mc_spaceship") + ((Instrumented) ? "_i" : "");
	}

}