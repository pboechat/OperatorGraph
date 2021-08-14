#define PGA_CORE_EXPORT 0
#define PGA_RENDERING_EXPORT 0

#include "DebugFlags.h"
#include "GlobalVariables.cuh"
#include "PGAFacade.h"

#include <pga/compiler/OperatorType.h>
#include <pga/compiler/ShapeType.h>
#include <pga/core/Core.h>
#include <pga/core/GPUTechnique.h>
#include <pga/rendering/GenerationFunctions.cuh>

#include <stdexcept>
#include <string.h>

// NOTE: DebugFlags.h have to come before Core.h because of partial template specializations!
using namespace PGA;
using namespace PGA::Shapes;
using namespace PGA::Operators;
using namespace PGA::Parameters;
using namespace PGA::Compiler;

ProcedureList procedureList = {
	// QUAD
	{ DISCARD, QUAD, 0 },
	{ EXTRUDE, QUAD, 0 },
	{ GENERATE, QUAD, 0 },
	{ GENERATE, QUAD, 1 },
	{ IF, QUAD, 0 },
	{ IFSIZELESS, QUAD, 0 },
	{ STOCHASTIC, QUAD, 0 },
	{ REPEAT, QUAD, 0 },
	{ REPLICATE, QUAD, 0 },
	{ ROTATE, QUAD, 0 },
	{ SCALE, QUAD, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, QUAD, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, QUAD, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, QUAD, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, QUAD, 0 },
	{ SUBDIV, QUAD, 0 },
	{ SWAPSIZE, QUAD, 0 },
	{ TRANSLATE, QUAD, 0 },
	
	// BOX
	{ COMPSPLIT, BOX, 0 },
	{ DISCARD, BOX, 0 },
	{ GENERATE, BOX, 0 },
	{ GENERATE, BOX, 1 },
	{ IF, BOX, 0 },
	{ IFSIZELESS, BOX, 0 },
	{ STOCHASTIC, BOX, 0 },
	{ REPEAT, BOX, 0 },
	{ REPLICATE, BOX, 0 },
	{ ROTATE, BOX, 0 },
	{ SCALE, BOX, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, BOX, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, BOX, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, BOX, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, BOX, 0 },
	{ SUBDIV, BOX, 0 },
	{ SWAPSIZE, BOX, 0 },
	{ TRANSLATE, BOX, 0 },

#if defined(INTERPRETER_SUPPORT_ALL_SHAPES)
	// DYNAMIC_CONVEX_POLYGON
	{ DISCARD, DYNAMIC_CONVEX_POLYGON, 0 },
	{ EXTRUDE, DYNAMIC_CONVEX_POLYGON, 0 },
	{ GENERATE, DYNAMIC_CONVEX_POLYGON, 0 },
	{ GENERATE, DYNAMIC_CONVEX_POLYGON, 1 },
	{ IF, DYNAMIC_CONVEX_POLYGON, 0 },
	{ IFSIZELESS, DYNAMIC_CONVEX_POLYGON, 0 },
	{ STOCHASTIC, DYNAMIC_CONVEX_POLYGON, 0 },
	{ REPEAT, DYNAMIC_CONVEX_POLYGON, 0 },
	{ REPLICATE, DYNAMIC_CONVEX_POLYGON, 0 },
	{ ROTATE, DYNAMIC_CONVEX_POLYGON, 0 },
	{ SCALE, DYNAMIC_CONVEX_POLYGON, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, DYNAMIC_CONVEX_POLYGON, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, DYNAMIC_CONVEX_POLYGON, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, DYNAMIC_CONVEX_POLYGON, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, DYNAMIC_CONVEX_POLYGON, 0 },
	{ SUBDIV, DYNAMIC_CONVEX_POLYGON, 0 },
	{ SWAPSIZE, DYNAMIC_CONVEX_POLYGON, 0 },
	{ TRANSLATE, DYNAMIC_CONVEX_POLYGON, 0 },

	// DYNAMIC_POLYGON
	{ DISCARD, DYNAMIC_POLYGON, 0 },
	{ EXTRUDE, DYNAMIC_POLYGON, 0 },
	{ GENERATE, DYNAMIC_POLYGON, 0 },
	{ GENERATE, DYNAMIC_POLYGON, 1 },
	{ IF, DYNAMIC_POLYGON, 0 },
	{ IFSIZELESS, DYNAMIC_POLYGON, 0 },
	{ STOCHASTIC, DYNAMIC_POLYGON, 0 },
	{ REPEAT, DYNAMIC_POLYGON, 0 },
	{ REPLICATE, DYNAMIC_POLYGON, 0 },
	{ ROTATE, DYNAMIC_POLYGON, 0 },
	{ SCALE, DYNAMIC_POLYGON, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, DYNAMIC_POLYGON, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, DYNAMIC_POLYGON, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, DYNAMIC_POLYGON, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, DYNAMIC_POLYGON, 0 },
	{ SUBDIV, DYNAMIC_POLYGON, 0 },
	{ SWAPSIZE, DYNAMIC_POLYGON, 0 },
	{ TRANSLATE, DYNAMIC_POLYGON, 0 },

	// DYNAMIC_CONVEX_RIGHT_PRISM
	{ COMPSPLIT, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ DISCARD, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ GENERATE, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ GENERATE, DYNAMIC_CONVEX_RIGHT_PRISM, 1 },
	{ IF, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ IFSIZELESS, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ STOCHASTIC, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ REPEAT, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ REPLICATE, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ ROTATE, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ SCALE, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ SUBDIV, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ SWAPSIZE, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },
	{ TRANSLATE, DYNAMIC_CONVEX_RIGHT_PRISM, 0 },

	// DYNAMIC_RIGHT_PRISM
	{ COMPSPLIT, DYNAMIC_RIGHT_PRISM, 0 },
	{ DISCARD, DYNAMIC_RIGHT_PRISM, 0 },
	{ GENERATE, DYNAMIC_RIGHT_PRISM, 0 },
	{ GENERATE, DYNAMIC_RIGHT_PRISM, 1 },
	{ IF, DYNAMIC_RIGHT_PRISM, 0 },
	{ IFSIZELESS, DYNAMIC_RIGHT_PRISM, 0 },
	{ STOCHASTIC, DYNAMIC_RIGHT_PRISM, 0 },
	{ REPEAT, DYNAMIC_RIGHT_PRISM, 0 },
	{ REPLICATE, DYNAMIC_RIGHT_PRISM, 0 },
	{ ROTATE, DYNAMIC_RIGHT_PRISM, 0 },
	{ SCALE, DYNAMIC_RIGHT_PRISM, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, DYNAMIC_RIGHT_PRISM, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, DYNAMIC_RIGHT_PRISM, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, DYNAMIC_RIGHT_PRISM, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, DYNAMIC_RIGHT_PRISM, 0 },
	{ SUBDIV, DYNAMIC_RIGHT_PRISM, 0 },
	{ SWAPSIZE, DYNAMIC_RIGHT_PRISM, 0 },
	{ TRANSLATE, DYNAMIC_RIGHT_PRISM, 0 },

	// TRIANGLE
	{ DISCARD, TRIANGLE, 0 },
	{ EXTRUDE, TRIANGLE, 0 },
	{ GENERATE, TRIANGLE, 0 },
	{ GENERATE, TRIANGLE, 1 },
	{ IF, TRIANGLE, 0 },
	{ IFSIZELESS, TRIANGLE, 0 },
	{ STOCHASTIC, TRIANGLE, 0 },
	{ REPEAT, TRIANGLE, 0 },
	{ REPLICATE, TRIANGLE, 0 },
	{ ROTATE, TRIANGLE, 0 },
	{ SCALE, TRIANGLE, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, TRIANGLE, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, TRIANGLE, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, TRIANGLE, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, TRIANGLE, 0 },
	{ SUBDIV, TRIANGLE, 0 },
	{ TRANSLATE, TRIANGLE, 0 },

	// PENTAGON
	{ DISCARD, PENTAGON, 0 },
	{ EXTRUDE, PENTAGON, 0 },
	{ GENERATE, PENTAGON, 0 },
	{ GENERATE, PENTAGON, 1 },
	{ IF, PENTAGON, 0 },
	{ IFSIZELESS, PENTAGON, 0 },
	{ STOCHASTIC, PENTAGON, 0 },
	{ REPEAT, PENTAGON, 0 },
	{ REPLICATE, PENTAGON, 0 },
	{ ROTATE, PENTAGON, 0 },
	{ SCALE, PENTAGON, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, PENTAGON, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, PENTAGON, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, PENTAGON, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, PENTAGON, 0 },
	{ SUBDIV, PENTAGON, 0 },
	{ TRANSLATE, PENTAGON, 0 },

	// HEXAGON
	{ DISCARD, HEXAGON, 0 },
	{ EXTRUDE, HEXAGON, 0 },
	{ GENERATE, HEXAGON, 0 },
	{ GENERATE, HEXAGON, 1 },
	{ IF, HEXAGON, 0 },
	{ IFSIZELESS, HEXAGON, 0 },
	{ STOCHASTIC, HEXAGON, 0 },
	{ REPEAT, HEXAGON, 0 },
	{ REPLICATE, HEXAGON, 0 },
	{ ROTATE, HEXAGON, 0 },
	{ SCALE, HEXAGON, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, HEXAGON, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, HEXAGON, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, HEXAGON, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, HEXAGON, 0 },
	{ SUBDIV, HEXAGON, 0 },
	{ TRANSLATE, HEXAGON, 0 },

	// HEPTAGON
	{ DISCARD, HEPTAGON, 0 },
	{ EXTRUDE, HEPTAGON, 0 },
	{ GENERATE, HEPTAGON, 0 },
	{ GENERATE, HEPTAGON, 1 },
	{ IF, HEPTAGON, 0 },
	{ IFSIZELESS, HEPTAGON, 0 },
	{ STOCHASTIC, HEPTAGON, 0 },
	{ REPEAT, HEPTAGON, 0 },
	{ REPLICATE, HEPTAGON, 0 },
	{ ROTATE, HEPTAGON, 0 },
	{ SCALE, HEPTAGON, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, HEPTAGON, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, HEPTAGON, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, HEPTAGON, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, HEPTAGON, 0 },
	{ SUBDIV, HEPTAGON, 0 },
	{ TRANSLATE, HEPTAGON, 0 },

	// OCTAGON
	{ DISCARD, OCTAGON, 0 },
	{ EXTRUDE, OCTAGON, 0 },
	{ GENERATE, OCTAGON, 0 },
	{ GENERATE, OCTAGON, 1 },
	{ IF, OCTAGON, 0 },
	{ IFSIZELESS, OCTAGON, 0 },
	{ STOCHASTIC, OCTAGON, 0 },
	{ REPEAT, OCTAGON, 0 },
	{ REPLICATE, OCTAGON, 0 },
	{ ROTATE, OCTAGON, 0 },
	{ SCALE, OCTAGON, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, OCTAGON, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, OCTAGON, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, OCTAGON, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, OCTAGON, 0 },
	{ SUBDIV, OCTAGON, 0 },
	{ TRANSLATE, OCTAGON, 0 },

	// PRISM3
	{ COMPSPLIT, PRISM3, 0 },
	{ DISCARD, PRISM3, 0 },
	{ GENERATE, PRISM3, 0 },
	{ GENERATE, PRISM3, 1 },
	{ IF, PRISM3, 0 },
	{ IFSIZELESS, PRISM3, 0 },
	{ STOCHASTIC, PRISM3, 0 },
	{ REPEAT, PRISM3, 0 },
	{ REPLICATE, PRISM3, 0 },
	{ ROTATE, PRISM3, 0 },
	{ SCALE, PRISM3, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, PRISM3, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, PRISM3, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, PRISM3, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, PRISM3, 0 },
	{ SUBDIV, PRISM3, 0 },
	{ TRANSLATE, PRISM3, 0 },

	// PRISM5
	{ COMPSPLIT, PRISM5, 0 },
	{ DISCARD, PRISM5, 0 },
	{ GENERATE, PRISM5, 0 },
	{ GENERATE, PRISM5, 1 },
	{ IF, PRISM5, 0 },
	{ IFSIZELESS, PRISM5, 0 },
	{ STOCHASTIC, PRISM5, 0 },
	{ REPEAT, PRISM5, 0 },
	{ REPLICATE, PRISM5, 0 },
	{ ROTATE, PRISM5, 0 },
	{ SCALE, PRISM5, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, PRISM5, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, PRISM5, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, PRISM5, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, PRISM5, 0 },
	{ SUBDIV, PRISM5, 0 },
	{ TRANSLATE, PRISM5, 0 },

	// PRISM6
	{ COMPSPLIT, PRISM6, 0 },
	{ DISCARD, PRISM6, 0 },
	{ GENERATE, PRISM6, 0 },
	{ GENERATE, PRISM6, 1 },
	{ IF, PRISM6, 0 },
	{ IFSIZELESS, PRISM6, 0 },
	{ STOCHASTIC, PRISM6, 0 },
	{ REPEAT, PRISM6, 0 },
	{ REPLICATE, PRISM6, 0 },
	{ ROTATE, PRISM6, 0 },
	{ SCALE, PRISM6, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, PRISM6, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, PRISM6, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, PRISM6, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, PRISM6, 0 },
	{ SUBDIV, PRISM6, 0 },
	{ TRANSLATE, PRISM6, 0 },

	// PRISM7
	{ COMPSPLIT, PRISM7, 0 },
	{ DISCARD, PRISM7, 0 },
	{ GENERATE, PRISM7, 0 },
	{ GENERATE, PRISM7, 1 },
	{ IF, PRISM7, 0 },
	{ IFSIZELESS, PRISM7, 0 },
	{ STOCHASTIC, PRISM7, 0 },
	{ REPEAT, PRISM7, 0 },
	{ REPLICATE, PRISM7, 0 },
	{ ROTATE, PRISM7, 0 },
	{ SCALE, PRISM7, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, PRISM7, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, PRISM7, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, PRISM7, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, PRISM7, 0 },
	{ SUBDIV, PRISM7, 0 },
	{ TRANSLATE, PRISM7, 0 },

	// PRISM8
	{ COMPSPLIT, PRISM8, 0 },
	{ DISCARD, PRISM8, 0 },
	{ GENERATE, PRISM8, 0 },
	{ GENERATE, PRISM8, 1 },
	{ IF, PRISM8, 0 },
	{ IFSIZELESS, PRISM8, 0 },
	{ STOCHASTIC, PRISM8, 0 },
	{ REPEAT, PRISM8, 0 },
	{ REPLICATE, PRISM8, 0 },
	{ ROTATE, PRISM8, 0 },
	{ SCALE, PRISM8, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, PRISM8, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, PRISM8, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, PRISM8, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, PRISM8, 0 },
	{ SUBDIV, PRISM8, 0 },
	{ TRANSLATE, PRISM8, 0 },

	// SPHERE
	{ DISCARD, SPHERE, 0 },
	{ GENERATE, SPHERE, 0 },
	{ GENERATE, SPHERE, 1 },
	{ IF, SPHERE, 0 },
	{ IFSIZELESS, SPHERE, 0 },
	{ STOCHASTIC, SPHERE, 0 },
	{ ROTATE, SPHERE, 0 },
	{ SCALE, SPHERE, 0 },
	{ SET_AS_DYNAMIC_CONVEX_POLYGON, SPHERE, 0 },
	{ SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, SPHERE, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_POLYGON, SPHERE, 0 },
	{ SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, SPHERE, 0 },
	{ TRANSLATE, SPHERE, 0 },
#endif

};

struct ProcList : T::List <
	// QUAD
	Proc<Quad, Discard>,
	Proc<Quad, Extrude<DynParam<0>, DynParam<1>, DCall<0>>>,
	Proc<Quad, Generate<false, 0, DynParam<0>>>,
	Proc<Quad, Generate<false, 1, DynParam<0>>>,
	Proc<Quad, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<Quad, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<Quad, RandomRule<DynParams>>,
	Proc<Quad, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Quad, Replicate<DynParams>>,
	Proc<Quad, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Quad, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Quad, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<Quad, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<Quad, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<Quad, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<Quad, Subdivide<DynParams>>,
	Proc<Quad, SwapSize<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Quad, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// BOX
	Proc<Box, ComponentSplit<false, DCall<0>, DCall<1>, DCall<2>>>,
	Proc<Box, Discard>,
	Proc<Box, Generate<true, 0, DynParam<0>>, 16>,
	Proc<Box, Generate<false, 1, DynParam<0>>>,
	Proc<Box, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<Box, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<Box, RandomRule<DynParams>>,
	Proc<Box, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Box, Replicate<DynParams>>,
	Proc<Box, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Box, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Box, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<Box, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<Box, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<Box, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<Box, Subdivide<DynParams>>,
	Proc<Box, SwapSize<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Box, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

#if defined(INTERPRETER_SUPPORT_ALL_SHAPES)
	// DYNAMIC_CONVEX_POLYGON
	Proc<DCPoly, Discard>,
	Proc<DCPoly, Extrude<DynParam<0>, DynParam<1>, DCall<0>>>,
	Proc<DCPoly, Generate<false, 0, DynParam<0>>>,
	Proc<DCPoly, Generate<false, 1, DynParam<0>>>,
	Proc<DCPoly, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<DCPoly, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<DCPoly, RandomRule<DynParams>>,
	Proc<DCPoly, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DCPoly, Replicate<DynParams>>,
	Proc<DCPoly, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DCPoly, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DCPoly, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<DCPoly, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<DCPoly, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<DCPoly, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<DCPoly, Subdivide<DynParams>>,
	Proc<DCPoly, SwapSize<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DCPoly, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// DYNAMIC_POLYGON
	Proc<DPoly, Discard>,
	Proc<DPoly, Extrude<DynParam<0>, DynParam<1>, DCall<0>>>,
	Proc<DPoly, Generate<false, 0, DynParam<0>>>,
	Proc<DPoly, Generate<false, 1, DynParam<0>>>,
	Proc<DPoly, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<DPoly, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<DPoly, RandomRule<DynParams>>,
	Proc<DPoly, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DPoly, Replicate<DynParams>>,
	Proc<DPoly, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DPoly, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DPoly, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<DPoly, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<DPoly, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<DPoly, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<DPoly, Subdivide<DynParams>>,
	Proc<DPoly, SwapSize<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DPoly, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// DYNAMIC_CONVEX_RIGHT_PRISM
	Proc<DCRPrism, ComponentSplit<false, DCall<0>, DCall<1>, DCall<2>>>,
	Proc<DCRPrism, Discard>,
	Proc<DCRPrism, Generate<false, 0, DynParam<0>>>,
	Proc<DCRPrism, Generate<false, 1, DynParam<0>>>,
	Proc<DCRPrism, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<DCRPrism, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<DCRPrism, RandomRule<DynParams>>,
	Proc<DCRPrism, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DCRPrism, Replicate<DynParams>>,
	Proc<DCRPrism, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DCRPrism, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DCRPrism, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<DCRPrism, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<DCRPrism, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<DCRPrism, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<DCRPrism, Subdivide<DynParams>>,
	Proc<DCRPrism, SwapSize<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DCRPrism, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// DYNAMIC_RIGHT_PRISM
	Proc<DRPrism, ComponentSplit<false, DCall<0>, DCall<1>, DCall<2>>>,
	Proc<DRPrism, Discard>,
	Proc<DRPrism, Generate<false, 0, DynParam<0>>>,
	Proc<DRPrism, Generate<false, 1, DynParam<0>>>,
	Proc<DRPrism, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<DRPrism, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<DRPrism, RandomRule<DynParams>>,
	Proc<DRPrism, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DRPrism, Replicate<DynParams>>,
	Proc<DRPrism, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DRPrism, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DRPrism, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<DRPrism, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<DRPrism, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<DRPrism, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<DRPrism, Subdivide<DynParams>>,
	Proc<DRPrism, SwapSize<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<DRPrism, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// TRIANGLE
	Proc<Triangle, Discard>,
	Proc<Triangle, Extrude<DynParam<0>, DynParam<1>, DCall<0>>>,
	Proc<Triangle, Generate<false, 0, DynParam<0>>>,
	Proc<Triangle, Generate<false, 1, DynParam<0>>>,
	Proc<Triangle, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<Triangle, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<Triangle, RandomRule<DynParams>>,
	Proc<Triangle, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Triangle, Replicate<DynParams>>,
	Proc<Triangle, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Triangle, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Triangle, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<Triangle, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<Triangle, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<Triangle, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<Triangle, Subdivide<DynParams>>,
	Proc<Triangle, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// PENTAGON
	Proc<Pentagon, Discard>,
	Proc<Pentagon, Extrude<DynParam<0>, DynParam<1>, DCall<0>>>,
	Proc<Pentagon, Generate<false, 0, DynParam<0>>>,
	Proc<Pentagon, Generate<false, 1, DynParam<0>>>,
	Proc<Pentagon, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<Pentagon, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<Pentagon, RandomRule<DynParams>>,
	Proc<Pentagon, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Pentagon, Replicate<DynParams>>,
	Proc<Pentagon, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Pentagon, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Pentagon, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<Pentagon, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<Pentagon, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<Pentagon, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<Pentagon, Subdivide<DynParams>>,
	Proc<Pentagon, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// HEXAGON
	Proc<Hexagon, Discard>,
	Proc<Hexagon, Extrude<DynParam<0>, DynParam<1>, DCall<0>>>,
	Proc<Hexagon, Generate<false, 0, DynParam<0>>>,
	Proc<Hexagon, Generate<false, 1, DynParam<0>>>,
	Proc<Hexagon, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<Hexagon, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<Hexagon, RandomRule<DynParams>>,
	Proc<Hexagon, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Hexagon, Replicate<DynParams>>,
	Proc<Hexagon, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Hexagon, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Hexagon, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<Hexagon, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<Hexagon, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<Hexagon, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<Hexagon, Subdivide<DynParams>>,
	Proc<Hexagon, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// HEPTAGON
	Proc<Heptagon, Discard>,
	Proc<Heptagon, Extrude<DynParam<0>, DynParam<1>, DCall<0>>>,
	Proc<Heptagon, Generate<false, 0, DynParam<0>>>,
	Proc<Heptagon, Generate<false, 1, DynParam<0>>>,
	Proc<Heptagon, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<Heptagon, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<Heptagon, RandomRule<DynParams>>,
	Proc<Heptagon, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Heptagon, Replicate<DynParams>>,
	Proc<Heptagon, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Heptagon, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Heptagon, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<Heptagon, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<Heptagon, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<Heptagon, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<Heptagon, Subdivide<DynParams>>,
	Proc<Heptagon, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// OCTAGON
	Proc<Octagon, Discard>,
	Proc<Octagon, Extrude<DynParam<0>, DynParam<1>, DCall<0>>>,
	Proc<Octagon, Generate<false, 0, DynParam<0>>>,
	Proc<Octagon, Generate<false, 1, DynParam<0>>>,
	Proc<Octagon, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<Octagon, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<Octagon, RandomRule<DynParams>>,
	Proc<Octagon, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Octagon, Replicate<DynParams>>,
	Proc<Octagon, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Octagon, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Octagon, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<Octagon, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<Octagon, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<Octagon, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<Octagon, Subdivide<DynParams>>,
	Proc<Octagon, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// PRISM3
	Proc<Prism3, ComponentSplit<false, DCall<0>, DCall<1>, DCall<2>>>,
	Proc<Prism3, Discard>,
	Proc<Prism3, Generate<false, 0, DynParam<0>>>,
	Proc<Prism3, Generate<false, 1, DynParam<0>>>,
	Proc<Prism3, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<Prism3, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<Prism3, RandomRule<DynParams>>,
	Proc<Prism3, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism3, Replicate<DynParams>>,
	Proc<Prism3, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism3, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism3, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<Prism3, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<Prism3, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<Prism3, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<Prism3, Subdivide<DynParams>>,
	Proc<Prism3, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// PRISM5
	Proc<Prism5, ComponentSplit<false, DCall<0>, DCall<1>, DCall<2>>>,
	Proc<Prism5, Discard>,
	Proc<Prism5, Generate<false, 0, DynParam<0>>>,
	Proc<Prism5, Generate<false, 1, DynParam<0>>>,
	Proc<Prism5, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<Prism5, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<Prism5, RandomRule<DynParams>>,
	Proc<Prism5, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism5, Replicate<DynParams>>,
	Proc<Prism5, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism5, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism5, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<Prism5, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<Prism5, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<Prism5, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<Prism5, Subdivide<DynParams>>,
	Proc<Prism5, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// PRISM6
	Proc<Prism6, ComponentSplit<false, DCall<0>, DCall<1>, DCall<2>>>,
	Proc<Prism6, Discard>,
	Proc<Prism6, Generate<false, 0, DynParam<0>>>,
	Proc<Prism6, Generate<false, 1, DynParam<0>>>,
	Proc<Prism6, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<Prism6, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<Prism6, RandomRule<DynParams>>,
	Proc<Prism6, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism6, Replicate<DynParams>>,
	Proc<Prism6, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism6, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism6, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<Prism6, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<Prism6, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<Prism6, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<Prism6, Subdivide<DynParams>>,
	Proc<Prism6, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// PRISM7
	Proc<Prism7, ComponentSplit<false, DCall<0>, DCall<1>, DCall<2>>>,
	Proc<Prism7, Discard>,
	Proc<Prism7, Generate<false, 0, DynParam<0>>>,
	Proc<Prism7, Generate<false, 1, DynParam<0>>>,
	Proc<Prism7, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<Prism7, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<Prism7, RandomRule<DynParams>>,
	Proc<Prism7, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism7, Replicate<DynParams>>,
	Proc<Prism7, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism7, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism7, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<Prism7, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<Prism7, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<Prism7, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<Prism7, Subdivide<DynParams>>,
	Proc<Prism7, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// PRISM8
	Proc<Prism8, ComponentSplit<false, DCall<0>, DCall<1>, DCall<2>>>,
	Proc<Prism8, Discard>,
	Proc<Prism8, Generate<false, 0, DynParam<0>>>,
	Proc<Prism8, Generate<false, 1, DynParam<0>>>,
	Proc<Prism8, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<Prism8, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<Prism8, RandomRule<DynParams>>,
	Proc<Prism8, Repeat<false, DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism8, Replicate<DynParams>>,
	Proc<Prism8, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism8, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Prism8, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<Prism8, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<Prism8, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<Prism8, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<Prism8, Subdivide<DynParams>>,
	Proc<Prism8, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,

	// SPHERE
	Proc<Sphere, Discard>,
	Proc<Sphere, Generate<false, 0, DynParam<0>>>,
	Proc<Sphere, Generate<false, 1, DynParam<0>>>,
	Proc<Sphere, If<DynParam<0>, DCall<0>, DCall<1>>>,
	Proc<Sphere, IfSizeLess<DynParam<0>, DynParam<1>, DCall<0>, DCall<1>>>,
	Proc<Sphere, RandomRule<DynParams>>,
	Proc<Sphere, Rotate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Sphere, Scale<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
	Proc<Sphere, SetAsDynamicConvexPolygon<DynParams>>,
	Proc<Sphere, SetAsDynamicConvexRightPrism<DynParams>>,
	Proc<Sphere, SetAsDynamicConcavePolygon<DynParams>>,
	Proc<Sphere, SetAsDynamicConcaveRightPrism<DynParams>>,
	Proc<Sphere, Translate<DynParam<0>, DynParam<1>, DynParam<2>, DCall<0>>>,
#endif

> {};

struct AxiomGenerator
{
	static unsigned int getNumAxioms()
	{
		return ::Host::numAxioms;
	}

	template <typename SymbolManagerT, typename QueueT>
	__host__ __device__ __inline__ static void generateAxiom(int axiomIndex, QueueT* queue)
	{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 2)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
		if (axiomIndex >= ::Device::numAxioms)
		{
			printf("axiomIndex >= Device::numAxioms (axiomIndex=%d, ::Device::numAxioms=%d)\n", axiomIndex, ::Device::numAxioms);
			asm("trap;");
		}
#else
		if (axiomIndex >= ::Host::numAxioms)
			throw std::runtime_error(("axiomIndex >= Host::numAxioms (axiomIndex=" + std::to_string(axiomIndex) + ", ::Host::numAxioms=" + std::to_string(::Host::numAxioms) + ")").c_str());
#endif
#endif
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
		const auto& axiom = ::Device::axioms[axiomIndex];
#else
		const auto& axiom = ::Host::axioms[axiomIndex];
#endif
		if (axiom.entryIndex == -1) return;
		if (axiom.shapeType == ShapeType::QUAD)
		{
			Quad quad;
			quad.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			quad.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			quad.setPosition(math::float3(0, 0, 0));
			quad.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(quad, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::BOX)
		{
			Box box;
			box.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			box.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			box.setPosition(math::float3(0, 0, 0));
			box.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(box, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
#if defined(INTERPRETER_SUPPORT_ALL_SHAPES)
		else if (axiom.shapeType == ShapeType::DYNAMIC_POLYGON)
		{
			DPoly prism(axiom.vertices, axiom.numVertices);
			prism.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			prism.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			prism.setPosition(math::float3(0, 0, 0));
			prism.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(prism, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::DYNAMIC_RIGHT_PRISM)
		{
			DRPrism shape(axiom.vertices, axiom.numVertices);
			shape.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			shape.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			shape.setPosition(math::float3(0, 0, 0));
			shape.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(shape, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::TRIANGLE)
		{
			Triangle poly;
			poly.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			poly.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			poly.setPosition(math::float3(0, 0, 0));
			poly.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(poly, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::PENTAGON)
		{
			Pentagon poly;
			poly.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			poly.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			poly.setPosition(math::float3(0, 0, 0));
			poly.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(poly, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::HEXAGON)
		{
			Hexagon poly;
			poly.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			poly.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			poly.setPosition(math::float3(0, 0, 0));
			poly.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(poly, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::HEPTAGON)
		{
			Heptagon poly;
			poly.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			poly.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			poly.setPosition(math::float3(0, 0, 0));
			poly.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(poly, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::OCTAGON)
		{
			Octagon poly;
			poly.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			poly.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			poly.setPosition(math::float3(0, 0, 0));
			poly.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(poly, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::PRISM3)
		{
			Prism3 prism;
			prism.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			prism.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			prism.setPosition(math::float3(0, 0, 0));
			prism.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(prism, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::PRISM5)
		{
			Prism5 prism;
			prism.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			prism.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			prism.setPosition(math::float3(0, 0, 0));
			prism.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(prism, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::PRISM6)
		{
			Prism6 prism;
			prism.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			prism.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			prism.setPosition(math::float3(0, 0, 0));
			prism.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(prism, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::PRISM7)
		{
			Prism7 prism;
			prism.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			prism.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			prism.setPosition(math::float3(0, 0, 0));
			prism.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(prism, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::PRISM8)
		{
			Prism8 prism;
			prism.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			prism.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			prism.setPosition(math::float3(0, 0, 0));
			prism.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(prism, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::DYNAMIC_CONVEX_POLYGON)
		{
			DCPoly prism(axiom.vertices, axiom.numVertices);
			prism.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			prism.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			prism.setPosition(math::float3(0, 0, 0));
			prism.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(prism, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::DYNAMIC_CONVEX_RIGHT_PRISM)
		{
			DCRPrism shape(axiom.vertices, axiom.numVertices);
			shape.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			shape.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			shape.setPosition(math::float3(0, 0, 0));
			shape.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(shape, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
		else if (axiom.shapeType == ShapeType::SPHERE)
		{
			Sphere sphere;
			sphere.setSeed(PGA::GlobalVars::getSeed(axiomIndex));
			sphere.setCustomAttribute(PGA::GlobalVars::getSeed(axiomIndex));
			sphere.setPosition(math::float3(0, 0, 0));
			sphere.setSize(math::float3(1, 1, 1));
			SymbolManagerT::dispatchAxiom(sphere, static_cast<unsigned int>(axiom.entryIndex), queue);
		}
#endif
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 2)
		else
		{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))

			printf("unsupported axiom shape type (axiomIndex=%d, axiom.shapeType=%d)\n", axiomIndex, axiom.shapeType);
			asm("trap;");
#else
			throw std::runtime_error(("unsupported axiom shape type (axiomIndex=" + std::to_string(axiomIndex) + ", axiom.shapeType=" + std::to_string(axiom.shapeType) + ")").c_str());
#endif
		}
#endif
	}

};

struct Configuration
{
#if defined(PGA_CPU)
	static const unsigned int MaxDerivationSteps = 100000;
#else
	static const PGA::GPU::Technique Technique = PGA::GPU::Technique::MEGAKERNEL;
	static const unsigned int QueueSize = 1024 * 32;
	static const unsigned int MaxSharedMemory = 0;
#endif

};

typedef PGA::SinglePhaseEvaluator<ProcList, AxiomGenerator, PGA::Rendering::GenFuncFilter, false, 0, 0, Configuration> Evaluator;
typedef std::unique_ptr<Evaluator, PGA::ReleaseCallback> EvaluatorPtr;
EvaluatorPtr g_evaluator;

PGA::Compiler::ProcedureList getProcedureList()
{
	return procedureList;
}

void initializePGA(const std::unique_ptr<PGA::DispatchTable>& dispatchTable)
{
	if (!g_evaluator)
		g_evaluator = EvaluatorPtr(new Evaluator());
	g_evaluator->initialize(dispatchTable->toDispatchTableEntriesPtr().get(), dispatchTable->entries.size());
}

void releasePGA()
{
	g_evaluator.release();
}

void destroyPGA()
{
	g_evaluator = 0;
}

double executePGA()
{
	return g_evaluator->execute();
}