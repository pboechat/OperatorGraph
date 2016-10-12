#pragma once

#include <string>
#include <stdexcept>

#include "OperatorType.h"
#include "ShapeType.h"

namespace PGA
{
	namespace Compiler
	{
		class EnumUtils
		{
		public:
			EnumUtils() = delete;

			static std::string toString(PGA::Compiler::OperatorType opType)
			{
				switch (opType)
				{
				case TRANSLATE:
					return "Translate";
				case ROTATE:
					return "Rotate";
				case SCALE:
					return "Scale";
				case EXTRUDE:
					return "Extrude";
				case COMPSPLIT:
					return "ComponentSplit";
				case SUBDIV:
					return "Subdivide";
				case REPEAT:
					return "Repeat";
				case DISCARD:
					return "Discard";
				case IF:
					return "If";
				case IFCOLLIDES:
					return "IfCollides";
				case IFSIZELESS:
					return "IfSizeLess";
				case GENERATE:
					return "Generate";
				case STOCHASTIC:
					return "RandomRule";
				case SET_AS_DYNAMIC_CONVEX_POLYGON:
					return "SetAsDynamicConvexPolygon";
				case SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM:
					return "SetAsDynamicConvexRightPrism";
				case SET_AS_DYNAMIC_CONCAVE_POLYGON:
					return "SetAsDynamicConcavePolygon";
				case SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM:
					return "SetAsDynamicConcaveRightPrism";
				case COLLIDER:
					return "Collider";
				case SWAPSIZE:
					return "SwapAxis";
				case REPLICATE:
					return "Replicate";
				default:
					throw std::runtime_error("PGA::Compiler::EnumUtils::toString(..): unknown operator type [opType=" + std::to_string(opType) + "]");
				}
			}

			static std::string toString(PGA::Compiler::ShapeType shapeType)
			{
				switch (shapeType)
				{
				case TRIANGLE:
					return "Triangle";
				case QUAD:
					return "Quad";
				case PENTAGON:
					return "Pentagon";
				case HEXAGON:
					return "Hexagon";
				case HEPTAGON:
					return "Heptagon";
				case OCTAGON:
					return "Octagon";
				case PRISM3:
					return "Prism3";
				case BOX:
					return "Box";
				case PRISM5:
					return "Prism5";
				case PRISM6:
					return "prism6";
				case PRISM7:
					return "Prism7";
				case PRISM8:
					return "Prism8";
				case DYNAMIC_CONVEX_POLYGON:
					return "DynamicPolygon<PGA::Constants::MaxNumSides, true>";
				case DYNAMIC_POLYGON:
					return "DynamicPolygon<PGA::Constants::MaxNumSides, false>";
				case DYNAMIC_CONVEX_RIGHT_PRISM:
					return "DynamicRightPrism<PGA::Constants::MaxNumSides, true>";
				case DYNAMIC_RIGHT_PRISM:
					return "DynamicRightPrism<PGA::Constants::MaxNumSides, false>";
				case SPHERE:
					return "Sphere";
				default:
					throw std::runtime_error("PGA::Compiler::EnumUtils::toString(..): unknown shape type [shapeType=" + std::to_string(shapeType) + "]");
				}
			}

		};

	}

}