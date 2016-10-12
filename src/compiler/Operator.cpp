#include <pga/compiler/EnumUtils.h>
#include <pga/compiler/Operator.h>

#include <stdexcept>

namespace PGA
{
	namespace Compiler
	{
		ShapeType Operator::nextShapeType(OperatorType opType, ShapeType shapeType, size_t succIdx)
		{
			if (opType == COMPSPLIT)
			{
				switch (shapeType)
				{
				case TRIANGLE:
				case QUAD:
				case HEXAGON:
				case HEPTAGON:
				case OCTAGON:
				case DYNAMIC_CONVEX_POLYGON:
					throw std::runtime_error("PGA::Compiler::Operator::nextShapeType(): shape is not supported for operator [shapeType=\"" + PGA::Compiler::EnumUtils::toString(shapeType) + "\", opType=\"" + PGA::Compiler::EnumUtils::toString(opType) + "\"]");
				case PRISM3:
					if (succIdx < 2)
						return TRIANGLE;
					else if (succIdx == 2)
						return QUAD;
					else
						throw std::runtime_error("PGA::Compiler::Operator::nextShapeType(): opType == COMPSPLIT && shapeType == PRISM3 && succIdx >= 3");
				case BOX:
					if (succIdx > 5)
						throw std::runtime_error("PGA::Compiler::Operator::nextShapeType(): opType == COMPSPLIT && shapeType == BOX and succIdx >= 4");
					return QUAD;
				case PRISM5:
					if (succIdx < 2)
						return PENTAGON;
					else if (succIdx < 5)
						return QUAD;
					else
						throw std::runtime_error("PGA::Compiler::Operator::nextShapeType(): opType == COMPSPLIT && shapeType == PRISM5 and succIdx >= 5");
				case PRISM6:
					if (succIdx < 2)
						return HEXAGON;
					else if (succIdx < 6)
						return QUAD;
					else
						throw std::runtime_error("PGA::Compiler::Operator::nextShapeType(): opType == COMPSPLIT && shapeType == PRISM6 and succIdx >= 6");
				case PRISM7:
					if (succIdx < 2)
						return HEPTAGON;
					else if (succIdx < 7)
						return QUAD;
					else
						throw std::runtime_error("PGA::Compiler::Operator::nextShapeType(): opType == COMPSPLIT && shapeType == PRISM7 and succIdx >= 7");
				case PRISM8:
					if (succIdx < 2)
						return OCTAGON;
					else if (succIdx < 8)
						return QUAD;
					else
						throw std::runtime_error("PGA::Compiler::Operator::nextShapeType(): opType == COMPSPLIT && shapeType == PRISM8 and succIdx >= 8");
				case DYNAMIC_CONVEX_RIGHT_PRISM:
					if (succIdx < 2)
						return DYNAMIC_CONVEX_POLYGON;
					else
						return QUAD;
				case DYNAMIC_RIGHT_PRISM:
					if (succIdx < 2)
						return DYNAMIC_POLYGON;
					else
						return QUAD;
				default:
					throw std::runtime_error("PGA::Compiler::Operator::nextShapeType(): shape is not supported for operator [shapeType=\"" + PGA::Compiler::EnumUtils::toString(shapeType) + "\", opType=\"" + PGA::Compiler::EnumUtils::toString(opType) + "\"]");
				}
			}
			else if (opType == EXTRUDE)
			{
				switch (shapeType)
				{
				case TRIANGLE:
					return PRISM3;
				case QUAD:
					return BOX;
				case PENTAGON:
					return PRISM5;
				case HEXAGON:
					return PRISM6;
				case HEPTAGON:
					return PRISM7;
				case OCTAGON:
					return PRISM8;
				case DYNAMIC_CONVEX_POLYGON:
					return DYNAMIC_CONVEX_RIGHT_PRISM;
				case DYNAMIC_POLYGON:
					return DYNAMIC_RIGHT_PRISM;
				case PRISM3:
				case BOX:
				case PRISM5:
				case PRISM6:
				case PRISM7:
				case PRISM8:
				case DYNAMIC_CONVEX_RIGHT_PRISM:
				default:
					throw std::runtime_error("PGA::Compiler::Operator::nextShapeType(): shape is not supported for operator [shapeType=\"" + PGA::Compiler::EnumUtils::toString(shapeType) + "\", opType=\"" + PGA::Compiler::EnumUtils::toString(opType) + "\"]");
				}
			}
			else if (opType == SET_AS_DYNAMIC_CONVEX_POLYGON)
			{
				return DYNAMIC_CONVEX_POLYGON;
			}
			else if (opType == SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM)
			{
				return DYNAMIC_CONVEX_RIGHT_PRISM;
			}
			else if (opType == SET_AS_DYNAMIC_CONCAVE_POLYGON)
			{
				return DYNAMIC_POLYGON;
			}
			else if (opType == SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM)
			{
				return DYNAMIC_RIGHT_PRISM;
			}
			else
				return shapeType;
			// FIXME: warning C4715
			return shapeType;
		}

		bool Operator::requiresEdgeParameter(OperatorType opType)
		{
			return (opType == SUBDIV || opType == STOCHASTIC);
		}

	};

};
