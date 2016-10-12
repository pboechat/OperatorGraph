#include <stdexcept>

#include <pga/compiler/ShapeAttribute.h>

#include "ParserShapeAttribute.h"

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			std::shared_ptr<PGA::Compiler::Parameter> toParameter(const ShapeAttribute& shapeAttribute)
			{
				switch (shapeAttribute.type)
				{
				case SHAPE_POS:
					return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::ShapeAttr(static_cast<PGA::Compiler::ShapeAttribute>(shapeAttribute.type), shapeAttribute.axis));
				case SHAPE_SIZE:
					return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::ShapeAttr(static_cast<PGA::Compiler::ShapeAttribute>(shapeAttribute.type), shapeAttribute.axis));
				case SHAPE_ROTATION:
					return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::ShapeAttr(static_cast<PGA::Compiler::ShapeAttribute>(shapeAttribute.type), shapeAttribute.axis, shapeAttribute.component));
				case SHAPE_NORMAL:
					return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::ShapeAttr(static_cast<PGA::Compiler::ShapeAttribute>(shapeAttribute.type), shapeAttribute.axis));
				case SHAPE_SEED:
					return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::ShapeAttr(static_cast<PGA::Compiler::ShapeAttribute>(shapeAttribute.type)));
				case SHAPE_CUSTOM_ATTR:
					return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::ShapeAttr(static_cast<PGA::Compiler::ShapeAttribute>(shapeAttribute.type)));
				default:
					throw std::runtime_error("PGA::Compiler::Parser::toParameter(): unknown shape attribute");
				}
			}

		}

	}

}