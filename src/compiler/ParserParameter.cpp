#include <stdexcept>
#include <algorithm>

#include <boost/variant/get.hpp>

#include <pga/core/Axis.h>
#include <pga/core/RepeatMode.h>

#include "ParserParameter.h"

namespace PGA
{
	namespace Compiler
	{
		namespace Parser
		{
			//////////////////////////////////////////////////////////////////////////
			std::shared_ptr<PGA::Compiler::Parameter> createParameter(Parameter param)
			{
				return createParameter(param, {}, {});
			}

			//////////////////////////////////////////////////////////////////////////
			std::shared_ptr<PGA::Compiler::Parameter> createParameter(Parameter param,
				const std::initializer_list<int>& allowed,
				const std::initializer_list<int>& forbidden)
			{
				auto adtParamType = param.which();
				if (std::find(allowed.begin(), allowed.end(), adtParamType) == allowed.end())
					if (std::find(forbidden.begin(), forbidden.end(), adtParamType) != forbidden.end())
						throw std::runtime_error("PGA::Compiler::Parser::createParameter(): trying to create a parameter of forbidden type");
				switch (adtParamType)
				{
				case 0:
					return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::Axis(static_cast<PGA::Axis>(boost::get<Axis>(param).type)));
				case 1:
					return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::RepeatMode(static_cast<PGA::RepeatMode>(boost::get<RepeatMode>(param).type)));
				case 2:
				{
					auto vec2 = boost::get<Vec2>(param);
					return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::Vec2(vec2.x, vec2.y));
				}
				case 3:
				{
					auto shapeAttribute = boost::get<ShapeAttribute>(param);
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
						return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::ShapeAttr(static_cast<PGA::Compiler::ShapeAttribute>(shapeAttribute.type), shapeAttribute.axis));
					case SHAPE_CUSTOM_ATTR:
						return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::ShapeAttr(static_cast<PGA::Compiler::ShapeAttribute>(shapeAttribute.type), shapeAttribute.axis));
					default:
						throw std::runtime_error("PGA::Compiler::Parser::createParameter(): unknown shape attribute");
					}
					break;
				}
				case 4:
					return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::Rand(boost::get<Rand>(param).min, boost::get<Rand>(param).max));
				case 5:
					return toParameter(boost::get<Expression>(param));
				case 6:
					return std::shared_ptr<PGA::Compiler::Parameter>(new PGA::Compiler::Scalar(boost::get<double>(param)));
				default:
					throw std::runtime_error("PGA::Compiler::Parser::createParameter(): unknown ADT parameter type");
				}
			}

		}

	}

}