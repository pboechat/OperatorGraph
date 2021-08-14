#include <math/vector.h>
#include <pga/compiler/Parameters.h>
#include <pga/compiler/ShapeAttribute.h>
#include <pga/core/EnumUtils.h>
#include <pga/core/PackUtils.h>

#include <stdexcept>

namespace PGA
{
	namespace Compiler
	{
		bool Parameter::compareValues(const std::vector<double>& otherValues) const
		{
			if (values.size() != otherValues.size()) return false;
			for (size_t i = 0; i < values.size(); i++)
				if (values[i] != otherValues[i]) return false;
			return true;
		}

		double Parameter::at(size_t i) const
		{
			return values[i];
		}

		size_t Parameter::size() const
		{
			return values.size();
		}

		unsigned int Parameter::getParameterLength() const
		{
			return parameterLength;
		}

		Scalar::Scalar(double value)
		{
			values.push_back(value);
		}

		ParameterType Scalar::getType() const
		{
			return ParameterType::PT_SCALAR;
		}

		// Source: http://stackoverflow.com/questions/9843999/calculate-number-of-decimal-places-for-a-float-value-without-libraries
		void Scalar::getValueAndNegativeOrderOfMagnitude(double dValue, int& iValue, unsigned int& negativeOrderOfMagnitude)
		{
			double threshold = 0.000000005;
			negativeOrderOfMagnitude = 0;
			while (abs(dValue - round(dValue)) > threshold)
			{
				dValue *= 10.0;
				threshold *= 10.0;
				negativeOrderOfMagnitude++;
			}
			iValue = (int)floor(dValue);
		}

		DispatchTable::Parameter Scalar::toDispatchTableParameter() const
		{
			return DispatchTable::Parameter(getType(), values);
		}

		void Scalar::print(std::ostream& out, bool plain /*= false*/) const
		{
			if (plain)
			{
				out << values[0];
			}
			else
			{
				int value;
				unsigned int NOoM;
				getValueAndNegativeOrderOfMagnitude(values[0], value, NOoM);
				if (NOoM > 0)
					out << "Scalar<" << value << ", " << NOoM << ">";
				else
					out << "Scalar<" << value << ">";
			}
		}

		void Scalar::encode(float* buffer, unsigned int pos) const
		{
			buffer[pos] = PackUtils::packOperand(OperandType::ORT_SCALAR, (float)values[0]);
		}

		bool Scalar::isEqual(const Parameter* other) const
		{
			const Scalar* otherParam = dynamic_cast<const Scalar*>(other);
			if (otherParam == 0)
				return false;
			return compareValues(otherParam->values);
		}

		Axis::Axis(PGA::Axis axis)
		{
			values.push_back(static_cast<double>(axis));
		}

		ParameterType Axis::getType() const
		{
			return ParameterType::PT_SCALAR;
		}

		DispatchTable::Parameter Axis::toDispatchTableParameter() const
		{
			return DispatchTable::Parameter(getType(), values);
		}

		void Axis::print(std::ostream& out, bool plain /*= false*/) const
		{
			if (!plain)
				out << "AxisParam<";
			out << EnumUtils::toString(static_cast<PGA::Axis>((unsigned int)values[0]), plain);
			if (!plain)
				out << ">";
		}

		void Axis::encode(float* buffer, unsigned int pos) const
		{
			buffer[pos] = PackUtils::packOperand(OperandType::ORT_SCALAR, (float)values[0]);
		}

		bool Axis::isEqual(const Parameter* other) const
		{
			const Axis* otherParam = dynamic_cast<const Axis*>(other);
			if (otherParam == 0)
				return false;
			return compareValues(otherParam->values);
		}

		RepeatMode::RepeatMode(PGA::RepeatMode repeatMode)
		{
			values.push_back(static_cast<double>(repeatMode));
		}

		ParameterType RepeatMode::getType() const
		{
			return ParameterType::PT_SCALAR;
		}

		DispatchTable::Parameter RepeatMode::toDispatchTableParameter() const
		{
			return DispatchTable::Parameter(getType(), values);
		}

		void RepeatMode::print(std::ostream& out, bool plain /*= false*/) const
		{
			if (!plain)
				out << "RepeatModeParam<";
			out << EnumUtils::toString(static_cast<PGA::RepeatMode>((unsigned int)values[0]), plain);
			if (!plain)
				out << ">";
		}

		void RepeatMode::encode(float* buffer, unsigned int pos) const
		{
			buffer[pos] = PackUtils::packOperand(OperandType::ORT_SCALAR, (float)values[0]);
		}

		bool RepeatMode::isEqual(const Parameter* other) const
		{
			const RepeatMode* otherParam = dynamic_cast<const RepeatMode*>(other);
			if (otherParam == 0)
				return false;
			return compareValues(otherParam->values);
		}

		Rand::Rand(double min, double max, double seed /*= 0*/)
		{
			values.push_back(min); 
			values.push_back(max); 
			values.push_back(seed);
		}

		ParameterType Rand::getType() const
		{
			return ParameterType::PT_RAND;
		}

		DispatchTable::Parameter Rand::toDispatchTableParameter() const
		{
			return DispatchTable::Parameter(getType(), values);
		}

		void Rand::print(std::ostream& out, bool plain /*= false*/) const
		{
			if (plain)
			{
				out << "rand(" << values[0] << ", " << values[1] << ")";
			}
			else
			{
				int min, max;
				unsigned int minNOoM, maxNOoM;
				Scalar::getValueAndNegativeOrderOfMagnitude(values[0], min, minNOoM);
				Scalar::getValueAndNegativeOrderOfMagnitude(values[1], max, maxNOoM);
				out << "Rand<" << min << ", " << max << ", " << minNOoM << ", " << maxNOoM << ">";
			}
		}

		void Rand::encode(float* buffer, unsigned int pos) const
		{
			buffer[pos] = PackUtils::packOperand(OperandType::ORT_RAND, PackUtils::packFloat2((float)values[0], (float)values[1]));
		}

		bool Rand::isEqual(const Parameter* other) const
		{
			const Rand* otherParam = dynamic_cast<const Rand*>(other);
			if (otherParam == 0)
				return false;
			return compareValues(otherParam->values);
		}

		ShapeAttr::ShapeAttr(ShapeAttribute type) : type(type)
		{
		}

		ShapeAttr::ShapeAttr(ShapeAttribute type, double axis) : type(type)
		{
			values.push_back(axis);
		}

		ShapeAttr::ShapeAttr(ShapeAttribute type, double axis, double component) : type(type)
		{
			values.push_back(axis);	
			values.push_back(component);
		}

		ParameterType ShapeAttr::getType() const
		{
			switch (type)
			{
			case ShapeAttribute::SHAPE_POS:
				return ParameterType::PT_SHAPE_ATTR;
			case ShapeAttribute::SHAPE_SIZE:
				return ParameterType::PT_SHAPE_ATTR;
			case ShapeAttribute::SHAPE_SEED:
				return ParameterType::PT_SHAPE_ATTR;
			case ShapeAttribute::SHAPE_CUSTOM_ATTR:
				return ParameterType::PT_SHAPE_ATTR;
			default:
				throw std::runtime_error("PGA::Compiler::ShapeAttrParam::getType(): unknown shape attribute [type=" + std::to_string(type) + "]");
			}
		}

		DispatchTable::Parameter ShapeAttr::toDispatchTableParameter() const
		{
			return DispatchTable::Parameter(getType(), { getShapeAttrIndex() });
		}

		void ShapeAttr::print(std::ostream& out, bool plain /*= false*/) const
		{
			switch (type)
			{
			case ShapeAttribute::SHAPE_POS:
				if (plain)
					out << "shape.pos.";
				else
					out << "ShapePos<";
				out << EnumUtils::toString(static_cast<PGA::Axis>((unsigned int)values[0]), plain);
				if (!plain)
					out << ">";
				break;
			case ShapeAttribute::SHAPE_SIZE:
				if (plain)
					out << "shape.size.";
				else
					out << "ShapeSize<";
				out << EnumUtils::toString(static_cast<PGA::Axis>((unsigned int)values[0]), plain);
				if (!plain)
					out << ">";
				break;
			case ShapeAttribute::SHAPE_ROTATION:
				if (plain)
					out << "shape.rotation.";
				else
					out << "ShapeRotation<";
				out << EnumUtils::toString(static_cast<PGA::Axis>((unsigned int)values[0]), plain);
				out << ", ";
				out << EnumUtils::toString(static_cast<PGA::Axis>((unsigned int)(values[1]), plain));
				if (!plain)
					out << ">";
				break;
			case ShapeAttribute::SHAPE_NORMAL:
				if (plain)
					out << "shape.normal.";
				else
					out << "ShapeNormal<";
				out << EnumUtils::toString(static_cast<PGA::Axis>((unsigned int)values[0]), plain);
				if (!plain)
					out << ">";
				break;
			case ShapeAttribute::SHAPE_SEED:
				if (plain)
					out << "shape.seed";
				else
					out << "ShapeSeed";
				break;
			case ShapeAttribute::SHAPE_CUSTOM_ATTR:
				if (plain)
					out << "shape.customAttr";
				else
					out << "ShapeCustomAttribute";
				break;
			default:
				throw std::runtime_error("PGA::Compiler::ShapeAttrParam::print(): unknown shape attribute");
			}
		}

		void ShapeAttr::encode(float* buffer, unsigned int pos) const
		{
			switch (type)
			{
			case SHAPE_POS:
			{
				buffer[pos] = PackUtils::packOperand(OperandType::ORT_SHAPE_ATTR, (float)((values[0] + 1) * 4 - 1));
			}
				break;
			case SHAPE_SIZE:
			{
				buffer[pos] = PackUtils::packOperand(OperandType::ORT_SHAPE_ATTR, (float)(values[0] + 12));
			}
				break;
			case SHAPE_ROTATION:
			{
				buffer[pos] = PackUtils::packOperand(OperandType::ORT_SHAPE_ATTR, (float)((values[1] * 4) + values[0]));
			}
				break;
			case SHAPE_NORMAL:
			{
				buffer[pos] = PackUtils::packOperand(OperandType::ORT_SHAPE_ATTR, (float)((values[0] * 4) + 2));
			}
				break;
			case SHAPE_SEED:
			{
				buffer[pos] = PackUtils::packOperand(OperandType::ORT_SHAPE_ATTR, 15.0f);
			}
				break;
			case SHAPE_CUSTOM_ATTR:
			{
				buffer[pos] = PackUtils::packOperand(OperandType::ORT_SHAPE_ATTR, 16.0f);
			}
				break;
			default:
				throw std::runtime_error("PGA::Compiler::ShapeAttrParam::encode(..): unknown shape attribute");
			}
		}

		bool ShapeAttr::isEqual(const Parameter* other) const
		{
			const ShapeAttr* otherParam = dynamic_cast<const ShapeAttr*>(other);
			if (otherParam == 0)
				return false;
			return compareValues(otherParam->values);
		}

		double ShapeAttr::at(size_t idx) const
		{
			if (idx > 0)
				throw std::runtime_error("PGA::Compiler::ShapeAttrParam::at(..): invalid value index");
			return getShapeAttrIndex();
		}

		double ShapeAttr::getShapeAttrIndex() const
		{
			switch (type)
			{
			case SHAPE_POS:
				return (values[0] + 1) * 4 - 1;
			case SHAPE_SIZE:
				return values[0] + 12;
			case SHAPE_SEED:
				return 15;
			case SHAPE_CUSTOM_ATTR:
				return 16;
			default:
				throw std::runtime_error("PGA::Compiler::ShapeAttrParam::getShapeAttrIndex(): unknown shape attribute");
			}
		}

		Vec2::Vec2(double x, double y)
		{
			this->values.push_back(x);
			this->values.push_back(y);
		}

		ParameterType Vec2::getType() const
		{
			return ParameterType::PT_VEC2;
		}

		DispatchTable::Parameter Vec2::toDispatchTableParameter() const
		{
			return DispatchTable::Parameter(getType(), values);
		}

		void Vec2::print(std::ostream& out, bool plain /*= false*/) const
		{
			if (plain)
			{
				out << "(" << values[0] << ", " << values[1] << ")";
			}
			else
			{
				int x, y;
				unsigned int xNOoM, yNOoM;
				Scalar::getValueAndNegativeOrderOfMagnitude(values[0], x, xNOoM);
				Scalar::getValueAndNegativeOrderOfMagnitude(values[1], y, yNOoM);
				out << "Vec2<" << x << ", " << y << ", " << xNOoM << ", " << yNOoM << ">";
			}
		}

		void Vec2::encode(float* buffer, unsigned int pos) const
		{
			// TODO:
			buffer[pos] = PackUtils::packOperand(OperandType::ORT_SCALAR, length(math::float2((float)values[0], (float)values[1])));
		}

		bool Vec2::isEqual(const Parameter* other) const
		{
			const Vec2* otherParam = dynamic_cast<const Vec2*>(other);
			if (otherParam == 0)
				return false;
			return compareValues(otherParam->values);
		}

		Exp::Exp(OperationType operationType, std::shared_ptr<Parameter> leftOperand, std::shared_ptr<Parameter> rightOperand) : operationType(operationType), leftOperand(leftOperand), rightOperand(rightOperand)
		{
			parameterLength = leftOperand->getParameterLength() + rightOperand->getParameterLength() + 1;
		}

		ParameterType Exp::getType() const
		{
			return ParameterType::PT_EXP;
		}

		DispatchTable::Parameter Exp::toDispatchTableParameter() const
		{
			auto bufferLength = leftOperand->getParameterLength() + rightOperand->getParameterLength() + 2;
			if (bufferLength > Constants::MaxNumParameterValues)
				throw std::runtime_error("PGA::Compiler::ExpressionParam::toDispatchTableParameter(): bufferLength > Constants::MaxNumParameterValues");
			float floatValues[Constants::MaxNumParameterValues];
			floatValues[0] = static_cast<float>(bufferLength - 1);
			encode(floatValues + 1, 0);
			std::vector<double> dValues;
			dValues.resize(bufferLength);
			std::copy(floatValues, floatValues + bufferLength, dValues.begin());
			return DispatchTable::Parameter(getType(), dValues);
		}

		void Exp::print(std::ostream& out, bool plain /*= false*/) const
		{
			if (plain)
			{
				out << "(";
				leftOperand->print(out, true); 
				out << " " << EnumUtils::toString(operationType, true) << " "; 
				rightOperand->print(out, true);
				out << ")";
			}
			else
			{
				out << "Exp<" << *(leftOperand) << ", " << *(rightOperand) << ", ";
				out << EnumUtils::toString(operationType, false);
				out << ">";
			}
		}

		void Exp::encode(float* buffer, unsigned int pos /*= 0*/) const
		{
			leftOperand->encode(buffer, pos);
			rightOperand->encode(buffer, pos + leftOperand->getParameterLength());
			buffer[pos + leftOperand->getParameterLength() + rightOperand->getParameterLength()] = PackUtils::packOperand(ORT_OP, (float)operationType);
		}

		bool Exp::isEqual(const Parameter* other) const
		{
			const Exp* otherParam = dynamic_cast<const Exp*>(other);
			if (otherParam == 0)
				return false;
			if (operationType != otherParam->operationType)
				return false;
			if (!leftOperand->isEqual(otherParam->leftOperand.get()))
				return false;
			if (!rightOperand->isEqual(otherParam->rightOperand.get()))
				return false;
			return true;
		}

	}

}

std::ostream& operator<<(std::ostream& os, const PGA::Compiler::Parameter& obj)
{
	obj.print(os);
	return os;
}