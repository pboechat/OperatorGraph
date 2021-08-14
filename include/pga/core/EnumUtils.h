#pragma once

#include "Axis.h"
#include "GPUTechnique.h"
#include "OperandType.h"
#include "OperationType.h"
#include "ParameterType.h"
#include "RepeatMode.h"

#include <stdexcept>
#include <string>

namespace PGA
{
	class EnumUtils
	{
	public:
		EnumUtils() = delete;

		static std::string toString(PGA::GPU::Technique technique, bool plain = false)
		{
			switch (technique)
			{
			case PGA::GPU::KERNELS:
				return (plain) ? "kernels" : "KERNELS";
			case PGA::GPU::DYN_PAR:
				return (plain) ? "dynamic parallelism" : "DYN_PAN";
			case PGA::GPU::MEGAKERNEL:
				return (plain) ? "megakernel" : "MEGAKERNEL";
			case PGA::GPU::MEGAKERNEL_LOCAL_QUEUES:
				return (plain) ? "megakernel with local queues" : "MEGAKERNEL_LOCAL_QUEUES";
			default:
				throw std::runtime_error("PGA::EnumUtils::toString(..): unknown GPU technique [technique=" + std::to_string(technique) + "]");
			}
			return "";
		}

		static std::string toString(PGA::Axis axis, bool plain = false)
		{
			switch (axis)
			{
			case Axis::X:
				return (plain) ? "x" : "X";
			case Axis::Y:
				return (plain) ? "y" : "Y";
			case Axis::Z:
				return (plain) ? "z" : "Z";
			default:
				throw std::runtime_error("PGA::EnumUtils::toString(..): unknown axis [axis=" + std::to_string(axis) + "]");
			}
			return "";
		}

		static std::string toString(PGA::RepeatMode repeatMode, bool plain = false)
		{
			switch (repeatMode)
			{
			case RepeatMode::ANCHOR_TO_START:
				return (plain) ? "anchor to start" : "ANCHOR_TO_START";
			case RepeatMode::ANCHOR_TO_END:
				return (plain) ? "anchor to end" : "ANCHOR_TO_END";
			case RepeatMode::ADJUST_TO_FILL:
				return (plain) ? "adjust to fill" : "ADJUST_TO_FILL";
			default:
				throw std::runtime_error("PGA::EnumUtils::toString(..): unknown repeat mode [repeatMode=" + std::to_string(repeatMode) + "]");
			}
			return "";
		}

		static std::string toString(PGA::ParameterType parameterType, bool plain = false)
		{
			switch (parameterType)
			{
			case PGA::ParameterType::PT_SCALAR:
				return (plain) ? "scalar parameter" : "PT_SCALAR";
			case PGA::ParameterType::PT_RAND:
				return (plain) ? "random function parameter" : "PT_RAND";
			case PGA::ParameterType::PT_SHAPE_ATTR:
				return (plain) ? "shape attribute parameter" : "PT_SHAPE_ATTR";
			case PGA::ParameterType::PT_EXP:
				return (plain) ? "expression parameter" : "PT_EXP";
			case PGA::ParameterType::PT_VEC2:
				return (plain) ? "vector 2 parameter" : "PT_VEC2";
			case PGA::ParameterType::PT_VEC4:
				return (plain) ? "vector 4 parameter" : "PT_VEC4";
			default:
				throw std::runtime_error("PGA::EnumUtils::toString(..): unknown parameter type [parameterType=" + std::to_string(parameterType) + "]");
			}
			return "";
		}

		static std::string toString(PGA::OperandType operandType, bool plain = false)
		{
			switch (operandType)
			{
			case PGA::OperandType::ORT_SCALAR:
				return (plain) ? "scalar operand type" : "ORT_SCALAR";
			case PGA::OperandType::ORT_OP:
				return (plain) ? "operation operand type" : "ORT_OP";
			case PGA::OperandType::ORT_SHAPE_ATTR:
				return (plain) ? "shape attribute operand type" : "ORT_SHAPE_ATTR";
			case PGA::OperandType::ORT_RAND:
				return (plain) ? "random function operand type" : "ORT_RAND";
			default:
				throw std::runtime_error("PGA::EnumUtils::toString(..): unknown operand type [operandType=" + std::to_string(operandType) + "]");
			}
			return "";
		}

		static std::string toString(PGA::OperationType operationType, bool plain = false)
		{
			switch (operationType)
			{
			case PGA::OperationType::OPT_ADD:
				return (plain) ? "+" : "Add";
			case PGA::OperationType::OPT_SUB:
				return (plain) ? "-" : "Sub";
			case PGA::OperationType::OPT_MULTI:
				return (plain) ? "*" : "Multi";
			case PGA::OperationType::OPT_DIV:
				return (plain) ? "/" : "Div";
			case PGA::OperationType::OPT_EQ:
				return (plain) ? "==" : "Eq";
			case PGA::OperationType::OPT_NEQ:
				return (plain) ? "!=" : "Neq";
			case PGA::OperationType::OPT_LT:
				return (plain) ? "<" : "Lt";
			case PGA::OperationType::OPT_GT:
				return (plain) ? ">" : "Gt";
			case PGA::OperationType::OPT_LEQ:
				return (plain) ? "<=" : "Leq";
			case PGA::OperationType::OPT_GEQ:
				return (plain) ? ">=" : "Geq";
			case PGA::OperationType::OPT_AND:
				return (plain) ? "&&" : "And";
			case PGA::OperationType::OPT_OR:
				return (plain) ? "||" : "Or";
			default:
				throw std::runtime_error("PGA::EnumUtils::toString(..): unknown operation type [operationType=" + std::to_string(operationType) + "]");
			}
			return "";
		}

	};

}
