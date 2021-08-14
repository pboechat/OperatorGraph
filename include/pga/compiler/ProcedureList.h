#pragma once

#include <pga/compiler/EnumUtils.h>
#include <pga/compiler/OperatorType.h>
#include <pga/compiler/ShapeType.h>

#include <initializer_list>
#include <vector>

namespace PGA
{
	namespace Compiler
	{
		class ProcedureList;

		class SingleOperatorProcedure
		{
		private:
			OperatorType opType;
			ShapeType shapeType;
			size_t genFuncIdx;

			friend class ProcedureList;

		public:
			SingleOperatorProcedure(OperatorType opType, ShapeType shapeType, size_t genFuncIdx)
				: opType(opType), shapeType(shapeType), genFuncIdx(genFuncIdx)
			{
			}

			bool operator==(const SingleOperatorProcedure& other) const
			{
				return opType == other.opType && shapeType == other.shapeType && genFuncIdx == other.genFuncIdx;
			}

		};

		class ProcedureList
		{
		private:
			std::vector<SingleOperatorProcedure> procedures;

		public:
			ProcedureList(const std::initializer_list<SingleOperatorProcedure>& procedures)
			{
				for (auto& procedure : procedures)
				{
					auto it = std::find(this->procedures.begin(), this->procedures.end(), procedure);
					if (it != this->procedures.end())
						throw std::runtime_error("PGA::Compiler::ProcedureList::ctor(): repeated procedure [procedure.opType=" + PGA::Compiler::EnumUtils::toString(procedure.opType) + ", procedure.shapeType=" + PGA::Compiler::EnumUtils::toString(procedure.shapeType) + "]");
					this->procedures.emplace_back(procedure);
				}
			}

			long indexOf(const SingleOperatorProcedure& procedure) const
			{
				return indexOf(std::move(procedure));
			}

			long indexOf(const SingleOperatorProcedure&& procedure) const
			{
				auto it = std::find(procedures.begin(), procedures.end(), procedure);
				if (it == procedures.end())
					return -1;
				return (long)std::distance(procedures.begin(), it);
			}

			std::string nameOf(size_t procIdx) const
			{
				std::string procName = PGA::Compiler::EnumUtils::toString(procedures.at(procIdx).opType);
				std::string shapeName = PGA::Compiler::EnumUtils::toString(procedures.at(procIdx).shapeType);
				return procName + "[" + shapeName + "]";
			}

		};

	}

}

namespace std
{
	template <>
	struct hash<PGA::Compiler::SingleOperatorProcedure> : public unary_function < PGA::Compiler::SingleOperatorProcedure, size_t >
	{
		size_t operator()(const PGA::Compiler::SingleOperatorProcedure& value) const
		{
			return 0;
		}
	};

}