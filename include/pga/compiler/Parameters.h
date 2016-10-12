#pragma once

#include <vector>
#include <memory>
#include <iostream>

#include <pga/core/Axis.h>
#include <pga/core/RepeatMode.h>
#include <pga/core/ParameterType.h>
#include <pga/core/OperationType.h>
#include <pga/core/DispatchTable.h>
#include <pga/compiler/ShapeAttribute.h>

namespace PGA
{
	namespace Compiler
	{
		//////////////////////////////////////////////////////////////////////////
		class Parameter
		{
		protected:
			unsigned int parameterLength = 1;
			std::vector<double> values;

			bool compareValues(const std::vector<double>& values) const;

		public:
			Parameter() = default;
			virtual ~Parameter() = default;

			size_t size() const;
			virtual double at(size_t idx) const;
			unsigned int getParameterLength() const;
			virtual ParameterType getType() const = 0;
			virtual DispatchTable::Parameter toDispatchTableParameter() const = 0;
			virtual void encode(float* buffer, unsigned int pos) const = 0;
			virtual void print(std::ostream& out, bool plain = false) const = 0;
			virtual bool isEqual(const Parameter* other) const = 0;

		};

		//////////////////////////////////////////////////////////////////////////
		class Scalar : public Parameter
		{
		public:
			Scalar(double value);
			virtual ~Scalar() = default;

			static void getValueAndNegativeOrderOfMagnitude(double dValue, int& iValue, unsigned int& negativeOrderOfMagnitude);

			virtual ParameterType getType() const;
			virtual DispatchTable::Parameter toDispatchTableParameter() const;
			virtual void print(std::ostream& out, bool plain = false) const;
			virtual void encode(float* buffer, unsigned int pos) const;
			virtual bool isEqual(const Parameter* other) const;

		};

		//////////////////////////////////////////////////////////////////////////
		class Axis : public Parameter
		{
		public:
			Axis(PGA::Axis axis);
			virtual ~Axis() = default;

			virtual ParameterType getType() const;
			virtual DispatchTable::Parameter toDispatchTableParameter() const;
			virtual void print(std::ostream& out, bool plain = false) const;
			virtual void encode(float* buffer, unsigned int pos) const;
			virtual bool isEqual(const Parameter* other) const;

		};

		//////////////////////////////////////////////////////////////////////////
		class RepeatMode : public Parameter
		{
		public:
			RepeatMode(PGA::RepeatMode repeatMode);
			virtual ~RepeatMode() = default;

			virtual ParameterType getType() const;
			virtual DispatchTable::Parameter toDispatchTableParameter() const;
			virtual void print(std::ostream& out, bool plain = false) const;
			virtual void encode(float* buffer, unsigned int pos) const;
			virtual bool isEqual(const Parameter* other) const;

		};

		//////////////////////////////////////////////////////////////////////////
		class Rand : public Parameter
		{
		public:
			Rand(double min, double max, double seed = 0);
			virtual ~Rand() = default;

			virtual ParameterType getType() const;
			virtual DispatchTable::Parameter toDispatchTableParameter() const;
			virtual void print(std::ostream& out, bool plain = false) const;
			virtual void encode(float* buffer, unsigned int pos) const;
			virtual bool isEqual(const Parameter* other) const;

		};

		//////////////////////////////////////////////////////////////////////////
		class ShapeAttr : public Parameter
		{
		private:
			ShapeAttribute type;
			double getShapeAttrIndex() const;

		public:
			ShapeAttr(ShapeAttribute type);
			ShapeAttr(ShapeAttribute type, double axis);
			ShapeAttr(ShapeAttribute type, double axis, double component);
			virtual ~ShapeAttr() = default;

			virtual double at(size_t idx) const;
			virtual ParameterType getType() const;
			virtual DispatchTable::Parameter toDispatchTableParameter() const;
			virtual void print(std::ostream& out, bool plain = false) const;
			virtual void encode(float* buffer, unsigned int pos) const;
			virtual bool isEqual(const Parameter* other) const;

		};

		//////////////////////////////////////////////////////////////////////////
		class Vec2 : public Parameter
		{
		public:
			Vec2(double x, double y);
			virtual ~Vec2() = default;

			virtual ParameterType getType() const;
			virtual DispatchTable::Parameter toDispatchTableParameter() const;
			virtual void print(std::ostream& out, bool plain = false) const;
			virtual void encode(float* buffer, unsigned int pos) const;
			virtual bool isEqual(const Parameter* other) const;

		};

		//////////////////////////////////////////////////////////////////////////
		class Exp : public Parameter
		{
		protected:
			OperationType operationType;
			std::shared_ptr<Parameter> leftOperand;
			std::shared_ptr<Parameter> rightOperand;

		public:
			Exp(OperationType operationType, std::shared_ptr<Parameter> leftOperand, std::shared_ptr<Parameter> rightOperand);
			virtual ~Exp() = default;

			virtual ParameterType getType() const;
			virtual DispatchTable::Parameter toDispatchTableParameter() const;
			virtual void print(std::ostream& out, bool plain = false) const;
			virtual void encode(float* buffer, unsigned int pos = 0) const;
			virtual bool isEqual(const Parameter* other) const;

		};

	}

}

//////////////////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& os, const PGA::Compiler::Parameter& obj);