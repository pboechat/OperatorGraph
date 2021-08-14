#pragma once

#include <pga/compiler/ShapeType.h>
#include <pga/core/Shapes.cuh>

namespace PGA
{
	namespace Compiler
	{
		template <typename ShapeT>
		struct ShapeToShapeType;

		template <>
		struct ShapeToShapeType < Shapes::Triangle >
		{
			static const ShapeType Result = ShapeType::TRIANGLE;

		};

		template <>
		struct ShapeToShapeType < Shapes::Quad >
		{
			static const ShapeType Result = ShapeType::QUAD;

		};

		template <>
		struct ShapeToShapeType < Shapes::Pentagon >
		{
			static const ShapeType Result = ShapeType::PENTAGON;

		};

		template <>
		struct ShapeToShapeType < Shapes::Hexagon >
		{
			static const ShapeType Result = ShapeType::HEXAGON;

		};

		template <>
		struct ShapeToShapeType < Shapes::Heptagon >
		{
			static const ShapeType Result = ShapeType::HEPTAGON;

		};

		template <>
		struct ShapeToShapeType < Shapes::Octagon >
		{
			static const ShapeType Result = ShapeType::OCTAGON;

		};

		template <unsigned int MaxNumVerticesT>
		struct ShapeToShapeType < Shapes::DynamicPolygon<MaxNumVerticesT, true> >
		{
			static const ShapeType Result = ShapeType::DYNAMIC_CONVEX_POLYGON;

		};

		template <unsigned int MaxNumVerticesT>
		struct ShapeToShapeType < Shapes::DynamicPolygon<MaxNumVerticesT, false> >
		{
			static const ShapeType Result = ShapeType::DYNAMIC_POLYGON;

		};

		template <>
		struct ShapeToShapeType < Shapes::Prism3 >
		{
			static const ShapeType Result = ShapeType::PRISM3;

		};

		template <>
		struct ShapeToShapeType < Shapes::Box >
		{
			static const ShapeType Result = ShapeType::BOX;

		};

		template <>
		struct ShapeToShapeType < Shapes::Prism5 >
		{
			static const ShapeType Result = ShapeType::PRISM5;

		};

		template <>
		struct ShapeToShapeType < Shapes::Prism6 >
		{
			static const ShapeType Result = ShapeType::PRISM6;

		};

		template <>
		struct ShapeToShapeType < Shapes::Prism7 >
		{
			static const ShapeType Result = ShapeType::PRISM7;

		};

		template <>
		struct ShapeToShapeType < Shapes::Prism8 >
		{
			static const ShapeType Result = ShapeType::PRISM8;

		};

		template <unsigned int MaxNumVerticesT>
		struct ShapeToShapeType < Shapes::DynamicRightPrism<MaxNumVerticesT, true> >
		{
			static const ShapeType Result = ShapeType::DYNAMIC_CONVEX_RIGHT_PRISM;

		};

		template <unsigned int MaxNumVerticesT>
		struct ShapeToShapeType < Shapes::DynamicRightPrism<MaxNumVerticesT, false> >
		{
			static const ShapeType Result = ShapeType::DYNAMIC_RIGHT_PRISM;

		};

	}

}
