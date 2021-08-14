#pragma once

#include <pga/core/DebugFlags.h>
#include <pga/core/Statistics.h>
#include <pga/core/TStdLib.h>

namespace T
{
	template <>
	struct IsEnabled < PGA::Statistics::Execution >
	{
#ifdef PGA_ENABLE_STATISTICS
		static const bool Result = true;
#else
		static const bool Result = false;
#endif

	};

	template <>
	struct IsEnabled< PGA::DebugFlags::SymbolDispatch >
	{
#ifdef PGA_ENABLE_DEBUG_FLAGS
		static const bool Result = true;
#else
		static const bool Result = false;
#endif

	};

	template <>
	struct IsEnabled < PGA::DebugFlags::EdgeTraversal >
	{
#ifdef PGA_ENABLE_DEBUG_FLAGS
		static const bool Result = true;
#else
		static const bool Result = false;
#endif

	};

	template <>
	struct IsEnabled < PGA::DebugFlags::VertexVisit >
	{
#ifdef PGA_ENABLE_DEBUG_FLAGS
		static const bool Result = true;
#else
		static const bool Result = false;
#endif

	};

	template <>
	struct IsEnabled < PGA::DebugFlags::ExpressionEvaluation >
	{
#ifdef PGA_ENABLE_DEBUG_FLAGS
		static const bool Result = true;
#else
		static const bool Result = false;
#endif

	};

	template <>
	struct IsEnabled < PGA::DebugFlags::AllOperators>
	{
#ifdef PGA_ENABLE_DEBUG_FLAGS
		static const bool Result = true;
#else
		static const bool Result = false;
#endif

	};

	template <>
	struct IsEnabled < PGA::DebugFlags::BVHTraversal>
	{
#ifdef PGA_ENABLE_DEBUG_FLAGS
		static const bool Result = true;
#else
		static const bool Result = false;
#endif

	};

	template <>
	struct IsEnabled < PGA::DebugFlags::CollisionCheck>
	{
#ifdef PGA_ENABLE_DEBUG_FLAGS
		static const bool Result = true;
#else
		static const bool Result = false;
#endif

	};

	template <>
	struct IsEnabled < PGA::DebugFlags::Collider>
	{
#ifdef PGA_ENABLE_DEBUG_FLAGS
		static const bool Result = true;
#else
		static const bool Result = false;
#endif

	};

}
