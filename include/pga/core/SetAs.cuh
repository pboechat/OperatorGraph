#pragma once

#include "Axis.h"
#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
#include "Parameters.cuh"
#include "Shapes.cuh"
#include "Symbol.cuh"
#include "SymbolDecorator.cuh"
#include "TStdLib.h"

#include <cuda_runtime_api.h>
#include <math/vector.h>

#include <stdexcept>
#include <string>

namespace PGA
{
	namespace Operators
	{
		template <typename NextOperatorT, typename NewShapeT, typename... Verts1T>
		class SetAs
		{
		private:
			static_assert(!T::EqualTypes<NextOperatorT, PGA::Parameters::DynParams>::Result, "PGA::Parameters::DynParams can only be used as a single parameter");

			template <typename... Verts2T>
			struct ForEachVertex;

			template <typename FirstT, typename... RemainderT>
			struct ForEachVertex < FirstT, RemainderT... >
			{
			private:
				static_assert(!T::EqualTypes<FirstT, PGA::Parameters::DynParams>::Result, "PGA::Parameters::DynParams can only be used as a single parameter");
				static const unsigned int VertexIndex = sizeof...(Verts1T)-sizeof...(RemainderT)-1;

			public:
				template <typename ShapeT, typename NewShapeT>
				__host__ __device__ __inline__ static void copyVertices(const Symbol<ShapeT>* symbol, Symbol<NewShapeT>& newSymbol)
				{
					newSymbol.vertices[VertexIndex] = FirstT::toFloat2(symbol);
					ForEachVertex<RemainderT...>::copyVertices(symbol, newSymbol);
				}

				__host__ __inline__ std::string toString()
				{
					return FirstT::toString() + ", " + ForEachVertex<Verts1T...>::toString();
				}

			};

			template <typename LastT>
			struct ForEachVertex < LastT >
			{
			private:
				static_assert(!T::EqualTypes<LastT, PGA::Parameters::DynParams>::Result, "PGA::Parameters::DynParams can only be used as a single parameter");
				static const unsigned int VertexIndex = sizeof...(Verts1T)-1;

			public:
				template <typename ShapeT, typename NewShapeT>
				__host__ __device__ __inline__ static void copyVertices(const Symbol<ShapeT>* symbol, Symbol<NewShapeT>& newSymbol)
				{
					newSymbol.vertices[VertexIndex] = LastT::toFloat2(symbol);
				}

				__host__ __inline__ std::string toString()
				{
					return LastT::toString();
				}

			};

			static const unsigned int NumSides = sizeof...(Verts1T);

		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				static_assert(NumSides <= Constants::MaxNumSides, "Cannot use set shape with no. of sides > Constants::MaxNumSides");
				Symbol<NewShapeT> newSymbol;
				newSymbol.numSides = NumSides;
				newSymbol.invert = false;
				ForEachVertex<Verts1T...>::copyVertices(symbol, newSymbol);
				newSymbol.setModel(symbol->getModel());
				newSymbol.setSize(symbol->getSize());
				newSymbol.setSeed(symbol->getSeed());
				newSymbol.setCustomAttribute(symbol->getCustomAttribute());
				SymbolDecorator<NextOperatorT>::run(symbol, &newSymbol);
				NextOperatorT::template execute<ContextT, NumThreadsT>(threadId, numThreads, queue, &newSymbol, shared);
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "SetAs";
			}

			__host__ __inline__ static std::string toString()
			{
				return "SetAs<" + NextOperatorT::toString() + ", " + Shapes::GetName<NewShapeT>::Result() + ", " + ForEachVertex<Verts1T...>::toString() + ">";
			}

		};

		template <typename NewShapeT, typename... Verts1T>
		class SetAs < PGA::Parameters::DynParams, NewShapeT, Verts1T... >
		{
		private:
			static_assert(sizeof...(Verts1T) == 0, "PGA::Parameters::DynParams can only be used as a single parameter");

		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				auto& entry = GlobalVars::getDispatchTableEntry(symbol->entryIndex);
				auto numSides = entry.numParameters;
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL >= 2)
				if (numSides > Constants::MaxNumSides)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("Cannot use set shape with no. of sides > Constants::MaxNumSides (numSides=%d, Constants::MaxNumSides=%d)\n", numSides, Constants::MaxNumSides);
					asm("trap;");
#else
					throw std::runtime_error(("Cannot use set shape with no. of sides > Constants::MaxNumSides (numSides=" + std::to_string(numSides) + ", Constants::MaxNumSides=" + std::to_string(Constants::MaxNumSides) + ")").c_str());
#endif
				}
#endif
				Symbol<NewShapeT> newSymbol;
				newSymbol.numSides = numSides;
				newSymbol.invert = false;
				for (auto i = 0; i <= numSides; i++)
					newSymbol.vertices[i] = Parameters::dynToFloat2(symbol, entry.parameters[i]);
				newSymbol.setModel(symbol->getModel());
				newSymbol.setSize(symbol->getSize());
				newSymbol.setSeed(symbol->getSeed());
				newSymbol.setCustomAttribute(symbol->getCustomAttribute());
				newSymbol.entryIndex = entry.successors[0].entryIndex;
				newSymbol.predecessor = symbol->entryIndex;
				unsigned int phaseIndex = entry.successors[0].phaseIndex;
				ContextT::Application::dispatchNonTerminal(newSymbol, phaseIndex, 0, queue);
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "SetAs";
			}

			__host__ __inline__ static std::string toString()
			{
				return "SetAs<DynParams, " + Shapes::GetName<NewShapeT>::Result() + ">";
			}

		};

		template <typename NextOperatorT, typename... Verts1T>
		class SetAsDynamicConvexPolygon : public SetAs < NextOperatorT, Shapes::DynamicPolygon<PGA::Constants::MaxNumSides, true>, Verts1T... > {};

		template <typename NextOperatorT, typename... Verts1T>
		class SetAsDynamicConvexRightPrism : public SetAs < NextOperatorT, Shapes::DynamicRightPrism<PGA::Constants::MaxNumSides, true>, Verts1T... > {};

		template <typename NextOperatorT, typename... Verts1T>
		class SetAsDynamicConcavePolygon : public SetAs < NextOperatorT, Shapes::DynamicPolygon<PGA::Constants::MaxNumSides, false>, Verts1T... > {};

		template <typename NextOperatorT, typename... Verts1T>
		class SetAsDynamicConcaveRightPrism : public SetAs < NextOperatorT, Shapes::DynamicRightPrism<PGA::Constants::MaxNumSides, false>, Verts1T... > {};

	}

}
