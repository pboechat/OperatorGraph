#pragma once

#include <pga/core/Operators.cuh>
#include <pga/compiler/OperatorType.h>
#include <pga/compiler/ShapeToShapeType.cuh>
#include <pga/compiler/ProcedureList.h>

namespace PGA
{
	namespace Compiler
	{
		template <typename ShapeT, typename OperatorT>
		struct SingleOperatorProcedureBuilder;

		template <typename ShapeT, unsigned int OperationCodeT, int PhaseIndexT, int EdgeIndexT, int EntryIndexT>
		struct SingleOperatorProcedureBuilder < ShapeT, Operators::FSCall<OperationCodeT, PhaseIndexT, EdgeIndexT, EntryIndexT> >
		{
			static void build(ProcedureList& procedureList)
			{
				// do nothing
			}

		};

		template <typename ShapeT, unsigned int OperationCodeT, unsigned int SuccessorOffsetT, int EdgeIndexT>
		struct SingleOperatorProcedureBuilder < ShapeT, Operators::PSCall<OperationCodeT, SuccessorOffsetT, EdgeIndexT> >
		{
			static void build(ProcedureList& procedureList)
			{
				// do nothing
			}

		};

		template <typename ShapeT, unsigned int SuccessorOffsetT, int EdgeIndexT>
		struct SingleOperatorProcedureBuilder < ShapeT, Operators::DCall<SuccessorOffsetT, EdgeIndexT> >
		{
			static void build(ProcedureList& procedureList)
			{
				// do nothing
			}

		};

		template <typename ShapeT, bool ParallelT, typename TopOperatorT, typename BottomOperatorT, typename SidesOperatorT>
		struct SingleOperatorProcedureBuilder < ShapeT, Operators::ComponentSplit<ParallelT, TopOperatorT, BottomOperatorT, SidesOperatorT> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::COMPSPLIT, 0);
				SingleOperatorProcedureBuilder<TopOperatorT>::build(procedureList);
				SingleOperatorProcedureBuilder<BottomOperatorT>::build(procedureList);
				SingleOperatorProcedureBuilder<SidesOperatorT>::build(procedureList);
			}

		};

		template <typename ShapeT>
		struct SingleOperatorProcedureBuilder < ShapeT, Operators::Discard >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::DISCARD, 0);
			}

		};

		template <typename ShapeT, typename AxisT, typename ExtentT, typename NextOperatorT>
		struct SingleOperatorProcedureBuilder < ShapeT, Operators::Extrude<AxisT, ExtentT, NextOperatorT> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::EXTRUDE, 0);
				SingleOperatorProcedureBuilder<NextOperatorT>::build(procedureList);
			}

		};

		template <typename ShapeT, bool ParallelT, unsigned int GenFuncIdxT, typename TerminalIdxT, typename Attr1T, typename Attr2T, typename Attr3T, typename Attr4T, typename Attr5T>
		struct SingleOperatorProcedureBuilder < Operators::Generate<ParallelT, GenFuncIdxT, TerminalIdxT, Attr1T, Attr2T, Attr3T, Attr4T, Attr5T> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::GENERATE, GenFuncIdxT);
			}

		};

		template <typename ShapeT, typename AxisT, typename SizeT, typename NextOperatorIfTrueT, typename NextOperatorIfFalseT>
		struct SingleOperatorProcedureBuilder < Operators::IfSizeLess<AxisT, SizeT, NextOperatorIfTrueT, NextOperatorIfFalseT> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::IFSIZELESS, 0);
				SingleOperatorProcedureBuilder<NextOperatorIfTrueT>::build(procedureList);
				SingleOperatorProcedureBuilder<NextOperatorIfFalseT>::build(procedureList);
			}

		};

		template <typename ShapeT, bool ParallelT, typename AxisT, typename RepetitionsExtentT, typename NextOperatorT>
		struct SingleOperatorProcedureBuilder < Operators::Repeat<ParallelT, AxisT, RepetitionsExtentT, NextOperatorT> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::REPEAT, 0);
				SingleOperatorProcedureBuilder<NextOperatorT>::build(procedureList);
			}

		};

		template <typename ShapeT, typename XT, typename YT, typename ZT, typename NextOperatorT>
		struct SingleOperatorProcedureBuilder < Operators::Scale<XT, YT, ZT, NextOperatorT> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::SCALE, 0);
				SingleOperatorProcedureBuilder<NextOperatorT>::build(procedureList);
			}

		};

		template <typename ShapeT, typename XT, typename YT, typename ZT, typename NextOperatorT>
		struct SingleOperatorProcedureBuilder < Operators::Rotate<XT, YT, ZT, NextOperatorT> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::ROTATE, 0);
				SingleOperatorProcedureBuilder<NextOperatorT>::build(procedureList);
			}

		};

		template <typename ShapeT, typename XT, typename YT, typename ZT, typename NextOperatorT>
		struct SingleOperatorProcedureBuilder < Operators::Translate<XT, YT, ZT, NextOperatorT> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::TRANSLATE, 0);
				SingleOperatorProcedureBuilder<NextOperatorT>::build(procedureList);
			}

		};

		template <typename ShapeT, typename NextOperatorT, typename... Verts1T>
		struct SingleOperatorProcedureBuilder < Operators::SetAsDynamicConvexPolygon<NextOperatorT, Verts1T> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::SET_AS_DYNAMIC_CONVEX_POLYGON, 0);
				SingleOperatorProcedureBuilder<NextOperatorT>::build(procedureList);
			}

		};

		template <typename ShapeT, typename NextOperatorT, typename... Verts1T>
		struct SingleOperatorProcedureBuilder < Operators::SetAsDynamicConvexRightPrism<NextOperatorT, Verts1T> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::SET_AS_DYNAMIC_CONVEX_RIGHT_PRISM, 0);
				SingleOperatorProcedureBuilder<NextOperatorT>::build(procedureList);
			}

		};

		template <typename ShapeT, typename NextOperatorT, typename... Verts1T>
		struct SingleOperatorProcedureBuilder < Operators::SetAsDynamicConcavePolygon<NextOperatorT, Verts1T> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::SET_AS_DYNAMIC_CONCAVE_POLYGON, 0);
				SingleOperatorProcedureBuilder<NextOperatorT>::build(procedureList);
			}

		};

		template <typename ShapeT, typename NextOperatorT, typename... Verts1T>
		struct SingleOperatorProcedureBuilder < Operators::SetAsDynamicConcaveRightPrism<NextOperatorT, Verts1T> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::SET_AS_DYNAMIC_CONCAVE_RIGHT_PRISM, 0);
				SingleOperatorProcedureBuilder<NextOperatorT>::build(procedureList);
			}

		};

		template <typename ShapeT, typename XT, typename YT, typename ZT, typename NextOperatorT>
		struct SingleOperatorProcedureBuilder < Operators::SwapSize<XT, YT, ZT, NextOperatorT> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::SWAPSIZE, 0);
				SingleOperatorProcedureBuilder<NextOperatorT>::build(procedureList);
			}

		};

		template <typename ShapeT, typename... ParamsT>
		struct SingleOperatorProcedureBuilder < Operators::Replicate<ParamsT...> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::REPLICATE, 0);
				SingleOperatorProcedureBuilder_Iterator2<ShapeT, ParamsT...>::build(procedureList);
			}

		};

		template <typename ShapeT, typename ColliderTagT, typename NextOperatorT>
		struct SingleOperatorProcedureBuilder < Operators::Collider<ColliderTagT, NextOperatorT> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::COLLIDER, 0);
				SingleOperatorProcedureBuilder<NextOperatorT>::build(procedureList);
			}

		};

		template <typename ShapeT, typename ColliderTagT, typename NextOperatorIfTrueT, typename NextOperatorIfFalseT>
		struct SingleOperatorProcedureBuilder < Operators::IfCollides<ColliderTagT, NextOperatorIfTrueT, NextOperatorIfFalseT> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::IFCOLLIDES, 0);
				SingleOperatorProcedureBuilder<NextOperatorIfTrueT>::build(procedureList);
				SingleOperatorProcedureBuilder<NextOperatorIfFalseT>::build(procedureList);
			}

		};

		template <typename ShapeT, typename ExpressionT, typename NextOperatorIfTrueT, typename NextOperatorIfFalseT>
		struct SingleOperatorProcedureBuilder < Operators::If<ExpressionT, NextOperatorIfTrueT, NextOperatorIfFalseT> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::IF, 0);
				SingleOperatorProcedureBuilder<NextOperatorIfTrueT>::build(procedureList);
				SingleOperatorProcedureBuilder<NextOperatorIfFalseT>::build(procedureList);
			}

		};

		template <typename ShapeT, typename... ParamsT>
		struct SingleOperatorProcedureBuilder_Iterator1;

		template <typename ShapeT, typename FirstT, typename... RemainderT>
		struct SingleOperatorProcedureBuilder_Iterator1 < ShapeT, FirstT, RemainderT... >
		{
		private:
			typedef typename FirstT::Value NextOperator;

		public:
			static void build(ProcedureList& procedureList)
			{
				SingleOperatorProcedureBuilder<ShapeT, NextOperator>::build(procedureList);
				SingleOperatorProcedureBuilder_Iterator1<ShapeT, RemainderT...>::build(procedureList);
			}

		};

		template <typename ShapeT, typename LastT>
		struct SingleOperatorProcedureBuilder_Iterator1 < ShapeT, LastT >
		{
		private:
			typedef typename LastT::Value NextOperator;

		public:
			static void build(ProcedureList& procedureList)
			{
				SingleOperatorProcedureBuilder<ShapeT, NextOperator>::build(procedureList);
			}

		};

		template <typename ShapeT, typename... ParamsT>
		struct SingleOperatorProcedureBuilder_Iterator2;

		template <typename ShapeT, typename FirstT, typename... RemainderT>
		struct SingleOperatorProcedureBuilder_Iterator2 < ShapeT, FirstT, RemainderT... >
		{
			static void build(ProcedureList& procedureList)
			{
				SingleOperatorProcedureBuilder<ShapeT, FirstT>::build(procedureList);
				SingleOperatorProcedureBuilder_Iterator1<ShapeT, RemainderT...>::build(procedureList);
			}

		};

		template <typename ShapeT, typename LastT>
		struct SingleOperatorProcedureBuilder_Iterator2 < ShapeT, LastT >
		{
			static void build(ProcedureList& procedureList)
			{
				SingleOperatorProcedureBuilder<ShapeT, LastT>::build(procedureList);
			}

		};

		template <typename ShapeT, typename AxisT, typename... ParamsT>
		struct SingleOperatorProcedureBuilder < Operators::Subdivide<AxisT, ParamsT...> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::SUBDIV, 0);
				SingleOperatorProcedureBuilder_Iterator1<ShapeT, ParamsT...>::build(procedureList);
			}

		};

		template <typename ShapeT, typename ChanceT, typename... ParamsT>
		struct SingleOperatorProcedureBuilder < Operators::RandomRule<ChanceT, ParamsT...> >
		{
			static void build(ProcedureList& procedureList)
			{
				procedureList.procedures.emplace_back(ShapeToShapeType<ShapeT>::Result, OperatorType::STOCHASTIC, 0);
				SingleOperatorProcedureBuilder_Iterator1<ShapeT, ParamsT...>::build(procedureList);
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename ProcedureListT, unsigned int LengthT, unsigned int IndexT>
		struct ProcedureListBuilder_Iterator
		{
		private:
			typedef typename ProcedureList::template ItemAt<IndexT>::Result Proc;

		public:
			static void build(ProcedureList& procedureList)
			{
				SingleOperatorProcedureBuilder<Proc::ExpectedData, Proc::FirstOperator>::build(procedureList);
				ProcedureListBuilder_Iterator<ProcedureListT, LengthT, IndexT>::build(procedureList);
			}

		};

		template <typename ProcedureListT, unsigned int IndexT>
		struct ProcedureListBuilder_Iterator < ProcedureListT, IndexT, IndexT >
		{
			static void build(ProcedureList& procedureList)
			{
				// do nothing
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename ProcedureListT>
		struct ProcedureListBuilder
		{
			static void build(ProcedureList& procedureList)
			{
				ProcedureListBuilder_Iterator<ProcedureListT, ProcedureListT::Length>::build(procedureList);
			}

		};

	}

}