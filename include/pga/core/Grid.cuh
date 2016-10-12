#pragma once

#include <cuda_runtime_api.h>

#include <math/vector.h>
#include <math/matrix.h>

#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
#include "CUDAException.h"
#include "Shapes.cuh"
#include "Random.cuh"

using namespace PGA::Shapes;

namespace PGA
{
	namespace Grid
	{
		namespace Device
		{
			__device__ unsigned int columns = 1;
			__device__ unsigned int rows = 1;
			__device__ unsigned int numElements = 1;

		}

		namespace Host
		{
			unsigned int columns = 1;
			unsigned int rows = 1;
			unsigned int numElements = 1;

		}

		namespace GlobalVars
		{
			__host__ __device__ unsigned int getColumns()
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				return Device::columns;
#else
				return Host::columns;
#endif
			}

			__host__ __device__ unsigned int getRows()
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				return Device::rows;
#else
				return Host::rows;
#endif
			}

			__host__ __device__ unsigned int getNumElements()
			{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				return Device::numElements;
#else
				return Host::numElements;
#endif
			}

			__host__ void __setGridSize(unsigned int columns, unsigned int rows)
			{
				Host::columns = columns;
				Host::rows = rows;
				PGA_CUDA_checkedCall(cudaMemcpyToSymbol(Device::columns, &columns, sizeof(unsigned int)));
				PGA_CUDA_checkedCall(cudaMemcpyToSymbol(Device::rows, &rows, sizeof(unsigned int)));
			}

			__host__ void __setNumElements(unsigned int numElements)
			{
				Host::numElements = numElements;
				PGA_CUDA_checkedCall(cudaMemcpyToSymbol(Device::numElements, &numElements, sizeof(unsigned int)));
			}

			__host__ void setGridSize(unsigned int columns, unsigned int rows, bool fill)
			{
				__setGridSize(columns, rows);
				if (fill)
				{
					__setNumElements(Host::rows * Host::columns);
				}
			}

			__host__ void setNumElements(unsigned int numElements, bool adapt)
			{
				unsigned int maxNumElements = Host::columns * Host::rows;
				if (adapt)
				{
					auto side = static_cast<unsigned int>(ceil(sqrt(static_cast<float>(numElements))));
					__setGridSize(side, side);
				}
				else
				{
					if (numElements > maxNumElements)
					{
						numElements = maxNumElements;
					}
				}
				__setNumElements(numElements);
			}

			__host__ void getGridSize(unsigned int& rows, unsigned int& columns)
			{
				rows = Host::rows;
				columns = Host::columns;
			}

		}

		template <typename ShapeT, int OperatorCodeT = -1>
		struct DefaultAxiomTraits
		{
			__host__ __device__ __inline__ static ShapeT getShape()
			{
				return ShapeT();
			}

			static const bool StaticDispatch = (OperatorCodeT > -1);
			static const int OperatorCode = OperatorCodeT;

			// NOTE: if StaticDispatchT evaluates to false, then AxiomTraitsT must declare getEntryIndex(..)
			__host__ __device__ __inline__ static int getEntryIndex()
			{
				return -1;
			}

			__host__ __device__ __inline__ static bool isEnabled(int axiomId)
			{
				return true;
			}

			__host__ __device__ __inline__ static math::float3 getSize(const ShapeT& shape)
			{
				return math::float3(1, 1, 1);
			}

			__host__ __device__ __inline__ static math::float3 getPosition(int row, int column, const ShapeT& shape)
			{
				math::float3 size = shape.getSize();
				return math::float3(column * (size.x * 1.5f), size.y * 0.5f, row * (size.z * 1.5f));
			}

		};

		namespace
		{
			template <typename AxiomTraitsT>
			struct AxiomDispatchingMethodSelector
			{
			private:
				template <bool StaticDispatchT, int DummyT = 0>
				struct Selector
				{
					template <typename SymbolManagerT, typename ShapeT, typename QueueT>
					__host__ __device__ __inline__ static void dispatchAxiom(ShapeT& shape, QueueT* queue)
					{
						SymbolManagerT::template dispatchAxiom<AxiomTraitsT::OperatorCode>(shape, queue);
					}

				};

				template <int DummyT>
				struct Selector < false, DummyT >
				{
					template <typename SymbolManagerT, typename ShapeT, typename QueueT>
					__host__ __device__ __inline__ static void dispatchAxiom(ShapeT& shape, QueueT* queue)
					{
						SymbolManagerT::dispatchAxiom(shape, AxiomTraitsT::getEntryIndex(), queue);
					}

				};

			public:
				typedef Selector<AxiomTraitsT::StaticDispatch> Result;

			};

			template <
				template <unsigned int> class AxiomTraitsT,
				unsigned int LengthT, 
				unsigned int Index = 0
			>
			struct ForEachStartingAxiom
			{
			private:
				typedef AxiomTraitsT<Index> CurrentAxiom;

			public:
				template <typename SymbolManagerT, typename QueueT>
				__host__ __device__ __inline__ static void generate(int id, int axiomId, int row, int column, QueueT* queue)
				{
					if (axiomId == Index)
					{
						if (CurrentAxiom::isEnabled(axiomId))
						{
							auto shape = CurrentAxiom::getShape();
							float seed = PGA::GlobalVars::getSeed(id);
							shape.setSeed(seed);
							shape.setCustomAttribute(seed);
							shape.setSize(CurrentAxiom::getSize(shape));
							shape.setPosition(CurrentAxiom::getPosition(row, column, shape));
							AxiomDispatchingMethodSelector<CurrentAxiom>::Result::template dispatchAxiom<SymbolManagerT>(shape, queue);
						}
					}
					else
					{
						ForEachStartingAxiom<AxiomTraitsT, LengthT, Index + 1>::template generate<SymbolManagerT>(id, axiomId, row, column, queue);
					}
				}

			};

			template <
				template <unsigned int> class AxiomTraitsT,
				unsigned int LengthT
			>
			struct ForEachStartingAxiom < AxiomTraitsT, LengthT, LengthT >
			{
				template <typename SymbolManagerT, typename QueueT>
				__host__ __device__ __inline__ static void generate(int id, int axiomId, int row, int column, QueueT* queue) {}

			};

		}

		template <
			template <unsigned int> class StartingAxiomTraitsT,
			unsigned int NumStartingAxioms,
			unsigned int Rows,
			unsigned int Columns,
			int _NumElements = -1
		>
		struct Static
		{
			static const unsigned int MaxNumElements = (Rows * Columns);
			static_assert((_NumElements < 0 || (_NumElements > static_cast<int>(MaxNumElements))), "Number of elements > max. number of elements for the grid");

			static const unsigned int NumElements = (_NumElements > 0) ? static_cast<unsigned int>(_NumElements) : MaxNumElements;

			__host__ static unsigned int getNumAxioms()
			{
				return NumStartingAxioms * NumElements;
			}

			template <typename SymbolManagerT, typename QueueT>
			__host__ __device__ __inline__ static void generateAxiom(int id, QueueT* queue)
			{
				int axiomId = id % NumStartingAxioms;
				int groupId = id / NumStartingAxioms;
				int column = groupId % Columns;
				int row = groupId / Rows;
				ForEachStartingAxiom<StartingAxiomTraitsT, NumStartingAxioms>::template generate<SymbolManagerT>(id, axiomId, column, row, queue);
			}

		};

		// NOTE: although we can create multiple instances of PGA with Grid::Dynamic as AxiomGenerator,
		// they should not be used *simultaneously* because they make use of the same global/device variables
		// (Host/Device::rows and Host/Device::columns)
		template <
			template <unsigned int> class StartingAxiomTraitsT,
			unsigned int NumStartingAxioms
		>
		struct Dynamic
		{
			__host__ static unsigned int getNumAxioms()
			{
				return NumStartingAxioms * Host::numElements;
			}

			template <typename SymbolManagerT, typename QueueT>
			__host__ __device__ __inline__ static void generateAxiom(int id, QueueT* queue)
			{
				int axiomId = id % NumStartingAxioms;
				int groupId = id / NumStartingAxioms;
				int column = groupId % GlobalVars::getColumns();
				int row = groupId / GlobalVars::getRows();
				ForEachStartingAxiom<StartingAxiomTraitsT, NumStartingAxioms>::template generate<SymbolManagerT>(id, axiomId, column, row, queue);
			}

		};

	};

}
