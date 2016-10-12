#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>

#include <pga/core/TStdLib.h>
#include <pga/core/Grid.cuh>
#include <pga/core/CUDAException.h>

namespace SceneUtils
{
	//////////////////////////////////////////////////////////////////////////
	const unsigned int MaxNumAttributes = 10;

	namespace Device
	{
		__device__ unsigned int attributes[MaxNumAttributes];

	}

	namespace Host
	{
		unsigned int attributes[MaxNumAttributes];

	}

	namespace GlobalVars
	{
		__host__ __device__ unsigned int getAttribute(unsigned int i)
		{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			return Device::attributes[i];
#else
			return Host::attributes[i];
#endif
		}

	}

	template <unsigned int NumAttributesT>
	struct AttributeSceneController
	{
	public:
		static const unsigned int _NumAttributes = T::Min<NumAttributesT, MaxNumAttributes>::Result;
		static const bool HasAttributes = true;

	private:
		static void updateAttributes()
		{
			CUDA_CHECKED_CALL(cudaMemcpyToSymbol(Device::attributes, &Host::attributes, _NumAttributes * sizeof(unsigned int)));
		}

	public:
		__host__ __device__ __inline__ static unsigned int getAttribute(unsigned int attributeIndex)
		{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			return Device::attributes[attributeIndex];
#else
			return Host::attributes[attributeIndex];
#endif
		}

		static void setAttribute(unsigned int attributeIndex, int attributeValue)
		{
			if (attributeIndex >= _NumAttributes || attributeValue < 0)
			{
				return;
			}
			Host::attributes[attributeIndex] = attributeValue;
			updateAttributes();
		}

		static void incrementAttribute(unsigned int attributeIndex)
		{
			setAttribute(attributeIndex, static_cast<int>(Host::attributes[attributeIndex]) + 1);
		}

		static void decrementAttribute(unsigned int attributeIndex)
		{
			setAttribute(attributeIndex, static_cast<int>(Host::attributes[attributeIndex]) - 1);
		}

	};

	//////////////////////////////////////////////////////////////////////////
	// FIXME: bad design!
	namespace GriddedSceneGlovalVars
	{
		unsigned int rows = 1;
		unsigned int columns = 1;
		unsigned int numElements = 1;

	}

	template <unsigned int MaxRowsT, unsigned int MaxColumnsT>
	struct GriddedSceneController
	{
	private:
		static const unsigned int _MaxRows = MaxRowsT;
		static const unsigned int _MaxColumns = MaxColumnsT;
		static const unsigned int _MaxNumElements = _MaxRows * _MaxColumns;

	public:
		static const bool IsGridded = true;

		static void incrementNumElements()
		{
			if (++GriddedSceneGlovalVars::numElements > _MaxNumElements)
			{
				GriddedSceneGlovalVars::numElements = _MaxNumElements;
			}
			PGA::Grid::GlobalVars::setNumElements(GriddedSceneGlovalVars::numElements, true);
			PGA::Grid::GlobalVars::getGridSize(GriddedSceneGlovalVars::rows, GriddedSceneGlovalVars::columns);
		}

		static void decrementNumElements()
		{
			if (GriddedSceneGlovalVars::numElements == 1)
			{
				return;
			}
			PGA::Grid::GlobalVars::setNumElements(--GriddedSceneGlovalVars::numElements, true);
			PGA::Grid::GlobalVars::getGridSize(GriddedSceneGlovalVars::rows, GriddedSceneGlovalVars::columns);
		}

		static void incrementGridSize()
		{
			if (++GriddedSceneGlovalVars::rows > _MaxRows)
			{
				GriddedSceneGlovalVars::rows = _MaxRows;
			}

			if (++GriddedSceneGlovalVars::columns > _MaxColumns)
			{
				GriddedSceneGlovalVars::columns = _MaxColumns;
			}

			PGA::Grid::GlobalVars::setGridSize(GriddedSceneGlovalVars::rows, GriddedSceneGlovalVars::columns, true);
			GriddedSceneGlovalVars::numElements = PGA::Grid::GlobalVars::getNumElements();
		}

		static void decrementGridSize()
		{
			if (GriddedSceneGlovalVars::rows > 1)
			{
				GriddedSceneGlovalVars::rows--;
			}

			if (GriddedSceneGlovalVars::columns > 1)
			{
				GriddedSceneGlovalVars::columns--;
			}

			PGA::Grid::GlobalVars::setGridSize(GriddedSceneGlovalVars::rows, GriddedSceneGlovalVars::columns, true);
			GriddedSceneGlovalVars::numElements = PGA::Grid::GlobalVars::getNumElements();
		}

		static unsigned int getNumElements()
		{
			return GriddedSceneGlovalVars::numElements;
		}

		static void setNumElements(unsigned int _numElements)
		{
			if (_numElements == 0)
			{
				// NOTE: min. num. elements for grid is always 1
				_numElements = 1;
			}
			else if (_numElements > _MaxNumElements)
			{
				_numElements = _MaxNumElements;
			}
			GriddedSceneGlovalVars::numElements = _numElements;
			PGA::Grid::GlobalVars::setNumElements(GriddedSceneGlovalVars::numElements, true);
			PGA::Grid::GlobalVars::getGridSize(GriddedSceneGlovalVars::rows, GriddedSceneGlovalVars::columns);
		}

		static void maximizeNumElements()
		{
			GriddedSceneGlovalVars::numElements = _MaxNumElements;
			PGA::Grid::GlobalVars::setNumElements(GriddedSceneGlovalVars::numElements, true);
			PGA::Grid::GlobalVars::getGridSize(GriddedSceneGlovalVars::rows, GriddedSceneGlovalVars::columns);
		}

		static unsigned int getRows()
		{
			return GriddedSceneGlovalVars::rows;
		}

		static unsigned int getColumns()
		{
			return GriddedSceneGlovalVars::columns;
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template <typename ControllerT>
	struct SceneInitialization
	{
	private:
		template <bool IsInitializableT /* false */, int DummyT = 0>
		struct Switch
		{
			static void initialize()
			{
			}

		};

		template <int DummyT>
		struct Switch < true, DummyT >
		{
			static void initialize()
			{
				ControllerT::initialize();
			}

		};

	public:
		static void initialize()
		{
			Switch<ControllerT::IsInitializable>::initialize();
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template <typename ControllerT>
	struct SceneGrid
	{
	private:
		template <bool IsGrided /* false */, int DummyT = 0>
		struct Switch
		{
			static void incrementNumElements()
			{
			}

			static void decrementNumElements()
			{
			}

			static void incrementGridSize()
			{
			}

			static void decrementGridSize()
			{
			}

			static void setNumElements(unsigned int numElements)
			{
			}

			static unsigned int getNumElements()
			{
				return 1;
			}

			static void maximizeNumElements()
			{
			}

		};

		template <int DummyT>
		struct Switch < true, DummyT >
		{
			static void incrementNumElements()
			{
				ControllerT::incrementNumElements();
			}

			static void decrementNumElements()
			{
				ControllerT::decrementNumElements();
			}

			static void incrementGridSize()
			{
				ControllerT::incrementGridSize();
			}

			static void decrementGridSize()
			{
				ControllerT::decrementGridSize();
			}

			static void setNumElements(unsigned int numElements)
			{
				ControllerT::setNumElements(numElements);
			}

			static unsigned int getNumElements()
			{
				return ControllerT::getNumElements();
			}

			static void maximizeNumElements()
			{
				ControllerT::maximizeNumElements();
			}

		};

	public:
		static void incrementNumElements()
		{
			Switch<ControllerT::IsGridded>::incrementNumElements();
		}

		static void decrementNumElements()
		{
			Switch<ControllerT::IsGridded>::decrementNumElements();
		}

		static void incrementGridSize()
		{
			Switch<ControllerT::IsGridded>::incrementGridSize();
		}

		static void decrementGridSize()
		{
			Switch<ControllerT::IsGridded>::decrementGridSize();
		}

		static void setNumElements(unsigned int numElements)
		{
			Switch<ControllerT::IsGridded>::setNumElements(numElements);
		}

		static unsigned int getNumElements()
		{
			return Switch<ControllerT::IsGridded>::getNumElements();
		}

		static void maximizeNumElements()
		{
			Switch<ControllerT::IsGridded>::maximizeNumElements();
		}


	};

	//////////////////////////////////////////////////////////////////////////
	template <typename ControllerT>
	struct SceneAttributes
	{
	private:
		template <bool HasAttributeT /* false */, int DummyT = 0>
		struct Switch
		{
			static void increment(unsigned int attributeIndex)
			{
			}

			static void decrement(unsigned int attributeIndex)
			{
			}

			static void setAttribute(unsigned int attributeIndex, unsigned int attributeValue)
			{
			}

		};

		template <int DummyT>
		struct Switch < true, DummyT >
		{
			static void increment(unsigned int attributeIndex)
			{
				ControllerT::incrementAttribute(attributeIndex);
			}

			static void decrement(unsigned int attributeIndex)
			{
				ControllerT::decrementAttribute(attributeIndex);
			}

			static void setAttribute(unsigned int attributeIndex, unsigned int attributeValue)
			{
				ControllerT::setAttribute(attributeIndex, attributeValue);
			}

		};

	public:
		static void increment(unsigned int attributeIndex)
		{
			Switch<ControllerT::HasAttributes>::increment(attributeIndex);
		}

		static void decrement(unsigned int attributeIndex)
		{
			Switch<ControllerT::HasAttributes>::decrement(attributeIndex);
		}

		static void setAttribute(unsigned int attributeIndex, unsigned int attributeValue)
		{
			Switch<ControllerT::HasAttributes>::setAttribute(attributeIndex, attributeValue);
		}

	};

	//////////////////////////////////////////////////////////////////////////
	template <typename ConfiguratorT>
	struct SceneConfiguration
	{
	private:
		template <bool IsFile /* false */, int DummyT = 0>
		struct Switch
		{
			static std::string getConfigurationString()
			{
				return ConfiguratorT::value();
			}

		};

		template <int DummyT>
		struct Switch < true, DummyT >
		{
			static std::string getConfigurationString()
			{
				std::ifstream file(ConfiguratorT::value());
				if (!file.good())
					throw std::runtime_error("configuration file not found: " + ConfiguratorT::value());
				return std::string((std::istreambuf_iterator<char>(file)), 
					std::istreambuf_iterator<char>());
			}

		};

	public:
		static std::string getConfigurationString()
		{
			return Switch<ConfiguratorT::IsFile>::getConfigurationString();
		}

	};

}
