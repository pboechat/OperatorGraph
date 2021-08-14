#pragma once

#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
#include "Parameters.cuh"
#include "ShapeGenerator.cuh"
#include "Shapes.cuh"
#include "Symbol.cuh"
#include "TStdLib.h"

#include <cuda_runtime_api.h>
#include <math/matrix.h>
#include <math/vector.h>

#include <cstdio>
#include <string>

namespace PGA
{
	namespace Operators
	{
		template <bool ParallelT, unsigned int GenFuncIdxT = 0, typename TerminalIdxT = Parameters::Scalar<0>, typename Attr1T = Parameters::Scalar<0>, typename Attr2T = Parameters::Scalar<0>, typename Attr3T = Parameters::Scalar<0>, typename Attr4T = Parameters::Scalar<0>, typename Attr5T = Parameters::Scalar<0> >
		class Generate;

		template <unsigned int GenFuncIdxT, typename TerminalIdxT, typename Attr1T, typename Attr2T, typename Attr3T, typename Attr4T, typename Attr5T>
		class Generate < false, GenFuncIdxT, TerminalIdxT, Attr1T, Attr2T, Attr3T, Attr4T, Attr5T >
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				ContextT::Application::template dispatchTerminal<GenFuncIdxT>(*symbol, TerminalIdxT::eval(symbol), symbol->getCustomAttribute(), Attr1T::eval(symbol), Attr2T::eval(symbol), Attr3T::eval(symbol), Attr4T::eval(symbol), Attr5T::eval(symbol));
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "Generate";
			}

			__host__ __inline__ static std::string toString()
			{
				return "Generate<false, " + std::to_string(GenFuncIdxT) + ">";
			}

		};

		template <unsigned int GenFuncIdxT, typename TerminalIdxT, typename Attr1T, typename Attr2T, typename Attr3T, typename Attr4T, typename Attr5T>
		class Generate < true, GenFuncIdxT, TerminalIdxT, Attr1T, Attr2T, Attr3T, Attr4T, Attr5T >
		{
		public:
			template <typename ContextT, unsigned int NumThreadsT, typename QueueT, typename ShapeT>
			__host__ __device__ __inline__ static void execute(int threadId, int numThreads, QueueT* queue, Symbol<ShapeT>* symbol, unsigned int* shared)
			{
				int localThreadId = threadId % NumThreadsT;
				ContextT::Application::template dispatchTerminalInParallel<GenFuncIdxT, ContextT, NumThreadsT>(localThreadId, *symbol, TerminalIdxT::eval(symbol), symbol->getCustomAttribute(), Attr1T::eval(symbol), Attr2T::eval(symbol), Attr3T::eval(symbol), Attr4T::eval(symbol), Attr5T::eval(symbol));
			}

			__host__ __device__ __inline__ static const char* name()
			{
				return "Generate";
			}

			__host__ __inline__ static std::string toString()
			{
				return "Generate<true, " + std::to_string(GenFuncIdxT) + ">";
			}

		};

	}

}
