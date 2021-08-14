#pragma once

#include <cuda_runtime_api.h>

#include <exception>
#include <string>

namespace PGA
{
	namespace CUDA
	{
		class Exception : public std::exception
		{
		private:
			static std::string getErrorString(cudaError error, const char* file, int line)
			{
				return std::string(file) + 
					'(' + std::to_string(static_cast<long long>(line)) + "): " + cudaGetErrorString(error);
			}

		public:
			Exception(cudaError error, const char* file, int line) : std::exception(getErrorString(error, file, line).c_str())
			{
			}

			Exception(cudaError error) : std::exception(cudaGetErrorString(error))
			{
			}

		};
		
		inline void checkError(cudaError error, const char* file, int line)
		{
			if (error != cudaSuccess)
				throw PGA::CUDA::Exception(error, file, line);
		}

		inline void checkError(const char* file, int  line)
		{
			checkError(cudaGetLastError(), file, line);
		}

		inline void checkError()
		{
			cudaError error = cudaGetLastError();
			if (error != cudaSuccess)
				throw PGA::CUDA::Exception(error);
		}
	  
	}

}

#define PGA_CUDA_checkedCall(__call) PGA::CUDA::checkError(__call, __FILE__, __LINE__)
#define PGA_CUDA_uncheckedCall(__call) __call; cudaGetLastError();
#define PGA_CUDA_checkError() PGA::CUDA::checkError(__FILE__, __LINE__)