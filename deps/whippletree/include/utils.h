
/*
* file created by    Markus Steinberger / steinberger ( at ) icg.tugraz.at
*
* modifications by
*/

#ifndef SOFTSHELL_TOOLS_UTILS_INCLUDED
#define SOFTSHELL_TOOLS_UTILS_INCLUDED


#include <string>
//#include <sstream>
#include <stdexcept>
#include <cuda_runtime_api.h>


namespace Softshell
{
    class error : public std::runtime_error
    {
    private:
      static __host__ std::string genErrorString(cudaError error, const char* file, int line)
      {
        //std::ostringstream msg;
        //msg << file << '(' << line << "): error: " << cudaGetErrorString(error);
        //return msg.str();
        return std::string(file) + '(' + std::to_string(static_cast<long long>(line)) + "): error: " + cudaGetErrorString(error);
      }
    public:
      __host__ error(cudaError error, const char* file, int line)
        : runtime_error(genErrorString(error, file, line))
      {
      }

      __host__ error(cudaError error)
        : runtime_error(cudaGetErrorString(error))
      {
      }

      __host__ error(const std::string& msg)
        : runtime_error(msg)
      {
      }
    };
  inline __host__ void checkError(cudaError error, const char* file, int line)
  {
#if defined(_DEBUG) || defined(NDEBUG)
    if (error != cudaSuccess)
      throw Softshell::error(error, file, line);
#endif
  }

  inline __host__ void checkError(const char* file, int  line)
  {
    checkError(cudaGetLastError(), file, line);
  }

  inline __host__ void checkError()
  {
#if defined(_DEBUG) || defined(NDEBUG)
    cudaError error = cudaGetLastError();
    if (error != cudaSuccess)
      throw Softshell::error(error);
#endif
  }

#define CUDA_CHECKED_CALL(call) Softshell::checkError(call, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() Softshell::checkError(__FILE__, __LINE__)
#define CUDA_IGNORE_CALL(call) call; cudaGetLastError();
}


#endif  // SOFTSHELL_TOOLS_UTILS_INCLUDED
