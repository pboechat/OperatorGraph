#include <pga/rendering/D3DException.h>

#include <string>

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			std::string getErrorMessage(HRESULT hresult, const char* file, int line)
			{
				std::string errorStr;
				switch (hresult)
				{
				case D3D11_ERROR_FILE_NOT_FOUND:
					errorStr = "D3D11_ERROR_FILE_NOT_FOUND";
				case D3D11_ERROR_TOO_MANY_UNIQUE_STATE_OBJECTS:
					errorStr = "D3D11_ERROR_TOO_MANY_UNIQUE_STATE_OBJECTS";
				case D3D11_ERROR_TOO_MANY_UNIQUE_VIEW_OBJECTS:
					errorStr = "D3D11_ERROR_TOO_MANY_UNIQUE_VIEW_OBJECTS";
				case D3D11_ERROR_DEFERRED_CONTEXT_MAP_WITHOUT_INITIAL_DISCARD:
					errorStr = "D3D11_ERROR_DEFERRED_CONTEXT_MAP_WITHOUT_INITIAL_DISCARD";
				case E_FAIL:
					errorStr = "E_FAIL";
				case E_INVALIDARG:
					errorStr = "E_INVALIDARG";
				case E_OUTOFMEMORY:
					errorStr = "E_OUTOFMEMORY";
				default:
					errorStr = "unknown error code";
				}
				return std::string(file) +
					'(' + std::to_string(static_cast<long long>(line)) + "): " + errorStr;
			}

			Exception::Exception(HRESULT hresult, const char* file, int line) : std::exception(getErrorMessage(hresult, file, line).c_str())
			{
			}

		}

	}

}