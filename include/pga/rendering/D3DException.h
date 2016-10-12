#pragma once

#include <exception>
#include <d3d11.h>

namespace PGA
{
	namespace Rendering
	{
		namespace D3D
		{
			class Exception : public std::exception
			{
			public:
				Exception(HRESULT hresult, const char* file, int line);

			};

			inline void checkError(HRESULT hresult, const char* file, int line)
			{
				if (FAILED(hresult))
					throw PGA::Rendering::D3D::Exception(hresult, file, line);
			}

		}

	}

}

#define PGA_Rendering_D3D_checkedCall(__call) PGA::Rendering::D3D::checkError(__call, __FILE__, __LINE__)
