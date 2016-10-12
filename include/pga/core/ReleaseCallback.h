#pragma once

namespace PGA
{
	struct ReleaseCallback
	{
		template <typename T>
		void operator()(T* obj)
		{
			if (obj)
				obj->release();
		}
		
	};

}