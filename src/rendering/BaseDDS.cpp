#include "BaseDDS.h"

namespace PGA
{
	namespace Rendering
	{
		namespace BaseDDS
		{
			std::istream& readSurface(std::istream& file, char* buffer, int width, int pixelSize)
			{
				return file.read(reinterpret_cast<char*>(buffer), sizeof(char) * width * pixelSize);
			}

			std::istream& readSurface(std::istream& file, char* buffer, int width, int height, int pixelSize)
			{
				for (int y = height - 1; y >= 0; --y)
					file.read(reinterpret_cast<char*>(buffer + y * width * pixelSize), sizeof(char) * width * pixelSize);
				return file;
			}

			std::ostream& writeSurface(std::ostream& file, const char* buffer, int width, int height, int pixelSize)
			{
				for (int y = height - 1; y >= 0; --y)
					file.write(reinterpret_cast<const char*>((buffer + y * width * pixelSize)), sizeof(const char*) * width * pixelSize);
				return file;
			}

			bool isDDS(std::istream& input)
			{
				char magic_num[4];
				input.read(reinterpret_cast<char*>(magic_num), sizeof(char) * 4);
				if (std::strncmp(magic_num, "DDS ", 4) == 0)
					return true;
				input.seekg(0);
				return false;
			}

		}

	}

}


