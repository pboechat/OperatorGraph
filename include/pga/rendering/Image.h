#pragma once

#include <algorithm>
#include <memory>

namespace PGA
{
	namespace Rendering
	{
		template <typename T>
		class Image
		{
		private:
			std::unique_ptr<T[]> rawData;
			unsigned int width;
			unsigned int height;

		public:
			Image(unsigned int width, unsigned int height)
				: rawData(new T[width * height]),
				width(width),
				height(height)
			{
			}

			Image(const Image& s)
				: rawData(new T[s.width * s.height]),
				width(s.width),
				height(s.height)
			{
				std::copy(&s.rawData[0], &s.rawData[0] + width * height, &rawData[0]);
			}

			Image(Image&& s)
				: rawData(move(s.rawData)),
				width(s.width),
				height(s.height)
			{
			}

			Image& operator =(const Image& s)
			{
				width = s.width;
				height = s.height;
				std::unique_ptr<T[]> buffer(new T[width * height]);
				std::copy(&s.rawData[0], &s.rawData[0] + width * height, &buffer[0]);
				rawData = std::move(buffer);
				return *this;
			}

			Image& operator =(Image&& s)
			{
				width = s.width;
				height = s.height;
				rawData = std::move(s.rawData);
				return *this;
			}

			T operator ()(unsigned int x, unsigned int y) const { return rawData[y * width + x]; }
			T& operator ()(unsigned int x, unsigned int y) { return rawData[y * width + x]; }

			friend unsigned int width(const Image& img)
			{
				return img.width;
			}

			friend unsigned int height(const Image& img)
			{
				return img.height;
			}

			friend const T* rawData(const Image& img)
			{
				return &img.rawData[0];
			}

			friend T* rawData(Image& img)
			{
				return &img.rawData[0];
			}

		};

	}

}