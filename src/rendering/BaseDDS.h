#pragma once

#include <iostream>

namespace PGA
{
	namespace Rendering
	{
		namespace BaseDDS
		{
			typedef unsigned int uint32_t;
			typedef unsigned int uint32_t;

#pragma pack(push, 1)
			const uint32_t DDSMagic = 0x20534444; // "DDS "

			struct DDSPixelFormat
			{
				uint32_t dwSize;
				uint32_t dwFlags;
				uint32_t dwFourCC;
				uint32_t dwRGBBitCount;
				uint32_t dwRBitMask;
				uint32_t dwGBitMask;
				uint32_t dwBBitMask;
				uint32_t dwABitMask;
			};

			struct DDSHeader
			{
				uint32_t dwSize;
				uint32_t dwFlags;
				uint32_t dwHeight;
				uint32_t dwWidth;
				uint32_t dwLinearSize;
				uint32_t dwDepth;
				uint32_t dwMipMapCount;
				uint32_t dwReserved1[11];
				DDSPixelFormat ddpf;
				uint32_t dwCaps;
				uint32_t dwCaps2;
				uint32_t dwCaps3;
				uint32_t dwCaps4;
				uint32_t dwReserved2;
			};

#pragma pack(pop)

			const unsigned long DDSD_CAPS = 0x00000001UL;
			const unsigned long DDSD_HEIGHT = 0x00000002UL;
			const unsigned long DDSD_WIDTH = 0x00000004UL;
			const unsigned long DDSD_PITCH = 0x00000008UL;
			const unsigned long DDSD_PIXELFORMAT = 0x00001000UL;
			const unsigned long DDSD_MIPMAPCOUNT = 0x00020000UL;
			const unsigned long DDSD_LINEARSIZE = 0x00080000UL;
			const unsigned long DDSD_DEPTH = 0x00800000UL;

			const unsigned long DDPF_RGB = 0x00000040UL;
			const unsigned long DDPF_FOURCC = 0x00000004UL;
			const unsigned long DDPF_ALPHAPIXELS = 0x00000001UL;
			const unsigned long DDPF_RGBA = DDPF_RGB | DDPF_ALPHAPIXELS;
			const unsigned long DDPF_LUMINANCE = 0x00020000UL;
			const unsigned long DDPF_ALPHA = 0x00000002UL;
			const unsigned long DDPF_BUMPDUDV = 0x00080000UL;

			const unsigned long DDSCAPS_TEXTURE = 0x00001000UL;
			const unsigned long DDSCAPS_COMPLEX = 0x00000008UL;
			const unsigned long DDSCAPS_MIPMAP = 0x00400000UL;

			const unsigned long DDSCAPS2_CUBEMAP = 0x00000200UL;

#ifndef MAKEFOURCC
#define MAKEFOURCC(ch0, ch1, ch2, ch3) \
		((uint32_t)(BYTE)(ch0) | ((uint32_t)(BYTE)(ch1) << 8) | \
		((uint32_t)(BYTE)(ch2) << 16) | ((uint32_t)(BYTE)(ch3) << 24 ))
#endif

#define ISBITMASK(r,g,b,a) ( ddpf.dwRBitMask == r && ddpf.dwGBitMask == g && ddpf.dwBBitMask == b && ddpf.dwABitMask == a )

#define FOURCC_DXT1	(MAKEFOURCC('D','X','T','1'))

			std::istream& readSurface(std::istream& file, char* buffer, int width, int pixelSize);
			std::istream& readSurface(std::istream& file, char* buffer, int width, int height, int pixelSize);
			std::ostream& writeSurface(std::ostream& file, const char* buffer, int width, int height, int pixel_size);
			bool isDDS(std::istream& input);

		}

	}
}
