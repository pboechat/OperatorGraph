#pragma once

#include <cstdint>

namespace PGA
{
	namespace Instrumentation
	{
		class PackUtils
		{
		private:
			template <unsigned int start, unsigned int length>
			static std::uint64_t unpackField(std::uint64_t v)
			{
				return (v >> start) & ~(~0ULL << length);
			}

			template <unsigned int start, unsigned int length>
			static std::uint32_t unpackField(std::uint32_t v)
			{
				return (v >> start) & ~(~0UL << length);
			}

			template <int i>
			static const std::uint64_t& get_element(const void* data)
			{
				return *(static_cast<const std::uint64_t*>(data) + i);
			}

		public:
			PackUtils() = delete;

			static unsigned long long unpack_t0(const unsigned char* data)
			{
				return get_element<0>(data);
			}

			static unsigned long long unpack_t1(const unsigned char* data)
			{
				return get_element<1>(data);
			}

			static unsigned int unpack_pid(const unsigned char* data)
			{
				return static_cast<int>(unpackField<53, 11>(get_element<2>(data)));
			}

			static unsigned int unpack_mpid(const unsigned char* data)
			{
				return static_cast<int>(unpackField<44, 8>(get_element<2>(data)));
			}

			static unsigned int unpack_active_threads(const unsigned char* data)
			{
				return static_cast<int>(unpackField<32, 12>(get_element<2>(data)));
			}

			static unsigned int unpack_overhead_counter(const unsigned char* data)
			{
				return static_cast<int>(unpackField<0, 32>(get_element<2>(data)));
			}

			static unsigned int unpack_dequeue_time(const unsigned char* data)
			{
				return static_cast<int>(unpackField<32, 32>(get_element<3>(data)));
			}

			static unsigned int unpack_num_enqueue_stats(const unsigned char* data)
			{
				return static_cast<int>(unpackField<0, 8>(get_element<3>(data)));
			}

			static unsigned int unpack_enqueue_stat_pid(const unsigned char* data, int i)
			{
				return unpackField<21, 11>(*(reinterpret_cast<const std::uint32_t*>(data) + 8 + i));
			}

			static unsigned int unpack_enqueue_stat_count(const unsigned char* data, int i)
			{
				return unpackField<0, 21>(*(reinterpret_cast<const std::uint32_t*>(data) + 8 + i));
			}

			static const unsigned char* next_element(const unsigned char* data, int num_enqueue_stats)
			{
				return data + 32 + (num_enqueue_stats + 3) / 4 * 16;
			}

		};

	}

}
