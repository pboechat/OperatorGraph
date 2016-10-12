#pragma once

// NOTE: see FIXME at the end of the file

#include <cstdint>
#include <map>
#include <tuple>

namespace PGA
{
	namespace CPU
	{
		class BaseContext
		{
		private:
			static int currentThreadIdx;
			static std::map<std::pair<uintptr_t, unsigned int>, void*> shflRegisters;

		public:
			static void setUpWarp(unsigned int size)
			{
				currentThreadIdx = -1;
				shflRegisters.clear();
			}

			static void setCurrentThreadIdx(int idx)
			{
				currentThreadIdx = idx;
			}

			static unsigned int shfl(unsigned int& variable, unsigned int idx)
			{
				auto loc = reinterpret_cast<uintptr_t>(&variable);
				std::pair<uintptr_t, unsigned int> key(loc, idx);
				auto it = shflRegisters.find(key);
				if (idx == currentThreadIdx)
				{
					// FIXME: checking invariants
					if (it != shflRegisters.end())
						throw std::runtime_error("it != shflRegisters.end()");
					shflRegisters[key] = reinterpret_cast<void*>(variable);
					return variable;
				}
				else
				{
					// FIXME: checking invariants
					if (it == shflRegisters.end())
						throw std::runtime_error("it == shflRegisters.end()");
					return static_cast<unsigned int>(reinterpret_cast<uintptr_t>(it->second));
				}
			}

		};

		// FIXME: currently cannot be included more than once because of these static member-variables
		int BaseContext::currentThreadIdx = 0;
		std::map<std::pair<uintptr_t, unsigned int>, void*> BaseContext::shflRegisters;

	}

}