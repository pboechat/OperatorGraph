#pragma once

#include <math/vector.h>

#include <cctype>
#include <functional>
#include <sstream>
#include <string>
#include <string.h>
#include <vector>

namespace PGA
{
	class StringUtils
	{
	public:
		StringUtils() = delete;

		static void replace(std::string& str, const std::string& from, const std::string& to)
		{
			size_t pos = str.find(from);
			if (pos == std::string::npos)
				return;
			str.replace(pos, from.length(), to);
		}

		static std::string replaceAll(const std::string& str, const std::string& search, const std::string& replace)
		{
			std::string copy(str);
			for (size_t pos = 0;; pos += replace.length())
			{
				pos = copy.find(search, pos);
				if (pos == std::string::npos) break;
				copy.erase(pos, search.length());
				copy.insert(pos, replace);
			}
			return copy;
		}

		static void replaceAll(std::string& str, const std::string& from, const std::string& to)
		{
			if (from.empty())
				return;
			size_t pos = 0;
			while ((pos = str.find(from, pos)) != std::string::npos)
			{
				str.replace(pos, from.length(), to);
				pos += to.length();
			}
		}

		static void split(const std::string &str, char delim, std::vector<std::string>& elems)
		{
			std::stringstream out(str);
			std::string item;
			while (std::getline(out, item, delim)) 
			{
				elems.push_back(item);
			}
		}

		static std::string ltrim(const std::string& str)
		{
			std::string copy(str);
			copy.erase(copy.begin(), std::find_if(copy.begin(), copy.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
			return copy;
		}

		static std::string rtrim(const std::string& str)
		{
			std::string copy(str);
			copy.erase(std::find_if(copy.rbegin(), copy.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), copy.end());
			return copy;
		}

		static std::string trim(const std::string& str)
		{
			return ltrim(rtrim(str));
		}

		static math::float2 parseVec2(const std::string& str)
		{
			std::vector<std::string> elems;
			split(str, ',', elems);
			if (elems.size() < 2)
				throw std::runtime_error("elems.size() < 2");
			auto pos = elems[0].find('(');
			if (pos == std::string::npos)
				pos = 0;
			else
				pos += 1;
			auto xStr = trim(elems[0].substr(pos, elems[0].length() - pos));
			float x = static_cast<float>(atof(xStr.c_str()));
			pos = elems[1].find(')');
			if (pos == std::string::npos)
				pos = elems[1].length();
			auto yStr = trim(elems[1].substr(0, pos));
			float y = static_cast<float>(atof(yStr.c_str()));
			return math::float2(x, y);
		}

		static math::float4 parseVec4(const std::string& str)
		{
			std::vector<std::string> elems;
			split(str, ',', elems);
			if (elems.size() < 2)
				throw std::runtime_error("elems.size() < 2");
			auto pos = elems[0].find('(');
			if (pos == std::string::npos)
				pos = 0;
			else
				pos += 1;
			auto xStr = trim(elems[0].substr(pos, elems[0].length() - pos));
			float x = static_cast<float>(atof(xStr.c_str()));
			auto yStr = trim(elems[1]);
			float y = static_cast<float>(atof(yStr.c_str()));
			auto zStr = trim(elems[2]);
			float z = static_cast<float>(atof(zStr.c_str()));
			pos = elems[3].find(')');
			if (pos == std::string::npos)
				pos = elems[3].length();
			auto wStr = trim(elems[3].substr(0, pos));
			float w = static_cast<float>(atof(wStr.c_str()));
			return math::float4(x, y, z, w);
		}

		static std::vector<math::float2>& parseVec2List(const std::string& str, std::vector<math::float2>& list)
		{
			std::vector<std::string> tokens;
			split(str, '(', tokens);
			for (auto& token : tokens)
			{
				if (token.empty())
					continue;
				list.emplace_back(parseVec2(token));
			}
			return list;
		}

	};

}