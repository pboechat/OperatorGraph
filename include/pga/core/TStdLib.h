#pragma once

#include <cstdio>

namespace T
{
	template <unsigned int A, unsigned int B>
	struct Max
	{
		static const unsigned int Result = (A > B) ? A : B;

	};

	template <unsigned int A, unsigned int B>
	struct Min
	{
		static const unsigned int Result = (A < B) ? A : B;

	};

	template<int X, unsigned int Y>
	struct Power
	{
		static const int Result = X * Power<X, Y - 1>::Result;

	};

	template<int X>
	struct Power < X, 0 >
	{
		static const int Result = 1;

	};

	template <unsigned int Value, unsigned int Exponent>
	struct NextPowerOfTwo_Try;

	template <bool B, unsigned int Value, unsigned int Exponent>
	struct NextPowerOfTwo_Select
	{
		static const int Result = (1 << Exponent);

	};

	template <unsigned int Value, unsigned int Exponent>
	struct NextPowerOfTwo_Select < false, Value, Exponent >
	{
		static const int Result = NextPowerOfTwo_Try<Value, Exponent + 1>::Result;
	};

	template <unsigned int Value, unsigned int Exponent>
	struct NextPowerOfTwo_Try
	{
		static const int Result = NextPowerOfTwo_Select< (1 << Exponent >= Value), Value, Exponent >::Result;

	};

	template <unsigned int Value>
	struct NextPowerOfTwo
	{
		static const int Result = NextPowerOfTwo_Try<Value, 1>::Result;

	};

	template <bool Expression, typename A>
	struct EnableIf
	{
	};

	template <typename A>
	struct EnableIf < true, A >
	{
		typedef A Result;

	};

	template <bool Expression, typename A, typename B = void>
	struct If
	{
		typedef A Result;

	};

	template <typename A, typename B>
	struct If < false, A, B >
	{
		typedef B Result;

	};

	template <typename... Types>
	struct EqualTypes;

	template <typename TypeA, typename TypeB>
	struct EqualTypes < TypeA, TypeB >
	{
		static const bool Result = false;

	};

	template <typename TypeA>
	struct EqualTypes < TypeA, TypeA >
	{
		static const bool Result = true;

	};

	template <typename TypeA, typename TypeB, typename... TypesRemainder>
	struct EqualTypes < TypeA, TypeB, TypesRemainder... >
	{
		static const bool Result = false;

	};

	template <typename TypeA, typename... TypesRemainder>
	struct EqualTypes < TypeA, TypeA, TypesRemainder... >
	{
		static const bool Result = EqualTypes<TypesRemainder...>::Result;

	};

	template <typename A, typename B>
	struct IsConvertible
	{
	private:
		typedef char Small;
		class Big { char dummy[2]; };
		static Small Test(B);
		static Big Test(...);
		static A MakeA();

	public:
		static const bool Result = sizeof(Test(MakeA())) == sizeof(Small);

	};

	template <typename KeyT, typename ValueT>
	struct Pair
	{
		typedef KeyT Key;
		typedef ValueT Value;

	};

	template <typename _First, typename _Second, typename _Third>
	struct Triple
	{
		// PERFORMANCE IMPROVEMENT?
		//typedef _First First;
		//typedef _Second Second;
		//typedef _Third Third;

	};

	template <unsigned int Index, typename Element, typename... Elements>
	struct _IndexOf;

	template <unsigned int Index, typename Element, typename Head, typename... Tail>
	struct _IndexOf < Index, Element, Head, Tail... >
	{
		static const int Result = _IndexOf<Index + 1, Element, Tail...>::Result;

	};

	template <unsigned int Index, typename Element, typename... Tail>
	struct _IndexOf < Index, Element, Element, Tail... >
	{
		static const int Result = static_cast<int>(Index);

	};

	template <unsigned int Index, typename Element>
	struct _IndexOf < Index, Element >
	{
		static const int Result = -1;

	};

	template <unsigned int StartingIndex, unsigned int Index, typename... Elements>
	struct _ItemAt;

	template <unsigned int CurrentIndex, unsigned int Index, typename Head, typename... Tail>
	struct _ItemAt < CurrentIndex, Index, Head, Tail... >
	{
		typedef typename _ItemAt<CurrentIndex + 1, Index, Tail...>::Result Result;

	};

	template <unsigned int Index, typename Head, typename... Tail>
	struct _ItemAt < Index, Index, Head, Tail... >
	{
		typedef Head Result;

	};

	template <typename Element, typename... Elements>
	struct _Contains;

	template <typename Element, typename Head, typename... Tail>
	struct _Contains < Element, Head, Tail... >
	{
		static const bool Result = _Contains<Element, Tail...>::Result;

	};

	template <typename Element, typename... Tail>
	struct _Contains < Element, Element, Tail... >
	{
		static const bool Result = true;

	};

	template <typename Element>
	struct _Contains < Element >
	{
		static const bool Result = false;

	};

	template <typename Key, typename... Elements>
	struct _ContainsKey;

	template <typename Key, typename HeadKey, typename HeadValue, typename... Tail>
	struct _ContainsKey < Key, Pair<HeadKey, HeadValue>, Tail... >
	{
		static const bool Result = _ContainsKey<Key, Tail...>::Result;

	};

	template <typename Key, typename HeadValue, typename... Tail>
	struct _ContainsKey < Key, Pair<Key, HeadValue>, Tail... >
	{
		static const bool Result = true;

	};

	template <typename Key>
	struct _ContainsKey < Key >
	{
		static const bool Result = false;

	};

	template <typename Key, typename... Elements>
	struct _Find;

	template <typename Key, typename HeadKey, typename HeadValue, typename... Tail>
	struct _Find < Key, Pair<HeadKey, HeadValue>, Tail... >
	{
		typedef typename _Find<Key, Tail...>::Result Result;

	};

	template <typename Key, typename HeadValue, typename... Tail>
	struct _Find < Key, Pair<Key, HeadValue>, Tail... >
	{
		typedef HeadValue Result;

	};

	template <typename... Elements>
	struct List
	{
		static const unsigned int Length = sizeof...(Elements);

		template <typename Element>
		struct IndexOf
		{
			static const int Result = _IndexOf<0, Element, Elements...>::Result;

		};

		template <unsigned int Index>
		struct ItemAt
		{
			static_assert(Index < Length, "index >= list length");
			typedef typename _ItemAt<0, Index, Elements...>::Result Result;

		};

		template <typename Element>
		struct Contains
		{
			static const bool Result = _Contains<Element, Elements...>::Result;

		};

		template <typename Key>
		struct ContainsKey
		{
			static const bool Result = _ContainsKey<Key, Elements...>::Result;

		};

		template <typename Key>
		struct Find
		{
			typedef typename _Find<Key, Elements...>::Result Result;

		};

	};

	// FIXME: STILL DOESN'T ACCEPT TYPES DERIVED FROM LIST!
	template <typename Element1, typename Element2, bool NoDuplicates = false>
	struct Append;

	template <typename Element1, typename Element2>
	struct Append < Element1, Element2, false >
	{
		typedef List<Element1, Element2> Result;

	};

	template <typename Element1, typename Element2>
	struct Append < Element1, Element2, true >
	{
		typedef List<Element1, Element2> Result;

	};

	template <typename Element>
	struct Append < Element, Element, true >
	{
		typedef Element Result;

	};

	template <typename Element1, typename Element2>
	struct Append < List<Element1>, List<Element2>, false >
	{
		typedef List<Element1, Element2> Result;

	};

	template <typename Element1, typename Element2>
	struct Append < List<Element1>, List<Element2>, true >
	{
		typedef List<Element1, Element2> Result;

	};

	template <typename Element, typename... Elements>
	struct Append < Element, List<Elements...>, false >
	{
		typedef List<Element, Elements...> Result;

	};

	template <typename Element, typename... Elements>
	struct Append < Element, List<Elements...>, true >
	{
		typedef typename If<_Contains<Element, List<Elements...> >::Result, List<Elements...>, List<Element, Elements...> >::Result Result;

	};

	template <typename Element, typename... Elements>
	struct Append < List<Elements...>, Element, false >
	{
		typedef List<Elements..., Element> Result;

	};

	template <typename Element, typename... Elements>
	struct Append < List<Elements...>, Element, true >
	{
		typedef typename If<_Contains<Element, List<Elements...> >::Result, List<Elements...>, List<Elements..., Element> >::Result Result;

	};

	template <typename... Elements1, typename... Elements2>
	struct Append < List<Elements1...>, List<Elements2...>, false >
	{
		typedef List < Elements1..., Elements2... > Result;

	};

	template <typename First1, typename... Elements1, typename First2, typename... Remainder2>
	struct Append < List<First1, Elements1...>, List<First2, Remainder2...>, true >
	{
	private:
		typedef typename Append< List<First1, Elements1...>, List<Remainder2...>, true >::Result TailIfTrue;
		typedef typename Append< List<First1, Elements1..., First2>, List<Remainder2...>, true >::Result TailIfFalse;

	public:
		typedef typename If<_Contains<First2, List<First1, Elements1...> >::Result, TailIfTrue, TailIfFalse >::Result Result;

	};

	template <typename... Elements>
	struct Append < List<Elements...>, List<Elements...>, true >
	{
		typedef List<Elements...> Result;

	};

	template <typename... Elements>
	struct Append < List<Elements...>, List<>, false >
	{
		typedef List<Elements...> Result;

	};

	template <typename... Elements>
	struct Append < List<Elements...>, List<>, true >
	{
		typedef List<Elements...> Result;

	};

	template <typename... Elements>
	struct Append < List<>, List<Elements...>, false >
	{
		typedef List<Elements...> Result;

	};

	template <typename... Elements>
	struct Append < List<>, List<Elements...>, true >
	{
		typedef List<Elements...> Result;

	};

	template <>
	struct Append < List<>, List<>, false >
	{
		typedef List<> Result;

	};

	template <>
	struct Append < List<>, List<>, true >
	{
		typedef List<> Result;

	};

	template <typename List, template <class> class Operation>
	struct ForEach;

	template <template <class> class Operation, typename First, typename... Remainder>
	struct ForEach < List<First, Remainder...>, Operation >
	{
		static void run()
		{
			Operation < First >::run();
			ForEach < List<Remainder...>, Operation >::run();
		}

	};

	template <template <class> class Operation>
	struct ForEach < List<>, Operation >
	{
		static void run()
		{
		}

	};

	template <typename Param>
	struct GetMacroTemplateParam;

	template <typename R, typename P1>
	struct GetMacroTemplateParam < R(P1) >
	{
		typedef P1 Result;

	};

	template <typename... Parameters>
	struct IsEnabled;

	template <typename SingleParameter>
	struct IsEnabled < SingleParameter >
	{
		static const bool Result = false;

	};

	template <typename First, typename... Remainder>
	struct IsEnabled < First, Remainder... >
	{
		static const bool Result = IsEnabled<First>::Result || IsEnabled<Remainder...>::Result;

	};

}
