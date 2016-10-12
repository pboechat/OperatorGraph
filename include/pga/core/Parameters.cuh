#pragma once

#include <cmath>
#include <string>
#include <stdexcept>
#include <cuda_runtime_api.h>

#include <math/vector.h>
#include <math/matrix.h>

#include "DebugFlags.h"
#include "GlobalConstants.h"
#include "GlobalVariables.cuh"
#include "Axis.h"
#include "RepeatMode.h"
#include "Shape.cuh"
#include "ParameterType.h"
#include "OperandType.h"
#include "OperationType.h"
#include "DispatchTableEntry.h"
#include "Random.cuh"
#include "Symbol.cuh"
#include "PackUtils.h"
#include "TStdLib.h"

namespace PGA
{
	namespace Parameters
	{
		//////////////////////////////////////////////////////////////////////////
		template <int ValueT, unsigned int NegativeOrderOfMagnitudeT = 0>
		class Scalar
		{
		public:
			static const unsigned int Length = 1;

			__host__ __device__ __inline__ static float getValue()
			{
				return ValueT / (float)T::Power<10, NegativeOrderOfMagnitudeT>::Result;
			}

			template <typename ShapeT>
			__host__ __device__ __inline__ static math::float2 toFloat2(const Symbol<ShapeT>* symbol)
			{
				return math::float2(getValue(), 0);
			}

			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return getValue();
			}

			__host__ __inline__ static DispatchTableEntry::Parameter toParameter()
			{
				DispatchTableEntry::Parameter parameter;
				parameter.type = ParameterType::PT_SCALAR;
				parameter.value[0] = getValue();
				return parameter;
			}

			__host__ __inline__ static void encode(float* b, unsigned int p)
			{
				b[p] = PackUtils::packOperand(ORT_SCALAR, getValue());
			}

			__host__ __inline__ static std::string toString()
			{
				return std::to_string(getValue());
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <int XT, int YT, unsigned int XNegativeOrderOfMagnitudeT = 3, unsigned int YNegativeOrderOfMagnitudeT = 3>
		class Vec2
		{
		public:
			static const unsigned int Length = 1;
			
			__host__ __device__ __inline__ static float getX()
			{
				return XT / (float)T::Power<10, XNegativeOrderOfMagnitudeT>::Result;
			}

			__host__ __device__ __inline__ static float getY()
			{
				return YT / (float)T::Power<10, YNegativeOrderOfMagnitudeT>::Result;
			}

			template <typename ShapeT>
			__host__ __device__ __inline__ static math::float2 toFloat2(const Symbol<ShapeT>* symbol)
			{
				return math::float2(getX(), getY());
			}

			__host__ __inline__ static DispatchTableEntry::Parameter toParameter()
			{
				DispatchTableEntry::Parameter parameter;
				parameter.type = ParameterType::PT_VEC2;
				parameter.value[0] = getX();
				parameter.value[1] = getY();
				return parameter;
			}

			__host__ __inline__ static void encode(float* b, unsigned int p)
			{
				// TODO:
				b[p] = PackUtils::packOperand(ORT_SCALAR, math::length2(math::float2(getX(), getY())));
			}

			__host__ __inline__ static std::string toString()
			{
				return "(" + std::to_string(getX()) + ", " + std::to_string(getY()) + ")";
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <PGA::Axis AxisT>
		class AxisParam : public Scalar< AxisT, 0 > {};

		//////////////////////////////////////////////////////////////////////////
		template <PGA::RepeatMode RepeatModeT>
		class RepeatModeParam : public Scalar< RepeatModeT, 0 > {};

		//////////////////////////////////////////////////////////////////////////
		template <unsigned int IndexT>
		struct ShapeAttr
		{
			static const unsigned int Length = 1;

			template <typename ShapeT>
			__host__ __device__  __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return symbol->at(IndexT);
			}

			__host__ __inline__ static DispatchTableEntry::Parameter toParameter()
			{
				DispatchTableEntry::Parameter parameter;
				parameter.value[0] = (float)IndexT;
				parameter.type = ParameterType::PT_SHAPE_ATTR;
				return parameter;
			}

			__host__ __inline__ static void encode(float* b, unsigned int p)
			{
				b[p] = PackUtils::packOperand(ORT_SHAPE_ATTR, (float)IndexT);
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <PGA::Axis AxisT>
		struct ShapePos : public ShapeAttr<(AxisT + 1) * 4 - 1>
		{
			__host__ __inline__ static std::string toString()
			{
				return "shape.pos[" + std::to_string(AxisT) + "]";
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <PGA::Axis AxisT, PGA::Axis ComponentT>
		struct ShapeRotation : public ShapeAttr<(ComponentT * 4) + AxisT>
		{
			__host__ __inline__ static std::string toString()
			{
				return "shape.rotation[" + std::to_string(Index) + "]";
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <PGA::Axis ComponentT>
		struct ShapeNormal : public ShapeRotation < PGA::Z, ComponentT > {};

		//////////////////////////////////////////////////////////////////////////
		template <PGA::Axis AxisT>
		struct ShapeSize : public ShapeAttr<(AxisT + 12)>
		{
			__host__ __inline__ static std::string toString()
			{
				return "shape.size[" + std::to_string(AxisT) + "]";
			}

		};

		//////////////////////////////////////////////////////////////////////////
		struct ShapeSeed : public ShapeAttr<15>
		{
			__host__ __inline__ static std::string toString()
			{
				return "shape.seed";
			}

		};

		//////////////////////////////////////////////////////////////////////////
		struct ShapeCustomAttribute : public ShapeAttr<16>
		{
			__host__ __inline__ static std::string toString()
			{
				return "shape.customAttr";
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <int MinT, int MaxT, unsigned int MinNegativeOrderOfMagnitudeT = 0, unsigned int MaxNegativeOrderOfMagnitudeT = 0>
		class Rand
		{
		public:
			static const unsigned int Length = 1;

			__host__ __device__ __inline__ static float getMin()
			{
				return MinT / (float)T::Power<10, MinNegativeOrderOfMagnitudeT>::Result;
			}

			__host__ __device__ __inline__ static float getMax()
			{
				return MaxT / (float)T::Power<10, MaxNegativeOrderOfMagnitudeT>::Result;
			}

			template <typename ShapeT>
			__host__ __device__ __inline__ static math::float2 toFloat2(const Symbol<ShapeT>* symbol)
			{
				return math::float2(getMin(), getMax());
			}

			template <typename ShapeT>
			__host__ __device__  __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return Random::range(symbol->getSeed(), getMin(), getMax());
			}

			__host__ __inline__ static DispatchTableEntry::Parameter toParameter()
			{
				DispatchTableEntry::Parameter parameter;
				parameter.type = ParameterType::PT_RAND;
				parameter.value[0] = getMin();
				parameter.value[1] = getMax();
				return parameter;
			}

			__host__ __inline__ static void encode(float* b, unsigned int p)
			{
				b[p] = PackUtils::packOperand(ORT_RAND, packFloat2(getMin(), getMax()));
			}

			__host__ __inline__ static std::string toString()
			{
				return "rand(" + std::to_string(getMin()) + ", " + std::to_string(getMax()) + ")";
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename AT, typename BT, OperationType OpTypeT>
		struct ExpOperator
		{
			__host__ __inline__ static void encode(float* b, unsigned int p)
			{
				AT::encode(b, p);
				BT::encode(b, p + AT::Length);
				b[p + AT::Length + BT::Length] = PackUtils::packOperand(ORT_OP, (float)OpTypeT);
			}

			__host__ __inline__ static std::string toString()
			{
				std::string opStr;
				switch (OpTypeT)
				{
				case OperationType::OPT_ADD:
					opStr = "+";
					break;
				case OperationType::OPT_SUB:
					opStr = "-";
					break;
				case OperationType::OPT_DIV:
					opStr = "/";
					break;
				case OperationType::OPT_MULTI:
					opStr = "*";
					break;
				case OperationType::OPT_EQ:
					opStr = "==";
					break;
				case OperationType::OPT_NEQ:
					opStr = "!=";
					break;
				case OperationType::OPT_LT:
					opStr = "<";
					break;
				case OperationType::OPT_GT:
					opStr = ">";
					break;
				case OperationType::OPT_LEQ:
					opStr = "<=";
					break;
				case OperationType::OPT_GEQ:
					opStr = ">=";
					break;
				default:
					// FIXME: checking invariants
					throw std::runtime_error("PGA::Parameters::ExpOperator::toString(): unknown expression operator type");
				}
				return AT::toString() + " " + opStr + " " + BT::toString();
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename OperandAT, typename OperandBT>
		struct Add : public ExpOperator < OperandAT, OperandBT, OperationType::OPT_ADD >
		{
			static const unsigned int Length = OperandAT::Length + OperandBT::Length + 1;

			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return OperandAT::eval(symbol) + OperandBT::eval(symbol);
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename OperandAT, typename OperandBT>
		struct Sub : public ExpOperator < OperandAT, OperandBT, OperationType::OPT_SUB >
		{
			static const unsigned int Length = OperandAT::Length + OperandBT::Length + 1;

			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return OperandAT::eval(symbol) - OperandBT::eval(symbol);
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename OperandAT, typename OperandBT>
		struct Multi : public ExpOperator < OperandAT, OperandBT, OperationType::OPT_MULTI >
		{
			static const unsigned int Length = OperandAT::Length + OperandBT::Length + 1;

			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return OperandAT::eval(symbol) * OperandBT::eval(symbol);
			}
		};

		//////////////////////////////////////////////////////////////////////////
		template <typename OperandAT, typename OperandBT>
		struct Div : public ExpOperator < OperandAT, OperandBT, OperationType::OPT_DIV >
		{
			static const unsigned int Length = OperandAT::Length + OperandBT::Length + 1;

			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return OperandAT::eval(symbol) / OperandBT::eval(symbol);
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename OperandAT, typename OperandBT>
		struct Eq : public ExpOperator < OperandAT, OperandBT, OperationType::OPT_EQ >
		{
			static const unsigned int Length = OperandAT::Length + OperandBT::Length + 1;

			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return OperandAT::eval(symbol) == OperandBT::eval(symbol);
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename OperandAT, typename OperandBT>
		struct Neq : public ExpOperator < OperandAT, OperandBT, OperationType::OPT_NEQ >
		{
			static const unsigned int Length = OperandAT::Length + OperandBT::Length + 1;

			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return OperandAT::eval(symbol) != OperandBT::eval(symbol);
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename OperandAT, typename OperandBT>
		struct Lt : public ExpOperator < OperandAT, OperandBT, OperationType::OPT_LT >
		{
			static const unsigned int Length = OperandAT::Length + OperandBT::Length + 1;

			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return OperandAT::eval(symbol) < OperandBT::eval(symbol);
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename OperandAT, typename OperandBT>
		struct Gt : public ExpOperator < OperandAT, OperandBT, OperationType::OPT_GT >
		{
			static const unsigned int Length = OperandAT::Length + OperandBT::Length + 1;

			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return OperandAT::eval(symbol) > OperandBT::eval(symbol);
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename OperandAT, typename OperandBT>
		struct Leq : public ExpOperator < OperandAT, OperandBT, OperationType::OPT_LEQ >
		{
			static const unsigned int Length = OperandAT::Length + OperandBT::Length + 1;

			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				float a = OperandAT::eval(symbol), b = OperandBT::eval(symbol);
				return a <= b;
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename OperandAT, typename OperandBT>
		struct Geq : public ExpOperator < OperandAT, OperandBT, OperationType::OPT_GEQ >
		{
			static const unsigned int Length = OperandAT::Length + OperandBT::Length + 1;

			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return OperandAT::eval(symbol) >= OperandBT::eval(symbol);
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename OperandAT, typename OperandBT>
		struct And : public ExpOperator < OperandAT, OperandBT, OperationType::OPT_AND >
		{
			static const unsigned int Length = OperandAT::Length + OperandBT::Length + 1;

			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return OperandAT::eval(symbol) != 0 && OperandBT::eval(symbol) != 0;
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename OperandAT, typename OperandBT>
		struct Or : public ExpOperator < OperandAT, OperandBT, OperationType::OPT_OR >
		{
			static const unsigned int Length = OperandAT::Length + OperandBT::Length + 1;

			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return OperandAT::eval(symbol) != 0 || OperandBT::eval(symbol) != 0;
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename OperandAT, typename OperandBT, template <class, class> class OpT>
		struct Exp
		{
			static const unsigned int Length = OperandAT::Length + OperandBT::Length + 1;

			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
				return OpT<OperandAT, OperandBT>::eval(symbol);
			}

			__host__ __inline__ static DispatchTableEntry::Parameter toParameter()
			{
				DispatchTableEntry::Parameter parameter;
				parameter.type = ParameterType::PT_EXP;
				parameter.value[0] = static_cast<float>(Length);
				Exp<OperandAT, OperandBT, OpT>::encode((&parameter.value[0]) + 1);
				return parameter;
			}

			__host__ __inline__ static void encode(float* b, unsigned int p = 0)
			{
				OpT<OperandAT, OperandBT>::encode(b, p);
			}

			__host__ __inline__ static std::string toString()
			{
				return "(" + OpT<OperandAT, OperandBT>::toString() + ")";
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <unsigned int LengthT>
		struct Stack
		{
		private:
			float values[LengthT];
			unsigned int top;

		public:
			__host__ __device__ __inline__ Stack() : top(0) {}

			__host__ __device__ __inline__ bool empty() const
			{
				return top == 0;
			}

			__host__ __device__ __inline__ float pop()
			{
				return values[--top];
			}

			__host__ __device__ __inline__ void push(float item)
			{
				values[top++] = item;
			}

		};

		//////////////////////////////////////////////////////////////////////////
		template <typename StackT>
		struct EncExp
		{
		private:
			__host__ __device__ __inline__ static float operate(OperationType op, float p1, float p2)
			{
				switch (op)
				{
				case OperationType::OPT_ADD:
					return p1 + p2;
				case OperationType::OPT_SUB:
					return p1 - p2;
				case OperationType::OPT_DIV:
					return p1 / p2;
				case OperationType::OPT_MULTI:
					return p1 * p2;
				case OperationType::OPT_EQ:
					return p1 == p2;
				case OperationType::OPT_NEQ:
					return p1 != p2;
				case OperationType::OPT_LT:
					return p1 < p2;
				case OperationType::OPT_GT:
					return p1 > p2;
				case OperationType::OPT_LEQ:
					return p1 <= p2;
				case OperationType::OPT_GEQ:
					return p1 >= p2;
				case OperationType::OPT_AND:
					return p1 != 0 && p2 != 0;
				case OperationType::OPT_OR:
					return p1 != 0 || p2 != 0;
				default:
				{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("PGA::Parameters::EncExp<StackT>::operate(..): unknown exp operation type [op=%d] (CUDA thread %d %d)\n", op, threadIdx.x, blockIdx.x);
					asm("trap;");
#else
					throw std::runtime_error(("PGA::Parameters::EncExp<StackT>::operate(..): unknown exp operation type [op=" + std::to_string(op) + "]").c_str());
#endif
#endif
				}
				}
				// warning C4715
				return 0;
			}

		public:
			template <typename ShapeT>
			__host__ __device__ __inline__ static float eval(const Symbol<ShapeT>* symbol, const float* b)
			{
				StackT stack;
				unsigned int size = static_cast<unsigned int>(b[0]);
				unsigned int i = 1;
				if (T::IsEnabled<DebugFlags::ExpressionEvaluation>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("Starting expression evaluation [size: %d] (CUDA thread %d %d)\n", size, threadIdx.x, blockIdx.x);
#else
					std::cout << "Starting expression evaluation [size: " << size << "]" << std::endl;
#endif
				while (i <= size)
				{
					OperandType p;
					float x;
					float p1, p2;
					PackUtils::unpackOperand(b[i], p, x);
					switch (p)
					{
					case OperandType::ORT_SCALAR:
					{
						if (T::IsEnabled<DebugFlags::ExpressionEvaluation>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("[%d] Pop'd a ORT_SCALAR with value %.5f (CUDA thread %d %d)\n", i, x, threadIdx.x, blockIdx.x);
#else
							std::cout << "[" << i << "] Pop'd a ORT_SCALAR with value " << x << std::endl;
#endif
						stack.push(x);
						break;
					}
					case OperandType::ORT_RAND:
						float min, max;
						PackUtils::unpackFloat2(x, min, max);
						if (T::IsEnabled<DebugFlags::ExpressionEvaluation>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("[%d] Pop'd a ORT_RAND with min %.5f and max %.5f (CUDA thread %d %d)\n", i, min, max, threadIdx.x, blockIdx.x);
#else
							std::cout << "[" << i << "] Pop'd a ORT_RAND with min " << min << " and max " << max << ")" << std::endl;
#endif
						stack.push(Random::range(symbol->getSeed(), min, max));
						break;
					case OperandType::ORT_SHAPE_ATTR:
					{
						unsigned int idx;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						idx = __float2uint_rn(x);
#else
						idx = static_cast<unsigned int>(round(x));
#endif
						if (T::IsEnabled<DebugFlags::ExpressionEvaluation>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("[%d] Pop'd a ORT_SHAPE_ATTR with idx %d [%.5f] (CUDA thread %d %d)\n", i, idx, x, threadIdx.x, blockIdx.x);
#else
							std::cout << "[" << i << "] Pop'd a ORT_SHAPE_ATTR with idx " << idx << " [x=" << x << "]" << std::endl;
#endif
						stack.push(symbol->at(idx));
						break;
					}
					case OperandType::ORT_OP:
					{
						OperationType opType;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						opType = static_cast<OperationType>(__float2uint_rn(x));
#else
						opType = static_cast<OperationType>(static_cast<unsigned int>(round(x)));
#endif
						if (T::IsEnabled<DebugFlags::ExpressionEvaluation>::Result)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
							printf("[%d] Pop'd a ORT_OP with type %d [%.5f] (CUDA thread %d %d)\n", i, opType, x, threadIdx.x, blockIdx.x);
#else
							std::cout << "[" << i << "] Pop'd a ORT_OP with type " << opType << " [x=" << x << "]" << std::endl;
#endif
						// NOTE: inverse the order of the operands because they are pop'd from stack, so first is right operand and second is left operand
						p2 = stack.pop();
						p1 = stack.pop();
						stack.push(operate(opType, p1, p2));
						break;
					}
					default:
					{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
						printf("PGA::Parameters::EncExp<StackT>::eval(..): unknown or invalid expression parameter type [p=%d] (CUDA thread %d %d)\n", p, threadIdx.x, blockIdx.x);
						asm("trap;");
#else
						throw std::runtime_error(("PGA::Parameters::EncExp<StackT>::eval(..): unknown or invalid expression parameter type [p=" + std::to_string(p) + "]").c_str());
#endif
#endif
						return 0;
					}
					}
					i++;
				}
				return stack.pop();
			}

		};

		//////////////////////////////////////////////////////////////////////////
		typedef Stack<Constants::MaxNumExpLevels> ExpStack;

		//////////////////////////////////////////////////////////////////////////
		template <typename ShapeT>
		__host__ __device__  __inline__ static float dynEval(const Symbol<ShapeT>* symbol, const DispatchTableEntry::Parameter& parameter)
		{
			switch (parameter.type)
			{
			case ParameterType::PT_SCALAR:
			{
				return parameter.value[0];
			}
			case ParameterType::PT_RAND:
			{
				float min = parameter.value[0];
				float max = parameter.value[1];
				float seed = symbol->getSeed();
				return Random::range(seed, min, max);
			}
			case ParameterType::PT_SHAPE_ATTR:
			{
				return symbol->at(static_cast<unsigned int>(parameter.value[0]));
			}
			case ParameterType::PT_EXP:
			{
				return EncExp<ExpStack>::eval(symbol, parameter.value);
			}
			/*case ParameterType::PT_VEC2:
			{
				// see Vec2::encode(..)
			}
			}*/
			default:
			{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				printf("PGA::Parameters::dynEval(..): unknown or invalid parameter type [parameter.type=%d] (CUDA thread %d %d)\n", parameter.type, threadIdx.x, blockIdx.x);
				asm("trap;");
#else
				throw std::runtime_error(("PGA::Parameters::dynEval(..): unknown or invalid parameter type [parameter.type=" + std::to_string(parameter.type) + "]").c_str());
#endif
#endif
			}
			}
			// warning C4715
			return 0;
		}

		//////////////////////////////////////////////////////////////////////////
		template <typename ShapeT>
		__host__ __device__ __inline__ static math::float2 dynToFloat2(const Symbol<ShapeT>* symbol, const DispatchTableEntry::Parameter& parameter)
		{
			switch (parameter.type)
			{
			case ParameterType::PT_SCALAR:
			{
				return math::float2(parameter.value[0], 0.0f);
			}
			case ParameterType::PT_VEC2:
			case ParameterType::PT_VEC4:
			case ParameterType::PT_RAND:
			{
				return math::float2(parameter.value[0], parameter.value[1]);
			}
			default:
			{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				printf("PGA::Parameters::dynToFloat2(..): unknown or invalid parameter type [parameter.type=%d] (CUDA thread %d %d)\n", parameter.type, threadIdx.x, blockIdx.x);
				asm("trap;");
#else
				throw std::runtime_error(("PGA::Parameters::dynToFloat2(..): unknown or invalid parameter type [parameter.type=" + std::to_string(parameter.type) + "]").c_str());
#endif
#endif
			}
			}
			// warning C4715
			return math::float2();
		}

		//////////////////////////////////////////////////////////////////////////
		template <unsigned int ParamIdxT>
		class DynParam
		{
		public:
			static const unsigned int Length = 1;

			template <typename ShapeT>
			__host__ __device__ __inline__ static math::float2 toFloat2(const Symbol<ShapeT>* symbol)
			{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				if (symbol->entryIndex < 0 || symbol->entryIndex >= Device::NumEntries)
				{
					printf("PGA::Parameters::DynParam<%d>::eval(..): invalid entry index [symbol->entryIndex=%d, Device::NumEntries=%d] (CUDA thread %d %d)\n", ParamIdxT, symbol->entryIndex, Device::NumEntries, threadIdx.x, blockIdx.x);
					asm("trap;");
				}
#else
				if (symbol->entryIndex < 0 || symbol->entryIndex >= Host::NumEntries)
					throw std::runtime_error(("PGA::Parameters::DynParam<" + std::to_string(ParamIdxT) + ">::eval(..): invalid entry index [symbol->entryIndex=" + std::to_string(symbol->entryIndex) + ", Host::NumEntries=" + std::to_string(Host::NumEntries) + "]").c_str());
#endif
#endif
				auto& entry = GlobalVars::getDispatchTableEntry(symbol->entryIndex);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
				if (ParamIdxT >= entry.numParameters)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("PGA::Parameters::DynParam<%d>::eval(..): out of boundaries dynamic parameter access [symbol->entryIndex=%d, entry.numParameters=%d] (CUDA thread %d %d)\n", ParamIdxT, symbol->entryIndex, entry.numParameters, threadIdx.x, blockIdx.x);
					asm("trap;");
#else
					throw std::runtime_error(("PGA::Parameters::DynParam<" + std::to_string(ParamIdxT) + ">::eval(..): out of boundaries dynamic parameter access [symbol->entryIndex=" + std::to_string(ParamIdxT) + ", entry.numParameters=" + std::to_string(entry.numParameters) + "]").c_str());
#endif
				}
#endif
				return dynToFloat2(symbol, entry.parameters[ParamIdxT]);
			}

			template <typename ShapeT>
			__host__ __device__  __inline__ static float eval(const Symbol<ShapeT>* symbol)
			{
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
				if (symbol->entryIndex < 0 || symbol->entryIndex >= Device::NumEntries)
				{
					printf("PGA::Parameters::DynParam<%d>::eval(..): invalid entry index [symbol->entryIndex=%d, Device::NumEntries=%d] (CUDA thread %d %d)\n", ParamIdxT, symbol->entryIndex, Device::NumEntries, threadIdx.x, blockIdx.x);
					asm("trap;");
				}
#else
				if (symbol->entryIndex < 0 || symbol->entryIndex >= Host::NumEntries)
					throw std::runtime_error(("PGA::Parameters::DynParam<" + std::to_string(ParamIdxT) + ">::eval(..): invalid entry index [symbol->entryIndex=" + std::to_string(symbol->entryIndex) + ", Host::NumEntries=" + std::to_string(Host::NumEntries) + "]").c_str());
#endif
#endif
				auto& entry = GlobalVars::getDispatchTableEntry(symbol->entryIndex);
#if defined(PGA_INVARIANT_CHECKING_LVL) && (PGA_INVARIANT_CHECKING_LVL != 0)
				if (ParamIdxT >= entry.numParameters)
				{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
					printf("PGA::Parameters::DynParam<%d>::eval(..): out of boundaries dynamic parameter access [symbol->entryIndex=%d, entry.numParameters=%d] (CUDA thread %d %d)\n", ParamIdxT, symbol->entryIndex, entry.numParameters, threadIdx.x, blockIdx.x);
					asm("trap;");
#else
					throw std::runtime_error(("PGA::Parameters::DynParam<" + std::to_string(ParamIdxT) + ">::eval(..): out of boundaries dynamic parameter access [symbol->entryIndex=" + std::to_string(symbol->entryIndex) + ", entry.numParameters=" + std::to_string(entry.numParameters) + "]").c_str());
#endif
				}
#endif
				return dynEval(symbol, entry.parameters[ParamIdxT]);
			}

			__host__ __inline__ static std::string toString()
			{
				return "dyn(" + std::to_string(ParamIdxT) + ")";
			}

		};

		class DynParams {};

	}

}
