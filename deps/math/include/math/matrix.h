


#ifndef INCLUDED_MATH_MATRIX
#define INCLUDED_MATH_MATRIX

#pragma once

#include <cassert>
#include <string>
#include <exception>
#include "vector.h"

#ifdef __CUDACC__
#define MATH_FUNCTION __host__ __device__
#else
#define MATH_FUNCTION  
#endif

#ifndef MATH_ALIGNMENT
/********************************************************************************/
/* source: http://stackoverflow.com/questions/12778949/cuda-memory-alignment	*/
/********************************************************************************/
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
#define MATH_ALIGNMENT(n) __align__(n)
#elif defined(__GNUC__)
#define MATH_ALIGNMENT(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#define MATH_ALIGNMENT(n) __declspec(align(n))
#else
#error Please provide a definition for MATH_ALIGNMENT macro for your host compiler!
#endif
#endif

namespace math
{
  template <typename T, unsigned int M, unsigned int N>
  class MATH_ALIGNMENT(16) matrix;

  template <typename T, unsigned int D>
  class MATH_ALIGNMENT(16) affine_matrix;

  template <typename T>
  class MATH_ALIGNMENT(16) matrix<T, 2U, 3U>
  {
  public:
    static const unsigned int m = 2U;
    static const unsigned int n = 3U;
    typedef T field_type;

    T _11, _12, _13;
    T _21, _22, _23;

    MATH_FUNCTION matrix() = default;

    MATH_FUNCTION explicit matrix(T a)
      : _11(a), _12(a), _13(a),
        _21(a), _22(a), _23(a)
    {
    }

    MATH_FUNCTION matrix(T m11, T m12, T m13,
           T m21, T m22, T m23)
      : _11(m11), _12(m12), _13(m13),
        _21(m21), _22(m22), _23(m23)
    {
    }

    MATH_FUNCTION matrix(const affine_matrix<T, 2U>& M)
      : _11(M._11), _12(M._12), _13(M._13),
        _21(M._21), _22(M._22), _23(M._23)
    {
    }

    static MATH_FUNCTION inline matrix from_rows(const vector<T,3U>& r1, const vector<T,3U>& r2)
    {
      return matrix(r1.x, r1.y, r1.z,
                    r2.x, r2.y, r2.z);
    }

    static MATH_FUNCTION inline matrix from_cols(const vector<T,2U>& c1, const vector<T,2U>& c2, const vector<T,2U>& c3)
    {
      return matrix(c1.x, c2.x, c3.x,
                    c1.y, c2.y, c3.y);
    }
    
    MATH_FUNCTION const vector<T,3U> row1() const
    {
      return vector<T,3U>(_11, _12, _13);
    }
    MATH_FUNCTION const vector<T,3U> row2() const
    {
      return vector<T,3U>(_21, _22, _23);
    }

    MATH_FUNCTION const vector<T,2U> column1() const
    {
      return vector<T,2U>(_11, _21);
    }
    MATH_FUNCTION const vector<T,2U> column2() const
    {
      return vector<T,2U>(_12, _22);
    }
    MATH_FUNCTION const vector<T,2U> column3() const
    {
      return vector<T,2U>(_13, _23);
    }


    MATH_FUNCTION friend inline matrix<T, 2U, 3U> transpose(const matrix& m)
    {
      return matrix<T, 2U, 3U>(m._11, m._21,
                               m._12, m._22,
                               m._13, m._23);
    }

    MATH_FUNCTION friend inline matrix operator +(const matrix& a, const matrix& b)
    {
      return matrix(a._11 + b._11, a._12 + b._12, a._13 + b._13,
                    a._21 + b._21, a._22 + b._22, a._23 + b._23);
    }

    MATH_FUNCTION friend inline matrix operator *(float f, const matrix& m)
    {
      return matrix(f * m._11, f * m._12, f * m._13,
                    f * m._21, f * m._22, f * m._23);
    }

    MATH_FUNCTION friend inline matrix operator *(const matrix& m, float f)
    {
      return f * m;
    }
  };

  /////////////////////////////////////////////////////////////////////////////

  template <typename T>
  class MATH_ALIGNMENT(16) matrix<T, 3U, 4U>
  {
  public:
    static const unsigned int m = 3U;
    static const unsigned int n = 4U;
    typedef T field_type;

    T _11, _12, _13, _14;
    T _21, _22, _23, _24;
    T _31, _32, _33, _34;

    MATH_FUNCTION matrix() = default;

    MATH_FUNCTION explicit matrix(T a)
      : _11(a), _12(a), _13(a), _14(a),
        _21(a), _22(a), _23(a), _24(a),
        _31(a), _32(a), _33(a), _34(a)
    {
    }

    MATH_FUNCTION matrix(T m11, T m12, T m13, T m14,
           T m21, T m22, T m23, T m24,
           T m31, T m32, T m33, T m34)
      : _11(m11), _12(m12), _13(m13), _14(m14),
        _21(m21), _22(m22), _23(m23), _24(m24),
        _31(m31), _32(m32), _33(m33), _34(m34)
    {
    }

    MATH_FUNCTION matrix(const affine_matrix<T, 3U>& M)
      : _11(M._11), _12(M._12), _13(M._13), _14(M._14),
        _21(M._21), _22(M._22), _23(M._23), _24(M._24),
        _31(M._31), _32(M._32), _33(M._33), _34(M._34)
    {
    }

    static MATH_FUNCTION inline matrix from_rows(const vector<T,4U>& r1, const vector<T,4U>& r2, const vector<T,4U>& r3)
    {
      return matrix(r1.x, r1.y, r1.z, r1.w,
                    r2.x, r2.y, r2.z, r2.w,
                    r3.x, r3.y, r3.z, r3.w);
    }
    static MATH_FUNCTION inline matrix from_cols(const vector<T,3U>& c1, const vector<T,3U>& c2, const vector<T,3U>& c3, const vector<T,3U>& c4)
    {
      return matrix(c1.x, c2.x, c3.x, c4.x,
                    c1.y, c2.y, c3.y, c4.y,
                    c1.z, c2.z, c3.z, c4.z);
    }

    MATH_FUNCTION const vector<T,4U> row1() const
    {
      return vector<T,4U>(_11, _12, _13, _14);
    }
    MATH_FUNCTION const vector<T,4U> row2() const
    {
      return vector<T,4U>(_21, _22, _23, _24);
    }
    MATH_FUNCTION const vector<T,4U> row3() const
    {
      return vector<T,4U>(_31, _32, _33, _34);
    }

    MATH_FUNCTION const vector<T,3U> column1() const
    {
      return vector<T,3U>(_11, _21, _31);
    }
    MATH_FUNCTION const vector<T,3U> column2() const
    {
      return vector<T,3U>(_12, _22, _32);
    }
    MATH_FUNCTION const vector<T,3U> column3() const
    {
      return vector<T,3U>(_13, _23, _33);
    }
    MATH_FUNCTION const vector<T,3U> column4() const
    {
      return vector<T,3U>(_14, _24, _34);
    }

    MATH_FUNCTION friend inline matrix transpose(const matrix& m)
    {
      return matrix(m._11, m._21, m._31, 
                    m._12, m._22, m._32, 
                    m._13, m._23, m._33, 
                    m._14, m._24, m._34);
    }

    MATH_FUNCTION friend inline matrix operator +(const matrix& a, const matrix& b)
    {
      return matrix(a._11 + b._11, a._12 + b._12, a._13 + b._13, a._14 + b._14,
                    a._21 + b._21, a._22 + b._22, a._23 + b._23, a._24 + b._24,
                    a._31 + b._31, a._32 + b._32, a._33 + b._33, a._34 + b._34);
    }

    MATH_FUNCTION friend inline matrix operator *(float f, const matrix& m)
    {
      return matrix(f * m._11, f * m._12, f * m._13, f * m._14,
                    f * m._21, f * m._22, f * m._23, f * m._24,
                    f * m._31, f * m._32, f * m._33, f * m._34);
    }

    MATH_FUNCTION friend inline matrix operator *(const matrix& m, float f)
    {
      return f * m;
    }

	// TODO: 3x4 * 4x3
    //MATH_FUNCTION friend inline matrix operator *(const matrix& a, const matrix& b)
    //{
    //  return matrix(a._11*b._11 + a._12*b._21 + a._13*b._31 + a._14*b._41,  a._11*b._12 + a._12*b._22 + a._13*b._32 + a._14*b._42,  a._11*b._13 + a._12*b._23 + a._13*b._33 + a._14*b._43,  a._11*b._14 + a._12*b._24 + a._13*b._34 + a._14*b._44,
    //                a._21*b._11 + a._22*b._21 + a._23*b._31 + a._24*b._41,  a._21*b._12 + a._22*b._22 + a._23*b._32 + a._24*b._42,  a._21*b._13 + a._22*b._23 + a._23*b._33 + a._24*b._43,  a._21*b._14 + a._22*b._24 + a._23*b._34 + a._24*b._44,
    //                a._31*b._11 + a._32*b._21 + a._33*b._31 + a._34*b._41,  a._31*b._12 + a._32*b._22 + a._33*b._32 + a._34*b._42,  a._31*b._13 + a._32*b._23 + a._33*b._33 + a._34*b._43,  a._31*b._14 + a._32*b._24 + a._33*b._34 + a._34*b._44,
    //                a._41*b._11 + a._42*b._21 + a._43*b._31 + a._44*b._41,  a._41*b._12 + a._42*b._22 + a._43*b._32 + a._44*b._42,  a._41*b._13 + a._42*b._23 + a._43*b._33 + a._44*b._43,  a._41*b._14 + a._42*b._24 + a._43*b._34 + a._44*b._44);
    //}

    MATH_FUNCTION friend inline vector<T, 4U> operator *(const vector<T, 3U>& v, const matrix& m)
    {
      return vector<T, 4U>(v.x*m._11 + v.y*m._21 + v.z*m._31,
                           v.x*m._12 + v.y*m._22 + v.z*m._32,
                           v.x*m._13 + v.y*m._23 + v.z*m._33,
                           v.x*m._14 + v.y*m._24 + v.z*m._34);
    }

    MATH_FUNCTION friend inline vector<T, 4U> operator *(const matrix& m, const vector<T, 4>& v)
    {
      return vector<T, 4U>(m._11*v.x + m._12*v.y + m._13*v.z + m._14*v.w,
                           m._21*v.x + m._22*v.y + m._23*v.z + m._24*v.w,
                           m._31*v.x + m._32*v.y + m._33*v.z + m._34*v.w,
						   1.0f);
    }

    MATH_FUNCTION static inline matrix rotateX(T angle)
    {
      float s = std::sin(angle);
      float c = std::cos(angle);
      return matrix(1, 0, 0, 0,
                    0, c,-s, 0,
                    0, s, c, 0);
    }
    MATH_FUNCTION static inline matrix rotateY(T angle)
    {
      float s = std::sin(angle);
      float c = std::cos(angle);
      return matrix(c, 0, s, 0,
                    0, 1, 0, 0,
                   -s, 0, c, 0);
    }
    MATH_FUNCTION static inline matrix rotateZ(T angle)
    {
      float s = std::sin(angle);
      float c = std::cos(angle);
      return matrix(c,-s, 0, 0,
                    s, c, 0, 0,
                    0, 0, 1, 0);
    }

	//MATH_FUNCTION static inline matrix translate(matrix& m, const vector<T,3>& dst)
	//{
	//  m._14 += dst.x; m._24 += dst.y; m._34 += dst.z;
	//  return m;
	//}

	//MATH_FUNCTION static inline matrix translate(const matrix& m, const vector<T,3>& dst)
	//{
	//  matrix M(m);
	//  M._14 += dst.x; M._24 += dst.y; M._34 += dst.z;
	//  return M;
	//}

	//MATH_FUNCTION static inline matrix translate(math::matrix<T,4U,4U>& m, const vector<T,3>& dst)
	//{
	//  m._14 += dst.x; m._24 += dst.y; m._34 += dst.z;
	//  return m;
	//}
  };


  /////////////////////////////////////////////////////////////////////////////
  template <typename T>
  class MATH_ALIGNMENT(16) matrix<T, 2U, 2U>
  {
  public:
    static const unsigned int m = 2U;
    static const unsigned int n = 2U;
    typedef T field_type;

    T _11, _12;
    T _21, _22;

    MATH_FUNCTION matrix() = default;

    MATH_FUNCTION explicit matrix(T a)
      : _11(a), _12(a),
        _21(a), _22(a)
    {
    }

    MATH_FUNCTION matrix(T m11, T m12,
           T m21, T m22)
      : _11(m11), _12(m12),
        _21(m21), _22(m22)
    {
    }

    static MATH_FUNCTION inline matrix from_rows(const vector<T,2U>& r1, const vector<T,2U>& r2)
    {
      return matrix(r1.x, r1.y,
                    r2.x, r2.y);
    }
    static MATH_FUNCTION inline matrix from_cols(const vector<T,2U>& c1, const vector<T,2U>& c2)
    {
      return matrix(c1.x, c2.x,
                    c1.y, c2.y);
    }
    
    MATH_FUNCTION const vector<T,2U> row1() const
    {
      return vector<T,3U>(_11, _12);
    }
    MATH_FUNCTION const vector<T,2U> row2() const
    {
      return vector<T,3U>(_21, _22);
    }

    MATH_FUNCTION const vector<T,2U> column1() const
    {
      return vector<T,2U>(_11, _21);
    }
    MATH_FUNCTION const vector<T,2U> column2() const
    {
      return vector<T,2U>(_12, _22);
    }

    MATH_FUNCTION friend inline matrix transpose(const matrix& m)
    {
      return matrix(m._11, m._21,
                    m._12, m._22);
    }

	MATH_FUNCTION static inline matrix scale(const vector<T, 2>& angle)
	{
		matrix M(angle.x, 0.0f,
			0.0f, angle.y);
		return M;
	}

    MATH_FUNCTION friend inline matrix operator *(const matrix& a, const matrix& b)
    {
      return matrix(a._11 * b._11 + a._12 * b._21, a._11 * b._12 + a._12 * b._22,
                    a._21 * b._11 + a._22 * b._21, a._21 * b._12 + a._22 * b._22);
    }
    MATH_FUNCTION friend inline vector<T,2u> operator *(const matrix& a, const vector<T,2u>& b)
    {
      return vector<T,2u>(a._11 * b.x + a._12 * b.y,
                          a._21 * b.x + a._22 * b.y);
    }

    MATH_FUNCTION friend inline matrix operator +(const matrix& a, const matrix& b)
    {
      return matrix(a._11 + b._11, a._12 + b._12,
                    a._21 + b._21, a._22 + b._22);
    }

    MATH_FUNCTION friend inline matrix operator *(T f, const matrix& m)
    {
      return matrix(f * m._11, f * m._12,
                    f * m._21, f * m._22);
    }

    MATH_FUNCTION friend inline matrix operator *(const matrix& m, T f)
    {
      return f * m;
    }

    MATH_FUNCTION static inline vector<T,2U> rotate(T angle, const vector<T,2U>& v)
    {
      float s = std::sin(angle);
      float c = std::cos(angle);
      return vector<T,2U>(c*v.x - s*v.y, s*v.x + c*v.y);
    }


    MATH_FUNCTION static inline matrix rotate(T angle)
    {
      float s = std::sin(angle);
      float c = std::cos(angle);
      return matrix(c, -s,
                    s,  c);
    }

    MATH_FUNCTION friend inline T trace(const matrix& M)
    {
      return M._11 + M._22;
    }
  };

  template <typename T>
  class MATH_ALIGNMENT(16) matrix<T, 3U, 3U>
  {
  public:
    static const unsigned int m = 3U;
    static const unsigned int n = 3U;
    typedef T field_type;

    T _11, _12, _13;
    T _21, _22, _23;
    T _31, _32, _33;

    MATH_FUNCTION matrix() {}

    MATH_FUNCTION explicit matrix(T a)
      : _11(a), _12(a), _13(a),
        _21(a), _22(a), _23(a),
        _31(a), _32(a), _33(a)
    {
    }

    MATH_FUNCTION matrix(T m11, T m12, T m13,
           T m21, T m22, T m23,
           T m31, T m32, T m33)
      : _11(m11), _12(m12), _13(m13),
        _21(m21), _22(m22), _23(m23),
        _31(m31), _32(m32), _33(m33)
    {
    }

    MATH_FUNCTION matrix(const affine_matrix<T, 2U>& M)
      : _11(M._11), _12(M._12), _13(M._13),
        _21(M._21), _22(M._22), _23(M._23),
        _31( 0.0f), _32( 0.0f), _33( 1.0f)
    {
    }

    static MATH_FUNCTION inline matrix from_rows(const vector<T,3U>& r1, const vector<T,3U>& r2, const vector<T,3U>& r3)
    {
      return matrix(r1.x, r1.y, r1.z,
                    r2.x, r2.y, r2.z,
                    r3.x, r3.y, r3.z);
    }
    static MATH_FUNCTION inline matrix from_cols(const vector<T,3U>& c1, const vector<T,3U>& c2, const vector<T,3U>& c3)
    {
      return matrix(c1.x, c2.x, c3.x,
                    c1.y, c2.y, c3.y,
                    c1.z, c2.z, c3.z);
    }

    MATH_FUNCTION const vector<T,3U> row1() const
    {
      return vector<T,3U>(_11, _12, _13);
    }
    MATH_FUNCTION const vector<T,3U> row2() const
    {
      return vector<T,3U>(_21, _22, _23);
    }
    MATH_FUNCTION const vector<T,3U> row3() const
    {
      return vector<T,3U>(_31, _32, _33);
    }

    MATH_FUNCTION const vector<T,3U> column1() const
    {
      return vector<T,3U>(_11, _21, _31);
    }
    MATH_FUNCTION const vector<T,3U> column2() const
    {
      return vector<T,3U>(_12, _22, _32);
    }
    MATH_FUNCTION const vector<T,3U> column3() const
    {
      return vector<T,3U>(_13, _23, _33);
    }
    
    MATH_FUNCTION friend inline matrix transpose(const matrix& m)
    {
      return matrix(m._11, m._21, m._31,
                    m._12, m._22, m._32,
                    m._13, m._23, m._33);
    }


    MATH_FUNCTION friend inline T determinant(const matrix& m)
    {
      return m._11*m._22*m._33 + m._12*m._23*m._31 + m._13*m._21*m._32 
           - m._13*m._22*m._31 - m._12*m._21*m._33 - m._11*m._23*m._32;
    }

    MATH_FUNCTION friend inline matrix operator +(const matrix& a, const matrix& b)
    {
      return matrix(a._11 + b._11, a._12 + b._12, a._13 + b._13,
                    a._21 + b._21, a._22 + b._22, a._23 + b._23,
                    a._31 + b._31, a._32 + b._32, a._33 + b._33);
    }

    MATH_FUNCTION friend inline matrix operator *(float f, const matrix& m)
    {
      return matrix(f * m._11, f * m._12, f * m._13,
                    f * m._21, f * m._22, f * m._23,
                    f * m._31, f * m._32, f * m._33);
    }

    MATH_FUNCTION friend inline matrix operator *(const matrix& m, float f)
    {
      return f * m;
    }

    MATH_FUNCTION friend inline matrix operator *(const matrix& a, const matrix& b)
    {
      return matrix(a._11*b._11 + a._12*b._21 + a._13*b._31, a._11*b._12 + a._12*b._22 + a._13*b._32, a._11*b._13 + a._12*b._23 + a._13*b._33,
                    a._21*b._11 + a._22*b._21 + a._23*b._31, a._21*b._12 + a._22*b._22 + a._23*b._32, a._21*b._13 + a._22*b._23 + a._23*b._33,
                    a._31*b._11 + a._32*b._21 + a._33*b._31, a._31*b._12 + a._32*b._22 + a._33*b._32, a._31*b._13 + a._32*b._23 + a._33*b._33);
    }

    MATH_FUNCTION friend inline vector<T, 3U> operator *(const vector<T, 3U>& v, const matrix& m)
    {
      return vector<T, 3U>(v.x*m._11 + v.y*m._21 + v.z*m._31,
                           v.x*m._12 + v.y*m._22 + v.z*m._32,
                           v.x*m._13 + v.y*m._23 + v.z*m._33);
    }

    MATH_FUNCTION friend inline vector<T, 3U> operator *(const matrix& m, const vector<T, 3U>& v)
    {
      return vector<T, 3U>(m._11*v.x + m._12*v.y + m._13*v.z,
                           m._21*v.x + m._22*v.y + m._23*v.z,
                           m._31*v.x + m._32*v.y + m._33*v.z);
    }

    MATH_FUNCTION static inline matrix rotateX(T angle)
    {
      float s = std::sin(angle);
      float c = std::cos(angle);
      return matrix(1,   0,  0,
                    0,   c, -s,
                    0,   s,  c);
    }
    MATH_FUNCTION static inline matrix rotateY(T angle)
    {
      float s = std::sin(angle);
      float c = std::cos(angle);
      return matrix( c, 0, s,
                     0, 1, 0,
                    -s, 0, c);
    }
    MATH_FUNCTION static inline matrix rotateZ(T angle)
    {
      float s = std::sin(angle);
      float c = std::cos(angle);
      return matrix( c, -s, 0,
                     s,  c, 0,
                     0,  0, 1);
    }

    MATH_FUNCTION static inline matrix rotateXYZ(vector<T,3> angles)
    {
      float sa = std::sin(angles.x);
      float sb = std::sin(angles.y);
      float sg = std::sin(angles.z);
      float ca = std::cos(angles.x);
      float cb = std::cos(angles.y);
      float cg = std::cos(angles.z);
      return matrix(cb*cg, cg*sa*sb-ca*sg, ca*cg*sb+sa*sg,
                      cb*sg, ca*cg+sa*sb*sg, -cg*sa+ca*sb*sg,
                      -sb,cb*sa,ca*cb);
    }

	MATH_FUNCTION static inline matrix scale(const vector<T, 3>& angle)
	{
		matrix M(angle.x, 0.0f, 0.0f,
			0.0f, angle.y, 0.0f,
			0.0f, 0.0f, angle.z);
		return M;
	}

    MATH_FUNCTION friend inline T trace(const matrix& M)
    {
      return M._11 + M._22 + M._33;
    }
  };


  /////////////////////////////////////////////////////////////////////////////

  template <typename T>
  class MATH_ALIGNMENT(16) matrix<T, 4U, 4U>
  {
  public:
    static const unsigned int m = 4U;
    static const unsigned int n = 4U;
    typedef T field_type;

	union
	{
		struct
		{
			T _11, _12, _13, _14;
			T _21, _22, _23, _24;
			T _31, _32, _33, _34;
			T _41, _42, _43, _44;
		};
		T _m[16];
	};

	/*T _11, _12, _13, _14;
	T _21, _22, _23, _24;
	T _31, _32, _33, _34;
	T _41, _42, _43, _44;*/

    MATH_FUNCTION matrix() = default;

    MATH_FUNCTION explicit matrix(T a)
      : _11(a), _12(a), _13(a), _14(a),
        _21(a), _22(a), _23(a), _24(a),
        _31(a), _32(a), _33(a), _34(a),
        _41(a), _42(a), _43(a), _44(a)
    {
    }

    MATH_FUNCTION matrix(T m11, T m12, T m13, T m14,
           T m21, T m22, T m23, T m24,
           T m31, T m32, T m33, T m34,
           T m41, T m42, T m43, T m44)
      : _11(m11), _12(m12), _13(m13), _14(m14),
        _21(m21), _22(m22), _23(m23), _24(m24),
        _31(m31), _32(m32), _33(m33), _34(m34),
        _41(m41), _42(m42), _43(m43), _44(m44)
    {
    }

    MATH_FUNCTION matrix(const affine_matrix<T, 3U>& M)
      : _11(M._11), _12(M._12), _13(M._13), _14(M._14),
        _21(M._21), _22(M._22), _23(M._23), _24(M._24),
        _31(M._31), _32(M._32), _33(M._33), _34(M._34),
        _41( 0.0f), _42( 0.0f), _43( 0.0f), _44( 1.0f)
    {
    }

    static MATH_FUNCTION inline matrix from_rows(const vector<T,4U>& r1, const vector<T,4U>& r2, const vector<T,4U>& r3, const vector<T,4U>& r4)
    {
      return matrix(r1.x, r1.y, r1.z, r1.w,
                    r2.x, r2.y, r2.z, r2.w,
                    r3.x, r3.y, r3.z, r3.w,
                    r4.x, r4.y, r4.z, r4.w);
    }
    static MATH_FUNCTION inline matrix from_cols(const vector<T,4U>& c1, const vector<T,4U>& c2, const vector<T,4U>& c3, const vector<T,4U>& c4)
    {
      return matrix(c1.x, c2.x, c3.x, c4.x,
                    c1.y, c2.y, c3.y, c4.y,
                    c1.z, c2.z, c3.z, c4.z,
                    c1.w, c2.w, c3.w, c4.w);
    }

    MATH_FUNCTION const vector<T,4U> row1() const
    {
      return vector<T,4U>(_11, _12, _13, _14);
    }
    MATH_FUNCTION const vector<T,4U> row2() const
    {
      return vector<T,4U>(_21, _22, _23, _24);
    }
    MATH_FUNCTION const vector<T,4U> row3() const
    {
      return vector<T,4U>(_31, _32, _33, _34);
    }
    MATH_FUNCTION const vector<T,4U> row4() const
    {
      return vector<T,4U>(_41, _42, _43, _44);
    }


    MATH_FUNCTION const vector<T,4U> column1() const
    {
      return vector<T,4U>(_11, _21, _31, _41);
    }
    MATH_FUNCTION const vector<T,4U> column2() const
    {
      return vector<T,4U>(_12, _22, _32, _42);
    }
    MATH_FUNCTION const vector<T,4U> column3() const
    {
      return vector<T,4U>(_13, _23, _33, _43);
    }
    MATH_FUNCTION const vector<T,4U> column4() const
    {
      return vector<T,4U>(_14, _24, _34, _44);
    }

    //MATH_FUNCTION friend inline matrix inverse(const matrix& m)
    //{
    //  matrix C(m._22*m._33*m._44 + m._23*m._34*m._42 + m._24*m._32*m._43 - m._42*m._33*m._24 - m._43*m._34*m._22 - m._44*m._32*m._23,
    //           m._42*m._33*m._14 + m._43*m._34*m._12 + m._44*m._32*m._13 - m._12*m._33*m._44 - m._13*m._34*m._42 - m._14*m._32*m._43,
    //           m._12*m._23*m._44 + m._13*m._24*m._42 + m._14*m._22*m._43 - m._42*m._23*m._14 - m._43*m._24*m._12 - m._44*m._22*m._13,
    //           m._32*m._23*m._14 + m._33*m._24*m._12 + m._34*m._22*m._13 - m._12*m._23*m._34 - m._13*m._24*m._32 - m._14*m._22*m._33,
    //           m._41*m._33*m._24 + m._43*m._34*m._21 + m._44*m._31*m._23 - m._21*m._33*m._44 - m._23*m._34*m._41 - m._24*m._31*m._43,
    //           m._11*m._33*m._44 + m._13*m._34*m._41 + m._14*m._31*m._43 - m._41*m._33*m._14 - m._43*m._34*m._11 - m._44*m._31*m._13,
    //           m._41*m._23*m._14 + m._43*m._24*m._11 + m._44*m._21*m._13 - m._11*m._23*m._44 - m._13*m._24*m._41 - m._14*m._21*m._43,
    //           m._11*m._23*m._34 + m._13*m._24*m._31 + m._14*m._21*m._33 - m._31*m._23*m._14 - m._33*m._24*m._11 - m._34*m._21*m._13,
    //           m._21*m._32*m._44 + m._22*m._34*m._41 + m._24*m._31*m._42 - m._41*m._32*m._24 - m._42*m._34*m._21 - m._44*m._31*m._22,
    //           m._41*m._32*m._14 + m._42*m._34*m._11 + m._44*m._31*m._12 - m._11*m._32*m._44 - m._12*m._34*m._41 - m._14*m._31*m._42,
    //           m._11*m._22*m._44 + m._12*m._24*m._41 + m._14*m._21*m._42 - m._41*m._22*m._14 - m._42*m._24*m._11 - m._44*m._21*m._12,
    //           m._31*m._22*m._14 + m._32*m._24*m._11 + m._34*m._21*m._12 - m._11*m._22*m._34 - m._12*m._24*m._31 - m._14*m._21*m._32,
    //           m._41*m._32*m._23 + m._42*m._33*m._21 + m._43*m._31*m._22 - m._21*m._32*m._43 - m._22*m._33*m._41 - m._23*m._31*m._42,
    //           m._11*m._32*m._43 + m._12*m._33*m._41 + m._13*m._31*m._42 - m._41*m._32*m._13 - m._42*m._33*m._11 - m._43*m._31*m._12,
    //           m._41*m._22*m._13 + m._42*m._23*m._11 + m._43*m._21*m._12 - m._11*m._22*m._43 - m._12*m._23*m._41 - m._13*m._21*m._42,
    //           m._11*m._22*m._33 + m._12*m._23*m._31 + m._13*m._21*m._32 - m._31*m._22*m._13 - m._32*m._23*m._11 - m._33*m._21*m._12);

    //  float d = 1.0f / (m._11 * C._11 + m._12 * C._21 + m._13 * C._31 + m._14 * C._41);

    //  return d * C;
    //}
    
    MATH_FUNCTION friend inline matrix transpose(const matrix& m)
    {
      return matrix(m._11, m._21, m._31, m._41,
                    m._12, m._22, m._32, m._42,
                    m._13, m._23, m._33, m._43,
                    m._14, m._24, m._34, m._44);
    }

    MATH_FUNCTION friend inline matrix operator +(const matrix& a, const matrix& b)
    {
      return matrix(a._11 + b._11, a._12 + b._12, a._13 + b._13, a._14 + b._14,
                    a._21 + b._21, a._22 + b._22, a._23 + b._23, a._24 + b._24,
                    a._31 + b._31, a._32 + b._32, a._33 + b._33, a._34 + b._34,
                    a._41 + b._41, a._42 + b._42, a._43 + b._43, a._44 + b._44);
    }

    MATH_FUNCTION friend inline matrix operator *(float f, const matrix& m)
    {
      return matrix(f * m._11, f * m._12, f * m._13, f * m._14,
                    f * m._21, f * m._22, f * m._23, f * m._24,
                    f * m._31, f * m._32, f * m._33, f * m._34,
                    f * m._41, f * m._42, f * m._43, f * m._44);
    }

    MATH_FUNCTION friend inline matrix operator *(const matrix& m, float f)
    {
      return f * m;
    }

    MATH_FUNCTION friend inline matrix operator *(const matrix& a, const matrix& b)
    {
      return matrix(a._11*b._11 + a._12*b._21 + a._13*b._31 + a._14*b._41, a._11*b._12 + a._12*b._22 + a._13*b._32 + a._14*b._42, a._11*b._13 + a._12*b._23 + a._13*b._33 + a._14*b._43, a._11*b._14 + a._12*b._24 + a._13*b._34 + a._14*b._44,
                    a._21*b._11 + a._22*b._21 + a._23*b._31 + a._24*b._41, a._21*b._12 + a._22*b._22 + a._23*b._32 + a._24*b._42, a._21*b._13 + a._22*b._23 + a._23*b._33 + a._24*b._43, a._21*b._14 + a._22*b._24 + a._23*b._34 + a._24*b._44,
                    a._31*b._11 + a._32*b._21 + a._33*b._31 + a._34*b._41, a._31*b._12 + a._32*b._22 + a._33*b._32 + a._34*b._42, a._31*b._13 + a._32*b._23 + a._33*b._33 + a._34*b._43, a._31*b._14 + a._32*b._24 + a._33*b._34 + a._34*b._44,
                    a._41*b._11 + a._42*b._21 + a._43*b._31 + a._44*b._41, a._41*b._12 + a._42*b._22 + a._43*b._32 + a._44*b._42, a._41*b._13 + a._42*b._23 + a._43*b._33 + a._44*b._43, a._41*b._14 + a._42*b._24 + a._43*b._34 + a._44*b._44);
    }

    MATH_FUNCTION friend inline vector<T, 4U> operator *(const vector<T, 4U>& v, const matrix& m)
    {
      return vector<T, 4U>(v.x*m._11 + v.y*m._21 + v.z*m._31 + v.w*m._41,
                           v.x*m._12 + v.y*m._22 + v.z*m._32 + v.w*m._42,
                           v.x*m._13 + v.y*m._23 + v.z*m._33 + v.w*m._43,
                           v.x*m._14 + v.y*m._24 + v.z*m._34 + v.w*m._44);
    }

    MATH_FUNCTION friend inline vector<T, 4U> operator *(const matrix& m, const vector<T, 4>& v)
    {
      return vector<T, 4U>(m._11*v.x + m._12*v.y + m._13*v.z + m._14*v.w,
                           m._21*v.x + m._22*v.y + m._23*v.z + m._24*v.w,
                           m._31*v.x + m._32*v.y + m._33*v.z + m._34*v.w,
                           m._41*v.x + m._42*v.y + m._43*v.z + m._44*v.w);
    }

	MATH_FUNCTION inline T operator[](unsigned int i) const
	{
	  return _m[i];
		/*if (i == 0)
			return _11;
		else if (i == 1)
			return _12;
		else if (i == 2)
			return _13;
		else if (i == 3)
			return _14;
		else if (i == 4)
			return _21;
		else if (i == 5)
			return _22;
		else if (i == 6)
			return _23;
		else if (i == 7)
			return _24;
		else if (i == 8)
			return _31;
		else if (i == 9)
			return _32;
		else if (i == 10)
			return _33;
		else if (i == 11)
			return _34;
		else if (i == 12)
			return _41;
		else if (i == 13)
			return _42;
		else if (i == 14)
			return _43;
		else if (i == 15)
			return _44;
		else
		{
			// FIXME: checking invariants
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
			printf("matrix::operator[](..): invalid attribute index [i=%d] (CUDA thread %d %d)\n", i, threadIdx.x, blockIdx.x);
			asm("trap;");
#else
			throw std::exception(("matrix::operator[](..): invalid attribute index [i=" + std::to_string(i) + "]").c_str());
#endif
			return 0;
		}*/
	}

    MATH_FUNCTION static inline matrix rotateX(T angle)
    {
      float s = std::sin(angle);
      float c = std::cos(angle);
      return matrix(1, 0, 0, 0,
                    0, c,-s, 0,
                    0, s, c, 0,
                    0, 0, 0, 1);
    }
    MATH_FUNCTION static inline matrix rotateY(T angle)
    {
      float s = std::sin(angle);
      float c = std::cos(angle);
      return matrix(c, 0, s, 0,
                    0, 1, 0, 0,
                   -s, 0, c, 0,
                    0, 0, 0, 1);
    }
    MATH_FUNCTION static inline matrix rotateZ(T angle)
    {
      float s = std::sin(angle);
      float c = std::cos(angle);
      return matrix(c,-s, 0, 0,
                    s, c, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1);
    }

    MATH_FUNCTION static inline matrix rotateXYZ(vector<T,3> angles)
    {
      float sa = std::sin(angles.x);
      float sb = std::sin(angles.y);
      float sg = std::sin(angles.z);
      float ca = std::cos(angles.x);
      float cb = std::cos(angles.y);
      float cg = std::cos(angles.z);
      return matrix(cb*cg, cg*sa*sb-ca*sg,  ca*cg*sb+sa*sg, 0,
                    cb*sg, ca*cg+sa*sb*sg, -cg*sa+ca*sb*sg, 0,
                      -sb,          cb*sa,           ca*cb, 0,
                        0,              0,               0, 1);
    }

	//MATH_FUNCTION static inline matrix translate(matrix& m, const vector<T,3>& dst)
	//{	  
	//  m._14 += dst.x; m._24 += dst.y; m._34 += dst.z;
	//  return m;
	//}

	MATH_FUNCTION static inline matrix translate(const matrix& m, const vector<T,3>& dst)
	{
	  matrix M(m);
	  M._14 += dst.x; M._24 += dst.y; M._34 += dst.z;
	  return M;
	}


	MATH_FUNCTION static inline matrix translate(const vector<T,3>& dst)
	{
	  return matrix(1.0f, 0.0f, 0.0f, dst.x,
               0.0f, 1.0f, 0.0f, dst.y,
               0.0f, 0.0f, 1.0f, dst.z,
               0.0f, 0.0f, 0.0f, 1.0f);
	}

	MATH_FUNCTION static inline matrix scale(const vector<T,3>& angle)
	{
	  matrix M(angle.x, 0.0f, 0.0f, 0.0f,
               0.0f, angle.y, 0.0f, 0.0f,
               0.0f, 0.0f, angle.z, 0.0f,
               0.0f, 0.0f, 0.0f, 1.0f);
	  return M;
	}

	MATH_FUNCTION inline const matrix& print() const
	{
		printf("\n--------------------------\n");
		printf("%f  %f  %f  %f\n", _11, _12, _13, _14);
		printf("%f  %f  %f  %f\n", _21, _22, _23, _24);
		printf("%f  %f  %f  %f\n", _31, _32, _33, _34);
		printf("%f  %f  %f  %f\n", _41, _42, _43, _44);
		printf("--------------------------\n");
		return *this;
	}

    MATH_FUNCTION friend inline T trace(const matrix& M)
    {
      return M._11 + M._22 + M._33 + M._44;
    }
  };


  template <typename T>
  class MATH_ALIGNMENT(16) affine_matrix<T, 2U>
  {
  public:
    typedef T field_type;

    T _11, _12, _13;
    T _21, _22, _23;

    MATH_FUNCTION affine_matrix() = default;

    MATH_FUNCTION explicit affine_matrix(T a)
      : _11(a), _12(a), _13(a),
        _21(a), _22(a), _23(a)
    {
    }

    MATH_FUNCTION affine_matrix(T m11, T m12, T m13,
                                T m21, T m22, T m23)
      : _11(m11), _12(m12), _13(m13),
        _21(m21), _22(m22), _23(m23)
    {
    }

     MATH_FUNCTION affine_matrix(const matrix<T, 2U, 3U>& M)
      : _11(M._11), _12(M._12), _13(M._13),
        _21(M._21), _22(M._22), _23(M._23)
    {
    }

     MATH_FUNCTION affine_matrix(const matrix<T, 3U, 3U>& M)
      : _11(M._11), _12(M._12), _13(M._13),
        _21(M._21), _22(M._22), _23(M._23)
    {
    }

    MATH_FUNCTION friend inline affine_matrix operator +(const affine_matrix& a, const affine_matrix& b)
    {
      return matrix<T, 3U, 3U>(a) + matrix<T, 3U, 3U>(b);
    }

    MATH_FUNCTION friend inline affine_matrix operator +(const affine_matrix& a, const matrix<T, 3U, 3U>& b)
    {
      return matrix<T, 3U, 3U>(a) + b;
    }

    MATH_FUNCTION friend inline affine_matrix operator +(const matrix<T, 3U, 3U>& a, const affine_matrix& b)
    {
      return a + matrix<T, 3U, 3U>(b);
    }

     MATH_FUNCTION friend inline affine_matrix operator *(const affine_matrix& a, const affine_matrix& b)
    {
      return matrix<T, 3U, 3U>(a) * matrix<T, 3U, 3U>(b);
    }

    MATH_FUNCTION friend inline const matrix<T, 3U, 3U> operator *(const affine_matrix& a, const matrix<T, 3U, 3U>& b)
    {
      return matrix<T, 3U, 3U>(a) * b;
    }

    MATH_FUNCTION friend inline const matrix<T, 3U, 3U> operator *(const matrix<T, 3U, 3U>& a, const affine_matrix& b)
    {
      return a * matrix<T, 3U, 3U>(b);
    }
  };

  template <typename T>
  class MATH_ALIGNMENT(16) affine_matrix<T, 3U>
  {
  public:
    typedef T field_type;

    T _11, _12, _13, _14;
    T _21, _22, _23, _24;
    T _31, _32, _33, _34;

    MATH_FUNCTION affine_matrix() = default;

    MATH_FUNCTION explicit affine_matrix(T a)
      : _11(a), _12(a), _13(a), _14(a),
        _21(a), _22(a), _23(a), _24(a),
        _31(a), _32(a), _33(a), _34(a)
    {
    }

    MATH_FUNCTION affine_matrix(T m11, T m12, T m13, T m14,
                                T m21, T m22, T m23, T m24,
                                T m31, T m32, T m33, T m34)
      : _11(m11), _12(m12), _13(m13), _14(m14),
        _21(m21), _22(m22), _23(m23), _24(m24),
        _31(m31), _32(m32), _33(m33), _34(m34)
    {
    }

     MATH_FUNCTION affine_matrix(const matrix<T, 3U, 4U>& M)
      : _11(M._11), _12(M._12), _13(M._13), _14(M._14),
        _21(M._21), _22(M._22), _23(M._23), _24(M._24),
        _31(M._31), _32(M._32), _33(M._33), _34(M._34)
    {
    }

     MATH_FUNCTION affine_matrix(const matrix<T, 4U, 4U>& M)
      : _11(M._11), _12(M._12), _13(M._13), _14(M._14),
        _21(M._21), _22(M._22), _23(M._23), _24(M._24),
        _31(M._31), _32(M._32), _33(M._33), _34(M._34)
    {
    }

    MATH_FUNCTION friend inline affine_matrix operator +(const affine_matrix& a, const affine_matrix& b)
    {
      return matrix<T, 4U, 4U>(a) + matrix<T, 4U, 4U>(b);
    }

    MATH_FUNCTION friend inline affine_matrix operator +(const affine_matrix& a, const matrix<T, 4U, 4U>& b)
    {
      return matrix<T, 4U, 4U>(a) + b;
    }

    MATH_FUNCTION friend inline affine_matrix operator +(const matrix<T, 4U, 4U>& a, const affine_matrix& b)
    {
      return a + matrix<T, 4U, 4U>(b);
    }

     MATH_FUNCTION friend inline affine_matrix operator *(const affine_matrix& a, const affine_matrix& b)
    {
      return matrix<T, 4U, 4U>(a) * matrix<T, 4U, 4U>(b);
    }

    MATH_FUNCTION friend inline const matrix<T, 4U, 4U> operator *(const affine_matrix& a, const matrix<T, 4U, 4U>& b)
    {
      return matrix<T, 4U, 4U>(a) * b;
    }

    MATH_FUNCTION friend inline const matrix<T, 4U, 4U> operator *(const matrix<T, 4U, 4U>& a, const affine_matrix& b)
    {
      return a * matrix<T, 4U, 4U>(b);
    }
  };


    
  template <typename T>
  MATH_FUNCTION inline T det(const matrix<T, 2U, 2U>& m)
  {
    return m._11*m._22 - m._21*m._12;
  }

  template <typename T>
  MATH_FUNCTION inline T det(const matrix<T, 3U, 3U>& m)
  {
    return m._11 * det(matrix<T, 2U, 2U>(m._22, m._23, m._32, m._33)) -
           m._12 * det(matrix<T, 2U, 2U>(m._21, m._23, m._31, m._33)) +
           m._13 * det(matrix<T, 2U, 2U>(m._21, m._22, m._31, m._32));
  }

  template <typename T>
  MATH_FUNCTION inline T det(const matrix<T, 4U, 4U>& m)
  {
    return m._11 * det(matrix<T, 3U, 3U>(m._22, m._23, m._24, m._32, m._33, m._34, m._42, m._43, m._44)) -
           m._12 * det(matrix<T, 3U, 3U>(m._21, m._23, m._24, m._31, m._33, m._34, m._41, m._43, m._44)) +
           m._13 * det(matrix<T, 3U, 3U>(m._21, m._22, m._24, m._31, m._32, m._34, m._41, m._42, m._44)) -
           m._14 * det(matrix<T, 3U, 3U>(m._21, m._22, m._23, m._31, m._32, m._33, m._41, m._42, m._43));
  }

  template <typename T>
  MATH_FUNCTION inline matrix<T, 3U, 3U> adj(const matrix<T, 3U, 3U>& m)
  {
    return transpose(matrix<T, 3U, 3U>(
      det(matrix<T, 2U, 2U>(m._22, m._23, m._32, m._33)),
     -det(matrix<T, 2U, 2U>(m._21, m._23, m._31, m._33)),
      det(matrix<T, 2U, 2U>(m._21, m._22, m._31, m._32)),

     -det(matrix<T, 2U, 2U>(m._12, m._13, m._32, m._33)),
      det(matrix<T, 2U, 2U>(m._11, m._13, m._31, m._33)),
     -det(matrix<T, 2U, 2U>(m._11, m._12, m._31, m._32)),

      det(matrix<T, 2U, 2U>(m._12, m._13, m._22, m._23)),
     -det(matrix<T, 2U, 2U>(m._11, m._13, m._21, m._23)),
      det(matrix<T, 2U, 2U>(m._11, m._12, m._21, m._22))
    ));
  }

  template <typename T>
  MATH_FUNCTION inline matrix<T, 4U, 4U> adj(const matrix<T, 4U, 4U>& m)
  {
    return transpose(matrix<T, 4U, 4U>(
      det(matrix<T, 3U, 3U>(m._22, m._23, m._24, m._32, m._33, m._34, m._42, m._43, m._44)),
     -det(matrix<T, 3U, 3U>(m._21, m._23, m._24, m._31, m._33, m._34, m._41, m._43, m._44)),
      det(matrix<T, 3U, 3U>(m._21, m._22, m._24, m._31, m._32, m._34, m._41, m._42, m._44)),
     -det(matrix<T, 3U, 3U>(m._21, m._22, m._23, m._31, m._32, m._33, m._41, m._42, m._43)),

     -det(matrix<T, 3U, 3U>(m._12, m._13, m._14, m._32, m._33, m._34, m._42, m._43, m._44)),
      det(matrix<T, 3U, 3U>(m._11, m._13, m._14, m._31, m._33, m._34, m._41, m._43, m._44)),
     -det(matrix<T, 3U, 3U>(m._11, m._12, m._14, m._31, m._32, m._34, m._41, m._42, m._44)),
      det(matrix<T, 3U, 3U>(m._11, m._12, m._13, m._31, m._32, m._33, m._41, m._42, m._43)),

      det(matrix<T, 3U, 3U>(m._12, m._13, m._14, m._22, m._23, m._24, m._42, m._43, m._44)),
     -det(matrix<T, 3U, 3U>(m._11, m._13, m._14, m._21, m._23, m._24, m._41, m._43, m._44)),
      det(matrix<T, 3U, 3U>(m._11, m._12, m._14, m._21, m._22, m._24, m._41, m._42, m._44)),
     -det(matrix<T, 3U, 3U>(m._11, m._12, m._13, m._21, m._22, m._23, m._41, m._42, m._43)),

     -det(matrix<T, 3U, 3U>(m._12, m._13, m._14, m._22, m._23, m._24, m._32, m._33, m._34)),
      det(matrix<T, 3U, 3U>(m._11, m._13, m._14, m._21, m._23, m._24, m._31, m._33, m._34)),
     -det(matrix<T, 3U, 3U>(m._11, m._12, m._14, m._21, m._22, m._24, m._31, m._32, m._34)),
      det(matrix<T, 3U, 3U>(m._11, m._12, m._13, m._21, m._22, m._23, m._31, m._32, m._33))
    ));
  }


  template <typename T, unsigned int N>
  MATH_FUNCTION inline matrix<T, N, N> inverse(const matrix<T, N, N>& M)
  {
    // TODO: optimize; compute det using adj
    return rcp(det(M)) * adj(M);
  }


  template <typename T, unsigned int D>
  MATH_FUNCTION inline affine_matrix<T, D> inverse(const affine_matrix<T, D>& M)
  {
    return affine_matrix<T, D>(inverse(matrix<T, D + 1, D + 1>(M)));
  }


  typedef matrix<float, 2U, 2U> float2x2;
  typedef matrix<float, 2U, 3U> float2x3;
  typedef matrix<float, 3U, 3U> float3x3;
  typedef matrix<float, 3U, 4U> float3x4;
  typedef matrix<float, 4U, 4U> float4x4;
  
  typedef affine_matrix<float, 2U> affine_float3x3;
  typedef affine_matrix<float, 3U> affine_float4x4;
  
  
  template <typename T>
  MATH_FUNCTION inline T identity();
  
  template <>
  MATH_FUNCTION inline float2x2 identity<float2x2>()
  {
    return float2x2(1.0f, 0.0f, 
                    0.0f, 1.0f);
  }

  template <>
  MATH_FUNCTION inline float3x3 identity<float3x3>()
  {
    return float3x3(1.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 1.0f);
  }
  
  template <>
  MATH_FUNCTION inline math::affine_float3x3 identity<math::affine_float3x3>()
  {
    return math::affine_float3x3(1.0f, 0.0f, 0.0f,
                                 0.0f, 1.0f, 0.0f);
  }
  
  template <>
  MATH_FUNCTION inline math::float4x4 identity<math::float4x4>()
  {
    return math::float4x4(1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f);
  }

  template <>
  MATH_FUNCTION inline math::affine_float4x4 identity<math::affine_float4x4>()
  {
    return math::affine_float4x4(1.0f, 0.0f, 0.0f, 0.0f,
                                 0.0f, 1.0f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 1.0f, 0.0f);
  }


}

//using math::float2x3;
//using math::float3x3;
//using math::math::float4x4;
#undef MATH_FUNCTION 

#endif  // INCLUDED_MATH_MATRIX
