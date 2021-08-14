#pragma once

#include <math/matrix.h>
#include <math/vector.h>

class Camera
{
private:
	float _aspect;
	float _fov2radians;
	float _nearZ;
	float _farZ;
	math::float3 _position;
	math::float3 _w;
	math::float3 _u;
	math::float3 _v;

public:
	struct UniformBuffer
	{
		math::float4x4 View;
		math::float4x4 Projection;
		math::float4x4 ViewProjection;
		math::float3 Position;

	};

	Camera(
		float aspect,
		float fovDegrees,
		float nearZ,
		float farZ,
		const math::float3& position,
		const math::float3& u = math::float3(1, 0, 0),
		const math::float3& v = math::float3(0, 1, 0),
		const math::float3& w = math::float3(0, 0, 1))
		:
		_aspect(aspect),
		_fov2radians(fovDegrees * 0.00872664625f),
		_nearZ(nearZ),
		_farZ(farZ),
		_position(position),
		_u(u),
		_v(v),
		_w(w)
	{
	}

	inline void writeToBuffer(UniformBuffer* buffer) const
	{
		auto _view = view();
		auto _projection = projection();
		buffer->View = _view;
		buffer->Projection = _projection;
		buffer->ViewProjection = _projection * _view;
		buffer->Position = _position;
	}

	inline math::float3& u() 
	{ 
		return _u; 
	}

	inline math::float3& v() 
	{ 
		return _v; 
	}

	inline math::float3& w() 
	{ 
		return _w; 
	}

	inline math::float3& position() 
	{ 
		return _position; 
	}

	inline math::float4x4 view() const
	{
		return math::float4x4(_u.x, _u.y, _u.z, -dot(_u, _position),
							  _v.x, _v.y, _v.z, -dot(_v, _position),
							  _w.x, _w.y, _w.z, -dot(_w, _position),
							  0, 0, 0, 1);
	}

	inline math::float4x4 projection() const
	{
		float xScale = 1.0f / std::tan(_fov2radians);
		float yScale = xScale / _aspect;
		float z1 = _farZ / (_farZ - _nearZ);
		float z2 = -_nearZ * _farZ / (_farZ - _nearZ);
		return math::float4x4(yScale, 0, 0, 0,
							  0, xScale, 0, 0,
							  0, 0, z1, z2,
							  0, 0, 1, 0);
	}

	inline math::float4x4 viewProjection() const 
	{ 
		return projection() * view(); 
	}

};
