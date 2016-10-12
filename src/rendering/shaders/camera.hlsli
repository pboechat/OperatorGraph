struct Camera
{
	float4x4 View;
	float4x4 Projection;
	float4x4 ViewProjection;
	float3 Position;

};

cbuffer CameraParameters : register(b0)
{
	Camera camera;

};
