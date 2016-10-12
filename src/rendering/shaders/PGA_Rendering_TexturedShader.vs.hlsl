#include <camera.hlsli>

//////////////
// TYPEDEFS //
//////////////
struct VSInput
{
	float4 position: POSITION;
	float3 normal: NORMAL;
	float2 uv: TEXCOORD0;
	float custom0: CUSTOM0;
	float4x4 modelMatrix: MODELMATRIX;
	float custom1: CUSTOM1;

};

struct PSInput
{
	float4 position: SV_POSITION;
	float3 normal: NORMAL;
	float2 uv: TEXCOORD0;
	float3 viewDirection: TEXCOORD1;

};

/////////////
// GLOBALS //
/////////////
cbuffer MaterialParameters : register(b2)
{
	float alphaThreshold;
	float useColorLookup;
	float useUvX0;
	float useUvY0;
	float2 uvScale0;
	float4 color0;

};

PSInput main(VSInput i)
{
	PSInput o;
	float4 world = mul(i.position, i.modelMatrix);
	o.position = mul(world, camera.ViewProjection);
	o.normal = mul(float4(i.normal, 0.0f), i.modelMatrix).xyz;
	o.uv = i.uv;
	o.viewDirection = camera.Position - world.xyz;
	float3 absNormal = abs(o.normal);
	if (absNormal.x > absNormal.y)
	{
		if (absNormal.x > absNormal.z)
			o.uv = float2(lerp(world.z, o.uv.x, useUvX0), lerp(world.y, o.uv.y, useUvY0));
		else
			o.uv = float2(lerp(world.x, o.uv.x, useUvX0), lerp(world.y, o.uv.y, useUvY0));
	}
	else
	{
		if (absNormal.y > absNormal.z)
			o.uv = float2(lerp(world.x, o.uv.x, useUvX0), lerp(world.z, o.uv.y, useUvY0));
		else
			o.uv = float2(lerp(world.x, o.uv.x, useUvX0), lerp(world.y, o.uv.y, useUvY0));
	}
	o.uv *= uvScale0.xy;
	return o;
}