#include <lighting.hlsli>

//////////////
// TYPEDEFS //
//////////////
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
	float4 color0;

};

float4 main(PSInput i) : SV_TARGET
{
	return lambert(dLight.direction, i.normal) * dLight.color * color0;
}